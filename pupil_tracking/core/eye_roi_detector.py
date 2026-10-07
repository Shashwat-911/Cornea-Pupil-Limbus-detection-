"""
Eye Region of Interest (ROI) Detector for Video Frames.

Automatically locates and crops eye regions from full-frame video
so the segmentation model receives the same input distribution it
was trained on (zoomed-in eye images).

Strategy hierarchy:
  1. Auto-detect if frame is already an eye closeup → skip ROI
  2. Use cached ROI from previous frame           → ~1 ms
  3. Haar cascade face → eye detection             → ~8 ms
  4. Intensity-based dark-blob fallback            → ~5 ms
  5. Full frame as last resort                     → 0 ms
"""

import cv2
import numpy as np
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ROIResult:
    """Result of eye ROI detection."""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    cropped: Optional[np.ndarray] = field(default=None, repr=False)
    is_closeup: bool = False
    from_cache: bool = False
    confidence: float = 0.0

    @property
    def valid(self) -> bool:
        return (self.width > 0
                and self.height > 0
                and self.cropped is not None
                and self.cropped.size > 0)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EyeROIDetector:
    """
    Detects and caches the eye region inside arbitrary video frames.

    Designed for speed: once the ROI is found it is cached for
    *cache_ttl* frames and only re-detected periodically or when
    detection confidence drops.
    """

    def __init__(
        self,
        cache_ttl: int = 12,
        padding_ratio: float = 0.6,
        min_eye_size: Tuple[int, int] = (25, 25),
        scale_factor: float = 1.15,
        min_neighbors: int = 4,
    ):
        self.cache_ttl = cache_ttl
        self.padding_ratio = padding_ratio
        self.min_eye_size = min_eye_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        # Haar cascades (built into OpenCV, zero extra deps)
        self._load_cascades()

        # Cache state
        self._cached_roi: Optional[Tuple[int, int, int, int]] = None
        self._cache_counter: int = 0
        self._closeup_mode: Optional[bool] = None
        self._consecutive_misses: int = 0

    # ------------------------------------------------------------------
    # Cascade loading
    # ------------------------------------------------------------------
    def _load_cascades(self):
        cascade_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        )
        self.eye_cascade = cv2.CascadeClassifier(
            os.path.join(cascade_dir, "haarcascade_eye.xml")
        )
        if self.face_cascade.empty():
            logger.warning("Face cascade failed to load – ROI detection disabled")
        if self.eye_cascade.empty():
            logger.warning("Eye cascade failed to load – ROI detection disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> ROIResult:
        """Return the best eye ROI for *frame*."""
        fh, fw = frame.shape[:2]

        # 0. Surgical microscope viewport detection for widescreen recording feeds (e.g. Elita)
        if fw >= int(fh * 1.25):
            viewport = self._detect_surgical_viewport(frame)
            if viewport is not None:
                vx, vy, vw, vh = viewport
                return ROIResult(
                    x=vx,
                    y=vy,
                    width=vw,
                    height=vh,
                    cropped=frame[vy:vy + vh, vx:vx + vw],
                    is_closeup=True,
                    from_cache=False,
                    confidence=0.98,
                )

        # 1. Closeup mode (decided once on first frame)
        if self._closeup_mode is None:
            self._closeup_mode = self._is_eye_closeup(frame)
        if self._closeup_mode:
            return ROIResult(0, 0, fw, fh, frame, True, False, 1.0)

        # 2. Cache hit
        if self._cached_roi and self._cache_counter > 0:
            self._cache_counter -= 1
            x, y, w, h = self._cached_roi
            if 0 <= x and 0 <= y and x + w <= fw and y + h <= fh:
                return ROIResult(x, y, w, h, frame[y:y+h, x:x+w],
                                 False, True, 0.85)

        # 3. Haar cascade detection
        roi = self._detect_haar(frame)
        if roi.valid:
            self._update_cache(roi)
            self._consecutive_misses = 0
            return roi

        # 4. Intensity-based fallback
        roi = self._detect_intensity(frame)
        if roi.valid:
            self._update_cache(roi)
            self._consecutive_misses = 0
            return roi

        # 5. Full frame fallback with rapid closeup lock-on
        self._consecutive_misses += 1
        if self._consecutive_misses >= 3:
            # Rapid recovery: if 3 consecutive frames cannot find a sub-region face/eye,
            # assume the image is an eye closeup/surgical microscope feed and lock closeup mode.
            self._closeup_mode = True
            return ROIResult(0, 0, fw, fh, frame, True, False, 0.8)
        return ROIResult(0, 0, fw, fh, frame, False, False, 0.5)

    def reset(self):
        """Reset all state (call when switching videos)."""
        self._cached_roi = None
        self._cache_counter = 0
        self._closeup_mode = None
        self._consecutive_misses = 0

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------
    def _detect_haar(self, frame: np.ndarray) -> ROIResult:
        if self.face_cascade.empty() or self.eye_cascade.empty():
            return ROIResult()

        gray = self._to_gray(frame)
        fh, fw = gray.shape

        # Downscale for speed if frame is large
        scale = 1.0
        if fw > 640:
            scale = 640.0 / fw
            small = cv2.resize(gray, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray

        faces = self.face_cascade.detectMultiScale(
            small, self.scale_factor, self.min_neighbors, minSize=(50, 50)
        )

        best_eye, best_area = None, 0
        for fx, fy, f_w, f_h in faces:
            eye_h = int(f_h * 0.6)
            roi = small[fy:fy + eye_h, fx:fx + f_w]
            eyes = self.eye_cascade.detectMultiScale(
                roi, 1.1, 3, minSize=self.min_eye_size
            )
            for ex, ey, ew, eh in eyes:
                area = ew * eh
                if area > best_area:
                    best_area = area
                    best_eye = (
                        int((fx + ex) / scale),
                        int((fy + ey) / scale),
                        int(ew / scale),
                        int(eh / scale),
                    )

        if best_eye is None:
            return ROIResult()

        x, y, w, h = self._pad_and_square(*best_eye, fw, fh)
        return ROIResult(x, y, w, h, frame[y:y+h, x:x+w], False, False, 0.9)

    def _detect_intensity(self, frame: np.ndarray) -> ROIResult:
        gray = self._to_gray(frame)
        fh, fw = gray.shape
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Try adaptive / percentile thresholding for dark pupil/iris region
        for thresh_mult in (0.45, 0.60, 0.35):
            thresh = int(np.mean(blurred) * thresh_mult)
            _, bw = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            best, best_score = None, 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 300 or area > fh * fw * 0.5:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0:
                    continue
                circularity = 4 * np.pi * area / (peri * peri)
                if circularity < 0.25:
                    continue
                score = area * circularity
                if score > best_score:
                    best_score = score
                    best = cv2.boundingRect(cnt)

            if best is not None:
                x, y, w, h = self._pad_and_square(*best, fw, fh)
                return ROIResult(x, y, w, h, frame[y:y+h, x:x+w], False, False, 0.70)

        return ROIResult()

    # ------------------------------------------------------------------
    # Closeup heuristic
    # ------------------------------------------------------------------
    def _is_eye_closeup(self, frame: np.ndarray) -> bool:
        gray = self._to_gray(frame)
        h, w = gray.shape
        m = min(h, w) // 4
        center_mean = float(gray[m:h - m, m:w - m].mean())
        full_mean = float(gray.mean())

        # If image dimensions suggest typical cropped video resolution or aspect ratio
        if center_mean < full_mean * 0.75:
            if self._has_circular_blob(gray, min_frac=0.003, max_frac=0.40):
                logger.info("Detected eye closeup (dark centre + circular structure)")
                return True

        # No face detected → typical for surgical/microscope eye closeup feeds
        if not self.face_cascade.empty():
            small = gray
            if w > 640:
                s = 640.0 / w
                small = cv2.resize(gray, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
            faces = self.face_cascade.detectMultiScale(
                small, 1.1, 3, minSize=(50, 50))
            if len(faces) == 0:
                if self._has_circular_blob(gray, 0.002, 0.45) or full_mean > 15:
                    logger.info("No face in frame → treating as eye closeup feed")
                    return True
        return False

    def _has_circular_blob(self, gray: np.ndarray,
                           min_frac: float, max_frac: float) -> bool:
        for thresh_mode in (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU, cv2.THRESH_BINARY_INV):
            if thresh_mode == cv2.THRESH_BINARY_INV:
                t_val = int(np.mean(gray) * 0.5)
                _, bw = cv2.threshold(gray, t_val, 255, thresh_mode)
            else:
                _, bw = cv2.threshold(gray, 0, 255, thresh_mode)
            contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            total = gray.shape[0] * gray.shape[1]
            for cnt in contours:
                a = cv2.contourArea(cnt)
                if min_frac * total < a < max_frac * total:
                    p = cv2.arcLength(cnt, True)
                    if p > 0 and 4 * np.pi * a / (p * p) > 0.25:
                        return True
        return False

    def _detect_surgical_viewport(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect circular surgical microscope camera viewport in widescreen feeds."""
        fh, fw = frame.shape[:2]
        gray = self._to_gray(frame)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = fh * fw * 0.12
        max_area = fh * fw * 0.65
        best_circle = None
        best_score = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * r * r
            if circle_area <= 0:
                continue
            circularity = area / circle_area
            if circularity > 0.70 and (0.30 * fh <= r <= 0.60 * fh):
                score = circularity * area
                if score > best_score:
                    best_score = score
                    best_circle = (int(round(cx)), int(round(cy)), int(round(r)))
        if best_circle is not None:
            cx, cy, r = best_circle
            x0 = max(0, cx - r)
            y0 = max(0, cy - r)
            x1 = min(fw, cx + r)
            y1 = min(fh, cy + r)
            return (x0, y0, x1 - x0, y1 - y0)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pad_and_square(self, x, y, w, h, fw, fh):
        pw, ph = int(w * self.padding_ratio), int(h * self.padding_ratio)
        x, y = max(0, x - pw), max(0, y - ph)
        w, h = min(fw - x, w + 2 * pw), min(fh - y, h + 2 * ph)
        size = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        x = max(0, min(cx - size // 2, fw - size))
        y = max(0, min(cy - size // 2, fh - size))
        w = min(fw - x, size)
        h = min(fh - y, size)
        return x, y, w, h

    def _update_cache(self, roi: ROIResult):
        self._cached_roi = (roi.x, roi.y, roi.width, roi.height)
        self._cache_counter = self.cache_ttl

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)