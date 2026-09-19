"""
ring_aware.py — Ring-Aware Preprocessing Pipeline

Provides specialised preprocessing for images with and without suction
rings.  The ring creates distinct visual artifacts that require different
handling compared to natural (pre-docked) eye images.

Two public classes
------------------
RingAwarePreprocessor
    Adaptive preprocessor that routes through a docked or pre-docked
    pipeline depending on the RingDetectionResult supplied at call time.

AdaptiveContourFilter
    Post-detection contour filter that applies spatial constraints when
    a suction ring is present (pupil must be inside the ring opening).

Data class
----------
PreprocessingResult
    Bundles every intermediate image and metadata produced by the
    preprocessor so that downstream stages can pick what they need.

Usage
-----
>>> from pupil_tracking.preprocessing.ring_aware import (
...     RingAwarePreprocessor,
...     PreprocessingResult,
...     AdaptiveContourFilter,
... )
>>> from pupil_tracking.core.deterministic_ring_detector import RingDetector
>>>
>>> ring_detector = RingDetector()
>>> ring_result = ring_detector.detect(image)
>>>
>>> preprocessor = RingAwarePreprocessor()
>>> prep = preprocessor.preprocess(image, ring_result)
>>>
>>> # prep.processed_image   — main preprocessed output
>>> # prep.roi_mask          — binary ROI (set when ring found)
>>> # prep.preprocessing_mode — "standard" or "ring_aware"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessingResult:
    """Complete output bundle of the preprocessing pipeline."""

    processed_image: np.ndarray
    """Main preprocessed image (float32 0-1 if normalised, else uint8)."""

    grayscale: np.ndarray
    """Grayscale conversion of the original input."""

    roi_mask: Optional[np.ndarray] = None
    """Binary uint8 mask — 255 inside the region of interest, 0 outside.
    Set only when a suction ring constrains the usable area."""

    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    """Bounding box (x, y, w, h) of the ROI. ``None`` when no ring."""

    ring_masked_image: Optional[np.ndarray] = None
    """Grayscale image with the ring boundary region replaced by the
    local mean intensity so that ring edges do not interfere with
    downstream contour detection."""

    preprocessing_mode: str = "standard"
    """Either ``"standard"`` (pre-docked) or ``"ring_aware"`` (docked)."""

    details: dict = field(default_factory=dict)
    """Free-form metadata for debugging and logging."""


# ═══════════════════════════════════════════════════════════════════════
#  RingAwarePreprocessor
# ═══════════════════════════════════════════════════════════════════════

class RingAwarePreprocessor:
    """
    Adaptive preprocessor that handles both docked and pre-docked images.

    For **pre-docked** images (no ring):
      * Standard pipeline: grayscale → median blur → CLAHE → normalise
      * Wider search area for pupil / limbus
      * Soft eyelid / eyelash suppression at top and bottom of frame

    For **docked** images (ring present):
      * Detect and mask the ring boundary region
      * Extract ROI inside the ring opening
      * Suppress ring-edge artifacts
      * Enhanced contrast within the smaller ring interior
      * Inpaint specular reflections from ring surface

    Parameters
    ----------
    median_kernel : int
        Kernel size for median blur (must be odd).
    clahe_clip : float
        CLAHE clip limit for contrast enhancement.
    clahe_grid : tuple of int
        CLAHE tile grid size ``(rows, cols)``.
    ring_inner_margin : int
        Pixels to shrink inward from the detected ring radius when
        building the inner ROI mask.  Avoids including ring-edge
        pixels in the usable area.
    normalize : bool
        If ``True`` the output image is float32 in ``[0, 1]``.
        If ``False`` the output remains uint8.
    """

    def __init__(
        self,
        median_kernel: int = 5,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        ring_inner_margin: int = 15,
        normalize: bool = True,
    ):
        self.median_kernel = median_kernel
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.ring_inner_margin = ring_inner_margin
        self.normalize = normalize

    # ── public entry point ────────────────────────────────────────

    def preprocess(
        self,
        image: np.ndarray,
        ring_result=None,
    ) -> PreprocessingResult:
        """
        Run the appropriate preprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image ``(H, W, 3)`` or grayscale ``(H, W)``.
        ring_result : RingDetectionResult or None
            Result from :class:`RingDetector`.  When ``None`` the
            standard (pre-docked) pipeline is used.

        Returns
        -------
        PreprocessingResult
        """
        # Lazy import to avoid circular dependency at module load time
        from pupil_tracking.core.deterministic_ring_detector import RingStatus

        has_ring = (
            ring_result is not None
            and ring_result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
        )

        if has_ring:
            return self._preprocess_docked(image, ring_result)
        else:
            return self._preprocess_predocked(image)

    # ── pre-docked (no ring) ─────────────────────────────────────

    def _preprocess_predocked(self, image: np.ndarray) -> PreprocessingResult:
        """
        Standard preprocessing for natural eye images.

        These images may contain visible eyelids, eyelashes, variable
        lighting, and a larger field of view with no artificial circular
        boundary.
        """
        h, w = image.shape[:2]

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Noise reduction
        denoised = cv2.medianBlur(gray, self.median_kernel)

        # 3. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_grid,
        )
        enhanced = clahe.apply(denoised)

        # 4. Soft eyelid / eyelash suppression
        #    Fade the top and bottom 15 % of the frame towards zero so
        #    that dark eyelash pixels do not dominate thresholding.
        eyelid_mask = np.ones((h, w), dtype=np.float32)
        margin = max(1, int(h * 0.15))
        for y in range(margin):
            factor = y / margin
            eyelid_mask[y, :] = factor
            eyelid_mask[h - 1 - y, :] = factor
        enhanced_masked = (enhanced.astype(np.float32) * eyelid_mask).astype(np.uint8)

        # 5. Normalise
        if self.normalize:
            processed = enhanced_masked.astype(np.float32) / 255.0
        else:
            processed = enhanced_masked

        return PreprocessingResult(
            processed_image=processed,
            grayscale=gray,
            roi_mask=None,
            roi_bbox=None,
            ring_masked_image=None,
            preprocessing_mode="standard",
            details={
                "eyelid_suppression": True,
                "eyelid_margin_px": margin,
                "clahe_clip": self.clahe_clip,
                "median_kernel": self.median_kernel,
            },
        )

    # ── docked (ring present) ────────────────────────────────────

    def _preprocess_docked(
        self,
        image: np.ndarray,
        ring_result,
    ) -> PreprocessingResult:
        """
        Ring-aware preprocessing for docked eye images.

        Key differences from the standard pipeline:

        1. Mask out the ring itself — its strong edges would otherwise
           dominate contour extraction and be mistaken for the pupil or
           limbus boundary.
        2. Extract a circular ROI from inside the ring opening.
        3. Apply tighter CLAHE within the smaller region.
        4. Suppress specular reflections caused by the ring surface.
        """
        h, w = image.shape[:2]

        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. Build inner ROI mask and ring-masked image
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        ring_masked_image = gray.copy()
        inner_radius: Optional[int] = None
        roi_bbox: Optional[Tuple[int, int, int, int]] = None

        if (
            ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            cx = int(ring_result.ring_center[0])
            cy = int(ring_result.ring_center[1])
            r = int(ring_result.ring_radius)

            # Inner ROI — everything inside the ring opening minus margin
            inner_radius = max(10, r - self.ring_inner_margin)
            cv2.circle(roi_mask, (cx, cy), inner_radius, 255, -1)

            # Ring band — a thick annulus covering the ring edge
            ring_band = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(ring_band, (cx, cy), r + 25, 255, -1)
            cv2.circle(ring_band, (cx, cy), max(1, r - 25), 0, -1)

            # Replace ring pixels with local mean so edges vanish
            roi_pixels = gray[roi_mask > 0]
            local_mean = int(np.mean(roi_pixels)) if roi_pixels.size > 0 else 128
            ring_masked_image[ring_band > 0] = local_mean

            # ROI bounding box
            x1 = max(0, cx - inner_radius)
            y1 = max(0, cy - inner_radius)
            x2 = min(w, cx + inner_radius)
            y2 = min(h, cy + inner_radius)
            roi_bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            # Ring detected but no geometry → fall back to centre region
            cx, cy = w // 2, h // 2
            r = int(min(h, w) * 0.35)
            inner_radius = r
            cv2.circle(roi_mask, (cx, cy), r, 255, -1)
            roi_bbox = (
                max(0, cx - r),
                max(0, cy - r),
                min(2 * r, w),
                min(2 * r, h),
            )

        # 3. Noise reduction
        denoised = cv2.medianBlur(ring_masked_image, self.median_kernel)

        # 4. CLAHE — tighter grid and slightly stronger clip for the
        #    smaller constrained region inside the ring
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip * 1.2,
            tileGridSize=(4, 4),
        )
        enhanced = clahe.apply(denoised)

        # 5. Apply ROI mask — zero out everything outside the ring
        enhanced_roi = cv2.bitwise_and(enhanced, enhanced, mask=roi_mask)

        # 6. Suppress specular reflections from ring surface
        reflections_suppressed = 0
        roi_pixel_values = enhanced_roi[roi_mask > 0]
        if roi_pixel_values.size > 0:
            reflection_thresh = np.percentile(roi_pixel_values, 97)
            bright_spots = (enhanced_roi > reflection_thresh) & (roi_mask > 0)
            reflections_suppressed = int(np.count_nonzero(bright_spots))
            if reflections_suppressed > 0:
                bright_mask_u8 = bright_spots.astype(np.uint8) * 255
                enhanced_roi = cv2.inpaint(
                    enhanced_roi, bright_mask_u8, 5, cv2.INPAINT_TELEA
                )

        # 7. Normalise
        if self.normalize:
            processed = enhanced_roi.astype(np.float32) / 255.0
        else:
            processed = enhanced_roi

        return PreprocessingResult(
            processed_image=processed,
            grayscale=gray,
            roi_mask=roi_mask,
            roi_bbox=roi_bbox,
            ring_masked_image=ring_masked_image,
            preprocessing_mode="ring_aware",
            details={
                "ring_center": (cx, cy),
                "ring_radius": r,
                "inner_radius": inner_radius,
                "reflections_suppressed": reflections_suppressed,
                "clahe_clip": self.clahe_clip * 1.2,
                "median_kernel": self.median_kernel,
                "ring_inner_margin": self.ring_inner_margin,
            },
        )


# ═══════════════════════════════════════════════════════════════════════
#  AdaptiveContourFilter
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveContourFilter:
    """
    Ring-aware contour filter for post-detection candidate pruning.

    When no ring is present the filter applies the standard area,
    circularity, and aspect-ratio gates.

    When a ring **is** present two additional constraints are enforced:

    * The contour centroid must lie inside the ring opening (within 75 %
      of the ring radius from the ring centre).
    * The contour equivalent radius must be smaller than 60 % of the
      ring radius — any larger and it is likely the ring itself rather
      than the pupil.

    Parameters
    ----------
    min_area : int
        Minimum contour area in pixels.
    max_area : int
        Maximum contour area in pixels.
    min_circularity : float
        Minimum ``4π·area / perimeter²`` value (0–1).
    max_aspect_ratio : float
        Maximum ratio between the major and minor axis of the
        minimum-area bounding rectangle.
    """

    def __init__(
        self,
        min_area: int = 150,
        max_area: int = 50000,
        min_circularity: float = 0.3,
        max_aspect_ratio: float = 3.0,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio

    def filter_contours(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple[int, ...],
        ring_result=None,
    ) -> List[np.ndarray]:
        """
        Filter contour candidates with optional ring constraints.

        Parameters
        ----------
        contours : list of np.ndarray
            Raw contour arrays from ``cv2.findContours``.
        image_shape : tuple
            ``(H, W)`` or ``(H, W, C)`` of the source image.
        ring_result : RingDetectionResult or None
            When supplied *and* the ring status is PRESENT, spatial
            constraints relative to the ring geometry are applied.

        Returns
        -------
        list of np.ndarray
            Contours that passed all filters.
        """
        # Lazy import
        from pupil_tracking.core.deterministic_ring_detector import RingStatus

        filtered: List[np.ndarray] = []
        h, w = image_shape[:2]

        has_ring = (
            ring_result is not None
            and ring_result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        )

        for cnt in contours:
            # ── basic geometric gates ─────────────────────────────
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            # Aspect ratio via minimum-area bounding rectangle
            if len(cnt) >= 5:
                _, (box_w, box_h), _ = cv2.minAreaRect(cnt)
                if box_w > 0 and box_h > 0:
                    aspect = max(box_w, box_h) / min(box_w, box_h)
                    if aspect > self.max_aspect_ratio:
                        continue

            # ── centroid ──────────────────────────────────────────
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # ── ring-specific spatial constraints ─────────────────
            if has_ring:
                ring_cx, ring_cy = ring_result.ring_center
                # Prefer inner_radius (opening) for spatial constraints
                ring_r_outer = ring_result.ring_radius
                ring_inner = (
                    ring_result.ring_inner_radius
                    if getattr(ring_result, "ring_inner_radius", None) is not None
                    else (ring_r_outer * 0.80 if ring_r_outer else None)
                )

                # Contour centre must be inside the ring opening (use inner radius)
                if ring_inner is not None:
                    dist = np.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)
                    max_allowed_dist = ring_inner * 0.85
                    if dist > max_allowed_dist:
                        continue

                # Equivalent radius must be much smaller than ring inner opening
                equiv_r = np.sqrt(area / np.pi)
                ref_r = ring_inner if ring_inner is not None else ring_r_outer
                if ref_r is not None and equiv_r > ref_r * 0.60:
                    # Likely the ring contour itself — skip
                    continue

            filtered.append(cnt)

        return filtered