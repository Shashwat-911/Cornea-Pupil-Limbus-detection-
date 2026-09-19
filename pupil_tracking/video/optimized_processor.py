"""
Optimised Video Processor â€” ACCURACY-FIRST + SPEED OPTIMIZED VERSION

Plan-aligned changes:
  A2  - Default resolution 192 â†’ 320 (accuracy-first)
  A5  - Suction ring marker masking in preprocessor
  A6  - ImageNormalizer for consistent illumination
  S1  - Batch inference via FastInference.detect_batch()
  S3  - Batch collection from decode-ahead queue
  S4  - Bilateral filter removed from fast path

Target: < 50 ms per frame (GPU), < 80 ms per frame (CPU)
"""

import csv
import gc
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.core.eye_roi_detector import EyeROIDetector, ROIResult
from pupil_tracking.utils.runtime_profile import detect_runtime_profile
from pupil_tracking.utils.types import assign_quality_grade
from pupil_tracking.video.temporal_smoother import TemporalSmoother
from pupil_tracking.core.deterministic_ring_detector import RingDetector, RingStatus

try:
    from pupil_tracking.core.detector import UnifiedDetector

    _HAS_UNIFIED = True
except ImportError:
    _HAS_UNIFIED = False

try:
    from pupil_tracking.ml.fast_inference import FastInference

    _HAS_FAST = True
except ImportError:
    _HAS_FAST = False

logger = logging.getLogger(__name__)

# Default batch size for GPU batching (S1)
_DEFAULT_BATCH_SIZE = 4


def _annotate_quality(det: Dict[str, Any]) -> Dict[str, Any]:
    """Attach unified confidence + quality labels to flat detection dicts."""
    pupil_detected = bool(det.get("pupil_detected", False))
    limbus_detected = bool(det.get("limbus_detected", False))

    if pupil_detected and limbus_detected:
        overall_conf = float(
            (
                float(det.get("pupil_confidence", 0.0))
                + float(det.get("limbus_confidence", 0.0))
            )
            / 2.0
        )
    elif pupil_detected:
        overall_conf = float(det.get("pupil_confidence", 0.0))
    elif limbus_detected:
        overall_conf = float(det.get("limbus_confidence", 0.0))
    else:
        overall_conf = 0.0

    det["overall_confidence"] = overall_conf
    det["overall_quality"] = assign_quality_grade(overall_conf).value
    return det


# ======================================================================
# ACCURACY-FIRST Video Preprocessing
# ======================================================================


class VideoPreprocessor:
    """
    ACCURACY-FIRST preprocessing for video frames.

    Ensures consistent normalisation regardless of illumination.
    Includes reflection removal, suction ring masking, and CLAHE.

    Plan alignment:
        A3 â€” reflection removal
        A5 â€” suction ring marker masking
        A6 â€” ImageNormalizer for brightness/contrast consistency
        S4 â€” bilateral filter removed from fast path
    """

    def __init__(
        self,
        denoise_strength: int = 3,
        clahe_clip: float = 2.0,
        clahe_grid: int = 4,
        sharpen: bool = False,
        fast_mode: bool = True,
        apply_normalizer: bool = True,
        suction_ring_removal: bool = True,
    ):
        self.denoise_strength = denoise_strength
        self.sharpen = sharpen
        self.fast_mode = fast_mode

        # CLAHE for normalisation (always create, used in both modes)
        self._clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            tileGridSize=(clahe_grid, clahe_grid),
        )

        # A3: Reflection removal for specular highlights
        from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover

        self._reflection_remover = ReflectionRemover(
            brightness_threshold=225,
            min_reflection_area=10,
            inpaint_radius=3,
            detect_red_highlights=True,
            red_threshold_offset=25,
        )

        # Temporal reflection filter for blinking lights
        try:
            from pupil_tracking.preprocessing.temporal_reflection_filter import (
                TemporalReflectionFilter,
            )

            self._temporal_filter = TemporalReflectionFilter(
                history_size=5,
                blink_threshold=0.3,
                min_stable_frames=2,
                dilation_size=3,
            )
        except ImportError:
            self._temporal_filter = None
            logger.debug("TemporalReflectionFilter not available")

        # A5: Suction ring marker masking
        self._ring_masker = None
        if suction_ring_removal:
            from pupil_tracking.preprocessing.suction_ring_masker import (
                SuctionRingMasker,
            )

            self._ring_masker = SuctionRingMasker()

        # A6: Full normaliser (CLAHE + brightness, skip heavy ops for speed)
        self._normalizer = None
        if apply_normalizer:
            try:
                from pupil_tracking.preprocessing.normalizer import ImageNormalizer

                self._normalizer = ImageNormalizer(
                    enable_clahe=True,
                    enable_brightness=True,
                    enable_white_balance=False,  # skip for speed
                    enable_gamma=False,  # skip for speed
                )
                logger.info("ImageNormalizer enabled for video preprocessing")
            except ImportError:
                logger.warning(
                    "ImageNormalizer not available â€” falling back to CLAHE only"
                )

        # Temporal filter state for blinking light detection
        self._current_stable_mask: Optional[np.ndarray] = None

    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply video-optimised preprocessing with proper normalisation."""
        if image is None or image.size == 0:
            return image

        out = image

        # A5: Light suction ring marker removal (skip diagnostics for speed)
        if self._ring_masker is not None:
            try:
                out, _ = self._ring_masker.remove(out)
            except Exception:
                pass

        # A3: Remove specular reflections (~0.3ms)
        out, _ = self._reflection_remover.remove(out, roi_mask=None)

        # Temporal filtering for blinking lights (new)
        # This identifies transient bright spots (blinking red lights)
        # and excludes them from affecting pupil detection
        if self._temporal_filter is not None:
            # Get stable reflection mask (excludes transient blinks)
            stable_mask = self._temporal_filter.process(out)
            # Store for later use in postprocessing to mask out blinks
            self._current_stable_mask = stable_mask

        # A6: Full normalisation pipeline if available
        if self._normalizer is not None:
            out = self._normalizer.normalize(out)
        else:
            # Fallback: CLAHE on luminance channel
            if len(out.shape) == 3 and out.shape[2] >= 3:
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            elif len(out.shape) == 2:
                out = self._clahe.apply(out)

        # S4: FAST MODE â€” skip bilateral filter and sharpening
        if self.fast_mode:
            return out

        # Non-fast mode: lighter bilateral filter
        if self.denoise_strength > 0:
            out = cv2.bilateralFilter(
                out,
                d=self.denoise_strength,
                sigmaColor=35,
                sigmaSpace=35,
            )

        # Optional sharpening
        if self.sharpen:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
            out = cv2.addWeighted(out, 1.2, blurred, -0.2, 0)

        return out


# ======================================================================
# Frame Quality Checker
# ======================================================================


class FrameQualityChecker:
    """
    Quality checker with permissive thresholds for surgical images.
    """

    def __init__(
        self,
        blur_threshold: float = 8.0,
        brightness_low: float = 8.0,
        brightness_high: float = 252.0,
        skip_check: bool = False,
    ):
        self.blur_thresh = blur_threshold
        self.bright_lo = brightness_low
        self.bright_hi = brightness_high
        self.skip_check = skip_check

    def is_usable(self, image: np.ndarray) -> Tuple[bool, str]:
        """Returns (usable, reason). ~0.1ms if skip_check=True."""
        if image is None or image.size == 0:
            return False, "empty_frame"

        if self.skip_check:
            return True, "ok"

        gray = (
            image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )

        mean_bright = float(gray.mean())
        if mean_bright < self.bright_lo:
            return False, "too_dark"
        if mean_bright > self.bright_hi:
            return False, "too_bright"

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.blur_thresh:
            return False, "too_blurry"

        return True, "ok"


@dataclass
class ManualCircularROI:
    """User-defined circular ROI stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height


@dataclass
class ManualRingAnnotation:
    """User-confirmed suction-ring circle stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    dot_count: int = 12

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height


# ======================================================================
# Threaded Frame Reader
# ======================================================================


class _FrameReader(threading.Thread):
    """Decode-ahead frame reader (S3)."""

    def __init__(
        self,
        cap: cv2.VideoCapture,
        q: queue.Queue,
        stride: int = 1,
        max_frames: Optional[int] = None,
        pause_predicate: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(daemon=True)
        self._cap = cap
        self._q = q
        self._stride = stride
        self._max = max_frames
        self._pause_predicate = pause_predicate
        self._stop_event = threading.Event()

    def run(self):
        idx = 0
        produced = 0
        while not self._stop_event.is_set():
            if self._pause_predicate is not None and self._pause_predicate():
                time.sleep(0.05)
                continue
            ok, frame = self._cap.read()
            if not ok:
                break
            if idx % self._stride == 0:
                try:
                    self._q.put((idx, frame), timeout=10.0)
                except queue.Full:
                    break
                produced += 1
                if self._max and produced >= self._max:
                    break
            idx += 1
        self._q.put(None)  # sentinel

    def stop(self):
        self._stop_event.set()


# ======================================================================
# Overlay Renderer
# ======================================================================


class _OverlayRenderer:
    """Draws detection results onto video frames."""

    PUPIL_COLOR = (0, 255, 0)
    LIMBUS_COLOR = (255, 180, 0)
    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)

    @classmethod
    def draw(
        cls,
        frame: np.ndarray,
        det: Dict[str, Any],
        frame_idx: int = 0,
        fps: float = 0.0,
    ) -> np.ndarray:
        vis = frame.copy()

        if det.get("pupil_detected"):
            cx = int(det.get("pupil_x", 0))
            cy = int(det.get("pupil_y", 0))
            r = int(det.get("pupil_radius", det.get("pupil_r", 0)))
            cv2.circle(vis, (cx, cy), r, cls.PUPIL_COLOR, 2)
            cv2.circle(vis, (cx, cy), 3, cls.PUPIL_COLOR, -1)

            if "pupil_major" in det and "pupil_minor" in det:
                axes = (
                    int(det["pupil_major"] / 2),
                    int(det["pupil_minor"] / 2),
                )
                angle = det.get("pupil_angle", 0)
                cv2.ellipse(
                    vis,
                    (cx, cy),
                    axes,
                    angle,
                    0,
                    360,
                    cls.PUPIL_COLOR,
                    2,
                    cv2.LINE_AA,
                )

        if det.get("limbus_detected"):
            lx = int(det.get("limbus_x", 0))
            ly = int(det.get("limbus_y", 0))
            lr = int(det.get("limbus_radius", det.get("limbus_r", 0)))
            if "limbus_major" in det and "limbus_minor" in det:
                axes = (
                    int(det["limbus_major"] / 2),
                    int(det["limbus_minor"] / 2),
                )
                angle = det.get("limbus_angle", 0)
                cv2.ellipse(
                    vis,
                    (lx, ly),
                    axes,
                    angle,
                    0,
                    360,
                    cls.LIMBUS_COLOR,
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(vis, (lx, ly), lr, cls.LIMBUS_COLOR, 2)

        lines = [f"Frame: {frame_idx}"]
        if fps > 0:
            lines.append(f"FPS: {fps:.1f}")
        if det.get("pupil_detected"):
            conf = det.get("pupil_confidence", 0)
            pr = det.get("pupil_radius", det.get("pupil_r", 0))
            lines.append(
                f"Pupil: ({int(det.get('pupil_x', 0))}, "
                f"{int(det.get('pupil_y', 0))})  "
                f"r={int(pr)}  conf={conf:.2f}"
            )
        else:
            lines.append("Pupil: not detected")
        if det.get("limbus_detected"):
            lr = det.get("limbus_radius", det.get("limbus_r", 0))
            lines.append(f"Limbus: r={int(lr)}")

        quality = det.get("overall_quality", det.get("frame_quality", ""))
        if quality:
            lines.append(f"Quality: {quality}")

        latency = det.get("latency_ms", 0)
        if latency > 0:
            lines.append(f"Latency: {latency:.0f} ms")

        y0 = 25
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(
                vis,
                (8, y0 - th - 4),
                (14 + tw, y0 + 4),
                cls.BG_COLOR,
                -1,
            )
            cv2.putText(
                vis,
                line,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cls.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            y0 += th + 12

        return vis


# ======================================================================
# ACCURACY-FIRST + SPEED-OPTIMIZED Main Processor
# ======================================================================


class OptimizedVideoProcessor:
    """
    Accuracy-first video processor with batch inference.

    Plan-aligned changes:
        A2  â€” Default resolution 192 â†’ 320
        A5  â€” Suction ring masking in preprocessor
        A6  â€” ImageNormalizer in preprocessor
        S1  â€” Batch inference via FastInference.detect_batch()
        S3  â€” Batch collection from decode-ahead queue
        S4  â€” Bilateral filter removed from fast path

    Key architecture:
        - When using FastInference: batch BATCH_SIZE frames through
          GPU in a single forward pass (S1)
        - When using UnifiedDetector: frame-by-frame (batch not
          supported by detector pipeline)
        - Decode-ahead thread fills queue while GPU processes (S3)
    """

    def __init__(
        self,
        model_path: str = "models/best_model.pth",
        device: str = "auto",
        input_size: int = 320,  # A2: 192 â†’ 320 (accuracy-first)
        half_precision: bool = True,
        use_compile: bool = True,
        enable_auto_roi: bool = True,
        roi_cache_ttl: int = 20,
        roi_padding: float = 0.5,
        process_noise: float = 1.5,
        measurement_noise: float = 3.0,
        denoise_strength: int = 3,
        fast_mode: bool = True,
        skip_quality_check: bool = False,
        batch_size: int = _DEFAULT_BATCH_SIZE,  # S1: batch size
        frame_callback: Optional[Callable] = None,
        config=None,
        adaptive_quality: bool = True,
        adaptive_stable_frames: int = 4,
        adaptive_quality_skip_stride: int = 1,
    ):
        """
        Parameters
        ----------
        input_size : int
            Model resolution. 320 = accurate (default), 256 = balanced,
            192 = fast.
        batch_size : int
            Number of frames per GPU batch (S1). 4 is a good default.
        fast_mode : bool
            Skip bilateral filter in preprocessing (S4).
        skip_quality_check : bool
            Disable blur/brightness checks.
        """
        logger.info("Initialising ACCURACY-FIRST OptimizedVideoProcessor ...")

        runtime_profile = detect_runtime_profile()
        requested_batch_size = max(1, int(batch_size))
        self._batch_size = requested_batch_size
        self._enable_auto_roi = bool(enable_auto_roi)
        self._manual_roi: Optional[ManualCircularROI] = None
        self._detection_enabled = True
        self._manual_ring: Optional[ManualRingAnnotation] = None
        self._manual_ring_priors_path = (
            Path(__file__).resolve().parents[2] / "manual_ring_priors.json"
        )
        self._manual_ring_priors = self._load_manual_ring_priors()

        # --- ROI detector with longer cache ---
        self.roi_detector = (
            EyeROIDetector(
                cache_ttl=roi_cache_ttl,
                padding_ratio=roi_padding,
            )
            if self._enable_auto_roi
            else None
        )

        # --- Detection backend ---
        # Prefer FastInference for video/camera: single-scale forward
        # pass at 320Ã—320 is much faster than UnifiedDetector's
        # multi-scale inference (3 passes at 448/512/640).
        # UnifiedDetector is better suited for single-image analysis.
        self._use_unified = False
        self._detector = None
        self._fast_engine = None
        self._ring_detector = RingDetector(classifier_path=None)

        if _HAS_FAST:
            try:
                self._fast_engine = FastInference(
                    model_path=model_path,
                    device=device,
                    input_size=input_size,
                    half_precision=half_precision,
                    use_compile=use_compile,
                )
                logger.info("Using FastInference for video (single-scale, fast)")
            except Exception as exc:
                logger.warning(
                    "FastInference init failed (%s), trying UnifiedDetector", exc
                )

        if self._fast_engine is None and _HAS_UNIFIED:
            try:
                self._detector = UnifiedDetector(model_path=model_path, config=config)
                self._detector.init_video_mode(
                    input_size=input_size,
                    half_precision=half_precision,
                    use_compile=use_compile,
                    device=device,
                )
                self._use_unified = True
                logger.info("Using UnifiedDetector (full pipeline, fallback)")
            except Exception as exc:
                logger.warning("UnifiedDetector init failed (%s)", exc)

        if not self._use_unified and not self._fast_engine:
            raise RuntimeError(
                "Neither FastInference nor UnifiedDetector could be "
                "initialised. Check model_path and dependencies."
            )

        backend_device = None
        if self._fast_engine is not None:
            backend_device = getattr(getattr(self._fast_engine, "device", None), "type", None)
        if backend_device == "cpu":
            self._batch_size = 1
        elif requested_batch_size <= 1:
            self._batch_size = max(1, runtime_profile.recommended_batch_size)

        # --- Temporal smoother ---
        self.smoother = TemporalSmoother(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
        )

        # --- ACCURACY-FIRST preprocessor (A5, A6) ---
        self.preprocessor = VideoPreprocessor(
            denoise_strength=denoise_strength,
            fast_mode=fast_mode,
            apply_normalizer=True,  # A6
            suction_ring_removal=True,  # A5
        )

        # --- Frame quality checker ---
        self.quality_checker = FrameQualityChecker(
            skip_check=skip_quality_check,
        )

        self.frame_callback = frame_callback
        self._input_size = input_size
        self._latency_history_ms: deque[float] = deque(maxlen=120)
        self._processing_history_ms: deque[float] = deque(maxlen=120)
        self._roi_history_ms: deque[float] = deque(maxlen=120)
        self._ring_redetect_interval_locked = 8
        self._ring_redetect_interval_absent = 3
        self._ring_redetect_interval_present = 2
        self._quality_fail_count = 0
        self._processed_frames = 0
        self._stale_frames = 0
        self._dropped_frames = 0
        self._last_source_frame_idx: Optional[int] = None
        self._adaptive_quality = bool(adaptive_quality)
        self._adaptive_stable_frames = max(1, int(adaptive_stable_frames))
        self._adaptive_quality_skip_stride = max(
            0, int(adaptive_quality_skip_stride)
        )
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0
        self._quality_check_skip_total = 0
        self._last_quality_usable = True
        self._last_ring_status = RingStatus.ABSENT.value
        self._backend_device = str(backend_device or "cpu")
        self._latency_budget_ms = 80.0 if self._backend_device == "cuda" else 165.0
        self._overload_streak = 0
        self._cached_reuse_streak = 0
        self._cached_reuse_total = 0
        self._max_cached_reuse_burst = 2
        self._last_valid_result: Optional[Dict[str, Any]] = None
        self._processing_max_dim_base = 1280.0 if self._backend_device == "cuda" else 960.0
        self._processing_max_dim_unstable = 1024.0 if self._backend_device == "cuda" else 864.0
        self._processing_max_dim_overload = 896.0 if self._backend_device == "cuda" else 768.0
        self._stale_frame_threshold_base_s = 0.16 if self._backend_device == "cuda" else 0.16
        self._last_processing_scale = 1.0
        self._degraded_processing_active = False

        logger.info(
            "OptimizedVideoProcessor ready | backend=%s | resolution=%d "
            "| fast_mode=%s | roi_cache=%d | batch_size=%d",
            "UnifiedDetector" if self._use_unified else "FastInference",
            input_size,
            fast_mode,
            roi_cache_ttl,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Stats for GUI display
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current processor stats for GUI status display."""
        roi_mode = "manual" if self._manual_roi is not None else (
            "auto" if self._enable_auto_roi else "off"
        )
        latency_avg = (
            sum(self._latency_history_ms) / len(self._latency_history_ms)
            if self._latency_history_ms
            else 0.0
        )
        processing_avg = (
            sum(self._processing_history_ms) / len(self._processing_history_ms)
            if self._processing_history_ms
            else 0.0
        )
        roi_avg = (
            sum(self._roi_history_ms) / len(self._roi_history_ms)
            if self._roi_history_ms
            else 0.0
        )
        latency_recent = self._recent_metric_avg(self._latency_history_ms, n=12)
        recent_processing_items = [
            float(v) for v in list(self._processing_history_ms)[-12:] if float(v) > 1.0
        ]
        processing_recent = (
            float(sum(recent_processing_items) / len(recent_processing_items))
            if recent_processing_items
            else self._recent_metric_avg(self._processing_history_ms, n=12)
        )
        return {
            "resolution": self._input_size,
            "frame_skip": 0,
            "roi_active": roi_mode != "off",
            "roi_mode": roi_mode,
            "backend": "UnifiedDetector" if self._use_unified else "FastInference",
            "batch_size": self._batch_size,
            "latency_avg_ms": latency_avg,
            "processing_avg_ms": processing_avg,
            "latency_recent_ms": latency_recent,
            "processing_recent_ms": processing_recent,
            "roi_avg_ms": roi_avg,
            "quality_fail_count": self._quality_fail_count,
            "processed_frames": self._processed_frames,
            "stale_frames": self._stale_frames,
            "dropped_frames": self._dropped_frames,
            "stable_tracking_streak": self._stable_tracking_streak,
            "adaptive_quality_active": self._adaptive_quality_active(),
            "quality_check_skips": getattr(self, "_quality_check_skip_total", 0),
            "cached_reuse_total": getattr(self, "_cached_reuse_total", 0),
            "latency_budget_ms": getattr(self, "_latency_budget_ms", 0.0),
            "overload_active": getattr(self, "_overload_streak", 0) > 0,
            "degraded_processing_active": getattr(self, "_degraded_processing_active", False),
            "processing_scale": getattr(self, "_last_processing_scale", 1.0),
            "stale_frame_threshold_ms": self.get_stale_frame_threshold_s() * 1000.0,
        }

    def set_manual_roi(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Install a manual circular ROI in source-frame coordinates."""
        frame_height = None
        frame_width = None
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_height = int(frame_shape[0])
            frame_width = int(frame_shape[1])
        self._manual_roi = ManualCircularROI(
            center_x=float(center_x),
            center_y=float(center_y),
            radius=max(1.0, float(radius)),
            frame_width=frame_width,
            frame_height=frame_height,
        )
        self._detection_enabled = True
        self.smoother.reset()
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0

    def clear_manual_roi(self) -> None:
        """Clear the user-defined ROI and stop detection until one is set."""
        self._manual_roi = None
        self._detection_enabled = False
        self.smoother.reset()
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0

    def _roi_required_result(self, frame_idx: int, started_at: float) -> Dict[str, Any]:
        self.smoother.reset()
        self._last_valid_result = None
        result = {
            "pupil_detected": False,
            "limbus_detected": False,
            "frame_quality": "roi_required",
            "overall_quality": "roi_required",
            "overall_confidence": 0.0,
            "ring_status": RingStatus.ABSENT.value,
            "ring_confidence": 0.0,
            "ring_center_x": None,
            "ring_center_y": None,
            "ring_radius": None,
            "manual_roi_active": False,
            "manual_ring_active": False,
            "image_category": "pre_docked",
            "corneal_reference_source": "limbus",
            "frame_idx": frame_idx,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "roi_from_cache": False,
            "roi_is_closeup": False,
            "roi_time_ms": 0.0,
            "quality_check_skipped": False,
            "reuse_cached_result": False,
        }
        self._record_metrics(result, result["latency_ms"], 0.0)
        return result

    def _load_manual_ring_priors(self) -> Dict[str, Any]:
        try:
            if self._manual_ring_priors_path.exists():
                with self._manual_ring_priors_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return data
        except Exception as exc:
            logger.debug("Manual ring priors could not be loaded: %s", exc)
        return {}

    def _save_manual_ring_priors(self) -> None:
        try:
            self._manual_ring_priors_path.parent.mkdir(parents=True, exist_ok=True)
            with self._manual_ring_priors_path.open("w", encoding="utf-8") as fh:
                json.dump(self._manual_ring_priors, fh, indent=2)
        except Exception as exc:
            logger.debug("Manual ring priors could not be saved: %s", exc)

    def _reference_circle_for_ring(
        self, frame_shape: Tuple[int, ...]
    ) -> Tuple[float, float, float]:
        h, w = frame_shape[:2]
        if self._manual_roi is not None:
            if (
                self._manual_roi.frame_width is None
                or self._manual_roi.frame_height is None
                or (
                    int(self._manual_roi.frame_width) == int(w)
                    and int(self._manual_roi.frame_height) == int(h)
                )
            ):
                return (
                    float(self._manual_roi.center_x),
                    float(self._manual_roi.center_y),
                    max(1.0, float(self._manual_roi.radius)),
                )
        return (w / 2.0, h / 2.0, max(1.0, min(w, h) * 0.42))

    def _learn_manual_ring(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        frame_shape: Tuple[int, ...],
        dot_count: int = 12,
    ) -> None:
        ref_x, ref_y, ref_r = self._reference_circle_for_ring(frame_shape)
        sample = {
            "count": 1,
            "offset_x_ratio": (float(center_x) - ref_x) / ref_r,
            "offset_y_ratio": (float(center_y) - ref_y) / ref_r,
            "radius_ratio": float(radius) / ref_r,
            "dot_count": max(1.0, float(dot_count)),
        }
        priors = dict(self._manual_ring_priors) if self._manual_ring_priors else {}
        count = int(priors.get("count", 0))
        if count <= 0:
            priors = sample
        else:
            total = float(count + 1)
            priors["count"] = count + 1
            for key in ("offset_x_ratio", "offset_y_ratio", "radius_ratio", "dot_count"):
                priors[key] = (
                    float(priors.get(key, sample[key])) * count + float(sample[key])
                ) / total
        self._manual_ring_priors = priors
        self._save_manual_ring_priors()

    def _suggest_ring_from_priors(
        self, frame_shape: Tuple[int, ...]
    ) -> Optional[Dict[str, float]]:
        priors = self._manual_ring_priors
        if not priors or int(priors.get("count", 0)) <= 0:
            return None
        ref_x, ref_y, ref_r = self._reference_circle_for_ring(frame_shape)
        radius = max(8.0, ref_r * float(priors.get("radius_ratio", 1.0)))
        center_x = ref_x + ref_r * float(priors.get("offset_x_ratio", 0.0))
        center_y = ref_y + ref_r * float(priors.get("offset_y_ratio", 0.0))
        h, w = frame_shape[:2]
        center_x = float(np.clip(center_x, radius, max(radius, w - radius)))
        center_y = float(np.clip(center_y, radius, max(radius, h - radius)))
        return {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "dot_count": max(1.0, float(priors.get("dot_count", 12.0))),
        }

    def set_manual_ring(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        frame_shape: Optional[Tuple[int, ...]] = None,
        dot_count: int = 12,
    ) -> None:
        """Install a user-confirmed suction ring and reuse it until cleared."""
        frame_height = None
        frame_width = None
        if frame_shape is not None and len(frame_shape) >= 2:
            frame_height = int(frame_shape[0])
            frame_width = int(frame_shape[1])
        self._manual_ring = ManualRingAnnotation(
            center_x=float(center_x),
            center_y=float(center_y),
            radius=max(1.0, float(radius)),
            frame_width=frame_width,
            frame_height=frame_height,
            dot_count=max(1, int(dot_count)),
        )
        if frame_shape is not None:
            self._learn_manual_ring(
                center_x=center_x,
                center_y=center_y,
                radius=radius,
                frame_shape=frame_shape,
                dot_count=dot_count,
            )
        self.smoother.reset()
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0

    def clear_manual_ring(self) -> None:
        """Disable the manual ring and drop any stale ring priors so detection really stops."""
        self._manual_ring = None
        self._manual_ring_priors = {}
        self._last_ring_status = RingStatus.ABSENT.value
        self.smoother.reset()
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0

    def get_manual_ring(self, frame_shape: Optional[Tuple[int, ...]] = None) -> Optional[Dict[str, float]]:
        if self._manual_ring is not None:
            if frame_shape is None or (
                self._manual_ring.frame_width in (None, int(frame_shape[1]))
                and self._manual_ring.frame_height in (None, int(frame_shape[0]))
            ):
                return {
                    "center_x": float(self._manual_ring.center_x),
                    "center_y": float(self._manual_ring.center_y),
                    "radius": float(self._manual_ring.radius),
                    "dot_count": float(self._manual_ring.dot_count),
                }
        if frame_shape is None:
            return None
        return self._suggest_ring_from_priors(frame_shape)

    def _manual_ring_payload(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self._manual_ring is None or not self._manual_ring.matches_frame(frame):
            return None
        return {
            "ring_status": RingStatus.PRESENT.value,
            "ring_confidence": 1.0,
            "ring_dot_count": int(self._manual_ring.dot_count),
            "ring_boundary_ratio": 0.0,
            "ring_center_x": float(self._manual_ring.center_x),
            "ring_center_y": float(self._manual_ring.center_y),
            "ring_radius": float(self._manual_ring.radius),
            "ring_locked": True,
            "ring_method": "manual_user",
            "manual_ring_active": True,
        }

    def _apply_learned_ring_prior(
        self,
        center_x: Optional[float],
        center_y: Optional[float],
        radius: Optional[float],
        frame_shape: Tuple[int, ...],
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if center_x is None or center_y is None or radius is None:
            return center_x, center_y, radius
        prior = self._suggest_ring_from_priors(frame_shape)
        if prior is None:
            return center_x, center_y, radius
        expected_radius = max(1.0, float(prior["radius"]))
        center_delta = math.hypot(
            float(center_x) - float(prior["center_x"]),
            float(center_y) - float(prior["center_y"]),
        ) / expected_radius
        radius_delta = abs(float(radius) - expected_radius) / expected_radius
        if center_delta > 0.24 or radius_delta > 0.18:
            return center_x, center_y, radius
        blend = 0.25
        cx = (1.0 - blend) * float(center_x) + blend * float(prior["center_x"])
        cy = (1.0 - blend) * float(center_y) + blend * float(prior["center_y"])
        rr = (1.0 - blend) * float(radius) + blend * expected_radius
        return cx, cy, rr

    def _ring_within_active_roi(
        self,
        ring_center_x: Optional[float],
        ring_center_y: Optional[float],
        ring_radius: Optional[float],
        frame_shape: Tuple[int, ...],
    ) -> bool:
        if (
            ring_center_x is None
            or ring_center_y is None
            or ring_radius is None
            or self._manual_roi is None
        ):
            return True

        h, w = frame_shape[:2]
        if (
            self._manual_roi.frame_width is not None
            and self._manual_roi.frame_height is not None
            and (
                int(self._manual_roi.frame_width) != int(w)
                or int(self._manual_roi.frame_height) != int(h)
            )
        ):
            return True

        roi = self._manual_roi
        dist = math.hypot(ring_center_x - roi.center_x, ring_center_y - roi.center_y)
        return dist <= max(4.0, roi.radius * 0.08) and ring_radius <= roi.radius * 1.02

    def update_runtime_settings(
        self,
        *,
        enable_auto_roi: Optional[bool] = None,
        roi_cache_ttl: Optional[int] = None,
        process_noise: Optional[float] = None,
        measurement_noise: Optional[float] = None,
    ) -> None:
        """Apply safe runtime settings without rebuilding the processor."""
        if enable_auto_roi is not None:
            enable = bool(enable_auto_roi)
            self._enable_auto_roi = enable
            if enable:
                ttl = int(roi_cache_ttl or 20)
                if self.roi_detector is None:
                    self.roi_detector = EyeROIDetector(cache_ttl=ttl, padding_ratio=0.5)
                else:
                    self.roi_detector.cache_ttl = ttl
            else:
                self.roi_detector = None

        if roi_cache_ttl is not None and self.roi_detector is not None:
            self.roi_detector.cache_ttl = int(roi_cache_ttl)

        if process_noise is not None:
            self.smoother.process_noise = float(process_noise)

        if measurement_noise is not None:
            self.smoother.measurement_noise = float(measurement_noise)

    def reset(self) -> None:
        """Reset temporal state (smoother, ROI cache) between videos."""
        self.smoother.reset()
        if self.roi_detector is not None:
            self.roi_detector.reset()
        if self.preprocessor._temporal_filter is not None:
            self.preprocessor._temporal_filter.reset()
        self._latency_history_ms.clear()
        self._processing_history_ms.clear()
        self._roi_history_ms.clear()
        self._quality_fail_count = 0
        self._processed_frames = 0
        self._stale_frames = 0
        self._dropped_frames = 0
        self._last_source_frame_idx = None
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0
        self._quality_check_skip_total = 0
        self._last_quality_usable = True
        self._last_ring_status = RingStatus.ABSENT.value
        self._overload_streak = 0
        self._cached_reuse_streak = 0
        self._cached_reuse_total = 0
        self._last_valid_result = None
        self._last_processing_scale = 1.0
        self._degraded_processing_active = False

    def note_stale_frame(self) -> None:
        """Record a stale frame dropped before processing."""
        self._stale_frames += 1

    def _cache_valid_result(self, det: Dict[str, Any]) -> None:
        if bool(det.get("pupil_detected", False)) or bool(det.get("limbus_detected", False)):
            self._last_valid_result = dict(det)

    def _recent_metric_avg(self, history: deque[float], n: int = 10) -> float:
        if not history:
            return 0.0
        items = list(history)[-n:]
        return float(sum(items) / max(len(items), 1))

    def _processing_pressure_active(self) -> bool:
        recent_processing = self._recent_metric_avg(self._processing_history_ms, n=8)
        recent_latency = self._recent_metric_avg(self._latency_history_ms, n=8)
        return bool(
            recent_processing > self._latency_budget_ms * 1.05
            or recent_latency > self._latency_budget_ms * 1.10
            or self._overload_streak > 1
        )

    def _should_reuse_cached_result(self) -> bool:
        if self._last_valid_result is None:
            self._overload_streak = 0
            self._cached_reuse_streak = 0
            return False
        recent_processing = self._recent_metric_avg(self._processing_history_ms, n=8)
        recent_latency = self._recent_metric_avg(self._latency_history_ms, n=8)
        overload = (
            recent_processing > self._latency_budget_ms
            or recent_latency > self._latency_budget_ms * 1.20
            or (self._stale_frames > 0 and recent_latency > self._latency_budget_ms * 0.85)
        )
        if overload:
            self._overload_streak += 1
        else:
            self._overload_streak = 0
            self._cached_reuse_streak = 0
            return False
        if self._stable_tracking_streak < 3:
            return False
        if self._cached_reuse_streak >= self._max_cached_reuse_burst:
            self._cached_reuse_streak = 0
            return False
        self._cached_reuse_streak += 1
        self._cached_reuse_total += 1
        return True

    def should_shed_input_frames(self) -> bool:
        recent_latency = self._recent_metric_avg(self._latency_history_ms, n=6)
        overload_streak = int(getattr(self, "_overload_streak", 0))
        latency_budget_ms = float(getattr(self, "_latency_budget_ms", 165.0))
        return bool(
            overload_streak >= 3
            or recent_latency > latency_budget_ms * 1.35
        )

    def get_stale_frame_threshold_s(self) -> float:
        base = float(getattr(self, "_stale_frame_threshold_base_s", 0.16))
        if self.should_shed_input_frames():
            return max(0.12, base * 0.90)
        return base

    def _prepare_processing_crop(
        self, eye_crop: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        h, w = eye_crop.shape[:2]
        max_dim = float(max(h, w))
        cap = self._processing_max_dim_base
        if self._stable_tracking_streak < 2 and self._last_ring_status != RingStatus.PRESENT.value:
            cap = min(cap, self._processing_max_dim_unstable)
        if self._processing_pressure_active():
            cap = min(cap, self._processing_max_dim_overload)
        if max_dim <= cap:
            self._degraded_processing_active = False
            self._last_processing_scale = 1.0
            return eye_crop, 1.0, 1.0

        scale = cap / max_dim
        new_w = max(self._input_size, int(round(w * scale)))
        new_h = max(self._input_size, int(round(h * scale)))
        resized = cv2.resize(eye_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = float(w) / float(new_w)
        scale_y = float(h) / float(new_h)
        self._degraded_processing_active = True
        self._last_processing_scale = 1.0 / max(scale, 1e-6)
        return resized, scale_x, scale_y

    def _reuse_cached_result(
        self,
        *,
        frame_idx: int,
        roi: ROIResult,
        roi_ms: float,
        quality_skipped: bool,
        reason: str,
        started_at: float,
    ) -> Dict[str, Any]:
        cached = dict(self._last_valid_result or {})
        cached["frame_idx"] = frame_idx
        cached["roi_from_cache"] = roi.from_cache
        cached["roi_is_closeup"] = roi.is_closeup
        cached["manual_roi_active"] = self._manual_roi is not None
        cached["manual_ring_active"] = bool(cached.get("manual_ring_active", False))
        cached["roi_time_ms"] = roi_ms
        cached["quality_check_skipped"] = quality_skipped
        cached["frame_quality"] = "ok"
        cached["reuse_cached_result"] = True
        cached["reuse_reason"] = reason
        cached["processing_time_ms"] = 0.1
        cached["latency_ms"] = (time.perf_counter() - started_at) * 1000.0
        return cached

    def note_source_frame(self, source_frame_idx: int) -> None:
        """Track gaps in source-frame numbering as dropped frames."""
        if self._last_source_frame_idx is not None and source_frame_idx > self._last_source_frame_idx + 1:
            self._dropped_frames += source_frame_idx - self._last_source_frame_idx - 1
        self._last_source_frame_idx = source_frame_idx

    def _record_metrics(
        self,
        det: Dict[str, Any],
        latency_ms: float,
        roi_ms: float = 0.0,
    ) -> None:
        self._processed_frames += 1
        self._latency_history_ms.append(float(latency_ms))
        self._processing_history_ms.append(
            float(det.get("processing_time_ms", latency_ms))
        )
        self._roi_history_ms.append(float(roi_ms))
        if str(det.get("frame_quality", "")).lower() != "ok":
            self._quality_fail_count += 1
        self._update_tracking_state(det)

    def _adaptive_quality_active(self) -> bool:
        return (
            self._adaptive_quality
            and self._manual_roi is not None
            and self._last_quality_usable
            and self._stable_tracking_streak >= self._adaptive_stable_frames
            and self._adaptive_quality_skip_stride > 0
        )

    def _evaluate_frame_quality(self, image: np.ndarray) -> Tuple[bool, str, bool]:
        if self.quality_checker.skip_check:
            return True, "ok", False

        if (
            self._adaptive_quality_active()
            and self._quality_check_skip_count < self._adaptive_quality_skip_stride
        ):
            self._quality_check_skip_count += 1
            self._quality_check_skip_total += 1
            return True, "ok", True

        usable, reason = self.quality_checker.is_usable(image)
        self._last_quality_usable = usable
        self._quality_check_skip_count = 0
        if not usable:
            self._stable_tracking_streak = 0
        return usable, reason, False

    def _update_tracking_state(self, det: Dict[str, Any]) -> None:
        conf = float(det.get("overall_confidence", 0.0) or 0.0)
        is_stable = (
            bool(det.get("pupil_detected", False))
            and bool(det.get("limbus_detected", False))
            and str(det.get("frame_quality", "")).lower() == "ok"
            and conf >= 0.75
        )
        if is_stable:
            self._stable_tracking_streak += 1
            return
        self._stable_tracking_streak = 0
        self._quality_check_skip_count = 0

    def _resolve_roi(self, frame: np.ndarray) -> ROIResult:
        """Choose the ROI with manual selection taking priority."""
        fh, fw = frame.shape[:2]

        if self._manual_roi is not None and self._manual_roi.matches_frame(frame):
            cx = float(np.clip(self._manual_roi.center_x, 0, fw - 1))
            cy = float(np.clip(self._manual_roi.center_y, 0, fh - 1))
            radius = max(1.0, float(self._manual_roi.radius))

            x0 = max(0, int(np.floor(cx - radius)))
            y0 = max(0, int(np.floor(cy - radius)))
            x1 = min(fw, int(np.ceil(cx + radius)))
            y1 = min(fh, int(np.ceil(cy + radius)))

            cropped = frame[y0:y1, x0:x1]
            if cropped.size > 0:
                return ROIResult(
                    x=x0,
                    y=y0,
                    width=x1 - x0,
                    height=y1 - y0,
                    cropped=cropped,
                    is_closeup=False,
                    from_cache=True,
                    confidence=1.0,
                )

        if self.roi_detector is not None:
            return self.roi_detector.detect(frame)

        return ROIResult(
            x=0,
            y=0,
            width=fw,
            height=fh,
            cropped=frame,
            is_closeup=False,
            from_cache=False,
            confidence=1.0,
        )

    # ------------------------------------------------------------------
    # Single frame processing (for camera / GUI mode)
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> Dict[str, Any]:
        """
        Process a single frame. Used for live/GUI mode.

        For video files, prefer process_video() which uses batching.
        """
        t0 = time.perf_counter()
        if not getattr(self, "_detection_enabled", True):
            return self._roi_required_result(frame_idx, t0)
        roi = ROIResult(
            x=0,
            y=0,
            width=frame.shape[1],
            height=frame.shape[0],
            cropped=frame,
            is_closeup=False,
            from_cache=False,
            confidence=1.0,
        )
        roi_ms = 0.0
        quality_skipped = False
        try:
            # 1. ROI detection (~1-5ms, cached = 0.1ms)
            roi_t0 = time.perf_counter()
            roi = self._resolve_roi(frame)
            roi_ms = (time.perf_counter() - roi_t0) * 1000.0
            eye_crop = roi.cropped if roi.valid else frame

            # 2. Quality check
            usable, reason, quality_skipped = self._evaluate_frame_quality(eye_crop)

            if not usable:
                raw_dict = dict(
                    pupil_detected=False,
                    limbus_detected=False,
                    frame_quality=reason,
                    overall_quality=reason,
                    quality_check_skipped=quality_skipped,
                )
                smoothed = self.smoother.smooth(raw_dict)
                smoothed["frame_idx"] = frame_idx
                smoothed["latency_ms"] = (time.perf_counter() - t0) * 1000
                smoothed["roi_from_cache"] = roi.from_cache
                smoothed["roi_is_closeup"] = roi.is_closeup
                smoothed["manual_roi_active"] = self._manual_roi is not None
                smoothed["roi_time_ms"] = roi_ms
                smoothed["quality_check_skipped"] = quality_skipped
                self._record_metrics(smoothed, smoothed["latency_ms"], roi_ms)
                return smoothed

            if self._should_reuse_cached_result():
                reused = self._reuse_cached_result(
                    frame_idx=frame_idx,
                    roi=roi,
                    roi_ms=roi_ms,
                    quality_skipped=quality_skipped,
                    reason="frame_budget",
                    started_at=t0,
                )
                self._record_metrics(reused, reused["latency_ms"], roi_ms)
                return reused

            # 4. Detection
            if self._use_unified:
                # 3. Preprocessing needed for UnifiedDetector (A3, A5, A6)
                preprocessed = self.preprocessor.process(eye_crop)
                eye_result = self._detector.detect_video_frame(
                    preprocessed,
                    frame_number=frame_idx,
                    roi_x=roi.x,
                    roi_y=roi.y,
                )
                raw_dict = self._detector.result_to_dict(eye_result)
                raw_dict["frame_quality"] = "ok"
            else:
                locked_ring = None
                should_redetect_ring = False
                manual_ring = self._manual_ring_payload(frame)
                proc_crop = eye_crop
                proc_to_eye_x = 1.0
                proc_to_eye_y = 1.0
                eye_h, eye_w = eye_crop.shape[:2]
                self._degraded_processing_active = False
                self._last_processing_scale = 1.0
                if manual_ring is None:
                    proc_crop, proc_to_eye_x, proc_to_eye_y = self._prepare_processing_crop(
                        eye_crop
                    )
                ring_result = (
                    self._ring_detector.detect(proc_crop) if should_redetect_ring else None
                )
                scale_x = eye_w / self._input_size
                scale_y = eye_h / self._input_size
                raw_dict = self._fast_engine.detect(
                    proc_crop,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    offset_x=float(roi.x),
                    offset_y=float(roi.y),
                )
                raw_dict["frame_quality"] = "ok"
                _annotate_quality(raw_dict)
                if manual_ring is not None:
                    raw_dict.update(manual_ring)
                elif ring_result is not None:
                    raw_dict["ring_status"] = ring_result.status.value
                    raw_dict["ring_confidence"] = ring_result.confidence
                    raw_dict["ring_dot_count"] = getattr(ring_result, "dot_count", 0)
                    raw_dict["ring_boundary_ratio"] = float(
                        getattr(ring_result, "details", {}).get("outer_boundary_ratio", 1.0)
                    )
                    if ring_result.ring_center is not None:
                        raw_dict["ring_center_x"] = (
                            ring_result.ring_center[0] * proc_to_eye_x + float(roi.x)
                        )
                        raw_dict["ring_center_y"] = (
                            ring_result.ring_center[1] * proc_to_eye_y + float(roi.y)
                        )
                    else:
                        raw_dict["ring_center_x"] = None
                        raw_dict["ring_center_y"] = None
                    raw_dict["ring_radius"] = (
                        ring_result.ring_radius * max(proc_to_eye_x, proc_to_eye_y)
                        if ring_result.ring_radius is not None
                        else None
                    )
                elif locked_ring is not None:
                    raw_dict.update(locked_ring)
                else:
                    raw_dict["ring_status"] = "ring_absent"
                    raw_dict["ring_confidence"] = 0.0
                    raw_dict["ring_dot_count"] = 0
                    raw_dict["ring_boundary_ratio"] = 1.0
                    raw_dict["ring_center_x"] = None
                    raw_dict["ring_center_y"] = None
                    raw_dict["ring_radius"] = None
                if (
                    raw_dict.get("ring_status") == RingStatus.PRESENT.value
                    and not raw_dict.get("manual_ring_active", False)
                ):
                    cx, cy, rr = self._apply_learned_ring_prior(
                        raw_dict.get("ring_center_x"),
                        raw_dict.get("ring_center_y"),
                        raw_dict.get("ring_radius"),
                        frame.shape,
                    )
                    raw_dict["ring_center_x"] = cx
                    raw_dict["ring_center_y"] = cy
                    raw_dict["ring_radius"] = rr
                if not self._ring_within_active_roi(
                    raw_dict.get("ring_center_x"),
                    raw_dict.get("ring_center_y"),
                    raw_dict.get("ring_radius"),
                    frame.shape,
                ):
                    raw_dict["ring_status"] = "ring_absent"
                    raw_dict["ring_confidence"] = 0.0
                    raw_dict["ring_dot_count"] = 0
                    raw_dict["ring_center_x"] = None
                    raw_dict["ring_center_y"] = None
                    raw_dict["ring_radius"] = None
                    raw_dict["manual_ring_active"] = False
                raw_dict["image_category"] = (
                    "docked"
                    if raw_dict.get("ring_status") == RingStatus.PRESENT.value
                    else "pre_docked"
                )
                raw_dict["corneal_reference_source"] = (
                    "suction_ring"
                    if raw_dict.get("ring_status") == RingStatus.PRESENT.value
                    and raw_dict.get("ring_center_x") is not None
                    else "limbus"
                )
                
                # Conditional Limbus Shrink
                # The ML model naturally tends to overestimate the iris/limbus
                # boundary in pre-docked images. Apply a conservative shrink
                # to help keep the detected limbus inside the expected eye ROI.
                if raw_dict.get("image_category") == "pre_docked" and raw_dict.get("limbus_detected", False):
                    shrink = 0.93
                    for key in ["limbus_radius", "limbus_r", "limbus_major", "limbus_minor"]:
                        if raw_dict.get(key) is not None:
                            raw_dict[key] = float(raw_dict[key]) * shrink

                eye_result = None
            raw_dict["quality_check_skipped"] = quality_skipped

            # 5. Temporal smoothing
            smoothed = self.smoother.smooth(raw_dict)
            if smoothed.get("ring_status") == "ring_present" and smoothed.get("ring_center_x") is not None:
                smoothed["image_category"] = "docked"
                smoothed["corneal_reference_source"] = "suction_ring"
            else:
                smoothed["image_category"] = "pre_docked"
                smoothed["corneal_reference_source"] = "limbus"

            # 6. Apply smoothed values back
            if self._use_unified and eye_result is not None:
                self._detector.apply_smoothed_dict(eye_result, smoothed)
                smoothed["_eye_result"] = eye_result

            # 7. Metadata
            smoothed["frame_idx"] = frame_idx
            smoothed["roi_from_cache"] = roi.from_cache
            smoothed["roi_is_closeup"] = roi.is_closeup
            smoothed["manual_roi_active"] = self._manual_roi is not None
            smoothed["manual_ring_active"] = bool(raw_dict.get("manual_ring_active", False))
            smoothed["latency_ms"] = (time.perf_counter() - t0) * 1000
            smoothed["roi_time_ms"] = roi_ms
            smoothed["reuse_cached_result"] = False
            self._last_ring_status = str(smoothed.get("ring_status", RingStatus.ABSENT.value))
            self._cache_valid_result(smoothed)
            self._record_metrics(smoothed, smoothed["latency_ms"], roi_ms)
            return smoothed
        except Exception as exc:
            logger.exception("process_frame failed, reusing last valid result if available: %s", exc)
            gc.collect()
            if self._last_valid_result is not None:
                reused = self._reuse_cached_result(
                    frame_idx=frame_idx,
                    roi=roi,
                    roi_ms=roi_ms,
                    quality_skipped=quality_skipped,
                    reason="exception_fallback",
                    started_at=t0,
                )
                self._record_metrics(reused, reused["latency_ms"], roi_ms)
                return reused
            raise

    # ------------------------------------------------------------------
    # S1: Prepare a batch of frames (ROI + quality + preprocess)
    # ------------------------------------------------------------------

    def _prepare_batch(
        self, raw_items: List[Tuple[int, np.ndarray]]
    ) -> Tuple[
        List[Dict[str, Any]],  # prepared items for batch inference
        List[Dict[str, Any]],  # already-resolved results (unusable frames)
    ]:
        """
        For each raw (frame_idx, frame):
            - detect ROI
            - check quality
            - preprocess

        Returns two lists:
            prepared : dicts with keys frame_idx, frame, roi,
                       preprocessed, t0 â€” ready for batch ML inference
            resolved : dicts that are already final results
                       (unusable frames get smoothed immediately)
        """
        prepared = []
        resolved = []

        for frame_idx, frame in raw_items:
            t0 = time.perf_counter()

            if not getattr(self, "_detection_enabled", True):
                resolved.append(self._roi_required_result(frame_idx, t0))
                continue

            roi_t0 = time.perf_counter()
            roi = self._resolve_roi(frame)
            roi_ms = (time.perf_counter() - roi_t0) * 1000.0
            eye_crop = roi.cropped if roi.valid else frame

            usable, reason, quality_skipped = self._evaluate_frame_quality(eye_crop)

            if not usable:
                raw_dict = dict(
                    pupil_detected=False,
                    limbus_detected=False,
                    frame_quality=reason,
                    overall_quality=reason,
                    quality_check_skipped=quality_skipped,
                )
                smoothed = self.smoother.smooth(raw_dict)
                smoothed["frame_idx"] = frame_idx
                smoothed["latency_ms"] = (time.perf_counter() - t0) * 1000
                smoothed["roi_from_cache"] = roi.from_cache
                smoothed["roi_is_closeup"] = roi.is_closeup
                smoothed["manual_roi_active"] = self._manual_roi is not None
                smoothed["roi_time_ms"] = roi_ms
                smoothed["quality_check_skipped"] = quality_skipped
                self._record_metrics(smoothed, smoothed["latency_ms"], roi_ms)
                resolved.append(smoothed)
                continue

            if self._use_unified:
                preprocessed = self.preprocessor.process(eye_crop)
            else:
                preprocessed, _, _ = self._prepare_processing_crop(eye_crop)

            prepared.append(
                {
                    "frame_idx": frame_idx,
                    "frame": frame,
                    "roi": roi,
                    "roi_ms": roi_ms,
                    "quality_check_skipped": quality_skipped,
                    "preprocessed": preprocessed,
                    "t0": t0,
                }
            )

        return prepared, resolved

    # ------------------------------------------------------------------
    # S1: Batch inference + post-processing
    # ------------------------------------------------------------------

    def _infer_batch_fast(self, prepared: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run batch ML inference on prepared frames using FastInference,
        then apply temporal smoothing per-frame (sequential).

        Returns list of final result dicts in input order.
        """
        if not prepared:
            return []

        # Collect inputs for detect_batch
        images = [p["preprocessed"] for p in prepared]
        offsets = [(float(p["roi"].x), float(p["roi"].y)) for p in prepared]

        try:
            batch_dets = self._fast_engine.detect_batch(images, offsets=offsets)
        except Exception as exc:
            logger.exception("Batch inference failed, falling back to cached/single-frame results: %s", exc)
            gc.collect()
            batch_dets = []
            for info in prepared:
                if self._last_valid_result is not None:
                    det = self._reuse_cached_result(
                        frame_idx=info["frame_idx"],
                        roi=info["roi"],
                        roi_ms=info.get("roi_ms", 0.0),
                        quality_skipped=bool(info.get("quality_check_skipped", False)),
                        reason="batch_exception_fallback",
                        started_at=info["t0"],
                    )
                else:
                    det = self.process_frame(info["frame"], info["frame_idx"])
                    det["already_final_batch_result"] = True
                batch_dets.append(det)

        results = []
        for info, raw_dict in zip(prepared, batch_dets):
            if raw_dict.get("already_final_batch_result", False):
                results.append(raw_dict)
                continue
            if raw_dict.get("reuse_cached_result", False):
                self._record_metrics(
                    raw_dict, raw_dict.get("latency_ms", 0.0), raw_dict.get("roi_time_ms", 0.0)
                )
                results.append(raw_dict)
                continue
            raw_dict["frame_quality"] = "ok"

            # Temporal smoothing (sequential â€” order preserved)
            _annotate_quality(raw_dict)
            smoothed = self.smoother.smooth(raw_dict)

            smoothed["frame_idx"] = info["frame_idx"]
            smoothed["roi_from_cache"] = info["roi"].from_cache
            smoothed["roi_is_closeup"] = info["roi"].is_closeup
            smoothed["manual_roi_active"] = self._manual_roi is not None
            smoothed["latency_ms"] = (time.perf_counter() - info["t0"]) * 1000
            smoothed["roi_time_ms"] = info.get("roi_ms", 0.0)
            smoothed["quality_check_skipped"] = bool(
                info.get("quality_check_skipped", False)
            )
            self._record_metrics(
                smoothed, smoothed["latency_ms"], smoothed["roi_time_ms"]
            )

            results.append(smoothed)

        return results

    # ------------------------------------------------------------------
    # Video processing â€” main loop with batching (S1, S3)
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        stride: int = 1,
        max_frames: Optional[int] = None,
        show_preview: bool = False,
        resize_output: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a video file with batch inference (S1).

        When using FastInference backend, frames are collected into
        batches of ``batch_size`` and processed in a single GPU
        forward pass.  When using UnifiedDetector, processing is
        frame-by-frame (detector does not support batching).
        """
        if self.roi_detector is not None:
            self.roi_detector.reset()
        self.smoother.reset()
        if self._use_unified:
            self._detector.reset()

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if resize_output:
            out_w, out_h = resize_output
        else:
            out_w, out_h = fw, fh

        estimated_proc = total_frames // stride if total_frames > 0 else 0
        if max_frames:
            estimated_proc = min(estimated_proc, max_frames)

        logger.info(
            "ACCURACY-FIRST Video: %s  %dx%d  %.1f fps  %d frames  "
            "stride=%d  batch=%d  â†’ ~%d to process",
            input_path,
            fw,
            fh,
            fps_in,
            total_frames,
            stride,
            self._batch_size,
            estimated_proc,
        )

        # Writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps_in / stride, (out_w, out_h)
            )
            if not writer.isOpened():
                # Fallback codec
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    output_path, fourcc, fps_in / stride, (out_w, out_h)
                )

        # CSV writer
        csv_file = None
        csv_writer = None
        if csv_path:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "frame_idx",
                    "pupil_detected",
                    "pupil_x",
                    "pupil_y",
                    "pupil_radius",
                    "pupil_confidence",
                    "limbus_detected",
                    "limbus_x",
                    "limbus_y",
                    "limbus_radius",
                    "latency_ms",
                    "overall_quality",
                ]
            )

        # S3: Decode-ahead thread
        frame_queue: queue.Queue = queue.Queue(maxsize=128)
        reader = _FrameReader(cap, frame_queue, stride, max_frames)
        reader.start()

        results: List[Dict[str, Any]] = []
        processed = 0
        t_start = time.perf_counter()
        fps_display = 0.0

        # Determine whether to use batch or frame-by-frame
        use_batch = (
            not self._use_unified
            and self._fast_engine is not None
            and self._batch_size > 1
        )

        try:
            if use_batch:
                # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # S1: BATCH PROCESSING LOOP
                # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                while True:
                    # Collect up to batch_size frames from queue
                    raw_items: List[Tuple[int, np.ndarray]] = []
                    sentinel_received = False

                    for _ in range(self._batch_size):
                        try:
                            item = frame_queue.get(timeout=30.0)
                        except queue.Empty:
                            sentinel_received = True
                            break
                        if item is None:
                            sentinel_received = True
                            break
                        raw_items.append(item)

                    if not raw_items and sentinel_received:
                        break

                    # Prepare: ROI + quality + preprocess
                    prepared, resolved = self._prepare_batch(raw_items)

                    # Add already-resolved (unusable) frames to results
                    for r in resolved:
                        results.append(r)
                        processed += 1

                    # Batch ML inference on prepared frames
                    batch_results = self._infer_batch_fast(prepared)

                    # Finalise each result (output, csv, callback)
                    for det in batch_results:
                        results.append(det)
                        processed += 1

                        elapsed = time.perf_counter() - t_start
                        if elapsed > 0:
                            fps_display = processed / elapsed

                        fidx = det.get("frame_idx", 0)

                        # Progress logging
                        if processed % 200 == 0 or processed == 1:
                            latency = det.get("latency_ms", 0)
                            pct = (
                                processed / estimated_proc * 100
                                if estimated_proc > 0
                                else 0
                            )
                            logger.info(
                                "Frame %d/%d (%.0f%%) %.1f ms/frame %.1f FPS",
                                processed,
                                estimated_proc,
                                pct,
                                latency,
                                fps_display,
                            )

                        # Find original frame for overlay
                        orig_frame = None
                        for fi, fr in raw_items:
                            if fi == fidx:
                                orig_frame = fr
                                break

                        # Write / display / callback
                        if orig_frame is not None and (
                            writer or show_preview or self.frame_callback
                        ):
                            vis = _OverlayRenderer.draw(
                                orig_frame, det, fidx, fps_display
                            )
                            if resize_output:
                                vis = cv2.resize(
                                    vis,
                                    (out_w, out_h),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            if writer:
                                writer.write(vis)
                            if show_preview:
                                cv2.imshow("Optimized Processor (Batch)", vis)
                                if cv2.waitKey(1) & 0xFF == ord("q"):
                                    reader.stop()
                                    sentinel_received = True
                            if self.frame_callback:
                                try:
                                    self.frame_callback(fidx, vis, det)
                                except Exception as cb_err:
                                    logger.warning("Callback error: %s", cb_err)

                        # CSV row
                        if csv_writer:
                            self._write_csv_row(csv_writer, det)

                    if sentinel_received:
                        break

            else:
                # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                # FRAME-BY-FRAME LOOP (UnifiedDetector path)
                # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                while True:
                    item = frame_queue.get(timeout=30.0)
                    if item is None:
                        break

                    frame_idx, frame = item
                    det = self.process_frame(frame, frame_idx)
                    results.append(det)
                    processed += 1

                    elapsed = time.perf_counter() - t_start
                    if elapsed > 0:
                        fps_display = processed / elapsed

                    if processed % 200 == 0 or processed == 1:
                        latency = det.get("latency_ms", 0)
                        pct = (
                            processed / estimated_proc * 100
                            if estimated_proc > 0
                            else 0
                        )
                        logger.info(
                            "Frame %d/%d (%.0f%%) %.1f ms/frame %.1f FPS",
                            processed,
                            estimated_proc,
                            pct,
                            latency,
                            fps_display,
                        )

                    if writer or show_preview or self.frame_callback:
                        vis = _OverlayRenderer.draw(frame, det, frame_idx, fps_display)
                        if resize_output:
                            vis = cv2.resize(
                                vis,
                                (out_w, out_h),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        if writer:
                            writer.write(vis)
                        if show_preview:
                            cv2.imshow("Optimized Processor", vis)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                reader.stop()
                                break
                        if self.frame_callback:
                            try:
                                self.frame_callback(frame_idx, vis, det)
                            except Exception as cb_err:
                                logger.warning("Callback error: %s", cb_err)

                    if csv_writer:
                        self._write_csv_row(csv_writer, det)

        except queue.Empty:
            logger.warning("Frame reader timed out")
        finally:
            reader.stop()
            reader.join(timeout=5.0)
            cap.release()
            if writer:
                writer.release()
            if csv_file:
                csv_file.close()
            if show_preview:
                cv2.destroyAllWindows()

        # Summary
        total_time = time.perf_counter() - t_start
        avg_latency = (
            float(np.mean([r.get("latency_ms", 0) for r in results])) if results else 0
        )
        detected_count = sum(1 for r in results if r.get("pupil_detected"))

        logger.info(
            "DONE (ACCURACY-FIRST)  %d frames in %.1f s  |  "
            "avg %.1f ms/frame  |  %.1f effective FPS  |  "
            "%d/%d pupil detected (%.0f%%)  |  batch=%s",
            processed,
            total_time,
            avg_latency,
            processed / total_time if total_time > 0 else 0,
            detected_count,
            processed,
            detected_count / processed * 100 if processed > 0 else 0,
            "yes" if use_batch else "no",
        )

        return results

    # ------------------------------------------------------------------
    # CSV helper
    # ------------------------------------------------------------------

    @staticmethod
    def _write_csv_row(csv_writer, det: Dict[str, Any]) -> None:
        """Write a single detection result to CSV."""
        csv_writer.writerow(
            [
                det.get("frame_idx", 0),
                det.get("pupil_detected", False),
                f"{det.get('pupil_x', 0):.1f}",
                f"{det.get('pupil_y', 0):.1f}",
                f"{det.get('pupil_radius', det.get('pupil_r', 0)):.1f}",
                f"{det.get('pupil_confidence', 0):.3f}",
                det.get("limbus_detected", False),
                f"{det.get('limbus_x', 0):.1f}",
                f"{det.get('limbus_y', 0):.1f}",
                f"{det.get('limbus_radius', det.get('limbus_r', 0)):.1f}",
                f"{det.get('latency_ms', 0):.1f}",
                det.get("overall_quality", ""),
            ]
        )

    # ------------------------------------------------------------------
    # Camera processing (unchanged except target_fps default)
    # ------------------------------------------------------------------

    def process_camera(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        target_fps: int = 40,
        flip_horizontal: bool = False,
        save_frames_dir: Optional[str] = None,
    ):
        """Process live camera feed."""
        self.roi_detector.reset()
        self.smoother.reset()
        if self._use_unified:
            self._detector.reset()

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        if save_frames_dir:
            Path(save_frames_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Camera %d opened at %s @ %d fps",
            camera_id,
            resolution,
            target_fps,
        )

        frame_idx = 0
        fps_counter = 0
        fps_time = time.perf_counter()
        display_fps = 0.0
        frame_delay = max(1, int(1000 / target_fps))
        consecutive_failures = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        logger.warning(
                            "Camera read failed %d times â€” stopping",
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                if flip_horizontal:
                    frame = cv2.flip(frame, 1)

                det = self.process_frame(frame, frame_idx)

                fps_counter += 1
                now = time.perf_counter()
                if now - fps_time >= 1.0:
                    display_fps = fps_counter / (now - fps_time)
                    fps_counter = 0
                    fps_time = now

                vis = _OverlayRenderer.draw(frame, det, frame_idx, display_fps)
                cv2.imshow("Pupil Tracker â€” Live", vis)

                if self.frame_callback:
                    try:
                        self.frame_callback(frame_idx, vis, det)
                    except Exception:
                        pass

                key = cv2.waitKey(frame_delay) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and save_frames_dir:
                    save_path = str(
                        Path(save_frames_dir) / f"frame_{frame_idx:06d}.jpg"
                    )
                    cv2.imwrite(save_path, vis)
                    logger.info("Saved %s", save_path)

                frame_idx += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()

    @property
    def detector(self) -> Optional["UnifiedDetector"]:
        """Access the underlying UnifiedDetector (or None)."""
        return self._detector if self._use_unified else None

    @staticmethod
    def save_results_json(results: List[Dict[str, Any]], path: str):
        """Save results list to a JSON file."""

        def _sanitise(v):
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v

        clean = []
        for r in results:
            row = {k: _sanitise(v) for k, v in r.items() if not k.startswith("_")}
            clean.append(row)

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info("Results saved to %s  (%d frames)", path, len(clean))


# ======================================================================
# SUPPORT CLASSES FOR ASYNC PROCESSING
# ======================================================================


class TrackingQuality:
    """Enum-like class for tracking quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    OK = "ok"
    POOR = "poor"
    LOST = "lost"


class FrameResult(dict):
    """
    Result from processing a single frame.

    Extends dict with convenient attribute access.
    Contains detection results, metadata, and quality metrics.
    """

    def __getattr__(self, key: str) -> Any:
        """Allow attribute-style access to dict keys."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"FrameResult has no attribute '{key}'")

    def __setattr__(self, key: str, val: Any) -> None:
        """Allow attribute-style setting of dict keys."""
        self[key] = val


class AsyncCapture(threading.Thread):
    """
    Asynchronous camera capture thread.

    Captures frames from camera in background thread,
    providing non-blocking frame retrieval for real-time processing.
    """

    def __init__(self, camera_id: int = 0, buffer_size: int = 2):
        """
        Initialize async camera capture.

        Args:
            camera_id: Camera device ID (default: 0)
            buffer_size: Size of frame buffer (older frames dropped)
        """
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.buffer_size = buffer_size

        self.cap = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.running = False
        self._lock = threading.Lock()
        self._exception = None
        self._frame_count = 0

    def run(self) -> None:
        """Capture frames in background thread."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            self.running = True
            consecutive_failures = 0

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        logger.warning(
                            "Camera read failed %d times â€” stopping capture",
                            consecutive_failures,
                        )
                        break
                    continue
                consecutive_failures = 0

                self._frame_count += 1
                item = (self._frame_count, frame, time.time())

                # Drop old frames if buffer is full
                try:
                    self.frame_queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.frame_queue.put_nowait(item)
                    except queue.Full:
                        pass

        except Exception as e:
            with self._lock:
                self._exception = e
            logger.error("AsyncCapture error: %s", e)
        finally:
            if self.cap is not None:
                self.cap.release()
            self.running = False

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Get the latest frame from camera.

        Args:
            timeout: Timeout in seconds for retrieving frame

        Returns:
            Frame (np.ndarray) or None if no frame available
        """
        try:
            item = self.frame_queue.get(timeout=timeout)
            if isinstance(item, tuple):
                return item[1]
            return item
        except queue.Empty:
            return None

    def read(self, timeout: float = 0.1):
        """
        Get the latest frame with metadata, draining stale frames.

        Always returns the most recent frame in the buffer so that
        live camera processing stays as close to real-time as possible.

        Args:
            timeout: Timeout in seconds for retrieving frame

        Returns:
            Tuple of (frame_number, frame, timestamp) or None
        """
        latest = None
        try:
            latest = self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        # Drain any remaining stale frames, keep only the newest
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def stop(self) -> None:
        """Stop capture thread."""
        self.running = False
        if self.is_alive():
            self.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()

    def get_error(self) -> Optional[Exception]:
        """Get any exception that occurred in background thread."""
        with self._lock:
            return self._exception

