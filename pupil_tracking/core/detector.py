# pupil_tracking/core/detector.py
"""
Unified detection orchestrator.

This is the SINGLE entry point for all detection -- images, video
frames, with or without suction ring.  It orchestrates:

    1. Grayscale mode handling (auto/force/off)          ← NEW
    2. Ring detection (classify docked vs pre-docked)
    3. Adaptive preprocessing (ring-aware)
    4. ML segmentation (primary)
    5. Smart contour fitting (circle vs ellipse auto-selection)
    6. Ring-constrained contour filtering
    7. Classical CV fallback (if ML fails entirely)
    8. Cross-validation between pupil, limbus, and ring
    9. Auto-calibration from limbus (px -> mm)
   10. Corneal centre + offset calculation (in mm)
   11. Quality grading (ring-aware)

Plan-aligned changes:
    - Cross-validation ratio relaxed 0.80 → 0.85 for surgical dilated pupils
    - init_video_mode() for video pipeline integration
    - detect_video_frame() with ROI offset handling
    - result_to_dict() / apply_smoothed_dict() for temporal smoother
    - detect_from_masks() for batch inference pipeline
    - Ring detection and adaptive pipeline integration
    - force_mode parameter for manual docked/pre_docked override
    - Ring-aware classical fallback with spatial constraints
    - Ring geometry in result_to_dict() output
    - Grayscale mode support (auto/force/off) with GUI toggle (NEW)
    - ONNX Runtime backend support for production distribution (NEW)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.ml.postprocess import (
    validate_pupil_limbus_pair,
    extract_ring_from_segmentation,
    RingSegmentationResult,
)
from pupil_tracking.core.corneal_center import CornealCenterCalculator
from pupil_tracking.core.ellipse_fitter import EllipseFitter
from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult, FitType
from pupil_tracking.core.deterministic_ring_detector import (
    RingDetector,
    RingDetectionResult,
    RingStatus,
)
from pupil_tracking.preprocessing.ring_aware import (
    RingAwarePreprocessor,
    AdaptiveContourFilter,
)
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    PupilDetection,
    LimbusDetection,
    CalibrationInfo,
    EllipseParams,
    FrameMetadata,
    DetectionMethod,
    DetectionQuality,
    ANATOMICAL_LIMITS,
    assign_quality_grade,
)
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger
from pupil_tracking.calibration.spatial_calibration import StabilizedCalibrator

from pupil_tracking.preprocessing.grayscale_handler import (
    GrayscaleHandler,
    GrayscaleMode,
    GrayscaleInfo,
)

logger = logging.getLogger(__name__)


class UnifiedDetector:
    """Production detection pipeline with smart circle/ellipse fitting
    and ring-aware adaptive processing.

    The detector automatically classifies each image as *docked*
    (suction ring present) or *pre-docked* (no ring) and adjusts
    preprocessing, contour filtering, and cross-validation accordingly.

    The SmartContourFitter automatically decides whether each
    structure (pupil, limbus) is best described by a circle or
    an ellipse, based on residual analysis and circularity metrics.

    Grayscale mode can be set to ``"auto"`` (detect and enhance
    grayscale inputs), ``"force"`` (always convert to enhanced
    grayscale), or ``"off"`` (pass through unchanged).  The mode
    can be toggled at runtime via :meth:`set_grayscale_mode` for
    GUI integration.

    The ML backend is auto-selected:
    - **ONNX Runtime** (preferred for distribution): 50MB, fast CPU
    - **PyTorch** (fallback for development): full training support

    Usage
    -----
    >>> detector = UnifiedDetector()
    >>> result = detector.detect(image_bgr)
    >>> if result.has_both:
    ...     print(result.corneal_center.offset_magnitude_mm)
    >>>
    >>> # Check ring status
    >>> print(result.ring_status)   # "ring_present" or "ring_absent"
    >>> print(result.image_category)  # "docked" or "pre_docked"
    >>>
    >>> # Force a specific mode
    >>> result = detector.detect(image_bgr, force_mode="docked")
    >>>
    >>> # Toggle grayscale mode (GUI button)
    >>> detector.set_grayscale_mode("force")
    >>> result = detector.detect(image_bgr)  # now processes as grayscale
    """

    # ================================================================
    # Construction
    # ================================================================

    def __init__(
        self,
        model_path: Optional[str] = None,
        ring_classifier_path: Optional[str] = None,
        config=None,
        grayscale_mode: str = "off",
    ) -> None:
        self.cfg = config or get_config()
        self.logger = get_logger()

        # --- ML backbone (auto-selects ONNX Runtime or PyTorch) ---
        self.ml_engine = self._create_ml_engine(model_path)

        # --- Smart fitter (circle-vs-ellipse auto-selection) ---
        self._fitter = SmartContourFitter(
            circularity_threshold=0.92,
            residual_ratio_threshold=1.15,
            use_ransac=True,
            subpixel_refine=True,
        )

        # --- Ring detector (NEW) ---
        if ring_classifier_path is None:
            ring_classifier_path = getattr(
                self.cfg.paths,
                "ring_classifier_path",
                "models/ring_classifier.pth",
            )
        self._ring_detector = RingDetector(
            classifier_path=ring_classifier_path,
            device=getattr(self.cfg.model, "device", "auto"),
            use_heuristic_fallback=True,
        )

        # --- Ring-aware preprocessor (NEW) ---
        self._ring_preprocessor = RingAwarePreprocessor()

        # --- Ring-aware contour filter (NEW) ---
        self._ring_contour_filter = AdaptiveContourFilter()

        # --- Corneal-centre calculator ---
        self.corneal_calc = CornealCenterCalculator(config=self.cfg)

        # --- Calibration state ---
        self._calibration = CalibrationInfo()

        # --- Stabilized calibrator (EMA + outlier rejection) ---
        corneal_mm = getattr(
            self.cfg.calibration,
            "corneal_diameter_mm",
            12.0,
        )
        self._stabilized_cal = StabilizedCalibrator(
            config=self.cfg.measurement_stabilization,
            corneal_diameter_mm=corneal_mm,
            mode=getattr(self.cfg.calibration, "mode", "ANATOMICAL_ANCHOR"),
            manual_px_per_mm=getattr(self.cfg.calibration, "manual_px_per_mm", None),
            ring_diameter_mm=getattr(self.cfg.calibration, "suction_ring_diameter_mm", 9.4),
        )

        # --- Video mode state ---
        self._video_mode = False
        self._video_fast_engine = None

        # --- Ring state (persists across frames for video) ---
        self._last_ring_result: Optional[RingDetectionResult] = None
        self._ring_stable_count: int = 0

        # --- Grayscale handler ---
        self._grayscale_handler = GrayscaleHandler(
            clahe_clip_limit=3.0,
            clahe_grid_size=(8, 8),
            channel_diff_threshold=3.0,
        )
        self._grayscale_mode = GrayscaleMode.from_string(grayscale_mode)
        self._last_grayscale_info: Optional[GrayscaleInfo] = None

        self.logger.info(
            "UnifiedDetector initialised (ML=%s, SmartFitter=enabled, "
            "RingDetector=%s, GrayscaleMode=%s)",
            "available" if self.ml_engine.available else "unavailable",
            "classifier+heuristic"
            if self._ring_detector.classifier is not None
            else "heuristic-only",
            self._grayscale_mode.name,
        )

    # ================================================================
    # ML engine creation (ONNX Runtime preferred, PyTorch fallback)
    # ================================================================

    def _create_ml_engine(self, model_path: Optional[str] = None):
        """
        Create the ML inference engine, preferring ONNX Runtime for
        production (50MB, fast CPU) with PyTorch fallback for development.

        The returned engine has the same interface:
          - .detect(image) or .infer(image) -> masks/result
          - .available (bool)
        """
        # ── Try ONNX Runtime first (production / distribution) ──
        try:
            from pupil_tracking.ml.onnx_inference import ONNXInference

            onnx_engine = ONNXInference(
                input_size=getattr(self.cfg.model, "input_size", 512),
                num_classes=getattr(self.cfg.model, "num_classes", 3),
                use_quantized=True,
                enable_gpu=getattr(self.cfg.model, "device", "auto") != "cpu",
            )

            if onnx_engine.is_loaded:
                self.logger.info(
                    "ML backend: ONNX Runtime (%s)",
                    onnx_engine.get_device_info().get("model", "unknown"),
                )
                return _ONNXEngineWrapper(onnx_engine, self.cfg)

        except ImportError:
            self.logger.debug("onnxruntime not installed, trying PyTorch")
        except Exception as e:
            self.logger.debug("ONNX backend failed: %s, trying PyTorch", e)

        # ── Fallback to PyTorch (development) ──
        try:
            from pupil_tracking.ml.inference import SegmentationInference

            engine = SegmentationInference(model_path=model_path, config=self.cfg)
            self.logger.info("ML backend: PyTorch")
            return engine
        except Exception as e:
            self.logger.error("Failed to create any ML backend: %s", e)
            return _DummyEngine()

    # ================================================================
    # Video mode initialisation
    # ================================================================

    def init_video_mode(
        self,
        input_size: int = 320,
        half_precision: bool = True,
        use_compile: bool = True,
        device: str = "auto",
    ) -> None:
        """Initialise video-specific settings.

        Called by OptimizedVideoProcessor to configure the detector
        for per-frame video inference with appropriate speed/accuracy
        trade-offs.

        Parameters
        ----------
        input_size : int
            Model input resolution for video frames.
        half_precision : bool
            Use FP16 on CUDA.
        device : str
            Device string ("cuda", "cpu", "auto").
        """
        self._video_mode = True

        # Enable temporal red light filtering for video
        self.set_red_light_temporal_mode(True)

        # Try to initialise a fast engine for video if ML engine
        # supports it, otherwise we just use the standard path
        # with adjusted parameters
        try:
            from pupil_tracking.ml.fast_inference import FastInference

            model_path = (
                self.ml_engine.model_path
                if hasattr(self.ml_engine, "model_path")
                else None
            )

            if model_path:
                self._video_fast_engine = FastInference(
                    model_path=model_path,
                    device=device,
                    input_size=input_size,
                    half_precision=half_precision,
                    use_compile=use_compile,
                )
                self.logger.info(
                    "Video mode: FastInference initialised (size=%d, half=%s)",
                    input_size,
                    half_precision,
                )
            else:
                self.logger.info(
                    "Video mode: using standard ML engine "
                    "(no model_path for FastInference)"
                )
        except Exception as exc:
            self.logger.warning(
                "Video mode: FastInference unavailable (%s), using standard ML engine",
                exc,
            )

    # ================================================================
    # Grayscale mode control
    # ================================================================

    @property
    def grayscale_mode(self) -> GrayscaleMode:
        """Current grayscale processing mode.

        Returns
        -------
        GrayscaleMode
            One of ``AUTO``, ``FORCE``, ``OFF``.
        """
        return self._grayscale_mode

    def set_grayscale_mode(self, mode: str | GrayscaleMode) -> None:
        """Change the grayscale processing mode at runtime.

        This is the method the GUI toggle button should call.

        Parameters
        ----------
        mode : str or GrayscaleMode
            ``"auto"`` — detect and enhance grayscale inputs.
            ``"force"`` — always convert to enhanced grayscale.
            ``"off"`` — pass through unchanged (default).
        """
        if isinstance(mode, str):
            new_mode = GrayscaleMode.from_string(mode)
        else:
            new_mode = mode

        if new_mode != self._grayscale_mode:
            self.logger.info(
                "Grayscale mode changed: %s → %s",
                self._grayscale_mode.name,
                new_mode.name,
            )
            self._grayscale_mode = new_mode

    @property
    def last_grayscale_info(self) -> Optional[GrayscaleInfo]:
        """Diagnostic info from the most recent grayscale processing.

        Returns ``None`` if no image has been processed yet or if
        grayscale mode is ``OFF``.

        Useful for the GUI to display whether grayscale enhancement
        was actually applied and what the contrast improvement was.
        """
        return self._last_grayscale_info

    def _apply_grayscale_mode(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[GrayscaleInfo]]:
        """Apply grayscale processing based on the current mode.

        This is called early in the detection pipeline, after format
        normalisation (image is guaranteed BGR 3-channel) but before
        ring detection and ML inference.

        Behaviour by mode
        ~~~~~~~~~~~~~~~~~

        - **OFF**: Return image unchanged, info=None.  This is the
          default and preserves the original pipeline exactly.
        - **AUTO**: Detect whether the image is grayscale.  If yes,
          enhance with CLAHE and replicate to 3 channels.  If no,
          pass through unchanged.
        - **FORCE**: Always convert to grayscale, enhance, and
          replicate to 3 channels — regardless of input type.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image of shape ``(H, W, 3)``.

        Returns
        -------
        (processed_image, grayscale_info)
            processed_image : np.ndarray
                ``(H, W, 3)`` uint8 image ready for the pipeline.
            grayscale_info : GrayscaleInfo or None
                Diagnostic metadata, or ``None`` if mode is OFF.
        """
        if self._grayscale_mode == GrayscaleMode.OFF:
            return image, None

        force = self._grayscale_mode == GrayscaleMode.FORCE
        processed, info = self._grayscale_handler.to_model_input(
            image,
            force_grayscale=force,
        )

        return processed, info

    # ================================================================
    # Red Light Filter Control
    # ================================================================

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable the red light filter.

        The red light filter specifically targets bright red LED/surgical
        lights that can temporarily distort pupil detection when they
        appear on the eye.

        Parameters
        ----------
        enabled : bool
            True to enable filtering, False to disable.
        """
        if hasattr(self.ml_engine, 'set_red_light_filter_enabled'):
            self.ml_engine.set_red_light_filter_enabled(enabled)
        self.logger.info("Red light filter enabled=%s", enabled)

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        """Enable temporal mode for red light filtering.

        Temporal mode tracks red light positions across video frames
        for more stable detection of blinking lights. This should
        be enabled when processing video streams.

        Parameters
        ----------
        enabled : bool
            True to enable temporal tracking, False to disable.
        """
        if hasattr(self.ml_engine, 'set_red_light_temporal_mode'):
            self.ml_engine.set_red_light_temporal_mode(enabled)
        self.logger.info("Red light temporal mode enabled=%s", enabled)

    def reset_red_light_temporal(self) -> None:
        """Reset temporal tracking state for red light filter.

        Call this when starting a new video or sequence to clear
        any cached temporal information.
        """
        if hasattr(self.ml_engine, 'reset_red_light_temporal'):
            self.ml_engine.reset_red_light_temporal()

    # ================================================================
    # Main detection entry point
    # ================================================================

    def detect(
        self,
        image: np.ndarray,
        frame_number: int = -1,
        source: str = "",
        force_mode: Optional[str] = None,
    ) -> EyeDetectionResult:
        """Detect pupil, limbus, corneal centre, and offset.

        All measurements include mm conversion when calibration
        is available (auto-calibrated from limbus diameter on
        first successful detection).

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image of shape (H, W, 3).
        frame_number : int
            Sequential frame index (-1 for single images).
        source : str
            Optional label for provenance tracking.
        force_mode : str or None
            Override ring auto-detection:
            ``"docked"`` — assume ring present.
            ``"pre_docked"`` — assume no ring.
            ``None`` — auto-detect (default).

        Returns
        -------
        EyeDetectionResult
            Always returned, never ``None``.
        """
        t0 = time.time()

        # -- Grayscale normalisation (Phase 1) -------------------------
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Apply grayscale mode processing
        image, grayscale_info = self._apply_grayscale_mode(image)
        self._last_grayscale_info = grayscale_info

        # -- Step 0: Ring detection ------------------------------------
        ring_result = self._detect_ring(image, force_mode)
        is_docked = ring_result.status in (
            RingStatus.PRESENT,
            RingStatus.PARTIAL,
        )

        # -- Step 1: Adaptive preprocessing ----------------------------
        prep_result = self._ring_preprocessor.preprocess(image, ring_result)

        # -- Step 2: ML segmentation -----------------------------------
        result = self.ml_engine.detect(image, frame_number=frame_number, source=source)

        # Attach ring info to result
        self._attach_ring_info(result, ring_result)

        # -- Step 3: Re-fit masks with SmartContourFitter --------------
        if hasattr(result, "_raw_mask") and result._raw_mask is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract ring from 4-class segmentation if available
            ring_seg = None
            raw_mask = result._raw_mask
            if raw_mask.max() >= 3:
                ring_seg = extract_ring_from_segmentation(raw_mask)
                # Cross-reference with ring detector result
                if ring_seg.detected and not is_docked:
                    # Segmentation found ring but classifier didn't —
                    # trust the segmentation as additional signal
                    is_docked = True
                    self._attach_ring_seg_info(result, ring_seg)

            pupil_fit, limbus_fit = self._extract_structure(
                raw_mask,
                gray,
                ring_result=ring_result,
            )
            self._apply_fit_to_result(
                result, 
                pupil_fit, 
                limbus_fit,
                force_limbus_overwrite=(not is_docked)
            )

        # -- Step 4: Classical fallback --------------------------------
        dc = self.cfg.detection
        if dc.enable_classical_fallback:
            if not result.pupil.detected:
                classical_pupil = self._classical_pupil(
                    image,
                    ring_result=ring_result,
                )
                if classical_pupil.detected:
                    classical_pupil.confidence *= dc.classical_confidence_penalty
                    result.pupil = classical_pupil

            if not result.limbus.detected:
                pupil_hint = None
                if result.pupil.detected and result.pupil.ellipse is not None:
                    pupil_hint = result.pupil.ellipse
                classical_limbus = self._classical_limbus(
                    image,
                    pupil_hint=pupil_hint,
                    ring_result=ring_result,
                )
                if classical_limbus.detected:
                    classical_limbus.confidence *= dc.classical_confidence_penalty
                    result.limbus = classical_limbus

        # -- Step 4b: Pre-docked limbus shrink correction ------------
        if (
            not is_docked
            and result.limbus.detected
            and result.limbus.ellipse is not None
        ):
            shrink_factor = self.cfg.detection.pre_docked_limbus_shrink_factor
            result.limbus.ellipse.set_radius(
                result.limbus.ellipse.radius * shrink_factor
            )

        # -- Step 5: Cross-validation and rejection --------------------
        if result.has_both:
            result = self._cross_validate_and_reject(
                result,
                ring_result=ring_result,
            )

        # -- Step 6: Calibration (modular: anatomical / fixed scale / ring) ----
        cal_mode = getattr(self._stabilized_cal, "mode", "ANATOMICAL_ANCHOR")
        if cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            self._calibration = self._stabilized_cal.update_from_limbus(result.limbus)
        elif cal_mode == "RING_REFLECTION":
            if ring_result.ring_radius is not None:
                self._calibration = self._stabilized_cal.update_from_ring(ring_result.ring_radius)
            elif (
                result.limbus.detected
                and result.limbus.ellipse is not None
                and result.limbus.confidence > 0.5
            ):
                cal = self._stabilized_cal.update_from_limbus(result.limbus)
                if cal.calibrated:
                    self._calibration = cal
        elif (
            result.limbus.detected
            and result.limbus.ellipse is not None
            and result.limbus.confidence > 0.5
        ):
            cal = self._stabilized_cal.update_from_limbus(result.limbus)
            if cal.calibrated:
                self._calibration = cal
        elif (
            not self._calibration.calibrated
            and is_docked
            and ring_result.ring_radius is not None
        ):
            self._calibrate_from_ring(ring_result)

        result.calibration = self._calibration

        # -- Step 7: Attach mm values ----------------------------------
        if self._calibration.calibrated:
            self._add_mm_values(result)

        # ============================================================
        # Heuristic: if limbus appears under-estimated while docked,
        # and we have a ring inner radius estimate, replace limbus
        # geometry with ring-based estimate (conservative correction).
        # This avoids the common failure where the ring or markers are
        # mistakenly fitted as the limbus producing very small limbus
        # diameters in mm (< 12 mm). We reduce confidence to reflect
        # the heuristic replacement.
        # ============================================================
        try:
            if (
                getattr(result, "ring_status", "pre_docked") == "ring_present"
                and getattr(result, "ring_inner_radius", None) is not None
                and result.limbus.detected
                and result.limbus.radius_mm is not None
            ):
                limbus_dia_mm = result.limbus.radius_mm * 2.0
                if limbus_dia_mm < 12.0:
                    ring_inner_px = float(result.ring_inner_radius)
                    if ring_inner_px > 4.0:
                        # Replace limbus ellipse with circle at ring centre
                        if getattr(result, "ring_center", None) is not None:
                            cx, cy = result.ring_center
                        else:
                            cx, cy = (result.limbus.ellipse.center_x, result.limbus.ellipse.center_y)

                        ep = result.limbus.ellipse
                        # Set semi-axes to ring_inner_px * 0.98 (slightly inside)
                        new_r = ring_inner_px * 0.98
                        ep.center_x = cx
                        ep.center_y = cy
                        ep.semi_major = new_r
                        ep.semi_minor = new_r
                        # Reduce confidence to indicate heuristic
                        result.limbus.confidence = min(result.limbus.confidence * 0.6, 0.85)
                        result.limbus.quality = assign_quality_grade(result.limbus.confidence)
                        # Update mm values
                        if self._calibration.calibrated:
                            result.limbus.radius_mm = ep.radius * self._calibration.mm_per_px
                            result.limbus.center_mm = self._calibration.point_px_to_mm((ep.center_x, ep.center_y))
                            self.logger.debug("Heuristic limbus replaced from ring_inner (%.1f px)", ring_inner_px)
        except Exception:
            pass

        # -- Step 8: Corneal centre + offset ---------------------------
        result.corneal_center = self.corneal_calc.calculate(
            result.pupil, result.limbus, self._calibration
        )
        if result.pupil.detected and result.pupil.ellipse is not None:
            self._blend_corneal_center_from_available(result, ring_result)

        # -- Step 9: Overall quality (ring-aware) ----------------------
        confs: list[float] = []
        if result.pupil.detected:
            confs.append(result.pupil.confidence)
        if result.limbus.detected:
            confs.append(result.limbus.confidence)
        # Ring confidence contributes modestly
        confs.append(ring_result.confidence * 0.3)

        if confs:
            result.overall_confidence = float(np.mean(confs))
        result.overall_quality = assign_quality_grade(result.overall_confidence)

        result.metadata.processing_time_ms = (time.time() - t0) * 1000.0

        self.logger.debug(
            "[%s] %s | pupil=%s (%.3f) | limbus=%s (%.3f) | "
            "ring=%s (%.3f) | gray=%s | quality=%s | %.1fms",
            source,
            "docked" if is_docked else "pre_docked",
            result.pupil.detected,
            result.pupil.confidence,
            result.limbus.detected,
            result.limbus.confidence,
            ring_result.status.value,
            ring_result.confidence,
            (grayscale_info.conversion_applied if grayscale_info else False),
            result.overall_quality,
            result.metadata.processing_time_ms,
        )

        return result

    def _blend_corneal_center_from_available(
        self,
        result: EyeDetectionResult,
        ring_result: RingDetectionResult,
    ) -> None:
        """Blend pupil, limbus, and suction-ring centres when available."""
        if result.pupil.ellipse is None:
            return

        pupil = result.pupil.ellipse
        points: list[tuple[float, float, str]] = [
            (pupil.center_x, pupil.center_y, "pupil"),
        ]
        weights: list[float] = [max(result.pupil.confidence, 1e-3)]

        if result.limbus.detected and result.limbus.ellipse is not None:
            points.append(
                (
                    result.limbus.ellipse.center_x,
                    result.limbus.ellipse.center_y,
                    "limbus",
                )
            )
            weights.append(max(result.limbus.confidence, 1e-3))

        if (
            ring_result.status == RingStatus.PRESENT
            and ring_result.ring_center is not None
        ):
            points.append(
                (ring_result.ring_center[0], ring_result.ring_center[1], "ring")
            )
            weights.append(max(ring_result.confidence, 1e-3))

        total_w = sum(weights)
        cx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
        cy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
        result.corneal_reference_source = "+".join(name for _, _, name in points)

        ox = pupil.center_x - cx
        oy = pupil.center_y - cy
        mag_px = math.hypot(ox, oy)

        result.corneal_center.valid = True
        result.corneal_center.center_px = (cx, cy)
        result.corneal_center.offset_px = (ox, oy)
        result.corneal_center.offset_magnitude_px = mag_px
        result.corneal_center.offset_angle_deg = math.degrees(math.atan2(oy, ox))
        result.corneal_center.confidence = min(
            result.pupil.confidence,
            max(
                ring_result.confidence if ring_result.status == RingStatus.PRESENT else 0.0,
                result.limbus.confidence if result.limbus.detected else 0.0,
            ),
        )
        if self._calibration.calibrated:
            result.corneal_center.center_mm = self._calibration.point_px_to_mm((cx, cy))
            dx_mm = ox * self._calibration.mm_per_px
            dy_mm = oy * self._calibration.mm_per_px
            result.corneal_center.offset_mm = (dx_mm, dy_mm)
            result.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)

    # ================================================================
    # Ring detection
    # ================================================================

    def _detect_ring(
        self,
        image: np.ndarray,
        force_mode: Optional[str] = None,
    ) -> RingDetectionResult:
        """Detect suction ring presence, with caching for video.

        In video mode, the ring status is unlikely to change between
        frames, so we cache the result and only re-detect periodically
        or when confidence drops.
        """
        if force_mode == "docked":
            return RingDetectionResult(
                status=RingStatus.PRESENT,
                confidence=1.0,
                method="forced",
            )
        elif force_mode == "pre_docked":
            return RingDetectionResult(
                status=RingStatus.ABSENT,
                confidence=1.0,
                method="forced",
            )

        # In video mode, reuse stable ring result for speed
        if (
            self._video_mode
            and self._last_ring_result is not None
            and self._ring_stable_count < 30  # Re-check every 30 frames
            and self._last_ring_result.confidence >= 0.80
        ):
            self._ring_stable_count += 1
            return self._last_ring_result

        # Full ring detection
        ring_result = self._ring_detector.detect(image)

        # Update stability tracking
        if (
            self._last_ring_result is not None
            and ring_result.status == self._last_ring_result.status
        ):
            self._ring_stable_count += 1
        else:
            self._ring_stable_count = 0

        self._last_ring_result = ring_result
        return ring_result

    def _attach_ring_info(
        self,
        result: EyeDetectionResult,
        ring_result: RingDetectionResult,
    ) -> None:
        """Attach ring detection info to the EyeDetectionResult.

        Adds ring-related attributes to the result object. These
        attributes may not exist on the base EyeDetectionResult
        class, so we set them dynamically for forward compatibility.
        """
        result.ring_status = ring_result.status.value
        result.ring_confidence = ring_result.confidence
        result.ring_contour = ring_result.ring_contour
        result.ring_center = ring_result.ring_center
        result.ring_radius = ring_result.ring_radius
        result.ring_mask = ring_result.ring_mask
        result.ring_dot_centers = list(getattr(ring_result, "dot_centers", []))
        result.ring_dot_count = int(getattr(ring_result, "dot_count", 0))
        result.ring_method = ring_result.method
        result.corneal_reference_source = getattr(
            ring_result,
            "corneal_reference_source",
            "limbus",
        )
        result.image_category = (
            "docked"
            if ring_result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
            else "pre_docked"
        )

    def _attach_ring_seg_info(
        self,
        result: EyeDetectionResult,
        ring_seg: RingSegmentationResult,
    ) -> None:
        """Update ring info from segmentation mask analysis."""
        if ring_seg.detected:
            if ring_seg.center is not None:
                result.ring_center = ring_seg.center
            if ring_seg.radius is not None:
                result.ring_radius = ring_seg.radius
            # Attach inner radius estimate when available for downstream heuristics
            if getattr(ring_seg, "inner_radius", None) is not None:
                result.ring_inner_radius = ring_seg.inner_radius
            result.ring_status = RingStatus.PRESENT.value

    # ================================================================
    # Video frame detection
    # ================================================================

    def detect_video_frame(
        self,
        image: np.ndarray,
        frame_number: int = 0,
        roi_x: float = 0.0,
        roi_y: float = 0.0,
        force_mode: Optional[str] = None,
    ) -> EyeDetectionResult:
        """Detect on a video frame, accounting for ROI offset.

        When the frame has been cropped to an eye ROI, the coordinates
        need to be offset back to full-frame space.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed BGR crop of the eye region.
        frame_number : int
            Frame index in the video.
        roi_x, roi_y : float
            Top-left corner of the ROI in the original frame.
        force_mode : str or None
            Override ring detection: ``"docked"``, ``"pre_docked"``,
            or ``None`` (auto).

        Returns
        -------
        EyeDetectionResult
        """
        result = self.detect(
            image,
            frame_number=frame_number,
            source="video",
            force_mode=force_mode,
        )

        # Offset coordinates from ROI space to full-frame space
        if roi_x != 0.0 or roi_y != 0.0:
            self._offset_result(result, roi_x, roi_y)

        return result

    def _offset_result(
        self,
        result: EyeDetectionResult,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """Shift all coordinates by (offset_x, offset_y)."""
        if result.pupil.detected and result.pupil.ellipse is not None:
            result.pupil.ellipse.center_x += offset_x
            result.pupil.ellipse.center_y += offset_y

        if result.limbus.detected and result.limbus.ellipse is not None:
            result.limbus.ellipse.center_x += offset_x
            result.limbus.ellipse.center_y += offset_y

        if result.corneal_center is not None and hasattr(
            result.corneal_center, "center_x"
        ):
            result.corneal_center.center_x += offset_x
            result.corneal_center.center_y += offset_y

        # Offset ring coordinates too
        if hasattr(result, "ring_center") and result.ring_center is not None:
            rx, ry = result.ring_center
            result.ring_center = (rx + offset_x, ry + offset_y)

    # ================================================================
    # detect_from_masks — for batch inference pipeline
    # ================================================================

    def detect_from_masks(
        self,
        image_bgr: np.ndarray,
        pupil_mask: np.ndarray,
        iris_mask: np.ndarray,
        pupil_confidence: float = 0.5,
        limbus_confidence: float = 0.5,
        ring_mask: Optional[np.ndarray] = None,
        force_mode: Optional[str] = None,
    ) -> EyeDetectionResult:
        """Build detection result from pre-computed segmentation masks.

        Used by the batch inference pipeline where FastInference has
        already produced masks and we want SmartContourFitter +
        cross-validation + calibration.

        Parameters
        ----------
        image_bgr : np.ndarray
            Original BGR image (for classical fallback and sub-pixel
            refinement).
        pupil_mask : np.ndarray
            Binary mask (0/255) for the pupil region.
        iris_mask : np.ndarray
            Binary mask (0/255) for the iris+pupil region.
        pupil_confidence, limbus_confidence : float
            ML-derived confidence values (A4).
        ring_mask : np.ndarray or None
            Binary mask (0/255) for the ring region (4-class models).
        force_mode : str or None
            Override ring detection mode.

        Returns
        -------
        EyeDetectionResult
        """
        t0 = time.time()

        # -- Grayscale normalisation (Phase 1) -------------------------
        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        # Apply grayscale mode processing
        image_bgr, grayscale_info = self._apply_grayscale_mode(image_bgr)
        self._last_grayscale_info = grayscale_info

        # Ring detection
        ring_result = self._detect_ring(image_bgr, force_mode)

        result = EyeDetectionResult()
        result.metadata = FrameMetadata()
        self._attach_ring_info(result, ring_result)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Build integer label mask: 0=bg, 1=pupil, 2=iris-only, 3=ring
        label_mask = np.zeros(pupil_mask.shape[:2], dtype=np.uint8)
        if ring_mask is not None:
            label_mask[ring_mask > 127] = 3
        label_mask[iris_mask > 127] = 2
        label_mask[pupil_mask > 127] = 1

        # Smart fit
        pupil_fit, limbus_fit = self._extract_structure(
            label_mask,
            gray,
            ring_result=ring_result,
        )

        # Apply pupil fit
        if pupil_fit is not None and pupil_fit.valid:
            ep = self._fit_result_to_ellipse_params(pupil_fit)
            fit_conf = self._fit_result_confidence(pupil_fit)
            # Combine ML confidence with fit confidence
            combined_conf = (pupil_confidence + fit_conf) / 2.0

            result.pupil.detected = True
            result.pupil.ellipse = ep
            result.pupil.confidence = combined_conf
            result.pupil.quality = assign_quality_grade(combined_conf)
            result.pupil.method = DetectionMethod.ML
            result.pupil.fit_type = pupil_fit.fit_type.value
            if pupil_fit.contour_points is not None:
                result.pupil.contour_points = pupil_fit.contour_points

        # Apply limbus fit
        if limbus_fit is not None and limbus_fit.valid:
            ep = self._fit_result_to_ellipse_params(limbus_fit)
            fit_conf = self._fit_result_confidence(limbus_fit)
            combined_conf = (limbus_confidence + fit_conf) / 2.0

            result.limbus.detected = True
            result.limbus.ellipse = ep
            result.limbus.confidence = combined_conf
            result.limbus.quality = assign_quality_grade(combined_conf)
            result.limbus.method = DetectionMethod.ML
            result.limbus.fit_type = limbus_fit.fit_type.value
            if limbus_fit.contour_points is not None:
                result.limbus.contour_points = limbus_fit.contour_points

        # Classical fallback
        dc = self.cfg.detection
        if dc.enable_classical_fallback:
            if not result.pupil.detected:
                classical_pupil = self._classical_pupil(
                    image_bgr,
                    ring_result=ring_result,
                )
                if classical_pupil.detected:
                    classical_pupil.confidence *= dc.classical_confidence_penalty
                    result.pupil = classical_pupil

            if not result.limbus.detected:
                pupil_hint = None
                if result.pupil.detected and result.pupil.ellipse is not None:
                    pupil_hint = result.pupil.ellipse
                classical_limbus = self._classical_limbus(
                    image_bgr,
                    pupil_hint=pupil_hint,
                    ring_result=ring_result,
                )
                if classical_limbus.detected:
                    classical_limbus.confidence *= dc.classical_confidence_penalty
                    result.limbus = classical_limbus

        # Cross-validation
        if result.has_both:
            result = self._cross_validate_and_reject(
                result,
                ring_result=ring_result,
            )

        # Calibration (stabilized)
        if (
            result.limbus.detected
            and result.limbus.ellipse is not None
            and result.limbus.confidence > 0.5
        ):
            cal = self._stabilized_cal.update_from_limbus(result.limbus)
            if cal.calibrated:
                self._calibration = cal

        if (
            not self._calibration.calibrated
            and ring_result.status == RingStatus.PRESENT
            and ring_result.ring_radius is not None
        ):
            self._calibrate_from_ring(ring_result)

        result.calibration = self._calibration

        if self._calibration.calibrated:
            self._add_mm_values(result)

        # Corneal centre
        result.corneal_center = self.corneal_calc.calculate(
            result.pupil, result.limbus, self._calibration
        )
        if result.pupil.detected and result.pupil.ellipse is not None:
            self._blend_corneal_center_from_available(result, ring_result)

        # Quality
        confs: list[float] = []
        if result.pupil.detected:
            confs.append(result.pupil.confidence)
        if result.limbus.detected:
            confs.append(result.limbus.confidence)
        confs.append(ring_result.confidence * 0.3)
        if confs:
            result.overall_confidence = float(np.mean(confs))
        result.overall_quality = assign_quality_grade(result.overall_confidence)

        result.metadata.processing_time_ms = (time.time() - t0) * 1000.0
        return result

    # ================================================================
    # Result → dict conversion (for temporal smoother)
    # ================================================================

    def result_to_dict(self, result: EyeDetectionResult) -> Dict[str, Any]:
        """Convert EyeDetectionResult to flat dict for temporal smoother.

        Keys are compatible with FastInference.detect() output so that
        the temporal smoother and CSV writer work identically regardless
        of which backend produced the detection.

        Includes ring-related and grayscale-related fields.
        """
        d: Dict[str, Any] = {
            "pupil_detected": False,
            "pupil_x": 0.0,
            "pupil_y": 0.0,
            "pupil_radius": 0.0,
            "pupil_r": 0.0,
            "pupil_confidence": 0.0,
            "limbus_detected": False,
            "limbus_x": 0.0,
            "limbus_y": 0.0,
            "limbus_radius": 0.0,
            "limbus_r": 0.0,
            "limbus_confidence": 0.0,
            "overall_quality": str(result.overall_quality)
            if result.overall_quality
            else "",
            "processing_time_ms": (
                result.metadata.processing_time_ms if result.metadata else 0.0
            ),
        }

        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            d["pupil_detected"] = True
            d["pupil_x"] = e.center_x
            d["pupil_y"] = e.center_y
            d["pupil_radius"] = e.radius
            d["pupil_r"] = e.radius
            d["pupil_confidence"] = result.pupil.confidence

            # Optional ellipse details
            d["pupil_major"] = e.semi_major
            d["pupil_minor"] = e.semi_minor
            d["pupil_angle"] = e.angle_deg

        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            d["limbus_detected"] = True
            d["limbus_x"] = e.center_x
            d["limbus_y"] = e.center_y
            d["limbus_radius"] = e.radius
            d["limbus_r"] = e.radius
            d["limbus_confidence"] = result.limbus.confidence
            d["wtw_horizontal_mm"] = getattr(result.limbus, "wtw_horizontal_mm", None)
            d["wtw_vertical_mm"] = getattr(result.limbus, "wtw_vertical_mm", None)
            d["wtw_mean_mm"] = getattr(result.limbus, "wtw_mean_mm", None)
            d["wtw_astigmatism_mm"] = getattr(result.limbus, "wtw_astigmatism_mm", None)
            d["is_wtw_measured"] = getattr(result.limbus, "is_wtw_measured", False)
            d["wtw_validity_status"] = getattr(result.limbus, "wtw_validity_status", "UNAVAILABLE")


        # Corneal center offset
        if result.corneal_center is not None:
            cc = result.corneal_center
            if hasattr(cc, "offset_magnitude_mm"):
                d["corneal_offset_mm"] = cc.offset_magnitude_mm
            if hasattr(cc, "offset_angle_deg"):
                d["corneal_offset_angle"] = cc.offset_angle_deg

        # Ring info
        d["ring_status"] = getattr(result, "ring_status", "unknown")
        d["ring_confidence"] = getattr(result, "ring_confidence", 0.0)
        d["image_category"] = getattr(result, "image_category", "unknown")
        d["corneal_reference_source"] = getattr(
            result,
            "corneal_reference_source",
            "limbus",
        )

        ring_center = getattr(result, "ring_center", None)
        if ring_center is not None:
            d["ring_center_x"] = ring_center[0]
            d["ring_center_y"] = ring_center[1]
        else:
            d["ring_center_x"] = None
            d["ring_center_y"] = None

        d["ring_radius"] = getattr(result, "ring_radius", None)
        d["ring_dot_count"] = getattr(result, "ring_dot_count", 0)

        # Grayscale info
        gs_info = self._last_grayscale_info
        d["grayscale_mode"] = self._grayscale_mode.name.lower()
        d["grayscale_applied"] = (
            gs_info.conversion_applied if gs_info is not None else False
        )
        d["grayscale_was_input_gray"] = (
            gs_info.was_grayscale if gs_info is not None else False
        )
        d["grayscale_contrast_before"] = (
            gs_info.contrast_before if gs_info is not None else None
        )
        d["grayscale_contrast_after"] = (
            gs_info.contrast_after if gs_info is not None else None
        )

        # Alerts
        if result.alerts:
            d["alerts"] = result.alerts

        return d

    def apply_smoothed_dict(
        self,
        result: EyeDetectionResult,
        smoothed: Dict[str, Any],
    ) -> None:
        """Apply temporally-smoothed values back to the result object.

        The temporal smoother may adjust positions and radii for
        jitter reduction.  This writes those smoothed values back
        into the EyeDetectionResult so that downstream consumers
        (GUI overlay, export) use the smoothed coordinates.
        """
        if result.pupil.detected and result.pupil.ellipse is not None:
            if "pupil_x" in smoothed:
                result.pupil.ellipse.center_x = float(smoothed["pupil_x"])
            if "pupil_y" in smoothed:
                result.pupil.ellipse.center_y = float(smoothed["pupil_y"])
            if "pupil_radius" in smoothed:
                r = float(smoothed["pupil_radius"])
                result.pupil.ellipse.semi_major = r
                result.pupil.ellipse.semi_minor = r

        if result.limbus.detected and result.limbus.ellipse is not None:
            if "limbus_x" in smoothed:
                result.limbus.ellipse.center_x = float(smoothed["limbus_x"])
            if "limbus_y" in smoothed:
                result.limbus.ellipse.center_y = float(smoothed["limbus_y"])
            if "limbus_radius" in smoothed:
                r = float(smoothed["limbus_radius"])
                result.limbus.ellipse.semi_major = r
                result.limbus.ellipse.semi_minor = r

    # ================================================================
    # Smart structure extraction from segmentation mask
    # ================================================================

    def _extract_structure(
        self,
        mask: np.ndarray,
        gray_image: Optional[np.ndarray] = None,
        ring_result: Optional[RingDetectionResult] = None,
    ) -> Tuple[Optional[FitResult], Optional[FitResult]]:
        """Extract pupil and limbus geometry from a segmentation mask
        using the SmartContourFitter (auto circle-vs-ellipse).

        When a ring is detected, applies spatial constraints to
        ensure the pupil and limbus fits lie inside the ring opening.

        Parameters
        ----------
        mask : np.ndarray
            Integer label mask where 1=pupil, 2=iris, 3=ring (optional).
        gray_image : np.ndarray, optional
            Grayscale image for sub-pixel refinement.
        ring_result : RingDetectionResult, optional
            Ring detection result for spatial constraints.

        Returns
        -------
        (pupil_fit, limbus_fit) : tuple of optional FitResult
        """
        is_docked = ring_result is not None and ring_result.status in (
            RingStatus.PRESENT,
            RingStatus.PARTIAL,
        )

        # --- Pupil (class 1) ---
        pupil_mask = (mask == 1).astype(np.uint8)

        # Apply ring ROI constraint
        if is_docked and ring_result.ring_center is not None:
            pupil_mask = self._apply_ring_roi(
                pupil_mask,
                ring_result,
                margin_frac=0.85,
            )

        pupil_fit = self._fitter.fit(pupil_mask, gray_image)

        # Validate pupil is inside ring
        if (
            pupil_fit is not None
            and pupil_fit.valid
            and is_docked
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            if not self._is_inside_ring(
                pupil_fit.center_x,
                pupil_fit.center_y,
                pupil_fit.radius,
                ring_result,
            ):
                self.logger.debug("Pupil fit rejected: outside ring opening")
                pupil_fit = None

        # --- Iris / Limbus (class 2; union with pupil) ---
        iris_mask = ((mask == 2) | (mask == 1)).astype(np.uint8)

        if is_docked and ring_result.ring_center is not None:
            iris_mask = self._apply_ring_roi(
                iris_mask,
                ring_result,
                margin_frac=0.95,
            )
        else:
            # Pre-docked: ML model systematically overestimates limbus into sclera.
            # Erode the mask to shrink it uniformly by ~7 pixels so subpixel
            # refinement can catch the true inward edge.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            iris_mask = cv2.erode(iris_mask, kernel, iterations=1)

        limbus_fit = self._fitter.fit(
            iris_mask, 
            gray_image, 
            pupil_hint=pupil_fit
        )

        # Validate pre-docking limbus concentricity and radius ratio
        if (
            not is_docked
            and pupil_fit is not None
            and pupil_fit.valid
            and limbus_fit is not None
            and limbus_fit.valid
        ):
            dx = pupil_fit.center_x - limbus_fit.center_x
            dy = pupil_fit.center_y - limbus_fit.center_y
            dist = math.hypot(dx, dy)
            if limbus_fit.radius > 0:
                offset_ratio = dist / limbus_fit.radius
                if offset_ratio > ANATOMICAL_LIMITS.MAX_CENTER_OFFSET_RATIO:
                    self.logger.debug(
                        "Pre-docking limbus fit rejected: center offset "
                        f"{offset_ratio:.2f} > {ANATOMICAL_LIMITS.MAX_CENTER_OFFSET_RATIO}"
                    )
                    limbus_fit = None

            if limbus_fit is not None and limbus_fit.radius > 0:
                ratio = pupil_fit.radius / limbus_fit.radius
                if (
                    ratio < ANATOMICAL_LIMITS.MIN_PUPIL_LIMBUS_RATIO
                    or ratio > ANATOMICAL_LIMITS.MAX_PUPIL_LIMBUS_RATIO
                ):
                    self.logger.debug(
                        "Pre-docking limbus fit rejected: radius ratio "
                        f"{ratio:.2f} out of bounds"
                    )
                    limbus_fit = None

        # Validate limbus is inside ring
        if (
            limbus_fit is not None
            and limbus_fit.valid
            and is_docked
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            if not self._is_inside_ring(
                limbus_fit.center_x,
                limbus_fit.center_y,
                limbus_fit.radius,
                ring_result,
                allow_partial=True,
            ):
                self.logger.debug("Limbus fit rejected: extends outside ring")
                limbus_fit = None

        return pupil_fit, limbus_fit

    def _apply_ring_roi(
        self,
        binary_mask: np.ndarray,
        ring_result: RingDetectionResult,
        margin_frac: float = 0.85,
    ) -> np.ndarray:
        """Zero out pixels outside the ring opening."""
        if ring_result.ring_center is None or ring_result.ring_radius is None:
            return binary_mask

        h, w = binary_mask.shape[:2]
        cx = int(ring_result.ring_center[0])
        cy = int(ring_result.ring_center[1])
        r = int(ring_result.ring_radius * margin_frac)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(roi_mask, (cx, cy), max(1, r), 1, -1)

        return cv2.bitwise_and(binary_mask, roi_mask)

    def _is_inside_ring(
        self,
        cx: float,
        cy: float,
        radius: float,
        ring_result: RingDetectionResult,
        allow_partial: bool = False,
    ) -> bool:
        """Check if a circle (cx, cy, radius) is inside the ring opening."""
        if ring_result.ring_center is None or ring_result.ring_radius is None:
            return True  # No constraint

        ring_cx, ring_cy = ring_result.ring_center
        ring_r = ring_result.ring_radius

        dist = math.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)

        if allow_partial:
            # Allow limbus to extend slightly outside
            return dist + radius <= ring_r * 1.1
        else:
            # Pupil must be well inside
            return dist <= ring_r * 0.80

    def _apply_fit_to_result(
        self,
        result: EyeDetectionResult,
        pupil_fit: Optional[FitResult],
        limbus_fit: Optional[FitResult],
        force_limbus_overwrite: bool = False,
    ) -> None:
        """Overwrite pupil/limbus in *result* with SmartFitter output
        when the new fit is valid and at least as confident."""
        if pupil_fit is not None and pupil_fit.valid:
            ep = self._fit_result_to_ellipse_params(pupil_fit)
            new_conf = self._fit_result_confidence(pupil_fit)

            if (not result.pupil.detected) or new_conf >= result.pupil.confidence:
                result.pupil.detected = True
                result.pupil.ellipse = ep
                result.pupil.confidence = new_conf
                result.pupil.quality = assign_quality_grade(new_conf)
                result.pupil.method = DetectionMethod.ML
                result.pupil.fit_type = pupil_fit.fit_type.value
                if pupil_fit.contour_points is not None:
                    result.pupil.contour_points = pupil_fit.contour_points

        if limbus_fit is not None and limbus_fit.valid:
            ep = self._fit_result_to_ellipse_params(limbus_fit)
            new_conf = self._fit_result_confidence(limbus_fit)

            if (
                not result.limbus.detected
                or force_limbus_overwrite
                or new_conf >= result.limbus.confidence
            ):
                if not is_docked and limbus_fit is not None:
                    from pupil_tracking.calibration.spatial_calibration import correct_pre_docked_limbus_ellipse
                    ep = correct_pre_docked_limbus_ellipse(ep, lower_quadrant_pct=0.015)

                result.limbus.detected = True
                result.limbus.ellipse = ep
                result.limbus.confidence = new_conf
                result.limbus.quality = assign_quality_grade(new_conf)
                result.limbus.method = DetectionMethod.ML
                result.limbus.fit_type = limbus_fit.fit_type.value
                if limbus_fit.contour_points is not None:
                    result.limbus.contour_points = limbus_fit.contour_points

    # ================================================================
    # FitResult → EllipseParams conversion
    # ================================================================

    @staticmethod
    def _fit_result_to_ellipse_params(fit: FitResult) -> EllipseParams:
        """Convert a SmartContourFitter ``FitResult`` into an
        ``EllipseParams`` used throughout the rest of the pipeline."""
        if fit.fit_type == FitType.CIRCLE:
            return EllipseParams(
                center_x=fit.center_x,
                center_y=fit.center_y,
                semi_major=fit.radius,
                semi_minor=fit.radius,
                angle_deg=0.0,
            )
        return EllipseParams(
            center_x=fit.center_x,
            center_y=fit.center_y,
            semi_major=fit.semi_major,
            semi_minor=fit.semi_minor,
            angle_deg=fit.angle_deg,
        )

    @staticmethod
    def _fit_result_confidence(fit: FitResult) -> float:
        """Derive a [0, 1] confidence from a FitResult.

        Combines the fitter's own quality score with a small
        penalty if the fit chose ellipse (slightly less constrained
        than a circle, so marginally more room for over-fitting).
        """
        base = fit.fit_quality if fit.fit_quality is not None else 0.5

        if fit.fit_type == FitType.CIRCLE:
            base = min(1.0, base * 1.05)

        return float(np.clip(base, 0.0, 1.0))

    # ================================================================
    # mm value attachment
    # ================================================================

    def _add_mm_values(self, result: EyeDetectionResult) -> None:
        """Attach mm-converted measurements to pupil and limbus."""
        cal = self._calibration
        if not cal.calibrated:
            return

        if result.pupil.detected and result.pupil.ellipse is not None:
            e = result.pupil.ellipse
            result.pupil.radius_mm = e.radius * cal.mm_per_px
            result.pupil.center_mm = (
                e.center_x * cal.mm_per_px,
                e.center_y * cal.mm_per_px,
            )

        if result.limbus.detected and result.limbus.ellipse is not None:
            e = result.limbus.ellipse
            result.limbus.radius_mm = e.radius * cal.mm_per_px
            result.limbus.center_mm = (
                e.center_x * cal.mm_per_px,
                e.center_y * cal.mm_per_px,
            )
            from pupil_tracking.calibration.spatial_calibration import evaluate_clinical_wtw
            h_wtw, v_wtw, m_wtw, astig, is_m, status = evaluate_clinical_wtw(result.limbus, cal)
            result.limbus.wtw_horizontal_mm = h_wtw
            result.limbus.wtw_vertical_mm = v_wtw
            result.limbus.wtw_mean_mm = m_wtw
            result.limbus.wtw_astigmatism_mm = astig
            result.limbus.is_wtw_measured = is_m
            result.limbus.wtw_validity_status = status

    # ================================================================
    # Calibration
    # ================================================================

    def set_calibration(self, cal: CalibrationInfo) -> None:
        """Manually inject a calibration (e.g. from a known target)."""
        self._calibration = cal
        self.logger.info(
            "Calibration set: px_per_mm=%.2f source=%s",
            cal.px_per_mm,
            cal.source,
        )

    def set_calibration_mode(
        self,
        mode: str,
        manual_px_per_mm: Optional[float] = None,
        corneal_diameter_mm: Optional[float] = None,
        ring_diameter_mm: Optional[float] = None,
    ) -> None:
        """Set active calibration mode and parameters."""
        self._stabilized_cal.set_mode(
            mode=mode,
            manual_px_per_mm=manual_px_per_mm,
            corneal_diameter_mm=corneal_diameter_mm,
            ring_diameter_mm=ring_diameter_mm,
        )
        self._calibration = self._stabilized_cal._current_best()
        self.logger.info(
            "Detector calibration mode set to %s (px_per_mm=%s)",
            mode,
            self._calibration.px_per_mm if self._calibration.calibrated else "dynamic",
        )

    def calibrate_from_limbus(
        self,
        limbus: LimbusDetection,
        corneal_diameter_mm: float = 12.0,
    ) -> CalibrationInfo:
        """Calibrate using the known average horizontal corneal diameter (HVID).

        Uses the semi-major axis (horizontal limbus diameter) so that
        horizontal WTW is anchored to the assumed HVID reference.
        """
        if not limbus.detected or limbus.ellipse is None:
            return CalibrationInfo()

        diameter_px = limbus.ellipse.semi_major * 2.0
        if diameter_px < 10:
            return CalibrationInfo()

        px_per_mm = diameter_px / corneal_diameter_mm
        mm_per_px = corneal_diameter_mm / diameter_px

        aspect = limbus.ellipse.circularity
        conf = limbus.confidence * 0.8
        if aspect < 0.85:
            conf *= aspect

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=mm_per_px,
            source="limbus_diameter",
            method="anatomical",
            reference_diameter_mm=corneal_diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            corneal_diameter_assumed_mm=corneal_diameter_mm,
        )
        self._calibration = cal
        return cal

    def _calibrate_from_ring(
        self,
        ring_result: RingDetectionResult,
    ) -> CalibrationInfo:
        """Calibrate using the known suction ring diameter.

        This is a fallback when limbus is not detected but the ring
        provides a known physical reference.

        The suction ring diameter is read from the calibration config.
        """
        if ring_result.ring_radius is None:
            return CalibrationInfo()

        ring_diameter_mm = getattr(
            self.cfg.calibration,
            "suction_ring_diameter_mm",
            9.4,
        )

        diameter_px = ring_result.ring_radius * 2.0
        if diameter_px < 20:
            return CalibrationInfo()

        px_per_mm = diameter_px / ring_diameter_mm
        mm_per_px = ring_diameter_mm / diameter_px

        conf = ring_result.confidence * 0.6  # Lower confidence than limbus

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=mm_per_px,
            source="suction_ring_diameter",
            method="ring_reflection",
            reference_diameter_mm=ring_diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            corneal_diameter_assumed_mm=None,
        )
        self._calibration = cal
        self.logger.info(
            "Calibrated from ring: px_per_mm=%.2f (conf=%.3f)",
            px_per_mm,
            conf,
        )
        return cal

    # ================================================================
    # Cross-validation with rejection (ring-aware)
    # ================================================================

    def _cross_validate_and_reject(
        self,
        result: EyeDetectionResult,
        ring_result: Optional[RingDetectionResult] = None,
    ) -> EyeDetectionResult:
        """Cross-validate pupil and limbus.

        REJECT physically impossible combinations:
        * pupil centre outside the limbus
        * pupil radius larger than limbus radius
        * pupil or limbus outside ring opening

        Plan change: ratio warning threshold relaxed 0.80 → 0.85
        for dilated surgical pupils under anaesthesia.
        """
        pe = result.pupil.ellipse
        le = result.limbus.ellipse

        if pe is None or le is None:
            return result

        # --- Centre-offset check ---
        dx = pe.center_x - le.center_x
        dy = pe.center_y - le.center_y
        offset = math.sqrt(dx * dx + dy * dy)

        if le.radius > 0:
            offset_ratio = offset / le.radius

            if offset_ratio > 1.0:
                self.logger.warning(
                    "Pupil centre outside limbus (ratio=%.2f). "
                    "Rejecting less confident.",
                    offset_ratio,
                )
                if result.pupil.confidence > result.limbus.confidence:
                    result.limbus = LimbusDetection()
                    result.alerts.append("Limbus rejected: pupil centre outside limbus")
                else:
                    result.pupil = PupilDetection()
                    result.alerts.append("Pupil rejected: centre outside limbus")
                return result

            if offset_ratio > 0.5:
                result.alerts.append(
                    f"Large pupil-limbus offset: {offset_ratio:.2f} of limbus radius"
                )
                result.pupil.confidence *= 0.8
                result.limbus.confidence *= 0.8

        # --- Size-ratio check ---
        if le.radius > 0 and pe.radius > 0:
            ratio = pe.radius / le.radius

            if ratio > 1.0:
                self.logger.warning(
                    "Pupil larger than limbus (ratio=%.2f).",
                    ratio,
                )
                if result.pupil.confidence > result.limbus.confidence:
                    result.limbus = LimbusDetection()
                    result.alerts.append("Limbus rejected: smaller than pupil")
                else:
                    result.pupil = PupilDetection()
                    result.alerts.append("Pupil rejected: larger than limbus")
                return result

            # PLAN CHANGE: 0.80 → 0.85 for dilated surgical pupils
            if ratio > 0.85:
                result.alerts.append(f"Unusual pupil/limbus ratio: {ratio:.2f}")
                result.pupil.confidence *= 0.7
                result.limbus.confidence *= 0.7

        # --- Ring containment check ---
        if (
            ring_result is not None
            and ring_result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            ring_cx, ring_cy = ring_result.ring_center
            ring_r = ring_result.ring_radius

            # Check pupil is inside ring
            pupil_dist = math.sqrt(
                (pe.center_x - ring_cx) ** 2 + (pe.center_y - ring_cy) ** 2
            )
            if pupil_dist > ring_r * 0.85:
                result.alerts.append(
                    f"Pupil centre near/outside ring opening: "
                    f"dist={pupil_dist:.1f} vs ring_r={ring_r:.1f}"
                )
                result.pupil.confidence *= 0.7

            # Check limbus is inside ring
            limbus_extent = (
                math.sqrt((le.center_x - ring_cx) ** 2 + (le.center_y - ring_cy) ** 2)
                + le.radius
            )

            if limbus_extent > ring_r * 1.1:
                result.alerts.append(
                    f"Limbus extends outside ring: "
                    f"extent={limbus_extent:.1f} vs ring_r={ring_r:.1f}"
                )
                result.limbus.confidence *= 0.7

            # Check limbus is not the ring itself
            if le.radius > ring_r * 0.85:
                result.alerts.append(
                    f"Limbus radius ({le.radius:.1f}) close to ring "
                    f"radius ({ring_r:.1f}) — may be detecting ring as limbus"
                )
                result.limbus.confidence *= 0.5

        # --- Geometric cross-validation from postprocess module ---
        ring_seg = None
        if ring_result is not None and ring_result.status == RingStatus.PRESENT:
            ring_seg = RingSegmentationResult(
                detected=True,
                center=ring_result.ring_center,
                radius=ring_result.ring_radius,
            )

        valid, issues = validate_pupil_limbus_pair(pe, le, ring=ring_seg)
        for issue in issues:
            if issue not in result.alerts:
                result.alerts.append(f"Cross-validation: {issue}")

        return result

    # ================================================================
    # Classical CV fallback — pupil (ring-aware)
    # ================================================================

    def _classical_pupil(
        self,
        image: np.ndarray,
        ring_result: Optional[RingDetectionResult] = None,
    ) -> PupilDetection:
        """Classical pupil detection using adaptive thresholding
        and SmartContourFitter for final geometry.

        When a ring is detected, the search is constrained to the
        ring opening area.
        """
        detection = PupilDetection()
        detection.method = DetectionMethod.CLASSICAL

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_diag = math.sqrt(h * h + w * w)

        min_radius = max(8, int(img_diag * 0.015))
        max_radius = int(img_diag * 0.25)
        min_area = max(100, int(math.pi * min_radius * min_radius * 0.5))

        # Ring-aware constraints
        is_docked = ring_result is not None and ring_result.status in (
            RingStatus.PRESENT,
            RingStatus.PARTIAL,
        )
        ring_roi_mask = None

        if (
            is_docked
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            ring_roi_mask = np.zeros((h, w), dtype=np.uint8)
            cx = int(ring_result.ring_center[0])
            cy = int(ring_result.ring_center[1])
            r = int(ring_result.ring_radius * 0.80)
            cv2.circle(ring_roi_mask, (cx, cy), max(1, r), 255, -1)

            # Tighter max_radius for docked images
            max_radius = min(max_radius, int(ring_result.ring_radius * 0.5))

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        best_fit: Optional[FitResult] = None
        best_score = 0.0
        best_contour = None

        for pct in [3, 5, 8, 12, 18, 25, 35]:
            thresh_val = np.percentile(blurred, pct)
            _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Apply ring ROI mask
            if ring_roi_mask is not None:
                binary = cv2.bitwise_and(binary, ring_roi_mask)

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or len(cnt) < 15:
                    continue

                cnt_mask = np.zeros_like(gray)
                cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
                fit = self._fitter.fit(cnt_mask, gray)

                if fit is None or not fit.valid:
                    continue
                if fit.radius < min_radius or fit.radius > max_radius:
                    continue

                # Ring containment check
                if (
                    is_docked
                    and ring_result.ring_center is not None
                    and ring_result.ring_radius is not None
                ):
                    if not self._is_inside_ring(
                        fit.center_x,
                        fit.center_y,
                        fit.radius,
                        ring_result,
                    ):
                        continue

                # Centrality score
                if is_docked and ring_result.ring_center is not None:
                    ring_cx, ring_cy = ring_result.ring_center
                    ring_r = ring_result.ring_radius or 1.0
                    dist = math.sqrt(
                        (fit.center_x - ring_cx) ** 2 + (fit.center_y - ring_cy) ** 2
                    )
                    centrality = max(0.0, 1.0 - dist / ring_r)
                else:
                    centrality = max(
                        0.0,
                        1.0
                        - (
                            abs(fit.center_x - w / 2) / (w / 2) * 0.5
                            + abs(fit.center_y - h / 2) / (h / 2) * 0.5
                        ),
                    )

                circ = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0

                mask_tmp = np.zeros_like(gray)
                cv2.drawContours(mask_tmp, [cnt], -1, 255, -1)
                darkness = 1.0 - (cv2.mean(gray, mask=mask_tmp)[0] / 255.0)

                fit_quality = fit.fit_quality if fit.fit_quality is not None else 0.5

                score = (
                    0.25 * centrality
                    + 0.25 * min(1.0, circ / 0.7)
                    + 0.25 * fit_quality
                    + 0.25 * darkness
                )

                if score > best_score:
                    best_score = score
                    best_fit = fit
                    best_contour = cnt

        if best_fit is not None and best_score > 0.20:
            detection.detected = True
            detection.ellipse = self._fit_result_to_ellipse_params(best_fit)
            detection.confidence = float(np.clip(best_score * 0.85, 0.0, 1.0))
            detection.quality = assign_quality_grade(detection.confidence)
            detection.contour_points = best_contour
            detection.fit_type = best_fit.fit_type.value

        return detection

    # ================================================================
    # Classical CV fallback — limbus (ring-aware)
    # ================================================================

    def _classical_limbus(
        self,
        image: np.ndarray,
        pupil_hint: Optional[EllipseParams] = None,
        ring_result: Optional[RingDetectionResult] = None,
    ) -> LimbusDetection:
        """Classical limbus detection using gradient edges + Hough,
        refined by SmartContourFitter.

        When a ring is detected, the search radius is constrained
        so the limbus cannot extend outside the ring opening.
        """
        detection = LimbusDetection()
        detection.method = DetectionMethod.CLASSICAL

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_diag = math.sqrt(h * h + w * w)

        min_radius = max(20, int(img_diag * 0.06))
        max_radius = int(img_diag * 0.45)

        if pupil_hint is not None and pupil_hint.is_valid:
            expected_min = pupil_hint.radius * 1.8
            expected_max = pupil_hint.radius * 5.0
            min_radius = max(min_radius, int(expected_min * 0.8))
            max_radius = min(max_radius, int(expected_max * 1.2))

        # Ring constraint: limbus must fit inside ring
        is_docked = ring_result is not None and ring_result.status in (
            RingStatus.PRESENT,
            RingStatus.PARTIAL,
        )
        if is_docked and ring_result.ring_radius is not None:
            max_radius = min(max_radius, int(ring_result.ring_radius * 0.90))

        if min_radius >= max_radius:
            max_radius = min_radius + 50

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Apply ring ROI to edges
        if (
            is_docked
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            roi = np.zeros_like(edges)
            cx = int(ring_result.ring_center[0])
            cy = int(ring_result.ring_center[1])
            r = int(ring_result.ring_radius * 0.95)
            cv2.circle(roi, (cx, cy), max(1, r), 255, -1)
            edges = cv2.bitwise_and(edges, roi)

        all_circles: list[list[float]] = []
        for dp, p1, p2 in [(1.5, 80, 40), (2.0, 60, 30)]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=max(50, max(h, w) // 4),
                param1=p1,
                param2=p2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is not None:
                all_circles.extend(circles[0].tolist())

        if not all_circles:
            return detection

        best_fit: Optional[FitResult] = None
        best_score = 0.0

        for cx, cy, r in all_circles:
            if r < min_radius or r > max_radius:
                continue

            if pupil_hint is not None and pupil_hint.is_valid:
                d = math.sqrt(
                    (cx - pupil_hint.center_x) ** 2 + (cy - pupil_hint.center_y) ** 2
                )
                if d > r * 0.35:
                    continue

            # Ring containment check
            if (
                is_docked
                and ring_result.ring_center is not None
                and ring_result.ring_radius is not None
            ):
                ring_cx, ring_cy = ring_result.ring_center
                ring_r = ring_result.ring_radius
                dist_to_ring = math.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)
                if dist_to_ring + r > ring_r * 1.05:
                    continue

            edge_pts: list[list[int]] = []
            for angle in np.linspace(0, 2 * np.pi, 360):
                for dr in range(-12, 13):
                    px = int(cx + (r + dr) * math.cos(angle))
                    py = int(cy + (r + dr) * math.sin(angle))
                    if 0 <= px < w and 0 <= py < h and edges[py, px] > 0:
                        edge_pts.append([px, py])
                        break

            if len(edge_pts) < 20:
                continue

            edge_mask = np.zeros_like(gray)
            pts_arr = np.array(edge_pts, dtype=np.int32)
            if len(pts_arr) >= 5:
                hull = cv2.convexHull(pts_arr)
                cv2.fillConvexPoly(edge_mask, hull, 1)

            fit = self._fitter.fit(edge_mask, gray)
            if fit is None or not fit.valid:
                continue
            if fit.radius < min_radius or fit.radius > max_radius:
                continue

            circ = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0
            centrality = max(
                0.0,
                1.0
                - (
                    abs(fit.center_x - w / 2) / (w / 2) * 0.5
                    + abs(fit.center_y - h / 2) / (h / 2) * 0.5
                ),
            )
            coverage = min(1.0, len(edge_pts) / 180.0)
            fit_quality = fit.fit_quality if fit.fit_quality is not None else 0.5

            score = (
                0.25 * min(1.0, circ / 0.7)
                + 0.25 * fit_quality
                + 0.25 * centrality
                + 0.25 * coverage
            )

            if pupil_hint is not None and pupil_hint.is_valid:
                d = math.sqrt(
                    (fit.center_x - pupil_hint.center_x) ** 2
                    + (fit.center_y - pupil_hint.center_y) ** 2
                )
                concentricity = max(0.0, 1.0 - d / max(r, 1))
                score = score * 0.7 + concentricity * 0.3

            # Bonus for concentricity with ring centre
            if is_docked and ring_result.ring_center is not None:
                ring_cx, ring_cy = ring_result.ring_center
                d_ring = math.sqrt(
                    (fit.center_x - ring_cx) ** 2 + (fit.center_y - ring_cy) ** 2
                )
                ring_concentricity = max(
                    0.0, 1.0 - d_ring / max(ring_result.ring_radius or 1, 1)
                )
                score = score * 0.85 + ring_concentricity * 0.15

            if score > best_score:
                best_score = score
                best_fit = fit

        if best_fit is not None and best_score > 0.20:
            detection.detected = True
            detection.ellipse = self._fit_result_to_ellipse_params(best_fit)
            detection.confidence = float(np.clip(best_score * 0.80, 0.0, 1.0))
            detection.quality = assign_quality_grade(detection.confidence)
            detection.fit_type = best_fit.fit_type.value

        return detection

    # ================================================================
    # State management
    # ================================================================

    def reset(self) -> None:
        """Clear calibration, calculator, ring state, and grayscale cache."""
        self._calibration = CalibrationInfo()
        self.corneal_calc.reset()
        self._last_ring_result = None
        self._ring_stable_count = 0
        self._last_grayscale_info = None
        self.logger.info("Detector state reset")

    @property
    def ring_detector(self) -> RingDetector:
        """Access the ring detector for direct queries."""
        return self._ring_detector

    @property
    def is_ring_detected(self) -> bool:
        """Whether the last processed image had a ring detected."""
        if self._last_ring_result is None:
            return False
        return self._last_ring_result.status in (
            RingStatus.PRESENT,
            RingStatus.PARTIAL,
        )

    @property
    def last_ring_result(self) -> Optional[RingDetectionResult]:
        """The most recent ring detection result."""
        return self._last_ring_result


# ====================================================================
# Engine wrapper classes (outside UnifiedDetector)
# ====================================================================

class _ONNXEngineWrapper:
    """
    Wraps ONNXInference to match the SegmentationInference interface
    that the rest of UnifiedDetector expects.

    SegmentationInference has:
      - .detect(image, frame_number, source) -> EyeDetectionResult
      - .available (bool)
      - .set_red_light_filter_enabled(bool)
      - .set_red_light_temporal_mode(bool)
      - .reset_red_light_temporal()
      - .model_path (str)

    ONNXInference has:
      - .infer(image, target_size) -> dict of masks
      - .is_loaded (bool)
    """

    def __init__(self, onnx_engine, config=None):
        self._engine = onnx_engine
        self._config = config
        self.available = onnx_engine.is_loaded
        self.model_path = None  # No .pth path for ONNX

        # Core preprocessors to clean image of reflections/red lights
        from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
        from pupil_tracking.preprocessing.suction_ring_masker import SuctionRingMasker

        self._reflection_remover = ReflectionRemover(
            brightness_threshold=220,
            min_reflection_area=15,
            inpaint_radius=5,
            detect_red_highlights=True,
            red_threshold_offset=25,
        )
        self._ring_masker = SuctionRingMasker()
        self._red_light_filter = None  # Lazy initialization
        self._red_light_enabled = True
        self._red_light_temporal_mode = True

    def _get_red_light_filter(self):
        """Lazy import and return red light filter."""
        from pupil_tracking.preprocessing.red_light_filter import RedLightFilter

        return RedLightFilter(
            red_threshold=200,
            dominance_offset=30,
            min_area=5,
            enable_inpaint=True,
            inpaint_radius=3,
            enable_temporal=self._red_light_temporal_mode,
        )

    def detect(
        self,
        image: np.ndarray,
        frame_number: int = -1,
        source: str = "",
        **kwargs,
    ) -> "EyeDetectionResult":
        """
        Run ONNX inference and return an EyeDetectionResult
        with _raw_mask attached for SmartContourFitter.
        """
        # Apply identical preprocessing before ONNX inference
        clean_bgr = image
        roi_mask = None
        if self._ring_masker is not None:
            try:
                clean_bgr, marker_mask, ring_result = self._ring_masker.remove_with_diagnostics(clean_bgr)
                if getattr(ring_result, "ring_centre", None) is not None:
                    cx, cy = int(round(ring_result.ring_centre[0])), int(round(ring_result.ring_centre[1]))
                    inner_r = int(round(ring_result.ring_inner_radius)) if getattr(ring_result, "ring_inner_radius", None) is not None else None
                    if inner_r is not None and inner_r > 4:
                        h, w = clean_bgr.shape[:2]
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.circle(roi_mask, (cx, cy), inner_r, 255, -1)
            except Exception:
                clean_bgr, _ = self._ring_masker.remove(clean_bgr)

        if self._reflection_remover is not None:
            clean_bgr, _ = self._reflection_remover.remove(clean_bgr, roi_mask=roi_mask)

        if self._red_light_enabled:
            if self._red_light_filter is None:
                self._red_light_filter = self._get_red_light_filter()
            if self._red_light_filter is not None:
                clean_bgr, _ = self._red_light_filter.apply(
                    clean_bgr, frame_number=frame_number
                )

        masks = self._engine.infer(clean_bgr)

        # Build an EyeDetectionResult with raw mask for downstream fitting
        result = EyeDetectionResult()
        result.metadata = FrameMetadata()
        result.metadata.frame_number = frame_number
        result.metadata.source = source

        # Build integer label mask: 0=bg, 1=pupil, 2=iris
        h, w = image.shape[:2]
        raw_mask = np.zeros((h, w), dtype=np.uint8)

        iris_mask = masks.get("iris", np.zeros((h, w), dtype=np.uint8))
        pupil_mask = masks.get("pupil", np.zeros((h, w), dtype=np.uint8))
        ring_mask = masks.get("ring", None)

        raw_mask[iris_mask > 127] = 2
        raw_mask[pupil_mask > 127] = 1
        if ring_mask is not None:
            raw_mask[ring_mask > 127] = 3

        result._raw_mask = raw_mask

        # Set initial confidence from mask quality
        pupil_pixels = (pupil_mask > 127).sum()
        iris_pixels = (iris_mask > 127).sum()

        if pupil_pixels > 100:
            result.pupil.detected = True
            result.pupil.confidence = 0.5  # Will be refined by SmartFitter
        if iris_pixels > 100:
            result.limbus.detected = True
            result.limbus.confidence = 0.5

        return result

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable red light filtering."""
        self._red_light_enabled = enabled
        if enabled:
            self._red_light_filter = self._get_red_light_filter()
        else:
            from pupil_tracking.preprocessing.red_light_filter import RedLightFilter
            self._red_light_filter = RedLightFilter(
                red_threshold=255,
                dominance_offset=1000,
                min_area=100000,
                enable_inpaint=False,
                enable_temporal=False,
            )

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        """Enable temporal mode for red light filtering (for video)."""
        self._red_light_temporal_mode = enabled
        if self._red_light_filter is None:
            self._red_light_filter = self._get_red_light_filter()
        if self._red_light_filter is not None:
            self._red_light_filter.enable_temporal = enabled
            if not enabled:
                self._red_light_filter.reset_temporal()

    def reset_red_light_temporal(self) -> None:
        """Reset temporal tracking state for red light filter."""
        if self._red_light_filter is not None:
            self._red_light_filter.reset_temporal()

    def get_device_info(self) -> dict:
        return self._engine.get_device_info()

    def __getattr__(self, name):
        """Forward unknown attributes to the underlying engine."""
        return getattr(self._engine, name)


class _DummyEngine:
    """
    Fallback engine when no ML backend is available.
    Returns empty results so classical fallback can still work.
    """

    available = False
    model_path = None

    def detect(self, image: np.ndarray, **kwargs) -> "EyeDetectionResult":
        result = EyeDetectionResult()
        result.metadata = FrameMetadata()
        h, w = image.shape[:2]
        result._raw_mask = np.zeros((h, w), dtype=np.uint8)
        return result

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        pass

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        pass

    def reset_red_light_temporal(self) -> None:
        pass

    def get_device_info(self) -> dict:
        return {"backend": "none", "provider": "none"}
