"""
Master registration engine for cyclotorsion detection.

Orchestrates the full pipeline:
    1. Validate inputs (two images + two detections)
    2. Shared high-speed polar unwrapping & reference frame caching
    3. Multi-stream execution (deterministic, phase, optical flow, sparse keypoints, landmarks)
    4. Fast-path cascade for sub-15ms real-time latency
    5. Multi-stream consensus fusion
    6. Temporal Kalman filtering for surgical video stabilization
    7. Return unified RegistrationResult
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.fusion import FusionEngine
from pupil_tracking.registration.polar import PolarImage, PolarUnwrapper
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.registration.streams.phase_correlation import PhaseCorrelationStream
from pupil_tracking.registration.streams.iris_code import IrisCodeStream
from pupil_tracking.registration.streams.polar_optical_flow import PolarOpticalFlowStream
from pupil_tracking.registration.streams.angular_profile import AngularProfileStream
from pupil_tracking.registration.streams.deep_matcher import DeepMatcherStream
from pupil_tracking.registration.streams.ink_tracker import InkTrackerStream
from pupil_tracking.registration.streams.vessel_tracker import VesselTrackerStream
from pupil_tracking.registration.streams.custom_feature import CustomFeatureStream
from pupil_tracking.registration.temporal_filter import CyclotorsionKalmanFilter
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    RegistrationQuality,
    RegistrationResult,
    StreamResult,
)

logger = logging.getLogger(__name__)


class RegistrationEngine:
    """Master engine for iris registration and cyclotorsion detection.

    Manages all detection streams and the fusion engine.  Provides
    the main ``register()`` method that accepts two eye images with
    their detections and returns a fused cyclotorsion result.

    Optimizations:
        - Reference image polar caching (zero repeat cost for fixed pre-op images).
        - Shared polar unwrapping & CLAHE enhancement across all polar streams.
        - Fast-path cascade: returns in <15ms if fast streams have high agreement.
        - Temporal Kalman filter for jitter suppression and coasting in video.
    """

    def __init__(self):
        cfg = get_config().registration
        self.enabled = getattr(cfg, "enabled", True)
        self.fast_cascade = getattr(cfg, "fast_cascade", False)
        self.enable_temporal_filter = getattr(cfg, "enable_temporal_filter", False)

        self.streams: Dict[str, BaseStream] = {}
        self.fusion = FusionEngine()

        # Shared precomputations
        self._unwrapper = PolarUnwrapper()
        self._enhancer = IrisEnhancer()
        self._ref_cache: Dict[str, Any] = {}

        # Temporal filter for video tracking
        self.temporal_filter = CyclotorsionKalmanFilter()

        # Initialise enabled streams
        if getattr(cfg, "enable_phase_correlation", True):
            self.streams["phase_correlation"] = PhaseCorrelationStream()

        if getattr(cfg, "enable_iris_code", True):
            self.streams["iris_code"] = IrisCodeStream()

        if getattr(cfg, "enable_angular_profile", True):
            self.streams["angular_profile"] = AngularProfileStream()

        if getattr(cfg, "enable_polar_optical_flow", True):
            self.streams["polar_optical_flow"] = PolarOpticalFlowStream()

        if getattr(cfg, "enable_deep_matcher", True):
            self.streams["deep_matcher"] = DeepMatcherStream()

        if getattr(cfg, "enable_ink_tracker", True):
            self.streams["ink_markers"] = InkTrackerStream()

        if getattr(cfg, "enable_vessel_tracker", True):
            self.streams["limbal_vessels"] = VesselTrackerStream()

        if getattr(cfg, "enable_custom_feature", False):
            stream = CustomFeatureStream()
            if stream.available:
                self.streams["custom_feature"] = stream
            else:
                logger.info("Custom feature stream skipped — no model available")

        logger.info(
            "RegistrationEngine initialised with %d streams (enabled=%s, fast_cascade=%s): %s",
            len(self.streams),
            self.enabled,
            self.fast_cascade,
            list(self.streams.keys()),
        )

    def _get_cache_key(self, img: np.ndarray, det: EyeDetectionResult) -> Tuple[int, float, float, float, float, float, float]:
        """Compute compact hash key for reference eye state."""
        p = det.pupil.ellipse
        l = det.limbus.ellipse
        return (
            id(img),
            round(p.center_x, 1) if p else 0.0,
            round(p.center_y, 1) if p else 0.0,
            round(p.radius, 1) if p else 0.0,
            round(l.center_x, 1) if l else 0.0,
            round(l.center_y, 1) if l else 0.0,
            round(l.radius, 1) if l else 0.0,
        )

    def _get_or_compute_reference_polar(
        self, img_ref: np.ndarray, det_ref: EyeDetectionResult
    ) -> Tuple[PolarImage, np.ndarray]:
        """Fetch cached polar-unwrapped reference image or compute it once."""
        key = self._get_cache_key(img_ref, det_ref)
        if self._ref_cache.get("key") == key:
            return self._ref_cache["polar"], self._ref_cache["enh"]

        polar_ref = self._unwrapper.unwrap_from_detection(img_ref, det_ref)
        if polar_ref.valid:
            enh_ref = self._enhancer.enhance_polar(polar_ref.image, polar_ref.mask)
        else:
            enh_ref = polar_ref.image

        self._ref_cache = {"key": key, "polar": polar_ref, "enh": enh_ref}
        return polar_ref, enh_ref

    def register(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
        fast_cascade: Optional[bool] = None,
        timestamp: Optional[float] = None,
    ) -> RegistrationResult:
        """Run all streams and fuse results.

        Parameters
        ----------
        img_ref : np.ndarray
            Reference pre-operative image.
        img_curr : np.ndarray
            Current intra-operative image.
        detection_ref, detection_curr : EyeDetectionResult
            Detections containing pupil and limbus.
        fast_cascade : bool, optional
            Override cascade mode: if True, returns immediately upon
            strong Tier-1 consensus (<15 ms).
        timestamp : float, optional
            Sequential timestamp in seconds for temporal Kalman filtering.
        """
        start = time.perf_counter()

        # Check if master registration is disabled (centration-only mode)
        if not self.enabled:
            logger.debug("Registration component disabled — running centration only")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
                quality=RegistrationQuality.NO_RESULT,
                metadata={"disabled": True, "mode": "centration_only"},
            )

        # Validate inputs
        if not detection_ref.has_both:
            logger.warning("Reference image lacks pupil+limbus detection")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
            )

        if not detection_curr.has_both:
            logger.warning("Current image lacks pupil+limbus detection")
            return RegistrationResult(
                total_processing_time_ms=(time.perf_counter() - start) * 1000.0,
            )

        use_cascade = self.fast_cascade if fast_cascade is None else fast_cascade

        # Shared precomputations: unwrap reference (cached) and current once
        polar_ref, enh_ref = self._get_or_compute_reference_polar(img_ref, detection_ref)

        polar_curr = self._unwrapper.unwrap_from_detection(img_curr, detection_curr)
        if polar_curr.valid:
            enh_curr = self._enhancer.enhance_polar(polar_curr.image, polar_curr.mask)
        else:
            enh_curr = polar_curr.image

        kwargs_polar = {
            "polar_ref": polar_ref,
            "polar_curr": polar_curr,
            "enh_ref": enh_ref,
            "enh_curr": enh_curr,
        }

        stream_results: Dict[str, StreamResult] = {}

        # Tier 1: Ultra-fast polar streams (2-8 ms each)
        tier1_names = ["phase_correlation", "iris_code", "angular_profile"]
        for name in tier1_names:
            stream = self.streams.get(name)
            if stream and stream.enabled:
                sr = stream.run(
                    img_ref, img_curr, detection_ref, detection_curr, **kwargs_polar
                )
                stream_results[name] = sr

        # Check Fast-Path Cascade early exit
        if use_cascade and len(stream_results) >= 2:
            valid_t1 = [sr for sr in stream_results.values() if sr.valid and sr.confidence >= 0.70]
            if len(valid_t1) >= 2:
                angles = [sr.torsion_deg for sr in valid_t1]
                spread = max(angles) - min(angles)
                if spread < 0.40:  # High inter-stream consensus
                    result = self.fusion.fuse(stream_results)
                    result.total_processing_time_ms = (time.perf_counter() - start) * 1000.0
                    result.metadata["cascade_fast_path"] = True
                    result.metadata["spread_deg"] = spread

                    if self.enable_temporal_filter:
                        result = self._apply_kalman(result, timestamp)

                    logger.debug(
                        "Fast cascade early exit: torsion=%.3f° conf=%.3f in %.1fms",
                        result.torsion_deg,
                        result.confidence,
                        result.total_processing_time_ms,
                    )
                    return result

        # Tier 2: Heavier / Scleral / Feature Matching streams
        tier2_names = [
            "polar_optical_flow",
            "deep_matcher",
            "ink_markers",
            "limbal_vessels",
            "custom_feature",
        ]
        for name in tier2_names:
            stream = self.streams.get(name)
            if stream and stream.enabled:
                if name == "polar_optical_flow":
                    sr = stream.run(
                        img_ref, img_curr, detection_ref, detection_curr, **kwargs_polar
                    )
                else:
                    sr = stream.run(img_ref, img_curr, detection_ref, detection_curr)
                stream_results[name] = sr

        # Fuse all streams
        result = self.fusion.fuse(stream_results)
        result.total_processing_time_ms = (time.perf_counter() - start) * 1000.0

        # Temporal filter update
        if self.enable_temporal_filter:
            result = self._apply_kalman(result, timestamp)

        logger.info(
            "Registration: torsion=%.3f° conf=%.3f quality=%s streams=%d/%d time=%.1fms",
            result.torsion_deg,
            result.confidence,
            result.quality.value,
            result.agreeing_streams,
            result.active_streams,
            result.total_processing_time_ms,
        )

        return result

    def _apply_kalman(
        self, result: RegistrationResult, timestamp: Optional[float]
    ) -> RegistrationResult:
        """Apply 1D constant-velocity Kalman filter for temporal smoothing."""
        raw_torsion = result.torsion_deg
        filt_angle, filt_vel, is_inlier = self.temporal_filter.update(
            raw_torsion if result.confidence > 0.3 else None,
            result.confidence,
            timestamp=timestamp,
        )
        result.torsion_deg = filt_angle
        result.metadata["raw_torsion_deg"] = raw_torsion
        result.metadata["torsion_velocity_deg_s"] = filt_vel
        result.metadata["kalman_inlier"] = is_inlier
        return result

    def reset_temporal_filter(self) -> None:
        """Reset temporal tracker (e.g. on new patient / sequence)."""
        self.temporal_filter.reset()
        self._ref_cache.clear()

    def clear_cache(self) -> None:
        """Clear cached reference polar unwrapping."""
        self._ref_cache.clear()

    def set_fast_cascade(self, enabled: bool) -> None:
        """Enable or disable fast-path cascade mode."""
        self.fast_cascade = bool(enabled)
        logger.info("RegistrationEngine fast_cascade set to: %s", self.fast_cascade)

    def set_temporal_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable Kalman temporal smoothing."""
        self.enable_temporal_filter = bool(enabled)
        logger.info("RegistrationEngine temporal_filter set to: %s", self.enable_temporal_filter)

    def get_stream_names(self) -> List[str]:
        """Return names of all initialised streams."""
        return list(self.streams.keys())

    def enable_stream(self, name: str) -> bool:
        """Enable a stream by name. Returns True if found."""
        if name in self.streams:
            self.streams[name].enabled = True
            return True
        return False

    def disable_stream(self, name: str) -> bool:
        """Disable a stream by name. Returns True if found."""
        if name in self.streams:
            self.streams[name].enabled = False
            return True
        return False

    def set_master_enabled(self, enabled: bool) -> None:
        """Enable or disable the entire registration component.

        When disabled, the system runs purely in centration mode without
        computing cyclotorsion or running registration streams.
        """
        self.enabled = bool(enabled)
        logger.info("Registration master enabled set to: %s", self.enabled)

    def set_ink_tracker_enabled(self, enabled: bool) -> None:
        """Enable or disable Stream C (purple limbal ink tracker)."""
        if enabled:
            self.enable_stream("ink_markers")
        else:
            self.disable_stream("ink_markers")

    def set_iris_features_enabled(self, enabled: bool) -> None:
        """Enable or disable iris landmark / feature extraction streams."""
        for name in ("custom_feature", "deep_matcher"):
            if enabled:
                self.enable_stream(name)
            else:
                self.disable_stream(name)

    def set_phase_correlation_enabled(self, enabled: bool) -> None:
        """Enable or disable polar FFT phase correlation stream."""
        if enabled:
            self.enable_stream("phase_correlation")
        else:
            self.disable_stream("phase_correlation")

    def register_pentacam(
        self,
        img_pentacam: np.ndarray,
        img_elita: np.ndarray,
        pentacam_result: Optional[Any] = None,
        elita_detection: Optional[EyeDetectionResult] = None,
        laterality: str = "OD",
        mode: str = "static",
    ):
        """Cross-modality registration between seated Pentacam IR and supine ELITA image."""
        if not self.enabled:
            logger.info("Registration disabled — skipping cross-modality registration (centration only mode)")
            from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult, RegistrationFailureKind
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.UNKNOWN_ERROR,
                failure_reason="Registration component is turned OFF in Settings (Centration Only Mode).",
                clinical_impact="DISABLED",
                quality_assessment="Bypassed (Centration Only Mode)",
                torsion_direction="NEUTRAL",
                laterality=laterality,
            )

        from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
        engine = CrossModalityRegistrationEngine()
        return engine.register(
            pentacam_image=img_pentacam,
            elita_image=img_elita,
            pentacam_result=pentacam_result,
            elita_detection=elita_detection,
            laterality=laterality,
            mode=mode,
        )
