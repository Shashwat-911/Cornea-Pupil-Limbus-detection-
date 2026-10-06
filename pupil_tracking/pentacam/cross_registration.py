"""Experimental cross-modality iris registration for Pentacam and ELITA images.

The implementation estimates angular shift from masked polar iris strips.
Synthetic rigid-rotation tests do not establish clinical
accuracy; pupil dilation, nonrigid deformation, reflections, and live-frame
latency require separate validation on representative clinical data.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import asdict
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.pentacam.cross_system import (
    CrossSystemRegistrationResult,
    RegistrationFailureKind,
    TransformationModel,
)
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.sitting import masked_angular_match, registration_mask, valid_geometry
from pupil_tracking.pentacam.types import PentacamDetectionResult
from pupil_tracking.registration.polar import PolarImage, PolarUnwrapper
from pupil_tracking.utils.types import EyeDetectionResult

logger = logging.getLogger(__name__)


class CrossModalityRegistrationEngine:
    """Master engine for Pentacam ↔ ELITA cross-modality cyclotorsion estimation."""

    def __init__(
        self,
        num_angles: int = 360,
        num_radial: int = 64,
        max_rotation_search_deg: float = 30.0,
    ) -> None:
        if num_angles < 180 or num_radial < 8 or not 0 < max_rotation_search_deg < 180:
            raise ValueError("Require >=180 angular samples, >=8 radial samples, and 0<search<180 degrees")
        self.num_angles = num_angles
        self.num_radial = num_radial
        self.max_rotation_search_deg = max_rotation_search_deg

        self.unwrapper = PolarUnwrapper(num_angles=num_angles, num_radial=num_radial, interpolation=cv2.INTER_LINEAR)
        self.pentacam_detector = PentacamIrisDetector()
        self._cached_reference_key: Optional[Tuple[Tuple[int, ...], str, bytes]] = None
        self._cached_reference_result: Optional[PentacamDetectionResult] = None
        self._polar_cache_key = None
        self._polar_cache = None
        self._angular_reference_cache = {}
        self.last_angular_match = None

        # Dynamic mode temporal filter state
        self._last_smooth_theta: Optional[float] = None

    def register(
        self,
        pentacam_image: np.ndarray,
        elita_image: np.ndarray,
        pentacam_result: Optional[PentacamDetectionResult] = None,
        elita_detection: Optional[EyeDetectionResult] = None,
        laterality: str = "OD",
        mode: str = "static",
    ) -> CrossSystemRegistrationResult:
        """Register seated Pentacam IR image with supine ELITA RGB/Grayscale image.

        Parameters
        ----------
        pentacam_image : np.ndarray
            Seated reference image (Pentacam IR or grayscale).
        elita_image : np.ndarray
            Supine surgery/diagnostic image (ELITA RGB or grayscale).
        pentacam_result : Optional[PentacamDetectionResult]
            Pre-computed Pentacam detection. If None, run PentacamIrisDetector.
        elita_detection : Optional[EyeDetectionResult]
            Pre-computed ELITA detection (from frozen UnifiedDetector).
        laterality : str
            Eye side: "OD" (Right) or "OS" (Left). Default is "OD".
        mode : str
            "static" (single high-quality snapshot) or "dynamic" (temporal stream).

        Returns
        -------
        CrossSystemRegistrationResult
            Validated cyclotorsion angle, clinical impact, confidence, and diagnostics.
        """
        t0 = time.perf_counter()
        laterality = laterality.upper()
        self.last_angular_match = None
        if laterality not in ("OD", "OS") or mode not in ("static", "dynamic"):
            raise ValueError("laterality must be OD/OS and mode static/dynamic")

        # 1. Validate inputs
        if pentacam_image is None or elita_image is None or pentacam_image.size == 0 or elita_image.size == 0:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.NO_PENTACAM if pentacam_image is None or pentacam_image.size == 0 else RegistrationFailureKind.NO_ELITA,
                failure_reason="Missing image input",
                laterality=laterality,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
        for image in (pentacam_image, elita_image):
            if image.dtype != np.uint8 or image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
                raise ValueError("Expected uint8 grayscale or BGR images")

        # Content key also catches in-place buffer updates and object-id reuse.
        contiguous_image = np.ascontiguousarray(pentacam_image)
        reference_key = (
            tuple(pentacam_image.shape), pentacam_image.dtype.str,
            hashlib.blake2b(contiguous_image.view(np.uint8), digest_size=16).digest(),
        )
        if pentacam_result is None:
            if reference_key == self._cached_reference_key:
                pentacam_result = self._cached_reference_result
            else:
                pentacam_result = self.pentacam_detector.detect(pentacam_image)
                if pentacam_result.valid:
                    self._cached_reference_key = reference_key
                    self._cached_reference_result = pentacam_result

        if (not pentacam_result.valid or not pentacam_result.geometry.pupil_detected
                or not pentacam_result.geometry.limbus_detected
                or not valid_geometry(pentacam_result.geometry.pupil, pentacam_result.geometry.limbus, pentacam_image.shape)):
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.INSUFFICIENT_PENTACAM_FEATURES,
                failure_reason=f"Pentacam detection failed: {pentacam_result.failure_reason}",
                laterality=laterality,
                pentacam_features_used=0,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. Validate ELITA detection
        if (elita_detection is None or not elita_detection.has_both
                or not valid_geometry(elita_detection.pupil.ellipse, elita_detection.limbus.ellipse, elita_image.shape)):
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.NO_ELITA,
                failure_reason="ELITA image lacks validated pupil and limbus geometry",
                laterality=laterality,
                pentacam_features_used=len(pentacam_result.feature_set.features),
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 4. Polar unwrap both irises (Daugman rubber-sheet model)
        geometry_key = tuple((e.center_x, e.center_y, e.semi_major, e.semi_minor, e.angle_deg)
                             for e in (pentacam_result.geometry.pupil, pentacam_result.geometry.limbus))
        polar_key = (reference_key, geometry_key, laterality, self.unwrapper.num_angles,
                     self.unwrapper.num_radial, self.unwrapper.inner_margin,
                     self.unwrapper.outer_margin, self.unwrapper.interpolation)
        if self._polar_cache_key != polar_key:
            self._polar_cache = self._unwrap_pentacam(pentacam_image, pentacam_result)
            self._polar_cache_key = polar_key
            self._last_smooth_theta = None
            self._angular_reference_cache.clear()
        polar_pentacam = self._polar_cache
        polar_elita = self.unwrapper.unwrap_from_detection(
            elita_image, elita_detection, validity_mask=registration_mask(elita_image))

        if not polar_pentacam.valid or not polar_elita.valid:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.COORDINATE_MISMATCH,
                failure_reason="Polar unwrapping failed for one or both modalities",
                laterality=laterality,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 5. Matcher performs shift-equivariant illumination normalization.
        enh_pentacam = polar_pentacam.image
        enh_elita = polar_elita.image

        # 6. Shift-dependent masked normalized correlation across radial bands.
        poc_theta, poc_conf, poc_psr = self._phase_correlate_cross(
            enh_pentacam, enh_elita, polar_pentacam.mask, polar_elita.mask
        )

        fused_theta = poc_theta
        # A texture-quality score, not a probability. The old landmark histogram
        # shortcut was not independent identity evidence and inflated confidence.
        fused_conf = poc_conf

        # Dynamic mode temporal filtering
        if mode == "dynamic" and poc_conf > 0:
            fused_theta = self._apply_temporal_smoothing(fused_theta, fused_conf)

        # 8. Calculate model-based toric metrics; these are not clinical outcomes.
        abs_theta = abs(fused_theta)

        if abs_theta <= 1.5:
            clinical_impact = "ACCEPTABLE"
            quality_desc = "Heuristic angular-offset bin: <=1.5 deg."
        elif abs_theta <= 3.0:
            clinical_impact = "BORDERLINE"
            quality_desc = "Heuristic angular-offset bin: 1.5-3.0 deg."
        else:
            clinical_impact = "CRITICAL"
            quality_desc = "Heuristic angular-offset bin: >3.0 deg."

        # Residual cylinder relative to planned correction (Alpins vector magnitude).
        theta_rad = math.radians(abs_theta)
        astigmatism_loss_pct = float(min(100.0, 2.0 * math.sin(theta_rad) * 100.0))
        toric_effectiveness_loss_pct = float(
            100.0 * (1.0 - math.cos(2.0 * theta_rad))
        )

        # Intorsion vs Excyclotorsion direction:
        # Fused theta > 0 = counter-clockwise rotation of the eye image
        # Fused theta < 0 = clockwise rotation of the eye image
        # OD (Right Eye):
        #   Clockwise = Intorsion (superior meridian moves nasally)
        #   Counter-Clockwise = Excyclotorsion (superior meridian moves temporally)
        # OS (Left Eye):
        #   Counter-Clockwise = Intorsion (superior meridian moves nasally)
        #   Clockwise = Excyclotorsion (superior meridian moves temporally)
        if abs_theta < 0.2:
            torsion_dir = "NEUTRAL"
        elif laterality == "OD":
            torsion_dir = "INTORSION" if fused_theta < 0 else "EXCYCLOTORSION"
        else:  # OS
            torsion_dir = "INTORSION" if fused_theta > 0 else "EXCYCLOTORSION"

        is_valid = math.isfinite(fused_theta) and poc_conf > 0
        fail_kind = RegistrationFailureKind.OK if is_valid else RegistrationFailureKind.WEAK_CORRESPONDENCE
        fail_reason = "" if is_valid else self.last_angular_match.reason

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return CrossSystemRegistrationResult(
            valid=is_valid,
            failure=fail_kind,
            failure_reason=fail_reason,
            transformation_model=TransformationModel.ROTATION_ONLY if is_valid else TransformationModel.NONE,
            rotation_deg=float(fused_theta),
            clinical_impact=clinical_impact if is_valid else "UNAVAILABLE",
            astigmatism_loss_percent=astigmatism_loss_pct if is_valid else 0.0,
            toric_effectiveness_loss_percent=toric_effectiveness_loss_pct if is_valid else 0.0,
            torsion_direction=torsion_dir if is_valid else "UNAVAILABLE",
            laterality=laterality,
            translation_x=0.0,
            translation_y=0.0,
            scale=1.0,
            elita_cyclotorsion_deg=0.0,
            final_sitting_to_supine_deg=float(fused_theta) if is_valid else None,
            confidence=fused_conf if is_valid else 0.0,
            quality_assessment=quality_desc if is_valid else "No accepted angular measurement",
            pentacam_features_used=0,
            elita_features_used=0,
            processing_time_ms=dt_ms,
            angular_diagnostics=asdict(self.last_angular_match),
        )

    def _unwrap_pentacam(
        self,
        image: np.ndarray,
        result: PentacamDetectionResult,
    ) -> PolarImage:
        """Unwrap Pentacam iris into polar coordinates."""
        geom = result.geometry
        pe = geom.pupil
        le = geom.limbus

        if pe is None or le is None:
            return PolarImage()

        return self.unwrapper.unwrap(
            image=image,
            pupil_center=(pe.center_x, pe.center_y),
            pupil_axes=(pe.semi_major, pe.semi_minor),
            pupil_angle_deg=pe.angle_deg,
            limbus_center=(le.center_x, le.center_y),
            limbus_axes=(le.semi_major, le.semi_minor),
            limbus_angle_deg=le.angle_deg,
            validity_mask=registration_mask(image),
        )

    def _phase_correlate_cross(
        self,
        enh_ref: np.ndarray,
        enh_curr: np.ndarray,
        mask_ref: Optional[np.ndarray],
        mask_curr: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """Compatibility tuple for the masked angular matcher; PSR is unused."""
        match = masked_angular_match(enh_ref, enh_curr, mask_ref, mask_curr,
                                     self.max_rotation_search_deg,
                                     reference_cache=self._angular_reference_cache)
        self.last_angular_match = match
        return match.angle_deg, (match.score if match.valid else 0.0), 0.0

    def _apply_temporal_smoothing(self, current_theta: float, conf: float) -> float:
        """Apply exponential moving average for dynamic mode tracking."""
        if self._last_smooth_theta is None:
            self._last_smooth_theta = current_theta
            return current_theta

        alpha = float(np.clip(conf * 0.6, 0.15, 0.85))
        diff = (current_theta - self._last_smooth_theta + 180.0) % 360.0 - 180.0
        smooth_theta = self._last_smooth_theta + alpha * diff
        smooth_theta = (smooth_theta + 180.0) % 360.0 - 180.0

        self._last_smooth_theta = smooth_theta
        return float(smooth_theta)
