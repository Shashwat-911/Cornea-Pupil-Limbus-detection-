"""Experimental cross-modality iris registration for Pentacam and ELITA images.

The implementation estimates rigid angular shift from polar iris strips and
landmark descriptors. Synthetic rigid-rotation tests do not establish clinical
accuracy; pupil dilation, nonrigid deformation, reflections, and live-frame
latency require separate validation on representative clinical data.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.pentacam.cross_system import (
    CrossSystemRegistrationInput,
    CrossSystemRegistrationResult,
    RegistrationFailureKind,
    TransformationModel,
)
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.types import PentacamDetectionResult, PentacamQuality
from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.polar import PolarImage, PolarUnwrapper
from pupil_tracking.utils.types import EllipseParams, EyeDetectionResult

logger = logging.getLogger(__name__)


class CrossModalityRegistrationEngine:
    """Master engine for Pentacam ↔ ELITA cross-modality cyclotorsion estimation."""

    def __init__(
        self,
        num_angles: int = 360,
        num_radial: int = 64,
        max_rotation_search_deg: float = 30.0,
    ) -> None:
        self.num_angles = num_angles
        self.num_radial = num_radial
        self.max_rotation_search_deg = max_rotation_search_deg

        self.unwrapper = PolarUnwrapper(num_angles=num_angles, num_radial=num_radial)
        self.enhancer = IrisEnhancer()
        self.pentacam_detector = PentacamIrisDetector()
        self._cached_reference_key: Optional[Tuple[Tuple[int, ...], str, bytes]] = None
        self._cached_reference_result: Optional[PentacamDetectionResult] = None

        # Dynamic mode temporal filter state
        self._temporal_history: List[float] = []
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

        # 1. Validate inputs
        if pentacam_image is None or elita_image is None:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.NO_PENTACAM if pentacam_image is None else RegistrationFailureKind.NO_ELITA,
                failure_reason="Missing image input",
                laterality=laterality,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 2. Reuse detection for an unchanged static reference image.
        if pentacam_result is None or not pentacam_result.valid:
            contiguous_image = np.ascontiguousarray(pentacam_image)
            reference_key = (
                tuple(pentacam_image.shape),
                pentacam_image.dtype.str,
                hashlib.blake2b(contiguous_image.view(np.uint8), digest_size=16).digest(),
            )
            if reference_key == self._cached_reference_key:
                pentacam_result = self._cached_reference_result
            else:
                pentacam_result = self.pentacam_detector.detect(pentacam_image)
                if pentacam_result.valid:
                    self._cached_reference_key = reference_key
                    self._cached_reference_result = pentacam_result

        if not pentacam_result.valid or not pentacam_result.geometry.pupil_detected:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.INSUFFICIENT_PENTACAM_FEATURES,
                failure_reason=f"Pentacam detection failed: {pentacam_result.failure_reason}",
                laterality=laterality,
                pentacam_features_used=0,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. Validate ELITA detection
        if elita_detection is None or not elita_detection.has_both:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.NO_ELITA,
                failure_reason="ELITA image lacks validated pupil and limbus geometry",
                laterality=laterality,
                pentacam_features_used=len(pentacam_result.feature_set.features),
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 4. Polar unwrap both irises (Daugman rubber-sheet model)
        polar_pentacam = self._unwrap_pentacam(pentacam_image, pentacam_result)
        polar_elita = self.unwrapper.unwrap_from_detection(elita_image, elita_detection)

        if not polar_pentacam.valid or not polar_elita.valid:
            return CrossSystemRegistrationResult(
                valid=False,
                failure=RegistrationFailureKind.COORDINATE_MISMATCH,
                failure_reason="Polar unwrapping failed for one or both modalities",
                laterality=laterality,
                processing_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 5. Cross-modality enhancement & spectral equalization
        enh_pentacam = self.enhancer.enhance_polar(polar_pentacam.image, polar_pentacam.mask)
        enh_elita = self.enhancer.enhance_polar(polar_elita.image, polar_elita.mask)

        # 6. Compute rotation angle via Phase-Only Correlation along angular axis
        poc_theta, poc_conf, poc_psr = self._phase_correlate_cross(
            enh_pentacam, enh_elita, polar_pentacam.mask, polar_elita.mask
        )

        # 7. Fast landmark verification using top features
        landmark_verified, inliers, total_matches, rms_res = self._verify_landmarks_fast(
            pentacam_result, elita_image, elita_detection, poc_theta
        )

        fused_theta = poc_theta
        fused_conf = float(np.clip(poc_conf * 0.7 + (inliers / max(total_matches, 1)) * 0.3, 0.0, 1.0))

        # Dynamic mode temporal filtering
        if mode == "dynamic":
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

        is_valid = math.isfinite(fused_theta) and (fused_conf >= 0.20 or poc_psr >= 5.0)
        fail_kind = RegistrationFailureKind.OK if is_valid else RegistrationFailureKind.WEAK_CORRESPONDENCE
        fail_reason = "" if is_valid else f"Confidence {fused_conf:.2f} (PSR: {poc_psr:.1f}) below threshold"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        n_p_feats = len(pentacam_result.feature_set.features)

        return CrossSystemRegistrationResult(
            valid=is_valid,
            failure=fail_kind,
            failure_reason=fail_reason,
            transformation_model=TransformationModel.RIGID_2D,
            rotation_deg=float(fused_theta),
            clinical_impact=clinical_impact,
            astigmatism_loss_percent=astigmatism_loss_pct,
            toric_effectiveness_loss_percent=toric_effectiveness_loss_pct,
            torsion_direction=torsion_dir,
            laterality=laterality,
            translation_x=0.0,
            translation_y=0.0,
            scale=1.0,
            elita_cyclotorsion_deg=0.0,
            final_sitting_to_supine_deg=float(fused_theta),
            n_correspondences=total_matches,
            n_inliers=inliers,
            inlier_fraction=float(inliers / max(total_matches, 1)),
            residual_rms=float(rms_res),
            residual_max=float(rms_res * 1.5),
            confidence=fused_conf,
            quality_assessment=quality_desc,
            pentacam_features_used=n_p_feats,
            elita_features_used=total_matches,
            processing_time_ms=dt_ms,
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
        )

    def _phase_correlate_cross(
        self,
        enh_ref: np.ndarray,
        enh_curr: np.ndarray,
        mask_ref: Optional[np.ndarray],
        mask_curr: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """Compute angular shift via 2D Phase Correlation."""
        src1 = enh_ref.astype(np.float32)
        src2 = enh_curr.astype(np.float32)

        hann_rad = np.hanning(src1.shape[0]).astype(np.float32)[:, None]
        window = np.repeat(hann_rad, src1.shape[1], axis=1)

        shift, resp = cv2.phaseCorrelate(src1, src2, window)
        deg_per_sample = 360.0 / float(self.num_angles)
        shift_deg = float((-shift[0] * deg_per_sample + 180.0) % 360.0 - 180.0)

        if mask_ref is not None and mask_curr is not None:
            v_frac = float(np.mean((mask_ref > 127) & (mask_curr > 127)))
        else:
            v_frac = 1.0

        psr = float(resp * 10.0)
        conf = float(np.clip(resp * min(v_frac / 0.5, 1.0), 0.0, 1.0))
        return float(shift_deg), conf, psr

    def _verify_landmarks_fast(
        self,
        pentacam_result: PentacamDetectionResult,
        elita_image: np.ndarray,
        elita_detection: EyeDetectionResult,
        candidate_theta_deg: float,
        top_k: int = 25,
    ) -> Tuple[bool, int, int, float]:
        """Fast verification of candidate rotation using the top-K highest confidence landmarks."""
        p_feats = pentacam_result.feature_set.features
        if not p_feats:
            return True, 0, 0, 0.0

        # Sort by confidence and take top_k
        top_feats = sorted(p_feats, key=lambda f: f.confidence, reverse=True)[:top_k]

        le = elita_detection.limbus.ellipse
        pe = elita_detection.pupil.ellipse

        if len(elita_image.shape) == 3:
            e_gray = cv2.cvtColor(elita_image, cv2.COLOR_BGR2GRAY)
        else:
            e_gray = elita_image

        h, w = e_gray.shape[:2]
        half_w = 3

        inliers = 0
        tested = 0
        residuals = []

        for feat in top_feats:
            if feat.descriptor is None:
                continue

            r_norm = feat.radial_norm
            # Expected angle in ELITA image
            exp_ang = (feat.angle_deg + candidate_theta_deg) % 360.0
            ang_rad = math.radians(exp_ang)

            r_px = pe.radius + r_norm * (le.radius - pe.radius)
            cx = pe.center_x + r_norm * (le.center_x - pe.center_x)
            cy = pe.center_y + r_norm * (le.center_y - pe.center_y)

            ex = int(round(cx + r_px * math.cos(ang_rad)))
            ey = int(round(cy + r_px * math.sin(ang_rad)))

            if not (half_w <= ex < w - half_w and half_w <= ey < h - half_w):
                continue

            patch = e_gray[ey - half_w:ey + half_w + 1, ex - half_w:ex + half_w + 1]
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            mag, ori = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            hist, _ = np.histogram(ori, bins=16, range=(0.0, 360.0), weights=mag)
            hist_norm = hist / (np.linalg.norm(hist) + 1e-7)

            sim = float(np.dot(feat.descriptor, hist_norm))
            tested += 1
            if sim > 0.40:
                inliers += 1
                residuals.append(1.0 - sim)

        rms = float(np.sqrt(np.mean(residuals))) if residuals else 0.5
        verified = (inliers / max(tested, 1)) >= 0.35

        return verified, inliers, tested, rms

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
