"""
Stream C: Deterministic Multi-Band Angular Intensity Profile Matcher for cyclotorsion detection.

Computes 1D circular normalized cross-correlation (NCC) of radial projection profiles
across the iris collarette, mid-iris, and ciliary zones.

Provides an ultra-fast (<0.2 ms), deterministic mathematical confirmation of rotational
shift with parabolic sub-pixel peak refinement.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class AngularProfileStream(BaseStream):
    """Deterministic 1D multi-band angular profile cross-correlation.

    Algorithm:
        1. Polar-unwrap both iris images using Daugman rubber sheet model
        2. Enhance polar textures with CLAHE and radial normalization
        3. Partition iris into multiple radial bands (collarette, mid-iris, ciliary)
        4. Compute 1D angular intensity profiles for each band
        5. Circular normalized cross-correlation (NCC) over surgical angular range
        6. Sub-pixel parabolic peak refinement
        7. Multi-band consensus voting and confidence scoring
    """

    def __init__(self, max_search_deg: float = 30.0):
        super().__init__(StreamName.ANGULAR_PROFILE)
        self.unwrapper = PolarUnwrapper()
        self.enhancer = IrisEnhancer()
        self.max_search_deg = max_search_deg

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
        polar_ref: Optional[Any] = None,
        polar_curr: Optional[Any] = None,
        enh_ref: Optional[np.ndarray] = None,
        enh_curr: Optional[np.ndarray] = None,
        **kwargs,
    ) -> StreamResult:
        """Compute cyclotorsion from angular intensity profiles."""
        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(metadata={"error": "incomplete_detection"})

        # Step 1: Polar unwrap (or reuse precomputed)
        if polar_ref is None:
            polar_ref = self.unwrapper.unwrap_from_detection(img_ref, detection_ref)
        if polar_curr is None:
            polar_curr = self.unwrapper.unwrap_from_detection(img_curr, detection_curr)

        if not polar_ref.valid or not polar_curr.valid:
            return self._make_result(metadata={"error": "polar_unwrap_failed"})

        # Step 2: Enhance (or reuse precomputed)
        if enh_ref is None:
            enh_ref = self.enhancer.enhance_polar(polar_ref.image, polar_ref.mask)
        if enh_curr is None:
            enh_curr = self.enhancer.enhance_polar(polar_curr.image, polar_curr.mask)

        num_radial, num_angles = enh_ref.shape[:2]
        deg_per_sample = 360.0 / float(num_angles)
        max_shift_samples = int(np.ceil(self.max_search_deg / deg_per_sample))

        # Check validity
        if polar_ref.mask is not None and polar_curr.mask is not None:
            combined_mask = (polar_ref.mask > 127) & (polar_curr.mask > 127)
            valid_frac = float(np.mean(combined_mask))
        else:
            valid_frac = 1.0

        if valid_frac < 0.25:
            return self._make_result(
                metadata={"error": "insufficient_valid_pixels", "valid_fraction": valid_frac}
            )

        # Step 3: Multi-band profile extraction
        # Band 1: Inner pupillary collarette (rows 10% to 40%)
        # Band 2: Mid-iris texture (rows 35% to 70%)
        # Band 3: Outer ciliary zone (rows 60% to 90%)
        b1_r1, b1_r2 = int(0.10 * num_radial), int(0.40 * num_radial)
        b2_r1, b2_r2 = int(0.35 * num_radial), int(0.70 * num_radial)
        b3_r1, b3_r2 = int(0.60 * num_radial), int(0.90 * num_radial)

        bands = [
            ("inner_collarette", b1_r1, b1_r2),
            ("mid_iris", b2_r1, b2_r2),
            ("outer_ciliary", b3_r1, b3_r2),
        ]

        band_estimates = []
        band_confidences = []

        for name, r1, r2 in bands:
            p_ref = np.mean(enh_ref[r1:r2, :], axis=0)
            p_curr = np.mean(enh_curr[r1:r2, :], axis=0)

            # High-pass filter profile to remove slow DC baseline drift
            kernel_size = max(num_angles // 16, 7)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smooth_ref = cv2.GaussianBlur(p_ref[:, None], (1, kernel_size), 0).flatten()
            smooth_curr = cv2.GaussianBlur(p_curr[:, None], (1, kernel_size), 0).flatten()
            sig_ref = p_ref - smooth_ref
            sig_curr = p_curr - smooth_curr

            shift_deg, ncc_score = self._correlate_profiles_1d(
                sig_ref, sig_curr, max_shift_samples, deg_per_sample
            )

            if ncc_score > 0.15:
                band_estimates.append(shift_deg)
                band_confidences.append(ncc_score)

        if not band_estimates:
            return self._make_result(
                metadata={"error": "low_correlation_all_bands"}
            )

        band_estimates = np.array(band_estimates)
        band_confidences = np.array(band_confidences)

        # Consensus weighting
        weights = band_confidences / np.sum(band_confidences)
        fused_deg = float(np.sum(band_estimates * weights))

        # Check band agreement
        max_diff = float(np.max(np.abs(band_estimates - fused_deg)))
        agreement_factor = 1.0 if max_diff < 1.5 else (0.7 if max_diff < 3.0 else 0.4)

        mean_ncc = float(np.mean(band_confidences))
        final_conf = float(np.clip(mean_ncc * agreement_factor * min(valid_frac / 0.5, 1.0), 0.0, 1.0))

        return self._make_result(
            torsion_deg=fused_deg,
            confidence=final_conf,
            inlier_count=len(band_estimates),
            metadata={
                "band_estimates": [float(x) for x in band_estimates],
                "band_confidences": [float(x) for x in band_confidences],
                "max_band_diff": float(max_diff),
                "valid_fraction": float(valid_frac),
            },
        )

    @staticmethod
    def _correlate_profiles_1d(
        sig_ref: np.ndarray,
        sig_curr: np.ndarray,
        max_shift: int,
        deg_per_sample: float,
    ) -> Tuple[float, float]:
        """Compute circular NCC over [-max_shift, +max_shift] with parabolic interpolation."""
        n = len(sig_ref)
        shifts = np.arange(-max_shift, max_shift + 1)
        ncc_values = np.zeros(len(shifts), dtype=np.float32)

        norm_ref = np.linalg.norm(sig_ref) + 1e-8

        for i, s in enumerate(shifts):
            # Circular roll
            rolled_curr = np.roll(sig_curr, s)
            norm_curr = np.linalg.norm(rolled_curr) + 1e-8
            ncc = float(np.dot(sig_ref, rolled_curr) / (norm_ref * norm_curr))
            ncc_values[i] = ncc

        best_idx = int(np.argmax(ncc_values))
        best_ncc = float(ncc_values[best_idx])
        best_shift = shifts[best_idx]

        # 3-point parabolic interpolation for sub-pixel peak
        if 0 < best_idx < len(shifts) - 1:
            y0 = ncc_values[best_idx - 1]
            y1 = ncc_values[best_idx]
            y2 = ncc_values[best_idx + 1]
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            if abs(denom) > 1e-8:
                delta = float((y0 - y2) / denom)
                delta = float(np.clip(delta, -0.5, 0.5))
            else:
                delta = 0.0
            sub_shift = float(best_shift) + delta
        else:
            sub_shift = float(best_shift)

        # Convert shift to degrees: positive sub_shift corresponds to counter-clockwise rotation
        torsion_deg = float(sub_shift * deg_per_sample)
        torsion_deg = (torsion_deg + 180.0) % 360.0 - 180.0

        return torsion_deg, best_ncc
