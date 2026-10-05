"""
Stream A: Phase-Only Correlation (POC) for cyclotorsion detection.

Computes the rotational shift between two polar-unwrapped iris images
using FFT-based phase correlation along the angular axis.

This is the most reliable baseline method, working well on images
with sufficient iris texture regardless of illumination changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class PhaseCorrelationStream(BaseStream):
    """FFT-based phase-only correlation for angular shift detection.

    Algorithm:
        1. Polar-unwrap both iris images
        2. Enhance with CLAHE + column normalisation
        3. Compute cross-power spectrum along angular axis
        4. Find peak in inverse FFT → angular shift = torsion
        5. Sub-pixel refinement via parabolic interpolation

    The confidence is derived from the peak-to-sidelobe ratio (PSR)
    and the percentage of valid pixels in both polar images.
    """

    def __init__(self):
        super().__init__(StreamName.PHASE_CORRELATION)
        cfg = get_config().registration
        self.unwrapper = PolarUnwrapper()
        self.enhancer = IrisEnhancer()
        self.upsample_factor = cfg.poc_upsample_factor

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
        """Compute torsion via phase correlation of polar iris images."""

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        # Step 1: Polar unwrap using Daugman rubber sheet (or reuse precomputed)
        if polar_ref is None:
            polar_ref = self.unwrapper.unwrap_from_detection(img_ref, detection_ref)
        if polar_curr is None:
            polar_curr = self.unwrapper.unwrap_from_detection(img_curr, detection_curr)

        if not polar_ref.valid or not polar_curr.valid:
            return self._make_result(
                metadata={"error": "polar_unwrap_failed"}
            )

        # Step 2: Enhance polar textures (or reuse precomputed)
        if enh_ref is None:
            enh_ref = self.enhancer.enhance_polar(polar_ref.image, polar_ref.mask)
        if enh_curr is None:
            enh_curr = self.enhancer.enhance_polar(polar_curr.image, polar_curr.mask)

        num_radial, num_angles = enh_ref.shape[:2]
        deg_per_sample = 360.0 / float(num_angles)

        # Step 3: Compute valid pixel coverage
        if polar_ref.mask is not None and polar_curr.mask is not None:
            combined_mask = (polar_ref.mask > 127) & (polar_curr.mask > 127)
            valid_frac = float(np.mean(combined_mask))
        else:
            valid_frac = 1.0

        if valid_frac < 0.25:
            return self._make_result(
                metadata={"error": "insufficient_valid_pixels", "valid_fraction": valid_frac}
            )

        # Step 4: 2D Fourier Phase Correlation with radial-only Hann window
        # (Radial window prevents boundary edge ringing; periodic angular axis needs no window)
        hann_rad = np.hanning(num_radial).astype(np.float32)[:, None]
        window = np.repeat(hann_rad, num_angles, axis=1)

        src1 = enh_ref.astype(np.float32)
        src2 = enh_curr.astype(np.float32)

        shift_global, resp_global = cv2.phaseCorrelate(src1, src2, window)
        ang_global = (-shift_global[0] * deg_per_sample + 180.0) % 360.0 - 180.0

        # Step 5: Multi-band radial cross-validation (pupillary collarette vs outer ciliary)
        half_r = max(num_radial // 2, 8)
        win_half = window[:half_r, :]
        shift_inner, resp_inner = cv2.phaseCorrelate(src1[:half_r, :], src2[:half_r, :], win_half)
        shift_outer, resp_outer = cv2.phaseCorrelate(src1[half_r:, :], src2[half_r:, :], win_half)

        ang_inner = (-shift_inner[0] * deg_per_sample + 180.0) % 360.0 - 180.0
        ang_outer = (-shift_outer[0] * deg_per_sample + 180.0) % 360.0 - 180.0

        diff_inner = abs((ang_inner - ang_global + 180.0) % 360.0 - 180.0)
        diff_outer = abs((ang_outer - ang_global + 180.0) % 360.0 - 180.0)

        # Consensus weighting
        agree_inner = diff_inner < 2.0
        agree_outer = diff_outer < 2.0

        if agree_inner and agree_outer:
            # Both radial zones agree — excellent quality
            fused_ang = float(ang_global)
            band_agreement = 1.0
        elif agree_inner and resp_inner >= resp_outer:
            # Inner zone cleaner (e.g. eyelids occluding outer zone)
            fused_ang = float(ang_inner)
            band_agreement = 0.75
        elif agree_outer:
            # Outer zone cleaner (e.g. pupil dilation/illumination artifact)
            fused_ang = float(ang_outer)
            band_agreement = 0.75
        else:
            fused_ang = float(ang_global)
            band_agreement = 0.5

        # Confidence: response * band agreement * validity
        base_resp = max(resp_global, resp_inner, resp_outer)
        confidence = float(np.clip(base_resp * band_agreement * min(valid_frac / 0.5, 1.0), 0.0, 1.0))

        return self._make_result(
            torsion_deg=fused_ang,
            confidence=confidence,
            metadata={
                "response": float(resp_global),
                "resp_inner": float(resp_inner),
                "resp_outer": float(resp_outer),
                "ang_inner": float(ang_inner),
                "ang_outer": float(ang_outer),
                "band_agreement": float(band_agreement),
                "valid_fraction": float(valid_frac),
                "num_angles": num_angles,
            },
        )

    def _phase_correlate_1d(
        self,
        signal_ref: np.ndarray,
        signal_curr: np.ndarray,
        num_angles: int,
    ) -> tuple:
        """1-D phase-only correlation with sub-sample refinement.

        Returns
        -------
        shift_deg : float
            Angular shift in degrees.
        psr : float
            Peak-to-sidelobe ratio.
        peak_val : float
            Peak correlation value.
        """
        n = len(signal_ref)

        # FFT
        fft_ref = np.fft.fft(signal_ref)
        fft_curr = np.fft.fft(signal_curr)

        # Cross-power spectrum
        cross = fft_ref * np.conj(fft_curr)
        magnitude = np.abs(cross)
        magnitude = np.maximum(magnitude, 1e-10)
        phase_only = cross / magnitude

        # Inverse FFT → correlation
        correlation = np.real(np.fft.ifft(phase_only))

        # Find peak
        peak_idx = np.argmax(correlation)
        peak_val = correlation[peak_idx]

        # Sub-pixel refinement via parabolic interpolation
        if 0 < peak_idx < n - 1:
            y_left = correlation[peak_idx - 1]
            y_center = correlation[peak_idx]
            y_right = correlation[peak_idx + 1]
            denom = 2.0 * (2.0 * y_center - y_left - y_right)
            if abs(denom) > 1e-10:
                sub_pixel = (y_left - y_right) / denom
            else:
                sub_pixel = 0.0
            refined_idx = peak_idx + sub_pixel
        else:
            refined_idx = float(peak_idx)

        # Handle wrap-around: if shift > half range, it's negative
        if refined_idx > n / 2:
            refined_idx -= n

        # Convert sample shift to degrees
        deg_per_sample = 360.0 / num_angles
        shift_deg = refined_idx * deg_per_sample

        # Peak-to-sidelobe ratio
        sidelobes = np.delete(correlation, peak_idx)
        sidelobe_mean = np.mean(sidelobes)
        sidelobe_std = np.std(sidelobes) + 1e-10
        psr = (peak_val - sidelobe_mean) / sidelobe_std

        return float(shift_deg), float(psr), float(peak_val)

    @staticmethod
    def _compute_confidence(psr: float, valid_frac: float, peak_val: float) -> float:
        """Compute confidence from PSR, valid fraction, and peak value.

        Higher PSR → sharper peak → more reliable estimate.
        Higher valid fraction → more data → more reliable.
        """
        # PSR confidence: sigmoid mapping, PSR=5 → ~0.5, PSR=15 → ~0.95
        psr_conf = 1.0 / (1.0 + np.exp(-0.3 * (psr - 8.0)))

        # Valid fraction penalty
        validity_conf = min(valid_frac / 0.6, 1.0)

        # Peak value check (should be positive and significant)
        peak_conf = min(max(peak_val, 0.0), 1.0)

        return float(np.clip(psr_conf * validity_conf * peak_conf, 0.0, 1.0))
