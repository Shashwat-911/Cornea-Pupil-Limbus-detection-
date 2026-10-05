"""
Stream G: Polar Lucas-Kanade Dense / Multi-Point Optical Flow.

Estimates cyclotorsion by tracking local iris texture patches along the
polar coordinate manifold:
    1. In polar coordinates (r, θ), ocular cyclotorsion maps strictly
       to horizontal displacement (Δθ = constant along the θ axis).
    2. Pyramidal Lucas-Kanade differential optical flow tracks salient
       iris crypts and collarette features.
    3. Radial displacement (|Δr| > threshold) is filtered out to decouple
       true ocular cyclotorsion from pupillary dilation / hippus.
    4. Robust median statistics and MAD (Median Absolute Deviation)
       reject outliers and calculate sub-pixel rotational displacement.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

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


class PolarOpticalFlowStream(BaseStream):
    """Polar coordinate Lucas-Kanade optical flow cyclotorsion stream."""

    def __init__(
        self,
        max_corners: int = 120,
        win_size: tuple[int, int] = (25, 15),
        max_level: int = 2,
    ):
        super().__init__(StreamName.POLAR_OPTICAL_FLOW)
        self.unwrapper = PolarUnwrapper()
        self.enhancer = IrisEnhancer()
        self.max_corners = max_corners
        self.win_size = win_size
        self.max_level = max_level
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def _prepare_uint8(self, img_float_or_uint: np.ndarray) -> np.ndarray:
        """Ensure polar image is single-channel uint8."""
        if img_float_or_uint.ndim == 3:
            img = cv2.cvtColor(img_float_or_uint, cv2.COLOR_BGR2GRAY)
        else:
            img = img_float_or_uint

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return self._clahe.apply(img)

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
        """Track polar optical flow displacements and compute torsion."""
        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(metadata={"error": "incomplete_detection"})

        # Step 1: Polar unwrap (reuse if provided)
        if polar_ref is None:
            polar_ref = self.unwrapper.unwrap_from_detection(img_ref, detection_ref)
        if polar_curr is None:
            polar_curr = self.unwrapper.unwrap_from_detection(img_curr, detection_curr)

        if not polar_ref.valid or not polar_curr.valid:
            return self._make_result(metadata={"error": "polar_unwrap_failed"})

        u_ref = self._prepare_uint8(polar_ref.image)
        u_curr = self._prepare_uint8(polar_curr.image)

        H, W = u_ref.shape[:2]
        deg_per_px = 360.0 / float(W)

        # Build mask for valid iris features (omit extreme inner/outer boundaries)
        mask_valid = (polar_ref.mask > 127).astype(np.uint8) if polar_ref.mask is not None else np.ones((H, W), dtype=np.uint8)
        mask_valid[:8, :] = 0
        mask_valid[-8:, :] = 0

        # Step 2: Detect salient features on reference polar image
        pts_ref = cv2.goodFeaturesToTrack(
            u_ref,
            maxCorners=self.max_corners,
            qualityLevel=0.015,
            minDistance=6,
            mask=mask_valid,
        )

        if pts_ref is None or len(pts_ref) < 8:
            return self._make_result(
                metadata={"error": "insufficient_features", "num_points": 0}
            )

        # Step 3: Track with Lucas-Kanade
        pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
            u_ref,
            u_curr,
            pts_ref,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.015),
        )

        status_mask = (status.ravel() == 1) & (err.ravel() < 25.0)
        if np.sum(status_mask) < 6:
            return self._make_result(
                metadata={"error": "tracking_failed", "tracked": int(np.sum(status_mask))}
            )

        p0 = pts_ref[status_mask].reshape(-1, 2)
        p1 = pts_curr[status_mask].reshape(-1, 2)

        # Displacements
        dx = p1[:, 0] - p0[:, 0]
        dy = p1[:, 1] - p0[:, 1]

        # Step 4: Outlier rejection
        # In polar coordinates, cyclotorsion is strictly horizontal (dx).
        # Vertical motion (dy) indicates non-rigid deformation (dilation / eyelashes).
        rigid_mask = np.abs(dy) <= 2.5
        if np.sum(rigid_mask) < 5:
            return self._make_result(
                metadata={"error": "no_rigid_consensus", "rigid_count": int(np.sum(rigid_mask))}
            )

        dx_rigid = dx[rigid_mask]

        # Robust statistics (Median + MAD)
        median_dx = float(np.median(dx_rigid))
        mad = float(np.median(np.abs(dx_rigid - median_dx)))
        inlier_mask = np.abs(dx_rigid - median_dx) <= max(1.5, 2.5 * mad)

        dx_inliers = dx_rigid[inlier_mask]
        num_inliers = len(dx_inliers)

        if num_inliers < 4:
            return self._make_result(
                metadata={"error": "too_few_inliers", "inliers": num_inliers}
            )

        final_shift_px = -float(np.mean(dx_inliers))
        torsion_deg = final_shift_px * deg_per_px

        # Confidence
        inlier_ratio = num_inliers / float(len(pts_ref))
        dispersion_penalty = max(0.0, 1.0 - (mad / 2.0))
        confidence = float(np.clip(inlier_ratio * 0.7 + dispersion_penalty * 0.3, 0.0, 1.0))

        return self._make_result(
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=num_inliers,
            metadata={
                "shift_px": final_shift_px,
                "mad_px": mad,
                "total_points": len(pts_ref),
                "tracked_points": len(p0),
                "inlier_points": num_inliers,
            },
        )
