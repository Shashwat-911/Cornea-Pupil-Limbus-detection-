"""
Stream B: Deep feature matching for cyclotorsion detection.

Uses ORB/SIFT feature detection with brute-force matching as a
classical fallback, with optional LoFTR/SuperGlue deep matcher
when kornia or equivalent is available.

This stream is particularly robust to illumination changes and
partial iris occlusion (eyelids).
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class DeepMatcherStream(BaseStream):
    """Deep/classical feature matching for rotation estimation.

    Algorithm:
        1. Extract iris crops from both images
        2. Detect keypoints (ORB or SIFT)
        3. Match descriptors with ratio test
        4. Compute rotation from matched point correspondences
        5. RANSAC to reject outliers

    Falls back to ORB if SIFT/LoFTR unavailable.
    """

    def __init__(self):
        super().__init__(StreamName.DEEP_MATCHER)
        cfg = get_config().registration
        self.max_keypoints = cfg.deep_matcher_max_keypoints
        self.enhancer = IrisEnhancer()

        # Try SIFT first (more robust), fallback to ORB
        try:
            self._detector = cv2.SIFT_create(nfeatures=self.max_keypoints)
            self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self._detector_name = "SIFT"
        except cv2.error:
            self._detector = cv2.ORB_create(nfeatures=self.max_keypoints)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self._detector_name = "ORB"

        logger.info("DeepMatcherStream using %s detector", self._detector_name)

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Match features between iris crops and compute rotation."""

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        le_ref = detection_ref.limbus.ellipse
        le_curr = detection_curr.limbus.ellipse

        # Extract iris crops
        crop_ref = self.enhancer.extract_iris_crop(
            img_ref,
            center=(le_ref.center_x, le_ref.center_y),
            radius=le_ref.radius,
        )
        crop_curr = self.enhancer.extract_iris_crop(
            img_curr,
            center=(le_curr.center_x, le_curr.center_y),
            radius=le_curr.radius,
        )

        # Convert to grayscale for detection
        gray_ref = self._to_gray(crop_ref['image'])
        gray_curr = self._to_gray(crop_curr['image'])

        # Detect and compute
        kp_ref, desc_ref = self._detector.detectAndCompute(gray_ref, None)
        kp_curr, desc_curr = self._detector.detectAndCompute(gray_curr, None)

        if desc_ref is None or desc_curr is None:
            return self._make_result(
                metadata={"error": "no_descriptors"}
            )

        if len(kp_ref) < 5 or len(kp_curr) < 5:
            return self._make_result(
                metadata={"error": "insufficient_keypoints",
                           "kp_ref": len(kp_ref),
                           "kp_curr": len(kp_curr)}
            )

        # Match with ratio test
        try:
            raw_matches = self._matcher.knnMatch(desc_ref, desc_curr, k=2)
        except cv2.error:
            return self._make_result(metadata={"error": "matching_failed"})

        # Lowe's ratio test
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return self._make_result(
                metadata={"error": "insufficient_matches",
                           "num_matches": len(good_matches)}
            )

        # Extract matched point coordinates in crop space
        pts_ref = np.float32([kp_ref[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])

        # Compute rotation from matched points relative to crop centres
        crop_center = np.array([crop_ref['target_size'] / 2.0,
                                 crop_ref['target_size'] / 2.0])

        angles_ref = np.arctan2(
            pts_ref[:, 1] - crop_center[1],
            pts_ref[:, 0] - crop_center[0],
        )
        angles_curr = np.arctan2(
            pts_curr[:, 1] - crop_center[1],
            pts_curr[:, 0] - crop_center[0],
        )

        # Angular differences
        diffs = (angles_curr - angles_ref + np.pi) % (2 * np.pi) - np.pi

        # RANSAC-style robust estimation
        torsion_rad, inlier_mask = self._robust_rotation_estimate(diffs)

        if torsion_rad is None:
            return self._make_result(
                metadata={"error": "ransac_failed"}
            )

        torsion_deg = float(np.degrees(torsion_rad))
        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = inlier_count / len(diffs)

        # Confidence from inlier ratio and match count
        consistency = 1.0 / (1.0 + np.std(diffs[inlier_mask]))
        match_conf = min(len(good_matches) / 50.0, 1.0)
        confidence = float(np.clip(
            inlier_ratio * consistency * match_conf, 0.0, 1.0
        ))

        return self._make_result(
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=inlier_count,
            metadata={
                "detector": self._detector_name,
                "total_matches": len(good_matches),
                "inlier_ratio": float(inlier_ratio),
                "kp_ref": len(kp_ref),
                "kp_curr": len(kp_curr),
            },
        )

    def _robust_rotation_estimate(self, diffs: np.ndarray, max_iters: int = 100):
        """RANSAC-style robust median + MAD filtering.

        Returns
        -------
        torsion_rad : float or None
        inlier_mask : np.ndarray or None
        """
        if len(diffs) < 3:
            return None, None

        best_inliers = None
        best_estimate = None
        best_count = 0

        for _ in range(max_iters):
            # Random sample
            idx = np.random.randint(0, len(diffs))
            hypothesis = diffs[idx]

            # Count inliers within threshold
            residuals = np.abs(
                (diffs - hypothesis + np.pi) % (2 * np.pi) - np.pi
            )
            inliers = residuals < np.radians(2.0)  # 2° threshold
            count = np.sum(inliers)

            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_estimate = hypothesis

        if best_count < 3:
            # Fallback to median + MAD
            median = np.median(diffs)
            mad = np.median(np.abs(diffs - median))
            best_inliers = np.abs(diffs - median) < 2.5 * max(mad, 0.005)
            if np.sum(best_inliers) < 3:
                return None, None

        # Refine with inlier mean
        torsion_rad = float(np.mean(diffs[best_inliers]))
        return torsion_rad, best_inliers

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
