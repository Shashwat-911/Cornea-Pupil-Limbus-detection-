"""
Stream D: Limbal vessel tracking for cyclotorsion detection.

Detects and tracks blood vessel patterns near the limbus using
morphological vessel enhancement (Frangi/ridge filter) and
bifurcation point matching.

Vessel patterns are highly stable across time (unlike iris texture
which changes with dilation), making them excellent landmarks.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

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


class VesselTrackerStream(BaseStream):
    """Track limbal vessel bifurcation points for rotation estimation.

    Algorithm:
        1. Extract limbal ring (just outside limbus boundary)
        2. Enhance vessels using morphological top-hat + Frangi-like filter
        3. Skeletonise vessel map
        4. Detect bifurcation points (3+ neighbours in skeleton)
        5. Match bifurcations between images by angular position
        6. Compute rotation from matched bifurcation pairs
    """

    def __init__(self):
        super().__init__(StreamName.LIMBAL_VESSELS)
        cfg = get_config().registration
        self.min_bifurcations = cfg.vessel_min_bifurcations
        self.enhancer = IrisEnhancer()

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Detect vessel bifurcations and compute rotation."""

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        # Extract and enhance limbal regions
        vessel_ref, center_ref = self._extract_vessel_map(img_ref, detection_ref)
        vessel_curr, center_curr = self._extract_vessel_map(img_curr, detection_curr)

        if vessel_ref is None or vessel_curr is None:
            return self._make_result(
                metadata={"error": "vessel_extraction_failed"}
            )

        # Detect bifurcation points
        bif_ref = self._detect_bifurcations(vessel_ref)
        bif_curr = self._detect_bifurcations(vessel_curr)

        if len(bif_ref) < self.min_bifurcations:
            return self._make_result(
                metadata={"error": "insufficient_ref_bifurcations",
                           "found": len(bif_ref)}
            )
        if len(bif_curr) < self.min_bifurcations:
            return self._make_result(
                metadata={"error": "insufficient_curr_bifurcations",
                           "found": len(bif_curr)}
            )

        # Compute angles relative to centres
        le_ref = detection_ref.limbus.ellipse
        le_curr = detection_curr.limbus.ellipse

        angles_ref = np.array([
            np.arctan2(y - center_ref[1], x - center_ref[0])
            for x, y in bif_ref
        ])
        angles_curr = np.array([
            np.arctan2(y - center_curr[1], x - center_curr[0])
            for x, y in bif_curr
        ])

        # Match by angular proximity
        matches = self._match_bifurcations(angles_ref, angles_curr)

        if len(matches) < self.min_bifurcations:
            return self._make_result(
                metadata={"error": "insufficient_matches",
                           "matched": len(matches)}
            )

        # Compute angular differences
        diffs = []
        for ri, ci in matches:
            diff = (angles_curr[ci] - angles_ref[ri] + np.pi) % (2 * np.pi) - np.pi
            diffs.append(diff)
        diffs = np.array(diffs)

        # Robust estimation
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        inliers = np.abs(diffs - median_diff) < 2.5 * max(mad, 0.01)

        if np.sum(inliers) < self.min_bifurcations:
            return self._make_result(
                metadata={"error": "too_few_inliers"}
            )

        torsion_rad = float(np.mean(diffs[inliers]))
        torsion_deg = float(np.degrees(torsion_rad))

        inlier_ratio = np.sum(inliers) / len(diffs)
        consistency = 1.0 / (1.0 + np.std(diffs[inliers]) * 10.0)
        confidence = float(np.clip(
            0.8 * inlier_ratio * consistency, 0.0, 1.0
        ))

        return self._make_result(
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=int(np.sum(inliers)),
            metadata={
                "bifurcations_ref": len(bif_ref),
                "bifurcations_curr": len(bif_curr),
                "matched": len(matches),
            },
        )

    def _extract_vessel_map(
        self,
        image: np.ndarray,
        detection: EyeDetectionResult,
    ) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
        """Extract and enhance vessel map from the limbal region.

        Returns
        -------
        vessel_map : np.ndarray or None
            Binary vessel map in original image coordinates.
        center : (float, float)
            Limbus centre.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        le = detection.limbus.ellipse
        center = (le.center_x, le.center_y)
        r_inner = le.radius * 0.95   # Just inside limbus
        r_outer = le.radius * 1.25   # Into sclera

        h, w = gray.shape
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        ring_mask = ((dist >= r_inner) & (dist <= r_outer)).astype(np.uint8) * 255

        if np.sum(ring_mask > 0) < 100:
            return None, center

        # Enhance vessels using morphological top-hat
        masked = cv2.bitwise_and(gray, gray, mask=ring_mask)

        # Multi-scale top-hat for vessel enhancement
        vessel_response = np.zeros_like(gray, dtype=np.float32)
        for ksize in [3, 5, 7]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            tophat = cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, kernel)
            vessel_response += tophat.astype(np.float32)

        # Normalise and threshold
        vessel_response = cv2.normalize(
            vessel_response, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        _, vessel_binary = cv2.threshold(
            vessel_response, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Apply ring mask
        vessel_binary = cv2.bitwise_and(vessel_binary, ring_mask)

        # Thin to skeleton
        vessel_binary = self._skeletonise(vessel_binary)

        return vessel_binary, center

    def _skeletonise(self, binary: np.ndarray) -> np.ndarray:
        """Morphological skeletonisation."""
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        img = binary.copy()

        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break

        return skeleton

    def _detect_bifurcations(
        self,
        skeleton: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """Detect bifurcation points in a skeleton image.

        A bifurcation is a pixel with 3 or more neighbours in the skeleton.
        """
        if skeleton is None or np.sum(skeleton > 0) < 10:
            return []

        # Count neighbours using convolution
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=np.float32)

        skel_binary = (skeleton > 0).astype(np.float32)
        neighbour_count = cv2.filter2D(skel_binary, -1, kernel)

        # Bifurcations: skeleton pixel with 3+ neighbours
        bifurcation_mask = (skel_binary > 0) & (neighbour_count >= 3)

        # Get coordinates
        ys, xs = np.where(bifurcation_mask)

        # Non-maximum suppression: merge nearby bifurcations
        if len(xs) == 0:
            return []

        points = list(zip(xs.astype(float), ys.astype(float)))
        merged = self._nms_points(points, min_dist=10.0)

        return merged

    @staticmethod
    def _nms_points(
        points: List[Tuple[float, float]],
        min_dist: float = 10.0,
    ) -> List[Tuple[float, float]]:
        """Merge nearby points via greedy non-maximum suppression."""
        if not points:
            return []

        remaining = list(points)
        merged = []

        while remaining:
            p = remaining.pop(0)
            cluster = [p]

            i = 0
            while i < len(remaining):
                dist = np.sqrt(
                    (remaining[i][0] - p[0])**2 +
                    (remaining[i][1] - p[1])**2
                )
                if dist < min_dist:
                    cluster.append(remaining.pop(i))
                else:
                    i += 1

            # Average the cluster
            cx = np.mean([c[0] for c in cluster])
            cy = np.mean([c[1] for c in cluster])
            merged.append((float(cx), float(cy)))

        return merged

    def _match_bifurcations(
        self,
        angles_ref: np.ndarray,
        angles_curr: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Match bifurcations by angular proximity (greedy nearest)."""
        matches = []
        used_curr = set()

        for ri, a_ref in enumerate(angles_ref):
            best_ci = -1
            best_dist = np.radians(20)  # Max 20° difference

            for ci, a_curr in enumerate(angles_curr):
                if ci in used_curr:
                    continue
                dist = abs((a_ref - a_curr + np.pi) % (2 * np.pi) - np.pi)
                if dist < best_dist:
                    best_dist = dist
                    best_ci = ci

            if best_ci >= 0:
                matches.append((ri, best_ci))
                used_curr.add(best_ci)

        return matches
