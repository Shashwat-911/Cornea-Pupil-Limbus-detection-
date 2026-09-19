"""
Stream C: Surgical ink marker tracking for cyclotorsion detection.

Detects coloured ink marks placed on the sclera/limbus by the surgeon
before the patient sits up. By comparing ink mark positions between
pre-op and intra-op images, cyclotorsion can be directly measured.

Ink marks are typically purple/blue (gentian violet) or green.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class InkTrackerStream(BaseStream):
    """Track surgical ink markers for rotation estimation.

    Algorithm:
        1. Convert to HSV colour space
        2. Threshold for ink colour (purple/blue range)
        3. Find contours, filter by size and position (near limbus)
        4. Match markers between images by angular position
        5. Compute rotation from matched marker pairs
    """

    def __init__(self):
        super().__init__(StreamName.INK_MARKERS)
        cfg = get_config().registration
        self.min_markers = cfg.ink_min_markers
        self.hsv_lower = np.array(cfg.ink_hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(cfg.ink_hsv_upper, dtype=np.uint8)

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Detect ink marks and compute rotation from their displacement."""

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        # Detect markers
        markers_ref = self._detect_ink_markers(img_ref, detection_ref)
        markers_curr = self._detect_ink_markers(img_curr, detection_curr)

        if len(markers_ref) < self.min_markers:
            return self._make_result(
                metadata={"error": "insufficient_ref_markers",
                           "markers_found": len(markers_ref)}
            )
        if len(markers_curr) < self.min_markers:
            return self._make_result(
                metadata={"error": "insufficient_curr_markers",
                           "markers_found": len(markers_curr)}
            )

        # Compute angles for all markers relative to limbus centres
        le_ref = detection_ref.limbus.ellipse
        le_curr = detection_curr.limbus.ellipse

        angles_ref = np.array([
            np.arctan2(my - le_ref.center_y, mx - le_ref.center_x)
            for mx, my, _ in markers_ref
        ])
        angles_curr = np.array([
            np.arctan2(my - le_curr.center_y, mx - le_curr.center_x)
            for mx, my, _ in markers_curr
        ])

        # Match markers by angular proximity
        matches = self._match_markers(angles_ref, angles_curr)

        if len(matches) < self.min_markers:
            return self._make_result(
                metadata={"error": "insufficient_matches",
                           "matched": len(matches)}
            )

        # Compute angular differences for matched pairs
        diffs = []
        for ri, ci in matches:
            diff = (angles_curr[ci] - angles_ref[ri] + np.pi) % (2 * np.pi) - np.pi
            diffs.append(diff)
        diffs = np.array(diffs)

        # Robust estimate
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        inliers = np.abs(diffs - median_diff) < 2.5 * max(mad, 0.01)

        if np.sum(inliers) < self.min_markers:
            return self._make_result(
                metadata={"error": "too_few_inliers"}
            )

        torsion_rad = float(np.mean(diffs[inliers]))
        torsion_deg = float(np.degrees(torsion_rad))

        # Confidence: high because ink markers are very reliable
        inlier_ratio = np.sum(inliers) / len(diffs)
        consistency = 1.0 / (1.0 + np.std(diffs[inliers]) * 10.0)
        confidence = float(np.clip(
            0.9 * inlier_ratio * consistency, 0.0, 1.0
        ))

        return self._make_result(
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=int(np.sum(inliers)),
            metadata={
                "markers_ref": len(markers_ref),
                "markers_curr": len(markers_curr),
                "matched": len(matches),
            },
        )

    def _detect_ink_markers(
        self,
        image: np.ndarray,
        detection: EyeDetectionResult,
    ) -> List[Tuple[float, float, float]]:
        """Detect coloured ink markers near the limbus.

        Returns list of (x, y, radius) tuples.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Threshold for ink colour
        ink_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Also try a second range for purple wrapping around hue=180
        hsv_lower2 = np.array([0, 50, 50], dtype=np.uint8)
        hsv_upper2 = np.array([10, 255, 255], dtype=np.uint8)
        ink_mask2 = cv2.inRange(hsv, hsv_lower2, hsv_upper2)
        ink_mask = cv2.bitwise_or(ink_mask, ink_mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by size and proximity to limbus
        le = detection.limbus.ellipse
        limbus_center = np.array([le.center_x, le.center_y])
        limbus_radius = le.radius

        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 2000:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            dist = np.sqrt((cx - limbus_center[0])**2 + (cy - limbus_center[1])**2)

            if 0.7 * limbus_radius < dist < 1.3 * limbus_radius:
                radius = np.sqrt(area / np.pi)
                markers.append((cx, cy, radius))

        return markers

    def _match_markers(
        self,
        angles_ref: np.ndarray,
        angles_curr: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Match markers by angular proximity (greedy nearest).

        Returns list of (ref_idx, curr_idx) pairs.
        """
        matches = []
        used_curr = set()

        for ri, a_ref in enumerate(angles_ref):
            best_ci = -1
            best_dist = np.radians(30)  # Max 30° difference

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
