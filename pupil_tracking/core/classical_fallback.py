# pupil_tracking/core/classical_fallback.py
"""Classical CV fallback detection for pupil and limbus.

Provides ring-aware classical detection using adaptive thresholding,
edge detection, and Hough transforms, refined by SmartContourFitter.
Extracted from :class:`UnifiedDetector` during the Phase-3 refactoring.

These are free functions that accept explicit parameters (fitter,
ring result, logger) instead of relying on instance state.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np

from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult
from pupil_tracking.core.structure_extraction import (
    fit_result_to_ellipse_params,
    is_inside_ring,
)
from pupil_tracking.core.deterministic_ring_detector import (
    RingDetectionResult,
    RingStatus,
)
from pupil_tracking.utils.types import (
    PupilDetection,
    LimbusDetection,
    EllipseParams,
    DetectionMethod,
    assign_quality_grade,
)

logger = logging.getLogger(__name__)


def classical_pupil_detection(
    image: np.ndarray,
    fitter: SmartContourFitter,
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

        max_radius = min(max_radius, int(ring_result.ring_radius * 0.5))

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    best_fit: Optional[FitResult] = None
    best_score = 0.0
    best_contour = None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for pct in [3, 5, 8, 12, 18, 25, 35]:
        thresh_val = np.percentile(blurred, pct)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

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
            fit = fitter.fit(cnt_mask, gray)

            if fit is None or not fit.valid:
                continue
            if fit.radius < min_radius or fit.radius > max_radius:
                continue

            if (
                is_docked
                and ring_result.ring_center is not None
                and ring_result.ring_radius is not None
            ):
                if not is_inside_ring(
                    fit.center_x,
                    fit.center_y,
                    fit.radius,
                    ring_result,
                ):
                    continue

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
        detection.ellipse = fit_result_to_ellipse_params(best_fit)
        detection.confidence = float(np.clip(best_score * 0.85, 0.0, 1.0))
        detection.quality = assign_quality_grade(detection.confidence)
        detection.contour_points = best_contour
        detection.fit_type = best_fit.fit_type.value

    return detection


def classical_limbus_detection(
    image: np.ndarray,
    fitter: SmartContourFitter,
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

        fit = fitter.fit(edge_mask, gray)
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
        detection.ellipse = fit_result_to_ellipse_params(best_fit)
        detection.confidence = float(np.clip(best_score * 0.80, 0.0, 1.0))
        detection.quality = assign_quality_grade(detection.confidence)
        detection.fit_type = best_fit.fit_type.value

    return detection
