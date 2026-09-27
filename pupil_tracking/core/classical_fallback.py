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
    for pct in [3, 5, 8, 12, 18, 25]:
        thresh_val = np.percentile(blurred[::2, ::2], pct)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        if ring_roi_mask is not None:
            binary = cv2.bitwise_and(binary, ring_roi_mask)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or len(cnt) < 5:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            circ_geom = 4.0 * math.pi * area / (peri * peri)
            if circ_geom < 0.30:
                continue

            (cx_c, cy_c), r_c = cv2.minEnclosingCircle(cnt)
            if r_c < min_radius or r_c > max_radius:
                continue

            if (
                is_docked
                and ring_result.ring_center is not None
                and ring_result.ring_radius is not None
            ):
                if not is_inside_ring(
                    cx_c,
                    cy_c,
                    r_c,
                    ring_result,
                ):
                    continue
                ring_cx, ring_cy = ring_result.ring_center
                ring_r = ring_result.ring_radius or 1.0
                dist = math.hypot(cx_c - ring_cx, cy_c - ring_cy)
                centrality = max(0.0, 1.0 - dist / ring_r)
            else:
                centrality = max(
                    0.0,
                    1.0
                    - (
                        abs(cx_c - w / 2) / (w / 2) * 0.5
                        + abs(cy_c - h / 2) / (h / 2) * 0.5
                    ),
                )

            bx, by, bw, bh = cv2.boundingRect(cnt)
            mean_val = float(np.mean(gray[by:by + bh, bx:bx + bw]))
            darkness = 1.0 - (mean_val / 255.0)

            pre_score = (
                0.35 * centrality
                + 0.35 * min(1.0, circ_geom / 0.7)
                + 0.30 * darkness
            )
            candidates.append((pre_score, cnt, circ_geom, centrality, darkness))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        for pre_score, cnt, circ_geom, centrality, darkness in candidates[:2]:
            cnt_mask = np.zeros_like(gray)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, -1)
            fit = fitter.fit(cnt_mask, gray)

            if fit is None or not fit.valid:
                continue
            if fit.radius < min_radius or fit.radius > max_radius:
                continue

            circ = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else circ_geom
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

        if best_score > 0.82:
            break

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
    refined by SmartContourFitter with vectorized edge sampling.

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

    candidate_circles = []
    for cx, cy, r in all_circles:
        if r < min_radius or r > max_radius:
            continue

        if pupil_hint is not None and pupil_hint.is_valid:
            d = math.hypot(cx - pupil_hint.center_x, cy - pupil_hint.center_y)
            if d > r * 0.35:
                continue
            radius_ratio = r / max(pupil_hint.radius, 1.0)
            if radius_ratio < 1.6 or radius_ratio > 5.5:
                continue

        if (
            is_docked
            and ring_result.ring_center is not None
            and ring_result.ring_radius is not None
        ):
            ring_cx, ring_cy = ring_result.ring_center
            ring_r = ring_result.ring_radius
            dist_to_ring = math.hypot(cx - ring_cx, cy - ring_cy)
            if dist_to_ring + r > ring_r * 1.05:
                continue

        if pupil_hint is not None and pupil_hint.is_valid:
            priority = -math.hypot(cx - pupil_hint.center_x, cy - pupil_hint.center_y)
        else:
            priority = -(abs(cx - w / 2) + abs(cy - h / 2))
        candidate_circles.append((priority, cx, cy, r))

    if not candidate_circles:
        return detection

    candidate_circles.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidate_circles[:3]

    best_fit: Optional[FitResult] = None
    best_score = 0.0

    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    dr_vals = np.arange(-10, 11)

    for _, cx, cy, r in top_candidates:
        r_mat = r + dr_vals[:, None]
        px_mat = np.clip(np.round(cx + r_mat * cos_a[None, :]).astype(np.int32), 0, w - 1)
        py_mat = np.clip(np.round(cy + r_mat * sin_a[None, :]).astype(np.int32), 0, h - 1)

        edge_hits = edges[py_mat, px_mat] > 0
        ray_has_hit = np.any(edge_hits, axis=0)

        if np.sum(ray_has_hit) < 15:
            continue

        first_hit_idx = np.argmax(edge_hits, axis=0)
        valid_ray_indices = np.where(ray_has_hit)[0]

        edge_pts = [
            [int(px_mat[first_hit_idx[i], i]), int(py_mat[first_hit_idx[i], i])]
            for i in valid_ray_indices
        ]

        pts_arr = np.array(edge_pts, dtype=np.float64)
        if len(pts_arr) < 12:
            continue

        fit = fitter.fit_contour(pts_arr)
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
        coverage = min(1.0, len(edge_pts) / 60.0)
        fit_quality = fit.fit_quality if fit.fit_quality is not None else 0.5

        score = (
            0.25 * min(1.0, circ / 0.7)
            + 0.25 * fit_quality
            + 0.25 * centrality
            + 0.25 * coverage
        )

        if pupil_hint is not None and pupil_hint.is_valid:
            d = math.hypot(fit.center_x - pupil_hint.center_x, fit.center_y - pupil_hint.center_y)
            concentricity = max(0.0, 1.0 - d / max(r, 1))
            score = score * 0.7 + concentricity * 0.3

        if is_docked and ring_result.ring_center is not None:
            ring_cx, ring_cy = ring_result.ring_center
            d_ring = math.hypot(fit.center_x - ring_cx, fit.center_y - ring_cy)
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
