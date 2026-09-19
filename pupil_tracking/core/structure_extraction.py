# pupil_tracking/core/structure_extraction.py
"""Smart structure extraction helpers for the detection pipeline.

Provides ring-aware contour fitting, fit validation, and result
population utilities extracted from :class:`UnifiedDetector` during
the Phase-3 refactoring.

All functions accept explicit parameters instead of relying on
instance state, making them independently testable and reusable.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult, FitType
from pupil_tracking.core.deterministic_ring_detector import (
    RingDetectionResult,
    RingStatus,
)
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    EllipseParams,
    DetectionMethod,
    ANATOMICAL_LIMITS,
    assign_quality_grade,
)

logger = logging.getLogger(__name__)


def fit_result_to_ellipse_params(fit: FitResult) -> EllipseParams:
    """Convert a SmartContourFitter ``FitResult`` into an
    ``EllipseParams`` used throughout the rest of the pipeline.
    """
    if fit.fit_type == FitType.CIRCLE:
        return EllipseParams(
            center_x=fit.center_x,
            center_y=fit.center_y,
            semi_major=fit.radius,
            semi_minor=fit.radius,
            angle_deg=0.0,
        )
    return EllipseParams(
        center_x=fit.center_x,
        center_y=fit.center_y,
        semi_major=fit.semi_major,
        semi_minor=fit.semi_minor,
        angle_deg=fit.angle_deg,
    )


def fit_result_confidence(fit: FitResult) -> float:
    """Derive a [0, 1] confidence from a FitResult.

    Combines the fitter's own quality score with a small
    penalty if the fit chose ellipse (slightly less constrained
    than a circle, so marginally more room for over-fitting).
    """
    base = fit.fit_quality if fit.fit_quality is not None else 0.5

    if fit.fit_type == FitType.CIRCLE:
        base = min(1.0, base * 1.05)

    return float(np.clip(base, 0.0, 1.0))


def is_inside_ring(
    cx: float,
    cy: float,
    radius: float,
    ring_result: RingDetectionResult,
    allow_partial: bool = False,
) -> bool:
    """Check if a circle (cx, cy, radius) is inside the ring opening."""
    if ring_result.ring_center is None or ring_result.ring_radius is None:
        return True  # No constraint

    ring_cx, ring_cy = ring_result.ring_center
    ring_r = ring_result.ring_radius

    dist = math.sqrt((cx - ring_cx) ** 2 + (cy - ring_cy) ** 2)

    if allow_partial:
        return dist + radius <= ring_r * 1.1
    else:
        return dist <= ring_r * 0.80


def apply_ring_roi(
    binary_mask: np.ndarray,
    ring_result: RingDetectionResult,
    margin_frac: float = 0.85,
) -> np.ndarray:
    """Zero out pixels outside the ring opening."""
    if ring_result.ring_center is None or ring_result.ring_radius is None:
        return binary_mask

    h, w = binary_mask.shape[:2]
    cx = int(ring_result.ring_center[0])
    cy = int(ring_result.ring_center[1])
    r = int(ring_result.ring_radius * margin_frac)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), max(1, r), 1, -1)

    return cv2.bitwise_and(binary_mask, roi_mask)


def apply_fit_to_result(
    result: EyeDetectionResult,
    pupil_fit: Optional[FitResult],
    limbus_fit: Optional[FitResult],
    force_limbus_overwrite: bool = False,
) -> None:
    """Overwrite pupil/limbus in *result* with SmartFitter output
    when the new fit is valid and at least as confident.
    """
    if pupil_fit is not None and pupil_fit.valid:
        ep = fit_result_to_ellipse_params(pupil_fit)
        new_conf = fit_result_confidence(pupil_fit)

        if (not result.pupil.detected) or new_conf >= result.pupil.confidence:
            result.pupil.detected = True
            result.pupil.ellipse = ep
            result.pupil.confidence = new_conf
            result.pupil.quality = assign_quality_grade(new_conf)
            result.pupil.method = DetectionMethod.ML
            result.pupil.fit_type = pupil_fit.fit_type.value
            if pupil_fit.contour_points is not None:
                result.pupil.contour_points = pupil_fit.contour_points

    if limbus_fit is not None and limbus_fit.valid:
        ep = fit_result_to_ellipse_params(limbus_fit)
        new_conf = fit_result_confidence(limbus_fit)

        if (
            not result.limbus.detected
            or force_limbus_overwrite
            or new_conf >= result.limbus.confidence
        ):
            result.limbus.detected = True
            result.limbus.ellipse = ep
            result.limbus.confidence = new_conf
            result.limbus.quality = assign_quality_grade(new_conf)
            result.limbus.method = DetectionMethod.ML
            result.limbus.fit_type = limbus_fit.fit_type.value
            if limbus_fit.contour_points is not None:
                result.limbus.contour_points = limbus_fit.contour_points


def extract_structure(
    mask: np.ndarray,
    fitter: SmartContourFitter,
    gray_image: Optional[np.ndarray] = None,
    ring_result: Optional[RingDetectionResult] = None,
) -> Tuple[Optional[FitResult], Optional[FitResult]]:
    """Extract pupil and limbus geometry from a segmentation mask
    using the SmartContourFitter (auto circle-vs-ellipse).

    When a ring is detected, applies spatial constraints to
    ensure the pupil and limbus fits lie inside the ring opening.

    Parameters
    ----------
    mask : np.ndarray
        Integer label mask where 1=pupil, 2=iris, 3=ring (optional).
    fitter : SmartContourFitter
        The fitter instance to use for contour fitting.
    gray_image : np.ndarray, optional
        Grayscale image for sub-pixel refinement.
    ring_result : RingDetectionResult, optional
        Ring detection result for spatial constraints.

    Returns
    -------
    (pupil_fit, limbus_fit) : tuple of optional FitResult
    """
    is_docked = ring_result is not None and ring_result.status in (
        RingStatus.PRESENT,
        RingStatus.PARTIAL,
    )

    # --- Pupil (class 1) ---
    pupil_mask = (mask == 1).astype(np.uint8)

    if is_docked and ring_result.ring_center is not None:
        pupil_mask = apply_ring_roi(
            pupil_mask,
            ring_result,
            margin_frac=0.85,
        )

    pupil_fit = fitter.fit(pupil_mask, gray_image)

    if (
        pupil_fit is not None
        and pupil_fit.valid
        and is_docked
        and ring_result.ring_center is not None
        and ring_result.ring_radius is not None
    ):
        if not is_inside_ring(
            pupil_fit.center_x,
            pupil_fit.center_y,
            pupil_fit.radius,
            ring_result,
        ):
            logger.debug("Pupil fit rejected: outside ring opening")
            pupil_fit = None

    # --- Iris / Limbus (class 2; union with pupil) ---
    iris_mask = ((mask == 2) | (mask == 1)).astype(np.uint8)

    if is_docked and ring_result.ring_center is not None:
        iris_mask = apply_ring_roi(
            iris_mask,
            ring_result,
            margin_frac=0.95,
        )
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        iris_mask = cv2.erode(iris_mask, kernel, iterations=1)

    limbus_fit = fitter.fit(
        iris_mask,
        gray_image,
        pupil_hint=pupil_fit,
    )

    # Validate pre-docking limbus concentricity and radius ratio
    if (
        not is_docked
        and pupil_fit is not None
        and pupil_fit.valid
        and limbus_fit is not None
        and limbus_fit.valid
    ):
        dx = pupil_fit.center_x - limbus_fit.center_x
        dy = pupil_fit.center_y - limbus_fit.center_y
        dist = math.hypot(dx, dy)
        if limbus_fit.radius > 0:
            offset_ratio = dist / limbus_fit.radius
            if offset_ratio > ANATOMICAL_LIMITS.MAX_CENTER_OFFSET_RATIO:
                logger.debug(
                    "Pre-docking limbus fit rejected: center offset "
                    f"{offset_ratio:.2f} > {ANATOMICAL_LIMITS.MAX_CENTER_OFFSET_RATIO}"
                )
                limbus_fit = None

        if limbus_fit is not None and limbus_fit.radius > 0:
            ratio = pupil_fit.radius / limbus_fit.radius
            if (
                ratio < ANATOMICAL_LIMITS.MIN_PUPIL_LIMBUS_RATIO
                or ratio > ANATOMICAL_LIMITS.MAX_PUPIL_LIMBUS_RATIO
            ):
                logger.debug(
                    "Pre-docking limbus fit rejected: radius ratio "
                    f"{ratio:.2f} out of bounds"
                )
                limbus_fit = None

    # Validate limbus is inside ring
    if (
        limbus_fit is not None
        and limbus_fit.valid
        and is_docked
        and ring_result.ring_center is not None
        and ring_result.ring_radius is not None
    ):
        if not is_inside_ring(
            limbus_fit.center_x,
            limbus_fit.center_y,
            limbus_fit.radius,
            ring_result,
            allow_partial=True,
        ):
            logger.debug("Limbus fit rejected: extends outside ring")
            limbus_fit = None

    return pupil_fit, limbus_fit
