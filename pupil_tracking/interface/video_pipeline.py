"""Shared post-detection frame processing for video pipelines.

Extracted from gui_app.py to eliminate duplication across the classic
and optimised video loops.  Handles smoothing, corneal-centre
calculation, ring-status propagation, weighted ring-centre override,
and iris detection — all pure post-processing steps that do NOT
perform detection themselves.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def process_frame_post_detect(
    *,
    result: Any,
    frame: np.ndarray,
    tracker: Any = None,
    corneal_calc: Any = None,
    iris_detector: Any = None,
    log_context: str = "frame",
    logger: Any = None,
) -> Any:
    """Run shared post-detection steps on a raw detection result.

    Steps applied (in order):
      1. Kalman smoothing via *tracker* (if provided)
      2. Corneal-centre calculation via *corneal_calc* (if both pupil
         and limbus detected)
      3. Propagate ring attributes from *result* to the smoothed object
      4. Weighted ring-centre override when ring is present
      5. Iris detection via *iris_detector* (if available)

    Returns the (possibly smoothed) result with all post-processing
    attributes assigned.  Safe to call with any combination of ``None``
    components — missing components are simply skipped.
    """
    # ── 1. Kalman smoothing ────────────────────────────────────────
    if tracker is not None:
        smoothed = tracker.update(result)
    else:
        smoothed = result

    # ── 2. Corneal-centre calculation ──────────────────────────────
    if smoothed.has_both and corneal_calc is not None:
        smoothed.corneal_center = corneal_calc.calculate(
            smoothed.pupil,
            smoothed.limbus,
            result.calibration,
        )

    # ── 3. Propagate ring attributes ──────────────────────────────
    smoothed.calibration = result.calibration
    smoothed.ring_status = getattr(result, "ring_status", "unknown")
    smoothed.ring_center = getattr(result, "ring_center", None)
    smoothed.ring_radius = getattr(result, "ring_radius", None)
    smoothed.ring_contour = getattr(result, "ring_contour", None)
    smoothed.ring_dot_count = getattr(result, "ring_dot_count", 0)
    smoothed.corneal_reference_source = getattr(
        result, "corneal_reference_source", "limbus"
    )

    # ── 4. Weighted ring-centre override ───────────────────────────
    if (
        smoothed.ring_status == "ring_present"
        and smoothed.ring_center is not None
        and getattr(smoothed, "pupil", None) is not None
        and getattr(smoothed.pupil, "ellipse", None) is not None
    ):
        _apply_weighted_ring_center(smoothed, result)

    # ── 5. Iris detection ─────────────────────────────────────────
    _detect_iris(smoothed, frame, iris_detector, log_context, logger)

    return smoothed


def _apply_weighted_ring_center(smoothed: Any, result: Any) -> None:
    """Override corneal centre with weighted pupil + limbus + ring centroid."""
    px = smoothed.pupil.ellipse.center_x
    py = smoothed.pupil.ellipse.center_y
    points = [(px, py, "pupil")]
    weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]

    if (
        getattr(smoothed, "limbus", None) is not None
        and getattr(smoothed.limbus, "ellipse", None) is not None
    ):
        points.append(
            (
                smoothed.limbus.ellipse.center_x,
                smoothed.limbus.ellipse.center_y,
                "limbus",
            )
        )
        weights.append(
            max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3)
        )

    points.append((smoothed.ring_center[0], smoothed.ring_center[1], "ring"))
    weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))

    total_w = sum(weights)
    rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
    rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w

    smoothed.corneal_reference_source = "+".join(
        name for _, _, name in points
    )
    smoothed.corneal_center.center_px = (rcx, rcy)
    smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
    smoothed.corneal_center.offset_magnitude_px = math.hypot(px - rcx, py - rcy)
    smoothed.corneal_center.offset_angle_deg = math.degrees(
        math.atan2(py - rcy, px - rcx)
    )
    smoothed.corneal_center.valid = True

    if result.calibration.calibrated:
        smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm(
            (rcx, rcy)
        )
        dx_mm = (px - rcx) * result.calibration.mm_per_px
        dy_mm = (py - rcy) * result.calibration.mm_per_px
        smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
        smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)


def _detect_iris(
    result: Any,
    frame: np.ndarray,
    iris_detector: Any = None,
    log_context: str = "frame",
    logger: Any = None,
) -> None:
    """Run iris detection on a result that has both pupil and limbus.

    Assigns ``result.iris_detection`` and ``result.iris_status``.
    Safe to call when *iris_detector* is ``None``.
    """
    if (
        iris_detector is not None
        and result.has_both
        and getattr(result.pupil, "ellipse", None) is not None
        and getattr(result.limbus, "ellipse", None) is not None
    ):
        try:
            iris_result = iris_detector.detect(
                frame, result.pupil.ellipse, result.limbus.ellipse
            )
            result.iris_detection = iris_result
            result.iris_status = iris_result.status
        except Exception as exc:
            if logger is not None:
                logger.debug(
                    "Iris detection failed (%s): %s", log_context, exc
                )
            result.iris_detection = None
            result.iris_status = None
    else:
        result.iris_detection = None
        result.iris_status = None
