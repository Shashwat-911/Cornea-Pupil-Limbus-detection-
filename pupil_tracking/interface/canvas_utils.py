"""Pure geometry and state helpers for canvas/event handling.

Extracted from gui_app.py.  These functions have NO GUI dependencies
and NO clinical logic — they operate on plain data (coordinates, ROIs,
image shapes).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def canvas_to_image_point(
    canvas_x: float,
    canvas_y: float,
    display_origin: Tuple[int, int],
    display_size: Tuple[int, int],
    display_scale: float,
    image_shape: Tuple[int, ...],
) -> Optional[Tuple[float, float]]:
    """Convert canvas pixel coordinates to original image coordinates.

    Returns ``(x, y)`` in image space, or ``None`` if the point is
    outside the displayed image region.
    """
    ox, oy = display_origin
    dw, dh = display_size
    if dw <= 0 or dh <= 0:
        return None
    if not (ox <= canvas_x <= ox + dw and oy <= canvas_y <= oy + dh):
        return None
    x = (canvas_x - ox) / max(display_scale, 1e-6)
    y = (canvas_y - oy) / max(display_scale, 1e-6)
    h, w = image_shape[:2]
    return (float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1)))


def active_manual_roi(
    manual_roi: Optional[Dict[str, float]],
    image_shape: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, float]]:
    """Return *manual_roi* if it matches the current image dimensions."""
    if manual_roi is None or image_shape is None:
        return None
    h, w = image_shape[:2]
    if (
        int(round(manual_roi.get("frame_width", w))) != w
        or int(round(manual_roi.get("frame_height", h))) != h
    ):
        return None
    return manual_roi


def active_manual_ring(
    manual_ring: Optional[Dict[str, float]],
    image_shape: Optional[Tuple[int, ...]],
) -> Optional[Dict[str, float]]:
    """Return *manual_ring* if it matches the current image dimensions."""
    if manual_ring is None or image_shape is None:
        return None
    h, w = image_shape[:2]
    if (
        int(round(manual_ring.get("frame_width", w))) != w
        or int(round(manual_ring.get("frame_height", h))) != h
    ):
        return None
    return manual_ring


def manual_roi_crop(
    frame: np.ndarray,
    roi: Dict[str, float],
) -> Optional[Tuple[np.ndarray, float, float]]:
    """Crop *frame* to the bounding box of a circular ROI.

    Returns ``(cropped, x_offset, y_offset)`` or ``None`` if the crop
    is empty.
    """
    h, w = frame.shape[:2]
    cx = float(np.clip(roi["center_x"], 0, w - 1))
    cy = float(np.clip(roi["center_y"], 0, h - 1))
    radius = max(1.0, float(roi["radius"]))
    x0 = max(0, int(np.floor(cx - radius)))
    y0 = max(0, int(np.floor(cy - radius)))
    x1 = min(w, int(np.ceil(cx + radius)))
    y1 = min(h, int(np.ceil(cy + radius)))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop, float(x0), float(y0)


def clamp_circle_center(
    cx: float,
    cy: float,
    radius: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float]:
    """Clamp a circle centre so the full circle stays inside the image."""
    return (
        float(np.clip(cx, radius, image_w - radius)),
        float(np.clip(cy, radius, image_h - radius)),
    )


def clamp_circle_radius(
    cx: float,
    cy: float,
    radius: float,
    image_w: int,
    image_h: int,
    min_radius: float = 8.0,
) -> float:
    """Clamp a circle radius so it stays inside the image bounds."""
    max_radius = min(cx, cy, image_w - cx, image_h - cy)
    return float(max(min_radius, min(radius, max_radius)))


def hit_test_circle(
    point_x: float,
    point_y: float,
    circle_cx: float,
    circle_cy: float,
    circle_radius: float,
    rim_fraction: float = 0.18,
) -> str:
    """Classify a click relative to a circle.

    Returns ``"move"``, ``"resize"``, or ``"outside"``.
    """
    import math
    distance = math.hypot(point_x - circle_cx, point_y - circle_cy)
    rim_threshold = max(10.0, circle_radius * rim_fraction)
    if abs(distance - circle_radius) <= rim_threshold:
        return "resize"
    elif distance < circle_radius:
        return "move"
    return "outside"
