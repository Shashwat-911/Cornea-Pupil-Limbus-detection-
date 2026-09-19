"""
Visualization for cyclotorsion detection results.

Provides overlay rendering for registration diagnostics, including:
    - Polar unwrapped images
    - Per-stream torsion indicators
    - Fused result with confidence interval
    - Matched feature correspondences
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.types import RegistrationResult, StreamResult

logger = logging.getLogger(__name__)


# Colour palette for streams
STREAM_COLOURS = {
    "phase_correlation": (255, 165, 0),    # Orange
    "deep_matcher": (0, 255, 128),          # Green
    "ink_markers": (200, 100, 200),         # Purple
    "limbal_vessels": (255, 80, 80),        # Red
    "custom_feature": (80, 200, 255),       # Cyan
}


def draw_registration_overlay(
    image: np.ndarray,
    result: RegistrationResult,
    detection_ref=None,
    detection_curr=None,
    show_streams: bool = True,
    show_arc: bool = True,
    show_legend: bool = True,
) -> np.ndarray:
    """Draw cyclotorsion result overlay on an image.

    Parameters
    ----------
    image : np.ndarray
        Image to draw on (will be copied).
    result : RegistrationResult
        Fused registration result.
    detection_ref, detection_curr
        Optional detection results for drawing iris boundaries.
    show_streams : bool
        Show per-stream torsion indicators.
    show_arc : bool
        Draw arc showing fused torsion angle.
    show_legend : bool
        Draw colour legend.

    Returns
    -------
    np.ndarray
        Image with overlays.
    """
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    h, w = vis.shape[:2]

    if not result.valid:
        cv2.putText(
            vis, "Registration: NO RESULT",
            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2,
        )
        return vis

    # Determine centre for arc drawing
    if detection_curr and detection_curr.has_limbus:
        le = detection_curr.limbus.ellipse
        center = (int(le.center_x), int(le.center_y))
        radius = int(le.radius)
    else:
        center = (w // 2, h // 2)
        radius = min(w, h) // 4

    # Draw fused torsion arc
    if show_arc:
        _draw_torsion_arc(vis, center, radius, result.torsion_deg, result.confidence)

    # Draw per-stream indicators
    if show_streams:
        _draw_stream_indicators(vis, center, radius, result)

    # Draw info text
    _draw_info_text(vis, result, h, w)

    # Draw legend
    if show_legend:
        _draw_legend(vis, result)

    return vis


def _draw_torsion_arc(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    torsion_deg: float,
    confidence: float,
):
    """Draw an arc showing the fused torsion angle."""
    # Reference line (12 o'clock)
    ref_end = (center[0], center[1] - radius)
    cv2.line(image, center, ref_end, (255, 255, 255), 2)

    # Rotated line
    angle_rad = math.radians(-torsion_deg)  # Negative for CW rotation
    rot_end = (
        int(center[0] + radius * math.sin(angle_rad)),
        int(center[1] - radius * math.cos(angle_rad)),
    )
    # Colour based on confidence
    if confidence >= 0.85:
        colour = (0, 255, 0)   # Green — surgical
    elif confidence >= 0.60:
        colour = (0, 255, 255) # Yellow — clinical
    elif confidence >= 0.30:
        colour = (0, 165, 255) # Orange — research
    else:
        colour = (0, 0, 255)   # Red — insufficient

    cv2.line(image, center, rot_end, colour, 3)

    # Arc between reference and rotated
    arc_radius = int(radius * 0.6)
    start_angle = -90  # 12 o'clock
    end_angle = start_angle - torsion_deg
    cv2.ellipse(
        image, center, (arc_radius, arc_radius),
        0, min(start_angle, end_angle), max(start_angle, end_angle),
        colour, 2,
    )

    # Angle label
    label_pos = (
        int(center[0] + arc_radius * 0.7 * math.sin(angle_rad / 2)),
        int(center[1] - arc_radius * 0.7 * math.cos(angle_rad / 2)),
    )
    cv2.putText(
        image, f"{torsion_deg:+.2f}°",
        label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2,
    )


def _draw_stream_indicators(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    result: RegistrationResult,
):
    """Draw small indicators for each stream's torsion estimate."""
    indicator_radius = int(radius * 0.8)

    for name, sr in result.stream_results.items():
        if not sr.valid:
            continue

        colour = STREAM_COLOURS.get(name, (200, 200, 200))
        angle_rad = math.radians(-sr.torsion_deg)

        # Small dot at the stream's estimated angle
        dot_pos = (
            int(center[0] + indicator_radius * math.sin(angle_rad)),
            int(center[1] - indicator_radius * math.cos(angle_rad)),
        )
        cv2.circle(image, dot_pos, 5, colour, -1)
        cv2.circle(image, dot_pos, 5, (255, 255, 255), 1)


def _draw_info_text(
    image: np.ndarray,
    result: RegistrationResult,
    h: int,
    w: int,
):
    """Draw text info panel."""
    texts = [
        f"Torsion: {result.torsion_deg:+.3f} deg",
        f"Confidence: {result.confidence:.2f} ({result.quality.value})",
        f"Streams: {result.agreeing_streams}/{result.active_streams} agree",
        f"CI: [{result.confidence_interval_deg[0]:+.2f}, "
        f"{result.confidence_interval_deg[1]:+.2f}]",
        f"Time: {result.total_processing_time_ms:.0f}ms",
    ]

    y_start = 30
    for i, text in enumerate(texts):
        cv2.putText(
            image, text,
            (10, y_start + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        cv2.putText(
            image, text,
            (10, y_start + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
        )


def _draw_legend(
    image: np.ndarray,
    result: RegistrationResult,
):
    """Draw colour legend for streams."""
    h = image.shape[0]
    x_start = image.shape[1] - 200
    y_start = h - 30 * len(result.stream_results) - 10

    for i, (name, sr) in enumerate(result.stream_results.items()):
        colour = STREAM_COLOURS.get(name, (200, 200, 200))
        y = y_start + i * 25
        cv2.circle(image, (x_start, y), 5, colour, -1)

        status = f"{sr.torsion_deg:+.2f}°" if sr.valid else "N/A"
        cv2.putText(
            image, f"{name}: {status}",
            (x_start + 15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )


def create_comparison_view(
    img_ref: np.ndarray,
    img_curr: np.ndarray,
    result: RegistrationResult,
) -> np.ndarray:
    """Create a side-by-side comparison view.

    Shows reference (left) and current (right) images with
    registration result overlaid on the current image.
    """
    if len(img_ref.shape) == 2:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)
    if len(img_curr.shape) == 2:
        img_curr = cv2.cvtColor(img_curr, cv2.COLOR_GRAY2BGR)

    # Resize to same height
    h = max(img_ref.shape[0], img_curr.shape[0])
    w_ref = int(img_ref.shape[1] * h / img_ref.shape[0])
    w_curr = int(img_curr.shape[1] * h / img_curr.shape[0])

    ref_resized = cv2.resize(img_ref, (w_ref, h))
    curr_resized = cv2.resize(img_curr, (w_curr, h))

    # Add labels
    cv2.putText(ref_resized, "REFERENCE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    curr_with_overlay = draw_registration_overlay(curr_resized, result)
    cv2.putText(curr_with_overlay, "CURRENT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Concatenate side by side
    combined = np.hstack([ref_resized, curr_with_overlay])
    return combined
