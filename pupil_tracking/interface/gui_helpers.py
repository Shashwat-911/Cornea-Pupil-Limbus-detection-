# pupil_tracking/interface/gui_helpers.py
"""Static helper functions and measurement capture rendering
extracted from :class:`PupilTrackingGUI` during the Phase-4
refactoring.

These are pure utility functions that don't depend on GUI state
(except through explicit parameters), making them independently
testable and reusable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from types import SimpleNamespace

import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────

_CIRCLE_DRAW_THRESHOLD: float = 0.95

_QUALITY_COLORS: Dict[str, str] = {
    "SURGICAL": "#00e676",
    "CLINICAL": "#2196f3",
    "RESEARCH": "#ff9800",
    "INSUFFICIENT": "#ff5722",
    "NO_DETECTION": "#9e9e9e",
    "INTERPOLATED": "#9c27b0",
}

_CAPTURE_GLYPH_MAP: Dict[str, str] = {
    "\u00b0": " deg",
    "\u00d7": "x",
    "\u2014": "-",
    "\u2013": "-",
    "\u2192": "->",
    "\u00b5": "u",
    "\u2265": ">=",
    "\u2264": "<=",
    "\u00b1": "+/-",
    "\u2022": "*",
    "\u26a0": "!",
    "\u26a1": "",
    "\u2713": "OK",
    "\u2717": "X",
}


# ── Static helpers ─────────────────────────────────────────────────


def hex_to_bgr(value: str) -> Tuple[int, int, int]:
    """Convert a hex colour string (``#RRGGBB``) to a BGR tuple for OpenCV."""
    value = value.lstrip("#")
    if len(value) != 6:
        return (200, 200, 200)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def ascii_for_capture(text: str) -> str:
    """Sanitize a UI string so ``cv2.putText`` renders it without
    ``?`` glyphs.

    Replaces known symbols with ASCII equivalents, then drops any
    remaining non-ASCII codepoint.
    """
    if not text:
        return text
    for uni, ascii_rep in _CAPTURE_GLYPH_MAP.items():
        if uni in text:
            text = text.replace(uni, ascii_rep)
    if any(ord(ch) > 127 for ch in text):
        text = text.encode("ascii", "ignore").decode("ascii")
    return text


def scale_ellipse(e: Any, scale: float) -> Any:
    """Return ellipse namespace with coordinates scaled for display."""
    return SimpleNamespace(
        center_x=e.center_x * scale,
        center_y=e.center_y * scale,
        radius=e.radius * scale,
        semi_major=e.semi_major * scale,
        semi_minor=e.semi_minor * scale,
        angle_deg=e.angle_deg,
    )


def draw_structure(
    out: np.ndarray,
    ellipse_data: Any,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> Tuple[int, int]:
    """Draw an ellipse (or circle if near-circular) on the output image."""
    ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
    if ellipse_data.semi_major > 0:
        ratio = ellipse_data.semi_minor / ellipse_data.semi_major
    else:
        ratio = 1.0
    if ratio > _CIRCLE_DRAW_THRESHOLD:
        r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
        cv2.circle(out, ct, r, color, thickness, cv2.LINE_AA)
    else:
        axes = (
            int(round(ellipse_data.semi_major)),
            int(round(ellipse_data.semi_minor)),
        )
        angle = int(round(ellipse_data.angle_deg))
        cv2.ellipse(out, ct, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)
    return ct


# ── Measurement capture rendering ──────────────────────────────────


def measurement_capture_sections(
    colors: Any,
    pv: Dict[str, Any],
    lv: Dict[str, Any],
    ov: Dict[str, Any],
    cv_vars: Dict[str, Any],
    proc_time_var: Any,
    latency_var: Any,
    latency_avg_var: Any,
    drop_var: Any,
    tracking_state_var: Any,
    fps_var: Any,
    frame_var: Any,
    image_size_var: Any,
    pipeline_var: Any,
    gray_mode_var_display: Any,
) -> List[Tuple[str, Tuple[int, int, int], List[Tuple[str, str]]]]:
    """Build structured list of measurement sections for rendered capture panel."""
    return [
        (
            "PUPIL",
            hex_to_bgr(colors.PUPIL),
            [
                ("Center", pv["center"].get()),
                ("Diameter (px)", pv["diameter_px"].get()),
                ("Diameter (mm)", pv["diameter_mm"].get()),
                ("Semi-Major (px)", pv["semi_major"].get()),
                ("Semi-Major (mm)", pv["semi_major_mm"].get()),
                ("Semi-Minor (px)", pv["semi_minor"].get()),
                ("Semi-Minor (mm)", pv["semi_minor_mm"].get()),
                ("Angle", pv["angle"].get()),
                ("Fit Type", pv["fit_type"].get()),
                ("Confidence", pv["confidence"].get()),
                ("Quality", pv["quality"].get()),
            ],
        ),
        (
            "LIMBUS",
            hex_to_bgr(colors.LIMBUS),
            [
                ("Center", lv["center"].get()),
                ("Diameter (px)", lv["diameter_px"].get()),
                ("Diameter (mm)", lv["diameter_mm"].get()),
                ("Semi-Major (px)", lv["semi_major"].get()),
                ("Semi-Major (mm)", lv["semi_major_mm"].get()),
                ("Semi-Minor (px)", lv["semi_minor"].get()),
                ("Semi-Minor (mm)", lv["semi_minor_mm"].get()),
                ("Angle", lv["angle"].get()),
                ("Fit Type", lv["fit_type"].get()),
                ("Confidence", lv["confidence"].get()),
                ("Quality", lv["quality"].get()),
            ],
        ),
        (
            "CORNEAL OFFSET",
            hex_to_bgr(colors.OFFSET),
            [
                ("Corneal Centre", ov["corneal_center"].get()),
                ("Offset (px)", ov["offset_px"].get()),
                ("Offset (mm)", ov["offset_mm"].get()),
                ("Offset dX,dY px", ov["offset_vec_px"].get()),
                ("Offset dX,dY mm", ov["offset_vec_mm"].get()),
                ("Offset Angle", ov["offset_angle"].get()),
                ("Pupil/Limbus", ov["pupil_limbus_ratio"].get()),
            ],
        ),
        (
            "CALIBRATION",
            hex_to_bgr(colors.CALIBRATION),
            [
                ("Source", cv_vars["source"].get()),
                ("px/mm", cv_vars["scale_px"].get()),
                ("mm/px", cv_vars["scale_mm"].get()),
                ("Reference", cv_vars["reference"].get()),
            ],
        ),
        (
            "PROCESSING",
            hex_to_bgr(colors.PROCESSING),
            [
                ("Proc. Time", proc_time_var.get()),
                ("Latency", latency_var.get()),
                ("Latency Avg", latency_avg_var.get()),
                ("Dropped/Stale", drop_var.get()),
                ("Tracking", tracking_state_var.get()),
                ("FPS", fps_var.get()),
                ("Frame", frame_var.get()),
                ("Image Size", image_size_var.get()),
                ("Pipeline", pipeline_var.get()),
                ("Grayscale", gray_mode_var_display.get()),
            ],
        ),
    ]


def render_measurements_capture(
    height: int,
    width: int,
    colors: Any,
    summary_quality_var: Any,
    summary_tracking_var: Any,
    summary_latency_var: Any,
    summary_pipeline_var: Any,
    sections: List[Tuple[str, Tuple[int, int, int], List[Tuple[str, str]]]],
) -> np.ndarray:
    """Render the full measurements panel as an OpenCV numpy array."""
    panel = np.full((height, width, 3), hex_to_bgr(colors.BG_SECONDARY), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), hex_to_bgr(colors.BORDER), 1)

    pad = max(10, height // 72)
    gutter = max(10, width // 42)
    inner_w = width - pad * 2
    col_w = max(220, (inner_w - gutter) // 2)
    title_font = max(0.42, min(0.72, height / 900.0))
    body_font = max(0.34, min(0.54, height / 1080.0))
    line_h = max(16, int(height / 36))
    row_gap = max(3, int(line_h * 0.2))
    section_gap = max(8, int(line_h * 0.55))
    summary_box_h = max(62, int(height * 0.09))
    summary_gap = max(8, gutter // 2)
    summary_w = max(120, (inner_w - summary_gap) // 2)

    fg_primary = hex_to_bgr(colors.FG_PRIMARY)
    fg_secondary = hex_to_bgr(colors.FG_SECONDARY)
    card_bg = hex_to_bgr(colors.BG_TERTIARY)
    quality_color = hex_to_bgr(
        _QUALITY_COLORS.get(summary_quality_var.get(), colors.FG_PRIMARY)
    )
    tracking_color = hex_to_bgr(
        {
            "Tracking Stable": colors.SURGICAL,
            "Tracking Acquiring": colors.CLINICAL,
            "Tracking Degraded": colors.RESEARCH,
            "No Detection": colors.INSUFFICIENT,
            "Ready": colors.ACCENT,
            "Waiting": colors.FG_SECONDARY,
        }.get(summary_tracking_var.get(), colors.FG_PRIMARY)
    )

    summaries = [
        ("QUALITY", summary_quality_var.get(), quality_color),
        ("TRACKING", summary_tracking_var.get(), tracking_color),
        ("LATENCY", summary_latency_var.get(), fg_primary),
        ("PIPELINE", summary_pipeline_var.get(), fg_primary),
    ]
    for idx, (label, value, color) in enumerate(summaries):
        row = idx // 2
        col = idx % 2
        x0 = pad + col * (summary_w + summary_gap)
        y0 = pad + row * (summary_box_h + summary_gap)
        x1 = min(width - pad, x0 + summary_w)
        cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), card_bg, -1)
        cv2.rectangle(panel, (x0, y0), (x1, y0 + summary_box_h), hex_to_bgr(colors.BORDER), 1)
        cv2.putText(panel, ascii_for_capture(label), (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg_secondary, 1, cv2.LINE_AA)
        cv2.putText(panel, ascii_for_capture(value or "---"), (x0 + 10, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    left_sections = sections[:2]
    right_sections = sections[2:]
    start_y = pad * 2 + summary_box_h * 2 + summary_gap

    def draw_section_column(items, x_start):
        y = start_y
        for title, accent, rows in items:
            box_h = max(60, 34 + len(rows) * (line_h + row_gap))
            if y + box_h > height - pad:
                box_h = max(40, height - pad - y)
            cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), card_bg, -1)
            cv2.rectangle(panel, (x_start, y), (x_start + col_w, min(height - pad, y + box_h)), hex_to_bgr(colors.BORDER), 1)
            cv2.putText(panel, ascii_for_capture(title), (x_start + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, title_font, accent, 2, cv2.LINE_AA)
            row_y = y + 48
            for label, value in rows:
                if row_y > y + box_h - 8:
                    break
                clean_label = ascii_for_capture(label.replace("_", " ").title())
                clean_value = ascii_for_capture((value or "---").replace("\n", " "))
                if len(clean_value) > 32:
                    clean_value = clean_value[:29] + "..."
                cv2.putText(panel, clean_label, (x_start + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_secondary, 1, cv2.LINE_AA)
                text_size = cv2.getTextSize(clean_value, cv2.FONT_HERSHEY_SIMPLEX, body_font, 1)[0]
                value_x = max(x_start + 140, x_start + col_w - 10 - text_size[0])
                cv2.putText(panel, clean_value, (value_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, body_font, fg_primary, 1, cv2.LINE_AA)
                row_y += line_h + row_gap
            y += box_h + section_gap

    draw_section_column(left_sections, pad)
    draw_section_column(right_sections, pad + col_w + gutter)
    return panel


def compose_capture_frame(
    image: np.ndarray,
    result: Any,
    colors: Any,
    summary_quality_var: Any,
    summary_tracking_var: Any,
    summary_latency_var: Any,
    summary_pipeline_var: Any,
    pv: Dict[str, Any],
    lv: Dict[str, Any],
    ov: Dict[str, Any],
    cv_vars: Dict[str, Any],
    proc_time_var: Any,
    latency_var: Any,
    latency_avg_var: Any,
    drop_var: Any,
    tracking_state_var: Any,
    fps_var: Any,
    frame_var: Any,
    image_size_var: Any,
    pipeline_var: Any,
    gray_mode_var_display: Any,
) -> np.ndarray:
    """Concatenate image + divider + measurement panel into composite frame."""
    img_h, img_w = image.shape[:2]
    panel_w = max(700, int(img_w * 0.62))
    sections = measurement_capture_sections(
        colors, pv, lv, ov, cv_vars,
        proc_time_var, latency_var, latency_avg_var, drop_var,
        tracking_state_var, fps_var, frame_var, image_size_var,
        pipeline_var, gray_mode_var_display,
    )
    panel = render_measurements_capture(
        img_h, panel_w, colors,
        summary_quality_var, summary_tracking_var,
        summary_latency_var, summary_pipeline_var,
        sections,
    )
    divider = np.full((img_h, 3, 3), hex_to_bgr(colors.BORDER), dtype=np.uint8)
    return np.concatenate([image, divider, panel], axis=1)
