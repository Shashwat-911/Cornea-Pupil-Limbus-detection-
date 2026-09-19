"""Overlay drawing functions extracted from PupilTrackingGUI.

Pure rendering — no clinical computation.  Reads display state from the
GUI instance passed as ``gui``.  All coordinate math stays at the call
site or in ``gui_helpers``.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Tuple

import cv2
import numpy as np
from types import SimpleNamespace

from pupil_tracking.interface.gui_helpers import scale_ellipse, draw_structure

_CIRCLE_DRAW_THRESHOLD: float = 0.95

# Iris-feature overlay (additive, purely visual — never affects detection).
_IRIS_FEATURE_COLOR: Tuple[int, int, int] = (255, 80, 255)
_IRIS_LOG = logging.getLogger("iris.overlay")
_IRIS_LOG_INTERVAL_S = 2.0
_IRIS_LAST_LOG = [0.0]


# ── Pure geometry helpers ────────────────────────────────────────────


def get_ellipse_intersection(
    ellipse: Any, px: float, py: float, dx: float, dy: float
) -> Tuple[float, float]:
    """Ray–ellipse intersection (returns display-pixel point)."""
    cx = ellipse.center_x
    cy = ellipse.center_y
    a = max(1.0, ellipse.semi_major)
    b = max(1.0, ellipse.semi_minor)
    angle_rad = math.radians(ellipse.angle_deg)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x_loc = (px - cx) * cos_a + (py - cy) * sin_a
    y_loc = -(px - cx) * sin_a + (py - cy) * cos_a
    dx_loc = dx * cos_a + dy * sin_a
    dy_loc = -dx * sin_a + dy * cos_a

    A = (dx_loc / a) ** 2 + (dy_loc / b) ** 2
    B = (x_loc * dx_loc) / (a ** 2) + (y_loc * dy_loc) / (b ** 2)
    C = (x_loc / a) ** 2 + (y_loc / b) ** 2 - 1.0

    disc = B ** 2 - A * C
    if disc < 0 or A == 0:
        return px + dx * a, py + dy * b

    t = (-B + math.sqrt(disc)) / A
    return px + t * dx, py + t * dy


# ── Pure drawing helpers ─────────────────────────────────────────────


def draw_filled_structure(
    out: np.ndarray,
    ellipse_data: Any,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Semi-transparent fill inside an ellipse / circle."""
    if alpha <= 0.0:
        return
    overlay = out.copy()
    ct = (int(round(ellipse_data.center_x)), int(round(ellipse_data.center_y)))
    ratio = (ellipse_data.semi_minor / ellipse_data.semi_major
             if ellipse_data.semi_major > 0 else 1.0)
    if ratio > _CIRCLE_DRAW_THRESHOLD:
        r = int(round((ellipse_data.semi_major + ellipse_data.semi_minor) / 2.0))
        cv2.circle(overlay, ct, r, color, -1, cv2.LINE_AA)
    else:
        axes = (int(round(ellipse_data.semi_major)),
                int(round(ellipse_data.semi_minor)))
        cv2.ellipse(overlay, ct, axes, int(round(ellipse_data.angle_deg)),
                    0, 360, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)


def draw_cross_section(out: np.ndarray, result: Any, scale: float) -> None:
    """Horizontal/vertical cross-section lines between pupil and limbus."""
    if not (
        result.pupil.detected
        and result.pupil.ellipse is not None
        and result.limbus.detected
        and result.limbus.ellipse is not None
    ):
        return

    p_ellipse = scale_ellipse(result.pupil.ellipse, scale)
    l_ellipse = scale_ellipse(result.limbus.ellipse, scale)

    p_cx, p_cy = p_ellipse.center_x, p_ellipse.center_y
    l_cx, l_cy = l_ellipse.center_x, l_ellipse.center_y

    p_up = get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, -1.0)
    p_dn = get_ellipse_intersection(p_ellipse, p_cx, p_cy, 0.0, 1.0)
    p_lt = get_ellipse_intersection(p_ellipse, p_cx, p_cy, -1.0, 0.0)
    p_rt = get_ellipse_intersection(p_ellipse, p_cx, p_cy, 1.0, 0.0)

    l_up = get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, -1.0)
    l_dn = get_ellipse_intersection(l_ellipse, l_cx, l_cy, 0.0, 1.0)
    l_lt = get_ellipse_intersection(l_ellipse, l_cx, l_cy, -1.0, 0.0)
    l_rt = get_ellipse_intersection(l_ellipse, l_cx, l_cy, 1.0, 0.0)

    green = (0, 255, 0)
    blue = (255, 100, 0)

    def _pt(p):
        return int(round(p[0])), int(round(p[1]))

    cv2.line(out, _pt(p_lt), _pt(p_rt), green, 1, cv2.LINE_AA)
    cv2.line(out, _pt(p_up), _pt(p_dn), green, 1, cv2.LINE_AA)
    cv2.line(out, _pt(l_lt), _pt(l_rt), blue, 1, cv2.LINE_AA)
    cv2.line(out, _pt(l_up), _pt(l_dn), blue, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fsz = max(0.3, 0.4 * scale)
    lbl = (220, 220, 220)

    ux, uy = _pt(l_up)
    cv2.putText(out, "270", (ux - int(10 * scale), uy - int(5 * scale)), font, fsz, lbl, 1, cv2.LINE_AA)
    dx, dy = _pt(l_dn)
    cv2.putText(out, "90", (dx - int(7 * scale), dy + int(12 * scale)), font, fsz, lbl, 1, cv2.LINE_AA)
    lx, ly = _pt(l_lt)
    cv2.putText(out, "0", (lx - int(15 * scale), ly + int(4 * scale)), font, fsz, lbl, 1, cv2.LINE_AA)
    rx, ry = _pt(l_rt)
    cv2.putText(out, "180", (rx + int(5 * scale), ry + int(4 * scale)), font, fsz, lbl, 1, cv2.LINE_AA)


# ── Iris-feature overlay (purely visual, additive) ────────────────────


def _log_iris_rendered(detected: int, sent: int, rendered: int) -> None:
    """Rate-limited runtime diagnostic for the iris-feature render path."""
    now = time.monotonic()
    if now - _IRIS_LAST_LOG[0] < _IRIS_LOG_INTERVAL_S:
        return
    _IRIS_LAST_LOG[0] = now
    _IRIS_LOG.info(
        "iris overlay: detected=%d accepted=%d rendered=%d",
        detected,
        sent,
        rendered,
    )


def draw_iris_feature_overlay(
    out: np.ndarray, result: Any, scale: float, label_y: int = 30
) -> None:
    """Draw accepted iris features at their source-image positions.

    Pure rendering of an existing ~detection result; never mutates it.  Called
    by the scaled live overlay and the full-resolution snapshot/export overlay.
    """
    iris_det = getattr(result, "iris_detection", None)
    if iris_det is None or not getattr(iris_det, "valid", False):
        return
    fs = getattr(iris_det, "feature_set", None)
    if fs is None:
        return
    feats = [f for f in fs.features if getattr(f, "valid", True)]
    if not feats:
        return

    r = max(2, int(round(3.0 * scale)))
    for f in feats:
        pt = (int(round(f.x * scale)), int(round(f.y * scale)))
        cv2.circle(out, pt, r, _IRIS_FEATURE_COLOR, -1, cv2.LINE_AA)

    cv2.putText(
        out,
        f"Iris: {len(feats)} features",
        (10, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, 0.45 * scale),
        _IRIS_FEATURE_COLOR,
        1,
        cv2.LINE_AA,
    )
    _log_iris_rendered(
        getattr(fs, "num_candidates", len(feats)),
        len(fs.features),
        len(feats),
    )


# ── State-dependent overlays (gui = PupilTrackingGUI instance) ──────


def draw_overlay_scaled(gui: Any, out: np.ndarray, result: Any, scale: float) -> None:
    """Main scaled overlay — ring, limbus, pupil, corneal centre, offset, alerts."""
    h, w = out.shape[:2]
    cal = result.calibration

    ring_status = getattr(result, "ring_status", "unknown")
    if ring_status == "ring_present":
        ring_center = getattr(result, "ring_center", None)
        ring_radius = getattr(result, "ring_radius", None)
        ring_contour = getattr(result, "ring_contour", None)
        if ring_center is not None and ring_radius is not None:
            cx = int(round(ring_center[0] * scale))
            cy = int(round(ring_center[1] * scale))
            rr = int(round(ring_radius * scale))
            if ring_contour is not None and len(ring_contour) >= 5:
                scaled = np.round(ring_contour.astype(np.float32) * scale).astype(np.int32)
                cv2.drawContours(out, [scaled], -1, (0, 0, 255), 2)
            else:
                cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
            if gui._show_ring_center.get():
                _base = max(4, int(10 * scale))
                _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
                if _cal_mm_per_px > 0:
                    _ring_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
                else:
                    _ring_cross_size = int(max(12, min(_base, 26)))
                cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, _ring_cross_size, 2, cv2.LINE_AA)
            if gui._show_measurements.get():
                label = f"R={ring_radius * 2.0:.0f}px"
                if cal.calibrated:
                    label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                cv2.putText(out, label, (cx + 10, cy - 18), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, 0.45 * scale), (0, 0, 255), 1, cv2.LINE_AA)

    if gui._show_limbus.get() and result.limbus.detected and result.limbus.ellipse is not None:
        e_orig = result.limbus.ellipse
        e = scale_ellipse(e_orig, scale)
        limbus_color = (255, 100, 0)
        limbus_alpha = gui._limbus_fill_alpha_var.get() / 100.0
        if limbus_alpha > 0:
            draw_filled_structure(out, e, limbus_color, limbus_alpha)
        ct = draw_structure(out, e, limbus_color)
        if gui._show_centers.get():
            cv2.circle(out, ct, max(2, int(4 * scale)), limbus_color, -1)
        if gui._show_measurements.get():
            dia_px = e_orig.radius * 2.0
            label = f"D={dia_px:.0f}px"
            if cal.calibrated:
                dia_mm = dia_px * cal.mm_per_px
                smaj_mm = e_orig.semi_major * cal.mm_per_px
                smin_mm = e_orig.semi_minor * cal.mm_per_px
                label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
            ft = getattr(e_orig, "fit_type", None) or getattr(result.limbus, "fit_type", None)
            if ft:
                label += f" [{ft}]"
            font_scale = max(0.3, 0.45 * scale)
            cv2.putText(out, label, (ct[0] + 10, ct[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, limbus_color, 1, cv2.LINE_AA)

    if gui._show_pupil.get() and result.pupil.detected and result.pupil.ellipse is not None:
        e_orig = result.pupil.ellipse
        e = scale_ellipse(e_orig, scale)
        pupil_color = (0, 255, 0)
        pupil_alpha = gui._pupil_fill_alpha_var.get() / 100.0
        if pupil_alpha > 0:
            draw_filled_structure(out, e, pupil_color, pupil_alpha)
        ct = draw_structure(out, e, pupil_color)
        if gui._show_centers.get():
            cv2.circle(out, ct, max(2, int(4 * scale)), pupil_color, -1)
        if gui._show_measurements.get():
            dia_px = e_orig.radius * 2.0
            label = f"D={dia_px:.0f}px"
            if cal.calibrated:
                label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
            ft = getattr(e_orig, "fit_type", None) or getattr(result.pupil, "fit_type", None)
            if ft:
                label += f" [{ft}]"
            font_scale = max(0.3, 0.45 * scale)
            cv2.putText(out, label, (ct[0] + 10, ct[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pupil_color, 1, cv2.LINE_AA)

    cc = getattr(result, "corneal_center", None)
    if gui._show_centers.get() and cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
        center_pt = (int(round(cc.center_px[0] * scale)), int(round(cc.center_px[1] * scale)))
        _base = max(4, int(10 * scale))
        cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
        if cal_mm_per_px > 0:
            cursor_size = int(max(10, round(_base + (0.5 / cal_mm_per_px) * scale)))
        else:
            cursor_size = int(max(12, min(_base, 26)))
        cv2.drawMarker(out, center_pt, (255, 255, 255), cv2.MARKER_CROSS, cursor_size, 2, cv2.LINE_AA)
        if gui._show_measurements.get():
            ref_name = getattr(result, "corneal_reference_source", "cornea")
            cv2.putText(out, f"Corneal Centre [{ref_name}]", (center_pt[0] + 12, center_pt[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.46 * scale), (0, 255, 255), 1, cv2.LINE_AA)

    if gui._show_offset.get() and result.has_both:
        p = result.pupil.ellipse
        p_pt = (int(round(p.center_x * scale)), int(round(p.center_y * scale)))
        if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
            ref_pt = (int(round(cc.center_px[0] * scale)), int(round(cc.center_px[1] * scale)))
            dx = p.center_x - cc.center_px[0]
            dy = p.center_y - cc.center_px[1]
        else:
            l = result.limbus.ellipse
            ref_pt = (int(round(l.center_x * scale)), int(round(l.center_y * scale)))
            dx = p.center_x - l.center_x
            dy = p.center_y - l.center_y
        cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
        if gui._show_centers.get():
            _base = max(4, int(10 * scale))
            _cal_mm_per_px = getattr(result.calibration, "mm_per_px", 0.0) or 0.0
            if _cal_mm_per_px > 0:
                _offset_cross_size = int(max(10, round(_base + (0.5 / _cal_mm_per_px) * scale)))
            else:
                _offset_cross_size = int(max(12, min(_base, 26)))
            cv2.drawMarker(out, ref_pt, (255, 255, 255), cv2.MARKER_CROSS, _offset_cross_size, 2, cv2.LINE_AA)
        if gui._show_measurements.get():
            offset_px = math.hypot(dx, dy)
            mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
            label = f"{offset_px:.1f}px"
            if cal.calibrated:
                label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
            font_scale = max(0.25, 0.4 * scale)
            cv2.putText(out, label, (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA)

    draw_cross_section(out, result, scale)

    font_scale_t = max(0.3, 0.5 * scale)
    cv2.putText(out, f"{result.metadata.processing_time_ms:.0f}ms", (w - max(80, int(100 * scale)), 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale_t, (180, 180, 180), 1)

    mode = gui._grayscale_mode_var.get()
    _ = mode

    font_scale_a = max(0.25, 0.4 * scale)
    for i, alert in enumerate(result.alerts[:3]):
        cv2.putText(out, alert[:80], (10, h - 15 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale_a, (0, 100, 255), 1)

    draw_ruler_overlay(gui, out, scale)

    draw_iris_feature_overlay(out, result, scale, label_y=30)


def draw_debug_overlay(gui: Any, out: np.ndarray, scale: float) -> None:
    """Performance / pipeline debug panel in bottom-right."""
    stats = gui._last_opt_stats
    if not stats:
        return
    h, w = out.shape[:2]
    pad = max(8, int(10 * scale))
    line_gap = max(16, int(18 * scale))
    font_scale = max(0.32, 0.42 * scale)
    lines = [
        f"Preset: {gui._performance_preset_var.get().replace('_', ' ').title()}",
        f"Pipeline: {stats.get('backend', gui._pipeline_var.get())}",
        f"Latency avg: {float(stats.get('latency_avg_ms', 0.0) or 0.0):.1f} ms",
        f"Proc avg: {float(stats.get('processing_avg_ms', 0.0) or 0.0):.1f} ms",
        f"ROI avg: {float(stats.get('roi_avg_ms', 0.0) or 0.0):.1f} ms",
        f"ROI mode: {str(stats.get('roi_mode', 'off')).title()}",
        f"Tracking: {gui._tracking_state_var.get() or '---'}",
        (
            "Adaptive quality: "
            + (
                f"ON (stable={int(stats.get('stable_tracking_streak', 0))}, "
                f"skips={int(stats.get('quality_check_skips', 0))})"
                if stats.get("adaptive_quality_active")
                else "OFF"
            )
        ),
        (
            f"Dropped/Stale: {int(stats.get('dropped_frames', 0))}/"
            f"{int(stats.get('stale_frames', 0))}"
        ),
        (
            "Overload protection: "
            + (
                f"ACTIVE (reuse={int(stats.get('cached_reuse_total', 0))})"
                if stats.get("overload_active")
                else f"Idle (reuse={int(stats.get('cached_reuse_total', 0))})"
            )
        ),
    ]
    box_width = max(220, int(w * 0.34))
    box_height = pad * 2 + line_gap * len(lines)
    x0 = max(0, w - box_width - pad)
    y0 = max(0, h - box_height - pad)
    cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (18, 18, 18), -1)
    cv2.rectangle(out, (x0, y0), (x0 + box_width, y0 + box_height), (70, 70, 70), 1)
    for idx, line in enumerate(lines):
        y = y0 + pad + line_gap * (idx + 1) - 4
        cv2.putText(out, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230, 230, 230), 1, cv2.LINE_AA)


def draw_manual_roi_overlay(gui: Any, out: np.ndarray, scale: float) -> None:
    """Manual ROI circle with dim-outside shading."""
    roi = gui._roi_preview if gui._roi_preview is not None else gui._active_manual_roi()
    if roi is None:
        return
    cx = int(round(roi["center_x"] * scale))
    cy = int(round(roi["center_y"] * scale))
    radius = max(1, int(round(roi["radius"] * scale)))
    is_editing = gui._roi_preview is not None and gui._roi_edit_active
    color = (0, 255, 255) if is_editing else (0, 220, 255)
    original = out.copy()
    mask = np.zeros(out.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
    shaded = out.copy()
    shaded[:] = (15, 15, 15)
    outside = cv2.bitwise_not(mask)
    out[:] = cv2.addWeighted(out, 0.45, shaded, 0.55, 0.0)
    inside_original = cv2.bitwise_and(original, original, mask=mask)
    outside_dimmed = cv2.bitwise_and(out, out, mask=outside)
    out[:] = cv2.add(inside_original, outside_dimmed)
    cv2.circle(out, (cx, cy), radius, color, 2, cv2.LINE_AA)
    if is_editing:
        cv2.circle(out, (cx, cy), 3, color, -1)
    handle_x = cx + radius
    handle_y = cy
    cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)

    dia_px = roi["radius"] * 2.0
    dia_str = f"Dia: {dia_px:.0f}px"
    if gui._current_result is not None and getattr(gui._current_result, "calibration", None) is not None and gui._current_result.calibration.calibrated:
        dia_mm = dia_px * gui._current_result.calibration.mm_per_px
        dia_str += f" ({dia_mm:.2f}mm)"

    label = f"ROI {dia_str} (Enter=lock)" if is_editing else f"ROI {dia_str}"
    font_scale = max(0.4, 0.5 * scale)
    (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    text_x = max(10, cx - radius - text_w - 10)
    if is_editing:
        cv2.putText(out, label, (text_x, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        caption = "ROI Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
        cv2.putText(out, caption, (max(10, cx - radius), max(20, cy - radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.48 * scale), color, 1, cv2.LINE_AA)


def draw_manual_ring_overlay(gui: Any, out: np.ndarray, scale: float) -> None:
    """Manual ring circle."""
    ring = gui._ring_preview if gui._ring_preview is not None else gui._active_manual_ring()
    if ring is None:
        return
    cx = int(round(ring["center_x"] * scale))
    cy = int(round(ring["center_y"] * scale))
    radius = max(1, int(round(ring["radius"] * scale)))
    is_editing = gui._ring_preview is not None and gui._ring_edit_active
    if (
        not is_editing
        and gui._current_result is not None
        and getattr(gui._current_result, "ring_status", "unknown") == "ring_present"
    ):
        return
    color = (40, 110, 255) if is_editing else (0, 0, 255)
    thickness = 2 if is_editing else 3
    cv2.circle(out, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), 3, color, -1)
    handle_x = cx + radius
    handle_y = cy
    cv2.circle(out, (handle_x, handle_y), max(4, int(6 * scale)), color, -1)

    dia_px = ring["radius"] * 2.0
    dia_str = f"Dia: {dia_px:.0f}px"
    if gui._current_result is not None and getattr(gui._current_result, "calibration", None) is not None and gui._current_result.calibration.calibrated:
        dia_mm = dia_px * gui._current_result.calibration.mm_per_px
        dia_str += f" ({dia_mm:.2f}mm)"

    label = f"Manual Ring {dia_str} (Enter=lock)" if is_editing else f"Manual Ring {dia_str}"
    cv2.putText(out, label, (cx + 10, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.45 * scale), color, 1, cv2.LINE_AA)
    cv2.circle(out, (handle_x, handle_y), max(5, int(7 * scale)), (20, 20, 20), 1)

    if is_editing:
        caption = "Ring Edit: drag move/resize, arrows nudge, Enter apply, Esc cancel"
        cv2.putText(out, caption, (max(10, cx - radius), max(20, cy - radius - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.48 * scale), color, 1, cv2.LINE_AA)


def draw_ruler_overlay(gui: Any, out: np.ndarray, scale: float) -> None:
    """Ruler calibration points and line."""
    if not getattr(gui, "_ruler_calibration_active", False) and not getattr(gui, "_ruler_points", None):
        return
    pts = getattr(gui, "_ruler_points", [])
    color = (0, 255, 255)
    for i, pt in enumerate(pts):
        cx = int(round(pt[0] * scale))
        cy = int(round(pt[1] * scale))
        cv2.circle(out, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 9, color, 2, cv2.LINE_AA)
        cv2.putText(out, f"P{i+1}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5 * scale), color, 1, cv2.LINE_AA)

    if len(pts) >= 2:
        p1 = (int(round(pts[0][0] * scale)), int(round(pts[0][1] * scale)))
        p2 = (int(round(pts[1][0] * scale)), int(round(pts[1][1] * scale)))
        cv2.line(out, p1, p2, color, 2, cv2.LINE_AA)
        dist_px = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
        known_mm = float(gui._ruler_known_dist_mm_var.get())
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        cv2.putText(out, f"{dist_px:.1f} px = {known_mm:.1f} mm ({dist_px/known_mm:.2f} px/mm)", (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.52 * scale), color, 2, cv2.LINE_AA)


def draw_overlay(gui: Any, image: np.ndarray, result: Any) -> np.ndarray:
    """Full-resolution overlay (snapshot / export path)."""
    out = image.copy()
    h, w = out.shape[:2]
    cal = result.calibration

    if gui._show_pupil.get() and result.pupil.detected and result.pupil.ellipse is not None:
        e = result.pupil.ellipse
        pupil_color = (0, 255, 0)
        ct = draw_structure(out, e, pupil_color)
        if gui._show_centers.get():
            cv2.circle(out, ct, 4, pupil_color, -1)
        if gui._show_measurements.get():
            dia_px = e.radius * 2.0
            label = f"D={dia_px:.0f}px"
            if cal.calibrated:
                label += f" ({dia_px * cal.mm_per_px:.2f}mm)"
            ft = getattr(e, "fit_type", None) or getattr(result.pupil, "fit_type", None)
            if ft:
                label += f" [{ft}]"
            cv2.putText(out, label, (ct[0] + 10, ct[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pupil_color, 1, cv2.LINE_AA)

    if gui._show_limbus.get() and result.limbus.detected and result.limbus.ellipse is not None:
        e = result.limbus.ellipse
        limbus_color = (255, 100, 0)
        ct = draw_structure(out, e, limbus_color)
        if gui._show_centers.get():
            cv2.circle(out, ct, 4, limbus_color, -1)
        if gui._show_measurements.get():
            dia_px = e.radius * 2.0
            label = f"D={dia_px:.0f}px"
            if cal.calibrated:
                dia_mm = dia_px * cal.mm_per_px
                smaj_mm = e.semi_major * cal.mm_per_px
                smin_mm = e.semi_minor * cal.mm_per_px
                label += f" ({dia_mm:.2f}mm  {smaj_mm:.2f}x{smin_mm:.2f})"
            ft = getattr(e, "fit_type", None) or getattr(result.limbus, "fit_type", None)
            if ft:
                label += f" [{ft}]"
            cv2.putText(out, label, (ct[0] + 10, ct[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, limbus_color, 1, cv2.LINE_AA)

    ring_status = getattr(result, "ring_status", "unknown")
    if ring_status == "ring_present":
        ring_center = getattr(result, "ring_center", None)
        ring_radius = getattr(result, "ring_radius", None)
        ring_contour = getattr(result, "ring_contour", None)
        if ring_center is not None and ring_radius is not None:
            cx = int(round(ring_center[0]))
            cy = int(round(ring_center[1]))
            rr = int(round(ring_radius))
            if ring_contour is not None and len(ring_contour) >= 5:
                cv2.drawContours(out, [ring_contour.astype(np.int32)], -1, (0, 0, 255), 2)
            else:
                cv2.circle(out, (cx, cy), rr, (0, 0, 255), 2, cv2.LINE_AA)
            if gui._show_ring_center.get():
                _base = max(4, int(10))
                _ring_cross_size = int(max(12, min(_base, 26)))
                cv2.drawMarker(out, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, _ring_cross_size, 2, cv2.LINE_AA)
            if gui._show_measurements.get():
                label = f"R={ring_radius * 2.0:.0f}px"
                if cal.calibrated:
                    label += f" ({ring_radius * 2.0 * cal.mm_per_px:.2f}mm)"
                cv2.putText(out, label, (cx + 10, cy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    if gui._show_offset.get() and result.has_both:
        p = result.pupil.ellipse
        p_pt = (int(round(p.center_x)), int(round(p.center_y)))
        cc = getattr(result, "corneal_center", None)
        if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
            ref_pt = (int(round(cc.center_px[0])), int(round(cc.center_px[1])))
            dx = p.center_x - cc.center_px[0]
            dy = p.center_y - cc.center_px[1]
        else:
            l = result.limbus.ellipse
            ref_pt = (int(round(l.center_x)), int(round(l.center_y)))
            dx = p.center_x - l.center_x
            dy = p.center_y - l.center_y
        cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)
        if gui._show_centers.get():
            cv2.drawMarker(out, ref_pt, (10, 10, 10), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.drawMarker(out, ref_pt, (255, 0, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        if gui._show_measurements.get():
            offset_px = math.hypot(dx, dy)
            mid = ((p_pt[0] + ref_pt[0]) // 2, (p_pt[1] + ref_pt[1]) // 2)
            label = f"{offset_px:.1f}px"
            if cal.calibrated:
                label += f" ({offset_px * cal.mm_per_px:.2f}mm)"
            cv2.putText(out, label, (mid[0] + 5, mid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    draw_cross_section(out, result, 1.0)

    quality = result.overall_quality.value if hasattr(result.overall_quality, "value") else str(result.overall_quality)
    color_map = {
        "SURGICAL": (0, 230, 118),
        "CLINICAL": (246, 182, 41),
        "RESEARCH": (38, 167, 255),
        "INSUFFICIENT": (80, 83, 239),
        "NO_DETECTION": (97, 97, 97),
    }
    badge_color = color_map.get(quality, (128, 128, 128))
    cv2.putText(out, f"{quality} ({result.overall_confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, badge_color, 2)
    cv2.putText(out, f"{result.metadata.processing_time_ms:.0f}ms", (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    for i, alert in enumerate(result.alerts[:3]):
        cv2.putText(out, alert[:80], (10, h - 15 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

    draw_iris_feature_overlay(out, result, 1.0, label_y=55)

    return out
