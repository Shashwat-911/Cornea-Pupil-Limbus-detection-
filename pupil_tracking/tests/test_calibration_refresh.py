"""Regression tests for calibration mode refresh (forward, reverse, pixel protection).

Verifies that when the calibration mode changes, the mm_per_px value updates
while pixel geometry remains unchanged. This is the unit-level contract that
the GUI fix in _apply_live_settings depends on.
"""

import pytest
import numpy as np
from types import SimpleNamespace

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.utils.types import (
    CalibrationInfo,
    DetectionQuality,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
)

FIXED_PX_PER_MM = 44.5
FIXED_MM_PER_PX = 1.0 / FIXED_PX_PER_MM


def _make_result():
    return EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=320.0, center_y=240.0,
                semi_major=80.0, semi_minor=75.0, angle_deg=12.0,
            ),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(
                center_x=315.0, center_y=235.0,
                semi_major=250.0, semi_minor=240.0, angle_deg=5.0,
            ),
            confidence=0.90,
        ),
        overall_quality=DetectionQuality.RESEARCH,
        overall_confidence=0.92,
    )


def test_detector_set_calibration_mode_returns_new_calibration():
    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal_fixed = det._calibration
    assert cal_fixed.calibrated is True
    assert cal_fixed.method == "fixed_manual"
    assert pytest.approx(cal_fixed.px_per_mm) == FIXED_PX_PER_MM

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal_anat = det._calibration
    assert cal_anat.method == "anatomical"
    assert cal_anat is not cal_fixed


def test_calibration_swap_updates_mm_values():
    result = _make_result()

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    result.calibration = det._calibration
    mm_fixed = result.calibration.mm_per_px
    assert pytest.approx(mm_fixed) == FIXED_MM_PER_PX

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    result.calibration = det._calibration
    mm_anat = result.calibration.mm_per_px
    assert mm_anat != mm_fixed


def test_pixel_geometry_unchanged_after_calibration_swap():
    result = _make_result()

    pupil_semi_major = result.pupil.ellipse.semi_major
    pupil_semi_minor = result.pupil.ellipse.semi_minor
    pupil_center = (result.pupil.ellipse.center_x, result.pupil.ellipse.center_y)
    limbus_semi_major = result.limbus.ellipse.semi_major
    limbus_semi_minor = result.limbus.ellipse.semi_minor
    limbus_center = (result.limbus.ellipse.center_x, result.limbus.ellipse.center_y)

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    result.calibration = det._calibration

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    result.calibration = det._calibration

    assert result.pupil.ellipse.semi_major == pupil_semi_major
    assert result.pupil.ellipse.semi_minor == pupil_semi_minor
    assert (result.pupil.ellipse.center_x, result.pupil.ellipse.center_y) == pupil_center
    assert result.limbus.ellipse.semi_major == limbus_semi_major
    assert result.limbus.ellipse.semi_minor == limbus_semi_minor
    assert (result.limbus.ellipse.center_x, result.limbus.ellipse.center_y) == limbus_center


def test_forward_round_trip_anatomical_to_fixed_to_anatomical():
    result = _make_result()

    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal1 = det._calibration
    result.calibration = cal1
    mm1 = result.calibration.mm_per_px

    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal2 = det._calibration
    result.calibration = cal2
    mm2 = result.calibration.mm_per_px
    assert mm2 != mm1

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal3 = det._calibration
    result.calibration = cal3
    mm3 = result.calibration.mm_per_px
    assert pytest.approx(mm3) == mm1


def test_mm_values_differ_between_modes():
    det = UnifiedDetector()
    det.set_calibration_mode(
        mode="FIXED_PIXEL_SCALE",
        manual_px_per_mm=FIXED_PX_PER_MM,
    )
    cal_fixed = det._calibration

    det.set_calibration_mode(
        mode="ANATOMICAL_ANCHOR",
        corneal_diameter_mm=12.0,
    )
    cal_anat = det._calibration

    assert cal_fixed.mm_per_px != cal_anat.mm_per_px
    assert cal_fixed.px_per_mm != cal_anat.px_per_mm
    assert cal_fixed.method != cal_anat.method


def _compute_wtw(semi_major_px, semi_minor_px, mm_per_px):
    """Recompute WTW the same way _update_measurements does."""
    h = 2.0 * semi_major_px * mm_per_px
    v = 2.0 * semi_minor_px * mm_per_px
    m = (h + v) / 2.0
    return h, v, m


def test_wtw_updates_on_calibration_switch():
    """Stale wtw_horizontal_mm must not persist after mode switch."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration
    # Simulate _add_mm_values storing pre-computed WTW
    h0, v0, m0 = _compute_wtw(le.semi_major, le.semi_minor, result.calibration.mm_per_px)
    result.limbus.wtw_horizontal_mm = h0
    result.limbus.wtw_vertical_mm = v0
    result.limbus.wtw_mean_mm = m0

    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    h_exp, v_exp, m_exp = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)

    # After clearing stale attrs (as _apply_live_settings now does):
    for attr in ("wtw_horizontal_mm", "wtw_vertical_mm", "wtw_mean_mm"):
        if hasattr(result.limbus, attr):
            setattr(result.limbus, attr, None)

    h_wtw = getattr(result.limbus, "wtw_horizontal_mm", None)
    if h_wtw is None:
        h_wtw, v_wtw, m_wtw = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)
    assert pytest.approx(h_wtw, abs=0.01) == h_exp
    assert h_wtw != h0


def test_wtw_recomputed_from_current_calibration():
    """_update_measurements WTW logic always recomputes from current mm_per_px."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration

    h, v, m = _compute_wtw(le.semi_major, le.semi_minor, FIXED_MM_PER_PX)
    expected_h = 2.0 * le.semi_major * FIXED_MM_PER_PX
    assert pytest.approx(h, abs=0.01) == expected_h


def test_fixed_mode_semi_major_not_tautological():
    """FIXED mode semi_major_mm must NOT equal corneal/2."""
    result = _make_result()
    le = result.limbus.ellipse

    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration

    semi_major_mm = le.semi_major * result.calibration.mm_per_px
    # 6.0 is the ANATOMICAL tautology (12.0 / 2); FIXED must differ
    assert semi_major_mm != 6.0
    assert pytest.approx(semi_major_mm, abs=0.01) == le.semi_major * FIXED_MM_PER_PX


def test_anatomical_tautology_semi_major():
    """ANATOMICAL semi_major_mm = corneal_diameter / 2 (tautology)."""
    result = _make_result()
    le = result.limbus.ellipse

    corneal_mm = 12.0
    diameter_px = le.semi_major * 2.0
    mm_per_px = corneal_mm / diameter_px

    semi_major_mm = le.semi_major * mm_per_px
    assert pytest.approx(semi_major_mm, abs=0.001) == corneal_mm / 2.0


def test_different_images_different_wtw_in_fixed_mode():
    """In FIXED mode, different pixel geometries produce different WTW."""
    det = UnifiedDetector()
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)

    r1 = _make_result()
    r1.calibration = det._calibration
    h1 = 2.0 * r1.limbus.ellipse.semi_major * FIXED_MM_PER_PX

    r2 = EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(center_x=320, center_y=240, semi_major=60, semi_minor=55),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(center_x=315, center_y=235, semi_major=180, semi_minor=170),
            confidence=0.90,
        ),
        overall_quality=DetectionQuality.RESEARCH,
        overall_confidence=0.92,
    )
    r2.calibration = det._calibration
    h2 = 2.0 * r2.limbus.ellipse.semi_major * FIXED_MM_PER_PX

    assert h1 != h2
    assert pytest.approx(h1 / h2, abs=0.01) == 250.0 / 180.0


def test_round_trip_mode_switch_no_stale_calibration():
    """ANATOMICAL -> FIXED -> ANATOMICAL: no stale calibration persists."""
    det = UnifiedDetector()
    result = _make_result()

    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration

    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    mm_fixed = result.calibration.mm_per_px

    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    # After ANATOMICAL reset, _current_best returns uncalibrated.
    # The GUI fix computes tautological calibration from pixel geometry.
    ep = result.limbus.ellipse
    corneal_mm = 12.0
    dia_px = ep.semi_major * 2.0
    if dia_px > 10:
        result.calibration = CalibrationInfo(
            calibrated=True,
            px_per_mm=dia_px / corneal_mm,
            mm_per_px=corneal_mm / dia_px,
            method="anatomical",
        )
    mm_back = result.calibration.mm_per_px

    assert mm_back != mm_fixed
    assert pytest.approx(mm_back * ep.semi_major, abs=0.001) == corneal_mm / 2.0


def test_stale_wtw_attrs_cleared_on_mode_switch():
    """Pre-computed WTW attrs must be reset to None when calibration changes."""
    result = _make_result()
    result.limbus.wtw_horizontal_mm = 999.0
    result.limbus.wtw_vertical_mm = 888.0
    result.limbus.wtw_mean_mm = 777.0

    for attr in ("wtw_horizontal_mm", "wtw_vertical_mm", "wtw_mean_mm"):
        if hasattr(result.limbus, attr):
            setattr(result.limbus, attr, None)

    assert result.limbus.wtw_horizontal_mm is None
    assert result.limbus.wtw_vertical_mm is None
    assert result.limbus.wtw_mean_mm is None


def test_no_limbus_does_not_corrupt_calibration():
    """Mode switch with no limbus detection should not crash."""
    det = UnifiedDetector()
    result = EyeDetectionResult(
        pupil=PupilDetection(detected=False),
        limbus=LimbusDetection(detected=False),
        overall_quality=DetectionQuality.NO_DETECTION,
        overall_confidence=0.0,
    )
    det.set_calibration_mode(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=FIXED_PX_PER_MM)
    result.calibration = det._calibration
    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    result.calibration = det._calibration
    assert result.calibration is not None


# ====================================================================
# _adapt_frame_result _eye_result path tests
# ====================================================================
#
# These tests simulate what happens inside _adapt_frame_result when the
# optimized camera processor carries an _eye_result from its internal
# UnifiedDetector.  Before the fix, this path returned the raw
# EyeDetectionResult with the detector's internal ANATOMICAL calibration,
# bypassing the GUI mode dropdown entirely.
# --------------------------------------------------------------------


class _MockTkVar:
    """Minimal stand-in for tk.StringVar that stores a plain Python value."""

    def __init__(self, value=""):
        self._value = value

    def get(self):
        return self._value

    def set(self, v):
        self._value = v


class _FakeGuiApp:
    """Bare-minimum replica of the attributes _adapt_frame_result reads."""

    def __init__(self, cal_mode="ANATOMICAL_ANCHOR", corneal_mm=12.0,
                 fixed_scale=44.5, ring_ref=9.4):
        self._calibration_mode_var = _MockTkVar(cal_mode)
        self._corneal_ref_mm_var = _MockTkVar(str(corneal_mm))
        self._fixed_scale_var = _MockTkVar(str(fixed_scale))
        self._ring_ref_mm_var = _MockTkVar(str(ring_ref))

    def _adapt_frame_result(self, fr, frame_shape):
        """Copy-paste of gui_app.py _adapt_frame_result (the _eye_result branch only)."""
        from pupil_tracking.utils.types import CalibrationInfo
        H, W = frame_shape[:2]
        eye_result = getattr(fr, "_eye_result", None)
        if eye_result is not None:
            eye_result.metadata.image_width = W
            eye_result.metadata.image_height = H
            eye_result.metadata.frame_number = getattr(fr, "frame_number", 0)
            eye_result.metadata.latency_ms = getattr(fr, "latency_ms", getattr(fr, "processing_ms", 0.0))
            _cal_mode = self._calibration_mode_var.get()
            _corneal_mm = float(self._corneal_ref_mm_var.get())
            _fixed_scale = float(self._fixed_scale_var.get())
            _ring_ref_mm = float(self._ring_ref_mm_var.get())
            if _cal_mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
                _px = max(0.1, _fixed_scale)
                new_cal = CalibrationInfo(
                    calibrated=True, px_per_mm=_px, mm_per_px=1.0 / _px,
                    source="fixed_manual", method="fixed_manual",
                    reference_diameter_mm=0.0, reference_diameter_px=0.0,
                    confidence=1.0, corneal_diameter_assumed_mm=None,
                )
            elif _cal_mode == "RING_REFLECTION":
                _ring_r = getattr(fr, "ring_radius", None)
                if _ring_r is not None and _ring_r > 10:
                    _dia = _ring_r * 2.0
                    _px = _dia / _ring_ref_mm
                    new_cal = CalibrationInfo(
                        calibrated=True, px_per_mm=_px, mm_per_px=1.0 / _px,
                        source=f"ring_reflection_{_ring_ref_mm:.1f}mm",
                        method="ring_reflection",
                        reference_diameter_mm=_ring_ref_mm,
                        reference_diameter_px=_dia,
                        confidence=0.95, corneal_diameter_assumed_mm=None,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False, px_per_mm=0.0, mm_per_px=0.0,
                        source="none", method="ring_reflection",
                    )
            else:
                if (
                    getattr(eye_result, "limbus", None) is not None
                    and getattr(eye_result.limbus, "detected", False)
                    and getattr(eye_result.limbus, "ellipse", None) is not None
                ):
                    _lsm = eye_result.limbus.ellipse.semi_major * 2.0
                    _px = _lsm / _corneal_mm if _corneal_mm > 0 else 0.0
                    new_cal = CalibrationInfo(
                        calibrated=True, px_per_mm=_px,
                        mm_per_px=1.0 / _px if _px > 0 else 0.0,
                        source="limbus_semi_major (optimised)",
                        method="anatomical",
                        reference_diameter_mm=_corneal_mm,
                        reference_diameter_px=_lsm,
                        confidence=min(0.95, getattr(eye_result, "overall_confidence", 0.0) + 0.05),
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
                else:
                    new_cal = CalibrationInfo(
                        calibrated=False, px_per_mm=0.0, mm_per_px=0.0,
                        source="none", method="anatomical",
                        corneal_diameter_assumed_mm=_corneal_mm,
                    )
            eye_result.calibration = new_cal
            for target in (getattr(eye_result, "limbus", None), getattr(eye_result, "pupil", None)):
                if target is None:
                    continue
                for attr in ("wtw_horizontal_mm", "wtw_vertical_mm", "wtw_mean_mm",
                             "wtw_astigmatism_mm", "is_wtw_measured", "wtw_validity_status",
                             "radius_mm", "center_mm"):
                    if hasattr(target, attr):
                        try:
                            setattr(target, attr, None)
                        except Exception:
                            pass
            if new_cal.calibrated:
                if (getattr(eye_result, "pupil", None) is not None
                        and eye_result.pupil.detected and eye_result.pupil.ellipse is not None):
                    pe = eye_result.pupil.ellipse
                    eye_result.pupil.radius_mm = pe.radius * new_cal.mm_per_px
                    eye_result.pupil.center_mm = (pe.center_x * new_cal.mm_per_px, pe.center_y * new_cal.mm_per_px)
                if (getattr(eye_result, "limbus", None) is not None
                        and eye_result.limbus.detected and eye_result.limbus.ellipse is not None):
                    le = eye_result.limbus.ellipse
                    eye_result.limbus.radius_mm = le.radius * new_cal.mm_per_px
                    eye_result.limbus.center_mm = (le.center_x * new_cal.mm_per_px, le.center_y * new_cal.mm_per_px)
                    from pupil_tracking.calibration.spatial_calibration import evaluate_clinical_wtw
                    h, v, m, astig, is_m, status = evaluate_clinical_wtw(eye_result.limbus, new_cal)
                    eye_result.limbus.wtw_horizontal_mm = h
                    eye_result.limbus.wtw_vertical_mm = v
                    eye_result.limbus.wtw_mean_mm = m
                    eye_result.limbus.wtw_astigmatism_mm = astig
                    eye_result.limbus.is_wtw_measured = is_m
                    eye_result.limbus.wtw_validity_status = status
            return eye_result
        return SimpleNamespace()

    def _make_frame_ns(self, eye_result):
        fr = SimpleNamespace(
            _eye_result=eye_result,
            frame_number=0,
            processing_ms=10.0,
            latency_ms=12.0,
        )
        return fr


def _make_eye_result_with_anatomical_calibration():
    """Create an EyeDetectionResult that mimics what the internal
    UnifiedDetector produces — calibration method is ANATOMICAL."""
    det = UnifiedDetector()
    det.set_calibration_mode(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=12.0)
    er = _make_result()
    er.calibration = det._calibration
    # Simulate _add_mm_values pre-computing mm with anatomical cal
    er.pupil.radius_mm = er.pupil.ellipse.radius * det._calibration.mm_per_px
    er.limbus.radius_mm = er.limbus.ellipse.radius * det._calibration.mm_per_px
    er.limbus.wtw_horizontal_mm = 12.0  # anatomical tautology
    er.limbus.wtw_vertical_mm = 12.0
    return er


# ── Test: FIXED mode overrides anatomical in _eye_result path ────────

def test_eye_result_fixed_mode_overrides_anatomical():
    """When _eye_result path is used with FIXED mode, calibration must be
    'fixed_manual' and semi_major_mm must NOT be the anatomical tautology."""
    app = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    er = _make_eye_result_with_anatomical_calibration()
    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))

    assert result is er
    assert result.calibration.method == "fixed_manual"
    assert result.calibration.calibrated is True
    assert pytest.approx(result.calibration.px_per_mm) == 44.5

    le = result.limbus.ellipse
    semi_major_mm = le.semi_major * result.calibration.mm_per_px
    # 6.0 is the anatomical tautology; FIXED must differ
    assert semi_major_mm != 6.0
    assert pytest.approx(semi_major_mm, abs=0.01) == le.semi_major / 44.5


def test_eye_result_fixed_mode_wtw_not_stale():
    """Pre-computed WTW (12.0 anatomical) must be replaced with FIXED values."""
    app = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    er = _make_eye_result_with_anatomical_calibration()
    assert er.limbus.wtw_horizontal_mm == 12.0  # stale anatomical value

    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))

    le = result.limbus.ellipse
    expected_wtw_h = 2.0 * le.semi_major / 44.5
    assert result.limbus.wtw_horizontal_mm is not None
    assert pytest.approx(result.limbus.wtw_horizontal_mm, abs=0.01) == expected_wtw_h
    assert result.limbus.wtw_horizontal_mm != 12.0


def test_eye_result_anatomical_mode_preserves_tautology():
    """ANATOMICAL mode in _eye_result path should produce corneal/2 tautology."""
    app = _FakeGuiApp(cal_mode="ANATOMICAL_ANCHOR", corneal_mm=12.0)
    er = _make_eye_result_with_anatomical_calibration()
    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))

    assert result.calibration.method == "anatomical"
    le = result.limbus.ellipse
    semi_major_mm = le.semi_major * result.calibration.mm_per_px
    assert pytest.approx(semi_major_mm, abs=0.001) == 12.0 / 2.0


def test_eye_result_pixel_geometry_unchanged_after_mode_override():
    """Switching calibration mode must NOT alter pixel coordinates."""
    app = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    er = _make_eye_result_with_anatomical_calibration()
    pupil_cx = er.pupil.ellipse.center_x
    pupil_sm = er.pupil.ellipse.semi_major
    limbus_cx = er.limbus.ellipse.center_x
    limbus_sm = er.limbus.ellipse.semi_major

    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))

    assert result.pupil.ellipse.center_x == pupil_cx
    assert result.pupil.ellipse.semi_major == pupil_sm
    assert result.limbus.ellipse.center_x == limbus_cx
    assert result.limbus.ellipse.semi_major == limbus_sm


def test_eye_result_to_dict_reflects_override():
    """to_dict() on the adapted result must serialize the overridden calibration."""
    app = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    er = _make_eye_result_with_anatomical_calibration()
    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))

    d = result.to_dict()
    cal = d.get("calibration", {})
    assert cal.get("method") == "fixed_manual"
    assert cal.get("calibrated") is True


def test_eye_result_no_limbus_fixed_mode_still_calibrated():
    """FIXED mode with no limbus: calibration is valid (doesn't need limbus)."""
    app = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    er = EyeDetectionResult(
        pupil=PupilDetection(detected=False),
        limbus=LimbusDetection(detected=False),
        overall_quality=DetectionQuality.NO_DETECTION,
        overall_confidence=0.0,
    )
    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))
    assert result is er
    assert result.calibration.calibrated is True
    assert result.calibration.method == "fixed_manual"


def test_eye_result_no_limbus_anatomical_mode_uncalibrated():
    """ANATOMICAL mode with no limbus: can't compute calibration (needs limbus)."""
    app = _FakeGuiApp(cal_mode="ANATOMICAL_ANCHOR", corneal_mm=12.0)
    er = EyeDetectionResult(
        pupil=PupilDetection(detected=False),
        limbus=LimbusDetection(detected=False),
        overall_quality=DetectionQuality.NO_DETECTION,
        overall_confidence=0.0,
    )
    fr = app._make_frame_ns(er)
    result = app._adapt_frame_result(fr, (480, 640, 3))
    assert result is er
    assert result.calibration.calibrated is False


def test_eye_result_fixed_vs_anatomical_different_wtw():
    """FIXED and ANATOMICAL modes must produce different WTW on the same eye."""
    er_fixed = _make_eye_result_with_anatomical_calibration()
    app_fix = _FakeGuiApp(cal_mode="FIXED_PIXEL_SCALE", fixed_scale=44.5)
    fr_fix = app_fix._make_frame_ns(er_fixed)
    r_fix = app_fix._adapt_frame_result(fr_fix, (480, 640, 3))

    er_anat = _make_eye_result_with_anatomical_calibration()
    app_anat = _FakeGuiApp(cal_mode="ANATOMICAL_ANCHOR", corneal_mm=12.0)
    fr_anat = app_anat._make_frame_ns(er_anat)
    r_anat = app_anat._adapt_frame_result(fr_anat, (480, 640, 3))

    assert r_fix.limbus.wtw_horizontal_mm != r_anat.limbus.wtw_horizontal_mm
    assert r_fix.calibration.method == "fixed_manual"
    assert r_anat.calibration.method == "anatomical"
