"""Unit tests for modules created during the Phase 3-9 refactoring.

Covers the pure-logic modules that had zero direct test coverage:
- core/corneal_center.py (calculate)
- core/validation.py (cross_validate_and_reject)
- core/structure_extraction.py (utility functions)
- video/video_models.py (annotate_quality, ManualCircularROI, ManualRingAnnotation)
- interface/gui_helpers.py (hex_to_bgr, ascii_for_capture, scale_ellipse)

Note: Tests for blend_from_available, blend_corneal_center_from_points, and
ring-radius kwargs on SmartContourFitter.fit() were removed — these APIs were
planned but never implemented; the logic lives elsewhere in the pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import pytest

from pupil_tracking.utils.types import (
    CalibrationInfo,
    EllipseParams,
    EyeDetectionResult,
)
from pupil_tracking.core.deterministic_ring_detector import (
    RingDetectionResult,
    RingStatus,
)
from pupil_tracking.core.smart_fitter import FitResult, FitType


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_ellipse(
    cx: float = 100.0,
    cy: float = 100.0,
    semi_major: float = 30.0,
    semi_minor: float = 25.0,
    angle: float = 0.0,
) -> EllipseParams:
    return EllipseParams(
        center_x=cx,
        center_y=cy,
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=angle,
    )


def _make_result(
    pupil_xy: Tuple[float, float] = (100.0, 100.0),
    pupil_r: float = 20.0,
    pupil_conf: float = 0.9,
    limbus_xy: Tuple[float, float] = (100.0, 100.0),
    limbus_r: float = 50.0,
    limbus_conf: float = 0.85,
    calibrated: bool = False,
) -> EyeDetectionResult:
    r = EyeDetectionResult()
    r.pupil.detected = True
    r.pupil.ellipse = _make_ellipse(pupil_xy[0], pupil_xy[1], pupil_r, pupil_r)
    r.pupil.confidence = pupil_conf
    r.limbus.detected = True
    r.limbus.ellipse = _make_ellipse(limbus_xy[0], limbus_xy[1], limbus_r, limbus_r)
    r.limbus.confidence = limbus_conf
    if calibrated:
        r.calibration = CalibrationInfo(calibrated=True, mm_per_px=0.02, px_per_mm=50.0)
    return r


def _make_ring(
    center: Tuple[float, float] = (100.0, 100.0),
    radius: float = 200.0,
    status: RingStatus = RingStatus.PRESENT,
    confidence: float = 0.9,
) -> RingDetectionResult:
    return RingDetectionResult(
        status=status,
        ring_center=center,
        ring_radius=radius,
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════════
# corneal_center.py
# ═══════════════════════════════════════════════════════════════════════


class TestCornealCenterCalculator:
    """Tests for CornealCenterCalculator."""

    def test_calculate_corneal_equals_limbus(self):
        from pupil_tracking.core.corneal_center import CornealCenterCalculator

        calc = CornealCenterCalculator()
        r = _make_result(pupil_xy=(105.0, 98.0), limbus_xy=(100.0, 100.0))
        result = calc.calculate(r.pupil, r.limbus, r.calibration)
        assert result.valid is True
        # Corneal centre = limbus centre (anatomical definition)
        assert result.center_px == pytest.approx((100.0, 100.0))

    def test_calculate_offset_is_pupil_minus_limbus(self):
        from pupil_tracking.core.corneal_center import CornealCenterCalculator

        calc = CornealCenterCalculator()
        r = _make_result(pupil_xy=(110.0, 100.0), limbus_xy=(100.0, 100.0))
        result = calc.calculate(r.pupil, r.limbus, r.calibration)
        assert result.offset_px == pytest.approx((10.0, 0.0))
        assert result.offset_magnitude_px == pytest.approx(10.0)
        assert result.offset_angle_deg == pytest.approx(0.0)

    def test_calculate_no_calibration_skips_mm(self):
        from pupil_tracking.core.corneal_center import CornealCenterCalculator

        calc = CornealCenterCalculator()
        r = _make_result(calibrated=False)
        result = calc.calculate(r.pupil, r.limbus, r.calibration)
        assert result.center_mm is None
        assert result.offset_magnitude_mm is None

    def test_calculate_with_calibration_computes_mm(self):
        from pupil_tracking.core.corneal_center import CornealCenterCalculator

        calc = CornealCenterCalculator()
        r = _make_result(
            pupil_xy=(110.0, 100.0),
            limbus_xy=(100.0, 100.0),
            calibrated=True,
        )
        result = calc.calculate(r.pupil, r.limbus, r.calibration)
        assert result.center_mm is not None
        assert result.offset_magnitude_mm is not None
        assert result.offset_magnitude_mm > 0

    def test_calculate_confidence_penalty_for_large_offset(self):
        from pupil_tracking.core.corneal_center import CornealCenterCalculator

        calc = CornealCenterCalculator()
        # Pupil centre far from limbus centre (>20% of limbus radius)
        r = _make_result(
            pupil_xy=(160.0, 100.0),
            limbus_xy=(100.0, 100.0),
            limbus_r=50.0,
            pupil_conf=0.9,
            limbus_conf=0.9,
        )
        result = calc.calculate(r.pupil, r.limbus, r.calibration)
        # Confidence should be penalized because offset_ratio = 60/50 = 1.2 > 0.2
        assert result.confidence < 0.9


# NOTE: TestBlendFromAvailable and TestBlendCornealCenterFromPoints were removed.
# They tested CornealCenterCalculator.blend_from_available() and the standalone
# blend_corneal_center_from_points() — APIs planned during Phase 3-9 refactoring
# but never implemented. The corneal-center blending logic lives in
# detector.py (Steps 7-8) instead.


# ═══════════════════════════════════════════════════════════════════════
# validation.py
# ═══════════════════════════════════════════════════════════════════════


class TestCrossValidateAndReject:
    """Tests for cross_validate_and_reject()."""

    def test_valid_pair_passes_through(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = _make_result(
            pupil_xy=(100.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=20.0,
            limbus_r=50.0,
        )
        result = cross_validate_and_reject(r)
        assert result.pupil.detected is True
        assert result.limbus.detected is True

    def test_pupil_outside_limbus_rejects(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = _make_result(
            pupil_xy=(200.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=20.0,
            limbus_r=50.0,
            pupil_conf=0.9,
            limbus_conf=0.5,
        )
        result = cross_validate_and_reject(r)
        # Pupil centre is outside limbus (dist=100 > radius=50)
        assert not result.pupil.detected or not result.limbus.detected
        assert any("outside limbus" in a for a in result.alerts)

    def test_pupil_larger_than_limbus_rejects(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = _make_result(
            pupil_xy=(100.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=60.0,
            limbus_r=50.0,
            pupil_conf=0.5,
            limbus_conf=0.9,
        )
        result = cross_validate_and_reject(r)
        assert not result.pupil.detected or not result.limbus.detected
        assert any("larger than limbus" in a for a in result.alerts)

    def test_large_offset_penalises_confidence(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = _make_result(
            pupil_xy=(130.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=20.0,
            limbus_r=50.0,
            pupil_conf=0.9,
            limbus_conf=0.9,
        )
        result = cross_validate_and_reject(r)
        # offset_ratio = 30/50 = 0.6 > 0.5 -> confidence penalised
        assert result.pupil.confidence < 0.9
        assert result.limbus.confidence < 0.9

    def test_ratio_above_085_penalises(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = _make_result(
            pupil_xy=(100.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=45.0,
            limbus_r=50.0,
            pupil_conf=0.9,
            limbus_conf=0.9,
        )
        result = cross_validate_and_reject(r)
        # ratio = 45/50 = 0.9 > 0.85
        assert result.pupil.confidence < 0.9
        assert any("ratio" in a for a in result.alerts)

    def test_ring_containment_penalty(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        ring = _make_ring(center=(100.0, 100.0), radius=60.0)
        r = _make_result(
            pupil_xy=(100.0, 100.0),
            limbus_xy=(100.0, 100.0),
            pupil_r=20.0,
            limbus_r=55.0,  # 55 > 60*0.85=51 -> triggers penalty
            pupil_conf=0.9,
            limbus_conf=0.9,
        )
        result = cross_validate_and_reject(r, ring_result=ring)
        # limbus radius 55 > ring radius 60 * 0.85 = 51 -> penalty
        assert result.limbus.confidence < 0.9

    def test_no_ellipses_returns_early(self):
        from pupil_tracking.core.validation import cross_validate_and_reject

        r = EyeDetectionResult()
        result = cross_validate_and_reject(r)
        assert result.pupil.detected is False
        assert result.limbus.detected is False


# ═══════════════════════════════════════════════════════════════════════
# structure_extraction.py
# ═══════════════════════════════════════════════════════════════════════


class TestFitResultToEllipseParams:
    def test_circle_becomes_symmetric_ellipse(self):
        from pupil_tracking.core.structure_extraction import fit_result_to_ellipse_params

        fit = FitResult(
            fit_type=FitType.CIRCLE,
            center_x=100.0,
            center_y=100.0,
            radius=25.0,
            fit_quality=0.9,
        )
        ep = fit_result_to_ellipse_params(fit)
        assert ep.center_x == 100.0
        assert ep.semi_major == 25.0
        assert ep.semi_minor == 25.0
        assert ep.angle_deg == 0.0

    def test_ellipse_preserves_axes(self):
        from pupil_tracking.core.structure_extraction import fit_result_to_ellipse_params

        fit = FitResult(
            fit_type=FitType.ELLIPSE,
            center_x=100.0,
            center_y=100.0,
            semi_major=30.0,
            semi_minor=20.0,
            angle_deg=45.0,
            fit_quality=0.8,
        )
        ep = fit_result_to_ellipse_params(fit)
        assert ep.semi_major == 30.0
        assert ep.semi_minor == 20.0
        assert ep.angle_deg == 45.0


class TestFitResultConfidence:
    def test_circle_gets_bonus(self):
        from pupil_tracking.core.structure_extraction import fit_result_confidence

        fit = FitResult(fit_type=FitType.CIRCLE, fit_quality=0.8)
        conf = fit_result_confidence(fit)
        assert conf == pytest.approx(min(1.0, 0.8 * 1.05))

    def test_ellipse_no_bonus(self):
        from pupil_tracking.core.structure_extraction import fit_result_confidence

        fit = FitResult(fit_type=FitType.ELLIPSE, fit_quality=0.8)
        conf = fit_result_confidence(fit)
        assert conf == pytest.approx(0.8)

    def test_clipped_to_0_1(self):
        from pupil_tracking.core.structure_extraction import fit_result_confidence

        fit = FitResult(fit_type=FitType.CIRCLE, fit_quality=1.5)
        conf = fit_result_confidence(fit)
        assert conf == 1.0

        fit2 = FitResult(fit_type=FitType.CIRCLE, fit_quality=-0.5)
        conf2 = fit_result_confidence(fit2)
        assert conf2 == 0.0


class TestIsInsideRing:
    def test_inside_ring(self):
        from pupil_tracking.core.structure_extraction import is_inside_ring

        ring = _make_ring(center=(100.0, 100.0), radius=200.0)
        assert is_inside_ring(100.0, 100.0, 20.0, ring) is True

    def test_outside_ring(self):
        from pupil_tracking.core.structure_extraction import is_inside_ring

        ring = _make_ring(center=(100.0, 100.0), radius=50.0)
        assert is_inside_ring(200.0, 100.0, 20.0, ring) is False

    def test_no_ring_center_returns_true(self):
        from pupil_tracking.core.structure_extraction import is_inside_ring

        ring = RingDetectionResult(status=RingStatus.ABSENT, confidence=0.0)
        assert is_inside_ring(100.0, 100.0, 20.0, ring) is True

    def test_allow_partial(self):
        from pupil_tracking.core.structure_extraction import is_inside_ring

        ring = _make_ring(center=(100.0, 100.0), radius=100.0)
        # dist=80 + radius=20 = 100 <= 100*1.1=110 -> True
        assert is_inside_ring(180.0, 100.0, 20.0, ring, allow_partial=True) is True


class TestApplyRingRoi:
    def test_masks_outside_ring(self):
        from pupil_tracking.core.structure_extraction import apply_ring_roi

        mask = np.ones((200, 200), dtype=np.uint8) * 255
        ring = _make_ring(center=(100.0, 100.0), radius=200.0)
        result = apply_ring_roi(mask, ring, margin_frac=0.5)
        # Centre should be preserved (non-zero), corners should be zeroed
        assert result[100, 100] > 0
        assert result[0, 0] == 0

    def test_no_ring_returns_unchanged(self):
        from pupil_tracking.core.structure_extraction import apply_ring_roi

        mask = np.ones((200, 200), dtype=np.uint8) * 255
        ring = RingDetectionResult(status=RingStatus.ABSENT, confidence=0.0)
        result = apply_ring_roi(mask, ring)
        assert np.array_equal(result, mask)


class TestApplyFitToResult:
    def test_populates_pupil(self):
        from pupil_tracking.core.structure_extraction import apply_fit_to_result

        r = EyeDetectionResult()
        fit = FitResult(
            fit_type=FitType.CIRCLE,
            center_x=100.0,
            center_y=100.0,
            radius=25.0,
            fit_quality=0.9,
            valid=True,
        )
        apply_fit_to_result(r, pupil_fit=fit, limbus_fit=None)
        assert r.pupil.detected is True
        assert r.pupil.ellipse.center_x == 100.0
        assert r.pupil.confidence > 0

    def test_does_not_overwrite_higher_confidence(self):
        from pupil_tracking.core.structure_extraction import apply_fit_to_result

        r = _make_result(pupil_conf=0.95)
        fit = FitResult(
            fit_type=FitType.CIRCLE,
            center_x=50.0,
            center_y=50.0,
            radius=10.0,
            fit_quality=0.5,
            valid=True,
        )
        apply_fit_to_result(r, pupil_fit=fit, limbus_fit=None)
        # Existing confidence 0.95 > new 0.5*1.05=0.525 -> no overwrite
        assert r.pupil.ellipse.center_x == 100.0


# ═══════════════════════════════════════════════════════════════════════
# video/video_models.py
# ═══════════════════════════════════════════════════════════════════════


class TestAnnotateQuality:
    def test_both_detected_averages(self):
        from pupil_tracking.video.video_models import annotate_quality

        det = {
            "pupil_detected": True,
            "limbus_detected": True,
            "pupil_confidence": 0.8,
            "limbus_confidence": 0.6,
        }
        result = annotate_quality(det)
        assert result["overall_confidence"] == pytest.approx(0.7)
        assert result["overall_quality"] is not None

    def test_pupil_only(self):
        from pupil_tracking.video.video_models import annotate_quality

        det = {
            "pupil_detected": True,
            "limbus_detected": False,
            "pupil_confidence": 0.8,
        }
        result = annotate_quality(det)
        assert result["overall_confidence"] == pytest.approx(0.8)

    def test_limbus_only(self):
        from pupil_tracking.video.video_models import annotate_quality

        det = {
            "pupil_detected": False,
            "limbus_detected": True,
            "limbus_confidence": 0.6,
        }
        result = annotate_quality(det)
        assert result["overall_confidence"] == pytest.approx(0.6)

    def test_none_detected(self):
        from pupil_tracking.video.video_models import annotate_quality

        det = {"pupil_detected": False, "limbus_detected": False}
        result = annotate_quality(det)
        assert result["overall_confidence"] == 0.0


class TestManualCircularROI:
    def test_matches_frame_matching(self):
        from pupil_tracking.video.video_models import ManualCircularROI

        roi = ManualCircularROI(100, 100, 50, frame_width=640, frame_height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert roi.matches_frame(frame) is True

    def test_matches_frame_mismatching(self):
        from pupil_tracking.video.video_models import ManualCircularROI

        roi = ManualCircularROI(100, 100, 50, frame_width=640, frame_height=480)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert roi.matches_frame(frame) is False

    def test_matches_frame_no_dimensions(self):
        from pupil_tracking.video.video_models import ManualCircularROI

        roi = ManualCircularROI(100, 100, 50)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert roi.matches_frame(frame) is True


class TestManualRingAnnotation:
    def test_matches_frame(self):
        from pupil_tracking.video.video_models import ManualRingAnnotation

        ann = ManualRingAnnotation(100, 100, 200, frame_width=640, frame_height=480)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert ann.matches_frame(frame) is True

    def test_no_dimensions_always_matches(self):
        from pupil_tracking.video.video_models import ManualRingAnnotation

        ann = ManualRingAnnotation(100, 100, 200)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert ann.matches_frame(frame) is True


class TestFrameResult:
    def test_attribute_access(self):
        from pupil_tracking.video.video_models import FrameResult

        fr = FrameResult(pupil_center=(100, 100), confidence=0.9)
        assert fr.pupil_center == (100, 100)
        assert fr.confidence == 0.9

    def test_attribute_error(self):
        from pupil_tracking.video.video_models import FrameResult

        fr = FrameResult()
        with pytest.raises(AttributeError):
            _ = fr.nonexistent


# ═══════════════════════════════════════════════════════════════════════
# interface/gui_helpers.py
# ═══════════════════════════════════════════════════════════════════════


class TestHexToBgr:
    def test_valid_hex(self):
        from pupil_tracking.interface.gui_helpers import hex_to_bgr

        assert hex_to_bgr("#FF0000") == (0, 0, 255)  # BGR
        assert hex_to_bgr("#00FF00") == (0, 255, 0)
        assert hex_to_bgr("#0000FF") == (255, 0, 0)

    def test_no_hash_prefix(self):
        from pupil_tracking.interface.gui_helpers import hex_to_bgr

        assert hex_to_bgr("FF0000") == (0, 0, 255)

    def test_invalid_length_returns_default(self):
        from pupil_tracking.interface.gui_helpers import hex_to_bgr

        assert hex_to_bgr("#FFF") == (200, 200, 200)
        assert hex_to_bgr("") == (200, 200, 200)


class TestAsciiForCapture:
    def test_unicode_replaced(self):
        from pupil_tracking.interface.gui_helpers import ascii_for_capture

        assert ascii_for_capture("\u2192") == "->"
        assert ascii_for_capture("\u00b5") == "u"
        assert ascii_for_capture("\u2265") == ">="
        assert ascii_for_capture("\u2264") == "<="
        assert ascii_for_capture("\u00b1") == "+/-"

    def test_pure_ascii_unchanged(self):
        from pupil_tracking.interface.gui_helpers import ascii_for_capture

        assert ascii_for_capture("Hello World 123") == "Hello World 123"

    def test_empty_string(self):
        from pupil_tracking.interface.gui_helpers import ascii_for_capture

        assert ascii_for_capture("") == ""

    def test_mixed_content(self):
        from pupil_tracking.interface.gui_helpers import ascii_for_capture

        result = ascii_for_capture("Pupil \u2192 100px")
        assert "\u2192" not in result
        assert "->" in result


class TestScaleEllipse:
    def test_scales_coordinates(self):
        from pupil_tracking.interface.gui_helpers import scale_ellipse

        e = SimpleNamespace(center_x=100.0, center_y=200.0, radius=27.5, semi_major=30.0, semi_minor=25.0, angle_deg=45.0)
        scaled = scale_ellipse(e, 2.0)
        assert scaled.center_x == 200.0
        assert scaled.center_y == 400.0
        assert scaled.radius == 55.0
        assert scaled.semi_major == 60.0
        assert scaled.semi_minor == 50.0
        assert scaled.angle_deg == 45.0


# ═══════════════════════════════════════════════════════════════════════
# video/video_overlay.py
# ═══════════════════════════════════════════════════════════════════════


class TestOverlayRenderer:
    def test_draw_returns_same_shape(self):
        from pupil_tracking.video.video_overlay import OverlayRenderer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det = {"pupil_detected": False, "limbus_detected": False}
        result = OverlayRenderer.draw(frame, det)
        assert result.shape == frame.shape

    def test_draw_with_pupil(self):
        from pupil_tracking.video.video_overlay import OverlayRenderer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        det = {
            "pupil_detected": True,
            "pupil_x": 320,
            "pupil_y": 240,
            "pupil_radius": 30,
            "limbus_detected": False,
        }
        result = OverlayRenderer.draw(frame, det)
        # When out=None, draws in-place on the frame
        assert result is frame
        assert not np.array_equal(result, original)

    def test_draw_with_out_buffer(self):
        from pupil_tracking.video.video_overlay import OverlayRenderer

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = np.zeros_like(frame)
        det = {"pupil_detected": False, "limbus_detected": False}
        result = OverlayRenderer.draw(frame, det, out=out)
        assert result is out
        # out should contain frame content (copied from frame)
        # Text overlays are drawn on top, so they won't be identical
        assert out.shape == frame.shape


# ═══════════════════════════════════════════════════════════════════════
# Limbus candidate evaluation (Phase 4 regression)
# ═══════════════════════════════════════════════════════════════════════


class TestLimbusCandidateEvaluation:
    """Tests for _apply_fit_to_result candidate-selection semantics.

    Verifies that:
    - a valid fit is accepted even when fit_confidence < 0.5
      (the old placeholder)
    - a stronger existing candidate is not overwritten by a weaker one
    - equal confidence preserves existing behavior
    - invalid/no-fit candidates are rejected
    """

    @staticmethod
    def _make_detector():
        from pupil_tracking.core.detector import UnifiedDetector
        return UnifiedDetector()

    @staticmethod
    def _make_fitResult(center_x=100.0, center_y=100.0, radius=50.0,
                        fit_quality=0.42, valid=True):
        from pupil_tracking.core.smart_fitter import FitResult, FitType
        f = FitResult()
        f.valid = valid
        f.center_x = center_x
        f.center_y = center_y
        f.radius = radius
        f.semi_major = radius
        f.semi_minor = radius
        f.fit_quality = fit_quality
        f.fit_type = FitType.CIRCLE
        f.circularity = 1.0
        return f

    def test_first_fit_accepted_below_old_placeholder(self):
        """A valid fit with confidence < 0.5 must be accepted
        when no prior fit exists (detected=True from wrapper,
        confidence=0.0 default)."""
        det = self._make_detector()
        r = _make_result()
        # Simulate wrapper state: detected=True, confidence=0.0
        r.limbus.detected = True
        r.limbus.confidence = 0.0

        fit = self._make_fitResult(fit_quality=0.42)
        det._apply_fit_to_result(r, pupil_fit=None, limbus_fit=fit)

        assert r.limbus.detected is True
        assert r.limbus.ellipse is not None
        assert r.limbus.confidence == pytest.approx(0.42 * 1.05, abs=0.01)

    def test_stronger_candidate_not_overwritten(self):
        """An existing candidate with confidence 0.67 must not be
        overwritten by a weaker candidate at 0.42."""
        det = self._make_detector()
        r = _make_result()
        r.limbus.detected = True
        r.limbus.confidence = 0.672

        weak_fit = self._make_fitResult(fit_quality=0.40)
        det._apply_fit_to_result(r, pupil_fit=None, limbus_fit=weak_fit)

        assert r.limbus.confidence == pytest.approx(0.672)

    def test_equal_confidence_preserves_existing(self):
        """A candidate with equal confidence preserves existing."""
        det = self._make_detector()
        r = _make_result()
        r.limbus.detected = True
        r.limbus.confidence = 0.50

        equal_fit = self._make_fitResult(fit_quality=0.50 / 1.05)
        det._apply_fit_to_result(r, pupil_fit=None, limbus_fit=equal_fit)

        # new_conf >= old: should be applied (equal is accepted)
        assert r.limbus.detected is True
        assert r.limbus.ellipse is not None

    def test_invalid_fit_rejected(self):
        """An invalid fit must not change the result."""
        det = self._make_detector()
        r = _make_result()
        r.limbus.detected = True
        r.limbus.confidence = 0.6

        bad_fit = self._make_fitResult(valid=False)
        det._apply_fit_to_result(r, pupil_fit=None, limbus_fit=bad_fit)

        assert r.limbus.confidence == pytest.approx(0.6)

    def test_no_fit_leaves_defaults(self):
        """When no fit is provided, result retains wrapper defaults."""
        det = self._make_detector()
        r = _make_result()
        r.limbus.detected = True
        r.limbus.confidence = 0.0

        det._apply_fit_to_result(r, pupil_fit=None, limbus_fit=None)

        assert r.limbus.detected is True
        assert r.limbus.confidence == 0.0


class TestHeuristicLimbusGuard:
    """Tests for the heuristic limbus replacement None guard."""

    def test_radius_mm_none_does_not_raise(self):
        """radius_mm=None must not raise TypeError in the heuristic path."""
        from pupil_tracking.utils.types import EyeDetectionResult
        from pupil_tracking.core.deterministic_ring_detector import (
            RingDetectionResult, RingStatus,
        )

        r = EyeDetectionResult()
        r.limbus.detected = True
        r.limbus.confidence = 0.5
        # radius_mm is None by default — no TypeError should occur
        assert r.limbus.radius_mm is None

        # Simulate the guard condition from detector.py
        ring_status = "PRESENT"
        ring_inner = 200.0
        should_enter = (
            ring_status == "PRESENT"
            and ring_inner is not None
            and r.limbus.detected
            and r.limbus.ellipse is not None
            and r.limbus.radius_mm is not None
            and r.limbus.radius_mm > 0
        )
        # Guard blocks entry when radius_mm is None
        assert should_enter is False

    def test_heuristic_triggers_when_radius_mm_present(self):
        """Heuristic should trigger when radius_mm is small and valid."""
        from pupil_tracking.utils.types import EyeDetectionResult, EllipseParams

        r = EyeDetectionResult()
        r.limbus.detected = True
        r.limbus.confidence = 0.5
        r.limbus.ellipse = EllipseParams(
            center_x=100.0, center_y=100.0,
            semi_major=50.0, semi_minor=50.0, angle_deg=0.0,
        )
        r.limbus.radius_mm = 5.0  # small diameter: 10mm < 12mm

        ring_status = "PRESENT"
        ring_inner = 200.0
        should_enter = (
            ring_status == "PRESENT"
            and ring_inner is not None
            and r.limbus.detected
            and r.limbus.ellipse is not None
            and r.limbus.radius_mm is not None
            and r.limbus.radius_mm > 0
        )
        assert should_enter is True
        assert r.limbus.radius_mm * 2.0 < 12.0


# ──────────────────────────────────────────────────────────────────────
# Phase 5 — Ring-constrained limbus fitting
# ──────────────────────────────────────────────────────────────────────


class TestRingConstrainedFitting:
    """Tests for ring-constraint logic in the detection pipeline."""

    def test_tighter_roi_refit_when_quality_low(self):
        """When fit quality < 0.50 and radius too large, detector should refit with tighter ROI."""
        from pupil_tracking.core.detector import UnifiedDetector
        from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult
        from pupil_tracking.core.deterministic_ring_detector import RingDetectionResult, RingStatus

        # Simulate: ring with inner=200, fit with quality=0.40 and radius=170
        # This should trigger tighter ROI refit (radius > 0.70 * ring_inner)
        ring = RingDetectionResult(
            status=RingStatus.PRESENT,
            confidence=0.9,
            ring_center=(200, 200),
            ring_radius=220,
            ring_inner_radius=200,
        )
        # The condition: quality < 0.50 AND radius > 0.70 * ring_inner
        fit_quality = 0.40
        fit_radius = 170.0
        ring_inner = 200.0

        should_refit = (
            fit_quality is not None
            and fit_quality < 0.50
            and ring_inner is not None
            and ring_inner > 0
            and fit_radius > ring_inner * 0.70
        )
        assert should_refit is True

    def test_no_refit_when_quality_good(self):
        """When fit quality >= 0.50, tighter ROI refit should NOT trigger."""
        fit_quality = 0.64
        fit_radius = 170.0
        ring_inner = 200.0

        should_refit = (
            fit_quality is not None
            and fit_quality < 0.50
            and ring_inner is not None
            and ring_inner > 0
            and fit_radius > ring_inner * 0.70
        )
        assert should_refit is False

    def test_no_refit_when_radius_within_bounds(self):
        """When radius <= 0.70 * ring_inner, tighter ROI refit should NOT trigger."""
        fit_quality = 0.40
        fit_radius = 130.0
        ring_inner = 200.0

        should_refit = (
            fit_quality is not None
            and fit_quality < 0.50
            and ring_inner is not None
            and ring_inner > 0
            and fit_radius > ring_inner * 0.70
        )
        assert should_refit is False

    def test_eye_01_unchanged_after_ring_constraint(self):
        """eye_01 (known good) should not regress from ring-constraint changes."""
        import cv2
        from pupil_tracking.core.detector import UnifiedDetector
        import logging
        logging.getLogger().setLevel(logging.WARNING)
        for n in ["pupil_tracking", "onnxruntime", "urllib3"]:
            logging.getLogger(n).setLevel(logging.WARNING)

        det = UnifiedDetector()
        img = cv2.imread("clinical_data/clean/eye_01.jpeg")
        r = det.detect(img, frame_number=-1, source="eye_01.jpeg")

        pe = r.pupil.ellipse
        le = r.limbus.ellipse
        assert abs(pe.center_x - 382.43) < 1.0
        assert abs(pe.center_y - 335.93) < 1.0
        assert abs(pe.semi_major - 82.96) < 1.0
        assert abs(r.pupil.confidence - 0.798) < 0.01
        assert abs(le.center_x - 382.24) < 1.0
        assert abs(le.center_y - 321.89) < 1.0
        assert abs(le.semi_major - 159.0) < 1.0
        assert abs(r.limbus.confidence - 0.525) < 0.02
