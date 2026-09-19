"""Unit tests for cyclotorsion and registration types & configs."""

import pytest
from pupil_tracking.utils.types import (
    StreamName,
    StreamResult,
    RegistrationResult,
    RegistrationQuality,
    assign_registration_grade,
)
from pupil_tracking.utils.config import get_config, RegistrationConfig


def test_stream_names():
    assert StreamName.PHASE_CORRELATION.value == "phase_correlation"
    assert StreamName.DEEP_MATCHER.value == "deep_matcher"
    assert StreamName.INK_MARKERS.value == "ink_markers"
    assert StreamName.LIMBAL_VESSELS.value == "limbal_vessels"
    assert StreamName.CUSTOM_FEATURE.value == "custom_feature"


def test_stream_result_validity():
    res = StreamResult(stream=StreamName.PHASE_CORRELATION, torsion_deg=2.5, confidence=0.8)
    assert res.valid is True

    invalid_res = StreamResult(stream=StreamName.PHASE_CORRELATION, torsion_deg=None, confidence=0.0)
    assert invalid_res.valid is False


def test_assign_registration_grade():
    assert assign_registration_grade(0.90) == RegistrationQuality.SURGICAL
    assert assign_registration_grade(0.70) == RegistrationQuality.CLINICAL
    assert assign_registration_grade(0.40) == RegistrationQuality.RESEARCH
    assert assign_registration_grade(0.10) == RegistrationQuality.INSUFFICIENT
    assert assign_registration_grade(float("nan")) == RegistrationQuality.NO_RESULT


def test_registration_config_defaults():
    cfg = get_config().registration
    assert isinstance(cfg, RegistrationConfig)
    assert cfg.enabled is True
    assert cfg.polar_num_angles == 360
    assert cfg.polar_num_radial == 64
    assert cfg.fusion_method == "weighted_median"
