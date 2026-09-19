"""Unit tests for cyclotorsion multi-stream fusion engine."""

import pytest
from pupil_tracking.registration.fusion import FusionEngine
from pupil_tracking.utils.types import (
    StreamName,
    StreamResult,
    RegistrationResult,
    RegistrationQuality,
)


def test_fusion_engine_empty():
    engine = FusionEngine()
    res = engine.fuse([])
    assert isinstance(res, RegistrationResult)
    assert res.valid is False
    assert res.quality == RegistrationQuality.NO_RESULT


def test_fusion_engine_agreement():
    engine = FusionEngine()
    s1 = StreamResult(stream=StreamName.PHASE_CORRELATION, torsion_deg=2.1, confidence=0.9)
    s2 = StreamResult(stream=StreamName.DEEP_MATCHER, torsion_deg=2.3, confidence=0.85)

    res = engine.fuse([s1, s2])
    assert res.valid is True
    assert 2.0 <= res.torsion_deg <= 2.4
    assert res.active_streams == 2
    assert res.agreeing_streams == 2


def test_fusion_engine_outlier_rejection():
    engine = FusionEngine()
    s1 = StreamResult(stream=StreamName.PHASE_CORRELATION, torsion_deg=3.0, confidence=0.9)
    s2 = StreamResult(stream=StreamName.DEEP_MATCHER, torsion_deg=3.1, confidence=0.88)
    s3 = StreamResult(stream=StreamName.INK_MARKERS, torsion_deg=25.0, confidence=0.2)  # Wild outlier

    res = engine.fuse([s1, s2, s3])
    assert res.valid is True
    assert 2.8 <= res.torsion_deg <= 3.3
