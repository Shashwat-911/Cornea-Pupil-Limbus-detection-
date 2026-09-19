"""Integration tests for RegistrationEngine."""

import numpy as np
import pytest
from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    EllipseParams,
    DetectionQuality,
    RegistrationResult,
)


def _make_dummy_detection(center=(100.0, 100.0), pupil_r=25.0, limbus_r=60.0):
    det = EyeDetectionResult()
    det.pupil.detected = True
    det.pupil.ellipse = EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=pupil_r,
        semi_minor=pupil_r,
        angle_deg=0.0,
    )
    det.pupil.confidence = 0.9
    det.pupil.quality = DetectionQuality.SURGICAL

    det.limbus.detected = True
    det.limbus.ellipse = EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=limbus_r,
        semi_minor=limbus_r,
        angle_deg=0.0,
    )
    det.limbus.confidence = 0.9
    det.limbus.quality = DetectionQuality.SURGICAL
    return det


def test_registration_engine_basic():
    engine = RegistrationEngine()
    assert len(engine.streams) > 0

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    y, x = np.ogrid[:200, :200]
    dist = np.sqrt((x - 100)**2 + (y - 100)**2)
    angle = np.arctan2(y - 100, x - 100)
    iris = (dist >= 25) & (dist <= 60)
    texture = (128 + 100 * np.sin(6 * angle)).astype(np.uint8)
    for c in range(3):
        img[iris, c] = texture[iris]

    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = engine.register(img, img, det1, det2)
    assert isinstance(res, RegistrationResult)
