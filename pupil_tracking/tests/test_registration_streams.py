"""Unit tests for cyclotorsion detection streams."""

import numpy as np
import pytest
from pupil_tracking.registration.streams.phase_correlation import PhaseCorrelationStream
from pupil_tracking.registration.streams.ink_tracker import InkTrackerStream
from pupil_tracking.registration.streams.vessel_tracker import VesselTrackerStream
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    EllipseParams,
    DetectionQuality,
    StreamName,
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


def test_phase_correlation_stream_zero_rotation():
    stream = PhaseCorrelationStream()
    assert stream.name == StreamName.PHASE_CORRELATION

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

    res = stream.run(img, img, det1, det2)
    assert res.stream == StreamName.PHASE_CORRELATION
    assert res.valid is True
    assert abs(res.torsion_deg) < 1.0  # Zero rotation check


def test_ink_tracker_stream_empty():
    stream = InkTrackerStream()
    assert stream.name == StreamName.INK_MARKERS

    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = stream.run(img1, img2, det1, det2)
    assert res.stream == StreamName.INK_MARKERS
    # No ink marks found in black images
    assert res.inlier_count == 0


def test_vessel_tracker_stream():
    stream = VesselTrackerStream()
    assert stream.name == StreamName.LIMBAL_VESSELS

    img1 = np.zeros((200, 200, 3), dtype=np.uint8)
    img2 = np.zeros((200, 200, 3), dtype=np.uint8)
    det1 = _make_dummy_detection()
    det2 = _make_dummy_detection()

    res = stream.run(img1, img2, det1, det2)
    assert res.stream == StreamName.LIMBAL_VESSELS
