"""Tests for new cyclotorsion registration streams and engine upgrades."""

import cv2
import numpy as np
import pytest

from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.streams.iris_code import IrisCodeStream
from pupil_tracking.registration.streams.polar_optical_flow import PolarOpticalFlowStream
from pupil_tracking.registration.temporal_filter import CyclotorsionKalmanFilter
from pupil_tracking.utils.types import (
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
    PupilDetection,
    StreamName,
)


@pytest.fixture
def synthetic_eye_pair():
    """Create a pair of synthetic eye images rotated by exactly +3.0 degrees."""
    size = 400
    cx, cy = 200.0, 200.0
    r_pupil = 40.0
    r_limbus = 100.0

    # Base image with simulated iris texture
    np.random.seed(42)
    img_ref = np.full((size, size, 3), 180, dtype=np.uint8)

    # Draw iris region
    y, x = np.ogrid[:size, :size]
    dist = np.hypot(x - cx, y - cy)
    iris_mask = (dist >= r_pupil) & (dist <= r_limbus)
    pupil_mask = dist < r_pupil

    # Add radial and angular texture variation
    angle = np.arctan2(y - cy, x - cx)
    pattern = (
        np.sin(18.0 * angle) * 40.0
        + np.cos(36.0 * angle) * 30.0
        + np.random.randn(size, size) * 20.0
    )
    img_ref[iris_mask] = np.clip(100.0 + pattern[iris_mask], 30, 220).astype(np.uint8)[:, None]
    img_ref[pupil_mask] = 15

    # Rotate by +3.0 degrees CCW
    target_deg = 3.0
    M = cv2.getRotationMatrix2D((cx, cy), target_deg, 1.0)
    img_curr = cv2.warpAffine(img_ref, M, (size, size), flags=cv2.INTER_CUBIC)

    det_ref = EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(cx, cy, r_pupil, r_pupil, 0.0),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(cx, cy, r_limbus, r_limbus, 0.0),
            confidence=0.95,
        ),
    )

    det_curr = EyeDetectionResult(
        pupil=PupilDetection(
            detected=True,
            ellipse=EllipseParams(cx, cy, r_pupil, r_pupil, 0.0),
            confidence=0.95,
        ),
        limbus=LimbusDetection(
            detected=True,
            ellipse=EllipseParams(cx, cy, r_limbus, r_limbus, 0.0),
            confidence=0.95,
        ),
    )

    return img_ref, img_curr, det_ref, det_curr, target_deg


def test_iris_code_stream(synthetic_eye_pair):
    img_ref, img_curr, det_ref, det_curr, target_deg = synthetic_eye_pair
    stream = IrisCodeStream()
    res = stream.run(img_ref, img_curr, det_ref, det_curr)

    assert res.stream == StreamName.IRIS_CODE
    assert res.valid
    assert res.torsion_deg is not None
    assert abs(res.torsion_deg - target_deg) < 0.25
    assert res.confidence > 0.50
    assert res.processing_time_ms > 0.0


def test_polar_optical_flow_stream(synthetic_eye_pair):
    img_ref, img_curr, det_ref, det_curr, target_deg = synthetic_eye_pair
    stream = PolarOpticalFlowStream()
    res = stream.run(img_ref, img_curr, det_ref, det_curr)

    assert res.stream == StreamName.POLAR_OPTICAL_FLOW
    assert res.valid
    assert res.torsion_deg is not None
    assert abs(res.torsion_deg - target_deg) < 0.25
    assert res.confidence > 0.50


def test_engine_fast_cascade(synthetic_eye_pair):
    img_ref, img_curr, det_ref, det_curr, target_deg = synthetic_eye_pair
    engine = RegistrationEngine()

    res_cascade = engine.register(
        img_ref, img_curr, det_ref, det_curr, fast_cascade=True
    )
    assert res_cascade.valid
    assert abs(res_cascade.torsion_deg - target_deg) < 0.25


def test_kalman_filter_smoothing():
    kf = CyclotorsionKalmanFilter()
    angle, vel, inlier = kf.update(3.0, 0.9, timestamp=0.0)
    assert abs(angle - 3.0) < 0.1
    assert inlier

    # Outlier rejection
    angle2, vel2, inlier2 = kf.update(45.0, 0.9, timestamp=0.033)
    assert not inlier2
    assert abs(angle2 - 3.0) < 1.0  # Should not jump to 45
