"""Numerical/rejection tests, independent of private patient files."""
import cv2
import numpy as np
import pytest

from pupil_tracking.pentacam.sitting import masked_angular_match, ocular_viewport
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.session import SittingRegistrationSession
from pupil_tracking.tests.test_pentacam_detector import create_synthetic_pentacam_image
from pupil_tracking.tests.test_cross_modality_registration import make_elita_detection, rotate_image


def strip(seed=5):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.normal(120, 30, (48, 720)).astype(np.float32), (5, 3), 0)


@pytest.mark.parametrize('rows', [8, 49, 64])
def test_reference_spectrum_cache_tracks_content_and_mask(rows):
    rng = np.random.default_rng(819)
    a = rng.normal(100, 20, (rows, 720)).astype(np.float32)
    b = np.roll(a, 7, axis=1)
    mask = np.ones_like(a, dtype=np.uint8)
    cache = {}
    for change in ('initial', 'reuse', 'texture', 'mask'):
        if change == 'texture':
            a[:] = np.roll(a, 3, axis=1)
        elif change == 'mask':
            mask[:, 20:240] = 0
        cached = masked_angular_match(a, b, mask, reference_cache=cache)
        fresh = masked_angular_match(a, b, mask)
        assert cached == fresh
        assert cached.valid
        assert len(cache) == 2


@pytest.mark.parametrize("angle", [-15.25, -4.75, -.25, 0., .25, 4.75, 15.25])
def test_masked_fractional_shift_and_reverse(angle):
    a = strip()
    freq = np.fft.rfftfreq(a.shape[1])
    b = np.fft.irfft(np.fft.rfft(a, axis=1) * np.exp(2j * np.pi * freq * angle * 2),
                     n=a.shape[1], axis=1).astype(np.float32)
    ma, mb = np.ones_like(a, np.uint8) * 255, np.ones_like(a, np.uint8) * 255
    ma[:, 70:140] = 0
    mb[:, 380:510] = 0
    a[ma == 0], b[mb == 0] = 255, 0
    result = masked_angular_match(a, b * .75 + 20, ma, mb)
    reverse = masked_angular_match(b, a, mb, ma)
    assert result.valid and reverse.valid
    assert abs(result.angle_deg - angle) < .15
    assert abs(result.angle_deg + reverse.angle_deg) < .02


def test_refuse_blank_unrelated_periodic_and_inadequate_overlap():
    a = strip()
    assert not masked_angular_match(a, strip(81)).valid
    assert not masked_angular_match(np.ones_like(a), np.ones_like(a)).valid
    mask = np.zeros_like(a, np.uint8)
    mask[:, :100] = 255
    assert not masked_angular_match(a, a, mask, mask).valid
    periodic = np.tile(np.sin(np.arange(720) * 2 * np.pi / 20), (48, 1)).astype(np.float32)
    assert not masked_angular_match(periodic, periodic).valid
    assert not masked_angular_match(a, np.roll(a, 80, axis=1), max_degrees=15).valid
    b = a.copy()
    b[0, 0] = np.nan
    assert not masked_angular_match(a, b).valid


def test_radial_disagreement_is_rejected():
    a = strip()
    b = a.copy()
    b[:16] = np.roll(a[:16], 14, axis=1)
    assert not masked_angular_match(a, b).valid


@pytest.mark.parametrize("value", [0, 15, 128, 255])
def test_uniform_frame_never_invents_an_eye(value):
    r = PentacamIrisDetector().detect(np.full((512, 512), value, np.uint8))
    assert not r.valid
    assert not r.geometry.pupil_detected


def test_viewport_and_coordinate_mapping():
    eye = create_synthetic_pentacam_image(with_ui=False)
    screenshot = np.full((900, 1200), 255, np.uint8)
    screenshot[160:800, 20:660] = cv2.resize(eye, (640, 640))
    x, y, w, h = ocular_viewport(screenshot)
    assert abs(x - 20) <= 4 and abs(y - 160) <= 4
    r = PentacamIrisDetector().detect(screenshot)
    assert r.valid
    assert abs(r.geometry.pupil.center_x - 340) < 5
    assert abs(r.geometry.pupil.center_y - 480) < 5
    assert r.image_width == 1200


def test_invalid_frame_cannot_update_temporal_state_or_report_final_angle():
    engine = CrossModalityRegistrationEngine()
    ref = create_synthetic_pentacam_image(with_ui=False)
    det = make_elita_detection((256, 256))
    good = engine.register(ref, rotate_image(ref, 3, (256, 256)), elita_detection=det, mode="dynamic")
    assert good.valid
    previous = engine._last_smooth_theta
    bad = engine.register(ref, np.full_like(ref, 120), elita_detection=det, mode="dynamic")
    assert not bad.valid and bad.final_sitting_to_supine_deg is None
    assert bad.confidence == 0 and bad.torsion_direction == "UNAVAILABLE"
    assert engine._last_smooth_theta == previous


def test_reference_geometry_change_invalidates_polar_cache():
    engine = CrossModalityRegistrationEngine()
    ref = create_synthetic_pentacam_image(with_ui=False)
    result = engine.pentacam_detector.detect(ref)
    det = make_elita_detection((256, 256))
    engine.register(ref, ref, result, det)
    old = engine._polar_cache
    result.geometry.pupil.center_x += 2
    engine.register(ref, ref, result, det)
    assert engine._polar_cache is not old


def test_session_identity_and_reference_replacement():
    session = SittingRegistrationSession()
    image = create_synthetic_pentacam_image(with_ui=False)
    assert not session.register(image, eye_id="a", laterality="OD").valid
    session.set_reference(image, eye_id="a", laterality="OD")
    result = session.register(rotate_image(image, -4.5, (256, 256)), eye_id="a", laterality="OD")
    assert result.valid and abs(result.rotation_deg + 4.5) < .5
    assert not session.register(image, eye_id="b", laterality="OD").valid
    assert not session.register(image, eye_id="a", laterality="OS").valid
    with pytest.raises(ValueError):
        session.set_reference(np.zeros_like(image), eye_id="a", laterality="OD")
    assert not session.register(image, eye_id="a", laterality="OD").valid


def test_session_maps_full_resolution_geometry_and_does_not_mutate_input():
    session = SittingRegistrationSession(max_size=384)
    image = create_synthetic_pentacam_image(with_ui=False)
    original = image.copy()
    session.set_reference(image, eye_id="a", laterality="OS")
    det = make_elita_detection((256, 256))
    result = session.register(rotate_image(image, 4.5, (256, 256)),
                              eye_id="a", laterality="OS", detection=det)
    assert result.valid and abs(result.rotation_deg - 4.5) < .5
    assert det.pupil.ellipse.center_x == 256
    np.testing.assert_array_equal(image, original)


def test_invalid_external_geometry_and_input_types_are_rejected():
    engine = CrossModalityRegistrationEngine()
    image = create_synthetic_pentacam_image(with_ui=False)
    det = make_elita_detection((256, 256))
    det.pupil.ellipse.center_x = float("nan")
    result = engine.register(image, image, elita_detection=det)
    assert not result.valid and result.final_sitting_to_supine_deg is None
    with pytest.raises(ValueError):
        engine.register(image.astype(float), image)
    with pytest.raises(ValueError):
        PentacamIrisDetector().detect(image.astype(float))


def test_sitting_import_does_not_load_torch():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
        "import sys; from pupil_tracking.pentacam.session import SittingRegistrationSession; "
        "assert 'torch' not in sys.modules"], check=True)
