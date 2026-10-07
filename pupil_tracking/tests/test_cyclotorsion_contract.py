"""Geometry and replaceable matcher regressions, without private data."""
import numpy as np
import pytest

from pupil_tracking.pentacam.matcher import checked_match
from pupil_tracking.pentacam.sitting import AngularMatch, valid_geometry
from pupil_tracking.registration.polar import PolarUnwrapper
from pupil_tracking.tests.test_cross_modality_registration import make_elita_detection
from pupil_tracking.tests.test_pentacam_detector import create_synthetic_pentacam_image
from pupil_tracking.pentacam.session import SittingRegistrationSession
from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.sitting import masked_angular_match
from pupil_tracking.tests.test_sitting_registration import strip


@pytest.mark.parametrize("rotation", [0, 25, 90, 155])
@pytest.mark.parametrize("offset", [(0, 0), (10, 5), (-12, 8)])
def test_decentered_rays_satisfy_actual_ellipse(rotation, offset):
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
    cx, cy = 100 + offset[0], 100 + offset[1]
    radii = PolarUnwrapper._ellipse_radii_at_angles(
        (cx, cy), (50, 30), rotation, (100, 100), angles)
    dx, dy = 100 + radii*np.cos(angles)-cx, 100 + radii*np.sin(angles)-cy
    t = np.deg2rad(rotation)
    x, y = dx*np.cos(t)+dy*np.sin(t), -dx*np.sin(t)+dy*np.cos(t)
    np.testing.assert_allclose((x/50)**2+(y/30)**2, 1, atol=1e-12)


def test_disjoint_or_crossing_pupil_is_not_valid_geometry():
    det = make_elita_detection((256, 256))
    pupil, limbus = det.pupil.ellipse, det.limbus.ellipse
    assert valid_geometry(pupil, limbus, (512, 512))
    pupil.center_x += 120
    assert not valid_geometry(pupil, limbus, (512, 512))


def test_invalid_input_type_and_nonfinite_masks_are_rejected():
    with pytest.raises(ValueError, match='numpy'):
        CrossModalityRegistrationEngine().register([], [])
    ref = strip()
    mask = np.ones_like(ref)
    mask[0, 0] = np.inf
    assert masked_angular_match(ref, ref, mask).reason == 'nonfinite_mask'


@pytest.mark.parametrize("angle,score,overlap", [(float('nan'), .9, .8),
    (float('inf'), .9, .8), (31, .9, .8), (3, 1.1, .8), (3, .9, 1.1)])
def test_bad_backend_cannot_publish_angle(angle, score, overlap):
    def backend(*args, **kwargs):
        return AngularMatch(angle_deg=angle, valid=True, score=score, overlap=overlap)
    strip = np.ones((64, 360), np.float32)
    result = checked_match(backend, strip, strip, strip, strip, 30, reference_cache={})
    assert not result.valid and result.reason == 'invalid_backend_measurement'


def test_session_supports_injected_model_without_replacing_geometry():
    seen = []
    def backend(ref, curr, mask_ref, mask_curr, max_degrees, *, reference_cache):
        seen.append((ref.shape, mask_curr.shape))
        return AngularMatch(angle_deg=2.5, valid=True, score=.8, overlap=.8, reason='')
    session = SittingRegistrationSession(matcher=backend)
    image = create_synthetic_pentacam_image(with_ui=False)
    session.set_reference(image, eye_id='synthetic', laterality='OD')
    result = session.register(image, eye_id='synthetic', laterality='OD')
    assert result.valid and result.rotation_deg == 2.5
    assert seen == [((64, 720), (64, 720))]


@pytest.mark.parametrize('seed', [7, 19, 101])
@pytest.mark.parametrize('angle', [-12.25, -3.25, .25, 5.75, 12.25])
@pytest.mark.parametrize('noise', [0, 3])
def test_rotation_under_noise_illumination_and_occlusion(seed, angle, noise):
    ref = strip(seed)
    frequencies = np.fft.rfftfreq(ref.shape[1])
    curr = np.fft.irfft(np.fft.rfft(ref, axis=1) *
        np.exp(2j*np.pi*frequencies*angle*2), n=ref.shape[1], axis=1)
    curr = curr*.7+25+np.random.default_rng(seed).normal(0, noise, ref.shape)
    mr, mc = np.ones_like(ref, np.uint8), np.ones_like(ref, np.uint8)
    mr[:, 80:190], mc[:, 400:510] = 0, 0
    result = masked_angular_match(ref, curr, mr, mc)
    assert result.valid
    assert abs(result.angle_deg-angle) < .15
