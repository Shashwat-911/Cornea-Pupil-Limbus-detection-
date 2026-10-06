"""Known-boundary tests separate from digital torsion validation."""
import cv2
import numpy as np
import pytest

from pupil_tracking.pentacam.refinement import refine_limbus, anatomy_candidates
from pupil_tracking.pentacam.types import PentacamGeometry
from pupil_tracking.utils.types import EllipseParams


@pytest.mark.parametrize('angle', [0, 15, 45, 90])
def test_decentered_ellipse_boundary_with_superior_occlusion(angle):
    image = np.full((512, 512), 180, np.uint8)
    cv2.ellipse(image, (269, 261), (155, 145), angle, 0, 360, 80, -1)
    cv2.circle(image, (252, 249), 55, 15, -1)
    # Superior lid edge must not pull the fit toward a pupil-centred circle.
    image[:155] = 145
    seed = EllipseParams(center_x=252, center_y=249, semi_major=150, semi_minor=150)
    result = refine_limbus(image, np.full_like(image, 255), seed)
    assert result is not None
    fit, points, residual = result
    assert np.hypot(fit.center_x-269, fit.center_y-261) < 3
    assert abs(fit.semi_major-155) < 3 and abs(fit.semi_minor-145) < 3
    assert residual < 2
    assert len(points) >= 45
    assert points[:, 1].min() > 155


def test_no_boundary_evidence_does_not_invent_refinement():
    seed = EllipseParams(center_x=256, center_y=256, semi_major=150, semi_minor=150)
    assert refine_limbus(np.full((512, 512), 100, np.uint8),
                         np.full((512, 512), 255, np.uint8), seed) is None


def test_anatomy_hypotheses_exclude_mask_and_never_claim_verified():
    image = np.full((256, 256), 120, np.uint8)
    cv2.circle(image, (190, 130), 6, 50, -1)
    geom = PentacamGeometry(pupil=EllipseParams(center_x=128, center_y=128,
        semi_major=30, semi_minor=30), limbus=EllipseParams(center_x=128,
        center_y=128, semi_major=100, semi_minor=100))
    mask = np.full_like(image, 255)
    candidates = anatomy_candidates(image, geom, mask)
    assert candidates
    assert all(not c.anatomy_verified and c.confidence == 0 for c in candidates)
    mask[:, 170:] = 0
    assert not anatomy_candidates(image, geom, mask)
