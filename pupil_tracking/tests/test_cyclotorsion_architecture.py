"""Unit and contract tests for cyclotorsion architecture, matchers, and training pipeline."""
import math
import numpy as np
import pytest

from pupil_tracking.pentacam.matcher import (
    BaseAngularMatcher,
    ClassicalFFTMatcher,
    LearnedAngularMatcher,
    EnsembleMatcher,
    create_matcher,
    checked_match,
)
from pupil_tracking.pentacam.sitting import AngularMatch, masked_angular_match
from pupil_tracking.pentacam.session import SittingRegistrationSession
from pupil_tracking.pentacam.training_data import CyclotorsionAugmentor
from pupil_tracking.tests.test_sitting_registration import strip
from pupil_tracking.tests.test_pentacam_detector import create_synthetic_pentacam_image
from pupil_tracking.tests.test_cross_modality_registration import rotate_image


def test_classical_matcher_conforms_to_protocol():
    matcher = ClassicalFFTMatcher()
    ref = strip(10)
    curr = np.roll(ref, 6, axis=1)
    res = matcher(ref, curr, np.ones_like(ref), np.ones_like(curr), 30.0, reference_cache={})
    assert isinstance(res, AngularMatch)
    assert res.valid
    assert matcher.total_calls == 1
    assert matcher.accepted_calls == 1
    assert matcher.acceptance_rate == 1.0
    assert matcher.average_latency_ms > 0


def test_learned_matcher_fallbacks_cleanly():
    # Model that rejects / is unconfident
    class LowConfidenceModel:
        def estimate_rotation(self, ref, curr, mr=None, mc=None, max_degrees=30.0, **kwargs):
            return AngularMatch(valid=False, score=0.1, reason="unconfident")

    ref = strip(12)
    curr = np.roll(ref, 8, axis=1)

    # 1. Fallback enabled -> delegates to classical matcher and succeeds
    learned_with_fallback = LearnedAngularMatcher(model=LowConfidenceModel(), enable_fallback=True)
    res = learned_with_fallback(ref, curr, None, None, 30.0, reference_cache={})
    assert res.valid
    assert abs(res.angle_deg - (-8 * 360 / 720)) < 0.2

    # 2. Fallback disabled -> returns model's rejection
    learned_no_fallback = LearnedAngularMatcher(model=LowConfidenceModel(), enable_fallback=False)
    res_no = learned_no_fallback(ref, curr, None, None, 30.0, reference_cache={})
    assert not res_no.valid
    assert res_no.reason == "unconfident"


def test_ensemble_matcher_consensus_and_disagreement():
    def matcher_a(ref, curr, mr, mc, max_deg, *, reference_cache):
        return AngularMatch(angle_deg=2.50, valid=True, score=0.8, overlap=0.9, peak_margin=0.2)

    def matcher_b_agree(ref, curr, mr, mc, max_deg, *, reference_cache):
        return AngularMatch(angle_deg=2.54, valid=True, score=0.85, overlap=0.9, peak_margin=0.2)

    def matcher_b_disagree(ref, curr, mr, mc, max_deg, *, reference_cache):
        return AngularMatch(angle_deg=5.10, valid=True, score=0.85, overlap=0.9, peak_margin=0.2)

    ref = strip(5)

    # 1. Consensus: within 1.0 deg tolerance
    ens_agree = EnsembleMatcher(matchers=[matcher_a, matcher_b_agree], max_discrepancy_deg=1.0)
    res = ens_agree(ref, ref, None, None, 30.0, reference_cache={})
    assert res.valid
    assert abs(res.angle_deg - 2.52) < 0.05
    assert res.band_spread_deg < 0.1

    # 2. Disagreement: > 1.0 deg tolerance
    ens_disagree = EnsembleMatcher(matchers=[matcher_a, matcher_b_disagree], max_discrepancy_deg=1.0)
    res_dis = ens_disagree(ref, ref, None, None, 30.0, reference_cache={})
    assert not res_dis.valid
    assert "ensemble_discrepancy" in res_dis.reason


def test_create_matcher_factory():
    c = create_matcher("classical")
    assert isinstance(c, ClassicalFFTMatcher)

    l = create_matcher("learned")
    assert isinstance(l, LearnedAngularMatcher)

    e = create_matcher("ensemble")
    assert isinstance(e, EnsembleMatcher)

    with pytest.raises(ValueError, match="Unknown matcher kind"):
        create_matcher("nonexistent")


def test_angular_match_fields_and_subpixel_bands():
    ref = strip(99)
    curr = np.roll(ref, 4, axis=1)
    res = masked_angular_match(ref, curr)
    assert res.valid
    assert hasattr(res, "band_angles")
    assert hasattr(res, "uncertainty_deg")
    assert len(res.band_angles) == 3
    assert all(math.isfinite(a) for a in res.band_angles)
    assert res.uncertainty_deg > 0
    # Continuous subpixel band spread should be tiny on rigid synthetic shift
    assert res.band_spread_deg < 0.10


def test_cyclotorsion_augmentor():
    aug = CyclotorsionAugmentor(max_delta_deg=5.0)
    ref = np.full((64, 360), 120, dtype=np.uint8)
    curr = ref.copy()
    mr = np.full((64, 360), 255, dtype=np.uint8)
    mc = np.full((64, 360), 255, dtype=np.uint8)

    ref_a, curr_a, mr_a, mc_a, rot_a = aug.augment_strip_pair(ref, curr, mr, mc, 0.0)
    assert ref_a.shape == (64, 360)
    assert curr_a.shape == (64, 360)
    assert mr_a.shape == (64, 360)
    assert mc_a.shape == (64, 360)
    assert math.isfinite(rot_a)


def test_iris_polar_models():
    try:
        import torch
        from pupil_tracking.pentacam.models import IrisPolarFeatureNet, IrisPolarSiameseNet
    except ImportError:
        pytest.skip("PyTorch not installed")

    fnet = IrisPolarFeatureNet(embedding_dim=32, base_channels=16)
    x = torch.randn(2, 1, 64, 360)
    emb = fnet(x)
    assert emb.shape == (2, 32, 360)

    model = IrisPolarSiameseNet(num_angles=360, embedding_dim=32)
    out = model(x, x)
    assert "pred_deg" in out
    assert "score" in out
    assert "peak_margin" in out
    assert out["pred_deg"].shape == (2,)

    # Test estimate_rotation
    strip_np = np.full((64, 360), 100, dtype=np.uint8)
    match_res = model.estimate_rotation(strip_np, strip_np, max_degrees=30.0)
    assert isinstance(match_res, AngularMatch)
