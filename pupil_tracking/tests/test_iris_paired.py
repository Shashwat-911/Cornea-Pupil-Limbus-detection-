"""Tests for the Phase IV synthetic-pair generator.

These tests are deterministic and synthetic; they validate the pair-generation
library in ``pupil_tracking/iris/paired.py`` (pure functions) rather than the
clinical/ML pipeline. Assertions pin down the controlled semantics the
correspondence/rotation-recovery evaluation depends on: exact ground-truth
recording, determinism, source immutability, and IMAGE-B-frame perturbation
masks.
"""

import numpy as np
import pytest

from pupil_tracking.iris.paired import (
    VALID_PERTURBATIONS,
    PairConfig,
    SyntheticPair,
    make_synthetic_pair,
)


# ── fixtures ───────────────────────────────────────────────────────────

def _synthetic_iris_bgr(size=200, rng_seed=7, texture=12.0):
    """Synthetic textured 'iris' test image (BGR)."""
    yy, xx = np.mgrid[0:size, 0:size]
    rad = np.sqrt((xx - size / 2.0) ** 2 + (yy - size / 2.0) ** 2)
    iris = (60 + 120 * (np.sin(rad * 0.5))).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(rng_seed)
    noise = rng.integers(-int(texture), int(texture), iris.shape).astype(np.int16)
    gray = np.clip(iris.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


# ── identity / immutability ────────────────────────────────────────────

def test_identity_config_returns_identical_a_and_b():
    src = _synthetic_iris_bgr()
    p = make_synthetic_pair(src, PairConfig(), name="id")
    assert isinstance(p, SyntheticPair)
    assert np.array_equal(p.image_a, p.image_b)
    assert p.gt_rotation_deg == 0.0
    assert p.gt_scale == 1.0
    assert p.perturbation is None
    assert p.occlusion_mask is None


def test_source_array_is_never_modified():
    src = _synthetic_iris_bgr()
    before = src.copy()
    cfg = PairConfig(rotation_deg=5.0, scale=1.03, translation_px=(4, -2),
                     perturbation="noise", seed=42)
    p = make_synthetic_pair(src, cfg)
    assert np.array_equal(src, before), "caller's array must be untouched"
    assert np.array_equal(p.image_a, before), "image_a must equal the source content"
    assert not np.shares_memory(p.image_a, src)


# ── ground truth recording ─────────────────────────────────────────────

def test_gt_rotation_scale_translation_recorded():
    src = _synthetic_iris_bgr()
    cfg = PairConfig(rotation_deg=-12.5, scale=0.97, translation_px=(3.0, 4.0),
                     center=(150.0, 50.0), seed=1)
    p = make_synthetic_pair(src, cfg, name="gt")
    assert p.gt_rotation_deg == pytest.approx(-12.5)
    assert p.gt_scale == pytest.approx(0.97)
    assert p.gt_translation_px == pytest.approx((3.0, 4.0))
    assert p.rotation_center == pytest.approx((150.0, 50.0))
    assert p.image_b.shape == src.shape


def test_to_dict_records_transform_and_geometry():
    src = _synthetic_iris_bgr()
    cfg = PairConfig(rotation_deg=3.0, scale=1.05, translation_px=(0.0, 0.0),
                     perturbation="blur", perturbation_params={"ksize": 5.0}, seed=2)
    p = make_synthetic_pair(src, cfg, name="d")
    d = p.to_dict()
    assert d["name"] == "d"
    assert d["gt_rotation_deg"] == pytest.approx(3.0)
    assert d["gt_scale"] == pytest.approx(1.05)
    assert d["gt_translation_px"] == [0.0, 0.0]
    assert d["perturbation"] == "blur"
    assert d["perturbation_params"] == {"ksize": 5.0}
    assert d["image_a_shape"] == list(src.shape)


# ── deterministic warp behaviour ───────────────────────────────────────

def test_same_seed_same_config_produces_identical_pair():
    src = _synthetic_iris_bgr()
    cfg = PairConfig(rotation_deg=5.0, scale=1.03, translation_px=(4, 0),
                     perturbation="noise", perturbation_params={"sigma": 6.0}, seed=42)
    p1 = make_synthetic_pair(src, cfg, name="x")
    p2 = make_synthetic_pair(src, cfg, name="x")
    assert np.array_equal(p1.image_b, p2.image_b)
    assert np.array_equal(p1.occlusion_mask, p2.occlusion_mask)


def test_different_seed_changes_noise_pair():
    src = _synthetic_iris_bgr()
    a = make_synthetic_pair(src, PairConfig(rotation_deg=3.0, perturbation="noise",
                                            perturbation_params={"sigma": 6.0}, seed=10))
    b = make_synthetic_pair(src, PairConfig(rotation_deg=3.0, perturbation="noise",
                                            perturbation_params={"sigma": 6.0}, seed=11))
    assert not np.array_equal(a.image_b, b.image_b)


def test_rotation_changes_content():
    src = _synthetic_iris_bgr()
    p0 = make_synthetic_pair(src, PairConfig(rotation_deg=0.0))
    p5 = make_synthetic_pair(src, PairConfig(rotation_deg=5.0))
    assert not np.array_equal(p0.image_b, p5.image_b)


def test_integer_translation_is_exact_shift():
    src = _synthetic_iris_bgr()
    p = make_synthetic_pair(src, PairConfig(translation_px=(5, 0)), name="t")
    h, w = src.shape[:2]
    # dst(x) = src(x - 5) for integer translation without resampling.
    assert np.array_equal(p.image_b[:, 5:, :], src[:, : w - 5, :])
    assert not np.array_equal(p.image_b, src)


def test_scale_changes_content():
    src = _synthetic_iris_bgr()
    p0 = make_synthetic_pair(src, PairConfig(scale=1.0))
    p1 = make_synthetic_pair(src, PairConfig(scale=1.05))
    assert not np.array_equal(p0.image_b, p1.image_b)


def test_custom_rotation_center_changes_output():
    src = _synthetic_iris_bgr()
    default = make_synthetic_pair(src, PairConfig(rotation_deg=15.0))
    custom = make_synthetic_pair(src, PairConfig(rotation_deg=15.0, center=(150.0, 50.0)))
    assert not np.array_equal(default.image_b, custom.image_b)


# ── perturbations ──────────────────────────────────────────────────────

def test_occlusion_perturbation_returns_b_frame_mask():
    src = _synthetic_iris_bgr()
    cfg = PairConfig(rotation_deg=3.0, perturbation="occlusion",
                     perturbation_params={"radius": 30.0}, seed=11)
    p = make_synthetic_pair(src, cfg, name="occ")
    m = p.occlusion_mask
    assert m is not None
    assert m.shape == src.shape[:2]
    assert m.dtype == bool
    assert m.any() and not m.all()
    assert not np.array_equal(p.image_b, p.image_a)


def test_reflection_perturbation_adds_saturated_pixels_no_mask():
    # The synthetic iris texture never reaches 255 (base ~180 + noise ~12), so a
    # white specular disc is observable as new fully-saturated pixels in B.
    src = _synthetic_iris_bgr()
    p = make_synthetic_pair(src, PairConfig(rotation_deg=-3.0, perturbation="reflection",
                                            perturbation_params={"radius": 15.0}, seed=11))
    assert p.occlusion_mask is None
    assert not np.array_equal(p.image_b, p.image_a)
    assert not (p.image_a == 255).any()
    assert (p.image_b == 255).any()


def test_all_valid_perturbations_accepted():
    src = _synthetic_iris_bgr()
    for kind in VALID_PERTURBATIONS:
        p = make_synthetic_pair(src, PairConfig(rotation_deg=1.0, perturbation=kind,
                                                perturbation_params={}, seed=5))
        assert p.image_b.shape == src.shape
        assert p.image_b.dtype == src.dtype


def test_perturbation_params_are_merged_and_recorded():
    src = _synthetic_iris_bgr()
    p = make_synthetic_pair(src, PairConfig(perturbation="noise",
                                            perturbation_params={"sigma": 10.0}, seed=0))
    # default sigma is 6.0; recorded params must be the caller's.
    assert p.perturbation_params == {"sigma": 10.0}
    assert p.to_dict()["perturbation_params"] == {"sigma": 10.0}


# ── error handling ─────────────────────────────────────────────────────

def test_invalid_perturbation_raises():
    src = _synthetic_iris_bgr()
    with pytest.raises(ValueError):
        make_synthetic_pair(src, PairConfig(perturbation="bogus"))


def test_empty_source_raises():
    with pytest.raises(ValueError):
        make_synthetic_pair(np.empty((0, 0, 3), np.uint8), PairConfig())
    with pytest.raises(ValueError):
        make_synthetic_pair(None, PairConfig())


def test_config_is_immutable_frozen():
    with pytest.raises(Exception):
        PairConfig(rotation_deg=1.0).rotation_deg = 2.0