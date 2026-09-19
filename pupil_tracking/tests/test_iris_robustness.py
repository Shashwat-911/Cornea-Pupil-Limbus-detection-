"""Phase II tests: repeatability & robustness evaluation helpers.

These tests are deterministic and synthetic. They validate the *evaluation
library* in ``pupil_tracking/iris/robustness.py`` (pure functions) rather than
re-running the clinical/ML pipeline, which is covered by the smoke script.

Assertions are chosen to be non-tautological: they fix the controlled semantics
of the metrics (correspondence, spatial distribution, inverse mapping) so the
report produced by Phase II is itself trustworthy.
"""

import numpy as np
import pytest

from pupil_tracking.iris import robustness as R
from pupil_tracking.iris.types import (
    IrisFeature,
    IrisFeatureSet,
    IrisStatus,
    IrisROI,
    IrisFeatureType,
)


# ── helpers ────────────────────────────────────────────────────────────

def _feat(angle_deg, radial_norm, confidence=0.5, x=0.0, y=0.0, idx=0):
    return IrisFeature(
        id=idx, x=x, y=y, angle_deg=angle_deg, radial_norm=radial_norm,
        scale=1.0, orientation_deg=angle_deg, feature_type=IrisFeatureType.TEXTURE,
        response=10.0, local_contrast=0.1, visibility=1.0, confidence=confidence,
    )


def _set(feats):
    return IrisFeatureSet(
        roi=IrisROI(valid=True), features=list(feats), num_candidates=len(feats),
        num_accepted=len(feats),
    )


# ── angular_gap ────────────────────────────────────────────────────────

def test_angular_gap_wraps_circularly():
    assert R.angular_gap(0.0, 359.0) == pytest.approx(1.0)
    assert R.angular_gap(350.0, 10.0) == pytest.approx(20.0)
    assert R.angular_gap(100.0, 100.0) == pytest.approx(0.0)


# ── repeatability_metrics ──────────────────────────────────────────────

def test_repeatability_identical_sets_is_one():
    feats = [_feat(a, 0.5) for a in [0, 60, 120, 180, 240, 300]]
    m = R.repeatability_metrics(feats, feats)
    assert m["repeatability_rate"] == pytest.approx(1.0)
    assert m["matched_count"] == len(feats)


def test_repeatability_disjoint_sets_is_zero():
    a = [_feat(a, 0.5) for a in [0, 90, 180, 270]]
    b = [_feat(a + 45.0, 0.7) for a in [0, 90, 180, 270]]  # far apart
    m = R.repeatability_metrics(a, b, ang_tol_deg=3.0, rad_tol=0.06)
    assert m["repeatability_rate"] == 0.0
    assert m["matched_count"] == 0


def test_repeatability_count_ratio_honest():
    # 4 baseline, 8 identical copies -> retained > 1 but repeatability still 1.0
    a = [_feat(a, 0.5, confidence=0.6 + i * 0.01) for i, a in enumerate([0, 90, 180, 270])]
    b = [f for f in a] + [f for f in a]  # duplicate set
    m = R.repeatability_metrics(a, b)
    assert m["repeatability_rate"] == pytest.approx(1.0)
    assert m["retained_feature_rate"] == pytest.approx(2.0)


def test_repeatability_tolerance_is_respected():
    # Features within tolerance match; outside do not.
    a = [_feat(0.0, 0.5)]
    b = [_feat(2.5, 0.5)]      # within 3.0 deg tol
    assert R.repeatability_metrics(a, b)["matched_count"] == 1
    b2 = [_feat(3.5, 0.5)]     # just outside 3.0 deg tol
    assert R.repeatability_metrics(a, b2)["matched_count"] == 0


def test_repeatability_empty_base_is_zero():
    m = R.repeatability_metrics([], [_feat(0, 0.5)])
    assert m["repeatability_rate"] == 0.0
    assert m["base_count"] == 0
    assert m["pert_count"] == 1


def test_repeatability_quality_orders_matches_deterministically():
    # When two baseline features compete for one perturbed feature, the greedy
    # matcher picks the highest-confidence one; result is deterministic.
    a = [
        _feat(0.0, 0.5, confidence=0.3, idx=0),
        _feat(1.0, 0.5, confidence=0.9, idx=1),  # both within 3deg of perturb target
    ]
    b = [_feat(1.2, 0.5, confidence=0.8)]
    m1 = R.repeatability_metrics(a, b)
    m2 = R.repeatability_metrics(a, b)
    assert m1 == m2
    # exactly one baseline feature matched
    assert m1["matched_count"] == 1


# ── spatial_distribution ───────────────────────────────────────────────

def test_spatial_distribution_uniform_ring():
    feats = [_feat(a, 0.5) for a in range(0, 360, 15)]  # 24 evenly round
    d = R.spatial_distribution(_set(feats))
    assert d["angular_coverage"] == pytest.approx(1.0)
    assert d["n_features"] == 24


def test_spatial_distribution_clumped_is_detected():
    feats = [_feat(5.0, 0.5) for _ in range(20)]  # all in one tiny angular band
    d = R.spatial_distribution(_set(feats))
    assert d["angular_coverage"] < 0.5
    assert d["concentration"] == pytest.approx(1.0)
    assert d["quadrant_count"] == 1


def test_spatial_distribution_entropy_uniform_is_max():
    feats = [_feat(a, 0.5) for a in range(0, 360, 30)]  # 12 -> 1 per sector
    d = R.spatial_distribution(_set(feats))
    assert d["angular_entropy"] == pytest.approx(1.0, abs=0.05)


def test_spatial_distribution_empty():
    d = R.spatial_distribution(_set([]))
    assert d["n_features"] == 0
    assert d["cell_coverage"] == 0.0


# ── baseline_statistics ────────────────────────────────────────────────

def test_baseline_statistics_counts_are_honest():
    from pupil_tracking.iris.types import IrisDetectionResult

    fs = _set([_feat(a, 0.5, confidence=0.4) for a in range(0, 360, 30)])
    fs.num_candidates = 25  # 13 rejected
    res = IrisDetectionResult(
        valid=True, status=IrisStatus.OK, feature_set=fs,
        processing_time_ms=12.3,
    )
    s = R.baseline_statistics(res)
    assert s["accepted"] == 12
    assert s["candidates"] == 25
    assert s["rejected"] == 13
    assert s["mean_quality"] == pytest.approx(0.4)


# ── map_point_back (inverse geometric mapping) ─────────────────────────

def test_map_point_back_translate_round_trip():
    x, y = 100.0, 200.0
    bx, by = R.map_point_back(x + 7, y - 4, "translate", (7, -4), 400, 400)
    assert (bx, by) == (pytest.approx(100.0), pytest.approx(200.0))


def test_map_point_back_rotate_round_trip():
    x, y = 150.0, 90.0
    deg = 5.0
    # forward rotation about image centre (400x400 -> centre 200,200)
    import math
    cx = cy = 200.0
    rad = math.radians(deg)
    ox, oy = x - cx, y - cy
    xp = cx + ox * math.cos(rad) - oy * math.sin(rad)
    yp = cy + ox * math.sin(rad) + oy * math.cos(rad)
    bx, by = R.map_point_back(xp, yp, "rotate", deg, 400, 400)
    assert bx == pytest.approx(x, abs=1e-6)
    assert by == pytest.approx(y, abs=1e-6)


def test_map_point_back_scale_round_trip():
    x, y = 180.0, 210.0
    f = 1.04
    new_w = int(round(400 * f)); new_h = int(round(400 * f))
    sx, sy = new_w / 400.0, new_h / 400.0
    tx, ty = (400 - new_w) / 2.0, (400 - new_h) / 2.0
    xp = x * sx + tx
    yp = y * sy + ty
    bx, by = R.map_point_back(xp, yp, "scale", f, 400, 400)
    assert bx == pytest.approx(x, abs=1e-6)
    assert by == pytest.approx(y, abs=1e-6)


# ── perturbations are deterministic ────────────────────────────────────

def test_perturbations_are_deterministic_per_seed():
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    for kind, value in [
        ("brightness", 25), ("contrast", 1.2), ("gamma", 0.8),
        ("noise", 3.0), ("blur", 5), ("sharpen", 0.4),
        ("translate", (5, 3)), ("rotate", 2.0), ("scale", 1.02),
    ]:
        a = R.PERTURBATIONS[kind](img, value, seed=7)
        b = R.PERTURBATIONS[kind](img, value, seed=7)
        assert np.array_equal(a, b), kind


def test_perturbation_output_shape_preserved():
    img = np.random.default_rng(1).integers(0, 255, (64, 64, 3)).astype(np.uint8)
    for kind, value in [
        ("brightness", 40), ("contrast", 0.6), ("gamma", 1.4),
        ("noise", 8.0), ("blur", 9), ("sharpen", 0.7),
        ("translate", (8, 8)), ("rotate", 5.0), ("scale", 0.95),
    ]:
        out = R.PERTURBATIONS[kind](img, value, seed=3)
        assert out.shape == img.shape, kind


# ── quality_stability_correlation ──────────────────────────────────────

def test_quality_stability_detects_positive_relationship():
    base = [
        _feat(a, 0.5, confidence=c)
        for a, c in zip(range(0, 360, 20), np.linspace(0.2, 0.95, 18))
    ]
    # perturbed set keeps only the highest-confidence baseline features -> strong
    # positive correlation between confidence and retention.
    keep = [f for f in base if f.confidence >= 0.6]
    q = R.quality_stability_correlation(base, keep)
    assert q["defined"] is True
    assert q["spearman_rho"] > 0.5


def test_quality_stability_undefined_when_all_retained():
    base = [_feat(a, 0.5, confidence=0.4 + i * 0.01) for i, a in enumerate(range(0, 360, 30))]
    q = R.quality_stability_correlation(base, list(base))
    assert q["defined"] is False
    assert q["spearman_rho"] == 0.0
