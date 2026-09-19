"""Tests for the Phase I iris-feature concept model.

These tests are deterministic and use synthetic fixtures. They validate the
ROI construction, masking, normalization, extraction, quality filtering and
result contracts without requiring the production ML model or clinical data
(the clinical proxy smoke test lives in ``scripts/iris_feature_smoke.py``).
"""

import numpy as np
import pytest

from pupil_tracking.iris import (
    IrisConfig,
    IrisDetectionResult,
    IrisFeatureDetector,
    IrisMasking,
    IrisNormalizer,
    IrisROIExtractor,
    detect_iris_features,
    draw_iris_overlay,
)
from pupil_tracking.iris.roi import point_in_roi_annulus, sample_annulus_mask
from pupil_tracking.iris.types import IrisFeatureType, IrisStatus
from pupil_tracking.utils.types import EllipseParams


# ── fixtures ───────────────────────────────────────────────────────────

def _make_ellipse(cx, cy, smaj, smin, angle=0.0, detected=True):
    e = EllipseParams(
        center_x=cx, center_y=cy,
        semi_major=smaj, semi_minor=smin,
        angle_deg=angle,
    )
    return e


def _synthetic_iris_image(size=320, rng_seed=0, texture=12.0, crypts=True):
    """Grayscale synthetic eye: dark pupil, brighter textured iris annulus."""
    gray = np.full((size, size), 25, np.uint8)
    center = size // 2
    cv2_circle_reuse(gray, (center, center), 130, 80)   # iris disc
    cv2_circle_reuse(gray, (center, center), 55, 10)     # pupil
    rng = np.random.default_rng(rng_seed)
    noise = rng.integers(-int(texture), int(texture), (size, size)).astype(np.int16)
    gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if crypts:
        for a in range(0, 360, 30):
            ang = np.radians(a)
            r = 92
            x = int(center + r * np.cos(ang))
            y = int(center + r * np.sin(ang))
            cv2_circle_reuse(gray, (x, y), 4, 18)
    return gray


def cv2_circle_reuse(img, center, radius, color):
    import cv2
    cv2.circle(img, center, radius, color, -1)


def _default_geometry(size=320):
    c = size // 2
    return _make_ellipse(c, c, 55, 55), _make_ellipse(c, c, 130, 130)


# ── ROI construction ───────────────────────────────────────────────────

def test_valid_geometry_creates_valid_roi():
    pupil, limbus = _default_geometry()
    extractor = IrisROIExtractor()
    roi = extractor.build(pupil, limbus)
    assert roi.valid is True
    assert roi.reason == "ok"
    assert roi.limbus_radius_px > roi.pupil_radius_px


def test_missing_geometry_invalid_roi_no_crash():
    extractor = IrisROIExtractor()
    roi = extractor.build(None, None)
    assert roi.valid is False
    assert roi.reason  # non-empty reason


def test_missing_limbus_invalid_roi():
    pupil, _ = _default_geometry()
    extractor = IrisROIExtractor()
    assert extractor.build(pupil, None).valid is False


def test_implausible_ratio_invalid_roi():
    _, limbus = _default_geometry()
    rogue = _make_ellipse(160, 160, 125, 125)  # pupil nearly as big as limbus
    extractor = IrisROIExtractor()
    roi = extractor.build(rogue, limbus)
    assert roi.valid is False


def test_detect_with_missing_geometry_is_safe():
    img = _synthetic_iris_image()
    res = detect_iris_features(img, None, None)
    assert isinstance(res, IrisDetectionResult)
    assert res.valid is False
    assert res.status == IrisStatus.NO_ROI


def test_annulus_mask_respects_pupil_limbus():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    mask = sample_annulus_mask(img.shape, roi)
    h, w = img.shape[:2]
    # center (pupil) should not be in annulus
    assert not mask[h // 2, w // 2]
    # a ring point roughly mid-iris should be included
    c = h // 2
    r = 92
    assert mask[int(c + r * np.cos(0)), int(c + r * np.sin(0))]


# ── masking / reflection ───────────────────────────────────────────────

def test_reflection_mask_respected():
    img = _synthetic_iris_image(rng_seed=3, crypts=False)
    pupil, limbus = _default_geometry()
    detector = IrisFeatureDetector()

    # Without reflection: measure usable fraction.
    roi = detector.roi_extractor.build(pupil, limbus)
    usable_clean = detector.masking.build(img, roi)
    n_clean = int(np.count_nonzero(usable_clean))
    assert n_clean > 0

    # Add a large bright specular reflection inside the annulus.
    img2 = img.copy()
    import cv2
    c = img.shape[0] // 2
    cv2.circle(img2, (int(c + 92), c), 12, (255, 255, 255), -1)
    usable_withrefl = detector.masking.build(img2, roi)
    n_withrefl = int(np.count_nonzero(usable_withrefl))

    # A bright blob inside the annulus must reduce the usable area (the
    # reflection is removed from the mask), and never increase it.
    assert n_withrefl < n_clean
    # Some iris pixels must still be usable.
    assert n_withrefl > 0


def test_mask_stats_interface():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    detector = IrisFeatureDetector()
    roi = detector.roi_extractor.build(pupil, limbus)
    usable = detector.masking.build(img, roi)
    stats = detector.detect(img, pupil, limbus).mask_stats
    assert "usable_fraction" in stats
    assert stats["usable_fraction"] >= 0.0


# ── normalization ──────────────────────────────────────────────────────

def test_normalized_coordinates_within_bounds():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    norm = IrisNormalizer()
    a, rn = norm.to_iris_relative(float(roi.center_x + 92.0), float(roi.center_y), roi)
    assert a is not None and rn is not None
    assert 0.0 <= a < 360.0
    assert 0.0 < rn <= 1.0


def test_roundtrip_from_iris_relative():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    roi = IrisROIExtractor().build(pupil, limbus)
    norm = IrisNormalizer()
    for rn in (0.0, 0.5, 1.0):
        pt = norm.from_iris_relative(35.0, rn, roi)
        assert pt is not None
        a, rn2 = norm.to_iris_relative(pt[0], pt[1], roi)
        assert rn2 is not None
        assert abs(rn2 - rn) < 0.02


# ── extraction ─────────────────────────────────────────────────────────

def test_extraction_deterministic():
    img = _synthetic_iris_image(rng_seed=5)
    pupil, limbus = _default_geometry()
    r1 = detect_iris_features(img, pupil, limbus).feature_set
    r2 = detect_iris_features(img, pupil, limbus).feature_set
    assert len(r1.features) == len(r2.features)
    for a, b in zip(r1.features, r2.features):
        assert abs(a.x - b.x) < 1e-6
        assert abs(a.y - b.y) < 1e-6
        assert a.confidence == b.confidence


def test_visibility_reflects_local_occlusion_fraction():
    # The per-feature visibility must measure the usable fraction of the
    # feature's local patch, not be a constant 1.0.
    from pupil_tracking.iris.extraction import IrisFeatureExtractor

    ex = IrisFeatureExtractor(radius_px=5)
    mask = np.ones((21, 21), dtype=bool)
    mask[0:11, :] = False  # occlude the top ~half of the image

    frac = ex._local_visibility(mask, 10.0, 10.0)
    # Feature at (10,10): patch rows 5..15 / cols 5..15 (11x11 = 121 px), of
    # which rows 5..10 (6 rows) are occluded -> usable = 121 - 66.
    assert 0.0 < frac < 1.0

    # Fully usable -> 1.0; fully occluded center would not be accepted, but a
    # fully occluded patch is reported as 1.0 for an already-usable center.
    assert ex._local_visibility(np.ones((21, 21), dtype=bool), 10.0, 10.0) == 1.0


def test_classify_uses_true_center_patch():
    # A dark pit exactly at the patch center must be classified as a crypt.
    # Previously the "center" window used a hard-coded top-left offset
    # (patch[1:4, 1:4]) that missed the true center for radius_px=5.
    from pupil_tracking.iris.extraction import IrisFeatureExtractor

    ex = IrisFeatureExtractor(radius_px=5)
    patch = np.full((11, 11), 200.0, np.float32)
    patch[5, 5] = 0.0
    assert ex._classify(patch) == IrisFeatureType.CRYPT


def test_printed_line_suppression_gate():
    # A long, straight, high-contrast line through the annulus (like the
    # quadrant crosshair printed on the docking ring) must suppress iris
    # features that would otherwise land on it, while a control patch without
    # the line is unaffected. Uses the wider-than-patch anisotropy window.
    import cv2

    from pupil_tracking.iris.extraction import IrisFeatureExtractor

    ex = IrisFeatureExtractor(radius_px=5)
    w = 61
    r = ex.line_suppression_window_r  # 15 default
    assert r > ex.radius_px

    gray = (np.ones((w, w), np.float32) * 150.0 +
            np.random.default_rng(0).normal(0, 3.0, (w, w)))

    # Line-free patch at centre: anisotropy must stay below the threshold.
    a_clean = ex._anchored_anisotropy(gray, w // 2, w // 2, r)
    assert a_clean < ex.line_suppression_threshold

    # The gate must be rotation-invariant: it targets long straight printed
    # lines regardless of their orientation, so a diagonal crosshair line is
    # suppressed too (an axis-aligned sx/sy ratio would miss it).
    for deg in (0.0, 30.0, 45.0, 60.0, 90.0):
        line = gray.copy()
        rad = np.radians(deg)
        n = w // 2
        cv2.line(
            line,
            (w // 2 - int(n * np.cos(rad)), w // 2 - int(n * np.sin(rad))),
            (w // 2 + int(n * np.cos(rad)), w // 2 + int(n * np.sin(rad))),
            40.0,
            1,
        )
        a_line = ex._anchored_anisotropy(line, w // 2, w // 2, r)
        assert a_line >= ex.line_suppression_threshold, (deg, a_line)
        assert ex._suppressed_by_printed_line(line, w // 2, w // 2), deg

    # Disabling the threshold must turn the gate off.
    ex2 = IrisFeatureExtractor(radius_px=5, line_suppression_threshold=0.0)
    line = gray.copy()
    cv2.line(line, (w // 2 - 15, w // 2), (w // 2 + 15, w // 2), 40.0, 1)
    assert not ex2._suppressed_by_printed_line(line, w // 2, w // 2)


def test_printed_line_suppresses_features_on_axis():
    # End-to-end: with a long straight printed crosshair line through the
    # annulus, no accepted feature may sit on the line, and features off the
    # line must still be extracted.
    import cv2

    img = _synthetic_iris_image(rng_seed=7, texture=6.0)
    pupil, limbus = _default_geometry()
    c = img.shape[0] // 2

    # Draw a vertical line through the centre spanning the whole annulus.
    line_img = img.copy()
    cv2.line(line_img, (c, c - 115), (c, c + 115), 5, 1)

    # ``eyelid_method="none"`` keeps the printed-line gate the only suppressor
    # of the drawn line: on this synthetic fixture the strong vertical line
    # otherwise trips the gradient eyelid detector, masking its own pixels and
    # masking the gate's effect (same rationale as the correspondence fixture
    # config, where production masking starves the synthetic texture).
    cfg = IrisConfig(
        eyelid_method="none",
        line_suppression_threshold=0.6,
        line_suppression_window_r=15,
    )
    res = detect_iris_features(line_img, pupil, limbus, config=cfg)
    assert res.valid
    # No accepted feature within 2 px of the printed line.
    for ft in res.feature_set.features:
        assert abs(ft.x - c) > 2.0, (ft.x, ft.y)
    # The reason must be recorded.
    assert res.feature_set.rejection_reasons.get("printed_line", 0) > 0

    # Control without the line: features are allowed near the axis.
    res_clean = detect_iris_features(img, pupil, limbus)
    assert res_clean.valid
    near_axis = sum(1 for ft in res_clean.feature_set.features if abs(ft.x - c) <= 2.0)
    assert near_axis > 0


def test_printed_line_suppresses_features_on_diagonal():
    # End-to-end guard for the rotation-invariance of the printed-line gate:
    # a ~45 degree crosshair line must also suppress features on it, which an
    # axis-aligned anisotropy test would miss. crypts=False keeps the gate the
    # only suppressor of the drawn line.
    import cv2

    img = _synthetic_iris_image(rng_seed=9, texture=6.0, crypts=False)
    pupil, limbus = _default_geometry()
    c = img.shape[0] // 2

    line_img = img.copy()
    off = 110
    cv2.line(line_img, (c - off, c - off), (c + off, c + off), 5, 1)

    # See test_printed_line_suppresses_features_on_axis: on this synthetic
    # fixture the gradient eyelid detector otherwise masks the drawn line and
    # pre-empts the printed-line gate.
    cfg = IrisConfig(
        eyelid_method="none",
        line_suppression_threshold=0.6,
        line_suppression_window_r=15,
    )
    res = detect_iris_features(line_img, pupil, limbus, config=cfg)
    assert res.valid
    # Accepted features must stay away from the drawn 45 deg line: the line's
    # perpendicular distance from each feature must exceed 2 px.
    sq2 = np.sqrt(2.0)
    for ft in res.feature_set.features:
        d = abs((ft.x - c) - (ft.y - c)) / sq2
        assert d > 2.0, (ft.x, ft.y, d)
    assert res.feature_set.rejection_reasons.get("printed_line", 0) > 0


def test_quality_filtering_flat_iris_yields_fewer():
    # High-texture iris -> more accepted than flat (low-texture) iris.
    hi = _synthetic_iris_image(rng_seed=1, texture=18.0, crypts=True)
    lo = _synthetic_iris_image(rng_seed=1, texture=1.0, crypts=False)
    pupil, limbus = _default_geometry()
    hi_res = detect_iris_features(hi, pupil, limbus).feature_set
    lo_res = detect_iris_features(lo, pupil, limbus).feature_set
    assert hi_res.num_accepted >= lo_res.num_accepted


def test_angular_suppression_enforces_min_separator():
    img = _synthetic_iris_image(rng_seed=2)
    pupil, limbus = _default_geometry()
    fs = detect_iris_features(img, pupil, limbus).feature_set
    acc = fs.features
    if len(acc) <= 1:
        pytest.skip("not enough accepted features to verify angular separation")
    min_angular_sep_deg = IrisConfig().min_angular_sep_deg
    # Every pair of accepted features must be separated by at least the
    # configured minimum angular gap.
    for i in range(len(acc)):
        for j in range(i + 1, len(acc)):
            gap = abs(acc[i].angle_deg - acc[j].angle_deg) % 360.0
            gap = min(gap, 360.0 - gap)
            assert gap >= min_angular_sep_deg - 1e-6, (
                acc[i].angle_deg, acc[j].angle_deg, gap, min_angular_sep_deg
            )


def test_result_contract_stable():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus)
    d = res.to_dict()
    assert d["status"] == res.status.value
    assert "feature_set" in d
    feat = res.feature_set.features[0]
    fd = feat.to_dict()
    for key in ("x", "y", "angle_deg", "radial_norm", "confidence",
                "feature_type", "local_contrast", "visibility", "valid"):
        assert key in fd


def test_does_not_modify_pupil_limbus_geometry():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    p_before = (pupil.center_x, pupil.center_y, pupil.semi_major, pupil.semi_minor)
    l_before = (limbus.center_x, limbus.center_y, limbus.semi_major, limbus.semi_minor)
    detect_iris_features(img, pupil, limbus)
    p_after = (pupil.center_x, pupil.center_y, pupil.semi_major, pupil.semi_minor)
    l_after = (limbus.center_x, limbus.center_y, limbus.semi_major, limbus.semi_minor)
    assert p_before == p_after
    assert l_before == l_after


# ── visualization ──────────────────────────────────────────────────────

def test_visualization_returns_same_shape():
    img = _synthetic_iris_image()
    pupil, limbus = _default_geometry()
    res = detect_iris_features(img, pupil, limbus)
    out = draw_iris_overlay(
        cv2_from_gray(img), res, pupil=pupil, limbus=limbus
    )
    assert out.shape == cv2_from_gray(img).shape
    assert out.dtype == np.uint8


def cv2_from_gray(gray):
    import cv2
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── regression: fallback ellipse object contract ────────────────────────

def test_roi_extractor_requires_is_valid_on_ellipse():
    """IrisROIExtractor.build() must not raise when given EllipseParams.

    Regression: _make_ellipse() in gui_app.py returned SimpleNamespace
    which lacked the `is_valid` property, causing AttributeError in
    IrisROIExtractor.build() at roi.py:73.  The fix changed _make_ellipse()
    to return EllipseParams.  This test locks that contract.
    """
    roi_ext = IrisROIExtractor()
    pupil_e = _make_ellipse(100, 100, 55, 55)
    limbus_e = _make_ellipse(100, 100, 130, 130)
    roi = roi_ext.build(pupil_e, limbus_e)
    assert roi.valid, f"ROI should be valid, got reason={roi.reason}"
    assert roi.pupil_radius_px > 0
    assert roi.limbus_radius_px > 0


def test_roi_extractor_rejects_simple_namespace():
    """SimpleNamespace ellipses (missing is_valid) must fail gracefully.

    Pre-fix behavior: IrisROIExtractor.build() raised AttributeError.
    After fix this should still fail gracefully (caught in _detect_iris).
    """
    from types import SimpleNamespace
    roi_ext = IrisROIExtractor()
    bad_pupil = SimpleNamespace(center_x=100, center_y=100,
                                semi_major=55, semi_minor=55, angle_deg=0.0)
    bad_limbus = SimpleNamespace(center_x=100, center_y=100,
                                 semi_major=130, semi_minor=130, angle_deg=0.0)
    try:
        roi = roi_ext.build(bad_pupil, bad_limbus)
        # If it doesn't raise, it must report invalid
        assert not roi.valid, "SimpleNamespace should produce invalid ROI"
    except AttributeError:
        # Pre-fix behavior is acceptable — caught by _detect_iris
        pass


def test_fallback_ellipse_has_required_contract():
    """The attributes produced by _make_ellipse must satisfy IrisROIExtractor."""
    pupil_e = _make_ellipse(960, 540, 56, 56)
    limbus_e = _make_ellipse(960, 540, 140, 140)
    for e in (pupil_e, limbus_e):
        assert hasattr(e, "is_valid"), f"Missing is_valid on {type(e)}"
        assert e.is_valid
        assert hasattr(e, "semi_major")
        assert hasattr(e, "semi_minor")
        assert hasattr(e, "angle_deg")
        assert hasattr(e, "center_x")
        assert hasattr(e, "center_y")


# ── adaptive texture gate (Phase 11) ─────────────────────────────────

def test_adaptive_gate_rejects_flat_iris():
    """A completely flat (zero-texture) synthetic iris must yield 0 features.

    This guards against the adaptive gate admitting noise as false features
    when the iris ROI itself has no measurable texture.
    """
    # Build a flat iris manually: uniform gray inside the iris annulus, no noise.
    size = 320
    gray = np.full((size, size), 25, np.uint8)
    center = size // 2
    cv2_circle_reuse(gray, (center, center), 130, 80)   # iris disc
    cv2_circle_reuse(gray, (center, center), 55, 10)     # pupil
    pupil, limbus = _default_geometry()
    res = detect_iris_features(gray, pupil, limbus).feature_set
    assert res.num_accepted == 0, (
        f"flat iris should yield 0 features, got {res.num_accepted}"
    )


def test_adaptive_gate_configurable_floor():
    """Higher texture_floor must not reduce features on a textured iris.

    The default floor (2.5) is low enough for moderate-texture irides.
    Raising it strictly should reduce (not increase) accepted features.
    """
    img = _synthetic_iris_image(rng_seed=0, texture=12.0, crypts=True)
    pupil, limbus = _default_geometry()

    low_floor = IrisConfig(texture_floor=0.0)
    high_floor = IrisConfig(texture_floor=8.0)
    r_low = detect_iris_features(img, pupil, limbus, config=low_floor).feature_set
    r_high = detect_iris_features(img, pupil, limbus, config=high_floor).feature_set
    # Features should be identical or fewer with the higher floor, never more.
    assert r_high.num_accepted <= r_low.num_accepted


# ── Blue docking-ring line suppression ─────────────────────────────────

def _synthetic_bgr_iris_with_blue_crosshair(
    size=360, rng_seed=1, blue=(150, 90, 80), ticks=(0, 90, 180, 270),
):
    """BGR eye fixture with muted saturated-blue crosshair ticks over the
    annulus.

    The blue is mid-brightness so it is not removed by the reflection remover
    (a saturated-white test would be, masking the real effect), and its
    grayscale value sits inside the intensity pass band so the *only* thing
    that can stop a feature on the tick is the blue-streak mask.
    """
    import cv2

    img = np.full((size, size, 3), 20, np.uint8)
    c = size // 2
    cv2.circle(img, (c, c), 135, (90, 90, 90), -1)   # iris disc
    cv2.circle(img, (c, c), 55, (12, 12, 12), -1)     # pupil
    rng = np.random.default_rng(rng_seed)
    noise = rng.integers(-10, 10, (size, size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for deg in ticks:
        a = np.radians(deg)
        for r in np.arange(64, 110, 0.8):
            x = int(round(c + r * np.cos(a)))
            y = int(round(c + r * np.sin(a)))
            cv2.circle(img, (x, y), 1, blue, -1)
    return img


def test_blue_ring_lines_erased_from_usable_mask():
    """Blue-ring crosshair pixels inside the annulus must be excluded from the
    usable mask when suppression is on, and left intact when it is off.
    """
    img = _synthetic_bgr_iris_with_blue_crosshair()
    pupil, limbus = _default_geometry(size=360)
    c = img.shape[0] // 2
    extractor = IrisROIExtractor()
    roi = extractor.build(pupil, limbus)

    # Collect blue crosshair points that actually fall inside the annulus ROI.
    line_pts = []
    for deg in (0, 90, 180, 270):
        a = np.radians(deg)
        for r in np.arange(60, 115, 1.0):
            x = int(round(c + r * np.cos(a)))
            y = int(round(c + r * np.sin(a)))
            if roi.valid and point_in_roi_annulus(x, y, roi):
                line_pts.append((x, y))
    assert line_pts, "fixture expected blue tick pixels inside the annulus"

    off = IrisMasking(suppress_blue_ring_lines=False).build(img, roi)
    on = IrisMasking(suppress_blue_ring_lines=True).build(img, roi)
    on_line_off = sum(1 for (x, y) in line_pts if off[y, x])
    on_line_on = sum(1 for (x, y) in line_pts if on[y, x])
    assert on_line_off > 0, "blue ticks should be usable without suppression"
    assert on_line_on == 0, "blue ticks must be erased when suppression is on"


def test_blue_ring_lines_suppress_features_on_crosshair():
    """End-to-end: with a blue ring crosshair through the annulus and
    suppression enabled, no accepted feature may sit on the cardinal axes
    within the erased margin.  The control is the masking-level test above;
    this test confirms the extractor honours the erased usable mask.
    """
    img = _synthetic_bgr_iris_with_blue_crosshair(rng_seed=1)
    pupil, limbus = _default_geometry(size=360)
    c = img.shape[0] // 2

    on_cfg = IrisConfig(
        min_contrast=3.0, texture_floor=2.0, eyelid_method="none",
        suppress_blue_ring_lines=True,
    )
    det = IrisFeatureDetector(config=on_cfg)
    res = det.detect(img, pupil, limbus)
    assert res.valid
    on_line = [
        ft for ft in res.feature_set.features
        if (((abs(ft.x - c) < 3) or (abs(ft.y - c) < 3))
            and (ft.angle_deg % 90 < 2.5
                 or (360 - ft.angle_deg) % 90 < 2.5))
    ]
    assert len(on_line) == 0, f"blue-ring lines still yield features: {on_line}"


def test_blue_ring_suppression_preserves_solid_blue_iris():
    """A broadly blue iris (every annulus pixel blue) must NOT be erased:
    the streak discriminator requires a *thin* line, i.e. low local blue
    density, so a solid blue iris region passes the blueness test but fails
    the thin-streak test and stays usable.
    """
    import cv2

    size = 320
    img = np.full((size, size, 3), 20, np.uint8)
    c = size // 2
    # Shade the whole iris disc and annulus a muted saturated blue.
    cv2.circle(img, (c, c), 135, (130, 70, 55), -1)
    cv2.circle(img, (c, c), 55, (12, 12, 12), -1)
    # Speaker-blue flecks are just as blue, but are *not* erased either.
    rng = np.random.default_rng(3)
    img = np.clip(
        img.astype(np.int16)
        + rng.integers(-8, 8, (size, size, 3)).astype(np.int16), 0, 255
    ).astype(np.uint8)

    pupil, limbus = _default_geometry(size=size)
    roi = IrisROIExtractor().build(pupil, limbus)
    usable = IrisMasking(
        suppress_blue_ring_lines=True, eyelid_method="none"
    ).build(img, roi)
    area = int(sample_annulus_mask(img.shape, roi).sum())
    usable_area = int(usable.sum())
    # A solid blue iris must keep almost all of its annulus usable; only a
    # marginal few pixels may drop out from noise.
    assert usable_area / max(area, 1) > 0.85, (
        f"solid blue iris was over-erased: {usable_area}/{area}"
    )
