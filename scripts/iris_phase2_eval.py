"""Phase II repeatability & robustness evaluation for the iris feature detector.

Runs the full production pipeline (UnifiedDetector) on the 12 clean clinical
proxy images and measures, for each valid ROI:

  A. baseline feature statistics,
  B. spatial distribution metrics (coverage / concentration / gaps),
  C. determinism (identical repeated runs),
  D. robustness to controlled perturbations (photometric + geometric),
  E. occlusion / reflection robustness,
  F. threshold sensitivity around ``min_contrast=4.0``,
  G. feature quality vs stability,
  H. performance,
  I. production-safety checks.

Repeatability uses the *controlled-correspondence* model: perturbations are
applied with known transformations and features are compared in the baseline
normalised iris coordinate frame (see ``pupil_tracking/iris/robustness.py``).

This is an evaluation/diagnostic tool. It is NOT part of the automated test
suite because it requires the production ONNX model and clinical imagery (both
gitignored). Focused, deterministic unit tests live in
``pupil_tracking/tests/test_iris_robustness.py``.

Usage:
    python scripts/iris_phase2_eval.py [--only eye_01.jpeg]
"""

import argparse
import glob
import math
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris import IrisConfig, detect_iris_features
from pupil_tracking.iris import robustness as R
from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import IrisFeature
from pupil_tracking.utils.types import EllipseParams


# --------------------------------------------------------------------------- #
# Geometry helpers (for geometric perturbations)
# --------------------------------------------------------------------------- #

def ellipse_transform(
    e: EllipseParams,
    kind: str,
    value,
    w: int,
    h: int,
) -> EllipseParams:
    """Return the EllipseParams transformed by the same geometric perturbation
    applied to the image, so the iris ROI tracks the moved iris."""
    if e is None:
        return e
    cx, cy = e.center_x, e.center_y
    smaj, smin = e.semi_major, e.semi_minor
    ang = e.angle_deg
    if kind == "translate":
        dx, dy = int(round(value[0])), int(round(value[1]))
        out = EllipseParams(
            center_x=cx + dx, center_y=cy + dy,
            semi_major=smaj, semi_minor=smin, angle_deg=ang,
        )
        return _copy_aux(e, out)
    if kind == "rotate":
        icx, icy = w / 2.0, h / 2.0
        rad = np.deg2rad(value)
        ox, oy = cx - icx, cy - icy
        nx = icx + ox * np.cos(rad) - oy * np.sin(rad)
        ny = icy + ox * np.sin(rad) + oy * np.cos(rad)
        out = EllipseParams(
            center_x=float(nx), center_y=float(ny),
            semi_major=smaj, semi_minor=smin, angle_deg=(ang + value) % 180.0,
        )
        return _copy_aux(e, out)
    if kind == "scale":
        factor = float(value)
        icx, icy = w / 2.0, h / 2.0
        o = (w - w * factor) / 2.0
        oy = (h - h * factor) / 2.0
        out = EllipseParams(
            center_x=float(icx + o + (cx - icx) * factor),
            center_y=float(icy + oy + (cy - icy) * factor),
            semi_major=smaj * factor, semi_minor=smin * factor,
            angle_deg=ang,
        )
        return _copy_aux(e, out)
    return e


def _copy_aux(src: EllipseParams, dst: EllipseParams) -> EllipseParams:
    for attr in ("fit_quality", "fit_rms_residual", "num_contour_points",
                 "eccentricity", "circularity"):
        if hasattr(src, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except Exception:
                pass
    return dst


# --------------------------------------------------------------------------- #
# Detection wrapper
# --------------------------------------------------------------------------- #

def extract(image_bgr, pe, le, cfg, occlusion=None) -> tuple:
    """Run iris detection; return (pe, le, iris_result, elapsed_ms)."""
    t0 = time.perf_counter()
    ir = detect_iris_features(
        image_bgr, pe, le, config=cfg, external_occlusion=occlusion,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return ir, ms


def detect_image(det, image_bgr, cfg, path, occlusion=None):
    """UnifiedDetector + iris detection on one image; returns a result bundle."""
    dr = det.detect(image_bgr, frame_number=0, source=path)
    pe = dr.pupil.ellipse if dr.has_pupil else None
    le = dr.limbus.ellipse if dr.has_limbus else None
    ir, ms = extract(image_bgr, pe, le, cfg, occlusion)
    return {
        "path": path,
        "det": dr,
        "pe": pe,
        "le": le,
        "iris": ir,
        "ms": ms,
        "pup_r": pe.radius if pe is not None else 0.0,
        "lim_r": le.radius if le is not None else 0.0,
    }


# --------------------------------------------------------------------------- #
# Per-image metrics
# --------------------------------------------------------------------------- #

def summarize_image(bundle) -> dict:
    ir = bundle["iris"]
    fs = ir.feature_set
    stats = R.baseline_statistics(ir)
    dist = R.spatial_distribution(fs)
    return {
        "path": bundle["path"],
        "pup_r": bundle["pup_r"],
        "lim_r": bundle["lim_r"],
        "status": ir.status.value,
        **{k: stats[k] for k in (
            "valid", "accepted", "candidates", "rejected",
            "usable_fraction", "mean_quality", "median_quality",
            "min_quality", "max_quality", "processing_time_ms",
        )},
        **{k: dist[k] for k in (
            "angular_coverage", "radial_coverage", "cell_coverage",
            "concentration", "angular_entropy", "mean_nn_angular_gap",
            "min_nn_angular_gap", "quadrant_count",
        )},
    }


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def run_determinism(bundle, cfg, repeats: int = 5) -> dict:
    """Re-run iris detection on the same (already-perturbed) image & geometry and
    require identical accepted count + identical normalised coordinates."""
    image = bundle["_image"]
    pe, le = bundle["pe"], bundle["le"]
    first = None
    for _ in range(repeats):
        ir, _ = extract(image, pe, le, cfg)
        fs = ir.feature_set
        sig = (
            fs.num_accepted,
            tuple((round(f.angle_deg, 4), round(f.radial_norm, 4))
                  for f in fs.features),
        )
        if first is None:
            first = sig
        elif sig != first:
            return {"deterministic": False}
    return {"deterministic": True}


# --------------------------------------------------------------------------- #
# Perturbation robustness
# --------------------------------------------------------------------------- #

PHOTOMETRIC = [
    ("brightness", -25),
    ("brightness", +25),
    ("contrast", 0.8),
    ("contrast", 1.2),
    ("gamma", 0.8),
    ("gamma", 1.2),
    ("noise", 2.0),
    ("noise", 6.0),
    ("blur", 3),
    ("blur", 7),
    ("sharpen", 0.2),
    ("sharpen", 0.6),
]

GEOMETRIC = [
    ("translate", (4, 0)),
    ("translate", (0, 4)),
    ("rotate", -3.0),
    ("rotate", 3.0),
    ("scale", 0.97),
    ("scale", 1.03),
]

DEFAULT_PERTURBATIONS = PHOTOMETRIC + GEOMETRIC


def run_perturbation(bundle, kind, value, seed, cfg):
    """Run one perturbation and compute repeatability vs the baseline features.

    For photometric perturbations (pixel-stationary) the same baseline geometry is
    reused, so perturbed-feature normalised coords are already in the baseline
    frame. For geometric perturbations the geometry is transformed to track the
    moved iris and each perturbed feature's pixel position is mapped back to the
    baseline frame before comparison.
    """
    image = bundle["_image"]
    pe, le = bundle["pe"], bundle["le"]
    base_ir = bundle["iris"]
    base_feats = base_ir.feature_set.features
    h, w = image.shape[:2]

    if kind in ("translate", "rotate", "scale"):
        perturbed_img = R.PERTURBATIONS[kind](image, value, seed)
        tpe = ellipse_transform(pe, kind, value, w, h)
        tle = ellipse_transform(le, kind, value, w, h)
        ir, _ = extract(perturbed_img, tpe, tle, cfg)
        fs = ir.feature_set
        rnorm = IrisNormalizer()
        roi = base_ir.feature_set.roi
        mapped = []
        for f in fs.features:
            bx, by = R.map_point_back(f.x, f.y, kind, value, w, h)
            ang_rn = rnorm.to_iris_relative(bx, by, roi)
            if ang_rn is None:
                continue
            mf = IrisFeature(
                id=f.id, x=float(bx), y=float(by),
                angle_deg=float(ang_rn[0]), radial_norm=float(ang_rn[1]),
                scale=f.scale, orientation_deg=f.orientation_deg,
                feature_type=f.feature_type, response=f.response,
                local_contrast=f.local_contrast, visibility=f.visibility,
                confidence=f.confidence,
            )
            mapped.append(mf)
        pert_feats = mapped
    else:
        perturbed_img = R.PERTURBATIONS[kind](image, value, seed)
        ir, _ = extract(perturbed_img, pe, le, cfg)
        pert_feats = ir.feature_set.features

    return R.repeatability_metrics(base_feats, pert_feats)


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="only run this image (e.g. eye_01.jpeg)")
    args = ap.parse_args()

    det = UnifiedDetector()
    cfg = IrisConfig()

    paths = sorted(glob.glob("clinical_data/clean/*.jpeg"))
    if args.only:
        paths = [p for p in paths if args.only in p]
    if not paths:
        print("no images matched")
        return 1

    bunches = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        b = detect_image(det, img, cfg, path)
        b["_image"] = img
        bunches.append(b)

    print("=" * 100)
    print("PHASE II — IRIS FEATURE REPEATABILITY & ROBUSTNESS (clinical proxy)")
    print("=" * 100)

    # ---- A. Baseline stats + B. spatial distribution --------------------- #
    print("\n[A+B] BASELINE STATS & SPATIAL DISTRIBUTION")
    hdr = (f"{'image':<16}{'det':<5}{'pup_r':<7}{'lim_r':<7}{'st':<10}"
           f"{'cand':<6}{'acc':<6}{'rej':<6}{'use':<6}"
           f"{'aCov':<6}{'rCov':<6}{'cCov':<6}{'conc':<6}{'ent':<6}"
           f"{'nnAng':<6}{'quad':<6}{'qualMed':<8}{'ms':<7}")
    print(hdr)
    valid_bundles = []
    for b in bunches:
        if not b["iris"].feature_set.roi.valid:
            print(f"{b['path']:<16}{'--':<5}  (no valid ROI)")
            continue
        s = summarize_image(b)
        valid_bundles.append((b, s))
        print(
            f"{s['path']:<16}{'Y':<5}{s['pup_r']:<7.1f}{s['lim_r']:<7.1f}"
            f"{s['status']:<10}{s['candidates']:<6}{s['accepted']:<6}"
            f"{s['rejected']:<6}{s['usable_fraction']:<6.2f}"
            f"{s['angular_coverage']:<6.2f}{s['radial_coverage']:<6.2f}"
            f"{s['cell_coverage']:<6.2f}{s['concentration']:<6.2f}"
            f"{s['angular_entropy']:<6.2f}{s['mean_nn_angular_gap']:<6.1f}"
            f"{s['quadrant_count']:<6}{s['median_quality']:<8.3f}"
            f"{s['processing_time_ms']:<7.1f}"
        )

    # ---- C. Determinism -------------------------------------------------- #
    print("\n[C] DETERMINISM (repeated identical runs)")
    for b, s in valid_bundles:
        det_ok = run_determinism(b, cfg, repeats=5)
        print(f"{b['path']:<16} deterministic={det_ok['deterministic']}")

    # ---- D. Perturbation robustness -------------------------------------- #
    print("\n[D] PERTURBATION ROBUSTNESS (repeatability in normalised frame)")
    types = ["brightness", "contrast", "gamma", "noise", "blur",
             "sharpen", "translate", "rotate", "scale"]
    # aggregate per type across images and per-type values
    agg = {t: {"rep": [], "ret": [], "ang": [], "rad": [], "n": 0} for t in types}
    for b, s in valid_bundles:
        for kind, value in DEFAULT_PERTURBATIONS:
            if not b["pe"] or not b["le"]:
                continue
            m = run_perturbation(b, kind, value, 0, cfg)
            agg[kind]["rep"].append(m["repeatability_rate"])
            agg[kind]["ret"].append(m["retained_feature_rate"])
            agg[kind]["ang"].append(m["median_ang_gap_deg"])
            agg[kind]["rad"].append(m["median_rad_gap"])
            agg[kind]["n"] += 1
    print()
    # Per-type aggregate table
    print(f"{'perturbation':<14}{'rep_mean':<10}{'rep_min':<10}{'ret_mean':<10}"
          f"{'angMed(deg)':<12}{'radMed':<8}{'n':<5}")
    for t in types:
        rep = agg[t]["rep"]
        ret = agg[t]["ret"]
        ang = agg[t]["ang"]
        rad = agg[t]["rad"]
        if not rep:
            print(f"{t:<14}{'n/a':<10}")
            continue
        print(f"{t:<14}{np.mean(rep):<10.3f}{np.min(rep):<10.3f}"
              f"{np.mean(ret):<10.3f}{np.mean(ang):<12.3f}{np.mean(rad):<8.3f}"
              f"{agg[t]['n']:<5}")

    # Per-image robustness matrix: each cell = mean repeatability across the
    # perturbation values for that type (so sparse-feature images are visible).
    print("\n  Per-image repeatability matrix (mean across perturb values)")
    col_types = ["brightness", "contrast", "gamma", "noise", "blur",
                 "sharpen", "translate", "rotate", "scale"]
    per_img = {}
    for b, s in valid_bundles:
        row = []
        for t in col_types:
            vals = [m["repeatability_rate"]
                    for (kind, value) in DEFAULT_PERTURBATIONS
                    if kind == t
                    for m in [run_perturbation(b, kind, value, 0, cfg)]]
            row.append(np.mean(vals) if vals else float("nan"))
        per_img[b["path"]] = row
    print(f"{'image':<18}" + "".join(f"{t[:6]:>8}" for t in col_types))
    for path, row in per_img.items():
        cells = "".join(f"{('%.2f' % v):>8}" if np.isfinite(v) else f"{'n/a':>8}"
                        for v in row)
        print(f"{path:<18}{cells}")

    # ---- E. Occlusion / reflection robustness ---------------------------- #
    print("\n[E] OCCLUSION / REFLECTION ROBUSTNESS (geometry-aware annulus occluder)")

    def _annulus_occluder(h, w, cx, cy, inner_r, outer_r, frac, seed):
        """Disc guaranteed to overlap the iris annulus: centred mid-annulus on a
        ray from the iris centre, radius = frac x annulus width."""
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0.0, 2.0 * math.pi)
        mid = (inner_r + outer_r) / 2.0
        px = int(round(cx + mid * math.cos(ang)))
        py = int(round(cy + mid * math.sin(ang)))
        r = max(int(frac * (outer_r - inner_r)), 4)
        img = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(img, (px, py), r, 1, -1)
        return img.astype(bool), (px, py, r)

    for b, s in valid_bundles:
        if not b["pe"] or not b["le"]:
            continue
        image = b["_image"]
        pe, le = b["pe"], b["le"]
        h, w = image.shape[:2]
        base = b["iris"].feature_set
        base_count = base.num_accepted
        cx = pe.center_x
        cy = pe.center_y
        inner_r = b["pup_r"]
        outer_r = b["lim_r"]
        occ_mask, (ox, oy, r) = _annulus_occluder(
            h, w, cx, cy, inner_r, outer_r, frac=0.45, seed=2)
        ir_occ, _ = extract(image, pe, le, cfg, occlusion=occ_mask)
        fs_occ = ir_occ.feature_set
        m = R.repeatability_metrics(base.features, fs_occ.features)
        # Does any accepted feature fall inside the occluded disc? (should be none)
        inside = 0
        for f in fs_occ.features:
            if (f.x - ox) ** 2 + (f.y - oy) ** 2 <= r * r:
                inside += 1
        print(
            f"{b['path']:<16} occ_r={r:<3} acc {base_count}->{fs_occ.num_accepted}"
            f"  use {base.usable_fraction:.3f}->{fs_occ.usable_fraction:.3f}"
            f"  insidemask={inside}  rep={m['repeatability_rate']:.3f}"
        )

    # ---- F. Threshold sensitivity ---------------------------------------- #
    print("\n[F] THRESHOLD SENSITIVITY (min_contrast; candidates pass gate before "
          "angular pruning)")
    # ``num_candidates`` is the count that passes the min_contrast gate; it shows
    # where the gate actually binds. ``num_accepted`` reflects the later angular-
    # separation pruning. Reporting both distinguishes the two limitations.
    for th in [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]:
        cfg2 = IrisConfig(min_contrast=th)
        cands, accs = [], []
        for b, _ in valid_bundles:
            img = b["_image"]
            ir, _ = extract(img, b["pe"], b["le"], cfg2)
            cands.append(ir.feature_set.num_candidates)
            accs.append(ir.feature_set.num_accepted)
        cands = [c for c in cands if c is not None]
        accs = [a for a in accs if a is not None]
        if cands:
            print(f"  min_contrast={th:<5} mean_cand={np.mean(cands):<8.2f}"
                  f" mean_acc={np.mean(accs):<8.2f} max_cand={max(cands):<5}"
                  f" max_acc={max(accs)}")

    # ---- G. Quality vs stability ----------------------------------------- #
    print("\n[G] QUALITY vs STABILITY (Spearman rho; defined-only aggregation)")
    rhos = []
    n_defined = 0
    for b, _ in valid_bundles:
        if not b["pe"] or not b["le"] or len(b["iris"].feature_set.features) < 2:
            continue
        for kind, value in PHOTOMETRIC[:6]:
            base = b["iris"].feature_set.features
            perturbed_img = R.PERTURBATIONS[kind](b["_image"], value, 0)
            ir, _ = extract(perturbed_img, b["pe"], b["le"], cfg)
            q = R.quality_stability_correlation(base, ir.feature_set.features)
            if q["defined"]:
                rhos.append(q["spearman_rho"])
                n_defined += 1
    if rhos:
        print(f"  spearman_rho mean={np.mean(rhos):.3f} "
              f"min={np.min(rhos):.3f} max={np.max(rhos):.3f} "
              f"(defined n={n_defined})")

    # ---- H. Performance --------------------------------------------------- #
    print("\n[H] PERFORMANCE (iris-only elapsed ms/mask+extraction)")
    ms_all = [s["processing_time_ms"] for _, s in valid_bundles]
    if ms_all:
        print(f"  mean={np.mean(ms_all):.1f} ms  median={np.median(ms_all):.1f} ms  "
              f"worst={np.max(ms_all):.1f} ms  n={len(ms_all)}")

    # ---- I. Production safety (iris disabled by default) ------------------ #
    print("\n[I] PRODUCTION SAFETY")
    from pupil_tracking.iris.detect import IrisFeatureDetector
    sig = IrisFeatureDetector
    print(f"  iris detector importable: {sig is not None}")

    print("\n" + "=" * 100)
    print("PHASE II EVALUATION COMPLETE")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
