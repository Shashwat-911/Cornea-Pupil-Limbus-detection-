"""Phase XI: sparse-feature analysis and honest-refusal experiment.

Runs the correspondence layer on the 5 clinical proxy images with both
consensus and global_hybrid rotation methods. Reports feature-level
metrics and evaluates an evidence gate for honest refusal.
"""

import argparse
import glob
import math
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, ".")

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.detect import detect_iris_features
from pupil_tracking.iris.paired import PairConfig, make_synthetic_pair
from pupil_tracking.iris.correspondence import (
    CorrespondenceConfig,
    MatchingBaseline,
    circular_distance,
    circular_mean,
    circular_std,
    estimate_correspondence,
    evaluate_pair,
    wrap_deg,
)
from pupil_tracking.utils.types import EllipseParams


CASE_IMAGES = ["eye_01.jpeg", "eye_02.jpeg", "eye_03.jpeg",
               "eye_11.jpeg", "eye_13.jpeg"]

ROTATIONS = [
    ("rot+1", 1.0), ("rot-1", -1.0),
    ("rot+3", 3.0), ("rot-3", -3.0),
    ("rot+5", 5.0), ("rot+6", 6.0),
]


def _scaled_ellipse(e: EllipseParams, s: float, cx: float, cy: float):
    if e is None:
        return None
    nx = cx + (e.center_x - cx) * s
    ny = cy + (e.center_y - cy) * s
    return EllipseParams(center_x=nx, center_y=ny,
                         semi_major=e.semi_major * s, semi_minor=e.semi_minor * s,
                         angle_deg=e.angle_deg)


def compute_feature_metrics(features):
    """Compute sparse-feature coverage metrics from a list of IrisFeature."""
    if not features:
        return {
            "feature_count": 0,
            "angular_span": 0.0,
            "largest_angular_gap": 360.0,
            "angular_coverage_ratio": 0.0,
            "occupied_angular_bins_30": 0,
            "radial_min": 0.0,
            "radial_max": 0.0,
            "radial_range": 0.0,
        }

    angles = sorted(float(f.angle_deg) % 360.0 for f in features)
    n = len(angles)

    # Angular span (smallest enclosing arc)
    if n < 2:
        span = 0.0
        largest_gap = 360.0
    else:
        gaps = [angles[i + 1] - angles[i] for i in range(n - 1)]
        gaps.append(angles[0] + 360.0 - angles[-1])
        largest_gap = max(gaps)
        span = 360.0 - largest_gap

    # Angular coverage ratio (span / 360)
    coverage = span / 360.0

    # Occupied 30-degree bins
    bins = set()
    for a in angles:
        bins.add(int(a // 30) % 12)

    # Radial coverage
    radials = [float(f.radial_norm) for f in features]
    r_min = min(radials)
    r_max = max(radials)

    return {
        "feature_count": n,
        "angular_span": span,
        "largest_angular_gap": largest_gap,
        "angular_coverage_ratio": coverage,
        "occupied_angular_bins_30": len(bins),
        "radial_min": r_min,
        "radial_max": r_max,
        "radial_range": r_max - r_min,
    }


def run_benchmark(rotation_method: str, det, paths):
    """Run the full rotation benchmark with a given method."""
    cfg = CorrespondenceConfig()
    rows = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        dr = det.detect(img, frame_number=0, source=path)
        if not dr.has_pupil or not dr.has_limbus:
            continue
        pe, le = dr.pupil.ellipse, dr.limbus.ellipse

        res_a = detect_iris_features(img, pe, le)
        if not res_a.feature_set.roi.valid:
            continue

        feats_a = res_a.feature_set.features
        fm = compute_feature_metrics(feats_a)

        for case, gt_rot in ROTATIONS:
            cx, cy = pe.center_x, pe.center_y
            pair_cfg = PairConfig(rotation_deg=gt_rot, scale=1.0,
                                  translation_px=(0.0, 0.0),
                                  center=(cx, cy), seed=0)
            pair = make_synthetic_pair(img, pair_cfg, name=case)
            res_b = detect_iris_features(
                pair.image_b, pe, le,
                external_occlusion=pair.occlusion_mask,
            )
            out = evaluate_pair(
                img, pair.image_b, res_a.feature_set, res_b.feature_set,
                gt_rotation_deg=gt_rot, gt_scale=1.0,
                baseline=MatchingBaseline.GEOMETRIC_DESCRIPTOR,
                rotation_method=rotation_method,
            )
            out["case"] = case
            out["image"] = path
            out.update(fm)
            rows.append(out)
    return rows


def classify(rows):
    """Classify TRUE-OK, FALSE-OK, FAILED."""
    true_ok = []
    false_ok = []
    failed = []
    for r in rows:
        mcd = r["min_circular_diff_deg"]
        ok = r["failure"] == "OK"
        if ok and mcd <= 1.0:
            true_ok.append(r)
        elif ok and mcd > 1.0:
            false_ok.append(r)
        else:
            failed.append(r)
    return true_ok, false_ok, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="only run this image")
    args = ap.parse_args()

    det = UnifiedDetector()
    paths = sorted(glob.glob("clinical_data/clean/*.jpeg"))
    if args.only:
        paths = [p for p in paths if args.only in p]
    paths = [p for p in paths if any(i in p for i in CASE_IMAGES)]

    # --- Run benchmarks ---
    print("=" * 120)
    print("PHASE XI — SPARSE FEATURE & CORRESPONDENCE ROBUSTNESS")
    print("=" * 120)

    methods = ["consensus", "global_hybrid"]
    all_results = {}
    for method in methods:
        t0 = time.perf_counter()
        rows = run_benchmark(method, det, paths)
        elapsed = time.perf_counter() - t0
        all_results[method] = rows
        print(f"\n[{method}] {len(rows)} cases, {elapsed:.1f}s")

    # --- Feature metrics per image ---
    print("\n" + "=" * 120)
    print("FEATURE METRICS PER IMAGE")
    print("=" * 120)
    header = f"{'image':<20} {'n_feat':>6} {'span':>7} {'gap':>7} {'cov':>6} {'bins':>5} {'r_min':>6} {'r_max':>6}"
    print(header)
    print("-" * 120)

    seen_images = {}
    for method in methods:
        for r in all_results[method]:
            img = r["image"]
            if img not in seen_images:
                seen_images[img] = r
    for img in sorted(seen_images):
        r = seen_images[img]
        print(f"{img:<20} {r['feature_count']:>6} {r['angular_span']:>7.1f} "
              f"{r['largest_angular_gap']:>7.1f} {r['angular_coverage_ratio']:>6.2f} "
              f"{r['occupied_angular_bins_30']:>5} {r['radial_min']:>6.2f} "
              f"{r['radial_max']:>6.2f}")

    # --- Rotation performance comparison ---
    print("\n" + "=" * 120)
    print("ROTATION PERFORMANCE COMPARISON")
    print("=" * 120)

    for method in methods:
        rows = all_results[method]
        true_ok, false_ok, failed = classify(rows)
        n_total = len(rows)
        n_ok = len(true_ok) + len(false_ok)
        mcds_ok = [r["min_circular_diff_deg"] for r in true_ok]
        mcds_all = [r["min_circular_diff_deg"] for r in rows]

        print(f"\n--- {method} ---")
        print(f"  TRUE-OK:  {len(true_ok)}/{n_total}")
        print(f"  FALSE-OK: {len(false_ok)}/{n_total}")
        print(f"  FAILED:   {len(failed)}/{n_total}")
        print(f"  Acceptance: {n_ok/n_total:.3f}")
        if mcds_ok:
            print(f"  Mean MCD (TRUE-OK only): {np.mean(mcds_ok):.3f}°")
        print(f"  Mean MCD (all):          {np.mean(mcds_all):.3f}°")
        if false_ok:
            print(f"  FALSE-OK details:")
            for r in false_ok:
                print(f"    {r['image']:<20} {r['case']:<12} mcd={r['min_circular_diff_deg']:.2f}° "
                      f"n_feat={r['feature_count']} span={r['angular_span']:.1f}° "
                      f"gap={r['largest_angular_gap']:.1f}°")

    # --- Metric correlation with correctness ---
    print("\n" + "=" * 120)
    print("METRIC CORRELATION WITH CORRECTNESS (global_hybrid)")
    print("=" * 120)
    rows_gh = all_results["global_hybrid"]
    true_ok_gh, false_ok_gh, failed_gh = classify(rows_gh)

    for metric in ["feature_count", "angular_span", "largest_angular_gap",
                    "angular_coverage_ratio", "occupied_angular_bins_30",
                    "radial_range", "global_inlier_count", "global_inlier_frac"]:
        vals_ok = [r[metric] for r in true_ok_gh]
        vals_fo = [r[metric] for r in false_ok_gh]
        vals_fail = [r[metric] for r in failed_gh]
        mean_ok = np.mean(vals_ok) if vals_ok else float("nan")
        mean_fo = np.mean(vals_fo) if vals_fo else float("nan")
        mean_fail = np.mean(vals_fail) if vals_fail else float("nan")
        print(f"  {metric:<30} TRUE-OK mean={mean_ok:>8.3f}  "
              f"FALSE-OK mean={mean_fo:>8.3f}  FAILED mean={mean_fail:>8.3f}")

    # --- Evidence gate investigation ---
    print("\n" + "=" * 120)
    print("EVIDENCE GATE INVESTIGATION (global_hybrid)")
    print("=" * 120)

    # Test various gate thresholds
    for gate_name, gate_key, thresholds in [
        ("min_angular_coverage", "angular_coverage_ratio", [0.20, 0.25, 0.30, 0.40, 0.50]),
        ("min_feature_count", "feature_count", [3, 5, 7, 9, 12]),
        ("min_inlier_count", "global_inlier_count", [2, 3, 4, 5]),
        ("min_inlier_frac", "global_inlier_frac", [0.30, 0.40, 0.50, 0.60]),
        ("min_occupied_bins", "occupied_angular_bins_30", [3, 4, 5, 6]),
    ]:
        print(f"\n  [{gate_name}]")
        for thresh in thresholds:
            retained = 0
            rejected_true = 0
            rejected_false = 0
            kept_true = 0
            kept_false = 0
            for r in rows_gh:
                if r[gate_key] >= thresh:
                    retained += 1
                    if r["min_circular_diff_deg"] <= 1.0 and r["failure"] == "OK":
                        kept_true += 1
                    elif r["failure"] == "OK" and r["min_circular_diff_deg"] > 1.0:
                        kept_false += 1
                else:
                    if r["min_circular_diff_deg"] <= 1.0 and r["failure"] == "OK":
                        rejected_true += 1
                    elif r["failure"] == "OK" and r["min_circular_diff_deg"] > 1.0:
                        rejected_false += 1
            n_total = len(rows_gh)
            n_ok_gh = len([r for r in rows_gh if r["failure"] == "OK"])
            print(f"    >= {thresh:<8} accept={retained/n_total:.3f} "
                  f"TRUE-OK kept={kept_true} rejected={rejected_true} "
                  f"FALSE-OK kept={kept_false} rejected={rejected_false}")

    # --- Best combined gate ---
    print("\n" + "=" * 120)
    print("BEST COMBINED GATE (global_hybrid)")
    print("=" * 120)

    # Test: angular_coverage >= 0.25 AND global_inlier_count >= 3
    best_gate = None
    best_score = -1
    for cov_min in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for inl_min in [2, 3, 4, 5]:
            kept_true = 0
            kept_false = 0
            rej_true = 0
            rej_false = 0
            for r in rows_gh:
                passes = (r["angular_coverage_ratio"] >= cov_min
                          and r["global_inlier_count"] >= inl_min)
                is_true = r["min_circular_diff_deg"] <= 1.0 and r["failure"] == "OK"
                is_false = r["failure"] == "OK" and r["min_circular_diff_deg"] > 1.0
                if passes:
                    if is_true:
                        kept_true += 1
                    elif is_false:
                        kept_false += 1
                else:
                    if is_true:
                        rej_true += 1
                    elif is_false:
                        rej_false += 1
            # Score: want max kept_true, min kept_false
            total_true = kept_true + rej_true
            total_false = kept_false + rej_false
            if total_true == 0:
                continue
            # Score: weighted: keep all true, reject all false
            score = kept_true / max(total_true, 1) - 2.0 * kept_false / max(total_false + total_true, 1)
            if score > best_score:
                best_score = score
                best_gate = {
                    "cov_min": cov_min,
                    "inl_min": inl_min,
                    "kept_true": kept_true,
                    "kept_false": kept_false,
                    "rej_true": rej_true,
                    "rej_false": rej_false,
                    "total_true": total_true,
                    "total_false": total_false,
                    "acceptance": (kept_true + kept_false) / len(rows_gh),
                }
    if best_gate:
        print(f"  Best gate: angular_coverage >= {best_gate['cov_min']} "
              f"AND global_inlier_count >= {best_gate['inl_min']}")
        print(f"    TRUE-OK: kept={best_gate['kept_true']}/{best_gate['total_true']}, "
              f"rejected={best_gate['rej_true']}")
        print(f"    FALSE-OK: kept={best_gate['kept_false']}, "
              f"rejected={best_gate['rej_false']}")
        print(f"    Acceptance: {best_gate['acceptance']:.3f}")

    # --- Eye-by-eye comparison ---
    print("\n" + "=" * 120)
    print("EYE-BY-EYE COMPARISON")
    print("=" * 120)
    for img in sorted(set(r["image"] for r in all_results["consensus"])):
        for method in methods:
            img_rows = [r for r in all_results[method] if r["image"] == img]
            true_ok, false_ok, failed = classify(img_rows)
            print(f"  {img:<20} {method:<18} TRUE-OK={len(true_ok)} "
                  f"FALSE-OK={len(false_ok)} FAILED={len(failed)}")
            for r in img_rows:
                tag = "TRUE-OK" if (r["failure"] == "OK" and r["min_circular_diff_deg"] <= 1.0) \
                    else "FALSE-OK" if r["failure"] == "OK" else "FAILED"
                print(f"    {r['case']:<12} mcd={r['min_circular_diff_deg']:>6.2f}° "
                      f"n_feat={r['feature_count']:>3} span={r['angular_span']:>6.1f}° "
                      f"gap={r['largest_angular_gap']:>6.1f}° "
                      f"inlier={r['global_inlier_count']:>2}/{r['n_matches']:<2} "
                      f"frac={r['global_inlier_frac']:.2f}  [{tag}]")
        print()

    print("=" * 120)
    print("PHASE XI ANALYSIS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    sys.exit(main() or 0)
