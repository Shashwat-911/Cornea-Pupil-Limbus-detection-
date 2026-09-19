"""Phase XVIII — Real ELITA paired-image cyclotorsion workflow.

Runs the full pipeline on the 12 clean clinical images:
  pupil/limbus detection → iris detection → pair correspondence

Pairs are created from all 5 valid-iris images. Ground truth is
unavailable; no accuracy claims are made.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.iris.correspondence import (
    CorrespondenceConfig,
    estimate_correspondence,
    MatchingBaseline,
)
from pupil_tracking.iris.types import IrisFeatureSet


# ── Configuration ────────────────────────────────────────────────────────── #

DATA_DIR = PROJECT_ROOT / "clinical_data" / "clean"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "phase18_output"
CORR_CONFIG = CorrespondenceConfig(evidence_gate=True)

# Images known to produce valid iris features from Phase XVII
VALID_IMAGES = ["eye_01.jpeg", "eye_02.jpeg", "eye_03.jpeg", "eye_11.jpeg", "eye_13.jpeg"]


# ── Helpers ──────────────────────────────────────────────────────────────── #

def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as BGR uint8."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def run_full_pipeline(
    detector: UnifiedDetector,
    iris_detector: IrisFeatureDetector,
    image: np.ndarray,
) -> Dict:
    """Run pupil/limbus detection then iris detection on a single image.

    Returns a dict with all intermediate results.
    """
    t0 = time.perf_counter()

    # Step 1: pupil/limbus detection (existing pipeline)
    geo = detector.detect(image)
    t_pupil = (time.perf_counter() - t0) * 1000.0

    has_pupil = geo.pupil.detected
    has_limbus = geo.limbus.detected

    # Step 2: iris feature detection (additive)
    t_iris_start = time.perf_counter()
    iris_result = iris_detector.detect(
        image,
        pupil=geo.pupil.ellipse if has_pupil else None,
        limbus=geo.limbus.ellipse if has_limbus else None,
    )
    t_iris = (time.perf_counter() - t_iris_start) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "geometry": geo,
        "has_pupil": has_pupil,
        "has_limbus": has_limbus,
        "pupil_detection_ms": round(t_pupil, 1),
        "iris_result": iris_result,
        "iris_features": len(iris_result.feature_set.features) if iris_result.valid else 0,
        "iris_coverage": round(iris_result.feature_set.region_coverage, 4) if iris_result.valid else 0.0,
        "iris_status": iris_result.status.value,
        "iris_detection_ms": round(t_iris, 1),
        "total_ms": round(total_ms, 1),
    }


def run_pair_correspondence(
    image_a: np.ndarray,
    image_b: np.ndarray,
    fs_a: IrisFeatureSet,
    fs_b: IrisFeatureSet,
) -> Dict:
    """Run correspondence between two feature sets. Returns result dict."""
    t0 = time.perf_counter()
    result = estimate_correspondence(
        image_a, image_b, fs_a, fs_b,
        baseline=MatchingBaseline.GEOMETRIC_DESCRIPTOR,
        config=CORR_CONFIG,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "valid": result.valid,
        "rotation_deg": round(result.estimated_rotation_deg, 2) if result.valid else None,
        "scale": round(result.estimated_scale, 4) if result.valid else None,
        "confidence": result.confidence.value if hasattr(result, "confidence") else None,
        "failure": result.failure.value,
        "failure_reason": result.failure_reason,
        "n_matches": result.n_matches,
        "feature_count": result.feature_count,
        "angular_coverage": round(result.angular_coverage_ratio, 4) if result.angular_coverage_ratio else 0.0,
        "occupied_bins": result.occupied_angular_bins,
        "processing_ms": round(ms, 1),
    }


# ── Main ─────────────────────────────────────────────────────────────────── #

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PHASE XVIII — REAL ELITA PAIRED-IMAGE CYCLOTORSION WORKFLOW")
    print("=" * 72)
    print(f"Data directory: {DATA_DIR}")
    print()

    # ── Step 1: Detect on all images ──────────────────────────────────── #
    print("STEP 1 — Running detection on all clean images")
    print("-" * 72)

    detector = UnifiedDetector()
    iris_detector = IrisFeatureDetector()

    all_results: Dict[str, Dict] = {}
    for img_file in sorted(DATA_DIR.glob("eye_*.jpeg")):
        name = img_file.name
        try:
            image = load_image(img_file)
            res = run_full_pipeline(detector, iris_detector, image)
            all_results[name] = {**res, "image": image, "path": str(img_file)}
            status_str = f"features={res['iris_features']}" if res["iris_features"] > 0 else res["iris_status"]
            print(f"  {name:16s}  pupil={'YES' if res['has_pupil'] else 'NO':3s}  "
                  f"limbus={'YES' if res['has_limbus'] else 'NO':3s}  "
                  f"iris={status_str:20s}  "
                  f"total={res['total_ms']:.0f}ms")
        except Exception as e:
            all_results[name] = {"error": str(e), "path": str(img_file)}
            print(f"  {name:16s}  ERROR: {e}")

    # ── Step 2: Identify valid iris images ────────────────────────────── #
    print()
    print("STEP 2 — Valid iris images")
    print("-" * 72)

    valid_names = [n for n, r in all_results.items()
                   if r.get("iris_features", 0) >= 3 and "error" not in r]
    print(f"  Valid iris images: {len(valid_names)}/{len(all_results)}")
    for n in valid_names:
        r = all_results[n]
        print(f"    {n:16s}  features={r['iris_features']:3d}  "
              f"coverage={r['iris_coverage']:.4f}  "
              f"status={r['iris_status']}")

    # ── Step 3: Create all pairs from valid images ────────────────────── #
    print()
    print("STEP 3 — Pair correspondence")
    print("-" * 72)

    pair_results: List[Dict] = []
    pair_idx = 0
    for i, name_a in enumerate(valid_names):
        for name_b in valid_names[i + 1:]:
            pair_idx += 1
            ra = all_results[name_a]
            rb = all_results[name_b]

            img_a = ra["image"]
            img_b = rb["image"]
            fs_a = ra["iris_result"].feature_set
            fs_b = rb["iris_result"].feature_set

            corr = run_pair_correspondence(img_a, img_b, fs_a, fs_b)

            pair_record = {
                "pair_id": pair_idx,
                "image_a": name_a,
                "image_b": name_b,
                "features_a": ra["iris_features"],
                "features_b": rb["iris_features"],
                "coverage_a": ra["iris_coverage"],
                "coverage_b": rb["iris_coverage"],
                **corr,
            }
            pair_results.append(pair_record)

            status_char = "OK" if corr["valid"] else "REJ"
            rot_str = f"{corr['rotation_deg']:+.2f}°" if corr["valid"] else "---"
            print(f"  Pair {pair_idx:2d}: {name_a} <-> {name_b}  "
                  f"matches={corr['n_matches']:2d}  "
                  f"rotation={rot_str:8s}  "
                  f"failure={corr['failure']:18s}  "
                  f"[{status_char}]")

    # ── Step 4: Summary ───────────────────────────────────────────────── #
    print()
    print("STEP 4 — Summary")
    print("-" * 72)

    n_valid_pairs = sum(1 for p in pair_results if p["valid"])
    n_rejected = sum(1 for p in pair_results if not p["valid"])
    n_degenerate = sum(1 for p in pair_results if p["failure"] == "DEGENERATE")
    n_low_evidence = sum(1 for p in pair_results if p["failure"] == "LOW_EVIDENCE")
    n_low_ncc = sum(1 for p in pair_results if p["failure"] == "LOW_NCC")

    print(f"  Total pairs:           {len(pair_results)}")
    print(f"  Valid (rotation est.): {n_valid_pairs}")
    print(f"  Rejected:              {n_rejected}")
    print(f"    DEGENERATE:          {n_degenerate}")
    print(f"    LOW_EVIDENCE:        {n_low_evidence}")
    print(f"    LOW_NCC:             {n_low_ncc}")

    if n_valid_pairs > 0:
        rotations = [p["rotation_deg"] for p in pair_results if p["valid"]]
        print(f"  Rotation range:        [{min(rotations):+.2f}°, {max(rotations):+.2f}°]")
        print(f"  Rotation mean:         {np.mean(rotations):+.2f}°")
        print(f"  Rotation std:          {np.std(rotations):.2f}°")

    print(f"  Ground truth:          UNAVAILABLE")
    print(f"  Accuracy claims:       NONE (no ground truth)")

    # ── Step 5: Write outputs ─────────────────────────────────────────── #
    print()
    print("STEP 5 — Writing outputs")
    print("-" * 72)

    # Per-image CSV
    csv_path = OUTPUT_DIR / "detection_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "pupil_detected", "limbus_detected",
            "iris_features", "iris_coverage", "iris_status",
            "pupil_detection_ms", "iris_detection_ms", "total_ms",
        ])
        for name, r in all_results.items():
            if "error" in r:
                writer.writerow([name, "ERROR", "ERROR", 0, 0, r["error"], 0, 0, 0])
            else:
                writer.writerow([
                    name, r["has_pupil"], r["has_limbus"],
                    r["iris_features"], r["iris_coverage"], r["iris_status"],
                    r["pupil_detection_ms"], r["iris_detection_ms"], r["total_ms"],
                ])
    print(f"  Detection CSV:  {csv_path}")

    # Pair results CSV
    pairs_csv_path = OUTPUT_DIR / "pair_results.csv"
    if pair_results:
        with open(pairs_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_results[0].keys()))
            writer.writeheader()
            writer.writerows(pair_results)
        print(f"  Pairs CSV:      {pairs_csv_path}")

    # Full JSON
    json_path = OUTPUT_DIR / "full_results.json"
    # Remove non-serializable items
    json_data = {}
    for name, r in all_results.items():
        json_r = {k: v for k, v in r.items() if k not in ("image", "geometry")}
        json_data[name] = json_r
    with open(json_path, "w") as f:
        json.dump({"images": json_data, "pairs": pair_results}, f, indent=2, default=str)
    print(f"  Full JSON:      {json_path}")

    # ── Runtime breakdown ─────────────────────────────────────────────── #
    print()
    print("STEP 6 — Runtime")
    print("-" * 72)
    pupil_times = [r["pupil_detection_ms"] for r in all_results.values() if "pupil_detection_ms" in r]
    iris_times = [r["iris_detection_ms"] for r in all_results.values() if "iris_detection_ms" in r]
    total_times = [r["total_ms"] for r in all_results.values() if "total_ms" in r]
    corr_times = [p["processing_ms"] for p in pair_results]

    if pupil_times:
        print(f"  Pupil/limbus detection: mean={np.mean(pupil_times):.0f}ms  "
              f"range=[{min(pupil_times):.0f}, {max(pupil_times):.0f}]ms")
    if iris_times:
        print(f"  Iris detection:         mean={np.mean(iris_times):.0f}ms  "
              f"range=[{min(iris_times):.0f}, {max(iris_times):.0f}]ms")
    if total_times:
        print(f"  Total per image:        mean={np.mean(total_times):.0f}ms  "
              f"range=[{min(total_times):.0f}, {max(total_times):.0f}]ms")
    if corr_times:
        print(f"  Correspondence:         mean={np.mean(corr_times):.0f}ms  "
              f"range=[{min(corr_times):.0f}, {max(corr_times):.0f}]ms")

    print()
    print("=" * 72)
    print("PHASE XVIII COMPLETE — STOP")
    print("=" * 72)


if __name__ == "__main__":
    main()
