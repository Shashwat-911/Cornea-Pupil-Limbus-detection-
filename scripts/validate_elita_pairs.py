#!/usr/bin/env python3
"""Minimal validation harness for real ELITA pre-dock/post-dock pairs.

Usage:
    python scripts/validate_elita_pairs.py manifest.json [--output results.json]

The manifest must be a JSON file with the structure defined in
IRIS_PHASE13_ELITA_VALIDATION_REPORT.md §6.

This script does NOT modify the algorithm. It only exercises the existing
pipeline on whatever data the manifest points to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure the repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.correspondence import estimate_correspondence
from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.correspondence import CorrespondenceConfig
from pupil_tracking.iris.detect import detect_iris_features


def _load_image(path: str) -> Optional[np.ndarray]:
    """Load an image from disk, returning BGR or None on failure."""
    if not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img if img is not None and img.size > 0 else None


def _run_pipeline(
    image: np.ndarray,
    detector: UnifiedDetector,
    iris_config: Optional[IrisConfig] = None,
) -> Dict[str, Any]:
    """Run pupil/limbus detection + iris feature extraction on one image."""
    t0 = time.perf_counter()
    det_result = detector.detect(image)
    t_det = time.perf_counter() - t0

    # UnifiedDetector returns PupilDetection/LimbusDetection; extract ellipses
    pupil = det_result.pupil.ellipse if det_result.pupil else None
    limbus = det_result.limbus.ellipse if det_result.limbus else None

    if pupil is None or limbus is None:
        return {
            "pupil_detected": pupil is not None,
            "limbus_detected": limbus is not None,
            "feature_count": 0,
            "angular_coverage_ratio": 0.0,
            "largest_angular_gap": 0.0,
            "detection_time_ms": round(t_det * 1000, 1),
            "iris_time_ms": 0.0,
            "error": "pupil/limbus not detected",
        }

    t1 = time.perf_counter()
    iris_result = detect_iris_features(image, pupil, limbus, config=iris_config)
    t_iris = time.perf_counter() - t1

    fs = iris_result.feature_set
    n_features = len(fs.features) if fs is not None else 0

    coverage = 0.0
    largest_gap = 0.0
    if fs is not None and n_features >= 2:
        angles = np.array([f.angle_deg for f in fs.features])
        angles_sorted = np.sort(angles % 360.0)
        gaps = np.diff(angles_sorted, append=angles_sorted[0] + 360.0)
        largest_gap = float(np.max(gaps))
        coverage = float(1.0 - largest_gap / 360.0)

    return {
        "pupil_detected": True,
        "limbus_detected": True,
        "pupil_radius_px": round(
            (pupil.semi_major + pupil.semi_minor) / 2.0, 1
        ),
        "limbus_radius_px": round(
            (limbus.semi_major + limbus.semi_minor) / 2.0, 1
        ),
        "feature_count": n_features,
        "angular_coverage_ratio": round(coverage, 4),
        "largest_angular_gap_deg": round(largest_gap, 2),
        "detection_time_ms": round(t_det * 1000, 1),
        "iris_time_ms": round(t_iris * 1000, 1),
        "error": None,
    }


def validate_pair(
    pair: Dict[str, Any],
    detector: UnifiedDetector,
    iris_config: Optional[IrisConfig] = None,
    corr_config: Optional[CorrespondenceConfig] = None,
) -> Dict[str, Any]:
    """Validate a single pre-dock / post-dock pair."""
    pair_id = pair.get("pair_id", "unknown")
    eye_side = pair.get("eye_side", "unknown")
    pre_path = pair.get("pre_dock_image", "")
    post_path = pair.get("post_dock_image", "")
    ref_rot = pair.get("reference_rotation_deg")

    result: Dict[str, Any] = {
        "pair_id": pair_id,
        "eye_side": eye_side,
        "pre_dock_path": pre_path,
        "post_dock_path": post_path,
        "reference_rotation_deg": ref_rot,
    }

    # Load images
    img_pre = _load_image(pre_path)
    img_post = _load_image(post_path)

    if img_pre is None:
        result["error"] = f"cannot load pre-dock image: {pre_path}"
        return result
    if img_post is None:
        result["error"] = f"cannot load post-dock image: {post_path}"
        return result

    # Run detection on both images
    pre_info = _run_pipeline(img_pre, detector, iris_config)
    post_info = _run_pipeline(img_post, detector, iris_config)

    result["pre_dock"] = pre_info
    result["post_dock"] = post_info

    if pre_info.get("error"):
        result["error"] = f"pre-dock: {pre_info['error']}"
        return result
    if post_info.get("error"):
        result["error"] = f"post-dock: {post_info['error']}"
        return result

    # Re-detect to get full feature sets for correspondence
    t0 = time.perf_counter()
    det_pre = detector.detect(img_pre)
    det_post = detector.detect(img_post)
    # UnifiedDetector returns PupilDetection/LimbusDetection; extract ellipses
    pupil_pre = det_pre.pupil.ellipse if det_pre.pupil else None
    limbus_pre = det_pre.limbus.ellipse if det_pre.limbus else None
    pupil_post = det_post.pupil.ellipse if det_post.pupil else None
    limbus_post = det_post.limbus.ellipse if det_post.limbus else None
    iris_pre = detect_iris_features(
        img_pre, pupil_pre, limbus_pre, config=iris_config
    )
    iris_post = detect_iris_features(
        img_post, pupil_post, limbus_post, config=iris_config
    )

    fs_pre = iris_pre.feature_set
    fs_post = iris_post.feature_set

    if fs_pre is None or len(fs_pre.features) < 4:
        result["error"] = "pre-dock: insufficient features"
        return result
    if fs_post is None or len(fs_post.features) < 4:
        result["error"] = "post-dock: insufficient features"
        return result

    # Run correspondence
    res = estimate_correspondence(
        img_pre, img_post, fs_pre, fs_post,
        rotation_method="global_hybrid",
        config=corr_config,
    )
    t_total = time.perf_counter() - t0

    result["correspondence"] = {
        "estimated_rotation_deg": round(res.estimated_rotation_deg, 3),
        "estimated_scale": round(res.estimated_scale, 4),
        "failure": res.failure.value,
        "failure_reason": res.failure_reason,
        "valid": res.valid,
        "feature_count_pre": len(fs_pre.features),
        "feature_count_post": len(fs_post.features),
        "global_inlier_count": res.global_inlier_count,
        "global_inlier_fraction": round(
            res.global_inlier_frac, 4
        ) if res.global_inlier_frac is not None else None,
        "processing_time_ms": round(t_total * 1000, 1),
    }

    # Compute error if reference rotation is available
    if ref_rot is not None:
        from pupil_tracking.iris.correspondence import circular_distance
        err = circular_distance(res.estimated_rotation_deg, float(ref_rot))
        result["correspondence"]["circular_error_deg"] = round(err, 3)
        result["correspondence"]["success_0_5_deg"] = err <= 0.5
        result["correspondence"]["success_1_0_deg"] = err <= 1.0
        result["correspondence"]["success_2_0_deg"] = err <= 2.0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate ELITA pre-dock/post-dock pairs"
    )
    parser.add_argument("manifest", help="Path to manifest JSON")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON path (default: stdout)",
    )
    parser.add_argument(
        "--evidence-gate", action="store_true",
        help="Enable evidence gate (disabled by default)",
    )
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    pairs = manifest.get("pairs", [])
    if not pairs:
        print("No pairs in manifest.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(pairs)} pairs from {args.manifest}", file=sys.stderr)

    detector = UnifiedDetector()
    iris_config = IrisConfig()

    corr_config_kwargs = {}
    if args.evidence_gate:
        corr_config_kwargs["evidence_gate"] = True
    corr_config = (
        CorrespondenceConfig(**corr_config_kwargs)
        if corr_config_kwargs
        else None
    )

    results = []
    for pair in pairs:
        pair_id = pair.get("pair_id", "?")
        print(f"  Validating {pair_id}...", file=sys.stderr, end=" ")
        r = validate_pair(pair, detector, iris_config, corr_config)
        status = "OK" if r.get("error") is None else f"FAIL: {r['error']}"
        print(status, file=sys.stderr)
        results.append(r)

    output = {
        "manifest": args.manifest,
        "total_pairs": len(pairs),
        "successful": sum(1 for r in results if r.get("error") is None),
        "failed": sum(1 for r in results if r.get("error") is not None),
        "evidence_gate_enabled": args.evidence_gate,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
