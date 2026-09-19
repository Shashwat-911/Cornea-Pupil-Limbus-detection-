"""Batch iris feature detection on clean clinical images.

Runs UnifiedDetector for pupil/limbus geometry, then detect_iris_features
for iris feature extraction on all eye_*.jpeg files in clinical_data/clean/.

Usage:
    python scripts/iris_batch_detect.py
"""

import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris.detect import detect_iris_features
from pupil_tracking.iris.config import IrisConfig


def main():
    image_dir = os.path.join(os.path.dirname(__file__), "..", "clinical_data", "clean")
    paths = sorted(glob.glob(os.path.join(image_dir, "eye_*.jpeg")))
    if not paths:
        print("No eye_*.jpeg images found in clinical_data/clean/")
        return 1

    print(f"Found {len(paths)} images\n")

    det = UnifiedDetector()
    cfg = IrisConfig()

    results = []
    total_iris_ms = 0.0
    total_detect_ms = 0.0

    for path in paths:
        filename = os.path.basename(path)

        t_load = time.perf_counter()
        img = cv2.imread(path)
        load_ms = (time.perf_counter() - t_load) * 1000.0

        if img is None:
            results.append({
                "filename": filename,
                "status": "UNREADABLE",
                "features": 0,
                "candidates": 0,
                "coverage": 0.0,
                "usable": 0.0,
                "confidence": 0.0,
                "detect_ms": 0.0,
                "iris_ms": 0.0,
                "total_ms": 0.0,
            })
            continue

        # Step 1: Unified detection for pupil/limbus geometry
        t_det = time.perf_counter()
        eye_result = det.detect(img, frame_number=0, source=filename)
        detect_ms = (time.perf_counter() - t_det) * 1000.0

        pupil = eye_result.pupil.ellipse if eye_result.has_pupil else None
        limbus = eye_result.limbus.ellipse if eye_result.has_limbus else None

        # Step 2: Iris feature detection
        t_iris = time.perf_counter()
        iris_result = detect_iris_features(img, pupil, limbus, config=cfg)
        iris_ms = (time.perf_counter() - t_iris) * 1000.0

        total_ms = detect_ms + iris_ms
        total_iris_ms += iris_ms
        total_detect_ms += detect_ms

        fs = iris_result.feature_set
        features = len(fs.features)
        coverage = fs.region_coverage
        usable = fs.usable_fraction

        # Confidence: average feature confidence if features exist
        conf = 0.0
        if features > 0:
            conf = sum(f.confidence for f in fs.features) / features

        results.append({
            "filename": filename,
            "status": iris_result.status.value,
            "features": features,
            "candidates": fs.num_candidates,
            "coverage": coverage,
            "usable": usable,
            "confidence": conf,
            "detect_ms": detect_ms,
            "iris_ms": iris_ms,
            "total_ms": total_ms,
        })

    # Sort by feature count descending
    results.sort(key=lambda r: r["features"], reverse=True)

    # Print per-image results
    print(f"{'filename':<18}{'status':<14}{'feat':<6}{'cand':<6}"
          f"{'cover':<8}{'usable':<8}{'conf':<8}{'det_ms':<9}{'iris_ms':<9}{'total':<8}")
    print("-" * 103)

    for r in results:
        print(
            f"{r['filename']:<18}{r['status']:<14}{r['features']:<6}{r['candidates']:<6}"
            f"{r['coverage']:<8.3f}{r['usable']:<8.3f}{r['confidence']:<8.3f}"
            f"{r['detect_ms']:<9.1f}{r['iris_ms']:<9.1f}{r['total_ms']:<8.1f}"
        )

    # Summary statistics
    n = len(results)
    valid = [r for r in results if r["status"] == "OK"]
    no_roi = [r for r in results if r["status"] == "NO_ROI"]
    no_feat = [r for r in results if r["status"] == "NO_FEATURES"]
    unreadable = [r for r in results if r["status"] == "UNREADABLE"]
    valid_features = [r for r in results if r["features"] >= 3]

    total_features = sum(r["features"] for r in results)
    total_candidates = sum(r["candidates"] for r in results)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total images            : {n}")
    print(f"OK (features > 0)      : {len(valid)}")
    print(f"NO_ROI                  : {len(no_roi)}")
    print(f"NO_FEATURES             : {len(no_feat)}")
    print(f"UNREADABLE              : {len(unreadable)}")
    print(f"")
    print(f"Total candidates        : {total_candidates}")
    print(f"Total accepted features : {total_features}")
    print(f"Mean features/image     : {total_features / max(n, 1):.1f}")
    print(f"Mean detect time        : {total_detect_ms / max(n, 1):.1f} ms")
    print(f"Mean iris time          : {total_iris_ms / max(n, 1):.1f} ms")
    print(f"")
    print(f"Images with valid iris features (features >= 3): {len(valid_features)}/{n}")
    if valid_features:
        print(f"  " + ", ".join(r["filename"] for r in valid_features))

    return 0


if __name__ == "__main__":
    sys.exit(main())
