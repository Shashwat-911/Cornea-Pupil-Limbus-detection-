"""Clinical proxy smoke validation for the Phase I iris-feature detector.

Runs the full production pipeline (UnifiedDetector) on the 12 clean clinical
proxy images and then applies the iris-feature detector to each image's
pupil/limbus geometry, reporting ROI validity, candidate/accepted feature
counts, coverage, usable fraction and timing.

This is a *validation/debug* tool. It is NOT part of the automated test suite
because it requires the production ONNX model and clinical imagery, both of
which are gitignored and may be absent on any given machine.

Usage:
    python scripts/iris_feature_smoke.py [--only eye_01.jpeg]
"""

import argparse
import glob
import re
import sys
import time

import cv2

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="only run this image (e.g. eye_01.jpeg)")
    args = ap.parse_args()

    from pupil_tracking.core.detector import UnifiedDetector
    from pupil_tracking.iris import IrisConfig, detect_iris_features

    det = UnifiedDetector()
    cfg = IrisConfig()

    paths = sorted(glob.glob("clinical_data/clean/*.jpeg"))
    if args.only:
        paths = [p for p in paths if args.only in p]
    if not paths:
        print("no images matched")
        return 1

    print(
        f"{'image':<18}{'det':<5}{'pup_r':<8}{'lim_r':<8}"
        f"{'cand':<6}{'acc':<6}{'cov':<7}{'use':<7}{'ms':<8}{'status'}"
    )
    n_ok = n_roi = n_acc = 0
    total_ms = 0.0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path:<18}  (unreadable)")
            continue
        dr = det.detect(img, frame_number=0, source=path)
        pe = dr.pupil.ellipse if dr.has_pupil else None
        le = dr.limbus.ellipse if dr.has_limbus else None
        pup_r = round(pe.radius, 1) if pe is not None else 0.0
        lim_r = round(le.radius, 1) if le is not None else 0.0
        t0 = time.perf_counter()
        ir = detect_iris_features(img, pe, le, config=cfg)
        ms = (time.perf_counter() - t0) * 1000.0
        fs = ir.feature_set
        name = path.replace("\\", "/")
        n_acc += fs.num_accepted
        total_ms += ms
        if ir.valid:
            n_ok += 1
        if fs.roi.valid:
            n_roi += 1
        print(
            f"{name:<18}{str(dr.has_pupil and dr.has_limbus):<5}"
            f"{str(pup_r):<8}{str(lim_r):<8}{fs.num_candidates:<6}"
            f"{fs.num_accepted:<6}{fs.region_coverage:<7.3f}"
            f"{fs.usable_fraction:<7.3f}{ms:<8.1f}{ir.status.value}"
        )

    n = max(len(paths), 1)
    print("\n--- summary ---")
    print(f"valid iris results : {n_ok}/{len(paths)}")
    print(f"valid ROI           : {n_roi}/{len(paths)}")
    print(f"total accepted feats: {n_acc}")
    print(f"mean accepted/image : {n_acc / n:.1f}")
    print(f"mean ms (mask+xtr)  : {total_ms / n:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
