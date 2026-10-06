"""Reproducible CPU benchmark; private images stay local, report contains no names.

Known digital rotations are numerical tests, NOT clinical torsion ground truth.
Run from repository root: python -m scripts.benchmark_sitting --data-root .
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.session import SittingRegistrationSession
from pupil_tracking.utils.types import EyeDetectionResult, PupilDetection, LimbusDetection


def eye_detection(result):
    return EyeDetectionResult(
        pupil=PupilDetection(detected=result.geometry.pupil_detected, ellipse=result.geometry.pupil),
        limbus=LimbusDetection(detected=result.geometry.limbus_detected, ellipse=result.geometry.limbus))


def transform_detection(result, matrix, angle):
    r = copy.deepcopy(result)
    for e in (r.geometry.pupil, r.geometry.limbus):
        e.center_x, e.center_y = map(float, matrix @ [e.center_x, e.center_y, 1])
        e.angle_deg = (e.angle_deg - angle) % 180
    return eye_detection(r)


def distribution(values):
    if not values:
        return None
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def baseline_class(ref, path, name):
    # Read a historical tracked file without checking out or changing user files.
    source = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True, encoding="utf-8")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("_sitting_baseline_" + name, loader=None))
    sys.modules[module.__name__] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return getattr(module, name)


def run(args):
    cv2.setNumThreads(args.threads)
    root = Path(args.data_root)
    files = sorted(root.glob("*.BMP")) + sorted(root.glob("sitting *.jpeg"))
    unique, seen = [], set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    if not unique:
        raise ValueError("No BMP or sitting JPEG inputs found")
    detector = PentacamIrisDetector()
    engine = CrossModalityRegistrationEngine(num_angles=720)
    old_detector = old_engine = None
    if args.baseline_ref:
        old_detector = baseline_class(args.baseline_ref, "pupil_tracking/pentacam/detector.py", "PentacamIrisDetector")()
        old_engine = baseline_class(args.baseline_ref, "pupil_tracking/pentacam/cross_registration.py", "CrossModalityRegistrationEngine")(num_angles=720)
    report = {
        "scope": "Digital rigid rotations of real screenshots; no clinical ground truth or confirmed cross-device pairs",
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__,
                        "platform": platform.system(), "processor": platform.processor(), "opencv_threads": args.threads},
        "baseline_ref": args.baseline_ref,
        "images": [], "trials": [], "negative_controls": [], "additional_screenshots": [],
    }
    images, detections = [], []
    for i, path in enumerate(unique):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unreadable input number {i}")
        # Warm up each implementation before repeat latency measurements.
        result = detector.detect(image)
        times = [detector.detect(image).processing_time_ms for _ in range(args.repeats)]
        old_times = []
        if old_detector:
            old_detector.detect(image)
            old_times = [old_detector.detect(image).processing_time_ms for _ in range(args.repeats)]
        report["images"].append({"id": f"sitting_{i:02}", "shape": image.shape, "valid": result.valid,
                                  "failure": result.failure_reason, "detection_ms": distribution(times),
                                  "baseline_detection_ms": distribution(old_times),
                                  "pupil_radius_px": result.geometry.pupil_radius_px,
                                  "limbus_radius_px": result.geometry.limbus_radius_px})
        images.append(image)
        detections.append(result)
        if not result.valid:
            continue
        session = SittingRegistrationSession()
        session.set_reference(image, eye_id=str(i), laterality="OD")
        session.register(image, eye_id=str(i), laterality="OD")
        h, w = image.shape[:2]
        center = (result.geometry.pupil.center_x, result.geometry.pupil.center_y)
        for condition in ("rigid", "illumination_translation_noise"):
            for angle in (-12., -5., -.75, 0., .75, 5., 12.):
                mat = cv2.getRotationMatrix2D(center, angle, 1)
                if condition != "rigid":
                    mat[:, 2] += [7, -5]
                current = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR)
                if condition != "rigid":
                    rng = np.random.default_rng(910 + i)
                    current = np.clip((current.astype(float) / 255) ** 1.25 * 225 + 8
                                      + rng.normal(0, 1.5, current.shape), 0, 255).astype(np.uint8)
                known = transform_detection(result, mat, angle)
                detected = detector.detect(current)
                for mode, current_detection in (("fixed_geometry", known), ("automatic_geometry", eye_detection(detected))):
                    if mode == "automatic_geometry" and not detected.valid:
                        report["trials"].append({"image": i, "condition": condition, "mode": mode,
                                                  "target_deg": angle, "valid": False, "failure": detected.failure_reason})
                        continue
                    res = engine.register(image, current, result, current_detection)
                    row = {"image": i, "condition": condition, "mode": mode, "target_deg": angle,
                           "valid": res.valid, "angle_deg": res.rotation_deg, "error_deg": abs(res.rotation_deg - angle),
                           "registration_ms": res.processing_time_ms, "failure": res.failure_reason}
                    if mode == "automatic_geometry":
                        row["detection_plus_registration_ms"] = detected.processing_time_ms + res.processing_time_ms
                    if old_engine:
                        baseline = old_engine.register(image, current, result, current_detection)
                        row.update(baseline_valid=baseline.valid, baseline_error_deg=abs(baseline.rotation_deg - angle),
                                   baseline_registration_ms=baseline.processing_time_ms)
                    report["trials"].append(row)
                session_result = session.register(current, eye_id=str(i), laterality="OD")
                report["trials"].append({"image": i, "condition": condition, "mode": "bounded_session",
                    "target_deg": angle, "valid": session_result.valid,
                    "angle_deg": session_result.rotation_deg,
                    "error_deg": abs(session_result.rotation_deg - angle),
                    "registration_ms": session_result.processing_time_ms,
                    "detection_plus_registration_ms": session_result.processing_time_ms,
                    "failure": session_result.failure_reason})
        print(f"Completed seated image {i + 1}/{len(unique)}", flush=True)
    # Different images are unconfirmed identity controls (not a clinical FAR set).
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if detections[i].valid and detections[j].valid:
                r = engine.register(images[i], images[j], detections[i], eye_detection(detections[j]))
                report["negative_controls"].append({"reference": i, "current": j, "accepted": r.valid})
    for i, path in enumerate(sorted(root.glob("WhatsApp*.jpeg"))):
        im = cv2.imread(str(path))
        r = detector.detect(im)
        report["additional_screenshots"].append({"id": f"additional_{i:02}", "valid": r.valid,
                                                "failure": r.failure_reason,
                                                "note": "Unpaired ELITA screenshot; not seated accuracy ground truth"})
    summary = {}
    for mode in ("fixed_geometry", "automatic_geometry", "bounded_session"):
        rows = [r for r in report["trials"] if r["mode"] == mode]
        good = [r for r in rows if r["valid"]]
        summary[mode] = {"total": len(rows), "accepted": len(good),
                         "accepted_error_deg": distribution([r["error_deg"] for r in good]),
                         "registration_ms": distribution([r["registration_ms"] for r in good]),
                         "accepted_above_1deg": sum(r["error_deg"] > 1 for r in good),
                         "baseline_accepted": sum(r.get("baseline_valid", False) for r in rows),
                         "baseline_accepted_error_deg": distribution([r["baseline_error_deg"] for r in rows if r.get("baseline_valid")]),
                         "baseline_registration_ms": distribution([r["baseline_registration_ms"] for r in rows if r.get("baseline_valid")]),
                         "detection_plus_registration_ms": distribution([r["detection_plus_registration_ms"] for r in good if "detection_plus_registration_ms" in r])}
    summary["unconfirmed_pair_acceptances"] = sum(r["accepted"] for r in report["negative_controls"])
    summary["unconfirmed_pairs"] = len(report["negative_controls"])
    report["summary"] = summary
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--report", default="output/sitting/benchmark.json")
    parser.add_argument("--baseline-ref")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
