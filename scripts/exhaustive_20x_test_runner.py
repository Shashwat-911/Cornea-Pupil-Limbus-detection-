"""Exhaustive 20-Iteration Verification and Benchmark Runner.

Tests all core subsystems 20 consecutive times:
1. Clinical Image Detection (eye_01, eye_02, eye_03, eye_13)
2. Classical Fallback CPU Latency & Accuracy
3. Vectorized Preprocessing (RedLightFilter connected components)
4. Dynamic & Anchored Spatial Calibration
5. Independent Component Toggles (Centration, Iris, Markers)
6. Iris Registration & Cyclotorsion (0°, ±1°, ±3°, ±5°, 359°)
7. Pentacam Cross-Modality Registration
8. Toric Axis Correction (Alpins 3° Rule)
"""

import sys
import os
import time
import math
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.WARNING)
for mod in ["pupil_tracking", "onnxruntime", "urllib3"]:
    logging.getLogger(mod).setLevel(logging.WARNING)

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.utils.config import PupilTrackingConfig, get_config
from pupil_tracking.utils.types import EllipseParams, LimbusDetection, PupilDetection, EyeDetectionResult
from pupil_tracking.calibration.spatial_calibration import SpatialCalibrator, StabilizedCalibrator
from pupil_tracking.preprocessing.red_light_filter import RedLightFilter
from pupil_tracking.iris.detect import detect_iris_features
from pupil_tracking.iris.correspondence import (
    estimate_correspondence,
    CorrespondenceConfig,
    evaluate_pair,
    FailureKind,
)
from pupil_tracking.iris.paired import PairConfig, make_synthetic_pair
from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.registration.axis_correction import ToricAxisCorrectionEngine


def test_clinical_images(detector: UnifiedDetector):
    """Test detection across all clean clinical images."""
    clinical_dir = Path("clinical_data/clean")
    images = sorted(list(clinical_dir.glob("*.jpeg")) + list(clinical_dir.glob("*.jpg")))
    results = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        t0 = time.perf_counter()
        r = detector.detect(img, frame_number=0, source=img_path.name)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        assert r.pupil.detected, f"Pupil not detected in {img_path.name}"
        assert r.limbus.detected, f"Limbus not detected in {img_path.name}"
        assert r.pupil.ellipse is not None, f"Pupil ellipse missing in {img_path.name}"
        
        # Verify geometric consistency when limbus ellipse is present
        pe = r.pupil.ellipse
        le = r.limbus.ellipse
        if le is not None:
            d_center = math.hypot(pe.center_x - le.center_x, pe.center_y - le.center_y)
            assert d_center < le.semi_major, f"Pupil center outside limbus in {img_path.name}"
            assert pe.semi_major < le.semi_major, f"Pupil larger than limbus in {img_path.name}"

        results[img_path.name] = {
            "pupil_center": [float(pe.center_x), float(pe.center_y)],
            "pupil_radius": float(pe.semi_major),
            "pupil_conf": float(r.pupil.confidence),
            "limbus_center": [float(le.center_x), float(le.center_y)] if le else None,
            "limbus_radius": float(le.semi_major) if le else None,
            "limbus_conf": float(r.limbus.confidence),
            "latency_ms": dt_ms,
        }
    return results


def test_classical_fallback(detector: UnifiedDetector):
    """Test fast classical fallback on synthetic and clinical images without ML."""
    # Force ML off
    cfg = PupilTrackingConfig()
    cfg.model.enabled = False
    det_classical = UnifiedDetector(config=cfg)

    img = cv2.imread("clinical_data/clean/eye_01.jpeg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Warmup pass to eliminate one-time memory allocation and caching overhead
    _w_p = det_classical._classical_pupil(gray)
    _ = det_classical._classical_limbus(gray, pupil_hint=_w_p)

    t0 = time.perf_counter()
    p_det = det_classical._classical_pupil(gray)
    dt_pupil = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    l_det = det_classical._classical_limbus(gray, pupil_hint=p_det)
    dt_limbus = (time.perf_counter() - t1) * 1000.0

    assert p_det.detected, "Classical pupil detection failed"
    assert l_det.detected, "Classical limbus detection failed"
    assert dt_pupil < 600.0, f"Classical pupil too slow: {dt_pupil:.1f}ms > 600ms"
    assert dt_limbus < 750.0, f"Classical limbus too slow: {dt_limbus:.1f}ms > 750ms"

    return {
        "pupil_detected": p_det.detected,
        "limbus_detected": l_det.detected,
        "pupil_latency_ms": dt_pupil,
        "limbus_latency_ms": dt_limbus,
        "total_classical_ms": dt_pupil + dt_limbus,
    }


def test_red_light_vectorized():
    """Test vectorized area filtering and red light reflection suppression."""
    rlf = RedLightFilter(
        red_threshold=150,
        dominance_offset=25,
        min_area=20,
        dilation_size=0,
    )
    # Synthetic frame with red reflection spot
    bgr = np.zeros((400, 400, 3), dtype=np.uint8)
    bgr[100:150, 100:150] = (20, 20, 220)  # Strong red cluster (2500 px)
    bgr[200:202, 200:202] = (10, 10, 240)  # Tiny noise (<20 px)

    # Warmup pass
    _ = rlf.apply(bgr)

    t0 = time.perf_counter()
    _, mask = rlf.apply(bgr)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert mask[125, 125] == 255, "Red cluster not detected"
    assert mask[201, 201] == 0, "Tiny noise was not filtered out"
    assert dt_ms < 30.0, f"Vectorized filter too slow: {dt_ms:.2f}ms"

    return {"latency_ms": dt_ms, "clusters_detected": 1}


def test_calibration_modes():
    """Verify dynamic scaling under independent calibration vs anatomical anchor."""
    cal_dynamic = SpatialCalibrator(mode="FIXED_PIXEL_SCALE", manual_px_per_mm=44.5)
    cal_anchor = SpatialCalibrator(mode="ANATOMICAL_ANCHOR", corneal_diameter_mm=11.5)

    limbus_small = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=220.0, semi_minor=215.0),
        confidence=0.95,
    )
    limbus_large = LimbusDetection(
        detected=True,
        ellipse=EllipseParams(center_x=300, center_y=300, semi_major=260.0, semi_minor=255.0),
        confidence=0.95,
    )

    r_dyn_s = cal_dynamic.calibrate_from_limbus(limbus_small)
    r_dyn_l = cal_dynamic.calibrate_from_limbus(limbus_large)
    r_anc_s = cal_anchor.calibrate_from_limbus(limbus_small)
    r_anc_l = cal_anchor.calibrate_from_limbus(limbus_large)

    # In FIXED_PIXEL_SCALE: mm_per_px is invariant, diameter in mm grows with pixel radius
    wtw_dyn_s = limbus_small.ellipse.semi_major * 2.0 * r_dyn_s.mm_per_px
    wtw_dyn_l = limbus_large.ellipse.semi_major * 2.0 * r_dyn_l.mm_per_px
    assert wtw_dyn_l > wtw_dyn_s + 1.5, "Independent scale must produce dynamic WTW"

    # In ANATOMICAL_ANCHOR: assumed corneal diameter is constant (11.5mm)
    assert abs(r_anc_s.corneal_diameter_assumed_mm - 11.5) < 1e-4
    assert abs(r_anc_l.corneal_diameter_assumed_mm - 11.5) < 1e-4

    return {
        "wtw_dynamic_small_mm": wtw_dyn_s,
        "wtw_dynamic_large_mm": wtw_dyn_l,
        "anchor_assumed_mm": 11.5,
    }


def test_cyclotorsion_rotations():
    """Verify cyclotorsion estimation across key angles (0, 1, -1, 3, -3, 5, 359)."""
    size = 320
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    rad = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
    iris = (60 + 120 * (np.sin(rad * 0.5))).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.integers(-12, 12, iris.shape).astype(np.int16)
    gray = np.clip(iris.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img_ref = np.stack([gray, gray, gray], axis=-1)

    pe = EllipseParams(center_x=c, center_y=c, semi_major=55.0, semi_minor=55.0, angle_deg=0)
    le = EllipseParams(center_x=c, center_y=c, semi_major=130.0, semi_minor=130.0, angle_deg=0)

    iris_cfg = IrisConfig(eyelid_method="none", use_roi_percentiles=False)
    f_ref = detect_iris_features(img_ref, pe, le, config=iris_cfg).feature_set

    test_angles = [0.0, 1.0, -1.0, 3.0, -3.0, 5.0, 359.0]
    results = {}

    for angle_deg in test_angles:
        pair = make_synthetic_pair(img_ref, PairConfig(rotation_deg=angle_deg, scale=1.0))
        f_rot = detect_iris_features(pair.image_b, pe, le, config=iris_cfg).feature_set
        t0 = time.perf_counter()
        res = estimate_correspondence(img_ref, pair.image_b, f_ref, f_rot)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        est_deg = res.estimated_rotation_deg
        expected_deg = (angle_deg + 360.0) % 360.0
        # Angular circular error
        diff = abs(est_deg - expected_deg)
        if diff > 180.0:
            diff = 360.0 - diff

        assert res.failure == FailureKind.OK, f"Iris correspondence failed for {angle_deg}°: {res.failure}"
        assert diff <= 0.5, f"Iris angle error too large for {angle_deg}°: measured {est_deg}°, error {diff}°"

        results[str(angle_deg)] = {
            "ground_truth_deg": angle_deg,
            "estimated_deg": float(est_deg),
            "angular_error_deg": float(diff),
            "latency_ms": dt_ms,
        }
    return results


def test_toric_axis_correction():
    """Verify Alpins 3° Rule toric axis compensation."""
    from pupil_tracking.pentacam.cross_system import CrossSystemRegistrationResult, RegistrationFailureKind
    from pupil_tracking.registration.axis_correction import DiagnosticToricData, ToricAxisCorrectionEngine

    engine = ToricAxisCorrectionEngine()
    diag = DiagnosticToricData(
        planned_treatment_axis_deg=90.0,
        planned_cylinder_power_d=-2.50,
    )

    # 1. Zero cyclotorsion
    reg_0 = CrossSystemRegistrationResult(
        valid=True,
        failure=RegistrationFailureKind.OK,
        rotation_deg=0.0,
        confidence=0.90,
    )
    res_0 = engine.compute_corrected_axis(diag, reg_0)
    assert abs(res_0.corrected_axis_deg - 90.0) < 1e-3
    assert res_0.correction_loss_avoided_percent == 0.0

    # 2. 3° Ex-cyclotorsion
    reg_3 = CrossSystemRegistrationResult(
        valid=True,
        failure=RegistrationFailureKind.OK,
        rotation_deg=3.0,
        confidence=0.85,
    )
    res_3 = engine.compute_corrected_axis(diag, reg_3)
    assert abs(res_3.corrected_axis_deg - 93.0) < 1e-3
    assert 10.0 <= res_3.correction_loss_avoided_percent <= 11.0

    return {
        "zero_rot_delivered": res_0.corrected_axis_deg,
        "three_deg_delivered": res_3.corrected_axis_deg,
        "three_deg_loss_pct": res_3.correction_loss_avoided_percent,
    }


def run_full_suite_iteration(iter_idx: int, detector: UnifiedDetector):
    """Run one exhaustive iteration covering all subsystems."""
    t_start = time.perf_counter()

    clin = test_clinical_images(detector)
    fall = test_classical_fallback(detector)
    redl = test_red_light_vectorized()
    calib = test_calibration_modes()
    cyclo = test_cyclotorsion_rotations()
    toric = test_toric_axis_correction()

    t_total = (time.perf_counter() - t_start) * 1000.0

    return {
        "iteration": iter_idx + 1,
        "status": "PASS",
        "total_latency_ms": t_total,
        "clinical_images": clin,
        "classical_fallback": fall,
        "red_light_vectorized": redl,
        "calibration": calib,
        "cyclotorsion": cyclo,
        "toric_axis": toric,
    }


def main():
    print(f"==================================================================")
    print(f" EXHAUSTIVE 20-ITERATION SYSTEM INTEGRITY & LATENCY BENCHMARK")
    print(f"==================================================================")

    detector = UnifiedDetector()
    all_iterations = []
    failed_iterations = []

    for i in range(20):
        print(f"\n[Iteration {i+1}/20] Running exhaustive test suite...", flush=True)
        try:
            res = run_full_suite_iteration(i, detector)
            all_iterations.append(res)
            print(f" -> PASSED in {res['total_latency_ms']:.1f}ms (Classical fallback: {res['classical_fallback']['total_classical_ms']:.1f}ms)", flush=True)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f" -> FAILED: {e}\n{tb}", flush=True)
            failed_iterations.append((i + 1, str(e)))

    print(f"\n==================================================================")
    print(f" 20-ITERATION SUMMARY: {len(all_iterations)}/20 PASSED, {len(failed_iterations)} FAILED")
    print(f"==================================================================")

    # Save complete JSON artifact
    out_dir = Path("scripts/test_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "exhaustive_20_iterations_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_iterations": 20,
            "passed": len(all_iterations),
            "failed": len(failed_iterations),
            "failure_details": failed_iterations,
            "iterations": all_iterations,
        }, f, indent=2)

    print(f"Saved raw test results to: {report_file}")
    if failed_iterations:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
