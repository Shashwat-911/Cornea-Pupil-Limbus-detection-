"""Backend Phase 3 — Detection Quality Audit (instrumentation only)."""

import json
import os
import sys
import time
import glob
import math

import cv2
import numpy as np

# Suppress noisy logging
import logging
logging.getLogger().setLevel(logging.WARNING)
for name in ["pupil_tracking", "onnxruntime", "urllib3"]:
    logging.getLogger(name).setLevel(logging.WARNING)

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.core.smart_fitter import SmartContourFitter, FitResult
from pupil_tracking.core.validation import cross_validate_and_reject
from pupil_tracking.utils.types import EyeDetectionResult


# ─── Helpers ───────────────────────────────────────────────────

def contour_stats(mask, label=""):
    """Compact contour statistics from a binary mask."""
    if mask is None or mask.sum() == 0:
        return {"count": 0, "total_area": 0, "largest_area": 0}
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
    return {
        "count": len(contours),
        "total_area": int(sum(areas)),
        "largest_area": int(areas[0]) if areas else 0,
        "areas_top5": [int(a) for a in areas[:5]],
    }


def mask_info(mask, name=""):
    """Compact mask statistics."""
    if mask is None:
        return {"name": name, "shape": None, "fg_pixels": 0}
    return {
        "name": name,
        "shape": list(mask.shape),
        "fg_pixels": int((mask > 0).sum()),
        "fg_pct": round(float((mask > 0).sum()) / max(mask.size, 1) * 100, 3),
    }


def fit_summary(fit_result, label=""):
    """Compact FitResult summary."""
    if fit_result is None:
        return None
    return {
        "label": label,
        "valid": fit_result.valid,
        "center": (round(fit_result.center_x, 2), round(fit_result.center_y, 2)),
        "radius": round(fit_result.radius, 2),
        "fit_type": fit_result.fit_type.value if hasattr(fit_result.fit_type, 'value') else str(fit_result.fit_type),
        "fit_quality": round(fit_result.fit_quality, 4),
        "num_contour_points": fit_result.num_contour_points,
        "circularity": round(fit_result.circularity, 4),
        "eccentricity": round(fit_result.eccentricity, 4),
        "rms_residual": round(fit_result.fit_rms_residual, 3),
    }


# ─── Main Audit ────────────────────────────────────────────────

def audit_image(det, fitter, img_path):
    """Run a single image through the full pipeline and collect diagnostics."""
    name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        return {"name": name, "error": "failed to read"}

    h, w = img.shape[:2]
    t0 = time.time()

    # ── A. Input ────────────────────────────────────────────
    audit = {
        "name": name,
        "input": {"width": w, "height": h, "channels": img.shape[2] if img.ndim == 3 else 1},
    }

    # ── B. Ring detection ───────────────────────────────────
    ring_result = det._detect_ring(img)
    audit["ring"] = {
        "status": ring_result.status.value,
        "confidence": round(ring_result.confidence, 4),
        "center": (round(ring_result.ring_center[0], 2), round(ring_result.ring_center[1], 2)) if ring_result.ring_center else None,
        "radius": round(ring_result.ring_radius, 2) if ring_result.ring_radius else None,
        "inner_radius": round(ring_result.ring_inner_radius, 2) if ring_result.ring_inner_radius else None,
    }

    # ── C. Preprocessing ────────────────────────────────────
    t_prep = time.time()
    prep_result = det._ring_preprocessor.preprocess(img, ring_result)
    t_prep = (time.time() - t_prep) * 1000
    audit["preprocessing_ms"] = round(t_prep, 1)

    # ── D. ML segmentation ──────────────────────────────────
    t_ml = time.time()
    ml_result = det.ml_engine.detect(img, frame_number=-1, source=name)
    t_ml = (time.time() - t_ml) * 1000
    audit["ml_inference_ms"] = round(t_ml, 1)

    raw_mask = ml_result._raw_mask if hasattr(ml_result, '_raw_mask') and ml_result._raw_mask is not None else np.zeros((h, w), dtype=np.uint8)

    pupil_mask = (raw_mask == 1).astype(np.uint8) * 255
    iris_mask = ((raw_mask == 2) | (raw_mask == 1)).astype(np.uint8) * 255
    ring_mask_ml = (raw_mask == 3).astype(np.uint8) * 255

    audit["ml_masks"] = {
        "pupil": mask_info(pupil_mask, "pupil"),
        "iris": mask_info(iris_mask, "iris"),
        "ring": mask_info(ring_mask_ml, "ring"),
        "raw_mask_values": {
            int(k): int((raw_mask == k).sum())
            for k in range(4)
            if (raw_mask == k).sum() > 0
        },
    }

    # ── E. Contours ─────────────────────────────────────────
    pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    iris_contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    audit["contours"] = {
        "pupil": contour_stats(pupil_mask, "pupil"),
        "iris": contour_stats(iris_mask, "iris"),
    }

    # ── F. Structure extraction (SmartContourFitter) ────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t_fit = time.time()
    pupil_fit, limbus_fit = det._extract_structure(raw_mask, gray, ring_result=ring_result)
    t_fit = (time.time() - t_fit) * 1000
    audit["fitting_ms"] = round(t_fit, 1)
    audit["fits"] = {
        "pupil": fit_summary(pupil_fit, "pupil"),
        "limbus": fit_summary(limbus_fit, "limbus"),
    }

    # ── G. Full pipeline detection ──────────────────────────
    t_detect = time.time()
    result = det.detect(img, frame_number=-1, source=name)
    t_detect = (time.time() - t_detect) * 1000
    audit["total_detect_ms"] = round(t_detect, 1)

    # ── H. Final result ─────────────────────────────────────
    p = result.pupil
    l = result.limbus
    cc = result.corneal_center
    cal = result.calibration

    audit["result"] = {
        "pupil": {
            "detected": p.detected,
            "center": (round(p.ellipse.center_x, 2), round(p.ellipse.center_y, 2)) if p.ellipse else None,
            "radius": round(p.ellipse.radius, 2) if p.ellipse else None,
            "diameter_mm": round(p.radius_mm * 2, 3) if p.radius_mm else None,
            "confidence": round(p.confidence, 4),
            "quality": getattr(p.quality, 'name', str(p.quality)) if p.quality else None,
            "method": getattr(p.method, 'name', str(p.method)) if p.method else None,
        },
        "limbus": {
            "detected": l.detected,
            "center": (round(l.ellipse.center_x, 2), round(l.ellipse.center_y, 2)) if l.ellipse else None,
            "radius": round(l.ellipse.radius, 2) if l.ellipse else None,
            "diameter_mm": round(l.radius_mm * 2, 3) if l.radius_mm else None,
            "confidence": round(l.confidence, 4),
            "quality": getattr(l.quality, 'name', str(l.quality)) if l.quality else None,
            "method": getattr(l.method, 'name', str(l.method)) if l.method else None,
        },
        "corneal_center": {
            "center": (round(cc.center_px[0], 2), round(cc.center_px[1], 2)) if cc.valid else None,
            "offset_mm": round(cc.offset_magnitude_mm, 3) if cc.offset_magnitude_mm else None,
            "offset_angle_deg": round(cc.offset_angle_deg, 2) if cc.valid else None,
            "confidence": round(cc.confidence, 4) if cc.valid else None,
            "valid": cc.valid,
        },
        "ring_status": getattr(result, 'ring_status', 'unknown'),
        "calibration": {
            "calibrated": cal.calibrated,
            "mm_per_px": round(cal.mm_per_px, 6) if cal.calibrated else None,
            "scale_px_per_mm": round(1.0 / cal.mm_per_px, 2) if cal.calibrated and cal.mm_per_px > 0 else None,
        },
        "overall_confidence": round(result.overall_confidence, 4),
        "overall_quality": getattr(result.overall_quality, 'name', str(result.overall_quality)) if result.overall_quality else None,
        "processing_time_ms": round(result.metadata.processing_time_ms, 1) if result.metadata.processing_time_ms else None,
        "alerts": result.alerts,
    }

    return audit


def audit_confidence_decomposition(det, img_path):
    """Decompose confidence calculation step by step."""
    name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Get masks
    ml_result = det.ml_engine.detect(img, frame_number=-1, source=name)
    raw_mask = ml_result._raw_mask if hasattr(ml_result, '_raw_mask') and ml_result._raw_mask is not None else np.zeros((h, w), dtype=np.uint8)

    # Get fits
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pupil_fit, limbus_fit = det._extract_structure(raw_mask, gray, ring_result=det._detect_ring(img))

    decomp = {}

    # Pupil confidence decomposition
    if pupil_fit and pupil_fit.valid:
        ml_conf = 0.5  # initial from wrapper
        fit_conf = det._fit_result_confidence(pupil_fit) if hasattr(det, '_fit_result_confidence') else 0.0
        combined = (ml_conf + fit_conf) / 2.0
        decomp["pupil"] = {
            "ml_confidence": round(ml_conf, 4),
            "fit_confidence": round(fit_conf, 4),
            "combined": round(combined, 4),
            "fit_type": pupil_fit.fit_type.value,
            "fit_quality": round(pupil_fit.fit_quality, 4),
            "circularity": round(pupil_fit.circularity, 4),
            "residual_ratio": round(pupil_fit.fit_rms_residual, 4),
        }
    else:
        decomp["pupil"] = {"detected": False, "reason": "no valid fit"}

    # Limbus confidence decomposition
    if limbus_fit and limbus_fit.valid:
        ml_conf = 0.5
        fit_conf = det._fit_result_confidence(limbus_fit) if hasattr(det, '_fit_result_confidence') else 0.0
        combined = (ml_conf + fit_conf) / 2.0
        decomp["limbus"] = {
            "ml_confidence": round(ml_conf, 4),
            "fit_confidence": round(fit_conf, 4),
            "combined": round(combined, 4),
            "fit_type": limbus_fit.fit_type.value,
            "fit_quality": round(limbus_fit.fit_quality, 4),
            "circularity": round(limbus_fit.circularity, 4),
            "residual_ratio": round(limbus_fit.fit_rms_residual, 4),
        }
    else:
        decomp["limbus"] = {"detected": False, "reason": "no valid fit"}

    return decomp


def audit_performance(det, img_path):
    """Detailed timing breakdown."""
    name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    timings = {}

    # Ring detection
    t = time.time()
    ring_result = det._detect_ring(img)
    timings["ring_ms"] = round((time.time() - t) * 1000, 1)

    # Preprocessing
    t = time.time()
    prep = det._ring_preprocessor.preprocess(img, ring_result)
    timings["preprocess_ms"] = round((time.time() - t) * 1000, 1)

    # ML inference
    t = time.time()
    ml_result = det.ml_engine.detect(img, frame_number=-1, source=name)
    timings["ml_inference_ms"] = round((time.time() - t) * 1000, 1)

    raw_mask = ml_result._raw_mask if hasattr(ml_result, '_raw_mask') and ml_result._raw_mask is not None else np.zeros((h, w), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Fitting
    t = time.time()
    pupil_fit, limbus_fit = det._extract_structure(raw_mask, gray, ring_result=ring_result)
    timings["fitting_ms"] = round((time.time() - t) * 1000, 1)

    # Total detect (includes everything)
    t = time.time()
    result = det.detect(img, frame_number=-1, source=name)
    timings["total_ms"] = round((time.time() - t) * 1000, 1)

    # Derived
    timings["post_fit_ms"] = round(timings["total_ms"] - timings["ml_inference_ms"] - timings["fitting_ms"] - timings["ring_ms"] - timings["preprocess_ms"], 1)

    return timings


# ─── Run Audit ─────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BACKEND PHASE 3 — DETECTION QUALITY AUDIT")
    print("=" * 70)

    # Initialize detector
    print("\n[1] Initializing UnifiedDetector...")
    det = UnifiedDetector()
    fitter = det._fitter
    print(f"    ML engine: {type(det.ml_engine).__name__}")
    print(f"    Fitter: {type(fitter).__name__}")

    # Find images
    images = sorted(glob.glob("clinical_data/clean/eye_*.jpeg"))
    print(f"\n[2] Found {len(images)} clinical images")

    # ── Run all 12 images ───────────────────────────────────
    print("\n[3] Running all images through pipeline...")
    all_audits = []
    for img_path in images:
        audit = audit_image(det, fitter, img_path)
        all_audits.append(audit)
        name = audit["name"]
        r = audit.get("result", {})
        p = r.get("pupil", {})
        l = r.get("limbus", {})
        cc = r.get("corneal_center", {})
        ring = audit.get("ring", {})
        print(f"  {name}: pupil={p.get('detected')} conf={p.get('confidence',0):.3f} "
              f"limbus={l.get('detected')} conf={l.get('confidence',0):.3f} "
              f"ring={ring.get('status')} "
              f"offset={cc.get('offset_mm','N/A')}mm "
              f"time={r.get('processing_time_ms','N/A')}ms")

    # ── Confidence decomposition ────────────────────────────
    print("\n[4] Confidence decomposition (4 representative images)...")
    decomps = {}
    for img_path in images[:4]:
        name = os.path.basename(img_path)
        decomps[name] = audit_confidence_decomposition(det, img_path)

    # ── Performance breakdown ───────────────────────────────
    print("\n[5] Performance breakdown (4 images)...")
    perf_images = [images[0], images[1], images[5], images[11]] if len(images) >= 12 else images[:4]
    perfs = {}
    for img_path in perf_images:
        name = os.path.basename(img_path)
        perfs[name] = audit_performance(det, img_path)
        p = perfs[name]
        print(f"  {name}: ring={p['ring_ms']}ms prep={p['preprocess_ms']}ms "
              f"ml={p['ml_inference_ms']}ms fit={p['fitting_ms']}ms "
              f"post_fit={p['post_fit_ms']}ms total={p['total_ms']}ms")

    # ── Coordinate system audit ─────────────────────────────
    print("\n[6] Coordinate system audit...")
    coord_issues = []
    for audit in all_audits:
        name = audit["name"]
        r = audit.get("result", {})
        inp = audit.get("input", {})
        w, h = inp["width"], inp["height"]
        p = r.get("pupil", {})
        l = r.get("limbus", {})
        cc = r.get("corneal_center", {})

        # Check if centers are within image bounds
        for det_name, det_data in [("pupil", p), ("limbus", l), ("corneal", cc)]:
            center = det_data.get("center")
            if center:
                cx, cy = center
                if cx < 0 or cx > w or cy < 0 or cy > h:
                    coord_issues.append(f"  {name}: {det_name} center ({cx},{cy}) outside image {w}x{h}")

    if coord_issues:
        print("  ISSUES FOUND:")
        for issue in coord_issues:
            print(f"    {issue}")
    else:
        print("  All centers within image bounds.")

    # ── Calibration audit ───────────────────────────────────
    print("\n[7] Calibration audit...")
    for audit in all_audits:
        name = audit["name"]
        r = audit.get("result", {})
        cal = r.get("calibration", {})
        l = r.get("limbus", {})
        if cal.get("calibrated") and l.get("detected"):
            lim_dia_mm = l.get("diameter_mm")
            lim_dia_px = (l.get("radius", 0) * 2) if l.get("radius") else None
            scale = cal.get("scale_px_per_mm")
            if lim_dia_mm and scale:
                expected_px = lim_dia_mm * scale
                actual_px = lim_dia_px
                if actual_px:
                    error_pct = abs(expected_px - actual_px) / max(expected_px, 1) * 100
                    print(f"  {name}: limbus_dia={lim_dia_mm:.3f}mm scale={scale:.2f}px/mm "
                          f"expected_px={expected_px:.1f} actual_px={actual_px:.1f} err={error_pct:.1f}%")

    # ── Ring interaction ────────────────────────────────────
    print("\n[8] Ring detection interaction...")
    docked = [a for a in all_audits if a.get("ring", {}).get("status") in ("ring_present", "partial")]
    pre_docked = [a for a in all_audits if a.get("ring", {}).get("status") not in ("ring_present", "partial")]
    print(f"  Docked images: {len(docked)} ({[a['name'] for a in docked]})")
    print(f"  Pre-docked images: {len(pre_docked)} ({[a['name'] for a in pre_docked]})")

    for a in docked:
        name = a["name"]
        r = a.get("result", {})
        p = r.get("pupil", {})
        l = r.get("limbus", {})
        ring = a.get("ring", {})
        print(f"  {name}: ring_r={ring.get('radius')} inner_r={ring.get('inner_radius')} "
              f"pupil_r={p.get('radius')} limbus_r={l.get('radius')}")

    # ── Validation alerts ───────────────────────────────────
    print("\n[9] Validation alerts...")
    for audit in all_audits:
        name = audit["name"]
        alerts = audit.get("result", {}).get("alerts", [])
        if alerts:
            print(f"  {name}: {len(alerts)} alerts")
            for a in alerts[:5]:
                print(f"    - {a}")
        else:
            print(f"  {name}: no alerts")

    # ── ONNX vs FastInference comparison ────────────────────
    print("\n[10] ONNX vs FastInference comparison (eye_01)...")
    try:
        from pupil_tracking.ml.fast_inference import FastInference
        fast_model = None
        model_paths = glob.glob("models/*.onnx") + glob.glob("models/*.pth")
        for mp in model_paths:
            if "segmentation" in mp.lower():
                try:
                    fast_model = FastInference(mp, device="cpu", input_size=320, use_half=False, use_compile=False)
                    break
                except Exception as e:
                    print(f"  FastInference load failed for {mp}: {e}")

        if fast_model:
            img = cv2.imread(images[0])
            # ONNX path
            onnx_result = det.ml_engine.detect(img, frame_number=-1, source="compare")
            onnx_mask = onnx_result._raw_mask if hasattr(onnx_result, '_raw_mask') else None

            # FastInference path
            t_fast = time.time()
            fast_result = fast_model.detect(img)
            t_fast = (time.time() - t_fast) * 1000

            print(f"  ONNX raw_mask values: {dict(zip(*np.unique(onnx_mask, return_counts=True))) if onnx_mask is not None else 'N/A'}")
            print(f"  FastInference time: {t_fast:.1f}ms")

            if fast_result:
                for k in ["pupil_center", "limbus_center", "pupil_radius", "limbus_radius"]:
                    if k in fast_result:
                        print(f"  FastInference {k}: {fast_result[k]}")
        else:
            print("  FastInference model not available for comparison")
    except Exception as e:
        print(f"  FastInference comparison failed: {e}")

    # ── Save full audit ─────────────────────────────────────
    output = {
        "images": all_audits,
        "confidence_decomposition": decomps,
        "performance": perfs,
    }
    with open("detection_audit.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n[11] Full audit saved to detection_audit.json")

    # ── Summary table ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION MATRIX SUMMARY")
    print("=" * 70)
    print(f"{'Image':<12} {'Pupil':<6} {'PConf':<7} {'Limbus':<7} {'LConf':<7} {'Ring':<14} {'Offset':<8} {'Quality':<10} {'Time':<8}")
    print("-" * 70)
    for a in all_audits:
        r = a.get("result", {})
        p = r.get("pupil", {})
        l = r.get("limbus", {})
        cc = r.get("corneal_center", {})
        ring = a.get("ring", {})
        print(f"{a['name']:<12} "
              f"{'Y' if p.get('detected') else 'N':<6} "
              f"{p.get('confidence',0):.3f}   "
              f"{'Y' if l.get('detected') else 'N':<7} "
              f"{l.get('confidence',0):.3f}   "
              f"{ring.get('status','?'):<14} "
              f"{str(cc.get('offset_mm','N/A'))+'mm':<8} "
              f"{r.get('overall_quality','?'):<10} "
              f"{r.get('processing_time_ms','?')}")


if __name__ == "__main__":
    main()
