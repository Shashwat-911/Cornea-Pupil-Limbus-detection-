import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import json
import time
import math
import numpy as np
from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.ml.onnx_inference import ONNXInference

# Load Ground Truth
with open('clinical_data/clean/annotations/annotations.json') as f:
    gt_data = json.load(f)

det = UnifiedDetector()
# Use FP32 ONNX model (DirectML accelerated)
det.ml_engine._engine = ONNXInference(use_quantized=False)

# Monkeypatch the fixes on det to test
orig_extract = det._extract_structure
def fast_extract(mask, gray_image=None, ring_result=None):
    is_docked = ring_result is not None and ring_result.status in ('ring_present', 'partial') or (hasattr(ring_result, 'status') and ring_result.status.name in ('PRESENT', 'PARTIAL'))
    pupil_mask = (mask == 1).astype(np.uint8)
    if is_docked and ring_result.ring_center is not None:
        pupil_mask = det._apply_ring_roi(pupil_mask, ring_result, margin_frac=0.85)
    pupil_fit = det._fitter.fit(pupil_mask, gray_image)

    iris_mask = ((mask == 2) | (mask == 1)).astype(np.uint8)
    if is_docked and ring_result.ring_center is not None:
        iris_mask = det._apply_ring_roi(iris_mask, ring_result, margin_frac=0.95)
    # No artificial 15x15 erosion!
    limbus_fit = det._fitter.fit(iris_mask, gray_image, pupil_hint=pupil_fit)
    return pupil_fit, limbus_fit
det._extract_structure = fast_extract

# Fix _apply_fit_to_result
orig_apply = det._apply_fit_to_result
def fixed_apply(result, pupil_fit, limbus_fit, force_limbus_overwrite=False, is_docked=False):
    if pupil_fit is not None and pupil_fit.valid:
        ep = det._fit_result_to_ellipse_params(pupil_fit)
        new_conf = det._fit_result_confidence(pupil_fit)
        if (not result.pupil.detected) or result.pupil.ellipse is None or new_conf >= result.pupil.confidence:
            result.pupil.detected = True
            result.pupil.ellipse = ep
            result.pupil.confidence = new_conf
            result.pupil.quality = det.assign_quality_grade(new_conf) if hasattr(det, 'assign_quality_grade') else result.pupil.quality
            result.pupil.fit_type = pupil_fit.fit_type.value
            result.pupil.contour_points = pupil_fit.contour_points

    if limbus_fit is not None and limbus_fit.valid:
        ep = det._fit_result_to_ellipse_params(limbus_fit)
        new_conf = det._fit_result_confidence(limbus_fit)
        if (not result.limbus.detected) or result.limbus.ellipse is None or force_limbus_overwrite or new_conf >= result.limbus.confidence:
            if not is_docked and limbus_fit is not None:
                from pupil_tracking.calibration.spatial_calibration import correct_pre_docked_limbus_ellipse
                ep = correct_pre_docked_limbus_ellipse(ep, lower_quadrant_pct=0.015)
            result.limbus.detected = True
            result.limbus.ellipse = ep
            result.limbus.confidence = new_conf
            result.limbus.quality = det.assign_quality_grade(new_conf) if hasattr(det, 'assign_quality_grade') else result.limbus.quality
            result.limbus.fit_type = limbus_fit.fit_type.value
            result.limbus.contour_points = limbus_fit.contour_points
det._apply_fit_to_result = fixed_apply

# Optimize smart_fitter subpixel refinement
fitter = det._fitter
orig_fit = fitter.fit
def fast_fit(binary_mask, gray_image=None, pupil_hint=None):
    mask = binary_mask.copy()
    if mask.max() == 1:
        mask = mask * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return det._fitter.__class__().fit_contour(np.empty((0, 2)))
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < fitter.min_area:
        return det._fitter.__class__().fit_contour(np.empty((0, 2)))
    pts = largest.reshape(-1, 2).astype(np.float64)
    if len(pts) < fitter.min_contour_points:
        return det._fitter.__class__().fit_contour(np.empty((0, 2)))

    if pupil_hint is not None and pupil_hint.valid:
        dx = pts[:, 0] - pupil_hint.center_x
        dy = pts[:, 1] - pupil_hint.center_y
        distances = np.hypot(dx, dy)
        median_dist = np.median(distances)
        upper_bound = median_dist * 1.15
        lower_bound = median_dist * 0.85
        mask_pts = (distances >= lower_bound) & (distances <= upper_bound)
        if np.sum(mask_pts) >= max(fitter.min_contour_points, int(len(pts) * 0.25)):
            pts = pts[mask_pts]

    # Subsample points along contour for 10x faster subpixel refinement
    if len(pts) > 180:
        step = len(pts) // 180
        pts_fit = pts[::step]
    else:
        pts_fit = pts

    if fitter.subpixel_refine and gray_image is not None:
        h, w = gray_image.shape[:2]
        # Fast float32 scharr
        gx = cv2.Scharr(gray_image, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray_image, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        s_rad = 3
        s_step = 0.25
        n_steps = int(s_rad / s_step)
        t_vals = (np.arange(-n_steps, n_steps + 1) * s_step).astype(np.float32)
        K = len(t_vals)

        ix = np.clip(np.round(pts_fit[:, 0]).astype(int), 0, w - 1)
        iy = np.clip(np.round(pts_fit[:, 1]).astype(int), 0, h - 1)
        pt_gx = gx[iy, ix]
        pt_gy = gy[iy, ix]
        g_len = np.hypot(pt_gx, pt_gy)
        g_len[g_len < 1e-6] = 1.0
        nx = pt_gx / g_len
        ny = pt_gy / g_len
        map_x = pts_fit[:, 0:1] + nx[:, None] * t_vals[None, :]
        map_y = pts_fit[:, 1:2] + ny[:, None] * t_vals[None, :]
        samples = cv2.remap(mag, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
        peak_idx = np.argmax(samples, axis=1)
        valid_peak = (peak_idx >= 1) & (peak_idx < K - 1)
        row = np.where(valid_peak)[0]
        idx = peak_idx[valid_peak]
        y_m1 = samples[row, idx - 1]
        y_0 = samples[row, idx]
        y_p1 = samples[row, idx + 1]
        denom = 2.0 * (2.0 * y_0 - y_m1 - y_p1)
        denom_ok = np.abs(denom) > 1e-12
        delta = np.zeros(len(row), dtype=np.float32)
        delta[denom_ok] = (y_m1[denom_ok] - y_p1[denom_ok]) / denom[denom_ok]
        best_t = t_vals[peak_idx]
        best_t[valid_peak] += delta * s_step
        refined = pts_fit.copy()
        refined[:, 0] += nx * best_t
        refined[:, 1] += ny * best_t
        pts_fit = refined
    return fitter.fit_contour(pts_fit)

fitter.fit = fast_fit

# Warmup
img0 = cv2.imread('clinical_data/clean/eye_01.jpeg')
det.detect(img0)

header = f"{'Image':<12} {'P_Err(px)':>10} {'L_Err(px)':>10} {'P_Rad':>8} {'L_Rad':>8} {'Quality':>12} {'Time(ms)':>10}"
print(header)
print('-' * len(header))

total_p_err = []
total_l_err = []
times = []

for name in sorted(gt_data.keys()):
    img_path = f'clinical_data/clean/{name}'
    img = cv2.imread(img_path)
    if img is None:
        continue
    t0 = time.perf_counter()
    res = det.detect(img)
    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)

    gt_p = gt_data[name]['annotations']['PUPIL']
    gt_l = gt_data[name]['annotations']['LIMBUS']
    gt_pr = (gt_p['semi_major'] + gt_p['semi_minor']) / 2.0
    gt_lr = (gt_l['semi_major'] + gt_l['semi_minor']) / 2.0

    p_r = res.pupil.ellipse.radius if res.pupil.ellipse else 0
    l_r = res.limbus.ellipse.radius if res.limbus.ellipse else 0

    p_err = abs(p_r - gt_pr)
    l_err = abs(l_r - gt_lr)
    total_p_err.append(p_err)
    total_l_err.append(l_err)

    q = res.overall_quality.name
    print(f"{name:<12} {p_err:>10.2f} {l_err:>10.2f} {p_r:>8.1f} {l_r:>8.1f} {q:>12} {dt:>10.1f}")

print('-' * len(header))
print(f"Mean Pupil Radius Error: {np.mean(total_p_err):.2f} px")
print(f"Mean Limbus Radius Error: {np.mean(total_l_err):.2f} px")
print(f"Mean Latency: {np.mean(times):.1f} ms  ({1000/np.mean(times):.1f} FPS)")
