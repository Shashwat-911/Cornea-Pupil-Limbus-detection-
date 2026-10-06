import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import json
import time
import numpy as np
from pupil_tracking.core.detector import UnifiedDetector

# Load Ground Truth
with open('clinical_data/clean/annotations/annotations.json') as f:
    gt_data = json.load(f)

det = UnifiedDetector()

# Warmup
img0 = cv2.imread('clinical_data/clean/eye_01.jpeg')
det.detect(img0)

header = "{:<12} {:>10} {:>10} {:>8} {:>8} {:>12} {:>10}".format(
    "Image", "P_Err(px)", "L_Err(px)", "P_Rad", "L_Rad", "Quality", "Time(ms)"
)
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
    print("{:<12} {:>10.2f} {:>10.2f} {:>8.1f} {:>8.1f} {:>12} {:>10.1f}".format(
        name, p_err, l_err, p_r, l_r, q, dt
    ))

print('-' * len(header))
print(f"Mean Pupil Radius Error: {np.mean(total_p_err):.2f} px")
print(f"Mean Limbus Radius Error: {np.mean(total_l_err):.2f} px")
print(f"Mean Latency: {np.mean(times):.1f} ms  ({1000/np.mean(times):.1f} FPS)")
