"""Render measured boundaries and explicitly unverified anatomy hypotheses.

Example: python -m scripts.review_sitting --input "sitting 1.jpeg" --output-dir output/review1
Outputs remain local; no patient images are uploaded.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.pentacam.detector import PentacamIrisDetector


def label(canvas, message, x, y, color=(230, 230, 230), scale=.58):
    cv2.putText(canvas, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(image, result):
    pupil = result.geometry.pupil
    limbus = result.geometry.refined_limbus or result.geometry.limbus
    overlays = [image.copy(), image.copy()]
    green, cyan, amber = (80, 235, 100), (240, 215, 80), (30, 180, 255)
    points = np.asarray(result.geometry.limbus_support_points)
    supported = np.zeros(180, bool)
    if len(points):
        t = np.deg2rad(limbus.angle_deg)
        dx, dy = points[:, 0]-limbus.center_x, points[:, 1]-limbus.center_y
        theta = np.degrees(np.arctan2((-dx*np.sin(t)+dy*np.cos(t))/limbus.semi_minor,
                                     (dx*np.cos(t)+dy*np.sin(t))/limbus.semi_major)) % 360
        for a in theta:
            for offset in range(-2, 3):
                supported[(int(a/2)+offset) % 180] = True
    for overlay in overlays:
        cv2.ellipse(overlay, ((pupil.center_x, pupil.center_y),
                             (2*pupil.semi_major, 2*pupil.semi_minor), pupil.angle_deg), green, 2, cv2.LINE_AA)
        for i in range(180):
            if supported[i] or i % 4 == 0:
                cv2.ellipse(overlay, (round(limbus.center_x), round(limbus.center_y)),
                    (round(limbus.semi_major), round(limbus.semi_minor)), limbus.angle_deg,
                    i*2, i*2+2, cyan if supported[i] else (160, 160, 160), 2 if supported[i] else 1, cv2.LINE_AA)
    selected = []
    for f in sorted(result.feature_set.features, key=lambda f: f.response, reverse=True):
        if all((f.x-g.x)**2+(f.y-g.y)**2 > (limbus.radius*.08)**2 for g in selected):
            selected.append(f)
        if len(selected) >= 45:
            break
    for f in selected:
        cv2.circle(overlays[0], (round(f.x), round(f.y)), 3, amber, 1, cv2.LINE_AA)
    counts = {'crypt_like_candidate': 0, 'furrow_like_candidate': 0}
    for f in result.anatomy_candidates:
        counts[f.kind] += 1
        if counts[f.kind] > 10:
            continue
        color = (90, 200, 255) if f.kind == 'crypt_like_candidate' else (240, 130, 235)
        prefix = 'C?' if f.kind == 'crypt_like_candidate' else 'F?'
        cv2.circle(overlays[1], (round(f.x), round(f.y)), 6, color, 1, cv2.LINE_AA)
        label(overlays[1], prefix, round(f.x)+7, round(f.y)-3, color, .40)
    r = limbus.semi_major
    h, w = image.shape[:2]
    x0, x1 = max(0, round(limbus.center_x-r*1.28)), min(w, round(limbus.center_x+r*1.28))
    y0, y1 = max(0, round(limbus.center_y-r*1.10)), min(h, round(limbus.center_y+r*1.15))
    pw, gap = 520, 18
    ph = round((y1-y0)*pw/(x1-x0))
    canvas = np.full((ph+220, pw*3+gap*4, 3), 24, np.uint8)
    label(canvas, 'LIMBUS + IRIS REVIEW | actual image processing', 18, 31, scale=.8)
    for i, (source, title) in enumerate(zip([image]+overlays,
            ['Original', 'Boundary support + texture keypoints', 'Anatomical hypotheses: expert review'])):
        x = gap+i*(pw+gap)
        label(canvas, title, x, 67, scale=.55)
        canvas[82:82+ph, x:x+pw] = cv2.resize(source[y0:y1, x0:x1], (pw, ph), interpolation=cv2.INTER_AREA)
    y = 82+ph+28
    label(canvas, 'GREEN pupil | CYAN supported limbus fit | GRAY DOTS inferred / unobserved boundary', 18, y)
    label(canvas, 'AMBER points: local texture keypoints, not confirmed anatomical landmarks.', 18, y+27, amber)
    label(canvas, 'C? dark-depression / crypt-like candidate | F? peripheral tangential ridge / furrow-like candidate', 18, y+54)
    label(canvas, 'Anatomical labels are unverified. No cyclotorsion angle from a single image. Hidden limbus is not measured.', 18, y+81, (170,170,170))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    image = cv2.imread(args.input)
    if image is None:
        raise ValueError('Cannot read input image')
    result = PentacamIrisDetector().detect(image, review_anatomy=True)
    if not result.valid:
        raise RuntimeError(result.failure_reason)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output/'review.png'), render(image, result)):
        raise RuntimeError('Cannot save review image')
    (output/'detections.json').write_text(json.dumps(result.to_dict(), indent=2), encoding='utf-8')
    print(json.dumps(dict(valid=result.valid, boundary_method=result.geometry.limbus_method,
        boundary_support_points=len(result.geometry.limbus_support_points),
        texture_keypoints=len(result.feature_set.features), anatomy_hypotheses=len(result.anatomy_candidates),
        processing_ms=result.processing_time_ms)))


if __name__ == '__main__':
    main()
