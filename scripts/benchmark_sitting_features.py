"""Known-warp keypoint repeatability; does not validate anatomical labels."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def strongest(result, limit=60):
    selected = []
    for f in sorted(result.feature_set.features, key=lambda x: x.response, reverse=True):
        if all((f.x-x)**2+(f.y-y)**2 >= 8**2 for x, y in selected):
            selected.append((f.x, f.y))
        if len(selected) == limit:
            break
    return np.asarray(selected, dtype=float).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.implementation_root).resolve()))
    from pupil_tracking.pentacam.detector import PentacamIrisDetector
    cv2.setNumThreads(1)
    rows, seen = [], set()
    root = Path(args.data_root)
    for path in sorted(root.glob('*.BMP')) + sorted(root.glob('sitting *.jpeg')):
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        source = cv2.imread(str(path), 0)
        source = cv2.resize(source, (640, round(source.shape[0]*640/source.shape[1])), interpolation=cv2.INTER_AREA)
        detector = PentacamIrisDetector()
        base = detector.detect(source)
        if not base.valid:
            continue
        p = base.geometry.pupil
        a = strongest(base)
        for angle in (-6.75, 3.25, 8.5):
            matrix = cv2.getRotationMatrix2D((p.center_x, p.center_y), angle, 1)
            frame = cv2.warpAffine(source, matrix, (source.shape[1], source.shape[0]))
            frame = np.clip(frame.astype(float)*.85+12, 0, 255).astype(np.uint8)
            # Isolate feature repeatability from automatic geometry changes;
            # automatic geometry/registration has its own benchmark.
            geometry = copy.deepcopy(base.geometry)
            ellipses = [geometry.pupil, geometry.limbus]
            if getattr(geometry, 'refined_limbus', None) is not None:
                ellipses.append(geometry.refined_limbus)
            for ellipse in ellipses:
                ellipse.center_x, ellipse.center_y = map(float, matrix @ [ellipse.center_x, ellipse.center_y, 1])
                ellipse.angle_deg = (ellipse.angle_deg-angle) % 180
            result = detector.detect(frame, geometry=geometry)
            b = strongest(result) if result.valid else np.empty((0, 2))
            expected = np.column_stack((a, np.ones(len(a)))) @ matrix.T
            pairs = np.linalg.norm(expected[:, None, :]-b[None, :, :], axis=-1)
            count = 0
            # Greedy one-to-one spatial correspondence within 3 pixels at 640.
            while pairs.size and np.min(pairs) <= 3:
                i, j = np.unravel_index(np.argmin(pairs), pairs.shape)
                count += 1
                pairs[i, :] = np.inf
                pairs[:, j] = np.inf
            rows.append(dict(image=len(seen)-1, angle=angle, reference_count=len(a),
                             current_count=len(b), repeated=count, fraction=count/max(1,len(a))))
    report = dict(scope='Known transformed geometry; top 60 spatially separated points, 3px one-to-one tolerance at 640px image width; synthetic warp plus brightness change, no anatomical labels',
                  images=len(seen), trials=rows,
                  mean_repeatability=float(np.mean([r['fraction'] for r in rows])) if rows else None)
    dest = Path(args.report)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'trials'}))


if __name__ == '__main__':
    main()
