"""Isolated warmed CPU session benchmark; reports anonymous image IDs only.

Use --implementation-root with a git-archive checkout to compare old code.
Run implementations sequentially on an otherwise idle computer.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--data-root', default='.')
    parser.add_argument('--report', required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    sys.path.insert(0, str(Path(args.implementation_root).resolve()))
    from pupil_tracking.pentacam.session import SittingRegistrationSession
    cv2.setNumThreads(1)
    root = Path(args.data_root)
    files = sorted(root.glob('*.BMP')) + sorted(root.glob('sitting *.jpeg'))
    rows, seen = [], set()
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Unreadable input')
        session = SittingRegistrationSession()
        session.set_reference(image, eye_id='benchmark', laterality='OD')
        h, w = image.shape[:2]
        frames = [cv2.warpAffine(image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1),
                                (w, h)) for angle in (-5, 0, 5)]
        for frame in frames:
            session.register(frame, eye_id='benchmark', laterality='OD')
        for _ in range(args.repeats):
            for angle, frame in zip((-5, 0, 5), frames):
                start = time.perf_counter()
                result = session.register(frame, eye_id='benchmark', laterality='OD')
                elapsed = (time.perf_counter() - start) * 1000
                rows.append(dict(image=len(seen)-1, target_deg=angle, valid=result.valid,
                                 angle_deg=result.rotation_deg if result.valid else None,
                                 latency_ms=elapsed))
    if not rows:
        raise ValueError('No benchmark images found')
    times = [r['latency_ms'] for r in rows]
    summary = dict(images=len(seen), frames=len(rows), accepted=sum(r['valid'] for r in rows),
                   median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                   max_error_deg=max((abs(r['angle_deg']-r['target_deg']) for r in rows if r['valid']), default=None))
    summary['processing_fps'] = 1000 / summary['median_ms']
    report = dict(scope='Warmed session, automatic geometry, BGR input, max_size=640, 720 angular samples; excludes capture, decoding and display',
                  environment=dict(python=platform.python_version(), opencv=cv2.__version__,
                                   numpy=np.__version__, processor=platform.processor(), threads=1),
                  summary=summary, trials=rows)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
