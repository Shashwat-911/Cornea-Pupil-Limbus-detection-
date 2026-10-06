"""CPU seated-eye video/camera runner with explicit rejection and JSONL output."""
import argparse
import json
import time
from pathlib import Path

import cv2

from pupil_tracking.pentacam import SittingRegistrationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source", required=True, help="Video filename or camera index, e.g. 0")
    parser.add_argument("--eye-id", required=True)
    parser.add_argument("--laterality", required=True, choices=("OD", "OS"))
    parser.add_argument("--output", default="output/sitting/video.jsonl")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes until end or Escape")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    session = SittingRegistrationSession()
    session.set_reference(cv2.imread(args.reference), eye_id=args.eye_id, laterality=args.laterality)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise ValueError("Could not open capture source")
    if isinstance(source, int):
        # Best effort; actual capture-to-display delay is backend dependent.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    path = Path(args.output)
    frame_index = 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output:
            while not args.max_frames or frame_index < args.max_frames:
                start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                result = session.register(frame, eye_id=args.eye_id, laterality=args.laterality, dynamic=True)
                record = result.to_dict()
                # Do not present a rejected candidate as a measured angle.
                record["rotation_deg"] = result.rotation_deg if result.valid else None
                record["frame_index"] = frame_index
                record["capture_and_processing_ms"] = (time.perf_counter() - start) * 1000
                output.write(json.dumps(record) + "\n")
                frame_index += 1
                if not args.headless:
                    label = f"Rotation {result.rotation_deg:+.2f} deg" if result.valid else "NO RELIABLE MATCH"
                    color = (0, 220, 0) if result.valid else (0, 80, 255)
                    display = frame.copy()
                    cv2.putText(display, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
                    cv2.imshow("Seated iris registration - engineering preview", display)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames; results saved to {path}")
    return 0 if frame_index else 2


if __name__ == "__main__":
    raise SystemExit(main())
