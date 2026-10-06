"""Register two confirmed same-eye seated images using CPU only."""
import argparse
import json
from pathlib import Path

import cv2

from pupil_tracking.pentacam.session import SittingRegistrationSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("current")
    parser.add_argument("--eye-id", required=True, help="Pseudonymous confirmed same-eye identifier")
    parser.add_argument("--laterality", choices=("OD", "OS"), required=True)
    parser.add_argument("--output", default="output/sitting/registration.json")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    session = SittingRegistrationSession()
    session.set_reference(cv2.imread(args.reference), eye_id=args.eye_id, laterality=args.laterality)
    result = session.register(cv2.imread(args.current), eye_id=args.eye_id, laterality=args.laterality)
    output = result.to_dict()
    output["rotation_deg"] = result.rotation_deg if result.valid else None
    output["scope"] = "Engineering estimate; positive angle is counter-clockwise in unmirrored image coordinates"
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
