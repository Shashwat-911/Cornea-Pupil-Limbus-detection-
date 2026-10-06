"""Create a static cyclotorsion review artifact.

The output contains side-by-side polar strips, a confidence/angle summary,
and JSON diagnostics. It is intended for doctor validation and annotation,
not for silently applying a treatment-axis correction.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.pentacam.cross_registration import CrossModalityRegistrationEngine
from pupil_tracking.pentacam.detector import PentacamIrisDetector
from pupil_tracking.utils.types import EyeDetectionResult, PupilDetection, LimbusDetection


def to_eye(result):
    return EyeDetectionResult(
        pupil=PupilDetection(detected=True, ellipse=result.geometry.pupil),
        limbus=LimbusDetection(detected=True, ellipse=result.geometry.limbus),
    )


def strip_image(polar):
    if polar is None or not polar.valid:
        return np.zeros((128, 720), np.uint8)
    image = np.clip(polar.image, 0, 255).astype(np.uint8)
    if polar.mask is not None:
        image[polar.mask == 0] = 0
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Seated Pentacam/IR image")
    parser.add_argument("--current", required=True, help="Post-dock/current ELITA image")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--laterality", choices=("OD", "OS"), required=True)
    parser.add_argument("--eye-id", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = cv2.imread(args.reference, cv2.IMREAD_COLOR)
    current = cv2.imread(args.current, cv2.IMREAD_COLOR)
    if reference is None or current is None:
        raise ValueError("Both images must be readable")

    pentacam = PentacamIrisDetector().detect(reference, extract_features=False, refine_boundary=False)
    if not pentacam.valid:
        raise RuntimeError(f"Reference rejected: {pentacam.failure_reason}")
    # The existing UnifiedDetector is the integration boundary for ELITA.
    from pupil_tracking.core.detector import UnifiedDetector
    elita_result = UnifiedDetector().detect(current)
    if not elita_result.has_both:
        raise RuntimeError("Current ELITA image lacks validated pupil and limbus geometry")
    engine = CrossModalityRegistrationEngine()
    result = engine.register(reference, current, pentacam_result=pentacam,
                             elita_detection=elita_result, laterality=args.laterality,
                             mode="static")
    reference_polar = engine._polar_cache
    current_polar = engine.unwrapper.unwrap_from_detection(current, elita_result)
    left, right = strip_image(reference_polar), strip_image(current_polar)
    width = max(left.shape[1], right.shape[1])
    left = cv2.resize(left, (width, 180), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (width, 180), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((470, width * 2 + 30, 3), np.uint8)
    canvas[80:260, 10:10+width] = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    canvas[80:260, 20+width:20+2*width] = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "PENTACAM / SEATED POLAR IRIS", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, .7, (220,220,220), 2)
    cv2.putText(canvas, "ELITA / CURRENT POLAR IRIS", (20+width, 55), cv2.FONT_HERSHEY_SIMPLEX, .7, (220,220,220), 2)
    angle = result.rotation_deg if result.valid else None
    summary = f"VALID={result.valid}  angle={angle:+.3f} deg  confidence={result.confidence:.3f}" if angle is not None else f"ABSTAINED  confidence={result.confidence:.3f}"
    cv2.putText(canvas, summary, (10, 315), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 220) if result.valid else (0, 0, 255), 2)
    cv2.putText(canvas, "Doctor validation required before axis adjustment.", (10, 355), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1)
    cv2.putText(canvas, "Save this JSON with the pre/post-dock images and laterality.", (10, 385), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 1)
    cv2.imwrite(str(output / "static_review.png"), canvas)
    report = {
        "eye_id": args.eye_id, "laterality": args.laterality, "mode": "static",
        "valid": bool(result.valid), "rotation_deg": angle,
        "confidence": result.confidence, "failure_reason": result.failure_reason,
        "angular_diagnostics": result.angular_diagnostics,
        "reference_polar_shape": list(reference_polar.image.shape) if reference_polar.valid else None,
        "current_polar_shape": list(current_polar.image.shape) if current_polar.valid else None,
        "doctor_validation": "PENDING",
    }
    (output / "static_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("valid", "rotation_deg", "confidence", "doctor_validation")}))


if __name__ == "__main__":
    main()
