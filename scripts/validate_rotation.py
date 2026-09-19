"""Validation experiment: Testing iris detection, centration, and cyclotorsion under +/- 3 degree rotation.

Steps:
1. Load a clinical ELITA image with clear iris visibility (e.g., eye_01.jpeg).
2. Create copies and apply exact +3.0° and -3.0° rotations around image/limbus center.
3. Run UnifiedDetector on both original and rotated images.
4. Run RegistrationEngine to compute cyclotorsion angle between original and rotated.
5. Compare pupil, limbus, corneal center, offset, and measured cyclotorsion against ground truth.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.registration.engine import RegistrationEngine


def rotate_image(image: np.ndarray, angle_deg: float, center: tuple[float, float] | None = None) -> np.ndarray:
    """Rotate image by angle_deg around center using cubic interpolation."""
    h, w = image.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated


def main():
    img_path = Path("clinical_data/clean/eye_01.jpeg")
    if not img_path.exists():
        img_path = Path("clinical_data/clean/eye_02.jpeg")

    print(f"Loading reference ELITA image: {img_path}")
    img_ref = cv2.imread(str(img_path))
    h, w = img_ref.shape[:2]

    detector = UnifiedDetector()
    reg_engine = RegistrationEngine()

    # Step 1: Detect on reference (non-rotated) image
    det_ref = detector.detect(img_ref)
    assert det_ref.has_both, "Reference detection must have both pupil and limbus"

    ref_pupil = det_ref.pupil.ellipse
    ref_limbus = det_ref.limbus.ellipse
    ref_cc = det_ref.corneal_center

    print("\n--- Non-Rotated Reference Image (0.0°) ---")
    print(f"Pupil Center: ({ref_pupil.center_x:.2f}, {ref_pupil.center_y:.2f}) px, Radius: {ref_pupil.radius:.2f} px")
    print(f"Limbus Center: ({ref_limbus.center_x:.2f}, {ref_limbus.center_y:.2f}) px, Radius: {ref_limbus.radius:.2f} px")
    print(f"Corneal Center: ({ref_cc.center_px[0]:.2f}, {ref_cc.center_px[1]:.2f}) px")
    print(f"Offset (Pupil - Limbus): dx={ref_cc.offset_px[0]:.2f} px, dy={ref_cc.offset_px[1]:.2f} px, magnitude={ref_cc.offset_magnitude_px:.2f} px, angle={ref_cc.offset_angle_deg:.2f}°")

    rot_center = (ref_limbus.center_x, ref_limbus.center_y)
    angles_to_test = [+3.0, -3.0]
    results = {}

    for angle in angles_to_test:
        print(f"\n==========================================")
        print(f"Testing Rotation: {angle:+.1f}°")
        print(f"==========================================")

        img_rot = rotate_image(img_ref, angle, center=rot_center)
        det_rot = detector.detect(img_rot)

        rot_pupil = det_rot.pupil.ellipse
        rot_limbus = det_rot.limbus.ellipse
        rot_cc = det_rot.corneal_center

        print(f"Rotated Pupil Center: ({rot_pupil.center_x:.2f}, {rot_pupil.center_y:.2f}) px, Radius: {rot_pupil.radius:.2f} px")
        print(f"Rotated Limbus Center: ({rot_limbus.center_x:.2f}, {rot_limbus.center_y:.2f}) px, Radius: {rot_limbus.radius:.2f} px")
        print(f"Rotated Corneal Center: ({rot_cc.center_px[0]:.2f}, {rot_cc.center_px[1]:.2f}) px")
        print(f"Rotated Offset: dx={rot_cc.offset_px[0]:.2f} px, dy={rot_cc.offset_px[1]:.2f} px, magnitude={rot_cc.offset_magnitude_px:.2f} px, angle={rot_cc.offset_angle_deg:.2f}°")

        # Check angular change in offset
        expected_offset_angle = ref_cc.offset_angle_deg + angle
        # Normalize to [-180, 180]
        expected_offset_angle = (expected_offset_angle + 180) % 360 - 180
        actual_offset_angle = (rot_cc.offset_angle_deg + 180) % 360 - 180
        offset_angle_diff = abs(actual_offset_angle - expected_offset_angle)
        if offset_angle_diff > 180:
            offset_angle_diff = 360 - offset_angle_diff

        print(f"Expected Offset Vector Angle: {expected_offset_angle:.2f}°, Actual: {actual_offset_angle:.2f}° (Difference: {offset_angle_diff:.2f}°)")
        print(f"Offset Magnitude Diff: {abs(rot_cc.offset_magnitude_px - ref_cc.offset_magnitude_px):.2f} px")

        # Step 4: Run Registration Engine for Cyclotorsion Estimation
        reg_result = reg_engine.register(img_ref, img_rot, det_ref, det_rot)

        print("\n--- Cyclotorsion Registration Engine Output ---")
        print(f"Valid: {reg_result.valid}")
        print(f"Fused Torsion Angle: {reg_result.torsion_deg:.3f}° (Target: {angle:+.1f}°)")
        error_deg = abs(reg_result.torsion_deg - angle)
        print(f"Cyclotorsion Error: {error_deg:.3f}°")
        print(f"Confidence: {reg_result.confidence:.3f}")
        print(f"Quality Grade: {reg_result.quality.value}")
        print(f"Active Streams: {reg_result.active_streams}, Agreeing Streams: {reg_result.agreeing_streams}")

        for s_name, s_res in reg_result.stream_results.items():
            s_ang = f"{s_res.torsion_deg:.3f}°" if s_res.torsion_deg is not None else "N/A"
            print(f"  - Stream [{s_name}]: torsion={s_ang}, conf={s_res.confidence:.3f}, inliers={s_res.inlier_count}, time={s_res.processing_time_ms:.1f}ms")

        results[str(angle)] = {
            "target_angle_deg": angle,
            "measured_torsion_deg": reg_result.torsion_deg,
            "error_deg": error_deg,
            "confidence": reg_result.confidence,
            "quality": reg_result.quality.value,
            "offset_magnitude_ref_px": ref_cc.offset_magnitude_px,
            "offset_magnitude_rot_px": rot_cc.offset_magnitude_px,
            "offset_angle_ref_deg": ref_cc.offset_angle_deg,
            "offset_angle_rot_deg": rot_cc.offset_angle_deg,
            "offset_angle_error_deg": offset_angle_diff,
        }

    out_json = Path("scripts/rotation_validation_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results to {out_json}")


if __name__ == "__main__":
    main()
