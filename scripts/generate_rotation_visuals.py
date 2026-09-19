"""Generate visual comparison figures for the +/-3 degree rotation experiment."""

import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.visualization import draw_registration_overlay
from scripts.validate_rotation import rotate_image


def draw_detection(image: np.ndarray, det) -> np.ndarray:
    vis = image.copy()
    if det.has_pupil:
        pe = det.pupil.ellipse
        cv2.ellipse(
            vis,
            (int(pe.center_x), int(pe.center_y)),
            (int(pe.semi_major), int(pe.semi_minor)),
            pe.angle_deg,
            0, 360,
            (0, 255, 255), 2,
        )
        cv2.circle(vis, (int(pe.center_x), int(pe.center_y)), 3, (0, 255, 255), -1)

    if det.has_limbus:
        le = det.limbus.ellipse
        cv2.ellipse(
            vis,
            (int(le.center_x), int(le.center_y)),
            (int(le.semi_major), int(le.semi_minor)),
            le.angle_deg,
            0, 360,
            (0, 255, 0), 2,
        )
        cv2.circle(vis, (int(le.center_x), int(le.center_y)), 3, (0, 255, 0), -1)

    if det.has_both:
        # Draw offset vector
        cv2.line(
            vis,
            (int(le.center_x), int(le.center_y)),
            (int(pe.center_x), int(pe.center_y)),
            (0, 0, 255), 2,
        )
    return vis


def main():
    img_path = Path("clinical_data/clean/eye_01.jpeg")
    img_ref = cv2.imread(str(img_path))
    detector = UnifiedDetector()
    reg_engine = RegistrationEngine()

    det_ref = detector.detect(img_ref)
    ref_limbus = det_ref.limbus.ellipse
    rot_center = (ref_limbus.center_x, ref_limbus.center_y)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Non-rotated reference
    overlay_ref = draw_detection(img_ref, det_ref)
    axes[0].imshow(cv2.cvtColor(overlay_ref, cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Reference ELITA Image (0.0°)\nPupil Center: ({det_ref.pupil.ellipse.center_x:.1f}, {det_ref.pupil.ellipse.center_y:.1f})\nOffset: {det_ref.corneal_center.offset_magnitude_px:.1f}px @ {det_ref.corneal_center.offset_angle_deg:.1f}°",
        fontsize=11, fontweight="bold",
    )
    axes[0].axis("off")

    angles = [+3.0, -3.0]
    for i, angle in enumerate(angles):
        img_rot = rotate_image(img_ref, angle, center=rot_center)
        det_rot = detector.detect(img_rot)
        reg_res = reg_engine.register(img_ref, img_rot, det_ref, det_rot)

        overlay_rot = draw_registration_overlay(
            img_rot, reg_res, detection_ref=det_ref, detection_curr=det_rot
        )
        overlay_rot = draw_detection(overlay_rot, det_rot)

        axes[i + 1].imshow(cv2.cvtColor(overlay_rot, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(
            f"Rotated {angle:+.1f}° (Ground Truth)\nMeasured Torsion: {reg_res.torsion_deg:+.2f}° (Conf: {reg_res.confidence:.2f})\nOffset: {det_rot.corneal_center.offset_magnitude_px:.1f}px @ {det_rot.corneal_center.offset_angle_deg:.1f}°",
            fontsize=11, fontweight="bold",
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    out_path = Path("scripts/rotation_validation_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved visual comparison to {out_path}")


if __name__ == "__main__":
    main()
