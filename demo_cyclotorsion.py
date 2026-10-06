#!/usr/bin/env python3
"""
Turnkey Cyclotorsion Live Demo for Mentors / Clinical Demonstrations.

Supports:
1. Video / Camera:
   Automatically uses Frame 0 as the 0.0° Baseline Reference,
   then tracks and displays dynamic cyclotorsion in real time.
   Usage:
       python demo_cyclotorsion.py --video path/to/video.mp4
       python demo_cyclotorsion.py --camera

2. Single Image:
   Runs an interactive / dynamic demonstration of cyclotorsion
   by testing rotation against ground truth angles (+3.0°, -3.0°).
   Usage:
       python demo_cyclotorsion.py --image clinical_data/clean/eye_01.jpeg
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.registration.engine import RegistrationEngine
from pupil_tracking.registration.visualization import draw_registration_overlay
from scripts.validate_rotation import rotate_image


def _run_seated_image_demo(image: np.ndarray, image_path: Path, test_angle: float, headless: bool):
    """Run the production seated detector/session on a same-eye rotation demo."""
    from pupil_tracking.pentacam.detector import PentacamIrisDetector
    from pupil_tracking.pentacam.session import SittingRegistrationSession

    detector = PentacamIrisDetector()
    reference_detection = detector.detect(image, review_anatomy=True)
    if not reference_detection.valid:
        return False
    pupil = reference_detection.geometry.pupil
    center = (pupil.center_x, pupil.center_y)
    rotated = rotate_image(image, test_angle, center=center)
    session = SittingRegistrationSession()
    session.set_reference(image, eye_id="demo", laterality="OD")
    registration = session.register(rotated, eye_id="demo", laterality="OD")
    if not registration.valid:
        print(f"[DEMO] Seated registration abstained: {registration.failure_reason}")
        return True
    print("\n[DEMO] Seated Pentacam path (production detector/session)")
    print(f"  Input: {image_path.name}")
    print(f"  Applied rotation: {test_angle:+.2f} degrees")
    print(f"  Measured rotation: {registration.rotation_deg:+.2f} degrees")
    print(f"  Absolute error: {abs(registration.rotation_deg-test_angle):.3f} degrees")
    print(f"  Confidence: {registration.confidence:.1%}")
    print(f"  Processing time: {registration.processing_time_ms:.1f} ms")
    print("  Feature points are repeatable texture keypoints; anatomy labels require expert review.")
    preview = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB) if rotated.ndim == 3 else cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
    geom = reference_detection.geometry
    limbus = geom.refined_limbus or geom.limbus
    cv2.ellipse(preview, (round(limbus.center_x), round(limbus.center_y)),
                (round(limbus.semi_major), round(limbus.semi_minor)), limbus.angle_deg,
                0, 360, (0, 220, 220), 2)
    cv2.putText(preview, f"Cyclotorsion {registration.rotation_deg:+.2f} deg | conf {registration.confidence:.2f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 220, 220), 2, cv2.LINE_AA)
    out_file = PROJECT_ROOT / "output" / "cyclotorsion_demo_result.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print(f"[DEMO] Saved visual demonstration to: {out_file}")
    if not headless:
        cv2.imshow("Cyclotorsion Detection Demo", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return True


def run_image_demo(image_path: Path, test_angle: float = 3.0, headless: bool = False):
    print(f"\n[DEMO] Loading image: {image_path}")
    img_ref = cv2.imread(str(image_path))
    if img_ref is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Pentacam/seated screenshots use the dedicated geometry/session contract.
    # This avoids presenting the general ELITA detector as a validated Pentacam
    # result and keeps the demo angle consistent with the production path.
    from pupil_tracking.pentacam.detector import PentacamIrisDetector
    seated_probe = PentacamIrisDetector().detect(img_ref, extract_features=False, refine_boundary=False)
    if seated_probe.valid and seated_probe.geometry.limbus_radius_px > seated_probe.geometry.pupil_radius_px * 1.5:
        if _run_seated_image_demo(img_ref, image_path, test_angle, headless):
            return

    detector = UnifiedDetector()
    engine = RegistrationEngine()

    print("[DEMO] Detecting pupil & limbus on reference frame...")
    det_ref = detector.detect(img_ref)
    if not det_ref.has_both:
        print("Error: Could not locate both pupil and limbus on reference image.")
        return

    rot_center = (det_ref.limbus.ellipse.center_x, det_ref.limbus.ellipse.center_y)
    print(f"[DEMO] Simulating clinical eye rotation of {test_angle:+.1f}°...")
    img_rot = rotate_image(img_ref, test_angle, center=rot_center)

    print("[DEMO] Detecting pupil & limbus on rotated eye...")
    det_rot = detector.detect(img_rot)

    # Detect Iris ROI and Iris Features on current eye
    from pupil_tracking.iris.detect import IrisFeatureDetector
    iris_det = IrisFeatureDetector()
    iris_res = iris_det.detect(img_rot, pupil=det_rot.pupil.ellipse, limbus=det_rot.limbus.ellipse)
    roi = iris_res.feature_set.roi

    print("\n[DEMO] Extracting Iris Annular ROI & Iris Texture Landmarks...")
    print(f"  Iris ROI Valid:         {roi.valid}")
    if roi.valid:
        in_r = roi.pupil_radius_px * (1.0 + roi.inner_inset_frac)
        out_r = roi.limbus_radius_px * (1.0 - roi.outer_inset_frac)
        print(f"  Iris Annulus Radii:     r_inner={in_r:.1f}px -> r_outer={out_r:.1f}px")
    print(f"  Detected Iris Features: {len(iris_res.feature_set.features)} micro-landmarks (crypts/furrows)")
    print(f"  Feature Spread Coverage:{getattr(iris_res.feature_set, 'region_coverage', 0.0):.1%}")

    print("\n[DEMO] Computing cyclotorsion via 5-stream registration engine...")
    reg_res = engine.register(img_ref, img_rot, det_ref, det_rot)

    print("\n" + "=" * 55)
    print("CYCLOTORSION DETECTION RESULT:")
    print(f"  Ground-Truth Applied Rotation: {test_angle:+.2f}°")
    print(f"  Detected Cyclotorsion Angle:   {reg_res.torsion_deg:+.2f}°")
    print(f"  Angular Absolute Error:       {abs(abs(reg_res.torsion_deg) - abs(test_angle)):.4f}°")
    print(f"  Tracking Confidence:           {reg_res.confidence:.1%}")
    print(f"  Clinical Safety Rating:        {reg_res.quality.value}")
    print("=" * 55 + "\n")

    # Render HUD overlay with Iris ROI + Features
    vis = draw_registration_overlay(img_rot, reg_res, detection_ref=det_ref, detection_curr=det_rot)
    
    # Draw Iris ROI concentric rings & features
    if roi.valid:
        rcx, rcy = int(round(roi.center_x)), int(round(roi.center_y))
        in_r = int(round(roi.pupil_radius_px * (1.0 + roi.inner_inset_frac)))
        out_r = int(round(roi.limbus_radius_px * (1.0 - roi.outer_inset_frac)))
        cv2.circle(vis, (rcx, rcy), in_r, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (rcx, rcy), out_r, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"IRIS ROI [{len(iris_res.feature_set.features)} features]",
            (rcx - 80, rcy - out_r - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

    for feat in iris_res.feature_set.features:
        fx, fy = int(round(feat.x)), int(round(feat.y))
        cv2.circle(vis, (fx, fy), 2, (0, 255, 255), -1)
        cv2.circle(vis, (fx, fy), 4, (255, 0, 255), 1, cv2.LINE_AA)
        ang = np.deg2rad(feat.orientation_deg)
        x2 = int(round(fx + 5.0 * np.cos(ang)))
        y2 = int(round(fy + 5.0 * np.sin(ang)))
        cv2.line(vis, (fx, fy), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

    # Save output preview
    out_file = PROJECT_ROOT / "output" / "cyclotorsion_demo_result.jpg"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    print(f"[DEMO] Saved visual demonstration to: {out_file}")

    if not headless:
        print("[DEMO] Displaying result window. Press any key or ESC to close.")
        cv2.imshow("Cyclotorsion Detection Demo - Surgical HUD", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream_demo(source, is_camera: bool = False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open stream source: {source}")
        return

    detector = UnifiedDetector()
    engine = RegistrationEngine()

    print("\n[DEMO] Initializing cyclotorsion tracker...")
    print("[DEMO] Step 1: Acquiring baseline Reference frame (Frame 0 = 0.0°)...")

    det_ref = None
    img_ref = None

    window_name = "Live Cyclotorsion Surgical HUD (Press ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            det_curr = detector.detect(frame)

            # Acquire reference frame if not yet locked
            if img_ref is None or det_ref is None:
                if det_curr.has_both:
                    img_ref = frame.copy()
                    det_ref = det_curr
                    print(f"[DEMO] Baseline locked at frame {frame_idx}!")
                else:
                    cv2.putText(
                        frame, "LOCKING BASELINE REFERENCE...", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
                    )
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                    continue

            # Compute cyclotorsion relative to baseline
            display = frame.copy()
            if det_curr.has_both:
                reg_res = engine.register(img_ref, frame, det_ref, det_curr)
                display = draw_registration_overlay(
                    display, reg_res, detection_ref=det_ref, detection_curr=det_curr
                )
            else:
                cv2.putText(
                    display, "INTERLOCK HALT: EYE OCCLUDED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset baseline
                print("[DEMO] Resetting baseline reference frame...")
                img_ref = None
                det_ref = None

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live Cyclotorsion Detection Demonstration")
    parser.add_argument("--image", type=str, help="Path to single eye image for rotation test")
    parser.add_argument("--video", type=str, help="Path to video file for live tracking")
    parser.add_argument("--camera", action="store_true", help="Use live webcam feed")
    parser.add_argument("--angle", type=float, default=3.0, help="Test rotation angle in degrees for image demo (default: 3.0)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window (saves result image only)")

    args = parser.parse_args()

    if args.image:
        run_image_demo(Path(args.image), test_angle=args.angle, headless=args.headless)
    elif args.video:
        run_stream_demo(args.video)
    elif args.camera:
        run_stream_demo(0, is_camera=True)
    else:
        # Default fallback: run on default clinical image
        default_img = PROJECT_ROOT / "clinical_data" / "clean" / "eye_01.jpeg"
        if default_img.exists():
            print(f"No arguments provided. Running default image demo on {default_img.name} with {args.angle:+.1f}° rotation.")
            run_image_demo(default_img, test_angle=args.angle, headless=args.headless)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
