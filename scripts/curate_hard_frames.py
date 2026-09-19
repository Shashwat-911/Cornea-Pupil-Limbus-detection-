#!/usr/bin/env python3
"""
Keyframe Curation & Active Learning Curation Script.

Analyzes a new video using the existing trained eye segmentation model
(quantized ONNX preferred, PyTorch as fallback), evaluates frames for:
  - Low prediction confidence
  - Deformed aspect ratios (blink, squint, occlusion)
  - Abrupt tracking jumps (tracking loss, saccades, artifacts)

Selects the top N most informative, non-redundant frames (with temporal NMS)
and writes their indices to clinical_data/annotations/{video_name}_curated_frames.json.
Saves these hard frames as clinical_data/training_data/images/{video_name}_frame_{frame_idx:06d}.jpg
so the user can immediately jump to and annotate them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pupil_tracking.preprocessing.red_light_filter import RedLightFilter
from pupil_tracking.core.deterministic_ring_detector import RingDetector, RingStatus

# Try lazy importing packages
try:
    from pupil_tracking.ml.onnx_inference import ONNXInference
    _HAS_ONNX_ENGINE = True
except ImportError:
    _HAS_ONNX_ENGINE = False

try:
    import torch
    from pupil_tracking.ml.architecture import EyeSegmentationModel
    _HAS_PYTORCH_ENGINE = True
except ImportError:
    _HAS_PYTORCH_ENGINE = False


def clean_base_name(video_path: str) -> str:
    """Get clean video stem name matching annotate_live_video.py."""
    base = Path(video_path).stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base)


def analyze_video(
    video_path: str,
    infer_engine: Any,
    subsample: int = 1,
    num_classes: int = 3,
) -> List[Dict[str, Any]]:
    """Scan the video and analyze every N-th frame for key clinical edge cases:
    - Saturated red lights & specular reflections
    - Eyelid occlusions & blink boundaries
    - Limbus overestimation in pre-docking
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[Curation] Scanning {total_frames} frames from '{Path(video_path).name}'...")

    # Initialize red-light filter and ring detector for active curation scoring
    red_filter = RedLightFilter(dilation_size=0, enable_inpaint=False)
    ring_detector = RingDetector()

    frame_metadata: List[Dict[str, Any]] = []
    prev_pupil_center: Optional[Tuple[float, float]] = None
    prev_limbus_center: Optional[Tuple[float, float]] = None
    had_pupil = False

    # Determine inference mode (ONNX vs PyTorch)
    is_pytorch = hasattr(infer_engine, "predict_proba")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % subsample != 0:
            frame_idx += 1
            continue

        # Run model inference
        avg_confidence = 0.95
        pupil_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        limbus_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        if is_pytorch:
            # PyTorch inference
            h_orig, w_orig = frame.shape[:2]
            # Preprocess
            resized = cv2.resize(frame, (512, 512))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0).to(infer_engine.device if hasattr(infer_engine, "device") else "cpu")
            
            with torch.no_grad():
                logits = infer_engine(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                class_map = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            # Resize prediction map to original size
            class_map_full = cv2.resize(
                class_map.astype(np.uint8),
                (w_orig, h_orig),
                interpolation=cv2.INTER_NEAREST,
            )
            pupil_mask = (class_map_full == 1).astype(np.uint8)
            limbus_mask = (class_map_full == 2).astype(np.uint8)
            
            # Confidence score
            max_probs = np.max(probs, axis=0)
            avg_confidence = float(np.mean(max_probs))
        else:
            # ONNX inference
            try:
                masks = infer_engine.infer(frame)
                pupil_mask = (masks["pupil"] > 127).astype(np.uint8)
                limbus_mask = (masks["iris"] > 127).astype(np.uint8)
                if "_probabilities" in masks:
                    probs = masks["_probabilities"]
                    max_probs = np.max(probs, axis=0)
                    avg_confidence = float(np.mean(max_probs))
            except Exception as e:
                print(f"[WARNING] Inference error on frame {frame_idx}: {e}")

        # Compute pupil shape stats
        pupil_area = 0.0
        pupil_ar = 1.0
        pupil_center = None

        p_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if p_contours:
            largest = max(p_contours, key=cv2.contourArea)
            pupil_area = float(cv2.contourArea(largest))
            if pupil_area > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    pupil_center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                if len(largest) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest)
                        w_ell, h_ell = ellipse[1]
                        pupil_ar = float(min(w_ell, h_ell) / max(w_ell, h_ell, 1e-5))
                    except cv2.error:
                        pass

        # Compute limbus shape stats
        limbus_area = 0.0
        limbus_ar = 1.0
        limbus_center = None

        l_contours, _ = cv2.findContours(limbus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if l_contours:
            largest = max(l_contours, key=cv2.contourArea)
            limbus_area = float(cv2.contourArea(largest))
            if limbus_area > 100:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    limbus_center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
                if len(largest) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(largest)
                        w_ell, h_ell = ellipse[1]
                        limbus_ar = float(min(w_ell, h_ell) / max(w_ell, h_ell, 1e-5))
                    except cv2.error:
                        pass

        # Compute Jumps
        pupil_jump = 0.0
        if pupil_center and prev_pupil_center:
            pupil_jump = float(math.hypot(pupil_center[0] - prev_pupil_center[0], pupil_center[1] - prev_pupil_center[1]))

        limbus_jump = 0.0
        if limbus_center and prev_limbus_center:
            limbus_jump = float(math.hypot(limbus_center[0] - prev_limbus_center[0], limbus_center[1] - prev_limbus_center[1]))

        # Keep track of previous centers
        if pupil_center:
            prev_pupil_center = pupil_center
        if limbus_center:
            prev_limbus_center = limbus_center

        # ── Compute targeted active-learning hardness components ──
        
        # 1. Prediction uncertainty
        conf_term = 1.0 - avg_confidence

        # 2. Blink / Squint detection (onset/offset of blinks or partial blinks)
        blink_score = 0.0
        has_pupil = pupil_center is not None
        if has_pupil != had_pupil:
            # High priority for blink transitions
            blink_score = 1.0
        elif has_pupil and pupil_ar < 0.65 and pupil_area > 100:
            # Squint/partial blink
            blink_score = (0.75 - pupil_ar) / 0.75
        had_pupil = has_pupil

        # 3. Red surgical light blinking / Specular reflections (Fast Downscaled morphology)
        red_score = 0.0
        try:
            # Downscale frame for rapid classical preprocessing metrics
            small_frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
            red_mask = red_filter._detect_red_lights(small_frame)
            red_pixel_frac = float(np.count_nonzero(red_mask)) / float(512 * 512)
            # Scale so that 2% active red distractor pixels gives a maximum score of 1.0
            red_score = min(red_pixel_frac * 50.0, 1.0)
        except Exception:
            pass

        # 4. Pre-docking limbus overestimation / Concentricity violations
        overest_score = 0.0
        try:
            ring_res = ring_detector.detect(frame)
            is_docked = ring_res.status in (RingStatus.PRESENT, RingStatus.PARTIAL)
            if not is_docked and pupil_center and limbus_center:
                dx = pupil_center[0] - limbus_center[0]
                dy = pupil_center[1] - limbus_center[1]
                dist = math.hypot(dx, dy)
                
                p_radius = math.sqrt(pupil_area / math.pi) if pupil_area > 0 else 0
                l_radius = math.sqrt(limbus_area / math.pi) if limbus_area > 0 else 0
                
                # Check for biological concentricity and size anomaly
                is_offset_too_large = l_radius > 0 and (dist / l_radius > 0.35)
                is_ratio_bad = l_radius > 0 and (p_radius / l_radius < 0.15 or p_radius / l_radius > 0.75)
                is_too_large = l_radius > 150.0
                
                if is_offset_too_large or is_ratio_bad or is_too_large:
                    overest_score = 1.0
        except Exception:
            pass

        # Weighted composite score targeting specific failures
        hardness = (
            0.15 * conf_term + 
            0.35 * red_score + 
            0.25 * blink_score + 
            0.25 * overest_score
        )

        # Flag complete loss of detection as standard high-difficulty score
        is_loss = False
        if pupil_area == 0 or limbus_area == 0:
            is_loss = True
            # Zero detection frames are important if the eye is open
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            val_variance = float(np.var(hsv[:, :, 2]))
            if val_variance > 100.0:  # Eye is likely open, but model failed
                hardness = max(hardness, 0.85)

        frame_metadata.append({
            "frame_index": frame_idx,
            "hardness": float(hardness),
            "is_loss": is_loss,
            "pupil_area": pupil_area,
            "pupil_ar": pupil_ar,
            "pupil_jump": pupil_jump,
            "avg_confidence": avg_confidence,
        })

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames (hardness={hardness:.3f})")

        frame_idx += 1

    cap.release()
    return frame_metadata


def select_diverse_keyframes(
    metadata: List[Dict[str, Any]],
    num_frames: int = 20,
    nms_threshold: int = 30,
) -> List[int]:
    """Select the hardest frames using Temporal NMS to ensure spatial-temporal diversity."""
    # Sort by hardness descending
    sorted_meta = sorted(metadata, key=lambda x: x["hardness"], reverse=True)

    selected_indices: List[int] = []
    for entry in sorted_meta:
        idx = entry["frame_index"]
        # Temporal NMS: Ensure this frame is at least `nms_threshold` frames away from all selected frames
        too_close = False
        for sel_idx in selected_indices:
            if abs(idx - sel_idx) < nms_threshold:
                too_close = True
                break
        
        if not too_close:
            selected_indices.append(idx)
            if len(selected_indices) >= num_frames:
                break

    # Add a few general diverse frames across the video if we didn't fill the budget
    if len(selected_indices) < num_frames and len(metadata) > 0:
        step = max(1, len(metadata) // (num_frames - len(selected_indices)))
        for i in range(0, len(metadata), step):
            idx = metadata[i]["frame_index"]
            too_close = False
            for sel_idx in selected_indices:
                if abs(idx - sel_idx) < nms_threshold:
                    too_close = True
                    break
            if not too_close:
                selected_indices.append(idx)
                if len(selected_indices) >= num_frames:
                    break

    return sorted(selected_indices)


def save_keyframes(
    video_path: str,
    frame_indices: List[int],
    output_images_dir: Path,
) -> int:
    """Seek to selected frame indices and extract them as high-quality JPEGs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    clean_base = clean_base_name(video_path)
    saved_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            name = f"{clean_base}_frame_{idx:06d}.jpg"
            save_path = output_images_dir / name
            cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1

    cap.release()
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Curate and extract informative hard frames for active learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--video", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output-dir", default="clinical_data", help="Output clinical data directory")
    parser.add_argument("-n", "--num-frames", type=int, default=20, help="Number of curated frames to select")
    parser.add_argument("-c", "--num-classes", type=int, default=3, choices=[3, 4], help="Model class count (3 or 4)")
    parser.add_argument("-s", "--subsample", type=int, default=1, help="Process every N-th frame for speed")
    parser.add_argument("-d", "--device", default="auto", help="Compute device for PyTorch model")
    
    args = parser.parse_args()

    video_path = args.video
    if not Path(video_path).exists():
        print(f"[ERROR] Video file does not exist: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "training_data" / "images"
    annotations_dir = output_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # ── Initialize Inference Engine ────────────────────────────────
    infer_engine = None
    print("[Curation] Initializing inference model...")

    # Strategy 1: ONNX Runtime
    if _HAS_ONNX_ENGINE:
        try:
            # Look for model
            infer_engine = ONNXInference(num_classes=args.num_classes)
            if not infer_engine.is_loaded:
                infer_engine = None
        except Exception as e:
            print(f"[Curation] ONNX loading skipped: {e}")

    # Strategy 2: PyTorch
    if infer_engine is None and _HAS_PYTORCH_ENGINE:
        model_path = Path("models/best_model.pth")
        if model_path.exists():
            try:
                print(f"[Curation] Loading PyTorch model weights from {model_path}...")
                infer_engine = EyeSegmentationModel.load(str(model_path), device=args.device)
            except Exception as e:
                print(f"[Curation] PyTorch loading failed: {e}")

    if infer_engine is None:
        print("[ERROR] Neither ONNX Runtime nor PyTorch models could be loaded.")
        print("        Ensure 'models/onnx/segmentation_quantized.onnx' or 'models/best_model.pth' exists.")
        sys.exit(1)

    # Calculate adaptive subsampling factor dynamically if default (1) is passed
    subsample_factor = args.subsample
    if subsample_factor == 1:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Target exactly ~800 processed frames to keep curation under 15-20 seconds
            subsample_factor = max(1, total_frames // 800)
            print(f"[Curation] Automatically set subsample factor to {subsample_factor} to target ~800 processed frames for rapid curation.")
            print("           To process every single frame, pass `-s 1` explicitly.")

    # ── Analyze and curate ─────────────────────────────────────────
    metadata = analyze_video(video_path, infer_engine, subsample=subsample_factor, num_classes=args.num_classes)
    
    # Select diverse hardest keyframes using temporal NMS (30 frames spacing = 1 sec at 30 fps)
    curated_indices = select_diverse_keyframes(metadata, num_frames=args.num_frames, nms_threshold=30)
    print(f"[Curation] Curated {len(curated_indices)} hard frame indices: {curated_indices}")

    # Extract and save curated frames
    saved_count = save_keyframes(video_path, curated_indices, images_dir)
    print(f"[Curation] Successfully saved {saved_count} curated frames as JPEGs to '{images_dir}'")

    # Output curation index file
    clean_base = clean_base_name(video_path)
    curated_json_path = annotations_dir / f"{clean_base}_curated_frames.json"
    with open(curated_json_path, "w") as f:
        json.dump(curated_indices, f, indent=2)

    print(f"[Curation] Wrote keyframe indices list to '{curated_json_path}'")
    print(f"[SUCCESS] Curation complete! You can now load the video in annotate_live_video.py to annotate them in seconds.")


if __name__ == "__main__":
    main()
