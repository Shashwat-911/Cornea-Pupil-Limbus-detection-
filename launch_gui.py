#!/usr/bin/env python3
"""
╔======================================================================╗
║           Medevplus IXcentai surgical grade                        ║
╚======================================================================╝

USAGE EXAMPLES:
-----------------------------------------------------------------------
  python launch_gui.py                                # Launch GUI
  python launch_gui.py gui                            # Launch GUI
  python launch_gui.py gui --stride 2 --no-fp16       # GUI + overrides
  python launch_gui.py image -i eye.jpg               # Single image
  python launch_gui.py video -i vid.mp4               # Video file
  python launch_gui.py video -i vid.mp4 --stride 3    # Skip frames
  python launch_gui.py video -i vid.mp4 -o out.mp4    # Save output
  python launch_gui.py camera                         # Live webcam

RING DETECTION & ADAPTIVE PIPELINE (NEW):
-----------------------------------------------------------------------
  --ring-mode MODE          auto / docked / pre_docked    [default: auto]
  --ring-classifier PATH    Path to ring classifier model
  --show-ring / --no-show-ring  Draw ring in output       [default: ON]

GRAYSCALE MODE (NEW):
-----------------------------------------------------------------------
  --grayscale MODE          off / auto / force            [default: off]
                            off   = RGB passthrough (original behaviour)
                            auto  = detect & enhance grayscale inputs
                            force = always convert to enhanced grayscale

  When grayscale is FORCE or AUTO (on grayscale input), the displayed
  image is converted to grayscale — like an IR camera feed — with
  coloured detection overlays drawn on top.

PIPELINE FLAGS (gui / video / camera):
-----------------------------------------------------------------------
  --stride N                Process every Nth frame       [default: 1]
  --resolution N            Inference resolution (px)     [default: profile]
  --target-fps F            Target processing FPS         [default: profile]
  --fp16 / --no-fp16        FP16 half-precision           [default: profile]
  --compile / --no-compile  torch.compile JIT             [default: profile]
  --roi / --no-roi          ROI tracking                  [default: ON]
  --optimized / --no-optimized  Use fast pipeline         [default: ON]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import AuditLogger, set_logger
from pupil_tracking.utils.runtime_profile import (
    apply_runtime_optimizations,
    detect_runtime_profile,
)


_RUNTIME_PROFILE = apply_runtime_optimizations(detect_runtime_profile())


# -- Constants ---------------------------------------------------
_CORNEAL_DIAMETER_MM = 12.0

_BANNER = "Medevplus IXcentai - Surgical Grade"


# ================================================================
# Optional fast-pipeline imports (graceful fallback)
# ================================================================

try:
    from pupil_tracking.ml.fast_inference import FastInference
    from pupil_tracking.video.optimized_processor import (
        OptimizedVideoProcessor,
        AsyncCapture,
        FrameResult,
        TrackingQuality,
    )

    _FAST_PIPELINE_AVAILABLE = True
    FastInferenceEngine = FastInference
except ImportError:
    _FAST_PIPELINE_AVAILABLE = False
    FastInferenceEngine = None
    OptimizedVideoProcessor = None
    AsyncCapture = None
    FrameResult = None
    TrackingQuality = None


# ==================================================================
# DISPLAY PATCH 1 of 4 — Grayscale display conversion helper
#
# When grayscale mode is active, this converts the frame to a
# 3-channel grayscale image for DISPLAY purposes.  The detection
# pipeline receives the original frame — it handles its own
# grayscale conversion internally.  This function only affects
# what the user SEES on screen and in output video.
#
# Result: grayscale background with coloured detection circles
# drawn on top — like looking through an IR camera with tracking.
# ==================================================================


def _convert_frame_for_display(
    frame: np.ndarray,
    detector: Any,
) -> np.ndarray:
    """Convert a frame to grayscale for display when mode is active.

    When the detector's grayscale mode is FORCE, this returns a
    3-channel grayscale image (gray replicated to BGR) so that
    coloured overlays (green pupil, blue limbus) are visible on
    a gray background.

    When mode is AUTO, the frame is only converted if the detector
    actually applied grayscale processing (i.e. the input was
    detected as grayscale).

    When mode is OFF, returns the original frame unchanged.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame from camera/video/image.
    detector : UnifiedDetector
        Detector instance — checked for current grayscale mode
        and last processing info.

    Returns
    -------
    np.ndarray
        Frame for display — always 3-channel BGR uint8.
        Either the original RGB or a grayscale-replicated version.
    """
    from pupil_tracking.preprocessing.grayscale_handler import GrayscaleMode

    mode = detector.grayscale_mode

    if mode == GrayscaleMode.OFF:
        return frame

    if mode == GrayscaleMode.FORCE:
        # Always show grayscale display
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if mode == GrayscaleMode.AUTO:
        # Show grayscale only if the detector actually converted
        gs_info = detector.last_grayscale_info
        if gs_info is not None and gs_info.conversion_applied:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return frame


# ==================================================================
# END DISPLAY PATCH 1
# ==================================================================


# ================================================================
# CLI — Image Processing
# ================================================================


def process_image(
    image_path: str,
    model_path: Optional[str] = None,
    device: str = "auto",
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process a single image and display results."""
    from pupil_tracking.core.detector import UnifiedDetector

    image = cv2.imread(image_path)
    if image is None:
        print(f"\n  ✗ ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    cfg = get_config()

    # Ring configuration overrides
    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    if ring_classifier:
        cfg.ring.classifier.classifier_path = ring_classifier

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )

    result = detector.detect(image, source=image_path, force_mode=force_mode)

    # -- calibration from limbus / ring --------------------------
    cal = result.calibration
    has_cal = cal.calibrated if cal else False
    mm_per_px = cal.mm_per_px if has_cal else 0.0

    # -- corneal centre reference (use blended when available) -------
    cc_ref = getattr(result, 'corneal_reference_source', 'pupil')
    if hasattr(result, 'corneal_center'):
        cc = result.corneal_center
        if cc.valid and hasattr(cc, 'center_px'):
            cc_ref = f"blended ({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f})"

    print(f"\n{'=' * 64}")
    print(f"  IMAGE ANALYSIS RESULTS")
    print(f"{'=' * 64}")
    print(f"  File:       {image_path}")
    print(
        f"  Size:       {result.metadata.image_width}"
        f" × {result.metadata.image_height} px"
    )
    print(f"  Category:   {getattr(result, 'image_category', 'unknown').upper()}")
    print(f"  Corneal Ref:{cc_ref}")

    ring_status = getattr(result, "ring_status", "unknown")
    ring_conf = getattr(result, "ring_confidence", 0.0)
    print(f"  Ring:       {ring_status} (conf: {ring_conf:.3f})")

    print(
        f"  Quality:    {result.overall_quality.value} "
        f"({result.overall_confidence:.3f})"
    )

    gs_info = detector.last_grayscale_info
    gs_mode_display = detector.grayscale_mode.name
    if gs_info is not None:
        gs_applied = "YES" if gs_info.conversion_applied else "NO"
        gs_input = "grayscale" if gs_info.was_grayscale else "RGB"
        print(
            f"  Grayscale:  mode={gs_mode_display}, applied={gs_applied}, "
            f"input={gs_input}"
        )
        if gs_info.conversion_applied:
            print(
                f"              contrast {gs_info.contrast_before:.1f} "
                f"→ {gs_info.contrast_after:.1f}"
            )
    else:
        print(f"  Grayscale:  mode={gs_mode_display} (not applied)")

    print(f"{'-' * 64}")

    # Ring Geometry
    if ring_status == "ring_present":
        rc = getattr(result, "ring_center", None)
        rr = getattr(result, "ring_radius", None)
        if rc and rr:
            print(f"\n  * SUCTION RING")
            print(f"    Center:       ({rc[0]:.1f}, {rc[1]:.1f}) px")
            print(f"    Radius:       {rr:.1f} px")
            print(f"    Diameter:     {rr * 2.0:.1f} px")
            if has_cal:
                print(f"    Diameter:     {rr * 2.0 * mm_per_px:.3f} mm")
            print(
                f"    Dots:         {getattr(result, 'ring_dot_count', 0)}"
            )

    # Pupil Geometry
    if result.pupil.detected:
        p = result.pupil.ellipse
        dia_px = p.radius * 2.0
        print(f"\n  * PUPIL")
        print(f"    Center:       ({p.center_x:.1f}, {p.center_y:.1f}) px")
        print(f"    Diameter:     {dia_px:.1f} px")
        if has_cal:
            print(f"    Diameter:     {dia_px * mm_per_px:.3f} mm")
        print(f"    Semi-axes:    {p.semi_major:.1f} × {p.semi_minor:.1f} px")
        print(f"    Angle:        {p.angle_deg:.1f}°")
        ft = getattr(p, "fit_type", None) or getattr(result.pupil, "fit_type", None)
        if ft:
            print(f"    Fit type:     {ft}")
        print(f"    Confidence:   {result.pupil.confidence:.3f}")
        print(f"    Quality:      {result.pupil.quality.value}")
    else:
        print(f"\n  o PUPIL:  NOT DETECTED")

    # Limbus Geometry
    if result.limbus.detected:
        el = result.limbus.ellipse
        dia_px = el.radius * 2.0
        print(f"\n  * LIMBUS")
        print(f"    Center:       ({el.center_x:.1f}, {el.center_y:.1f}) px")
        print(f"    Diameter:     {dia_px:.1f} px")
        if has_cal:
            print(f"    Diameter:     {dia_px * mm_per_px:.3f} mm")
        print(f"    Semi-axes:    {el.semi_major:.1f} × {el.semi_minor:.1f} px")
        print(f"    Angle:        {el.angle_deg:.1f}°")
        ft = getattr(el, "fit_type", None) or getattr(result.limbus, "fit_type", None)
        if ft:
            print(f"    Fit type:     {ft}")
        print(f"    Confidence:   {result.limbus.confidence:.3f}")
        print(f"    Quality:      {result.limbus.quality.value}")
    else:
        print(f"\n  o LIMBUS:  NOT DETECTED")

    if result.corneal_center.valid:
        cc = result.corneal_center
        print(f"\n  * CORNEAL CENTRE & OFFSET")
        print(f"    Centre:       ({cc.center_px[0]:.1f}, {cc.center_px[1]:.1f}) px")
        print(
            f"    Reference:    "
            f"{getattr(result, 'corneal_reference_source', 'limbus')}"
        )
        print(f"    Offset:       {cc.offset_magnitude_px:.1f} px")
        if cc.offset_magnitude_mm is not None:
            print(f"    Offset:       {cc.offset_magnitude_mm:.3f} mm")
        print(f"    Angle:        {cc.offset_angle_deg:.1f}°")
        for alert in getattr(cc, "alerts", []):
            print(f"    ! {alert}")

    if has_cal:
        print(f"\n  * CALIBRATION")
        print(f"    Source:       {cal.source}")
        print(f"    Scale:        {cal.px_per_mm:.2f} px/mm")
        print(f"    Scale:        {cal.mm_per_px:.4f} mm/px")

    print(f"\n{'-' * 64}")
    print(f"  Processing time: {result.metadata.processing_time_ms:.1f} ms")
    print(f"{'=' * 64}")

    # ==============================================================
    # DISPLAY PATCH 2 of 4 — Convert image for display
    #
    # The detector already processed the original image internally.
    # Now convert the DISPLAY image to grayscale if mode is active.
    # Detection overlays (green/blue circles) will be drawn on the
    # grayscale background, giving the IR camera look.
    # ==============================================================
    display_frame = _convert_frame_for_display(image, detector)
    annotated = _draw_cli_overlay(
        display_frame,
        result,
        show_ring=show_ring,
        grayscale_mode=gs_mode_display,
    )
    # ==============================================================
    # END DISPLAY PATCH 2
    # ==============================================================
    cv2.imshow("Detection Result", annotated)
    print("\n  Press any key to close the window…\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ================================================================
# CLI — Video File Processing
# ================================================================


def process_video(
    video_path: str,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process a video file with optional optimised pipeline."""
    from pupil_tracking.core.detector import UnifiedDetector
    from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
    from pupil_tracking.core.corneal_center import CornealCenterCalculator

    cfg = get_config()

    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )
    tracker = EyeKalmanTracker(config=cfg)
    corneal_calc = CornealCenterCalculator(config=cfg)

    # -- read CLI flags (with defaults) --------------------------
    stride = getattr(args, "stride", 1) or 1
    resolution = getattr(args, "resolution", 320) or 320
    use_fp16 = getattr(args, "fp16", True)
    use_compile = getattr(args, "compile", True)
    use_roi = getattr(args, "roi", True)
    target_fps = getattr(args, "target_fps", 20.0) or 20.0
    use_optimized = getattr(args, "optimized", True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n  ✗ ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'=' * 64}")
    print(f"  VIDEO PROCESSING")
    print(f"{'=' * 64}")
    print(f"  File:         {video_path}")
    print(f"  Resolution:   {width} × {height} @ {fps:.1f} FPS")
    print(f"  Total frames: {total_frames}")
    print(f"{'-' * 64}")
    print(f"  Ring Mode:    {ring_mode.upper()}")
    print(f"  Grayscale:    {grayscale_mode.upper()}")
    print(f"  Stride:       {stride}  (every {stride} frame(s))")
    print(f"  Inference:    {resolution} px")
    print(f"  FP16:         {'ON' if use_fp16 else 'OFF'}")
    print(f"  Compile:      {'ON' if use_compile else 'OFF'}")
    print(f"  ROI:          {'ON' if use_roi else 'OFF'}")
    print(f"  Target FPS:   {target_fps:.1f}")

    # -- output video writer -------------------------------------
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = fps / stride
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
        print(f"  Output:       {output_path}")

    # -- try optimised pipeline ----------------------------------
    fast_engine = None
    opt_processor = None

    if use_optimized and _FAST_PIPELINE_AVAILABLE:
        model_file = _find_model_path(detector, cfg)
        if model_file:
            try:
                opt_processor = OptimizedVideoProcessor(
                    model_path=model_file,
                    device="auto",
                    input_size=resolution,
                    half_precision=use_fp16,
                    use_compile=use_compile,
                    enable_auto_roi=use_roi,
                    roi_cache_ttl=getattr(args, "roi_cache", 5),
                    process_noise=getattr(args, "kalman_process_noise", 0.03),
                    measurement_noise=getattr(args, "kalman_measure_noise", 0.1),
                    batch_size=_RUNTIME_PROFILE.recommended_batch_size,
                )
                print(
                    f"  Pipeline:     ✓ OPTIMISED "
                    f"(FP16={use_fp16}, ROI={use_roi}, "
                    f"batch={opt_processor.get_stats().get('batch_size')})"
                )
            except Exception as exc:
                print(f"  Pipeline:     ! Optimised failed: {exc}")
                print(f"                  Falling back to classic")
                opt_processor = None

    if opt_processor is None:
        print(f"  Pipeline:     Classic (UnifiedDetector)")

    print(f"{'=' * 64}\n")

    # -- processing loop -----------------------------------------
    results_history: List[Dict[str, Any]] = []
    raw_frame_idx = 0
    processed_count = 0
    start_time = time.monotonic()

    gs_mode_display = grayscale_mode.upper()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame_idx += 1

            # stride
            if stride > 1 and (raw_frame_idx % stride) != 0:
                continue

            processed_count += 1

            # -- detect ------------------------------------------
            if opt_processor is not None:
                try:
                    frame_result = opt_processor.process_frame(frame, processed_count)
                    result_dict = _frame_result_to_dict(frame_result)
                except Exception:
                    result = detector.detect(
                        frame,
                        frame_number=processed_count,
                        source=video_path,
                        force_mode=force_mode,
                    )
                    result_dict = result.to_dict()
            else:
                result = detector.detect(
                    frame,
                    frame_number=processed_count,
                    source=video_path,
                    force_mode=force_mode,
                )
                smoothed = tracker.update(result)
                if smoothed.has_both:
                    smoothed.corneal_center = corneal_calc.calculate(
                        smoothed.pupil,
                        smoothed.limbus,
                        result.calibration,
                    )
                smoothed.calibration = result.calibration
                if (
                    getattr(result, "ring_status", "unknown") == "ring_present"
                    and getattr(result, "ring_center", None) is not None
                    and getattr(smoothed, "pupil", None) is not None
                    and getattr(smoothed.pupil, "ellipse", None) is not None
                ):
                    px = smoothed.pupil.ellipse.center_x
                    py = smoothed.pupil.ellipse.center_y
                    points = [(px, py, "pupil")]
                    weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]
                    if getattr(smoothed, "limbus", None) is not None and getattr(smoothed.limbus, "ellipse", None) is not None:
                        points.append(
                            (
                                smoothed.limbus.ellipse.center_x,
                                smoothed.limbus.ellipse.center_y,
                                "limbus",
                            )
                        )
                        weights.append(max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3))
                    points.append((result.ring_center[0], result.ring_center[1], "ring"))
                    weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                    total_w = sum(weights)
                    rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                    rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                    smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                    smoothed.corneal_center.center_px = (rcx, rcy)
                    smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                    smoothed.corneal_center.offset_magnitude_px = math.hypot(px - rcx, py - rcy)
                    smoothed.corneal_center.offset_angle_deg = math.degrees(
                        math.atan2(py - rcy, px - rcx)
                    )
                    smoothed.corneal_center.valid = True
                    if result.calibration.calibrated:
                        smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm((rcx, rcy))
                        dx_mm = (px - rcx) * result.calibration.mm_per_px
                        dy_mm = (py - rcy) * result.calibration.mm_per_px
                        smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
                        smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)
                    smoothed.corneal_reference_source = getattr(
                        result,
                        "corneal_reference_source",
                        "suction_ring",
                    )
                result_dict = smoothed.to_dict()

            results_history.append(result_dict)

            # ======================================================
            # DISPLAY PATCH 3 of 4 — Convert video frame for display
            #
            # detector.detect(frame) already received the ORIGINAL
            # frame — the detector handles grayscale internally.
            # Now convert the display copy so the output video and
            # any preview shows grayscale with coloured overlays.
            # ======================================================
            if writer is not None:
                display_frame = _convert_frame_for_display(frame, detector)
                if opt_processor is not None:
                    annotated = _draw_cli_overlay_from_dict(
                        display_frame,
                        result_dict,
                        show_ring=show_ring,
                        grayscale_mode=gs_mode_display,
                    )
                else:
                    annotated = _draw_cli_overlay(
                        display_frame,
                        smoothed,
                        show_ring=show_ring,
                        grayscale_mode=gs_mode_display,
                    )
                writer.write(annotated)
            # ======================================================
            # END DISPLAY PATCH 3
            # ======================================================

            # -- progress bar ------------------------------------
            if total_frames > 0:
                pct = raw_frame_idx / total_frames * 100
                elapsed = time.monotonic() - start_time
                rate = raw_frame_idx / elapsed if elapsed > 0 else 0
                eta = (total_frames - raw_frame_idx) / rate if rate > 0 else 0

                bar_len = 35
                filled = int(bar_len * raw_frame_idx / total_frames)
                bar = "█" * filled + "░" * (bar_len - filled)

                sys.stdout.write(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"fr {raw_frame_idx}/{total_frames}  "
                    f"proc {processed_count}  "
                    f"ETA {eta:.0f}s  "
                    f"({rate:.1f} fr/s)    "
                )
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n  ! Interrupted by user")

    # -- cleanup -------------------------------------------------
    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.monotonic() - start_time
    avg_fps = processed_count / elapsed if elapsed > 0 else 0

    print(f"\n\n{'=' * 64}")
    print(f"  COMPLETE")
    print(f"{'-' * 64}")
    print(f"  Processed:    {processed_count} frames")
    print(f"  Elapsed:      {elapsed:.1f} s")
    print(f"  Average FPS:  {avg_fps:.1f}")

    # -- auto-export CSV alongside output video ------------------
    if output_path:
        csv_path = str(Path(output_path).with_suffix(".csv"))
        _export_results_csv(results_history, csv_path)
        print(f"  Results CSV:  {csv_path}")
        print(f"  Output video: {output_path}")

    print(f"{'=' * 64}\n")


# ================================================================
# CLI — Camera Processing
# ================================================================


def process_camera(
    camera_id: int = 0,
    model_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
) -> None:
    """Process live camera feed."""
    from pupil_tracking.core.detector import UnifiedDetector
    from pupil_tracking.video.kalman_tracker import EyeKalmanTracker
    from pupil_tracking.core.corneal_center import CornealCenterCalculator

    cfg = get_config()
    ring_classifier = getattr(args, "ring_classifier", None)
    ring_mode = getattr(args, "ring_mode", "auto")
    force_mode = None if ring_mode == "auto" else ring_mode
    show_ring = getattr(args, "show_ring", True)

    grayscale_mode = getattr(args, "grayscale", "off")

    detector = UnifiedDetector(
        model_path=model_path,
        ring_classifier_path=ring_classifier,
        config=cfg,
        grayscale_mode=grayscale_mode,
    )
    tracker = EyeKalmanTracker(config=cfg)
    corneal_calc = CornealCenterCalculator(config=cfg)

    # -- CLI flags -----------------------------------------------
    resolution = getattr(args, "resolution", 320) or 320
    use_fp16 = getattr(args, "fp16", True)
    use_compile = getattr(args, "compile", True)
    use_roi = getattr(args, "roi", True)
    target_fps = getattr(args, "target_fps", 20.0) or 20.0
    stride = getattr(args, "stride", 1) or 1
    use_optimized = getattr(args, "optimized", True)

    print(f"\n{'=' * 64}")
    print(f"{'=' * 64}")
    print(f"  IMAGE ANALYSIS RESULTS")
    print(f"{'=' * 64}")
    print(f"  Device:       camera {camera_id}")
    print(f"  Resolution:   {resolution} px inference")
    print(f"  Target FPS:   {target_fps}")
    print(f"  Stride:       {stride}")
    print(f"  Ring Mode:    {ring_mode.upper()}")
    print(f"  Grayscale:    {grayscale_mode.upper()}")
    print(f"  FP16:         {'ON' if use_fp16 else 'OFF'}")
    print(f"  Compile:      {'ON' if use_compile else 'OFF'}")
    print(f"  ROI:          {'ON' if use_roi else 'OFF'}")
    print(f"{'-' * 64}")
    print(f"  CONTROLS:")
    print(f"    [Q]       Quit")
    print(f"    [S]       Save snapshot")
    print(f"    [SPACE]   Pause / Resume")
    print(f"    [G]       Toggle grayscale mode (OFF → AUTO → FORCE)")
    print(f"{'=' * 64}\n")

    # -- try optimised pipeline ----------------------------------
    opt_processor = None
    async_capture = None

    if use_optimized and _FAST_PIPELINE_AVAILABLE:
        model_file = _find_model_path(detector, cfg)
        if model_file:
            try:
                opt_processor = OptimizedVideoProcessor(
                    model_path=model_file,
                    device="auto",
                    input_size=resolution,
                    half_precision=use_fp16,
                    use_compile=use_compile,
                    enable_auto_roi=use_roi,
                    roi_cache_ttl=getattr(args, "roi_cache", 5),
                    process_noise=getattr(args, "kalman_process_noise", 0.03),
                    measurement_noise=getattr(args, "kalman_measure_noise", 0.1),
                    batch_size=_RUNTIME_PROFILE.recommended_batch_size,
                )
                async_capture = AsyncCapture(
                    camera_id,
                    buffer_size=_RUNTIME_PROFILE.recommended_capture_buffer,
                )
                async_capture.start()
                print(
                    "  Pipeline: ✓ OPTIMISED "
                    f"(batch={opt_processor.get_stats().get('batch_size')})\n"
                )
            except Exception as exc:
                print(f"  Pipeline: ! Optimised failed: {exc}")
                print(f"            Falling back to classic\n")
                opt_processor = None
                if async_capture is not None:
                    try:
                        async_capture.stop()
                    except Exception:
                        pass
                    async_capture = None

    # -- classic fallback ----------------------------------------
    cap = None
    if opt_processor is None:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"  ✗ ERROR: Cannot open camera {camera_id}")
            sys.exit(1)
        print("  Pipeline: Classic (UnifiedDetector)\n")

    # -- processing loop -----------------------------------------
    frame_count = 0
    paused = False
    snapshot_count = 0
    fps_counter = 0
    fps_timer = time.monotonic()
    current_fps = 0.0

    _GRAYSCALE_CYCLE = ["off", "auto", "force"]

    try:
        while True:
            # -- pause -------------------------------------------
            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    paused = False
                    print("  ▶ Resumed")
                elif key == ord("q"):
                    break
                continue

            # -- read frame --------------------------------------
            frame = None
            if async_capture is not None:
                data = async_capture.read(timeout=0.1)
                if data is not None:
                    _, frame, _ = data
            elif cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break

            if frame is None:
                continue

            frame_count += 1

            # -- stride ------------------------------------------
            if stride > 1 and (frame_count % stride) != 0:
                continue

            # ======================================================
            # DISPLAY PATCH 4 of 4 — Convert camera frame for display
            #
            # KEY SEPARATION:
            #   - detector.detect(frame) gets the ORIGINAL frame
            #     (detector handles grayscale internally for model)
            #   - display_frame is the GRAYSCALE version shown to user
            #     (overlays are drawn on this — IR camera look)
            #
            # This conversion happens BEFORE detection so the display
            # frame is ready regardless of which detection path runs.
            # ======================================================
            display_frame = _convert_frame_for_display(frame, detector)
            # ======================================================
            # END DISPLAY PATCH 4 (frame conversion)
            # ======================================================

            # -- detect ------------------------------------------
            if opt_processor is not None:
                try:
                    frame_result = opt_processor.process_frame(frame, frame_count)
                    res_dict = _frame_result_to_dict(frame_result)
                    # Draw overlays on the DISPLAY frame (gray or RGB)
                    annotated = _draw_cli_overlay_from_dict(
                        display_frame,
                        res_dict,
                        show_ring=show_ring,
                        grayscale_mode=detector.grayscale_mode.name,
                    )
                except Exception:
                    annotated = display_frame.copy()
            else:
                result = detector.detect(
                    frame,
                    frame_number=frame_count,
                    source=f"camera:{camera_id}",
                    force_mode=force_mode,
                )
                smoothed = tracker.update(result)
                if smoothed.has_both:
                    smoothed.corneal_center = corneal_calc.calculate(
                        smoothed.pupil,
                        smoothed.limbus,
                        result.calibration,
                    )
                smoothed.calibration = result.calibration

                # Assign ring status manually from raw result to smoothed
                smoothed.ring_status = getattr(result, "ring_status", "unknown")
                smoothed.ring_center = getattr(result, "ring_center", None)
                smoothed.ring_radius = getattr(result, "ring_radius", None)
                smoothed.ring_contour = getattr(result, "ring_contour", None)
                smoothed.ring_dot_count = getattr(result, "ring_dot_count", 0)
                smoothed.corneal_reference_source = getattr(
                    result,
                    "corneal_reference_source",
                    "limbus",
                )
                if (
                    smoothed.ring_status == "ring_present"
                    and smoothed.ring_center is not None
                    and getattr(smoothed, "pupil", None) is not None
                    and getattr(smoothed.pupil, "ellipse", None) is not None
                ):
                    px = smoothed.pupil.ellipse.center_x
                    py = smoothed.pupil.ellipse.center_y
                    points = [(px, py, "pupil")]
                    weights = [max(getattr(smoothed.pupil, "confidence", 0.0), 1e-3)]
                    if getattr(smoothed, "limbus", None) is not None and getattr(smoothed.limbus, "ellipse", None) is not None:
                        points.append(
                            (
                                smoothed.limbus.ellipse.center_x,
                                smoothed.limbus.ellipse.center_y,
                                "limbus",
                            )
                        )
                        weights.append(max(getattr(smoothed.limbus, "confidence", 0.0), 1e-3))
                    points.append((smoothed.ring_center[0], smoothed.ring_center[1], "ring"))
                    weights.append(max(getattr(result, "ring_confidence", 0.0), 1e-3))
                    total_w = sum(weights)
                    rcx = sum(pt[0] * w for pt, w in zip(points, weights)) / total_w
                    rcy = sum(pt[1] * w for pt, w in zip(points, weights)) / total_w
                    smoothed.corneal_reference_source = "+".join(name for _, _, name in points)
                    smoothed.corneal_center.center_px = (rcx, rcy)
                    smoothed.corneal_center.offset_px = (px - rcx, py - rcy)
                    smoothed.corneal_center.offset_magnitude_px = math.hypot(px - rcx, py - rcy)
                    smoothed.corneal_center.offset_angle_deg = math.degrees(
                        math.atan2(py - rcy, px - rcx)
                    )
                    smoothed.corneal_center.valid = True
                    if result.calibration.calibrated:
                        smoothed.corneal_center.center_mm = result.calibration.point_px_to_mm((rcx, rcy))
                        dx_mm = (px - rcx) * result.calibration.mm_per_px
                        dy_mm = (py - rcy) * result.calibration.mm_per_px
                        smoothed.corneal_center.offset_mm = (dx_mm, dy_mm)
                        smoothed.corneal_center.offset_magnitude_mm = math.hypot(dx_mm, dy_mm)

                # Draw overlays on the DISPLAY frame (gray or RGB)
                annotated = _draw_cli_overlay(
                    display_frame,
                    smoothed,
                    show_ring=show_ring,
                    grayscale_mode=detector.grayscale_mode.name,
                )

            # -- FPS ---------------------------------------------
            fps_counter += 1
            now = time.monotonic()
            if (now - fps_timer) >= 1.0:
                current_fps = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            h, w = annotated.shape[:2]
            cv2.putText(
                annotated,
                f"FPS: {current_fps:.1f}",
                (w - 160, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pupil Tracker — Camera", annotated)

            # -- keyboard ----------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
                print("  ⏸ Paused  (press SPACE to resume)")
            elif key == ord("s"):
                snapshot_count += 1
                snap_path = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(snap_path, annotated)
                print(f"  📸 Snapshot saved: {snap_path}")
            elif key == ord("g"):
                current_mode = detector.grayscale_mode.name.lower()
                try:
                    idx = _GRAYSCALE_CYCLE.index(current_mode)
                except ValueError:
                    idx = 0
                next_mode = _GRAYSCALE_CYCLE[(idx + 1) % len(_GRAYSCALE_CYCLE)]
                detector.set_grayscale_mode(next_mode)
                print(f"  🔲 Grayscale mode: {next_mode.upper()}")

    except KeyboardInterrupt:
        print("\n  ! Interrupted")

    # -- cleanup -------------------------------------------------
    if async_capture is not None:
        try:
            async_capture.stop()
        except Exception:
            pass
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n  Done — {frame_count} frames captured\n")


# ================================================================
# CLI Overlay Drawing Helpers
# ================================================================


def _draw_cli_overlay(
    image: np.ndarray,
    result: Any,
    show_ring: bool = True,
    grayscale_mode: str = "",
) -> np.ndarray:
    """Draw detection overlay for CLI display.
    Works with both EyeDetectionResult and SimpleNamespace."""
    out = image.copy()
    h, w = out.shape[:2]

    cal = getattr(result, "calibration", None)
    has_cal = cal is not None and getattr(cal, "calibrated", False)
    mm_per_px = cal.mm_per_px if has_cal else 0.0

    # -- ring ----------------------------------------------------
    ring_status = getattr(result, "ring_status", "unknown")
    if show_ring and ring_status == "ring_present":
        rc = getattr(result, "ring_center", None)
        rr = getattr(result, "ring_radius", None)
        if rc and rr:
            cx, cy = int(round(rc[0])), int(round(rc[1]))
            r = int(round(rr))
            contour = getattr(result, "ring_contour", None)
            if contour is not None and len(contour) >= 5:
                cv2.drawContours(out, [contour.astype(np.int32)], -1, (0, 0, 255), 2)
            else:
                cv2.circle(out, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)
            label = f"R D={rr * 2.0:.0f}px"
            if has_cal:
                label += f" ({rr * 2.0 * mm_per_px:.2f}mm)"
            cv2.putText(
                out,
                label,
                (cx + 12, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # -- pupil ---------------------------------------------------
    if (
        getattr(result, "pupil", None)
        and getattr(result.pupil, "detected", False)
        and getattr(result.pupil, "ellipse", None) is not None
    ):
        e = result.pupil.ellipse
        ct = (int(round(e.center_x)), int(round(e.center_y)))
        r = int(round(e.radius))
        cv2.circle(out, ct, r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, ct, 4, (0, 255, 0), -1)

        dia_px = e.radius * 2.0
        label = f"P D={dia_px:.0f}px"
        if has_cal:
            label += f" ({dia_px * mm_per_px:.2f}mm)"
        cv2.putText(
            out,
            label,
            (ct[0] + 12, ct[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # -- limbus --------------------------------------------------
    if (
        getattr(result, "limbus", None)
        and getattr(result.limbus, "detected", False)
        and getattr(result.limbus, "ellipse", None) is not None
    ):
        e = result.limbus.ellipse
        ct = (int(round(e.center_x)), int(round(e.center_y)))
        r = int(round(e.radius))
        cv2.circle(out, ct, r, (255, 100, 0), 2, cv2.LINE_AA)
        cv2.circle(out, ct, 4, (255, 100, 0), -1)

        dia_px = e.radius * 2.0
        label = f"L D={dia_px:.0f}px"
        if has_cal:
            label += f" ({dia_px * mm_per_px:.2f}mm)"
        cv2.putText(
            out,
            label,
            (ct[0] + 12, ct[1] + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 100, 0),
            1,
            cv2.LINE_AA,
        )

    # -- offset line ---------------------------------------------
    if getattr(result, "has_both", False):
        pe = result.pupil.ellipse
        p_pt = (int(round(pe.center_x)), int(round(pe.center_y)))
        cc = getattr(result, "corneal_center", None)
        if cc is not None and getattr(cc, "valid", False) and getattr(cc, "center_px", None):
            ref_pt = (int(round(cc.center_px[0])), int(round(cc.center_px[1])))
        else:
            le = result.limbus.ellipse
            ref_pt = (int(round(le.center_x)), int(round(le.center_y)))
        cv2.line(out, p_pt, ref_pt, (0, 255, 255), 2, cv2.LINE_AA)

    # -- quality badge -------------------------------------------
    quality = "---"
    confidence = 0.0
    oq = getattr(result, "overall_quality", None)
    if oq is not None:
        quality = oq.value if hasattr(oq, "value") else str(oq)
    confidence = getattr(result, "overall_confidence", 0.0)

    color_map = {
        "SURGICAL": (0, 204, 0),
        "CLINICAL": (255, 204, 0),
        "RESEARCH": (0, 165, 255),
        "INSUFFICIENT": (0, 0, 255),
        "NO_DETECTION": (128, 128, 128),
        "EXCELLENT": (0, 204, 0),
        "GOOD": (255, 204, 0),
        "FAIR": (0, 165, 255),
        "POOR": (0, 0, 255),
        "UNUSABLE": (128, 128, 128),
    }
    badge_color = color_map.get(quality, (128, 128, 128))

    cv2.putText(
        out,
        f"{quality} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        badge_color,
        2,
    )

    if ring_status == "ring_present":
        cv2.putText(
            out,
            "RING DETECTED",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 200, 0),
            2,
        )

    # -- grayscale mode indicator --------------------------------
    if grayscale_mode and grayscale_mode.upper() != "OFF":
        gs_colors = {
            "AUTO": (255, 255, 0),
            "FORCE": (0, 255, 255),
        }
        gs_color = gs_colors.get(grayscale_mode.upper(), (200, 200, 200))
        gs_label = f"GRAY: {grayscale_mode.upper()}"
        cv2.putText(
            out,
            gs_label,
            (w - 180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gs_color,
            2,
            cv2.LINE_AA,
        )

    # -- processing time -----------------------------------------
    meta = getattr(result, "metadata", None)
    if meta is not None:
        proc_ms = getattr(meta, "processing_time_ms", 0)
        cv2.putText(
            out,
            f"{proc_ms:.0f}ms",
            (w - 110, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )

    return out


def _draw_cli_overlay_from_dict(
    image: np.ndarray,
    result_dict: Dict[str, Any],
    show_ring: bool = True,
    grayscale_mode: str = "",
) -> np.ndarray:
    """Draw overlay from a serialised result dict."""
    out = image.copy()
    h, w = out.shape[:2]

    ring_status = result_dict.get("ring_status", "unknown")
    if show_ring and ring_status == "ring_present":
        rx = result_dict.get("ring_center_x")
        ry = result_dict.get("ring_center_y")
        rr = result_dict.get("ring_radius")
        if rx is not None and ry is not None and rr is not None:
            cx, cy = int(round(rx)), int(round(ry))
            r = int(round(rr))
            cv2.circle(out, (cx, cy), r, (0, 0, 255), 2, cv2.LINE_AA)

    pupil = result_dict.get("pupil", {})
    pe = pupil.get("ellipse", {})
    if pupil.get("detected") and pe.get("center_x") is not None:
        ct = (int(round(pe["center_x"])), int(round(pe["center_y"])))
        r = int(round(pe.get("radius", 0)))
        if r > 0:
            cv2.circle(out, ct, r, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(out, ct, 4, (0, 255, 0), -1)

    limbus = result_dict.get("limbus", {})
    le = limbus.get("ellipse", {})
    if limbus.get("detected") and le.get("center_x") is not None:
        ct = (int(round(le["center_x"])), int(round(le["center_y"])))
        r = int(round(le.get("radius", 0)))
        if r > 0:
            cv2.circle(out, ct, r, (255, 100, 0), 2, cv2.LINE_AA)
            cv2.circle(out, ct, 4, (255, 100, 0), -1)

    quality = result_dict.get("overall_quality", "---")
    conf = result_dict.get("overall_confidence", 0.0)
    cv2.putText(
        out,
        f"{quality} ({conf:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 204, 0),
        2,
    )

    if ring_status == "ring_present":
        cv2.putText(
            out,
            "RING DETECTED",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 200, 0),
            2,
        )

    # -- grayscale mode indicator --------------------------------
    if grayscale_mode and grayscale_mode.upper() != "OFF":
        gs_colors = {
            "AUTO": (255, 255, 0),
            "FORCE": (0, 255, 255),
        }
        gs_color = gs_colors.get(grayscale_mode.upper(), (200, 200, 200))
        gs_label = f"GRAY: {grayscale_mode.upper()}"
        cv2.putText(
            out,
            gs_label,
            (w - 180, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            gs_color,
            2,
            cv2.LINE_AA,
        )

    return out


# ================================================================
# Utility Helpers
# ================================================================


def _find_model_path(detector: Any, cfg: Any) -> Optional[str]:
    """Locate the trained model checkpoint."""
    candidates: List[str] = []

    if isinstance(cfg, dict):
        cfg_path = cfg.get("model_path")
        if cfg_path:
            candidates.append(str(cfg_path))

    if detector is not None:
        eng = getattr(detector, "ml_engine", None)
        if eng is not None:
            for attr in ("model_path", "_model_path"):
                p = getattr(eng, attr, None)
                if p:
                    candidates.append(str(p))

    candidates += [
        "models/best_model.pth",
        "model/best_model.pth",
        "pupil_tracking/models/best_model.pth",
    ]

    for c in candidates:
        if Path(c).is_file():
            return str(Path(c).resolve())
    return None


def _frame_result_to_dict(fr: Any) -> Dict[str, Any]:
    """Convert an OptimizedVideoProcessor FrameResult to dict."""
    d: Dict[str, Any] = {
        "metadata": {
            "frame_number": fr.frame_number,
            "processing_time_ms": fr.processing_ms,
        },
        "overall_quality": (fr.quality.value if fr.quality else "NO_DETECTION"),
        "overall_confidence": fr.confidence,
    }

    if fr.pupil_center and fr.pupil_axes:
        semi_a = max(fr.pupil_axes) / 2.0
        semi_b = min(fr.pupil_axes) / 2.0
        d["pupil"] = {
            "detected": True,
            "confidence": fr.confidence,
            "fit_type": getattr(fr, "pupil_fit_type", None),
            "ellipse": {
                "center_x": fr.pupil_center[0],
                "center_y": fr.pupil_center[1],
                "radius": semi_a,
                "semi_major": semi_a,
                "semi_minor": semi_b,
            },
        }
    else:
        d["pupil"] = {"detected": False, "ellipse": {}}

    if fr.limbus_center and fr.limbus_axes:
        semi_a = max(fr.limbus_axes) / 2.0
        semi_b = min(fr.limbus_axes) / 2.0
        d["limbus"] = {
            "detected": True,
            "confidence": min(0.95, fr.confidence + 0.05),
            "fit_type": getattr(fr, "limbus_fit_type", None),
            "ellipse": {
                "center_x": fr.limbus_center[0],
                "center_y": fr.limbus_center[1],
                "radius": semi_a,
                "semi_major": semi_a,
                "semi_minor": semi_b,
            },
        }
    else:
        d["limbus"] = {"detected": False, "ellipse": {}}

    if fr.limbus_axes is not None:
        limbus_full_dia = max(fr.limbus_axes)
        if limbus_full_dia > 0:
            px_per_mm = limbus_full_dia / _CORNEAL_DIAMETER_MM
            d["calibration"] = {
                "calibrated": True,
                "mm_per_px": 1.0 / px_per_mm,
                "px_per_mm": px_per_mm,
            }
        else:
            d["calibration"] = {"calibrated": False}
    else:
        d["calibration"] = {"calibrated": False}

    if fr.pupil_center and fr.limbus_center:
        dx = fr.pupil_center[0] - fr.limbus_center[0]
        dy = fr.pupil_center[1] - fr.limbus_center[1]
        mag_px = math.hypot(dx, dy)
        cal = d["calibration"]
        d["corneal_center"] = {
            "offset_magnitude_px": mag_px,
            "offset_magnitude_mm": (
                mag_px * cal["mm_per_px"] if cal.get("calibrated") else None
            ),
            "offset_angle_deg": math.degrees(math.atan2(dy, dx)),
        }
    else:
        d["corneal_center"] = {}

    d["ring_status"] = getattr(fr, "ring_status", "unknown")
    d["ring_confidence"] = getattr(fr, "ring_confidence", 0.0)
    d["image_category"] = getattr(fr, "image_category", "unknown")
    d["corneal_reference_source"] = getattr(
        fr,
        "corneal_reference_source",
        "limbus",
    )
    if hasattr(fr, "ring_center") and fr.ring_center is not None:
        d["ring_center_x"], d["ring_center_y"] = fr.ring_center
    if hasattr(fr, "ring_radius") and fr.ring_radius is not None:
        d["ring_radius"] = fr.ring_radius
    d["ring_dot_count"] = getattr(fr, "ring_dot_count", 0)

    return d


def _export_results_csv(results: List[Dict[str, Any]], csv_path: str) -> None:
    """Export results list to CSV."""
    if not results:
        return

    rows: List[Dict[str, Any]] = []
    prev_pupil_px = None
    prev_limbus_px = None
    for r in results:
        pupil = r.get("pupil", {})
        limbus = r.get("limbus", {})
        pe = pupil.get("ellipse", {})
        le = limbus.get("ellipse", {})
        cc = r.get("corneal_center", {})
        meta = r.get("metadata", {})
        cal_info = r.get("calibration", {})
        mm_px = cal_info.get("mm_per_px", 0)

        pupil_dia_px = float(pe.get("semi_major", pe.get("radius", 0))) * 2.0 if pe.get("semi_major") is not None or pe.get("radius") else ""
        limbus_dia_px = float(le.get("semi_major", le.get("radius", 0))) * 2.0 if le.get("semi_major") is not None or le.get("radius") else ""
        pupil_dia_mm = pupil_dia_px * mm_px if isinstance(pupil_dia_px, (int, float)) and mm_px else ""
        limbus_dia_mm = limbus_dia_px * mm_px if isinstance(limbus_dia_px, (int, float)) and mm_px else ""
        # compute deltas vs previous frame (px)
        pupil_delta = ""
        limbus_delta = ""
        try:
            if prev_pupil_px is not None and isinstance(pupil_dia_px, (int, float)):
                pupil_delta = pupil_dia_px - prev_pupil_px
        except Exception:
            pupil_delta = ""
        try:
            if prev_limbus_px is not None and isinstance(limbus_dia_px, (int, float)):
                limbus_delta = limbus_dia_px - prev_limbus_px
        except Exception:
            limbus_delta = ""

        rows.append(
            {
                "frame": meta.get("frame_number", ""),
                "ring_status": r.get("ring_status", ""),
                "pupil_detected": pupil.get("detected", False),
                "pupil_cx_px": pe.get("center_x", ""),
                "pupil_cy_px": pe.get("center_y", ""),
                "pupil_diameter_px": pupil_dia_px,
                "pupil_diameter_mm": pupil_dia_mm,
                "pupil_diameter_delta_px": pupil_delta,
                "pupil_fit_type": pupil.get("fit_type", ""),
                "pupil_confidence": pupil.get("confidence", ""),
                "limbus_detected": limbus.get("detected", False),
                "limbus_cx_px": le.get("center_x", ""),
                "limbus_cy_px": le.get("center_y", ""),
                "limbus_diameter_px": limbus_dia_px,
                "limbus_diameter_mm": limbus_dia_mm,
                "limbus_diameter_delta_px": limbus_delta,
                "limbus_fit_type": limbus.get("fit_type", ""),
                "limbus_confidence": limbus.get("confidence", ""),
                "offset_px": cc.get("offset_magnitude_px", ""),
                "offset_mm": cc.get("offset_magnitude_mm", ""),
                "offset_angle_deg": cc.get("offset_angle_deg", ""),
                "px_per_mm": cal_info.get("px_per_mm", ""),
                "quality": r.get("overall_quality", ""),
                "grayscale_mode": r.get("grayscale_mode", ""),
                "grayscale_applied": r.get("grayscale_applied", ""),
            }
        )

        # update previous frame values
        try:
            prev_pupil_px = float(pupil_dia_px) if isinstance(pupil_dia_px, (int, float)) else None
        except Exception:
            prev_pupil_px = None
        try:
            prev_limbus_px = float(limbus_dia_px) if isinstance(limbus_dia_px, (int, float)) else None
        except Exception:
            prev_limbus_px = None

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        # Append summary statistics: average pupil/limbus diameter pre/post dock
        try:
            # compute averages grouped by ring_status
            pre_pupil = []
            post_pupil = []
            pre_limbus = []
            post_limbus = []
            for r in rows:
                try:
                    rs = r.get("ring_status", "")
                    pd = r.get("pupil_diameter_mm")
                    ld = r.get("limbus_diameter_mm")
                    if isinstance(pd, (int, float)) and pd > 0:
                        if rs == "ring_present" or rs == "PRESENT":
                            post_pupil.append(pd)
                        else:
                            pre_pupil.append(pd)
                    if isinstance(ld, (int, float)) and ld > 0:
                        if rs == "ring_present" or rs == "PRESENT":
                            post_limbus.append(ld)
                        else:
                            pre_limbus.append(ld)
                except Exception:
                    continue

            def mean_or_empty(arr):
                return float(sum(arr) / len(arr)) if arr else ""

            summary = {
                "frame": "SUMMARY",
                "ring_status": "",
                "pupil_diameter_mm": mean_or_empty(pre_pupil),
                "limbus_diameter_mm": mean_or_empty(pre_limbus),
                "pupil_diameter_delta_px": "(pre_avg)",
            }
            writer.writerow({k: summary.get(k, "") for k in rows[0].keys()})
            summary2 = {
                "frame": "SUMMARY",
                "ring_status": "docked_avg",
                "pupil_diameter_mm": mean_or_empty(post_pupil),
                "limbus_diameter_mm": mean_or_empty(post_limbus),
                "pupil_diameter_delta_px": "(docked_avg)",
            }
            writer.writerow({k: summary2.get(k, "") for k in rows[0].keys()})
            # write diff row
            try:
                diff_pupil = ""
                if isinstance(summary2.get("pupil_diameter_mm"), float) and isinstance(summary.get("pupil_diameter_mm"), float):
                    diff_pupil = summary2.get("pupil_diameter_mm") - summary.get("pupil_diameter_mm")
                diff_row = {k: "" for k in rows[0].keys()}
                diff_row["frame"] = "SUMMARY"
                diff_row["ring_status"] = "docked_minus_pre"
                diff_row["pupil_diameter_mm"] = diff_pupil
                writer.writerow(diff_row)
            except Exception:
                # fail silently on diff row
                pass
        except Exception:
            # never fail CSV export for summary
            pass


# ================================================================
# Argument Parser
# ================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launch_gui.py",
        description=(
            "\n"
            "==============================================================\n"
            "Medevplus IXcentai surgical grade\n"
            "Deep learning pupil/iris detection & tracking\n"
            "==============================================================\n"
        ),
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=42, width=100
        ),
        epilog=(
            "\n"
            "----------------------- EXAMPLES -----------------------\n"
            "\n"
            "  Launch GUI (default):\n"
            "    python launch_gui.py\n"
            "    python launch_gui.py gui\n"
            "\n"
            "  Process single image:\n"
            "    python launch_gui.py image -i eye_photo.jpg\n"
            "    python launch_gui.py image -i eye.jpg --grayscale force\n"
            "\n"
            "  Process video file:\n"
            "    python launch_gui.py video -i recording.mp4\n"
            "    python launch_gui.py video -i rec.mp4 -o output.mp4\n"
            "    python launch_gui.py video -i rec.mp4 --grayscale force\n"
            "\n"
            "  Live camera:\n"
            "    python launch_gui.py camera\n"
            "    python launch_gui.py camera --grayscale force\n"
            "    python launch_gui.py camera  # press [G] to toggle\n"
            "\n"
            "---------------------------------------------------------\n"
        ),
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="gui",
        choices=["gui", "image", "video", "camera"],
        help="Processing mode: gui / image / video / camera",
    )

    io_group = parser.add_argument_group("INPUT / OUTPUT")
    io_group.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        metavar="PATH",
        help="Input file path (image or video)",
    )
    io_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Output file path",
    )
    io_group.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to model weights (.pth)",
    )
    io_group.add_argument(
        "--camera-id",
        type=int,
        default=0,
        metavar="ID",
        help="Camera device index (default: 0)",
    )
    io_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto)",
    )

    ring_group = parser.add_argument_group("RING DETECTION")
    ring_group.add_argument(
        "--ring-mode",
        type=str,
        default="auto",
        choices=["auto", "docked", "pre_docked"],
        help="Ring detection mode (default: auto)",
    )
    ring_group.add_argument(
        "--ring-classifier",
        type=str,
        default="models/ring_classifier.pth",
        metavar="PATH",
        help="Path to ring classifier model",
    )
    ring_group.add_argument(
        "--show-ring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw ring outline in output",
    )

    gray_group = parser.add_argument_group("GRAYSCALE MODE")
    gray_group.add_argument(
        "--grayscale",
        type=str,
        default="off",
        choices=["off", "auto", "force"],
        metavar="MODE",
        help=(
            "Grayscale processing mode.\n"
            "  off   = RGB passthrough (default)\n"
            "  auto  = detect grayscale -> enhance -> process\n"
            "  force = always convert to grayscale (IR look)\n"
            "In camera mode, press [G] to cycle modes."
        ),
    )

    pipe_group = parser.add_argument_group("PIPELINE")
    pipe_group.add_argument(
        "--optimized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use optimised pipeline",
    )
    pipe_group.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=_RUNTIME_PROFILE.recommended_fp16,
        help="FP16 half-precision",
    )
    pipe_group.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=_RUNTIME_PROFILE.recommended_compile,
        help="torch.compile JIT",
    )

    vid_group = parser.add_argument_group("VIDEO")
    vid_group.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1)",
    )
    vid_group.add_argument(
        "--resolution",
        type=int,
        default=_RUNTIME_PROFILE.recommended_resolution,
        metavar="PX",
        help="Inference resolution (default: machine profile)",
    )
    vid_group.add_argument(
        "--target-fps",
        type=float,
        default=_RUNTIME_PROFILE.recommended_target_fps,
        metavar="FPS",
        help="Target processing FPS (default: machine profile)",
    )

    roi_group = parser.add_argument_group("ROI & TRACKING")
    roi_group.add_argument(
        "--roi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ROI tracking",
    )
    roi_group.add_argument(
        "--roi-cache",
        type=int,
        default=5,
        metavar="N",
        help="ROI cache lifetime in frames (default: 5)",
    )
    roi_group.add_argument(
        "--kalman-process-noise",
        type=float,
        default=0.03,
        metavar="F",
        help="Kalman process noise (default: 0.03)",
    )
    roi_group.add_argument(
        "--kalman-measure-noise",
        type=float,
        default=0.1,
        metavar="F",
        help="Kalman measurement noise (default: 0.1)",
    )

    # Calibration group
    cal_group = parser.add_argument_group("Calibration Options")
    cal_group.add_argument(
        "--calibration-mode",
        type=str,
        default="FIXED_PIXEL_SCALE",
        choices=["ANATOMICAL_ANCHOR", "FIXED_PIXEL_SCALE", "RING_REFLECTION"],
        help="Calibration mode: ANATOMICAL_ANCHOR, FIXED_PIXEL_SCALE, RING_REFLECTION (default: FIXED_PIXEL_SCALE)",
    )
    cal_group.add_argument(
        "--px-per-mm",
        type=float,
        default=58.2,
        metavar="F",
        help="Fixed manual scale in px/mm for FIXED_PIXEL_SCALE mode (default: 58.2)",
    )
    cal_group.add_argument(
        "--corneal-diameter-mm",
        type=float,
        default=12.0,
        metavar="F",
        help="Assumed horizontal corneal diameter in mm (default: 12.0)",
    )
    cal_group.add_argument(
        "--ring-diameter-mm",
        type=float,
        default=9.4,
        metavar="F",
        help="Known reference suction ring diameter in mm (default: 9.4)",
    )

    return parser



# ================================================================
# Main Entry Point
# ================================================================


# ================================================================
# Startup Self Check
# ================================================================


def _run_startup_self_check(args: argparse.Namespace, cfg: Any) -> int:
    warnings: List[str] = []
    errors: List[str] = []

    log_dir = Path(cfg.paths.log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        errors.append(f"Cannot create log directory: {log_dir} ({exc})")

    if getattr(args, "input", None):
        input_path = Path(args.input)
        if not input_path.exists():
            errors.append(f"Input not found: {input_path}")
        elif args.mode in {"image", "video"} and not input_path.is_file():
            errors.append(f"Input is not a file: {input_path}")

    optimized_requested = bool(getattr(args, "optimized", True))
    if optimized_requested:
        if not _FAST_PIPELINE_AVAILABLE:
            warnings.append("Optimized pipeline unavailable - classic pipeline will be used.")
        else:
            model_candidates = []
            if getattr(args, "model", None):
                model_candidates.append(Path(args.model))
            model_candidates.extend(
                [
                    Path("models/best_model.pth"),
                    Path("models/segmentation_quantized.onnx"),
                ]
            )
            if not any(candidate.is_file() for candidate in model_candidates):
                warnings.append(
                    "No optimized model artifact found in expected locations - optimized mode may fall back."
                )

    for msg in warnings:
        print(f"  [WARN] {msg}")
    for msg in errors:
        print(f"  [ERROR] {msg}")
    return 1 if errors else 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(_BANNER)
    print(
        f"Runtime profile: {_RUNTIME_PROFILE.name} "
        f"({int(round(_RUNTIME_PROFILE.ram_gb))} GB RAM, "
        f"{_RUNTIME_PROFILE.physical_cores} cores)"
    )

    cfg = get_config()
    logger = AuditLogger(log_dir=cfg.paths.log_dir)
    set_logger(logger)

    startup_status = _run_startup_self_check(args, cfg)
    if startup_status != 0:
        logger.close()
        sys.exit(startup_status)

    if args.mode == "image":
        if not args.input:
            print("  ✗ ERROR: --input / -i required for image mode")
            sys.exit(1)
        if not Path(args.input).is_file():
            print(f"  ✗ ERROR: File not found: {args.input}\n")
            sys.exit(1)
        process_image(args.input, model_path=args.model, device=args.device, args=args)

    elif args.mode == "video":
        if not args.input:
            print("  ✗ ERROR: --input / -i required for video mode")
            sys.exit(1)
        if not Path(args.input).is_file():
            print(f"  ✗ ERROR: File not found: {args.input}\n")
            sys.exit(1)
        process_video(
            args.input, output_path=args.output, model_path=args.model, args=args
        )

    elif args.mode == "camera":
        process_camera(camera_id=args.camera_id, model_path=args.model, args=args)

    elif args.mode == "gui":
        from pupil_tracking.interface.gui_app import PupilTrackingGUI, launch_gui

        def _launch_with_overrides() -> None:
            import tkinter as _tk
            from pupil_tracking.interface.theme import DarkTheme

            root = _tk.Tk()
            try:
                import os
                import sys
                base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
                icon_path = os.path.join(base_dir, "build", "assets", "icon.ico")
                if os.path.exists(icon_path):
                    root.iconbitmap(icon_path)
            except Exception:
                pass
                
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass
            colors = DarkTheme.apply(root)
            app = PupilTrackingGUI(root, colors=colors)
            app._use_optimized_var.set(args.optimized)
            app._fp16_var.set(args.fp16)
            app._compile_var.set(getattr(args, "compile", True))
            app._resolution_var.set(args.resolution)
            app._stride_var.set(args.stride)
            app._target_fps_var.set(args.target_fps)
            app._roi_var.set(args.roi)
            app._roi_cache_var.set(args.roi_cache)
            app._kalman_process_var.set(args.kalman_process_noise)
            app._kalman_measure_var.set(args.kalman_measure_noise)
            if hasattr(app, "_res_display"):
                app._res_display.set(str(args.resolution))
            if hasattr(app, "_kp_display"):
                app._kp_display.set(f"{args.kalman_process_noise:.3f}")
            if hasattr(app, "_calibration_mode_var"):
                app._calibration_mode_var.set(args.calibration_mode)
            if hasattr(app, "_fixed_scale_var"):
                app._fixed_scale_var.set(args.px_per_mm)
            if hasattr(app, "_corneal_ref_mm_var"):
                app._corneal_ref_mm_var.set(args.corneal_diameter_mm)
            if hasattr(app, "_ring_ref_mm_var"):
                app._ring_ref_mm_var.set(args.ring_diameter_mm)
            if hasattr(app, "_grayscale_mode_var"):
                app._grayscale_mode_var.set(args.grayscale)
            app._invalidate_fast_engine()
            print("  ✓ CLI overrides applied to GUI Settings tab\n")
            root.mainloop()

        _defaults = _build_parser().parse_args([])
        _any_override = any(
            getattr(args, k) != getattr(_defaults, k)
            for k in [
                "optimized",
                "fp16",
                "compile",
                "resolution",
                "stride",
                "target_fps",
                "roi",
                "roi_cache",
                "kalman_process_noise",
                "kalman_measure_noise",
                "grayscale",
                "calibration_mode",
                "px_per_mm",
                "corneal_diameter_mm",
                "ring_diameter_mm",
            ]
        )

        if _any_override:
            print("  Launching GUI with CLI overrides…\n")
            _launch_with_overrides()
        else:
            print("  Launching GUI…\n")
            launch_gui()

    logger.close()


if __name__ == "__main__":
    main()
