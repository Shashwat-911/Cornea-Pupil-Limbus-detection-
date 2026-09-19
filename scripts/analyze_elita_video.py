"""Phase XX-A: ELITA video baseline analysis.

Runs the existing pupil/limbus/iris pipeline on manager-provided ELITA videos
and produces a quantitative baseline report. No algorithmic changes.

Usage:
    python scripts/analyze_elita_video.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import EllipseParams


# ── Data classes for results ──────────────────────────────────────────


@dataclass
class FrameResult:
    frame_index: int
    timestamp_s: float
    pupil_detected: bool = False
    pupil_confidence: float = 0.0
    pupil_center: Optional[Tuple[float, float]] = None
    pupil_major: float = 0.0
    pupil_minor: float = 0.0
    limbus_detected: bool = False
    limbus_confidence: float = 0.0
    limbus_center: Optional[Tuple[float, float]] = None
    limbus_major: float = 0.0
    limbus_minor: float = 0.0
    has_both: bool = False
    iris_status: str = "NOT_RUN"
    iris_features: int = 0
    iris_coverage: float = 0.0
    iris_spatial_spread: Optional[float] = None
    pupil_ms: float = 0.0
    limbus_ms: float = 0.0
    iris_ms: float = 0.0
    total_ms: float = 0.0
    failure_reason: str = ""


@dataclass
class VideoInventory:
    filename: str
    filepath: str
    resolution: Tuple[int, int] = (0, 0)
    fps: float = 0.0
    frame_count: int = 0
    duration_s: float = 0.0
    codec: str = ""
    file_size_mb: float = 0.0


@dataclass
class VideoAnalysis:
    inventory: VideoInventory
    frame_results: List[FrameResult] = field(default_factory=list)
    total_frames_sampled: int = 0
    pupil_success_count: int = 0
    limbus_success_count: int = 0
    iris_success_count: int = 0
    iris_skip_count: int = 0
    iris_no_roi_count: int = 0


# ── Video inventory ───────────────────────────────────────────────────


def inventory_video(filepath: str) -> VideoInventory:
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {filepath}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    dur = fc / fps if fps > 0 else 0
    sz_mb = os.path.getsize(filepath) / 1024 / 1024

    cap.release()

    return VideoInventory(
        filename=os.path.basename(filepath),
        filepath=filepath,
        resolution=(w, h),
        fps=fps,
        frame_count=fc,
        duration_s=dur,
        codec=codec,
        file_size_mb=sz_mb,
    )


# ── Frame sampling strategy ───────────────────────────────────────────


def select_sample_frames(total_frames: int, max_samples: int = 20) -> List[int]:
    """Select frames spread across the video for baseline analysis."""
    if total_frames <= max_samples:
        return list(range(total_frames))

    indices = set()
    # Always include first and last
    indices.add(0)
    indices.add(total_frames - 1)

    # Evenly spread remaining samples
    remaining = max_samples - 2
    step = (total_frames - 1) / (remaining + 1)
    for i in range(1, remaining + 1):
        idx = int(step * i)
        indices.add(min(idx, total_frames - 1))

    # Add some near-boundary samples for transitions
    for offset in [1, 2, total_frames - 2, total_frames - 3]:
        if 0 <= offset < total_frames:
            indices.add(offset)

    return sorted(indices)


# ── Pipeline runner ───────────────────────────────────────────────────


def run_pipeline_on_frame(
    frame: np.ndarray,
    iris_detector: IrisFeatureDetector,
    detector=None,
    frame_index: int = 0,
    timestamp_s: float = 0.0,
) -> FrameResult:
    """Run the full pupil -> limbus -> iris pipeline on a single frame."""
    result = FrameResult(frame_index=frame_index, timestamp_s=timestamp_s)

    # ── Pupil + Limbus detection ──
    t0 = time.perf_counter()
    try:
        det_result = detector.detect(frame, frame_number=frame_index)
    except Exception as exc:
        result.failure_reason = f"detect_error: {exc}"
        result.total_ms = (time.perf_counter() - t0) * 1000
        return result
    t1 = time.perf_counter()

    result.pupil_ms = (t1 - t0) * 1000

    # Extract pupil info
    if det_result.pupil.detected and det_result.pupil.ellipse is not None:
        result.pupil_detected = True
        result.pupil_confidence = det_result.pupil.confidence
        e = det_result.pupil.ellipse
        result.pupil_center = (e.center_x, e.center_y)
        result.pupil_major = e.semi_major
        result.pupil_minor = e.semi_minor

    # Extract limbus info
    if det_result.limbus.detected and det_result.limbus.ellipse is not None:
        result.limbus_detected = True
        result.limbus_confidence = det_result.limbus.confidence
        e = det_result.limbus.ellipse
        result.limbus_center = (e.center_x, e.center_y)
        result.limbus_major = e.semi_major
        result.limbus_minor = e.semi_minor

    result.has_both = det_result.has_both

    # ── Iris detection ──
    if (
        result.has_both
        and det_result.pupil.ellipse is not None
        and det_result.limbus.ellipse is not None
    ):
        t2 = time.perf_counter()
        try:
            iris_result = iris_detector.detect(
                frame, det_result.pupil.ellipse, det_result.limbus.ellipse
            )
            t3 = time.perf_counter()
            result.iris_ms = (t3 - t2) * 1000
            result.iris_status = iris_result.status.name
            result.iris_features = len(iris_result.feature_set.features)
            result.iris_coverage = iris_result.feature_set.region_coverage

            # Compute spatial spread of iris features
            if iris_result.feature_set.features:
                feats = iris_result.feature_set.features
                xs = [f.x for f in feats]
                ys = [f.y for f in feats]
                cx = np.mean(xs)
                cy = np.mean(ys)
                dists = np.sqrt((np.array(xs) - cx) ** 2 + (np.array(ys) - cy) ** 2)
                result.iris_spatial_spread = float(np.std(dists))
        except Exception as exc:
            t3 = time.perf_counter()
            result.iris_ms = (t3 - t2) * 1000
            result.iris_status = f"ERROR: {exc}"
            result.failure_reason = f"iris_error: {exc}"
    elif not result.has_both:
        result.iris_status = "SKIPPED_NO_LIMBUS" if result.pupil_detected else "SKIPPED_NO_PUPIL"
        result.iris_skip_count = 1

    result.total_ms = (time.perf_counter() - t0) * 1000
    return result


# ── Main analysis ─────────────────────────────────────────────────────


def analyze_video(
    video_path: str,
    max_samples: int = 20,
) -> VideoAnalysis:
    """Full pipeline analysis on one video."""
    print(f"\n{'='*70}")
    print(f"  Analyzing: {os.path.basename(video_path)}")
    print(f"{'='*70}")

    # Inventory
    inv = inventory_video(video_path)
    print(f"  Resolution: {inv.resolution[0]}x{inv.resolution[1]}")
    print(f"  FPS: {inv.fps:.2f}  Frames: {inv.frame_count}  Duration: {inv.duration_s:.1f}s")
    print(f"  Codec: {inv.codec}  Size: {inv.file_size_mb:.1f} MB")

    analysis = VideoAnalysis(inventory=inv)

    # Select sample frames
    sample_indices = select_sample_frames(inv.frame_count, max_samples)
    print(f"  Sampling {len(sample_indices)} frames: {sample_indices[:5]}...{sample_indices[-3:]}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video")
        return analysis

    # Init detectors
    print("  Initializing detectors...")
    cfg = get_config()
    iris_detector = IrisFeatureDetector()

    # Init the main detector (UnifiedDetector)
    try:
        from pupil_tracking.core.detector import UnifiedDetector
        main_detector = UnifiedDetector(config=cfg)
        print(f"  Main detector ready (ML: {'available' if main_detector.ml_engine.available else 'unavailable'})")
    except Exception as exc:
        print(f"  ERROR init main detector: {exc}")
        cap.release()
        return analysis

    # Process sampled frames
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {frame_idx}: READ FAILED")
            continue

        timestamp_s = frame_idx / inv.fps if inv.fps > 0 else 0

        fr = run_pipeline_on_frame(
            frame, iris_detector, main_detector,
            frame_index=frame_idx, timestamp_s=timestamp_s,
        )
        analysis.frame_results.append(fr)

        # Accumulate stats
        analysis.total_frames_sampled += 1
        if fr.pupil_detected:
            analysis.pupil_success_count += 1
        if fr.limbus_detected:
            analysis.limbus_success_count += 1
        if fr.iris_status == "OK":
            analysis.iris_success_count += 1
        if "SKIPPED" in fr.iris_status:
            analysis.iris_skip_count += 1

        status_str = (
            f"P={'OK' if fr.pupil_detected else 'FAIL'} "
            f"L={'OK' if fr.limbus_detected else 'FAIL'} "
            f"I={fr.iris_status}"
            f"({'N/A' if fr.iris_features == 0 else fr.iris_features} feat, "
            f"{fr.iris_coverage*100:.2f}%)" if fr.iris_features > 0 or "OK" in fr.iris_status
            else f"I={fr.iris_status}"
        )
        print(
            f"  [{i+1:2d}/{len(sample_indices)}] Frame {frame_idx:5d} "
            f"({timestamp_s:6.1f}s) | {status_str} | {fr.total_ms:.0f}ms"
        )

    cap.release()

    # Summary
    n = max(analysis.total_frames_sampled, 1)
    print(f"\n  --- Summary for {inv.filename} ---")
    print(f"  Frames sampled: {analysis.total_frames_sampled}")
    print(f"  Pupil success:  {analysis.pupil_success_count}/{n} ({100*analysis.pupil_success_count/n:.0f}%)")
    print(f"  Limbus success: {analysis.limbus_success_count}/{n} ({100*analysis.limbus_success_count/n:.0f}%)")
    print(f"  Iris success:   {analysis.iris_success_count}/{n} ({100*analysis.iris_success_count/n:.0f}%)")
    print(f"  Iris skipped:   {analysis.iris_skip_count}/{n} ({100*analysis.iris_skip_count/n:.0f}%)")

    return analysis


# ── Report generation ─────────────────────────────────────────────────


def generate_report(analyses: List[VideoAnalysis], output_dir: str) -> str:
    """Generate the PHASE_XX_A markdown report."""
    lines = [
        "# Phase XX-A: Real ELITA Video Baseline Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Dataset Inventory",
        "",
        "| Property | " + " | ".join(a.inventory.filename for a in analyses) + " |",
        "|---|" + "|".join(["---"] * len(analyses)) + "|",
    ]

    props = [
        ("Resolution", lambda i: f"{i.resolution[0]}x{i.resolution[1]}"),
        ("FPS", lambda i: f"{i.fps:.2f}"),
        ("Frame Count", lambda i: f"{i.frame_count:,}"),
        ("Duration", lambda i: f"{i.duration_s:.1f}s ({i.duration_s/60:.1f} min)"),
        ("Codec", lambda i: i.codec),
        ("File Size", lambda i: f"{i.file_size_mb:.1f} MB"),
    ]

    for label, fn in props:
        row = f"| {label} | " + " | ".join(fn(a.inventory) for a in analyses) + " |"
        lines.append(row)

    lines.extend([
        "",
        "> **Note:** No assumptions are made about laterality, pairing, or clinical meaning.",
        "> These are engineering observations only.",
        "",
        "---",
        "",
        "## 2. Methodology",
        "",
        "- Frames sampled strategically across each video (beginning, early-middle, middle, late-middle, end, plus boundary samples)",
        "- Each sampled frame processed through the full pipeline: `UnifiedDetector.detect()` -> `IrisFeatureDetector.detect()`",
        "- Timing measured per-stage with `time.perf_counter()`",
        "- No Kalman smoothing applied (raw detection output)",
        "- Iris detection skipped when `has_both` is False (requires both pupil and limbus ellipses)",
        "",
        "---",
        "",
        "## 3. Per-Video Results",
        "",
    ])

    for a in analyses:
        inv = a.inventory
        n = max(a.total_frames_sampled, 1)
        lines.extend([
            f"### {inv.filename}",
            "",
            f"- Frames sampled: {a.total_frames_sampled}",
            f"- Pupil success: {a.pupil_success_count}/{n} ({100*a.pupil_success_count/n:.0f}%)",
            f"- Limbus success: {a.limbus_success_count}/{n} ({100*a.limbus_success_count/n:.0f}%)",
            f"- Iris success (status=OK): {a.iris_success_count}/{n} ({100*a.iris_success_count/n:.0f}%)",
            f"- Iris skipped (missing prerequisite): {a.iris_skip_count}/{n} ({100*a.iris_skip_count/n:.0f}%)",
            "",
        ])

        # Per-frame details
        if a.frame_results:
            lines.extend([
                "#### Frame-by-Frame Detail",
                "",
                "| Frame | Time | Pupil | Limbus | Iris Status | Features | Coverage | Time (ms) |",
                "|-------|------|-------|--------|-------------|----------|----------|-----------|",
            ])
            for fr in a.frame_results:
                p = "OK" if fr.pupil_detected else "FAIL"
                l = "OK" if fr.limbus_detected else "FAIL"
                feat = str(fr.iris_features) if fr.iris_features > 0 else "-"
                cov = f"{fr.iris_coverage*100:.2f}%" if fr.iris_coverage > 0 else "-"
                lines.append(
                    f"| {fr.frame_index} | {fr.timestamp_s:.1f}s | {p} | {l} "
                    f"| {fr.iris_status} | {feat} | {cov} | {fr.total_ms:.0f} |"
                )
            lines.append("")

    # Aggregate stats
    total_sampled = sum(a.total_frames_sampled for a in analyses)
    total_pupil = sum(a.pupil_success_count for a in analyses)
    total_limbus = sum(a.limbus_success_count for a in analyses)
    total_iris = sum(a.iris_success_count for a in analyses)
    total_skip = sum(a.iris_skip_count for a in analyses)

    lines.extend([
        "---",
        "",
        "## 4. Aggregate Detection Rates",
        "",
        f"- **Total frames analyzed:** {total_sampled}",
        f"- **Pupil success:** {total_pupil}/{total_sampled} ({100*total_pupil/max(total_sampled,1):.0f}%)",
        f"- **Limbus success:** {total_limbus}/{total_sampled} ({100*total_limbus/max(total_sampled,1):.0f}%)",
        f"- **Iris success:** {total_iris}/{total_sampled} ({100*total_iris/max(total_sampled,1):.0f}%)",
        f"- **Iris skipped:** {total_skip}/{total_sampled} ({100*total_skip/max(total_sampled,1):.0f}%)",
        "",
    ])

    # Iris feature statistics
    all_iris_frames = []
    for a in analyses:
        for fr in a.frame_results:
            if fr.iris_features > 0:
                all_iris_frames.append(fr)

    lines.extend([
        "---",
        "",
        "## 5. Iris Feature Statistics",
        "",
    ])

    if all_iris_frames:
        feat_counts = [fr.iris_features for fr in all_iris_frames]
        coverages = [fr.iris_coverage for fr in all_iris_frames]
        spreads = [fr.iris_spatial_spread for fr in all_iris_frames if fr.iris_spatial_spread is not None]
        iris_times = [fr.iris_ms for fr in all_iris_frames]

        lines.extend([
            f"- **Frames with iris features:** {len(all_iris_frames)}/{total_sampled}",
            f"- **Feature count range:** {min(feat_counts)} - {max(feat_counts)}",
            f"- **Feature count mean:** {np.mean(feat_counts):.1f}",
            f"- **Feature count median:** {np.median(feat_counts):.1f}",
            f"- **Coverage range:** {min(coverages)*100:.3f}% - {max(coverages)*100:.3f}%",
            f"- **Coverage mean:** {np.mean(coverages)*100:.3f}%",
        ])
        if spreads:
            lines.extend([
                f"- **Spatial spread (std of dist from centroid):** {np.mean(spreads):.1f} px (mean), {np.min(spreads):.1f} - {np.max(spreads):.1f} px (range)",
            ])
        lines.extend([
            f"- **Iris detection time:** {np.mean(iris_times):.0f} ms (mean), {np.min(iris_times):.0f} - {np.max(iris_times):.0f} ms (range)",
        ])
    else:
        lines.append("**No frames produced valid iris features.**")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Failure Breakdown",
        "",
    ])

    # Classify failures
    pupil_fail = 0
    limbus_fail = 0
    iris_error = 0
    iris_no_roi = 0
    iris_ok = 0
    iris_skip_pupil = 0
    iris_skip_limbus = 0
    other_fail = 0

    for a in analyses:
        for fr in a.frame_results:
            if not fr.pupil_detected:
                pupil_fail += 1
            elif not fr.limbus_detected:
                limbus_fail += 1
            elif fr.iris_status == "OK":
                iris_ok += 1
            elif "NO_ROI" in fr.iris_status:
                iris_no_roi += 1
            elif "ERROR" in fr.iris_status:
                iris_error += 1
            elif "SKIPPED_NO_PUPIL" in fr.iris_status:
                iris_skip_pupil += 1
            elif "SKIPPED_NO_LIMBUS" in fr.iris_status:
                iris_skip_limbus += 1
            else:
                other_fail += 1

    lines.extend([
        f"| Failure Type | Count | % of Total |",
        f"|---|---|---|",
        f"| Pupil detection failure | {pupil_fail} | {100*pupil_fail/max(total_sampled,1):.0f}% |",
        f"| Limbus detection failure (after pupil OK) | {limbus_fail} | {100*limbus_fail/max(total_sampled,1):.0f}% |",
        f"| Iris skipped (no pupil) | {iris_skip_pupil} | {100*iris_skip_pupil/max(total_sampled,1):.0f}% |",
        f"| Iris skipped (no limbus) | {iris_skip_limbus} | {100*iris_skip_limbus/max(total_sampled,1):.0f}% |",
        f"| Iris NO_ROI (pupil+limbus OK but no ROI) | {iris_no_roi} | {100*iris_no_roi/max(total_sampled,1):.0f}% |",
        f"| Iris detection error | {iris_error} | {100*iris_error/max(total_sampled,1):.0f}% |",
        f"| Iris OK | {iris_ok} | {100*iris_ok/max(total_sampled,1):.0f}% |",
        f"| Other/unclassified | {other_fail} | {100*other_fail/max(total_sampled,1):.0f}% |",
        "",
    ])

    # Temporal observations
    lines.extend([
        "---",
        "",
        "## 7. Temporal Observations",
        "",
    ])

    for a in analyses:
        if len(a.frame_results) < 3:
            continue
        lines.append(f"### {a.inventory.filename}")
        lines.append("")

        # Check feature count stability
        iris_frames = [fr for fr in a.frame_results if fr.iris_features > 0]
        if len(iris_frames) >= 2:
            feat_list = [fr.iris_features for fr in iris_frames]
            cov_list = [fr.iris_coverage for fr in iris_frames]
            lines.append(f"- Feature counts across {len(iris_frames)} iris-positive frames: {feat_list}")
            lines.append(f"- Coverage values: {[f'{c*100:.3f}%' for c in cov_list]}")

            # Simple stability check
            if len(feat_list) >= 2:
                cv = np.std(feat_list) / max(np.mean(feat_list), 1)
                lines.append(f"- Feature count coefficient of variation: {cv:.2f} ({'stable' if cv < 0.3 else 'moderate' if cv < 0.6 else 'unstable'})")
        else:
            lines.append(f"- Only {len(iris_frames)} frames with iris features; insufficient for temporal analysis")
        lines.append("")

    # Performance
    lines.extend([
        "---",
        "",
        "## 8. Performance Measurements",
        "",
    ])

    for a in analyses:
        if not a.frame_results:
            continue
        all_total = [fr.total_ms for fr in a.frame_results]
        all_pupil = [fr.pupil_ms for fr in a.frame_results]
        all_iris = [fr.iris_ms for fr in a.frame_results if fr.iris_ms > 0]

        lines.extend([
            f"### {a.inventory.filename}",
            "",
            f"- **Total pipeline time:** {np.mean(all_total):.0f} ms (mean), {np.min(all_total):.0f} - {np.max(all_total):.0f} ms (range)",
            f"- **Pupil+Limbus detection:** {np.mean(all_pupil):.0f} ms (mean), {np.min(all_pupil):.0f} - {np.max(all_pupil):.0f} ms (range)",
        ])
        if all_iris:
            lines.append(
                f"- **Iris detection (when run):** {np.mean(all_iris):.0f} ms (mean), {np.min(all_iris):.0f} - {np.max(all_iris):.0f} ms (range)"
            )
        else:
            lines.append("- **Iris detection:** never ran (no frames with both pupil+limbus)")
        lines.append("")

    # Root cause
    lines.extend([
        "---",
        "",
        "## 9. Root Cause of Empty/Weak Iris Detection",
        "",
    ])

    if iris_skip_limbus > 0:
        lines.extend([
            f"The dominant cause of empty iris detection is **limbus detection failure**.",
            f"Of {total_sampled} frames analyzed, {limbus_fail} failed limbus detection,",
            f"causing {iris_skip_limbus} iris detection skips.",
            "",
            "The `_detect_iris()` guard condition requires `result.has_both` (both pupil AND limbus",
            "detected with valid ellipses). When limbus fails, iris detection is skipped entirely.",
            "",
            f"If iris NO_ROI cases ({iris_no_roi}) also exist, these represent frames where pupil+limbus",
            "were detected but the iris annular region was too small or degenerate.",
        ])
    elif iris_no_roi > 0:
        lines.extend([
            f"Pupil and limbus detection succeed on most frames, but iris detection",
            f"returns NO_ROI for {iris_no_roi} frames. This means the annular region between",
            "pupil and limbus ellipses is too small or geometrically degenerate.",
        ])
    else:
        lines.extend([
            "Iris detection runs successfully when pupil+limbus are both detected.",
            "Any weakness is in upstream detection, not iris processing itself.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 10. What Is Working",
        "",
    ])

    if total_pupil > 0:
        lines.append(f"- Pupil detection: {total_pupil}/{total_sampled} success")
    if total_limbus > 0:
        lines.append(f"- Limbus detection: {total_limbus}/{total_sampled} success")
    if total_iris > 0:
        lines.append(f"- Iris detection: {total_iris}/{total_sampled} success (when prerequisites met)")
    if all_iris_frames:
        lines.append(f"- Iris feature extraction produces valid features with measurable coverage and spatial spread")
        lines.append(f"- Iris detection runs in ~{np.mean([fr.iris_ms for fr in all_iris_frames]):.0f} ms per frame")

    lines.extend([
        "",
        "---",
        "",
        "## 11. What Is Failing",
        "",
    ])

    if pupil_fail > 0:
        lines.append(f"- Pupil detection fails on {pupil_fail}/{total_sampled} frames")
    if limbus_fail > 0:
        lines.append(f"- Limbus detection fails on {limbus_fail}/{total_sampled} frames (primary iris blocker)")
    if iris_skip_limbus > 0:
        lines.append(f"- Iris detection skipped on {iris_skip_limbus} frames due to missing limbus")
    if iris_no_roi > 0:
        lines.append(f"- Iris NO_ROI on {iris_no_roi} frames (degenerate annular region)")

    lines.extend([
        "",
        "---",
        "",
        "## 12. Recommended Phase XX-B Changes",
        "",
        "Based on the baseline analysis:",
        "",
        "1. **If limbus failure is the primary blocker:** Focus Phase XX-B on improving limbus detection",
        "   robustness for video frames (motion blur, lighting variation, partial occlusion).",
        "",
        "2. **If iris NO_ROI is the issue:** Adjust ROI inset parameters or add fallback ROI",
        "   construction when the annular region is degenerate.",
        "",
        "3. **If iris features are insufficient:** Tune feature extraction parameters",
        "   (min_contrast, num_angles, num_radii) for real surgical video characteristics.",
        "",
        "4. **Temporal tracking:** If iris features are stable across consecutive frames,",
        "   implement feature tracking/matching for cyclotorsion estimation.",
        "",
        "5. **Performance:** If pupil+limbus detection is too slow for real-time video,",
        "   consider frame skipping or pipeline optimization.",
        "",
        "---",
        "",
        "> **Disclaimer:** These videos establish engineering detection behavior but do not by themselves",
        "> establish a clinically valid cyclotorsion measurement. No assumptions are made about",
        "> pre-dock/post-dock pairing, laterality, or ground truth.",
        "",
    ])

    report = "\n".join(lines)

    # Write report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "PHASE_XX_A_REAL_VIDEO_BASELINE.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report written to: {report_path}")

    # Also write raw JSON data
    json_data = []
    for a in analyses:
        vid_data = {
            "inventory": {
                "filename": a.inventory.filename,
                "resolution": list(a.inventory.resolution),
                "fps": a.inventory.fps,
                "frame_count": a.inventory.frame_count,
                "duration_s": a.inventory.duration_s,
                "codec": a.inventory.codec,
                "file_size_mb": a.inventory.file_size_mb,
            },
            "summary": {
                "total_frames_sampled": a.total_frames_sampled,
                "pupil_success": a.pupil_success_count,
                "limbus_success": a.limbus_success_count,
                "iris_success": a.iris_success_count,
                "iris_skip": a.iris_skip_count,
            },
            "frames": [
                {
                    "frame_index": fr.frame_index,
                    "timestamp_s": fr.timestamp_s,
                    "pupil_detected": fr.pupil_detected,
                    "pupil_confidence": fr.pupil_confidence,
                    "pupil_center": fr.pupil_center,
                    "limbus_detected": fr.limbus_detected,
                    "limbus_confidence": fr.limbus_confidence,
                    "limbus_center": fr.limbus_center,
                    "has_both": fr.has_both,
                    "iris_status": fr.iris_status,
                    "iris_features": fr.iris_features,
                    "iris_coverage": fr.iris_coverage,
                    "iris_spatial_spread": fr.iris_spatial_spread,
                    "pupil_ms": fr.pupil_ms,
                    "limbus_ms": fr.limbus_ms,
                    "iris_ms": fr.iris_ms,
                    "total_ms": fr.total_ms,
                    "failure_reason": fr.failure_reason,
                }
                for fr in a.frame_results
            ],
        }
        json_data.append(vid_data)

    json_path = os.path.join(output_dir, "phase_xx_a_baseline_data.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON data written to: {json_path}")

    return report_path


# ── Entry point ───────────────────────────────────────────────────────


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "_phase_artifacts")

    # Discover ELITA videos in project root
    video_files = []
    for f in os.listdir(project_root):
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            video_files.append(os.path.join(project_root, f))

    if not video_files:
        print("ERROR: No video files found in project root")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s):")
    for vf in video_files:
        print(f"  {os.path.basename(vf)}")

    analyses = []
    for vf in sorted(video_files):
        try:
            a = analyze_video(vf, max_samples=20)
            analyses.append(a)
        except Exception as exc:
            print(f"  ERROR analyzing {vf}: {exc}")

    if analyses:
        generate_report(analyses, output_dir)
        print("\n  Phase XX-A baseline analysis complete.")
    else:
        print("\n  ERROR: No videos analyzed successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
