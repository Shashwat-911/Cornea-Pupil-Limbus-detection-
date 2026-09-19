#!/usr/bin/env python3
"""
evaluate_ring_detection.py — Evaluate ring detection accuracy.

Compares ring detector output against ground-truth labels and prints
a comprehensive classification report including accuracy, precision,
recall, F1 score, and a confusion matrix.

Can evaluate:
  • The CNN classifier alone (``--classifier``)
  • The heuristic detector alone (``--heuristic-only``)
  • The combined detector (default — CNN + heuristic)

Prerequisites
-------------
1. A directory of eye images.
2. A ``ring_labels.json`` ground-truth file (from ``annotate_ring_data.py``).
3. Optionally a trained ring classifier (``models/ring_classifier.pth``).

Usage
-----
::

    # Evaluate combined detector (CNN + heuristic)
    python scripts/evaluate_ring_detection.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json \\
        --classifier models/ring_classifier.pth

    # Evaluate heuristic only (no trained classifier needed)
    python scripts/evaluate_ring_detection.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json \\
        --heuristic-only

    # Save detailed per-image results to CSV
    python scripts/evaluate_ring_detection.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json \\
        --classifier models/ring_classifier.pth \\
        --output-csv evaluation_results.csv

    # Show misclassified images interactively
    python scripts/evaluate_ring_detection.py \\
        --image-dir clinical_data/training_data/images \\
        --labels clinical_data/ring_labels.json \\
        --classifier models/ring_classifier.pth \\
        --show-errors
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Ensure project root is importable ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pupil_tracking.core.deterministic_ring_detector import (
    RingDetector,
    HeuristicRingDetector,
    RingDetectionResult,
    RingStatus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate ring detection accuracy against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument(
        "--image-dir", type=str, required=True,
        help="Directory containing eye images",
    )
    p.add_argument(
        "--labels", type=str, required=True,
        help="Path to ring_labels.json ground-truth file",
    )

    # Detector configuration
    p.add_argument(
        "--classifier", type=str, default=None,
        help="Path to trained ring classifier .pth (omit for heuristic-only)",
    )
    p.add_argument(
        "--heuristic-only", action="store_true",
        help="Evaluate only the heuristic detector (ignore classifier)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device (default: auto-detect)",
    )

    # Output
    p.add_argument(
        "--output-csv", type=str, default=None,
        help="Save per-image results to CSV file",
    )
    p.add_argument(
        "--show-errors", action="store_true",
        help="Display misclassified images in an OpenCV window",
    )
    p.add_argument(
        "--confidence-threshold", type=float, default=0.50,
        help="Confidence threshold for PRESENT/ABSENT decision (default: 0.50)",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Result container
# ═══════════════════════════════════════════════════════════════════════

class EvalRecord:
    """Per-image evaluation record."""

    __slots__ = (
        "filename", "gt_ring", "gt_visibility",
        "pred_ring", "pred_status", "confidence",
        "method", "correct", "latency_ms",
    )

    def __init__(
        self,
        filename: str,
        gt_ring: bool,
        gt_visibility: str,
        pred_ring: bool,
        pred_status: str,
        confidence: float,
        method: str,
        latency_ms: float,
    ):
        self.filename = filename
        self.gt_ring = gt_ring
        self.gt_visibility = gt_visibility
        self.pred_ring = pred_ring
        self.pred_status = pred_status
        self.confidence = confidence
        self.method = method
        self.correct = (gt_ring == pred_ring)
        self.latency_ms = latency_ms

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "gt_ring": self.gt_ring,
            "gt_visibility": self.gt_visibility,
            "pred_ring": self.pred_ring,
            "pred_status": self.pred_status,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "correct": self.correct,
            "latency_ms": round(self.latency_ms, 2),
        }


# ═══════════════════════════════════════════════════════════════════════
#  Metrics computation
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(records: List[EvalRecord]) -> Dict[str, float]:
    """Compute classification metrics from evaluation records."""
    tp = sum(1 for r in records if r.gt_ring and r.pred_ring)
    tn = sum(1 for r in records if not r.gt_ring and not r.pred_ring)
    fp = sum(1 for r in records if not r.gt_ring and r.pred_ring)
    fn = sum(1 for r in records if r.gt_ring and not r.pred_ring)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Average confidence
    avg_conf = np.mean([r.confidence for r in records]) if records else 0.0
    avg_conf_correct = np.mean(
        [r.confidence for r in records if r.correct]
    ) if any(r.correct for r in records) else 0.0
    avg_conf_wrong = np.mean(
        [r.confidence for r in records if not r.correct]
    ) if any(not r.correct for r in records) else 0.0

    # Latency
    avg_latency = np.mean([r.latency_ms for r in records]) if records else 0.0
    p95_latency = np.percentile(
        [r.latency_ms for r in records], 95
    ) if records else 0.0

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "avg_confidence": float(avg_conf),
        "avg_confidence_correct": float(avg_conf_correct),
        "avg_confidence_wrong": float(avg_conf_wrong),
        "avg_latency_ms": float(avg_latency),
        "p95_latency_ms": float(p95_latency),
    }


# ═══════════════════════════════════════════════════════════════════════
#  CSV export
# ═══════════════════════════════════════════════════════════════════════

def save_csv(records: List[EvalRecord], path: str) -> None:
    """Write per-image evaluation results to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename", "gt_ring", "gt_visibility",
        "pred_ring", "pred_status", "confidence",
        "method", "correct", "latency_ms",
    ]

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())

    logger.info("Results saved to %s", out)


# ═══════════════════════════════════════════════════════════════════════
#  Error visualisation
# ═══════════════════════════════════════════════════════════════════════

def show_misclassified(
    records: List[EvalRecord],
    image_dir: Path,
    max_display: int = 800,
) -> None:
    """Display misclassified images in an OpenCV window for review."""
    errors = [r for r in records if not r.correct]
    if not errors:
        print("\n  ✅ No misclassified images — nothing to show.\n")
        return

    print(f"\n  Showing {len(errors)} misclassified images.")
    print("  Press any key for next, Q to stop.\n")

    window_name = "Misclassified"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    for i, rec in enumerate(errors):
        img_path = image_dir / rec.filename
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Resize for display
        h, w = image.shape[:2]
        if max(h, w) > max_display:
            scale = max_display / max(h, w)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Overlay info
        dh, dw = image.shape[:2]

        # Background bar
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (dw, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Text
        cv2.putText(
            image,
            f"[{i+1}/{len(errors)}] {rec.filename}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
        )
        gt_text = f"Ground Truth: {'RING' if rec.gt_ring else 'NO RING'}"
        pred_text = (
            f"Prediction:   {'RING' if rec.pred_ring else 'NO RING'} "
            f"(conf={rec.confidence:.3f})"
        )
        cv2.putText(
            image, gt_text, (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
        )
        cv2.putText(
            image, pred_text, (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1,
        )

        cv2.imshow(window_name, image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation loop
# ═══════════════════════════════════════════════════════════════════════

def evaluate(
    image_dir: Path,
    labels: Dict[str, dict],
    detector: RingDetector,
) -> List[EvalRecord]:
    """Run the detector on every labelled image and collect results."""
    records: List[EvalRecord] = []
    total = len(labels)

    for i, (fname, info) in enumerate(sorted(labels.items())):
        img_path = image_dir / fname
        if not img_path.exists():
            logger.debug("Image not found (skipping): %s", img_path)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Cannot read image (skipping): %s", img_path)
            continue

        gt_ring = bool(info.get("ring_present", False))
        gt_visibility = info.get("ring_visibility", "unknown")

        t0 = time.perf_counter()
        result = detector.detect(image)
        latency = (time.perf_counter() - t0) * 1000.0

        pred_ring = result.status in (RingStatus.PRESENT, RingStatus.PARTIAL)

        rec = EvalRecord(
            filename=fname,
            gt_ring=gt_ring,
            gt_visibility=gt_visibility,
            pred_ring=pred_ring,
            pred_status=result.status.value,
            confidence=result.confidence,
            method=result.method,
            latency_ms=latency,
        )
        records.append(rec)

        # Progress
        if (i + 1) % 25 == 0 or (i + 1) == total:
            correct_so_far = sum(1 for r in records if r.correct)
            logger.info(
                "  Processed %d/%d — running accuracy: %.3f",
                i + 1, total, correct_so_far / len(records),
            )

    return records


# ═══════════════════════════════════════════════════════════════════════
#  Report printing
# ═══════════════════════════════════════════════════════════════════════

def print_report(
    metrics: Dict[str, float],
    records: List[EvalRecord],
) -> None:
    """Print a formatted evaluation report."""

    print()
    print("=" * 64)
    print("  RING DETECTION EVALUATION REPORT")
    print("=" * 64)

    # Confusion matrix
    print()
    print("  Confusion Matrix:")
    print("  " + "-" * 40)
    print(f"                    Predicted")
    print(f"                    Ring    No-Ring")
    print(f"  Actual Ring     {metrics['tp']:5d}    {metrics['fn']:5d}")
    print(f"  Actual No-Ring  {metrics['fp']:5d}    {metrics['tn']:5d}")
    print("  " + "-" * 40)

    # Metrics
    print()
    print("  Classification Metrics:")
    print("  " + "-" * 40)
    print(f"  Total images:    {metrics['total']}")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print()
    print(f"  Accuracy:        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall:          {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score:        {metrics['f1']:.4f}  ({metrics['f1']*100:.1f}%)")
    print(f"  Specificity:     {metrics['specificity']:.4f}  ({metrics['specificity']*100:.1f}%)")
    print("  " + "-" * 40)

    # Confidence analysis
    print()
    print("  Confidence Analysis:")
    print("  " + "-" * 40)
    print(f"  Avg confidence (all):     {metrics['avg_confidence']:.4f}")
    print(f"  Avg confidence (correct): {metrics['avg_confidence_correct']:.4f}")
    print(f"  Avg confidence (wrong):   {metrics['avg_confidence_wrong']:.4f}")
    print("  " + "-" * 40)

    # Latency
    print()
    print("  Latency:")
    print("  " + "-" * 40)
    print(f"  Average:  {metrics['avg_latency_ms']:.1f} ms")
    print(f"  P95:      {metrics['p95_latency_ms']:.1f} ms")
    print("  " + "-" * 40)

    # Misclassified list
    errors = [r for r in records if not r.correct]
    if errors:
        print()
        print(f"  Misclassified Images ({len(errors)}):")
        print("  " + "-" * 40)
        # Show up to 20
        for r in errors[:20]:
            gt_str = "RING" if r.gt_ring else "NO_RING"
            pred_str = "RING" if r.pred_ring else "NO_RING"
            print(
                f"    {r.filename:<30s}  "
                f"GT={gt_str:<8s}  "
                f"Pred={pred_str:<8s}  "
                f"Conf={r.confidence:.3f}"
            )
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
        print("  " + "-" * 40)

    # Quality assessment
    print()
    acc = metrics["accuracy"]
    if acc >= 0.95:
        print("  ✅ EXCELLENT — Ring detection is highly reliable.")
    elif acc >= 0.90:
        print("  ✅ GOOD — Ring detection works well for most images.")
    elif acc >= 0.80:
        print("  ⚠️  FAIR — Some misclassifications; consider more training data.")
    elif acc >= 0.70:
        print("  ⚠️  POOR — Significant errors; review misclassified images.")
    else:
        print("  ❌ UNRELIABLE — Detector needs retraining or more data.")

    if metrics["fp"] > metrics["fn"]:
        print("     Bias: More false positives (seeing rings where there are none).")
        print("     Fix:  Add more no-ring examples to training data.")
    elif metrics["fn"] > metrics["fp"]:
        print("     Bias: More false negatives (missing actual rings).")
        print("     Fix:  Add more ring examples, especially edge cases.")

    print()
    print("=" * 64)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Load ground truth ─────────────────────────────────────────
    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error("Label file not found: %s", labels_path)
        sys.exit(1)

    with open(labels_path, "r") as f:
        labels = json.load(f)

    logger.info("Loaded %d ground-truth labels from %s", len(labels), labels_path)

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    # ── Initialise detector ───────────────────────────────────────
    if args.heuristic_only:
        logger.info("Evaluating HEURISTIC detector only (no CNN classifier)")
        detector = RingDetector(
            classifier_path=None,
            device=args.device,
            use_heuristic_fallback=True,
        )
    else:
        classifier_path = args.classifier
        if classifier_path and not Path(classifier_path).exists():
            logger.warning(
                "Classifier not found at %s — falling back to heuristic",
                classifier_path,
            )
            classifier_path = None

        detector = RingDetector(
            classifier_path=classifier_path,
            device=args.device,
            use_heuristic_fallback=True,
        )

        if detector.classifier is not None:
            logger.info("Evaluating COMBINED detector (CNN + heuristic)")
        else:
            logger.info("Evaluating HEURISTIC detector (classifier not loaded)")

    # ── Run evaluation ────────────────────────────────────────────
    print()
    print("  Running evaluation...")
    print()

    records = evaluate(image_dir, labels, detector)

    if not records:
        logger.error("No images were evaluated — check paths and labels")
        sys.exit(1)

    # ── Compute and display metrics ───────────────────────────────
    metrics = compute_metrics(records)
    print_report(metrics, records)

    # ── Save CSV if requested ─────────────────────────────────────
    if args.output_csv:
        save_csv(records, args.output_csv)

    # ── Show misclassified if requested ───────────────────────────
    if args.show_errors:
        show_misclassified(records, image_dir)


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()