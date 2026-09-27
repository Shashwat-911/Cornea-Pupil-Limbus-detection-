"""
Automated clinical-image accuracy test.

Loads every image from ``clinical_data/clean/`` and asserts 100%
pupil + limbus detection with sane geometry.

Run:
    python -m pytest pupil_tracking/tests/test_clinical_accuracy.py -v

Or run standalone for a summary table:
    python pupil_tracking/tests/test_clinical_accuracy.py

Plan-aligned:
    - Tests pupil detection on all clinical images
    - Tests limbus detection on all clinical images
    - Tests confidence > 0.3 for both
    - Tests pupil centre is inside limbus boundary
    - Prints per-image summary table
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Locate project root
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Discover clinical images
# ---------------------------------------------------------------------------
_CLEAN_DIR = _PROJECT_ROOT / "clinical_data" / "clean"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_clinical_images() -> List[Path]:
    """Find all clinical images in the clean directory."""
    if not _CLEAN_DIR.is_dir():
        # Try alternative locations
        alt_dirs = [
            _PROJECT_ROOT / "data" / "clinical" / "clean",
            _PROJECT_ROOT / "data" / "clean",
            _PROJECT_ROOT / "images" / "clinical",
            _PROJECT_ROOT / "clinical_images",
        ]
        for alt in alt_dirs:
            if alt.is_dir():
                return sorted(
                    p for p in alt.iterdir()
                    if p.suffix.lower() in _IMAGE_EXTENSIONS
                )
        return []

    return sorted(
        p for p in _CLEAN_DIR.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


_CLINICAL_IMAGES = _find_clinical_images()


# ---------------------------------------------------------------------------
# Helper: get detection result with flexible API
# ---------------------------------------------------------------------------

def _detect_image(detector, image_bgr: np.ndarray) -> Dict[str, Any]:
    """Run detection and normalise the result to a flat dict.

    Handles both UnifiedDetector (returns EyeDetectionResult) and
    FastInference (returns flat dict) APIs.
    """
    raw = detector.detect(image_bgr)

    # If it's already a dict (FastInference.detect())
    if isinstance(raw, dict):
        return raw

    # EyeDetectionResult from UnifiedDetector
    result: Dict[str, Any] = {
        "pupil_detected": False,
        "pupil_x": 0.0,
        "pupil_y": 0.0,
        "pupil_radius": 0.0,
        "pupil_confidence": 0.0,
        "limbus_detected": False,
        "limbus_x": 0.0,
        "limbus_y": 0.0,
        "limbus_radius": 0.0,
        "limbus_confidence": 0.0,
    }

    if hasattr(raw, "pupil") and raw.pupil is not None:
        p = raw.pupil
        if getattr(p, "detected", False):
            result["pupil_detected"] = True
            result["pupil_confidence"] = getattr(p, "confidence", 0.0)

            if hasattr(p, "ellipse") and p.ellipse is not None:
                result["pupil_x"] = getattr(p.ellipse, "center_x", 0.0)
                result["pupil_y"] = getattr(p.ellipse, "center_y", 0.0)
                result["pupil_radius"] = getattr(p.ellipse, "radius", 0.0)
            elif hasattr(p, "center_x"):
                result["pupil_x"] = p.center_x
                result["pupil_y"] = p.center_y
                result["pupil_radius"] = getattr(p, "radius", 0.0)

    if hasattr(raw, "limbus") and raw.limbus is not None:
        l = raw.limbus
        if getattr(l, "detected", False):
            result["limbus_detected"] = True
            result["limbus_confidence"] = getattr(l, "confidence", 0.0)

            if hasattr(l, "ellipse") and l.ellipse is not None:
                result["limbus_x"] = getattr(l.ellipse, "center_x", 0.0)
                result["limbus_y"] = getattr(l.ellipse, "center_y", 0.0)
                result["limbus_radius"] = getattr(l.ellipse, "radius", 0.0)
            elif hasattr(l, "center_x"):
                result["limbus_x"] = l.center_x
                result["limbus_y"] = l.center_y
                result["limbus_radius"] = getattr(l, "radius", 0.0)

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    """Create a detector once for all tests.

    Tries UnifiedDetector first, then FastInference as fallback.
    """
    # Try UnifiedDetector
    try:
        from pupil_tracking.core.detector import UnifiedDetector
        det = UnifiedDetector()
        # Warm up
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            det.detect(dummy)
        except Exception:
            pass
        return det
    except (ImportError, Exception) as exc:
        print(f"UnifiedDetector not available ({exc}), trying FastInference")

    # Try FastInference
    try:
        from pupil_tracking.ml.fast_inference import FastInference
        model_path = str(_PROJECT_ROOT / "models" / "best_model.pth")
        if not os.path.exists(model_path):
            # Search for any .pth file
            models_dir = _PROJECT_ROOT / "models"
            if models_dir.is_dir():
                pth_files = list(models_dir.glob("*.pth"))
                if pth_files:
                    model_path = str(pth_files[0])

        det = FastInference(model_path=model_path)
        return det
    except (ImportError, Exception) as exc:
        pytest.skip(f"No detector available: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    len(_CLINICAL_IMAGES) == 0,
    reason=f"No clinical images found in {_CLEAN_DIR} or alternative locations",
)
class TestClinicalDetection:
    """Each clinical image must yield a valid pupil AND limbus detection."""

    @pytest.fixture(autouse=True)
    def _inject_detector(self, detector):
        self.detector = detector

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert img is not None, f"Failed to load {path}"
        return img

    # ---- Pupil detection ─────────────────────────────────────

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_pupil_detected(self, img_path: Path):
        """Pupil must be detected in every clinical image."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)
        assert result.get("pupil_detected", False), (
            f"Pupil NOT detected in {img_path.name}"
        )

    # ---- Limbus detection ────────────────────────────────────

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_limbus_detected(self, img_path: Path):
        """Limbus must be detected in every clinical image."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)
        assert result.get("limbus_detected", False), (
            f"Limbus NOT detected in {img_path.name}"
        )

    # ---- Confidence thresholds ───────────────────────────────

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_pupil_confidence(self, img_path: Path):
        """Pupil confidence must exceed 0.3."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)

        if not result.get("pupil_detected", False):
            pytest.skip("Pupil not detected — covered by other test")

        conf = result.get("pupil_confidence", 0.0)
        assert conf > 0.3, (
            f"Pupil confidence too low ({conf:.3f}) in {img_path.name}"
        )

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_limbus_confidence(self, img_path: Path):
        """Limbus confidence must exceed 0.3."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)

        if not result.get("limbus_detected", False):
            pytest.skip("Limbus not detected — covered by other test")

        conf = result.get("limbus_confidence", 0.0)
        assert conf > 0.3, (
            f"Limbus confidence too low ({conf:.3f}) in {img_path.name}"
        )

    # ---- Geometric sanity: pupil inside limbus ───────────────

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_pupil_inside_limbus(self, img_path: Path):
        """Pupil centre must be inside the limbus boundary."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)

        if not result.get("pupil_detected") or not result.get("limbus_detected"):
            pytest.skip("Detection missing — covered by other tests")

        pcx = result.get("pupil_x", 0)
        pcy = result.get("pupil_y", 0)
        lcx = result.get("limbus_x", 0)
        lcy = result.get("limbus_y", 0)
        lr = result.get("limbus_radius", 0)

        if lr < 1:
            pytest.skip("Limbus radius too small for geometric check")

        dist = np.sqrt((pcx - lcx) ** 2 + (pcy - lcy) ** 2)
        assert dist < lr, (
            f"Pupil centre ({pcx:.0f},{pcy:.0f}) is outside limbus "
            f"(centre {lcx:.0f},{lcy:.0f}, r={lr:.0f}) "
            f"in {img_path.name}  (dist={dist:.0f})"
        )

    # ---- Radius ratio sanity ─────────────────────────────────

    @pytest.mark.parametrize(
        "img_path",
        _CLINICAL_IMAGES,
        ids=[p.stem for p in _CLINICAL_IMAGES],
    )
    def test_radius_ratio(self, img_path: Path):
        """Pupil radius must be smaller than limbus radius."""
        img = self._load(img_path)
        result = _detect_image(self.detector, img)

        if not result.get("pupil_detected") or not result.get("limbus_detected"):
            pytest.skip("Detection missing — covered by other tests")

        pr = result.get("pupil_radius", 0)
        lr = result.get("limbus_radius", 0)

        if lr < 1 or pr < 1:
            pytest.skip("Radii too small for ratio check")

        ratio = pr / lr
        assert ratio < 0.85, (
            f"Pupil/limbus ratio too large ({ratio:.2f}) "
            f"in {img_path.name}  (pupil_r={pr:.0f}, limbus_r={lr:.0f})"
        )

        assert ratio > 0.10, (
            f"Pupil/limbus ratio suspiciously small ({ratio:.2f}) "
            f"in {img_path.name}"
        )


# ---------------------------------------------------------------------------
# Standalone summary
# ---------------------------------------------------------------------------

def _print_summary():
    """Run detection on all clinical images and print a summary table.

    Useful for quick visual verification outside pytest.
    """
    print("\n" + "=" * 78)
    print("  CLINICAL IMAGE DETECTION SUMMARY")
    print("=" * 78)

    images = _find_clinical_images()
    if not images:
        print(f"  No images found in {_CLEAN_DIR} or alternative locations")
        print("  Searched:")
        print(f"    {_CLEAN_DIR}")
        for alt in [
            _PROJECT_ROOT / "data" / "clinical" / "clean",
            _PROJECT_ROOT / "data" / "clean",
            _PROJECT_ROOT / "images" / "clinical",
            _PROJECT_ROOT / "clinical_images",
        ]:
            print(f"    {alt}")
        return

    # Try to create detector
    det = None
    try:
        from pupil_tracking.core.detector import UnifiedDetector
        det = UnifiedDetector()
    except Exception:
        pass

    if det is None:
        try:
            from pupil_tracking.ml.fast_inference import FastInference
            model_path = str(_PROJECT_ROOT / "models" / "best_model.pth")
            if not os.path.exists(model_path):
                models_dir = _PROJECT_ROOT / "models"
                if models_dir.is_dir():
                    pth_files = list(models_dir.glob("*.pth"))
                    if pth_files:
                        model_path = str(pth_files[0])
            det = FastInference(model_path=model_path)
        except Exception as exc:
            print(f"  ERROR: Could not create detector: {exc}")
            return

    # Header
    hdr = (
        f"  {'Image':<28} {'Pupil':>6} {'P-Conf':>7} "
        f"{'Limbus':>7} {'L-Conf':>7} {'Ratio':>6} {'Time':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr.strip())))

    total_pass = 0
    total_fail = 0

    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  {p.name:<28} {'LOAD FAIL':>6}")
            total_fail += 1
            continue

        try:
            t0 = time.time()
            r = _detect_image(det, img)
            elapsed_ms = (time.time() - t0) * 1000

            pupil_ok = "OK" if r.get("pupil_detected") else "FAIL"
            limbus_ok = "OK" if r.get("limbus_detected") else "FAIL"
            pc = f"{r.get('pupil_confidence', 0):.2f}"
            lc = f"{r.get('limbus_confidence', 0):.2f}"

            pr = r.get("pupil_radius", 0)
            lr = r.get("limbus_radius", 0)
            ratio = f"{pr / lr:.2f}" if lr > 0 else "-"

            time_str = f"{elapsed_ms:.0f}ms"

            both_ok = r.get("pupil_detected") and r.get("limbus_detected")
            if both_ok:
                total_pass += 1
            else:
                total_fail += 1

            print(
                f"  {p.name:<28} {pupil_ok:>6} {pc:>7} "
                f"{limbus_ok:>7} {lc:>7} {ratio:>6} {time_str:>8}"
            )
        except Exception as exc:
            print(f"  {p.name:<28} {'ERR':>6}  {str(exc)[:40]}")
            total_fail += 1

    print("  " + "-" * (len(hdr.strip())))
    total = total_pass + total_fail
    pct = 100 * total_pass / total if total > 0 else 0
    status = "PASS" if total_fail == 0 else "FAIL"
    print(
        f"  {status}  {total_pass}/{total} images detected "
        f"({pct:.0f}%)"
    )
    print("=" * 78)


if __name__ == "__main__":
    _print_summary()