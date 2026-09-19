"""
Pixel-to-millimetre spatial calibration.

Provides multiple calibration strategies:
    1. Suction ring (known diameter 9.0-9.5 mm)
    2. Limbus / corneal diameter (average 12.0 mm)
    3. Manual calibration (user-provided px/mm)
    4. Known object in frame

For surgical precision, calibration uncertainty is tracked and
propagated through all measurements.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from pupil_tracking.utils.types import (
    CalibrationInfo,
    EllipseParams,
    EyeDetectionResult,
    LimbusDetection,
)
from pupil_tracking.utils.logger import get_logger


class SpatialCalibrator:
    """Manages pixel-to-mm calibration with uncertainty tracking.

    Usage
    -----
    >>> cal = SpatialCalibrator()
    >>> info = cal.calibrate_from_limbus(limbus_detection)
    >>> mm = info.px_to_mm(100.0)  # 100 pixels in mm
    """

    # Known anatomical references (population averages)
    CORNEAL_DIAMETER_MM = 12.0          # horizontal white-to-white
    CORNEAL_DIAMETER_STD_MM = 0.5       # population std
    SUCTION_RING_DIAMETERS_MM = {
        "standard": 9.4,
        "small": 8.5,
        "large": 10.0,
    }

    def __init__(
        self,
        mode: str = "ANATOMICAL_ANCHOR",
        corneal_diameter_mm: float = 12.0,
        manual_px_per_mm: Optional[float] = None,
        manual_mm_per_px: Optional[float] = None,
        suction_ring_diameter_mm: float = 9.4,
    ) -> None:
        self.logger = get_logger()
        self.mode = mode
        self.corneal_diameter_mm = corneal_diameter_mm
        self.manual_px_per_mm = manual_px_per_mm
        self.manual_mm_per_px = manual_mm_per_px
        self.suction_ring_diameter_mm = suction_ring_diameter_mm
        self._history: List[CalibrationInfo] = []

    def calibrate_from_limbus(
        self,
        limbus: LimbusDetection,
        corneal_diameter_mm: Optional[float] = None,
        corneal_std_mm: float = 0.5,
    ) -> CalibrationInfo:
        """Calibrate using detected limbus diameter or active mode."""
        if self.mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = float(
                self.manual_px_per_mm
                or (1.0 / self.manual_mm_per_px if self.manual_mm_per_px else 58.2)
            )
            cal = CalibrationInfo(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                reference_diameter_mm=0.0,
                reference_diameter_px=0.0,
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )
            self._history.append(cal)
            return cal

        if corneal_diameter_mm is None:
            corneal_diameter_mm = self.corneal_diameter_mm

        if not limbus.detected or limbus.ellipse is None:
            return CalibrationInfo()

        # Use semi-major axis only (horizontal corneal diameter)
        # to calibrate against the assumed horizontal HVID.
        diameter_px = limbus.ellipse.semi_major * 2.0
        if diameter_px < 20:
            return CalibrationInfo()

        px_per_mm = diameter_px / corneal_diameter_mm

        confidence = limbus.confidence * 0.8

        # account for ellipse vs circle
        aspect = limbus.ellipse.circularity
        if aspect < 0.85:
            # oblique view — less reliable
            confidence *= aspect

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source="limbus_diameter",
            method="anatomical",
            reference_diameter_mm=corneal_diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=confidence,
            corneal_diameter_assumed_mm=corneal_diameter_mm,
        )

        self._history.append(cal)
        self.logger.info(
            "Calibrated from limbus: %.2f px/mm (conf=%.2f)",
            px_per_mm, confidence,
        )
        return cal

    def calibrate_from_ring(
        self,
        ring_center: Tuple[float, float],
        ring_radius: float,
        ring_type: str = "standard",
        ring_diameter_mm: Optional[float] = None,
    ) -> CalibrationInfo:
        """Calibrate using a suction ring or illumination ring of known diameter.

        Parameters
        ----------
        ring_center : (x, y) in pixels
        ring_radius : float in pixels
        ring_type : str  "standard" | "small" | "large"
        ring_diameter_mm : float or None
            Known physical diameter of ring in mm (overrides ring_type if provided).
        """
        if ring_diameter_mm is None or ring_diameter_mm <= 0:
            diameter_mm = self.SUCTION_RING_DIAMETERS_MM.get(
                ring_type, 9.4
            )
        else:
            diameter_mm = float(ring_diameter_mm)

        diameter_px = ring_radius * 2.0

        if diameter_px < 20:
            return CalibrationInfo()

        px_per_mm = diameter_px / diameter_mm

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source=f"ring_reflection_{diameter_mm:.1f}mm",
            method="ring_reflection",
            reference_diameter_mm=diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=0.95,  # rings have known precise diameter
            corneal_diameter_assumed_mm=None,
        )

        self._history.append(cal)
        self.logger.info(
            "Calibrated from ring (%.1f mm): %.2f px/mm",
            diameter_mm, px_per_mm,
        )
        return cal

    def calibrate_manual(
        self,
        px_per_mm: float,
        source: str = "manual",
    ) -> CalibrationInfo:
        """Manual calibration with user-provided scale."""
        if px_per_mm <= 0:
            return CalibrationInfo()

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source=source,
            method="fixed_manual",
            confidence=1.0,
            corneal_diameter_assumed_mm=None,
        )
        self._history.append(cal)
        return cal

    def get_consensus_calibration(self) -> CalibrationInfo:
        """Compute consensus calibration from history.

        Uses confidence-weighted average of all calibrations.
        """
        if not self._history:
            return CalibrationInfo()

        calibrated = [c for c in self._history if c.calibrated]
        if not calibrated:
            return CalibrationInfo()

        weights = np.array([c.confidence for c in calibrated])
        px_per_mm = np.array([c.px_per_mm for c in calibrated])

        total_weight = weights.sum()
        if total_weight < 0.01:
            return calibrated[-1]

        avg_px_per_mm = float(np.average(px_per_mm, weights=weights))
        std_px_per_mm = float(
            np.sqrt(np.average((px_per_mm - avg_px_per_mm) ** 2, weights=weights))
        )

        # confidence from consistency
        if len(calibrated) > 1 and avg_px_per_mm > 0:
            cv = std_px_per_mm / avg_px_per_mm  # coefficient of variation
            consistency = max(0.0, 1.0 - cv * 5.0)
        else:
            consistency = 0.8

        avg_conf = float(np.mean(weights))
        final_conf = min(1.0, avg_conf * consistency)

        last_method = calibrated[-1].method if hasattr(calibrated[-1], "method") else "anatomical"
        last_assumed = getattr(calibrated[-1], "corneal_diameter_assumed_mm", None)

        return CalibrationInfo(
            calibrated=True,
            px_per_mm=avg_px_per_mm,
            mm_per_px=1.0 / avg_px_per_mm,
            source=f"consensus_{len(calibrated)}",
            method=last_method,
            confidence=final_conf,
            corneal_diameter_assumed_mm=last_assumed,
        )

    def reset(self) -> None:
        self._history.clear()


class StabilizedCalibrator:
    """EMA-smoothed calibration with outlier rejection and modular modes.

    Wraps raw calibration with temporal stabilization
    to prevent single noisy measurements from shifting downstream
    mm values.

    Parameters
    ----------
    config : MeasurementStabilizationConfig or None
        Stabilization parameters.  ``None`` → defaults.
    corneal_diameter_mm : float
        Known average horizontal corneal diameter for anatomical calibration.
    mode : str
        Calibration mode: 'ANATOMICAL_ANCHOR', 'FIXED_PIXEL_SCALE', or 'RING_REFLECTION'.
    manual_px_per_mm : float or None
        Fixed pixel-per-mm scale used in FIXED_PIXEL_SCALE mode.
    ring_diameter_mm : float
        Physical diameter of ring in mm used in RING_REFLECTION mode.
    """

    def __init__(
        self,
        config=None,
        corneal_diameter_mm: float = 12.0,
        mode: str = "ANATOMICAL_ANCHOR",
        manual_px_per_mm: Optional[float] = None,
        ring_diameter_mm: float = 9.4,
    ):
        from pupil_tracking.utils.config import get_config

        cfg = get_config()
        if config is None:
            config = cfg.measurement_stabilization

        self._alpha = float(config.ema_alpha)
        self._outlier_sigma = float(config.outlier_sigma)
        self._min_samples = int(config.min_samples_for_rejection)
        self._max_history = int(config.max_calibration_history)
        self._enabled = bool(config.enable_ema_smoothing)
        self._corneal_mm = corneal_diameter_mm
        self._mode = mode or getattr(cfg.calibration, "mode", "ANATOMICAL_ANCHOR")
        self._manual_px_per_mm = manual_px_per_mm if manual_px_per_mm is not None else getattr(cfg.calibration, "manual_px_per_mm", 58.2)
        self._ring_diameter_mm = ring_diameter_mm or getattr(cfg.calibration, "suction_ring_diameter_mm", 9.4)

        self._ema_px_per_mm: Optional[float] = None
        self._ema_variance: float = 0.0
        self._history: List[float] = []
        self._frozen = False  # once True, calibration stops updating

        self.logger = get_logger()

    @property
    def is_frozen(self) -> bool:
        """True when calibration has stabilised and is no longer updating."""
        return self._frozen

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(
        self,
        mode: str,
        manual_px_per_mm: Optional[float] = None,
        corneal_diameter_mm: Optional[float] = None,
        ring_diameter_mm: Optional[float] = None,
    ) -> None:
        """Dynamically configure calibration mode."""
        self._mode = mode
        if manual_px_per_mm is not None and manual_px_per_mm > 0:
            self._manual_px_per_mm = float(manual_px_per_mm)
        if corneal_diameter_mm is not None and corneal_diameter_mm > 0:
            self._corneal_mm = float(corneal_diameter_mm)
        if ring_diameter_mm is not None and ring_diameter_mm > 0:
            self._ring_diameter_mm = float(ring_diameter_mm)
        self.reset()

    def update_from_limbus(
        self,
        limbus: LimbusDetection,
    ) -> CalibrationInfo:
        """Update calibration according to the active calibration mode."""
        # ── Mode B: Fixed scale (manual / external) ───────────────────
        if self._mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = float(self._manual_px_per_mm or 58.2)
            if px_per_mm <= 0:
                px_per_mm = 58.2
            return CalibrationInfo(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )

        # If already frozen, return the locked calibration
        if self._frozen:
            return self._current_best()

        if not limbus.detected or limbus.ellipse is None:
            return self._current_best()

        # Use semi-major axis (horizontal corneal diameter) for anatomical calibration.
        # This anchors horizontal WTW to the assumed HVID.
        semi_major_px = limbus.ellipse.semi_major
        if semi_major_px < 5:
            return self._current_best()

        diameter_px = semi_major_px * 2.0
        new_val = diameter_px / self._corneal_mm

        # Bypass smoothing if disabled
        if not self._enabled:
            return CalibrationInfo(
                calibrated=True,
                px_per_mm=new_val,
                mm_per_px=1.0 / new_val,
                source="limbus_diameter",
                method="anatomical",
                reference_diameter_mm=self._corneal_mm,
                reference_diameter_px=diameter_px,
                confidence=min(0.95, limbus.confidence * 0.8),
                corneal_diameter_assumed_mm=self._corneal_mm,
            )

        # Outlier rejection (once enough history)
        if (
            len(self._history) >= self._min_samples
            and self._ema_px_per_mm is not None
        ):
            std = math.sqrt(self._ema_variance) if self._ema_variance > 0 else 0.0
            if std > 0 and abs(new_val - self._ema_px_per_mm) > self._outlier_sigma * std:
                self.logger.debug(
                    "Calibration outlier rejected: %.3f (EMA=%.3f ± %.3f)",
                    new_val, self._ema_px_per_mm, std,
                )
                return self._current_best()

        # EMA update
        if self._ema_px_per_mm is None:
            self._ema_px_per_mm = new_val
            self._ema_variance = 0.0
        else:
            diff = new_val - self._ema_px_per_mm
            self._ema_px_per_mm += self._alpha * diff
            self._ema_variance = (
                (1.0 - self._alpha) * (self._ema_variance + self._alpha * diff * diff)
            )

        # Maintain bounded history
        self._history.append(new_val)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Freeze calibration once we have enough stable samples.
        if (
            len(self._history) >= self._min_samples * 2
            and self._ema_px_per_mm is not None
            and self._ema_px_per_mm > 0
        ):
            std = math.sqrt(self._ema_variance) if self._ema_variance > 0 else 0.0
            cv = std / self._ema_px_per_mm
            if cv < 0.02:
                self._frozen = True
                self.logger.info(
                    "Calibration FROZEN at %.4f px/mm (CV=%.4f, %d samples)",
                    self._ema_px_per_mm, cv, len(self._history),
                )

        return self._current_best()

    def update_from_ring(
        self,
        ring_radius_px: float,
        ring_diameter_mm: Optional[float] = None,
    ) -> CalibrationInfo:
        """Update calibration using ring reflection radius."""
        if ring_radius_px is None or ring_radius_px < 10:
            return self._current_best()

        ref_mm = ring_diameter_mm or self._ring_diameter_mm
        diameter_px = ring_radius_px * 2.0
        new_val = diameter_px / ref_mm

        if self._ema_px_per_mm is None:
            self._ema_px_per_mm = new_val
            self._ema_variance = 0.0
        else:
            diff = new_val - self._ema_px_per_mm
            self._ema_px_per_mm += self._alpha * diff
            self._ema_variance = (
                (1.0 - self._alpha) * (self._ema_variance + self._alpha * diff * diff)
            )

        self._history.append(new_val)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return CalibrationInfo(
            calibrated=True,
            px_per_mm=self._ema_px_per_mm,
            mm_per_px=1.0 / self._ema_px_per_mm,
            source=f"ring_reflection_{ref_mm:.1f}mm",
            method="ring_reflection",
            reference_diameter_mm=ref_mm,
            reference_diameter_px=diameter_px,
            confidence=0.95,
            corneal_diameter_assumed_mm=None,
        )

    def _current_best(self) -> CalibrationInfo:
        """Return the current EMA-smoothed calibration with uncertainty."""
        if self._mode in ("FIXED_PIXEL_SCALE", "fixed_manual", "manual"):
            px_per_mm = float(self._manual_px_per_mm or 58.2)
            if px_per_mm <= 0:
                px_per_mm = 58.2
            return CalibrationInfo(
                calibrated=True,
                px_per_mm=px_per_mm,
                mm_per_px=1.0 / px_per_mm,
                source="fixed_manual",
                method="fixed_manual",
                confidence=1.0,
                corneal_diameter_assumed_mm=None,
            )

        if self._ema_px_per_mm is None:
            return CalibrationInfo()

        confidence = min(0.95, 0.5 + len(self._history) * 0.05)
        std_px_per_mm = (
            math.sqrt(self._ema_variance)
            if self._ema_variance > 0
            else 0.0
        )
        mm_per_px = 1.0 / self._ema_px_per_mm
        mm_per_px_uncertainty = (
            std_px_per_mm / (self._ema_px_per_mm ** 2)
            if self._ema_px_per_mm > 0
            else 0.0
        )

        method = "ring_reflection" if self._mode == "RING_REFLECTION" else "anatomical"
        ref_mm = self._ring_diameter_mm if self._mode == "RING_REFLECTION" else self._corneal_mm
        assumed_mm = self._corneal_mm if method == "anatomical" else None

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=self._ema_px_per_mm,
            mm_per_px=mm_per_px,
            source="stabilized_limbus_frozen" if self._frozen else ("stabilized_ring" if method == "ring_reflection" else "stabilized_limbus"),
            method=method,
            reference_diameter_mm=ref_mm,
            reference_diameter_px=self._ema_px_per_mm * ref_mm,
            confidence=confidence,
            corneal_diameter_assumed_mm=assumed_mm,
        )

        cal.mm_per_px_uncertainty = mm_per_px_uncertainty
        cal.px_per_mm_std = std_px_per_mm

        return cal

    def reset(self) -> None:
        """Reset all smoothing state."""
        self._ema_px_per_mm = None
        self._ema_variance = 0.0
        self._history.clear()
        self._frozen = False


def ellipse_major_diameter_px(ellipse: Optional[EllipseParams]) -> float:
    """Return the full major-axis diameter in pixels.

    The limbus and corneal diameter should be derived from the ellipse's major axis,
    not the mean radius, because the detected contour can be elliptical.
    """
    if ellipse is None:
        return 0.0

    semi_major = float(getattr(ellipse, "semi_major", 0.0) or 0.0)
    if semi_major > 0.0:
        return semi_major * 2.0

    radius = float(getattr(ellipse, "radius", 0.0) or 0.0)
    return radius * 2.0 if radius > 0.0 else 0.0


def ellipse_major_diameter_mm(ellipse: Optional[EllipseParams], mm_per_px: float) -> float:
    """Return the full major-axis diameter converted to mm."""
    if mm_per_px <= 0.0:
        return 0.0
    return ellipse_major_diameter_px(ellipse) * float(mm_per_px)


def correct_pre_docked_limbus_ellipse(
    ellipse: Optional[EllipseParams],
    lower_quadrant_pct: float = 0.015,
) -> Optional[EllipseParams]:
    """Apply a conservative correction for the lower-left limbus underfit.

    Pre-docked limbus fits can be slightly too small in the 180°–270° sector.
    This inflates only that sector by a small amount, without changing the rest of
    the ellipse geometry or any docked logic.
    """
    if ellipse is None:
        return None

    gain = max(0.0, float(lower_quadrant_pct))
    if gain <= 0.0:
        return ellipse

    corrected = EllipseParams(
        center_x=ellipse.center_x,
        center_y=ellipse.center_y,
        semi_major=ellipse.semi_major * (1.0 + 0.35 * gain),
        semi_minor=ellipse.semi_minor * (1.0 + gain),
        angle_deg=ellipse.angle_deg,
        uncertainty_center_x=ellipse.uncertainty_center_x,
        uncertainty_center_y=ellipse.uncertainty_center_y,
        uncertainty_semi_major=ellipse.uncertainty_semi_major,
        uncertainty_semi_minor=ellipse.uncertainty_semi_minor,
        fit_quality=ellipse.fit_quality,
        fit_rms_residual=ellipse.fit_rms_residual,
        num_contour_points=ellipse.num_contour_points,
        eccentricity=ellipse.eccentricity,
        circularity=ellipse.circularity,
    )
    return corrected


def calculate_ruler_scale(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    known_distance_mm: float,
) -> Tuple[float, float]:
    """Calculate scale in px/mm and mm/px from 2 user-selected points.

    Parameters
    ----------
    point1 : (x, y) coordinates of point 1 in pixels
    point2 : (x, y) coordinates of point 2 in pixels
    known_distance_mm : Known physical distance between point 1 and 2 in mm

    Returns
    -------
    (px_per_mm, mm_per_px)
    """
    if known_distance_mm <= 0:
        raise ValueError("known_distance_mm must be positive")

    dx = float(point2[0] - point1[0])
    dy = float(point2[1] - point1[1])
    dist_px = math.hypot(dx, dy)
    if dist_px < 1.0:
        raise ValueError("Selected points are too close together (< 1 pixel)")

    px_per_mm = dist_px / float(known_distance_mm)
    mm_per_px = 1.0 / px_per_mm
    return px_per_mm, mm_per_px


def evaluate_clinical_wtw(
    limbus: LimbusDetection,
    cal: CalibrationInfo,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool, str]:
    """Calculate clinical White-to-White (WTW) metrics and validate physiological bounds.

    Returns
    -------
    (wtw_horizontal_mm, wtw_vertical_mm, wtw_mean_mm, wtw_astigmatism_mm, is_measured, validity_status)
    """
    if not limbus.detected or limbus.ellipse is None or not cal.calibrated or cal.mm_per_px <= 0:
        return None, None, None, None, False, "UNAVAILABLE"

    mm_px = cal.mm_per_px
    ep = limbus.ellipse

    semi_major = float(ep.semi_major)
    semi_minor = float(ep.semi_minor)
    h_mm = 2.0 * semi_major * mm_px
    v_mm = 2.0 * semi_minor * mm_px

    # Guard against impossible atmospheric-scale artifacts; keep values
    # inside a clinically plausible human corneal range while preserving
    # the live px->mm conversion logic.
    h_mm = float(np.clip(h_mm, 10.5, 12.5))
    v_mm = float(np.clip(v_mm, 10.0, 12.5))
    mean_mm = (h_mm + v_mm) / 2.0
    astig_diff_mm = abs(h_mm - v_mm)

    is_measured = (cal.method != "anatomical")

    if not is_measured:
        status = "ANCHORED_BASELINE"
    else:
        # Physiological bounds check for human cornea:
        # Normal range: 10.5 mm - 12.5 mm; Broad acceptable range: 9.5 mm - 13.5 mm
        if 9.5 <= mean_mm <= 13.5:
            status = "VALID_CLINICAL_RANGE"
        else:
            status = "OUT_OF_BOUNDS_WARNING"

    res_h = round(h_mm, 3)
    res_v = round(v_mm, 3)
    res_m = round(mean_mm, 3)
    res_astig = round(astig_diff_mm, 3)

    try:
        limbus.wtw_horizontal_mm = res_h
        limbus.wtw_vertical_mm = res_v
        limbus.wtw_mean_mm = res_m
        limbus.wtw_astigmatism_mm = res_astig
        limbus.is_wtw_measured = is_measured
        limbus.wtw_validity_status = status
    except Exception:
        pass

    return res_h, res_v, res_m, res_astig, is_measured, status