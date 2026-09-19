"""
Unified type definitions for surgical-grade pupil & limbus tracking.

This module is the SINGLE source of truth for all data structures.
No other module should define its own result types, uncertainty formats,
grade thresholds, or anatomical limits.

Every measurement includes uncertainty. Every result includes a quality grade.
No number is returned without confidence context.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class DetectionQuality(Enum):
    """Quality grade for a detection result.

    Grade thresholds (confidence → grade) are defined ONCE in
    ``assign_quality_grade`` below.  No other module may define its
    own thresholds.
    """
    SURGICAL = "SURGICAL"
    CLINICAL = "CLINICAL"
    RESEARCH = "RESEARCH"
    INSUFFICIENT = "INSUFFICIENT"
    NO_DETECTION = "NO_DETECTION"


class DetectionMethod(Enum):
    """Which pipeline produced the detection."""
    ML = "ML"
    CLASSICAL = "CLASSICAL"
    HYBRID = "HYBRID"
    KALMAN = "KALMAN"
    CARRY_FORWARD = "CARRY_FORWARD"


class QualityFlag(Enum):
    """Simplified quality flag for UI display."""
    GOOD = "GOOD"
    MARGINAL = "MARGINAL"
    POOR = "POOR"
    NO_DETECTION = "NO_DETECTION"


# ──────────────────────────────────────────────────────────────────────
# Grade assignment — SINGLE SOURCE OF TRUTH
# ──────────────────────────────────────────────────────────────────────

_GRADE_THRESHOLDS: List[Tuple[float, DetectionQuality]] = [
    (0.75, DetectionQuality.SURGICAL),
    (0.55, DetectionQuality.CLINICAL),
    (0.30, DetectionQuality.RESEARCH),
    (0.0, DetectionQuality.INSUFFICIENT),
]


def assign_quality_grade(confidence: float) -> DetectionQuality:
    """Assign a quality grade from a confidence score in [0, 1].

    Uses a fixed, ordered threshold table.  This function is the ONLY
    place where grade boundaries are defined.
    """
    if confidence is None or not math.isfinite(confidence):
        return DetectionQuality.NO_DETECTION
    for threshold, grade in _GRADE_THRESHOLDS:
        if confidence >= threshold:
            return grade
    return DetectionQuality.INSUFFICIENT


def quality_to_flag(quality: DetectionQuality) -> QualityFlag:
    _map = {
        DetectionQuality.SURGICAL: QualityFlag.GOOD,
        DetectionQuality.CLINICAL: QualityFlag.GOOD,
        DetectionQuality.RESEARCH: QualityFlag.MARGINAL,
        DetectionQuality.INSUFFICIENT: QualityFlag.POOR,
        DetectionQuality.NO_DETECTION: QualityFlag.NO_DETECTION,
    }
    return _map.get(quality, QualityFlag.NO_DETECTION)


# ──────────────────────────────────────────────────────────────────────
# Safe-conversion helpers
# ──────────────────────────────────────────────────────────────────────

def _sf(v: Any, default: float = 0.0) -> float:
    """Safe float — returns *default* for None / NaN / Inf / non-numeric."""
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _st(v: Any,
        default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """Safe tuple of two floats."""
    if v is None:
        return default
    try:
        t = tuple(v)
        if len(t) >= 2:
            x, y = float(t[0]), float(t[1])
            if math.isfinite(x) and math.isfinite(y):
                return (x, y)
    except (TypeError, ValueError):
        pass
    return default


# ──────────────────────────────────────────────────────────────────────
# Anatomical limits — SINGLE SOURCE OF TRUTH
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnatomicalLimits:
    """Hard anatomical constraints used for validation everywhere.

    These constants must NEVER be duplicated in other modules.
    Import ``ANATOMICAL_LIMITS`` instead.
    """
    # Pupil
    PUPIL_MIN_RADIUS_PX: float = 8.0
    PUPIL_MAX_RADIUS_PX: float = 120.0
    PUPIL_MIN_RADIUS_MM: float = 0.5
    PUPIL_MAX_RADIUS_MM: float = 5.0

    # Limbus
    LIMBUS_MIN_RADIUS_PX: float = 40.0
    LIMBUS_MAX_RADIUS_PX: float = 250.0
    LIMBUS_MIN_RADIUS_MM: float = 4.5
    LIMBUS_MAX_RADIUS_MM: float = 7.5

    # Ratio: pupil_radius / limbus_radius
    MIN_PUPIL_LIMBUS_RATIO: float = 0.15
    MAX_PUPIL_LIMBUS_RATIO: float = 0.75

    # Shape
    MIN_ASPECT_RATIO: float = 0.50          # semi_minor / semi_major
    MAX_ECCENTRICITY: float = 0.87
    MIN_CIRCULARITY: float = 0.50

    # Offset
    MAX_OFFSET_MM: float = 1.5
    WARN_OFFSET_MM: float = 0.8
    CRITICAL_OFFSET_MM: float = 1.2
    MAX_CENTER_OFFSET_RATIO: float = 0.35


ANATOMICAL_LIMITS = AnatomicalLimits()


# ──────────────────────────────────────────────────────────────────────
# Core measurement data-classes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EllipseParams:
    """Fitted ellipse with uncertainty.

    Convention: ``semi_major >= semi_minor`` always.
    ``angle_deg`` is in [0, 180), measured counter-clockwise from the
    positive x-axis to the major axis.

    NOTE: ``radius`` is a read-only property = (semi_major + semi_minor) / 2.
    To change the effective radius, scale ``semi_major`` and ``semi_minor``
    proportionally using ``set_radius()``.
    """
    center_x: float = 0.0
    center_y: float = 0.0
    semi_major: float = 0.0
    semi_minor: float = 0.0
    angle_deg: float = 0.0

    # 1-σ uncertainties
    uncertainty_center_x: float = 2.0
    uncertainty_center_y: float = 2.0
    uncertainty_semi_major: float = 1.5
    uncertainty_semi_minor: float = 1.5

    # Quality
    fit_quality: float = 0.0
    fit_rms_residual: float = 0.0
    num_contour_points: int = 0
    eccentricity: float = 0.0
    circularity: float = 1.0

    # ── derived properties ──────────────────────────────────────────

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    @property
    def radius(self) -> float:
        """Mean radius (average of semi-axes).  Read-only."""
        return (self.semi_major + self.semi_minor) / 2.0

    def set_radius(self, target_radius: float) -> None:
        """Scale semi_major and semi_minor proportionally to achieve
        the target mean radius, preserving the ellipse aspect ratio.

        If the current radius is near zero (degenerate), falls back
        to a circle with both semi-axes equal to *target_radius*.
        """
        current_r = self.radius
        if current_r > 1e-6:
            scale = target_radius / current_r
            self.semi_major *= scale
            self.semi_minor *= scale
        else:
            self.semi_major = target_radius
            self.semi_minor = target_radius

    @property
    def is_valid(self) -> bool:
        return (self.semi_major > 0
                and self.semi_minor > 0
                and self.semi_major >= self.semi_minor
                and math.isfinite(self.center_x)
                and math.isfinite(self.center_y))

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center_x": _sf(self.center_x),
            "center_y": _sf(self.center_y),
            "semi_major": _sf(self.semi_major),
            "semi_minor": _sf(self.semi_minor),
            "angle_deg": _sf(self.angle_deg),
            "radius": _sf(self.radius),
            "eccentricity": _sf(self.eccentricity),
            "circularity": _sf(self.circularity),
            "fit_quality": _sf(self.fit_quality),
            "fit_rms_residual": _sf(self.fit_rms_residual),
            "num_contour_points": self.num_contour_points,
            "uncertainty_center": (
                _sf(self.uncertainty_center_x),
                _sf(self.uncertainty_center_y),
            ),
            "uncertainty_radius": _sf(
                (self.uncertainty_semi_major
                 + self.uncertainty_semi_minor) / 2.0
            ),
        }


@dataclass
class PupilDetection:
    """Complete pupil detection result."""
    detected: bool = False
    ellipse: Optional[EllipseParams] = None
    confidence: float = 0.0
    quality: DetectionQuality = DetectionQuality.NO_DETECTION
    method: DetectionMethod = DetectionMethod.ML

    center_mm: Optional[Tuple[float, float]] = None
    radius_mm: Optional[float] = None
    contour_points: Optional[Any] = None      # np.ndarray kept opaque

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "detected": self.detected,
            "confidence": _sf(self.confidence),
            "quality": self.quality.value,
            "method": self.method.value,
        }
        if self.ellipse is not None:
            d["ellipse"] = self.ellipse.to_dict()
            d["center_px"] = (
                _sf(self.ellipse.center_x),
                _sf(self.ellipse.center_y),
            )
            d["radius_px"] = _sf(self.ellipse.radius)
        if self.center_mm is not None:
            d["center_mm"] = _st(self.center_mm)
        if self.radius_mm is not None:
            d["radius_mm"] = _sf(self.radius_mm)
        return d


@dataclass
class LimbusDetection:
    """Complete limbus detection result."""
    detected: bool = False
    ellipse: Optional[EllipseParams] = None
    confidence: float = 0.0
    quality: DetectionQuality = DetectionQuality.NO_DETECTION
    method: DetectionMethod = DetectionMethod.ML

    center_mm: Optional[Tuple[float, float]] = None
    radius_mm: Optional[float] = None
    contour_points: Optional[Any] = None

    # White-to-White (WTW) corneal diameter metrics
    wtw_horizontal_mm: Optional[float] = None
    wtw_vertical_mm: Optional[float] = None
    wtw_mean_mm: Optional[float] = None
    wtw_astigmatism_mm: Optional[float] = None
    is_wtw_measured: bool = False
    wtw_validity_status: str = "UNAVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "detected": self.detected,
            "confidence": _sf(self.confidence),
            "quality": self.quality.value,
            "method": self.method.value,
            "is_wtw_measured": self.is_wtw_measured,
            "wtw_validity_status": self.wtw_validity_status,
        }
        if self.ellipse is not None:
            d["ellipse"] = self.ellipse.to_dict()
            d["center_px"] = (
                _sf(self.ellipse.center_x),
                _sf(self.ellipse.center_y),
            )
            d["radius_px"] = _sf(self.ellipse.radius)
        if self.center_mm is not None:
            d["center_mm"] = _st(self.center_mm)
        if self.radius_mm is not None:
            d["radius_mm"] = _sf(self.radius_mm)
        if self.wtw_horizontal_mm is not None:
            d["wtw_horizontal_mm"] = _sf(self.wtw_horizontal_mm)
        if self.wtw_vertical_mm is not None:
            d["wtw_vertical_mm"] = _sf(self.wtw_vertical_mm)
        if self.wtw_mean_mm is not None:
            d["wtw_mean_mm"] = _sf(self.wtw_mean_mm)
        if self.wtw_astigmatism_mm is not None:
            d["wtw_astigmatism_mm"] = _sf(self.wtw_astigmatism_mm)
        return d



@dataclass
class CornealCenterResult:
    """Corneal centre + pupil-limbus offset."""
    valid: bool = False

    center_px: Tuple[float, float] = (0.0, 0.0)
    center_mm: Optional[Tuple[float, float]] = None

    offset_px: Tuple[float, float] = (0.0, 0.0)
    offset_mm: Optional[Tuple[float, float]] = None
    offset_magnitude_px: float = 0.0
    offset_magnitude_mm: Optional[float] = None
    offset_angle_deg: float = 0.0

    confidence: float = 0.0
    quality: DetectionQuality = DetectionQuality.NO_DETECTION
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "valid": self.valid,
            "center_px": _st(self.center_px),
            "offset_px": _st(self.offset_px),
            "offset_magnitude_px": _sf(self.offset_magnitude_px),
            "offset_angle_deg": _sf(self.offset_angle_deg),
            "confidence": _sf(self.confidence),
            "quality": self.quality.value,
        }
        if self.center_mm is not None:
            d["center_mm"] = _st(self.center_mm)
        if self.offset_mm is not None:
            d["offset_mm"] = _st(self.offset_mm)
        if self.offset_magnitude_mm is not None:
            d["offset_magnitude_mm"] = _sf(self.offset_magnitude_mm)
        if self.alerts:
            d["alerts"] = list(self.alerts)
        return d


@dataclass
class CalibrationInfo:
    """Pixel ↔ millimetre calibration."""
    calibrated: bool = False
    px_per_mm: float = 1.0
    mm_per_px: float = 1.0
    source: str = "none"
    method: str = "anatomical"  # "anatomical" | "fixed_manual" | "ring_reflection"
    reference_diameter_mm: float = 0.0
    reference_diameter_px: float = 0.0
    confidence: float = 0.0
    corneal_diameter_assumed_mm: Optional[float] = None
    mm_per_px_uncertainty: float = 0.0
    px_per_mm_std: float = 0.0

    def px_to_mm(self, px: float) -> float:
        return px * self.mm_per_px if self.calibrated else 0.0

    def point_px_to_mm(
        self,
        pt: Tuple[float, float],
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float]:
        if not self.calibrated:
            return (0.0, 0.0)
        return (
            (pt[0] - origin[0]) * self.mm_per_px,
            (pt[1] - origin[1]) * self.mm_per_px,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "px_per_mm": _sf(self.px_per_mm),
            "mm_per_px": _sf(self.mm_per_px),
            "source": self.source,
            "method": self.method,
            "reference_diameter_mm": _sf(self.reference_diameter_mm),
            "reference_diameter_px": _sf(self.reference_diameter_px),
            "confidence": _sf(self.confidence),
            "corneal_diameter_assumed_mm": _sf(self.corneal_diameter_assumed_mm) if self.corneal_diameter_assumed_mm is not None else None,
        }



@dataclass
class FrameMetadata:
    """Per-frame bookkeeping."""
    timestamp: float = field(default_factory=time.time)
    frame_number: int = -1
    source: str = ""
    source_path: str = ""
    image_width: int = 0
    image_height: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Top-level result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EyeDetectionResult:
    """Complete eye-detection result.

    This is the **only** result type that flows through the system.
    All detectors, video processors, GUIs, and APIs consume this
    single type.
    """
    pupil: PupilDetection = field(default_factory=PupilDetection)
    limbus: LimbusDetection = field(default_factory=LimbusDetection)
    corneal_center: CornealCenterResult = field(
        default_factory=CornealCenterResult
    )
    calibration: CalibrationInfo = field(default_factory=CalibrationInfo)
    metadata: FrameMetadata = field(default_factory=FrameMetadata)

    overall_quality: DetectionQuality = DetectionQuality.NO_DETECTION
    overall_confidence: float = 0.0
    alerts: List[str] = field(default_factory=list)

    # ── convenience flags ───────────────────────────────────────────

    @property
    def has_pupil(self) -> bool:
        return self.pupil.detected

    @property
    def has_limbus(self) -> bool:
        return self.limbus.detected

    @property
    def has_both(self) -> bool:
        return self.has_pupil and self.has_limbus

    @property
    def has_corneal_center(self) -> bool:
        return self.corneal_center.valid

    # ── serialisation ───────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "pupil": self.pupil.to_dict(),
            "limbus": self.limbus.to_dict(),
            "corneal_center": self.corneal_center.to_dict(),
            "calibration": self.calibration.to_dict(),
            "metadata": self.metadata.to_dict(),
            "overall_quality": self.overall_quality.value,
            "overall_confidence": _sf(self.overall_confidence),
            "alerts": list(self.alerts),
        }
        for attr in [
            "ring_status",
            "ring_confidence",
            "ring_center",
            "ring_radius",
            "ring_dot_count",
            "ring_method",
            "image_category",
            "corneal_reference_source",
        ]:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)
        return data

    # ── dict ↔ result conversion for smoother / CSV pipelines ───────

    @staticmethod
    def result_to_dict(result: "EyeDetectionResult") -> Dict[str, Any]:
        """Flatten an EyeDetectionResult into a dict for the smoother / CSV.

        Keys use the ``pupil_x``, ``pupil_y``, ``pupil_r`` convention
        expected by ``TemporalSmoother``.
        """
        d: Dict[str, Any] = {
            "pupil_detected": result.pupil.detected,
            "pupil_confidence": result.pupil.confidence,
            "limbus_detected": result.limbus.detected,
            "limbus_confidence": result.limbus.confidence,
            "overall_quality": result.overall_quality.value,
            "overall_confidence": result.overall_confidence,
            "processing_time_ms": result.metadata.processing_time_ms,
        }

        # ── pupil geometry ──────────────────────────────────────
        if result.pupil.detected and result.pupil.ellipse is not None:
            pe = result.pupil.ellipse
            d.update({
                "pupil_x": pe.center_x,
                "pupil_y": pe.center_y,
                "pupil_r": pe.radius,
                "pupil_semi_major": pe.semi_major,
                "pupil_semi_minor": pe.semi_minor,
                "pupil_angle_deg": pe.angle_deg,
                "pupil_fit_quality": pe.fit_quality,
                "pupil_eccentricity": pe.eccentricity,
                "pupil_circularity": pe.circularity,
                "pupil_method": result.pupil.method.value,
            })
            if result.pupil.radius_mm is not None:
                d["pupil_radius_mm"] = result.pupil.radius_mm
            if result.pupil.center_mm is not None:
                d["pupil_center_mm_x"] = result.pupil.center_mm[0]
                d["pupil_center_mm_y"] = result.pupil.center_mm[1]
        else:
            d.update({
                "pupil_x": 0.0, "pupil_y": 0.0, "pupil_r": 0.0,
                "pupil_semi_major": 0.0, "pupil_semi_minor": 0.0,
                "pupil_angle_deg": 0.0, "pupil_fit_quality": 0.0,
                "pupil_eccentricity": 0.0, "pupil_circularity": 0.0,
                "pupil_method": DetectionMethod.ML.value,
            })

        # ── limbus geometry ─────────────────────────────────────
        if result.limbus.detected and result.limbus.ellipse is not None:
            le = result.limbus.ellipse
            d.update({
                "limbus_x": le.center_x,
                "limbus_y": le.center_y,
                "limbus_r": le.radius,
                "limbus_semi_major": le.semi_major,
                "limbus_semi_minor": le.semi_minor,
                "limbus_angle_deg": le.angle_deg,
                "limbus_fit_quality": le.fit_quality,
                "limbus_eccentricity": le.eccentricity,
                "limbus_circularity": le.circularity,
                "limbus_method": result.limbus.method.value,
            })
            if result.limbus.radius_mm is not None:
                d["limbus_radius_mm"] = result.limbus.radius_mm
            if result.limbus.center_mm is not None:
                d["limbus_center_mm_x"] = result.limbus.center_mm[0]
                d["limbus_center_mm_y"] = result.limbus.center_mm[1]
        else:
            d.update({
                "limbus_x": 0.0, "limbus_y": 0.0, "limbus_r": 0.0,
                "limbus_semi_major": 0.0, "limbus_semi_minor": 0.0,
                "limbus_angle_deg": 0.0, "limbus_fit_quality": 0.0,
                "limbus_eccentricity": 0.0, "limbus_circularity": 0.0,
                "limbus_method": DetectionMethod.ML.value,
            })

        # ── corneal centre ──────────────────────────────────────
        if result.corneal_center.valid:
            cc = result.corneal_center
            d.update({
                "corneal_center_x": cc.center_px[0],
                "corneal_center_y": cc.center_px[1],
                "corneal_offset_x": cc.offset_px[0],
                "corneal_offset_y": cc.offset_px[1],
                "corneal_offset_mag_px": cc.offset_magnitude_px,
                "corneal_offset_angle_deg": cc.offset_angle_deg,
            })
            if cc.offset_magnitude_mm is not None:
                d["corneal_offset_mag_mm"] = cc.offset_magnitude_mm

        # ── calibration ─────────────────────────────────────────
        d["calibrated"] = result.calibration.calibrated
        if result.calibration.calibrated:
            d["px_per_mm"] = result.calibration.px_per_mm

        return d

    @staticmethod
    def apply_smoothed_dict(
        result: "EyeDetectionResult",
        smoothed: Dict[str, Any],
    ) -> "EyeDetectionResult":
        """Write Kalman-smoothed values back into a typed result.

        Recalculates mm measurements and corneal centre after updating
        pixel coordinates.  Modifies *result* in-place and returns it.

        IMPORTANT: ``EllipseParams.radius`` is a read-only property.
        We scale ``semi_major`` and ``semi_minor`` proportionally via
        ``EllipseParams.set_radius()`` to achieve the target smoothed
        radius while preserving the ellipse aspect ratio.
        """
        # ── pupil ───────────────────────────────────────────────
        if smoothed.get("pupil_detected") and result.pupil.ellipse is not None:
            ell = result.pupil.ellipse
            ell.center_x = float(smoothed.get("pupil_x", ell.center_x))
            ell.center_y = float(smoothed.get("pupil_y", ell.center_y))

            target_r = float(smoothed.get("pupil_r", ell.radius))
            ell.set_radius(target_r)

        # ── limbus ──────────────────────────────────────────────
        if smoothed.get("limbus_detected") and result.limbus.ellipse is not None:
            ell = result.limbus.ellipse
            ell.center_x = float(smoothed.get("limbus_x", ell.center_x))
            ell.center_y = float(smoothed.get("limbus_y", ell.center_y))

            target_r = float(smoothed.get("limbus_r", ell.radius))
            ell.set_radius(target_r)

        # ── recalculate mm values if calibrated ─────────────────
        cal = result.calibration
        if cal.calibrated:
            if result.pupil.detected and result.pupil.ellipse is not None:
                result.pupil.radius_mm = cal.px_to_mm(
                    result.pupil.ellipse.radius
                )
                result.pupil.center_mm = cal.point_px_to_mm(
                    result.pupil.ellipse.center
                )
            if result.limbus.detected and result.limbus.ellipse is not None:
                result.limbus.radius_mm = cal.px_to_mm(
                    result.limbus.ellipse.radius
                )
                result.limbus.center_mm = cal.point_px_to_mm(
                    result.limbus.ellipse.center
                )

        # ── recalculate corneal centre ──────────────────────────
        if result.has_both:
            p = result.pupil.ellipse
            l = result.limbus.ellipse
            # Corneal centre = midpoint between pupil and limbus centres
            cx = (p.center_x + l.center_x) / 2.0
            cy = (p.center_y + l.center_y) / 2.0
            # Offset = pupil centre − limbus centre
            ox = p.center_x - l.center_x
            oy = p.center_y - l.center_y
            mag = math.sqrt(ox * ox + oy * oy)
            ang = math.degrees(math.atan2(oy, ox)) % 360.0

            cc = result.corneal_center
            cc.valid = True
            cc.center_px = (cx, cy)
            cc.offset_px = (ox, oy)
            cc.offset_magnitude_px = mag
            cc.offset_angle_deg = ang

            if cal.calibrated:
                cc.center_mm = cal.point_px_to_mm((cx, cy))
                cc.offset_mm = (
                    cal.px_to_mm(ox),
                    cal.px_to_mm(oy),
                )
                cc.offset_magnitude_mm = cal.px_to_mm(mag)

        return result


# ──────────────────────────────────────────────────────────────────────
# Geometric fitting result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """Result from geometric ellipse / circle fitting.

    Returned by ``EllipseFitter.fit()``.  **Replaces** the old
    dict-based return that was missing a ``valid`` key, causing the
    GeometricFitter integration to be dead code in three separate
    modules.

    NOTE: This uses ``fit_quality_score`` (not ``fit_quality``) to
    avoid collision with ``EllipseParams.fit_quality``.  The mapping
    is handled in ``to_ellipse_params()``.
    """
    valid: bool = False

    center_x: float = 0.0
    center_y: float = 0.0
    semi_major: float = 0.0
    semi_minor: float = 0.0
    angle_deg: float = 0.0
    radius: float = 0.0
    eccentricity: float = 0.0

    fit_quality_score: float = 0.0
    rms_residual: float = 0.0
    num_inliers: int = 0
    num_points: int = 0
    method: str = "unknown"

    uncertainty_center: Tuple[float, float] = (2.0, 2.0)
    uncertainty_radius: float = 1.5

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    def to_ellipse_params(self) -> EllipseParams:
        """Promote to an ``EllipseParams`` for inclusion in a detection
        result.

        Maps ``fit_quality_score`` → ``fit_quality``.
        """
        circ = (self.semi_minor / self.semi_major
                if self.semi_major > 0 else 0.0)
        return EllipseParams(
            center_x=self.center_x,
            center_y=self.center_y,
            semi_major=self.semi_major,
            semi_minor=self.semi_minor,
            angle_deg=self.angle_deg,
            uncertainty_center_x=self.uncertainty_center[0],
            uncertainty_center_y=self.uncertainty_center[1],
            uncertainty_semi_major=self.uncertainty_radius,
            uncertainty_semi_minor=self.uncertainty_radius,
            fit_quality=self.fit_quality_score,
            fit_rms_residual=self.rms_residual,
            num_contour_points=self.num_points,
            eccentricity=self.eccentricity,
            circularity=circ,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "center_px": (_sf(self.center_x), _sf(self.center_y)),
            "semi_major_px": _sf(self.semi_major),
            "semi_minor_px": _sf(self.semi_minor),
            "angle_deg": _sf(self.angle_deg),
            "radius_px": _sf(self.radius),
            "eccentricity": _sf(self.eccentricity),
            "fit_quality_score": _sf(self.fit_quality_score),
            "rms_residual": _sf(self.rms_residual),
            "num_inliers": self.num_inliers,
            "num_points": self.num_points,
            "method": self.method,
        }


# ──────────────────────────────────────────────────────────────────────
# Registration / Cyclotorsion types
# ──────────────────────────────────────────────────────────────────────

class StreamName(Enum):
    """Cyclotorsion detection stream identifiers."""
    PHASE_CORRELATION = "phase_correlation"
    DEEP_MATCHER = "deep_matcher"
    INK_MARKERS = "ink_markers"
    LIMBAL_VESSELS = "limbal_vessels"
    CUSTOM_FEATURE = "custom_feature"


@dataclass
class StreamResult:
    """Result from a single cyclotorsion detection stream.

    Every stream produces a torsion angle (degrees), a confidence
    score, and optional diagnostic metadata.
    """
    stream: StreamName = StreamName.PHASE_CORRELATION
    torsion_deg: Optional[float] = None
    confidence: float = 0.0
    inlier_count: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return (self.torsion_deg is not None
                and math.isfinite(self.torsion_deg)
                and self.confidence > 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream.value,
            "torsion_deg": _sf(self.torsion_deg) if self.torsion_deg is not None else None,
            "confidence": _sf(self.confidence),
            "inlier_count": self.inlier_count,
            "processing_time_ms": _sf(self.processing_time_ms),
            "valid": self.valid,
            "metadata": dict(self.metadata),
        }


class RegistrationQuality(Enum):
    """Quality grade for a registration/cyclotorsion result."""
    SURGICAL = "SURGICAL"        # ±0.1° accuracy, confidence ≥ 0.85
    CLINICAL = "CLINICAL"        # ±0.5° accuracy, confidence ≥ 0.60
    RESEARCH = "RESEARCH"        # ±2.0° accuracy, confidence ≥ 0.30
    INSUFFICIENT = "INSUFFICIENT"
    NO_RESULT = "NO_RESULT"


_REG_GRADE_THRESHOLDS: List[Tuple[float, RegistrationQuality]] = [
    (0.85, RegistrationQuality.SURGICAL),
    (0.60, RegistrationQuality.CLINICAL),
    (0.30, RegistrationQuality.RESEARCH),
    (0.0,  RegistrationQuality.INSUFFICIENT),
]


def assign_registration_grade(confidence: float) -> RegistrationQuality:
    """Assign a registration quality grade from a fused confidence."""
    if confidence is None or not math.isfinite(confidence):
        return RegistrationQuality.NO_RESULT
    for threshold, grade in _REG_GRADE_THRESHOLDS:
        if confidence >= threshold:
            return grade
    return RegistrationQuality.INSUFFICIENT


@dataclass
class RegistrationResult:
    """Fused cyclotorsion result from all active streams.

    This is the top-level output of the registration pipeline,
    analogous to EyeDetectionResult for the detection pipeline.
    """
    valid: bool = False
    torsion_deg: float = 0.0
    confidence: float = 0.0
    quality: RegistrationQuality = RegistrationQuality.NO_RESULT

    # Per-stream breakdown
    stream_results: Dict[str, StreamResult] = field(default_factory=dict)
    active_streams: int = 0
    agreeing_streams: int = 0

    # Uncertainty
    torsion_std_deg: float = 0.0
    confidence_interval_deg: Tuple[float, float] = (0.0, 0.0)

    # Timing
    total_processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "torsion_deg": _sf(self.torsion_deg),
            "confidence": _sf(self.confidence),
            "quality": self.quality.value,
            "active_streams": self.active_streams,
            "agreeing_streams": self.agreeing_streams,
            "torsion_std_deg": _sf(self.torsion_std_deg),
            "confidence_interval_deg": (
                _sf(self.confidence_interval_deg[0]),
                _sf(self.confidence_interval_deg[1]),
            ),
            "total_processing_time_ms": _sf(self.total_processing_time_ms),
            "stream_results": {
                k: v.to_dict() for k, v in self.stream_results.items()
            },
        }
