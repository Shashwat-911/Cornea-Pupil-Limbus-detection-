"""Pentacam detection result types.

This module defines the data contract for Pentacam image detection.
It is intentionally minimal and isolated from the ELITA iris pipeline.

All types are additive and do not modify existing detection results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.utils.types import EllipseParams


class PentacamImageType(Enum):
    """Classification of Pentacam image content."""
    SCHEIMPFLUG_CROSS_SECTION = "SCHEIMPFLUG_CROSS_SECTION"
    ANTERIOR_SEGMENT = "ANTERIOR_SEGMENT"
    CORNEAL_MAP = "CORNEAL_MAP"
    UNKNOWN = "UNKNOWN"


class PentacamDetectionStatus(Enum):
    """Overall status of Pentacam detection."""
    OK = "OK"
    NO_IMAGE = "NO_IMAGE"
    NO_PUPIL = "NO_PUPIL"
    NO_LIMBUS = "NO_LIMBUS"
    INSUFFICIENT_FEATURES = "INSUFFICIENT_FEATURES"
    DEGENERATE = "DEGENERATE"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class PentacamQuality(Enum):
    """Quality grade for Pentacam detection."""
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    MARGINAL = "MARGINAL"
    POOR = "POOR"
    NO_DETECTION = "NO_DETECTION"


@dataclass
class PentacamGeometry:
    """Detected anatomical geometry from a Pentacam image.

    Coordinates are in Pentacam image pixel space.
    The coordinate system is device-specific and may differ from ELITA.
    """
    pupil: Optional[EllipseParams] = None
    limbus: Optional[EllipseParams] = None

    pupil_detected: bool = False
    limbus_detected: bool = False

    pupil_radius_px: float = 0.0
    limbus_radius_px: float = 0.0
    pupil_limbus_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "pupil_detected": self.pupil_detected,
            "limbus_detected": self.limbus_detected,
            "pupil_radius_px": round(self.pupil_radius_px, 2),
            "limbus_radius_px": round(self.limbus_radius_px, 2),
            "pupil_limbus_ratio": round(self.pupil_limbus_ratio, 4),
        }
        if self.pupil is not None:
            d["pupil"] = self.pupil.to_dict()
        if self.limbus is not None:
            d["limbus"] = self.limbus.to_dict()
        return d


@dataclass
class PentacamFeature:
    """A single detected feature from a Pentacam image.

    Features are in Pentacam image pixel coordinates.
    The coordinate system is device-specific.
    """
    id: int = -1
    x: float = 0.0
    y: float = 0.0
    angle_deg: float = 0.0
    radial_norm: float = 0.5
    response: float = 0.0
    confidence: float = 0.0
    valid: bool = True
    descriptor: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "angle_deg": float(self.angle_deg),
            "radial_norm": float(self.radial_norm),
            "response": float(self.response),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "descriptor_len": (
                len(self.descriptor) if self.descriptor is not None else 0
            ),
        }


@dataclass
class PentacamFeatureSet:
    """Container for features extracted from a Pentacam image."""
    features: List[PentacamFeature] = field(default_factory=list)
    num_candidates: int = 0
    num_accepted: int = 0
    angular_coverage_ratio: float = 0.0
    largest_angular_gap_deg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": [f.to_dict() for f in self.features],
            "num_candidates": self.num_candidates,
            "num_accepted": self.num_accepted,
            "angular_coverage_ratio": round(self.angular_coverage_ratio, 4),
            "largest_angular_gap_deg": round(self.largest_angular_gap_deg, 2),
        }


@dataclass
class PentacamDetectionResult:
    """Top-level result of Pentacam image detection.

    This is the primary object a future cross-system matcher will consume.
    It carries Pentacam-side geometry, features, and quality assessment.
    """
    valid: bool = False
    status: PentacamDetectionStatus = PentacamDetectionStatus.NO_IMAGE
    image_type: PentacamImageType = PentacamImageType.UNKNOWN

    geometry: PentacamGeometry = field(default_factory=PentacamGeometry)
    feature_set: PentacamFeatureSet = field(default_factory=PentacamFeatureSet)

    image_width: int = 0
    image_height: int = 0
    coordinate_system: str = "pentacam_pixel"

    quality: PentacamQuality = PentacamQuality.NO_DETECTION
    confidence: float = 0.0
    failure_reason: str = ""

    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "image_type": self.image_type.value,
            "geometry": self.geometry.to_dict(),
            "feature_set": self.feature_set.to_dict(),
            "image_width": self.image_width,
            "image_height": self.image_height,
            "coordinate_system": self.coordinate_system,
            "quality": self.quality.value,
            "confidence": round(self.confidence, 4),
            "failure_reason": self.failure_reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
