"""Type definitions for the iris-feature concept model.

PHASE I scope
-------------
This phase implements **iris feature detection only**. It defines the data
contract that the (future) feature-matching, registration, and cyclotorsion
stages will consume, but it does **not** implement any matching, registration,
rotation estimation, or cyclotorsion logic.

The contract is intentionally kept small and extensible so a later phase can
build correspondence on top of ``IrisFeatureSet`` without changing the way
features are extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class IrisFeatureType(Enum):
    """Category of an extracted iris feature.

    The classical baseline labels every extracted feature with a coarse
    anatomical category derived from its local structure, so downstream
    consumers can reason about feature kind without re-deriving it.
    """
    TEXTURE = "TEXTURE"          # generic local texture patch
    CRYPT = "CRYPT"              # dark, locally-contrasty pit-like structure
    FURROW = "FURROW"            # elongated groove-like structure
    UNKNOWN = "UNKNOWN"          # could not be classified

    @classmethod
    def from_name(cls, name: Optional[str]) -> "IrisFeatureType":
        if not name:
            return cls.UNKNOWN
        try:
            return cls(name.upper())
        except ValueError:
            return cls.UNKNOWN


class IrisStatus(Enum):
    """Overall status of the iris detection run."""
    OK = "OK"
    NO_ROI = "NO_ROI"            # pupil/limbus geometry insufficient
    NO_FEATURES = "NO_FEATURES"  # ROI valid but no features passed filtering


@dataclass
class IrisROI:
    """Annular iris region of interest derived from pupil/limbus geometry.

    Coordinates are in source-image pixel space. The ROI is the ring of iris
    tissue between the pupil ellipse and the limbus ellipse, optionally
    contracted by ``inner_inset_frac`` / ``outer_inset_frac`` fractions of the
    local pupil/limbus radius to avoid pupil-proximal and limbus-ambiguous
    pixels.
    """
    valid: bool = False
    reason: str = ""

    center_x: float = 0.0
    center_y: float = 0.0
    pupil_semi_major: float = 0.0
    pupil_semi_minor: float = 0.0
    pupil_angle_deg: float = 0.0
    limbus_semi_major: float = 0.0
    limbus_semi_minor: float = 0.0
    limbus_angle_deg: float = 0.0

    inner_inset_frac: float = 0.10
    outer_inset_frac: float = 0.10

    pupil_radius_px: float = 0.0     # mean pupil semi-axis at center angle
    limbus_radius_px: float = 0.0    # mean limbus semi-axis at center angle

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "pupil_semi_major": float(self.pupil_semi_major),
            "pupil_semi_minor": float(self.pupil_semi_minor),
            "pupil_angle_deg": float(self.pupil_angle_deg),
            "limbus_semi_major": float(self.limbus_semi_major),
            "limbus_semi_minor": float(self.limbus_semi_minor),
            "limbus_angle_deg": float(self.limbus_angle_deg),
            "inner_inset_frac": float(self.inner_inset_frac),
            "outer_inset_frac": float(self.outer_inset_frac),
            "pupil_radius_px": float(self.pupil_radius_px),
            "limbus_radius_px": float(self.limbus_radius_px),
        }


@dataclass
class IrisFeature:
    """A single detected iris feature candidate.

    Both raw image-pixel coordinates and iris-relative coordinates are
    retained. ``radial_norm`` is a value in (0, 1] measured from the pupil
    boundary toward the limbus boundary; ``angle_deg`` is measured
    counter-clockwise from the positive x-axis in the image plane (matching
    the existing ellipse angle convention).

    The descriptor is a small, deterministic patch-based vector so that later
    phases can compare features without re-extracting.
    """
    id: int = -1

    # image-pixel position
    x: float = 0.0
    y: float = 0.0

    # iris-relative coordinates
    angle_deg: float = 0.0
    radial_norm: float = 0.5

    # feature characteristics
    scale: float = 1.0
    orientation_deg: float = 0.0
    feature_type: IrisFeatureType = IrisFeatureType.TEXTURE

    # strength / quality
    response: float = 0.0
    local_contrast: float = 0.0
    visibility: float = 1.0
    confidence: float = 0.0
    valid: bool = True

    # deterministic local descriptor (np.ndarray kept opaque)
    # When CNN encoding is active, this is a 128-d L2-normalized vector
    # instead of the classical 16-bin histogram.
    descriptor: Optional[np.ndarray] = None

    # CNN embedding flag (True when descriptor comes from CNN encoder)
    cnn_descriptor: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "x": float(self.x),
            "y": float(self.y),
            "angle_deg": float(self.angle_deg),
            "radial_norm": float(self.radial_norm),
            "scale": float(self.scale),
            "orientation_deg": float(self.orientation_deg),
            "feature_type": self.feature_type.value,
            "response": float(self.response),
            "local_contrast": float(self.local_contrast),
            "visibility": float(self.visibility),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "descriptor_len": (
                len(self.descriptor) if self.descriptor is not None else 0
            ),
            "cnn_descriptor": bool(self.cnn_descriptor),
        }
        return d


@dataclass
class IrisFeatureSet:
    """Container for all features extracted from one image.

    This is the primary object a future registration/matching stage will
    consume: it carries the reference iris geometry and every feature with
    both pixel and iris-relative coordinates plus confidence.
    """
    roi: IrisROI = field(default_factory=IrisROI)
    features: List[IrisFeature] = field(default_factory=list)

    num_candidates: int = 0
    num_accepted: int = 0
    region_coverage: float = 0.0        # fraction of usable iris area covered
    usable_fraction: float = 0.0        # fraction of annulus not occluded/reflective
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roi": self.roi.to_dict(),
            "features": [f.to_dict() for f in self.features],
            "num_candidates": self.num_candidates,
            "num_accepted": self.num_accepted,
            "region_coverage": float(self.region_coverage),
            "usable_fraction": float(self.usable_fraction),
            "rejection_reasons": dict(self.rejection_reasons),
        }


@dataclass
class IrisDetectionResult:
    """Top-level result of a Phase I iris-feature detection run."""
    valid: bool = False
    status: IrisStatus = IrisStatus.NO_ROI
    feature_set: IrisFeatureSet = field(default_factory=IrisFeatureSet)

    mask_stats: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "feature_set": self.feature_set.to_dict(),
            "mask_stats": dict(self.mask_stats),
            "processing_time_ms": float(self.processing_time_ms),
        }
