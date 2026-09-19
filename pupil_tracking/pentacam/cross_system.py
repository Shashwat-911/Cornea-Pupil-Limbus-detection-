"""Cross-system registration types for Pentacam ↔ ELITA correspondence.

This module defines the data contract for cross-system registration between
Pentacam sitting images and ELITA supine images. It is ADDITIVE and ISOLATED.

No existing types are modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pupil_tracking.pentacam.types import PentacamDetectionResult
from pupil_tracking.iris.types import IrisDetectionResult
from pupil_tracking.iris.correspondence import CorrespondenceResult


class RegistrationFailureKind(Enum):
    """Why cross-system registration failed or was refused."""
    OK = "OK"
    NO_PENTACAM = "NO_PENTACAM"
    NO_ELITA = "NO_ELITA"
    INSUFFICIENT_PENTACAM_FEATURES = "INSUFFICIENT_PENTACAM_FEATURES"
    INSUFFICIENT_ELITA_FEATURES = "INSUFFICIENT_ELITA_FEATURES"
    WEAK_CORRESPONDENCE = "WEAK_CORRESPONDENCE"
    AMBIGUOUS_CORRESPONDENCE = "AMBIGUOUS_CORRESPONDENCE"
    INCONSISTENT_TRANSFORMATION = "INCONSISTENT_TRANSFORMATION"
    EXCESSIVE_RESIDUAL = "EXCESSIVE_RESIDUAL"
    COORDINATE_MISMATCH = "COORDINATE_MISMATCH"
    IMAGE_QUALITY_FAILURE = "IMAGE_QUALITY_FAILURE"
    MISSING_METADATA = "MISSING_METADATA"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class TransformationModel(Enum):
    """Type of geometric transformation estimated."""
    RIGID_2D = "RIGID_2D"              # rotation + translation
    SIMILARITY_2D = "SIMILARITY_2D"    # rotation + translation + uniform scale
    AFFINE_2D = "AFFINE_2D"            # full affine (6 parameters)
    NONE = "NONE"                       # no transformation estimated


@dataclass
class CrossSystemRegistrationInput:
    """Input to the cross-system registration algorithm.

    This is the minimal set of information required to attempt
    Pentacam ↔ ELITA correspondence.
    """
    # Pentacam side
    pentacam: Optional[PentacamDetectionResult] = None

    # ELITA side
    elita_supine: Optional[IrisDetectionResult] = None
    elita_cyclotorsion: Optional[CorrespondenceResult] = None

    # Metadata
    pentacam_coordinate_system: str = "pentacam_pixel"
    elita_coordinate_system: str = "elita_pixel"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_pentacam": self.pentacam is not None and self.pentacam.valid,
            "has_elita_supine": self.elita_supine is not None and self.elita_supine.valid,
            "has_elita_cyclotorsion": (
                self.elita_cyclotorsion is not None
                and self.elita_cyclotorsion.valid
            ),
            "pentacam_coordinate_system": self.pentacam_coordinate_system,
            "elita_coordinate_system": self.elita_coordinate_system,
        }


@dataclass
class CrossSystemRegistrationResult:
    """Output of cross-system registration.

    Contains the estimated transformation between Pentacam and ELITA
    coordinate systems, along with confidence and quality metrics.
    """
    valid: bool = False
    failure: RegistrationFailureKind = RegistrationFailureKind.NO_PENTACAM
    failure_reason: str = ""

    # Transformation
    transformation_model: TransformationModel = TransformationModel.NONE
    rotation_deg: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    scale: float = 1.0
    transform_matrix: Optional[np.ndarray] = None  # 2x3 or 3x3

    # Composition with ELITA cyclotorsion
    # final_sitting_to_supine_deg = pentacam_to_elita_rotation + elita_cyclotorsion
    elita_cyclotorsion_deg: float = 0.0
    final_sitting_to_supine_deg: Optional[float] = None

    # Correspondence quality
    n_correspondences: int = 0
    n_inliers: int = 0
    inlier_fraction: float = 0.0
    residual_rms: float = 0.0
    residual_max: float = 0.0

    # Confidence
    confidence: float = 0.0
    quality_assessment: str = ""

    # Diagnostics
    pentacam_features_used: int = 0
    elita_features_used: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "valid": self.valid,
            "failure": self.failure.value,
            "failure_reason": self.failure_reason,
            "transformation_model": self.transformation_model.value,
            "rotation_deg": round(self.rotation_deg, 4),
            "translation_x": round(self.translation_x, 2),
            "translation_y": round(self.translation_y, 2),
            "scale": round(self.scale, 4),
            "elita_cyclotorsion_deg": round(self.elita_cyclotorsion_deg, 4),
            "final_sitting_to_supine_deg": (
                round(self.final_sitting_to_supine_deg, 4)
                if self.final_sitting_to_supine_deg is not None
                else None
            ),
            "n_correspondences": self.n_correspondences,
            "n_inliers": self.n_inliers,
            "inlier_fraction": round(self.inlier_fraction, 4),
            "residual_rms": round(self.residual_rms, 4),
            "residual_max": round(self.residual_max, 4),
            "confidence": round(self.confidence, 4),
            "quality_assessment": self.quality_assessment,
            "pentacam_features_used": self.pentacam_features_used,
            "elita_features_used": self.elita_features_used,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }
        if self.transform_matrix is not None:
            d["transform_matrix"] = self.transform_matrix.tolist()
        return d
