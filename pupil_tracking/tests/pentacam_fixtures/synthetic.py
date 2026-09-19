"""Synthetic test fixtures for Pentacam types.

ALL FIXTURES IN THIS MODULE ARE SYNTHETIC — NOT CLINICAL DATA.

These fixtures exist ONLY to verify:
- Result schema correctness
- Coordinate handling
- Deterministic execution
- Serialization round-trips

They must NEVER be used to claim:
- Pentacam detection accuracy
- Clinical performance
- Cross-system registration accuracy
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from pupil_tracking.pentacam.types import (
    PentacamDetectionResult,
    PentacamDetectionStatus,
    PentacamFeature,
    PentacamFeatureSet,
    PentacamGeometry,
    PentacamImageType,
    PentacamQuality,
)
from pupil_tracking.utils.types import EllipseParams


def make_synthetic_ellipse(
    center: Tuple[float, float] = (320.0, 240.0),
    radius: float = 50.0,
    eccentricity: float = 0.1,
) -> EllipseParams:
    """Create a synthetic ellipse for testing. SYNTHETIC — NOT CLINICAL."""
    semi_major = radius * (1.0 + eccentricity / 2.0)
    semi_minor = radius * (1.0 - eccentricity / 2.0)
    return EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_deg=0.0,
        fit_quality=0.95,
        fit_rms_residual=0.5,
        num_contour_points=36,
        eccentricity=eccentricity,
        circularity=1.0 - eccentricity,
    )


def make_synthetic_geometry(
    pupil_center: Tuple[float, float] = (320.0, 240.0),
    pupil_radius: float = 50.0,
    limbus_radius: float = 150.0,
) -> PentacamGeometry:
    """Create synthetic pupil/limbus geometry. SYNTHETIC — NOT CLINICAL."""
    pupil = make_synthetic_ellipse(pupil_center, pupil_radius, eccentricity=0.05)
    limbus = make_synthetic_ellipse(pupil_center, limbus_radius, eccentricity=0.08)
    return PentacamGeometry(
        pupil=pupil,
        limbus=limbus,
        pupil_detected=True,
        limbus_detected=True,
        pupil_radius_px=pupil_radius,
        limbus_radius_px=limbus_radius,
        pupil_limbus_ratio=pupil_radius / limbus_radius,
    )


def make_synthetic_features(
    center: Tuple[float, float] = (320.0, 240.0),
    pupil_radius: float = 50.0,
    limbus_radius: float = 150.0,
    num_features: int = 36,
) -> PentacamFeatureSet:
    """Create synthetic iris features in a regular lattice. SYNTHETIC — NOT CLINICAL."""
    features = []
    iris_radius = (pupil_radius + limbus_radius) / 2.0
    for i in range(num_features):
        angle_deg = i * (360.0 / num_features)
        angle_rad = math.radians(angle_deg)
        x = center[0] + iris_radius * math.cos(angle_rad)
        y = center[1] + iris_radius * math.sin(angle_rad)
        features.append(PentacamFeature(
            id=i,
            x=x,
            y=y,
            angle_deg=angle_deg,
            radial_norm=0.5,
            response=0.8,
            confidence=0.9,
            valid=True,
        ))

    angles = sorted(f.angle_deg for f in features)
    gaps = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
    gaps.append(360.0 - angles[-1] + angles[0])
    largest_gap = max(gaps)
    coverage = 1.0 - largest_gap / 360.0

    return PentacamFeatureSet(
        features=features,
        num_candidates=num_features,
        num_accepted=num_features,
        angular_coverage_ratio=coverage,
        largest_angular_gap_deg=largest_gap,
    )


def make_synthetic_detection_result(
    image_width: int = 640,
    image_height: int = 480,
    num_features: int = 36,
) -> PentacamDetectionResult:
    """Create a complete synthetic detection result. SYNTHETIC — NOT CLINICAL."""
    center = (image_width / 2.0, image_height / 2.0)
    geometry = make_synthetic_geometry(center)
    feature_set = make_synthetic_features(
        center,
        geometry.pupil_radius_px,
        geometry.limbus_radius_px,
        num_features,
    )
    return PentacamDetectionResult(
        valid=True,
        status=PentacamDetectionStatus.OK,
        image_type=PentacamImageType.SCHEIMPFLUG_CROSS_SECTION,
        geometry=geometry,
        feature_set=feature_set,
        image_width=image_width,
        image_height=image_height,
        coordinate_system="synthetic_test",
        quality=PentacamQuality.GOOD,
        confidence=0.9,
        failure_reason="",
        processing_time_ms=1.0,
    )
