"""Deterministic unit tests for Pentacam types and synthetic fixtures.

ALL TESTS USE SYNTHETIC DATA ONLY — NOT CLINICAL DATA.

These tests verify:
- Schema correctness
- Coordinate handling
- Deterministic execution
- Serialization round-trips
- Geometry validation
- Feature coverage computation
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

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

# ── Import synthetic fixtures ───────────────────────────────────────
from pupil_tracking.tests.pentacam_fixtures.synthetic import (
    make_synthetic_detection_result,
    make_synthetic_ellipse,
    make_synthetic_features,
    make_synthetic_geometry,
)


# ── EllipseParams ───────────────────────────────────────────────────

class TestEllipseParams:
    def test_basic_construction(self):
        e = EllipseParams(
            center_x=100.0, center_y=200.0,
            semi_major=50.0, semi_minor=45.0,
            angle_deg=10.0,
        )
        assert e.center == (100.0, 200.0)
        assert e.radius == 47.5
        assert e.is_valid is True

    def test_radius_property(self):
        e = EllipseParams(semi_major=60.0, semi_minor=40.0)
        assert e.radius == 50.0

    def test_set_radius(self):
        e = EllipseParams(semi_major=60.0, semi_minor=40.0)
        e.set_radius(100.0)
        assert e.radius == pytest.approx(100.0)
        assert e.semi_major > e.semi_minor

    def test_to_dict_round_trip(self):
        e = make_synthetic_ellipse()
        d = e.to_dict()
        assert "center_x" in d
        assert "semi_major" in d
        assert "radius" in d


# ── PentacamGeometry ────────────────────────────────────────────────

class TestPentacamGeometry:
    def test_synthetic_geometry(self):
        geom = make_synthetic_geometry()
        assert geom.pupil_detected is True
        assert geom.limbus_detected is True
        assert geom.pupil_radius_px == 50.0
        assert geom.limbus_radius_px == 150.0
        assert geom.pupil_limbus_ratio == pytest.approx(50.0 / 150.0)

    def test_geometry_to_dict(self):
        geom = make_synthetic_geometry()
        d = geom.to_dict()
        assert d["pupil_detected"] is True
        assert d["limbus_detected"] is True
        assert "pupil" in d
        assert "limbus" in d

    def test_geometry_without_pupil(self):
        geom = PentacamGeometry(pupil_detected=False, limbus_detected=True)
        d = geom.to_dict()
        assert d["pupil_detected"] is False
        assert "pupil" not in d


# ── PentacamFeature ─────────────────────────────────────────────────

class TestPentacamFeature:
    def test_feature_construction(self):
        f = PentacamFeature(id=0, x=100.0, y=200.0, angle_deg=45.0)
        assert f.id == 0
        assert f.x == 100.0
        assert f.valid is True

    def test_feature_to_dict(self):
        f = PentacamFeature(id=1, x=50.0, y=60.0, angle_deg=90.0)
        d = f.to_dict()
        assert d["id"] == 1
        assert d["x"] == 50.0


# ── PentacamFeatureSet ──────────────────────────────────────────────

class TestPentacamFeatureSet:
    def test_synthetic_features(self):
        fs = make_synthetic_features(num_features=36)
        assert len(fs.features) == 36
        assert fs.num_accepted == 36
        assert fs.angular_coverage_ratio > 0.9

    def test_feature_coverage_computation(self):
        fs = make_synthetic_features(num_features=72)
        assert fs.angular_coverage_ratio > 0.95

    def test_sparse_features(self):
        fs = make_synthetic_features(num_features=8)
        assert len(fs.features) == 8
        assert fs.angular_coverage_ratio < 1.0

    def test_feature_set_to_dict(self):
        fs = make_synthetic_features(num_features=12)
        d = fs.to_dict()
        assert "features" in d
        assert len(d["features"]) == 12
        assert "angular_coverage_ratio" in d


# ── PentacamDetectionResult ─────────────────────────────────────────

class TestPentacamDetectionResult:
    def test_synthetic_result(self):
        r = make_synthetic_detection_result()
        assert r.valid is True
        assert r.status == PentacamDetectionStatus.OK
        assert r.geometry.pupil_detected is True
        assert len(r.feature_set.features) == 36

    def test_result_to_dict(self):
        r = make_synthetic_detection_result()
        d = r.to_dict()
        assert d["valid"] is True
        assert d["status"] == "OK"
        assert "geometry" in d
        assert "feature_set" in d

    def test_result_json_round_trip(self):
        r = make_synthetic_detection_result()
        d = r.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["valid"] is True
        assert loaded["status"] == "OK"

    def test_failed_result(self):
        r = PentacamDetectionResult(
            valid=False,
            status=PentacamDetectionStatus.NO_PUPIL,
            failure_reason="pupil not detected",
        )
        d = r.to_dict()
        assert d["valid"] is False
        assert d["status"] == "NO_PUPIL"


# ── Determinism ─────────────────────────────────────────────────────

class TestDeterminism:
    def test_repeated_synthetic_same_result(self):
        r1 = make_synthetic_detection_result()
        r2 = make_synthetic_detection_result()
        d1 = r1.to_dict()
        d2 = r2.to_dict()
        assert json.dumps(d1) == json.dumps(d2)

    def test_different_sizes(self):
        r1 = make_synthetic_detection_result(image_width=640, image_height=480)
        r2 = make_synthetic_detection_result(image_width=1024, image_height=768)
        assert r1.image_width == 640
        assert r2.image_width == 1024


# ── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_feature_set(self):
        fs = PentacamFeatureSet()
        d = fs.to_dict()
        assert len(d["features"]) == 0

    def test_zero_radius_geometry(self):
        geom = PentacamGeometry(
            pupil_detected=False,
            limbus_detected=False,
            pupil_radius_px=0.0,
            limbus_radius_px=0.0,
        )
        d = geom.to_dict()
        assert d["pupil_radius_px"] == 0.0

    def test_no_detection_result(self):
        r = PentacamDetectionResult()
        d = r.to_dict()
        assert d["valid"] is False
        assert d["status"] == "NO_IMAGE"


# ── Cross-System Registration Types ─────────────────────────────────

from pupil_tracking.pentacam.cross_system import (
    CrossSystemRegistrationInput,
    CrossSystemRegistrationResult,
    RegistrationFailureKind,
    TransformationModel,
)


class TestCrossSystemRegistrationInput:
    def test_empty_input(self):
        inp = CrossSystemRegistrationInput()
        d = inp.to_dict()
        assert d["has_pentacam"] is False
        assert d["has_elita_supine"] is False

    def test_with_pentacam(self):
        pentacam = make_synthetic_detection_result()
        inp = CrossSystemRegistrationInput(pentacam=pentacam)
        d = inp.to_dict()
        assert d["has_pentacam"] is True
        assert d["has_elita_supine"] is False

    def test_coordinate_systems(self):
        inp = CrossSystemRegistrationInput(
            pentacam_coordinate_system="test_coord",
            elita_coordinate_system="elita_pixel",
        )
        d = inp.to_dict()
        assert d["pentacam_coordinate_system"] == "test_coord"


class TestCrossSystemRegistrationResult:
    def test_empty_result(self):
        r = CrossSystemRegistrationResult()
        d = r.to_dict()
        assert d["valid"] is False
        assert d["failure"] == "NO_PENTACAM"

    def test_successful_registration(self):
        r = CrossSystemRegistrationResult(
            valid=True,
            failure=RegistrationFailureKind.OK,
            transformation_model=TransformationModel.SIMILARITY_2D,
            rotation_deg=5.0,
            translation_x=10.0,
            translation_y=-5.0,
            scale=1.02,
            n_correspondences=20,
            n_inliers=18,
            inlier_fraction=0.9,
            confidence=0.85,
        )
        d = r.to_dict()
        assert d["valid"] is True
        assert d["failure"] == "OK"
        assert d["rotation_deg"] == 5.0
        assert d["n_inliers"] == 18

    def test_with_transform_matrix(self):
        mat = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]])
        r = CrossSystemRegistrationResult(
            valid=True,
            transform_matrix=mat,
        )
        d = r.to_dict()
        assert "transform_matrix" in d
        assert len(d["transform_matrix"]) == 2

    def test_with_cyclotorsion_composition(self):
        r = CrossSystemRegistrationResult(
            valid=True,
            rotation_deg=3.0,
            elita_cyclotorsion_deg=2.5,
            final_sitting_to_supine_deg=5.5,
        )
        d = r.to_dict()
        assert d["final_sitting_to_supine_deg"] == 5.5

    def test_failure_kinds(self):
        for kind in RegistrationFailureKind:
            r = CrossSystemRegistrationResult(failure=kind)
            d = r.to_dict()
            assert d["failure"] == kind.value

    def test_transformation_models(self):
        for model in TransformationModel:
            r = CrossSystemRegistrationResult(transformation_model=model)
            d = r.to_dict()
            assert d["transformation_model"] == model.value

    def test_json_round_trip(self):
        r = CrossSystemRegistrationResult(
            valid=True,
            failure=RegistrationFailureKind.OK,
            rotation_deg=7.5,
            confidence=0.9,
        )
        import json
        d = r.to_dict()
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["valid"] is True
        assert loaded["rotation_deg"] == 7.5
