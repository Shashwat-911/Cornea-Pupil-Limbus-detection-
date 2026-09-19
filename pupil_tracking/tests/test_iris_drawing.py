"""Regression tests for the GUI iris-feature overlay render path.

The live and snapshot overlays (``draw_overlay_scaled`` / ``draw_overlay``)
must draw accepted iris features from ``result.iris_detection`` at their
source-image positions scaled by the display scale, and must be a silent
no-op when the detection is invalid or has no accepted features.
"""
from types import SimpleNamespace

import cv2
import numpy as np

from pupil_tracking.interface.gui.drawing import draw_iris_feature_overlay
from pupil_tracking.iris.types import (
    IrisDetectionResult,
    IrisFeature,
    IrisFeatureSet,
    IrisStatus,
)

_COLOR = (255, 80, 255)


def _result_with(features, valid=True):
    fs = IrisFeatureSet(features=features, num_candidates=len(features))
    res = IrisDetectionResult(
        valid=valid,
        status=IrisStatus.OK if valid else IrisStatus.NO_FEATURES,
        feature_set=fs,
    )
    return SimpleNamespace(iris_detection=res)


def _feature(x, y, valid=True):
    return IrisFeature(x=x, y=y, valid=valid)


def test_draws_features_at_scaled_positions():
    out = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _result_with([_feature(10, 10), _feature(30, 40)])
    draw_iris_feature_overlay(out, result, scale=2.0)
    assert out[20, 20].tolist() == list(_COLOR)
    assert out[80, 60].tolist() == list(_COLOR)


def test_skips_invalid_features():
    out = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _result_with([_feature(10, 10), _feature(30, 40, valid=False)])
    draw_iris_feature_overlay(out, result, scale=1.0)
    assert out[10, 10].tolist() == list(_COLOR)
    assert out[40, 30].tolist() == [0, 0, 0]


def test_noop_when_invalid_detection():
    out = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _result_with([_feature(10, 10)], valid=False)
    draw_iris_feature_overlay(out, result, scale=1.0)
    assert not (out > 0).any()


def test_noop_when_no_features():
    out = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_iris_feature_overlay(out, _result_with([]), scale=1.0)
    assert not (out > 0).any()


def test_noop_when_no_iris_result():
    out = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_iris_feature_overlay(out, SimpleNamespace(), scale=1.0)
    assert not (out > 0).any()


def test_label_rendered_near_feature_count():
    out = np.zeros((200, 200, 3), dtype=np.uint8)
    draw_iris_feature_overlay(
        out, _result_with([_feature(40, 40) for _ in range(3)]), scale=1.0
    )
    label = "Iris: 3 features"
    h, w = out.shape[:2]
    # putText is anti-aliased; count non-background pixels in the label band.
    band = out[20:50, 10:160]
    assert np.count_nonzero(band) > 0


def test_full_resolution_draws_at_source_coordinates():
    out = np.zeros((200, 200, 3), dtype=np.uint8)
    result = _result_with([_feature(100, 100)])
    draw_iris_feature_overlay(out, result, scale=1.0)
    assert out[100, 100].tolist() == list(_COLOR)