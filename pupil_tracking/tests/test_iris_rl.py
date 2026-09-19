"""Tests for the Phase V RL parameter tuning agent.

These tests are deterministic and do not require a trained agent or torch
(rl_agent.py degrades gracefully without torch; the Q-network tests are
skipped via importorskip).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.rl_agent import (
    _ACTION_DIM,
    _ACTION_NAMES,
    _DEFAULT_VALUES,
    _SAFE_RANGES,
    _STATE_DIM,
    IrisParamAgent,
)

from pupil_tracking.iris.types import (
    IrisDetectionResult,
    IrisFeature,
    IrisFeatureSet,
    IrisFeatureType,
    IrisROI,
    IrisStatus,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_roi(size=320):
    return IrisROI(
        center_x=size / 2,
        center_y=size / 2,
        pupil_semi_major=55,
        pupil_semi_minor=50,
        limbus_semi_major=130,
        limbus_semi_minor=125,
        pupil_radius_px=55,
        limbus_radius_px=130,
        valid=True,
    )


def _make_result(n_features=20, coverage=0.3, usable=0.5, confidence=0.8, contrast=6.0):
    roi = _make_roi()
    fs = IrisFeatureSet(roi=roi)
    fs.region_coverage = coverage
    fs.usable_fraction = usable
    for i in range(n_features):
        fs.features.append(
            IrisFeature(
                id=i,
                x=160.0 + i,
                y=160.0 - i,
                angle_deg=i % 360,
                radial_norm=0.5,
                feature_type=IrisFeatureType.CRYPT,
                local_contrast=contrast,
                confidence=confidence,
                visibility=0.9,
                valid=True,
                descriptor=np.zeros(16, dtype=np.float32),
            )
        )
    return IrisDetectionResult(
        valid=True,
        status=IrisStatus.OK,
        feature_set=fs,
        mask_stats={"intensity_p50": 128.0, "usable_fraction": usable},
    )


@pytest.fixture
def agent():
    a = IrisParamAgent(buffer_size=64, batch_size=4, epsilon_start=0.0)
    assert a.available
    return a


def test_action_contract():
    assert _STATE_DIM == 8
    assert _ACTION_DIM == 6
    assert len(_ACTION_NAMES) == _ACTION_DIM
    # safe ranges are sane
    assert _SAFE_RANGES["min_contrast"][0] < _SAFE_RANGES["min_contrast"][1]
    assert _SAFE_RANGES["texture_floor"][0] < _SAFE_RANGES["texture_floor"][1]
    assert _SAFE_RANGES["max_features"][0] < _SAFE_RANGES["max_features"][1]


def test_state_from_result_shape_and_clip():
    result = _make_result(n_features=10)
    state = IrisParamAgent.state_from_result(result, IrisConfig())
    assert state.shape == (_STATE_DIM,)
    assert state.dtype == np.float32
    assert np.all(np.abs(state) <= 1.0)


def test_state_from_result_empty_result():
    config = IrisConfig()
    result = _make_result(n_features=0)
    state = IrisParamAgent.state_from_result(result, config)
    assert state.shape == (_STATE_DIM,)
    assert np.all(np.abs(state) <= 1.0)


def test_apply_action_min_contrast():
    agent = IrisParamAgent()
    config = IrisConfig(min_contrast=4.0, texture_floor=2.5, max_features=120)
    # increase min_contrast (action 1)
    out = agent.apply_action(config, 1)
    assert out.min_contrast == pytest.approx(4.5)


def test_apply_action_safe_range_clamp():
    agent = IrisParamAgent(safety_clamp_sigma=100.0)
    config = IrisConfig(min_contrast=9.8, texture_floor=2.5, max_features=120)
    # increasing min_contrast would exceed 10.0 -> clamped to the safe-range max
    out = agent.apply_action(config, 1)
    assert out.min_contrast == pytest.approx(10.0)


def test_apply_action_safety_clamp_resets_to_default():
    # safety_clamp_sigma must reject extreme deviations
    agent = IrisParamAgent(safety_clamp_sigma=0.05)
    config = IrisConfig(min_contrast=4.0, texture_floor=2.5, max_features=120)
    # raising min_contrast by 0.5 across many apply calls is rejected
    for _ in range(200):
        agent.apply_action(config, 1)
    # after the first action the safety clamp resets deviation, so value stays at default
    assert config.min_contrast == pytest.approx(_DEFAULT_VALUES["min_contrast"], abs=0.51)


def test_apply_action_max_features_bounds():
    agent = IrisParamAgent()
    config = IrisConfig(max_features=120)
    out = agent.apply_action(config, 5)  # increase max_features
    assert out.max_features >= 120  # clamped at 120 (already max, stays)
    out = agent.apply_action(config, 4)  # decrease max_features
    assert out.max_features == 110


def test_reward_too_few_features_negative():
    agent = IrisParamAgent()
    r = agent.compute_reward(feature_count=2, coverage=0.5, confidence=0.8, usable_fraction=0.5)
    assert r < 0.0


def test_reward_too_many_features_negative():
    agent = IrisParamAgent()
    r = agent.compute_reward(feature_count=90, coverage=0.5, confidence=0.8, usable_fraction=0.5)
    assert r < 0.0


def test_reward_good_yield_positive():
    agent = IrisParamAgent()
    r = agent.compute_reward(feature_count=40, coverage=0.5, confidence=0.8, usable_fraction=0.6)
    assert r > 0.0


def test_select_action_deterministic_within_range():
    agent = IrisParamAgent(epsilon_start=0.0)
    state = np.zeros(_STATE_DIM, dtype=np.float32)
    for _ in range(50):
        a = agent.select_action(state, deterministic=True)
        assert 0 <= a < _ACTION_DIM


def test_replay_buffer_and_update_returns_loss():
    agent = IrisParamAgent(buffer_size=64, batch_size=4, epsilon_start=0.0)
    assert agent.update() is None  # empty buffer

    s = np.zeros(_STATE_DIM, dtype=np.float32)
    for _ in range(8):
        agent.store_transition(s, 1, 1.0, s, False)
    loss = agent.update()
    assert loss is not None  # buffer has 8 >= batch_size 4
    assert np.isfinite(loss)


def test_replay_buffer_update_requires_minimum():
    agent = IrisParamAgent(buffer_size=64, batch_size=4, epsilon_start=0.0)
    assert agent.update() is None  # empty buffer


def test_save_load_roundtrip(tmp_path):
    agent = IrisParamAgent(epsilon_start=0.3)
    agent.epsilon = 0.3
    agent._current_values["min_contrast"] = 3.0
    p = str(tmp_path / "agent.pth")
    agent.save(p)
    assert p and __import__("pathlib").Path(p).is_file()

    loaded = IrisParamAgent(epsilon_start=0.0)
    loaded.load(p)
    assert loaded.get_current_values()["min_contrast"] == 3.0


def test_reset_to_defaults():
    agent = IrisParamAgent()
    agent.apply_action(IrisConfig(), 1)
    agent.reset_to_defaults()
    assert agent.get_current_values() == _DEFAULT_VALUES


def test_rl_integration_detector_runs_without_crash():
    """The detector must not crash when rl_enabled but no model is present."""
    import cv2

    from pupil_tracking.iris.detect import IrisFeatureDetector
    from pupil_tracking.utils.types import EllipseParams

    img = np.full((320, 320, 3), 25, np.uint8)
    cv2.circle(img, (160, 160), 130, (80, 80, 80), -1)
    cv2.circle(img, (160, 160), 55, (10, 10, 10), -1)

    pupil = EllipseParams(center_x=160, center_y=160, semi_major=55, semi_minor=50, angle_deg=0.0)
    limbus = EllipseParams(center_x=160, center_y=160, semi_major=130, semi_minor=120, angle_deg=0.0)

    config = IrisConfig(rl_enabled=True)
    detector = IrisFeatureDetector(config=config)
    result = detector.detect(img, pupil, limbus)
    assert isinstance(result, IrisDetectionResult)
    assert result.valid in (True, False)