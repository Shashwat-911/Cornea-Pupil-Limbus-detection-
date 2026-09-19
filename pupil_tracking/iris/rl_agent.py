"""Reinforcement learning agent for adaptive iris detection parameters.

A lightweight DQN (Deep Q-Network) agent that adjusts iris detection
thresholds in real-time based on frame quality feedback. Observes detection
metrics and selects parameter adjustments to maximize feature yield.

Safety-constrained: actions are clamped to predefined ranges and rejected
when they deviate excessively from defaults.

Usage
-----
>>> from pupil_tracking.iris.rl_agent import IrisParamAgent
>>> agent = IrisParamAgent()
>>> state = IrisParamAgent.state_from_result(result, config)
>>> action = agent.select_action(state)
>>> adjusted_config = agent.apply_action(config, action)
"""

from __future__ import annotations

import json
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

# State dimensions: [feature_count, coverage, contrast, confidence,
#                    brightness, usable_fraction, frame_delta, prev_reward]
_STATE_DIM = 8

# Actions: 0-5
# 0: decrease min_contrast
# 1: increase min_contrast
# 2: decrease texture_floor
# 3: increase texture_floor
# 4: decrease max_features
# 5: increase max_features
_ACTION_DIM = 6

_ACTION_NAMES = [
    "decrease_min_contrast",
    "increase_min_contrast",
    "decrease_texture_floor",
    "increase_texture_floor",
    "decrease_max_features",
    "increase_max_features",
]

_ACTION_DELTAS = {
    0: ("min_contrast", -0.5),
    1: ("min_contrast", +0.5),
    2: ("texture_floor", -0.3),
    3: ("texture_floor", +0.3),
    4: ("max_features", -10),
    5: ("max_features", +10),
}

_DEFAULT_VALUES = {
    "min_contrast": 4.0,
    "texture_floor": 2.5,
    "max_features": 120,
}

_SAFE_RANGES = {
    "min_contrast": (1.0, 10.0),
    "texture_floor": (0.5, 8.0),
    "max_features": (20, 120),
}

_MODEL_SEARCH_PATHS = [
    "models/iris_rl_agent.onnx",
    "models/iris_rl_agent.pth",
]


@dataclass
class Transition:
    """Single experience tuple for replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class _QNetwork(nn.Module):
    """Small feedforward Q-network for parameter tuning."""

    def __init__(self, state_dim: int = _STATE_DIM, action_dim: int = _ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IrisParamAgent:
    """DQN agent for adaptive iris detection parameter tuning.

    Parameters
    ----------
    buffer_size : int
        Experience replay buffer capacity.
    gamma : float
        Discount factor for future rewards.
    learning_rate : float
        Adam optimizer learning rate.
    epsilon_start : float
        Initial exploration rate.
    epsilon_end : float
        Minimum exploration rate.
    epsilon_decay : float
        Exponential decay factor per step.
    target_update_freq : int
        Steps between target network updates.
    batch_size : int
        Mini-batch size for Q-network updates.
    safety_clamp_sigma : float
        Reject actions whose resulting parameter deviates more than this
        many standard deviations from the default value.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        batch_size: int = 32,
        safety_clamp_sigma: float = 2.0,
        model_path: Optional[str] = None,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.safety_clamp_sigma = safety_clamp_sigma

        self._buffer: Deque[Transition] = deque(maxlen=buffer_size)
        self._step_count = 0
        self._current_values: Dict[str, float] = dict(_DEFAULT_VALUES)

        self._q_net: Optional[nn.Module] = None
        self._target_net: Optional[nn.Module] = None
        self._optimizer = None

        if model_path is None:
            for p in _MODEL_SEARCH_PATHS:
                if Path(p).is_file():
                    model_path = p
                    break

        if _HAS_TORCH:
            self._q_net = _QNetwork()
            self._target_net = _QNetwork()
            self._target_net.load_state_dict(self._q_net.state_dict())
            self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=learning_rate)

            if model_path and Path(model_path).is_file():
                self.load(model_path)

    @property
    def available(self) -> bool:
        return self._q_net is not None

    def state_from_result(
        result: Any,
        config: Any,
        prev_result: Optional[Any] = None,
    ) -> np.ndarray:
        """Extract 8-d state vector from detection result and config.

        Parameters
        ----------
        result : IrisDetectionResult
        config : IrisConfig
        prev_result : IrisDetectionResult | None

        Returns
        -------
        np.ndarray (8,) float32
        """
        fs = result.feature_set if hasattr(result, "feature_set") else None

        feature_count = len(fs.features) if fs else 0
        coverage = fs.region_coverage if fs else 0.0
        usable = fs.usable_fraction if fs else 0.0

        contrasts = [f.local_contrast for f in fs.features] if fs and fs.features else [0.0]
        confidences = [f.confidence for f in fs.features] if fs and fs.features else [0.0]
        mean_contrast = float(np.mean(contrasts))
        mean_confidence = float(np.mean(confidences))

        mask_stats = getattr(result, "mask_stats", {})
        brightness = mask_stats.get("intensity_p50", 128.0) / 255.0

        prev_feature_count = 0
        if prev_result is not None:
            pfs = getattr(prev_result, "feature_set", None)
            if pfs and pfs.features:
                prev_feature_count = len(pfs.features)

        frame_delta = min(max((feature_count - prev_feature_count) / 120.0, -1.0), 1.0)
        prev_reward = 0.0

        state = np.array(
            [
                feature_count / 120.0,
                coverage,
                mean_contrast / 20.0,
                mean_confidence,
                brightness,
                usable,
                frame_delta,
                prev_reward,
            ],
            dtype=np.float32,
        )
        return np.clip(state, -1.0, 1.0)

    state_from_result = staticmethod(state_from_result)

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """Select an action using epsilon-greedy policy.

        Parameters
        ----------
        state : (8,) float32
            Current state vector.
        deterministic : bool
            If True, always exploit (no exploration).

        Returns
        -------
        int
            Action index (0-5).
        """
        if not self.available:
            return 0

        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, _ACTION_DIM - 1)

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0)
            q_values = self._q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def apply_action(self, config: Any, action: int) -> Any:
        """Apply an action to the config, with safety clamping.

        Parameters
        ----------
        config : IrisConfig
            Current configuration (modified in-place and returned).
        action : int
            Action index (0-5).

        Returns
        -------
        IrisConfig
            Modified configuration.
        """
        if action not in _ACTION_DELTAS:
            return config

        param_name, delta = _ACTION_DELTAS[action]
        current = getattr(config, param_name, _DEFAULT_VALUES[param_name])

        new_val = current + delta
        lo, hi = _SAFE_RANGES[param_name]

        default = _DEFAULT_VALUES[param_name]
        std = abs(delta) * 2
        if abs(new_val - default) > self.safety_clamp_sigma * std:
            new_val = default

        if isinstance(config, dict):
            config[param_name] = type(_DEFAULT_VALUES[param_name])(max(lo, min(hi, new_val)))
        else:
            setattr(config, param_name, type(_DEFAULT_VALUES[param_name])(max(lo, min(hi, new_val))))

        self._current_values[param_name] = getattr(config, param_name, new_val)
        return config

    def compute_reward(
        self,
        feature_count: int,
        coverage: float,
        confidence: float,
        usable_fraction: float = 0.0,
    ) -> float:
        """Compute reward for the current frame.

        Reward is proportional to feature yield * quality, with penalties
        for too few or too many features.

        Parameters
        ----------
        feature_count : int
            Number of accepted features.
        coverage : float
            Region coverage fraction.
        confidence : float
            Mean feature confidence.
        usable_fraction : float
            Fraction of usable iris pixels.

        Returns
        -------
        float
            Reward value.
        """
        if feature_count < 5:
            reward = -10.0 + feature_count * 0.5
        elif feature_count > 80:
            reward = -5.0 + (80 - feature_count) * 0.1
        else:
            reward = feature_count * coverage * confidence * usable_fraction

        return float(reward)

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self._buffer.append(Transition(state, action, reward, next_state, done))

    def update(self) -> Optional[float]:
        """Perform one training step if enough data is available.

        Returns
        -------
        float or None
            Loss value if updated, None otherwise.
        """
        if not self.available or len(self._buffer) < self.batch_size:
            return None

        batch = random.sample(list(self._buffer), self.batch_size)

        states = torch.from_numpy(np.stack([t.state for t in batch]))
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch]))
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32)

        q_values = self._q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self._target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)

        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._q_net.parameters(), 1.0)
        self._optimizer.step()

        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._target_net.load_state_dict(self._q_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save agent state."""
        if not self.available:
            raise RuntimeError("No Q-network to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net_state_dict": self._q_net.state_dict(),
                "target_net_state_dict": self._target_net.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self._step_count,
                "current_values": self._current_values,
                "state_dim": _STATE_DIM,
                "action_dim": _ACTION_DIM,
            },
            path,
        )
        logger.info("Iris RL agent saved: %s", path)

    def load(self, path: str) -> None:
        """Load agent state."""
        if not self.available:
            raise RuntimeError("No Q-network loaded")

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._q_net.load_state_dict(ckpt["q_net_state_dict"])
        self._target_net.load_state_dict(ckpt["target_net_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._step_count = ckpt.get("step_count", 0)
        self._current_values = ckpt.get("current_values", dict(_DEFAULT_VALUES))
        logger.info("Iris RL agent loaded: %s (step=%d, eps=%.3f)", path, self._step_count, self.epsilon)

    def get_current_values(self) -> Dict[str, float]:
        """Return the current adjusted parameter values."""
        return dict(self._current_values)

    def reset_to_defaults(self) -> None:
        """Reset all parameters to defaults."""
        self._current_values = dict(_DEFAULT_VALUES)
