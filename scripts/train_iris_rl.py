#!/usr/bin/env python3
"""Train the RL parameter-tuning agent for adaptive iris detection.

The agent learns to adjust min_contrast / texture_floor / max_features based
on frame-quality feedback. The "environment" is the classical iris detector
run on a video sequence.

Usage
-----
python scripts/train_iris_rl.py \
    --video path/to/video.mp4 \
    --episodes 500 --epochs 200 --save models/iris_rl_agent
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.detect import IrisFeatureDetector
from pupil_tracking.iris.rl_agent import IrisParamAgent
from pupil_tracking.utils.logger import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL iris parameter agent")
    p.add_argument("--video", required=True, help="Video file to train on")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=100, help="Max frames per episode")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer-size", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--save", type=str, default="models/iris_rl_agent.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--simulate-pupil", type=float, default=60.0, help="Synthetic pupil radius px")
    p.add_argument("--simulate-limbus", type=float, default=160.0, help="Synthetic limbus radius px")
    p.add_argument("--fit-sample", type=int, default=0, help="Use a single frame for synthetic env (0 = full video)")
    return p.parse_args()


def _extract_frame_pair(cap) -> tuple:
    """Read next frame; return (frame_bgr or None)."""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return (None, None) if not ret else (frame, frame)


class _IrisEnv:
    """RL environment wrapping IrisFeatureDetector.

    Each step runs detection with the current config, returns state,
    reward, and whether the episode is done.
    """

    def __init__(
        self,
        detector: IrisFeatureDetector,
        frame_source,
        pupil: dict,
        limbus: dict,
        max_steps: int = 100,
    ):
        self.detector = detector
        self.frame_source = frame_source
        self.pupil = pupil
        self.limbus = limbus
        self.max_steps = max_steps
        self.step_count = 0
        self.frames = self._preload_frames()

    def _preload_frames(self) -> list:
        frames = []
        while True:
            frame, _ = self.frame_source()
            if frame is None:
                break
            frames.append(frame)
            if len(frames) >= 300:
                break
        return frames

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.detector.config = copy.copy(self.detector.config)
        self.detector.config.min_contrast = 4.0
        self.detector.config.texture_floor = 2.5
        self.detector.config.max_features = 120
        self.detector.extractor.min_contrast = 4.0
        self.detector.extractor.texture_floor = 2.5
        self.detector.extractor.max_features = 120
        self.detector._prev_result = None

        frame = self.frames[0]
        result = self.detector.detect(frame, self.pupil, self.limbus)
        return IrisParamAgent.state_from_result(result, self.detector.config)

    def step(self, action: int) -> tuple:
        if self.step_count >= self.max_steps or not self.frames:
            return (
                IrisParamAgent.state_from_result(None, self.detector.config),
                0.0,
                True,
                {},
            )

        # Apply the action to the detector config
        self.detector.config = self.detector._rl_agent.apply_action(
            copy.copy(self.detector.config), action
        ) if self.detector._rl_agent else self.detector.config

        self.detector.extractor.min_contrast = self.detector.config.min_contrast
        self.detector.extractor.texture_floor = self.detector.config.texture_floor
        self.detector.extractor.max_features = self.detector.config.max_features

        frame = self.frames[self.step_count % len(self.frames)]
        result = self.detector.detect(frame, self.pupil, self.limbus)
        self.step_count += 1

        fs = result.feature_set
        n_features = len(fs.features) if fs else 0
        coverage = fs.region_coverage if fs else 0.0
        usable = fs.usable_fraction if fs else 0.0

        confidences = [f.confidence for f in fs.features] if fs and fs.features else [0.0]
        mean_conf = float(np.mean(confidences))

        reward = IrisParamAgent.compute_reward(
            feature_count=n_features,
            coverage=coverage,
            confidence=mean_conf,
            usable_fraction=usable,
        )

        done = self.step_count >= self.max_steps
        next_state = IrisParamAgent.state_from_result(result, self.detector.config)
        return next_state, reward, done, {}


def _make_synthetic_pupil(center, radius):
    from pupil_tracking.utils.types import EllipseParams

    return EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=radius,
        semi_minor=radius * 0.9,
        angle_deg=0.0,
    )


def _make_synthetic_limbus(center, radius):
    from pupil_tracking.utils.types import EllipseParams

    return EllipseParams(
        center_x=center[0],
        center_y=center[1],
        semi_major=radius,
        semi_minor=radius * 0.95,
        angle_deg=0.0,
    )


def main():
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build the detector (classical base; no CNN during RL training)
    config = IrisConfig()
    detector = IrisFeatureDetector(config=config)

    # Load frame source
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", args.video)
        sys.exit(1)

    frame, _ = _extract_frame_pair(cap)
    if frame is None:
        logger.error("No frames in video")
        sys.exit(1)

    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2

    pupil = _make_synthetic_pupil((cx, cy), args.simulate_pupil)
    limbus = _make_synthetic_limbus((cx, cy), args.simulate_limbus)

    # Rewind to frame 0 for the environment
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    env = _IrisEnv(
        detector=detector,
        frame_source=(lambda: _extract_frame_pair(cap)),
        pupil=pupil,
        limbus=limbus,
        max_steps=args.max_steps,
    )

    agent = IrisParamAgent(
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=100,
        batch_size=args.batch_size,
    )

    # Wire agent into detector for RL actions
    detector._rl_agent = agent

    episode_rewards = []
    for episode in range(args.episodes):
        state = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update() if len(agent._buffer) >= args.batch_size else None

            if loss is not None:
                pass  # update() already does the training step

            ep_reward += reward
            state = next_state
            steps += 1

            if steps >= args.max_steps:
                break

        episode_rewards.append(ep_reward)
        avg = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0

        if episode % 50 == 0 or episode == args.episodes - 1:
            logger.info(
                "Episode %4d/%d  reward=%8.2f  avg20=%8.2f  eps=%.3f  params=%s",
                episode + 1, args.episodes, ep_reward, avg, agent.epsilon,
                {k: round(v, 2) for k, v in agent.get_current_values().items()},
            )

    cap.release()
    agent.save(args.save)
    logger.info("Saved RL agent: %s", args.save)


if __name__ == "__main__":
    main()