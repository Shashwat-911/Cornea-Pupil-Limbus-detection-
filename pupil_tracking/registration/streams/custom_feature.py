"""
Stream E: Custom-trained iris feature model.

Uses a DINOv2-based model trained via the standalone annotation tool
to extract iris landmarks (crypts, furrows, junctions) and compute
rotation from correspondences.

The model is loaded from an ONNX file exported by the training pipeline.
If no model is available, this stream gracefully reports unavailability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pupil_tracking.registration.enhancement import IrisEnhancer
from pupil_tracking.registration.streams.base import BaseStream
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class CustomFeatureStream(BaseStream):
    """Uses a custom-trained model to extract iris landmarks
    and compute rotation from correspondences.

    The model outputs:
        - keypoint_heatmap [1, 1, H, W]
        - descriptors [1, D, H, W]

    If the model file does not exist, this stream is automatically
    disabled and returns gracefully.
    """

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(StreamName.CUSTOM_FEATURE)
        cfg = get_config().registration
        self.model_path = Path(
            model_path or cfg.custom_feature_model_path
        )
        self.available = False
        self.session = None
        self.enhancer = IrisEnhancer()

        if self.model_path.exists():
            self._load_model()
        else:
            logger.info(
                "Custom feature model not found at %s — stream disabled",
                self.model_path,
            )

    def _load_model(self):
        """Load ONNX model with runtime provider auto-selection."""
        try:
            import onnxruntime as ort
            providers = []
            available = ort.get_available_providers()
            for p in ['CUDAExecutionProvider', 'DmlExecutionProvider',
                       'CPUExecutionProvider']:
                if p in available:
                    providers.append(p)
            if not providers:
                providers = ['CPUExecutionProvider']

            self.session = ort.InferenceSession(
                str(self.model_path), providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.available = True
            logger.info(
                "Custom feature model loaded: %s (providers: %s)",
                self.model_path, providers,
            )
        except Exception as e:
            logger.error("Failed to load custom model: %s", e)

    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Extract landmarks, match, compute rotation."""

        if not self.available:
            return self._make_result(
                metadata={"error": "model_not_available"}
            )

        if not detection_ref.has_both or not detection_curr.has_both:
            return self._make_result(
                metadata={"error": "incomplete_detection"}
            )

        le_ref = detection_ref.limbus.ellipse
        le_curr = detection_curr.limbus.ellipse

        # Extract iris crops
        crop_ref = self.enhancer.extract_iris_crop(
            img_ref,
            center=(le_ref.center_x, le_ref.center_y),
            radius=le_ref.radius,
            target_size=self.input_shape[2] if self.input_shape[2] else 224,
        )
        crop_curr = self.enhancer.extract_iris_crop(
            img_curr,
            center=(le_curr.center_x, le_curr.center_y),
            radius=le_curr.radius,
            target_size=self.input_shape[2] if self.input_shape[2] else 224,
        )

        # Run inference
        keypoints_ref, descriptors_ref = self._infer(crop_ref['image'])
        keypoints_curr, descriptors_curr = self._infer(crop_curr['image'])

        if len(keypoints_ref) < 5 or len(keypoints_curr) < 5:
            return self._make_result(
                metadata={"error": "insufficient_keypoints",
                           "kp_ref": len(keypoints_ref),
                           "kp_curr": len(keypoints_curr)}
            )

        # Match descriptors
        matches = self._match_descriptors(descriptors_ref, descriptors_curr)

        if len(matches) < 3:
            return self._make_result(
                metadata={"error": "insufficient_matches",
                           "num_matches": len(matches)}
            )

        # Compute rotation from matched keypoints
        pts_ref = keypoints_ref[matches[:, 0]]
        pts_curr = keypoints_curr[matches[:, 1]]

        crop_center = np.array([crop_ref['target_size'] / 2.0,
                                 crop_ref['target_size'] / 2.0])

        angles_ref = np.arctan2(
            pts_ref[:, 1] - crop_center[1],
            pts_ref[:, 0] - crop_center[0],
        )
        angles_curr = np.arctan2(
            pts_curr[:, 1] - crop_center[1],
            pts_curr[:, 0] - crop_center[0],
        )

        diffs = (angles_curr - angles_ref + np.pi) % (2 * np.pi) - np.pi

        # Robust median + MAD
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        inliers = np.abs(diffs - median_diff) < 2.5 * max(mad, 0.005)

        if np.sum(inliers) < 3:
            return self._make_result(
                metadata={"error": "too_few_inliers"}
            )

        torsion_rad = float(np.mean(diffs[inliers]))
        torsion_deg = float(np.degrees(torsion_rad))

        inlier_ratio = np.sum(inliers) / len(diffs)
        consistency = 1.0 / (1.0 + np.std(diffs[inliers]))
        confidence = float(np.clip(
            inlier_ratio * consistency, 0.0, 1.0
        ))

        return self._make_result(
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=int(np.sum(inliers)),
            metadata={
                "total_matches": len(matches),
                "keypoints_ref": len(keypoints_ref),
                "keypoints_curr": len(keypoints_curr),
                "model_path": str(self.model_path),
            },
        )

    def _infer(self, image: np.ndarray):
        """Run ONNX inference, return keypoints and descriptors."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Preprocess: HWC → CHW, normalise
        img_norm = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_norm - mean) / std
        img_chw = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: img_chw})

        # Expected: heatmap [1, 1, H, W], descriptor_map [1, D, H, W]
        heatmap = outputs[0][0, 0]       # [H, W]
        descriptor_map = outputs[1][0]   # [D, H, W]

        keypoints, descriptors = self._extract_keypoints_from_heatmap(
            heatmap, descriptor_map, top_k=200, nms_radius=4
        )
        return keypoints, descriptors

    def _extract_keypoints_from_heatmap(
        self,
        heatmap: np.ndarray,
        descriptor_map: np.ndarray,
        top_k: int = 200,
        nms_radius: int = 4,
    ):
        """Non-maximum suppression on heatmap to extract keypoints."""
        from scipy.ndimage import maximum_filter

        local_max = maximum_filter(heatmap, size=nms_radius * 2 + 1)
        peaks = (heatmap == local_max) & (heatmap > 0.05)

        ys, xs = np.where(peaks)
        scores = heatmap[ys, xs]

        if len(scores) > top_k:
            top_idx = np.argsort(scores)[-top_k:]
            ys, xs = ys[top_idx], xs[top_idx]

        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

        keypoints = np.stack([xs, ys], axis=1).astype(np.float32)

        # Sample descriptors at keypoint locations
        D = descriptor_map.shape[0]
        descriptors = descriptor_map[:, ys, xs].T  # [N, D]
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8
        descriptors = descriptors / norms

        return keypoints, descriptors

    def _match_descriptors(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.85,
    ) -> np.ndarray:
        """Mutual nearest-neighbour with ratio test."""
        if len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0, 2), dtype=int)

        sim = desc1 @ desc2.T  # Cosine similarity

        # For each in desc1, find top 2 in desc2
        if sim.shape[1] < 2:
            return np.empty((0, 2), dtype=int)

        top2_idx = np.argsort(sim, axis=1)[:, -2:]
        top2_sim = np.take_along_axis(sim, top2_idx, axis=1)

        # Ratio test
        ratio = top2_sim[:, 0] / (top2_sim[:, 1] + 1e-8)
        pass_ratio = ratio < ratio_threshold

        # Mutual check
        best_idx = top2_idx[:, 1]
        matches = []
        for i, j in enumerate(best_idx):
            if not pass_ratio[i]:
                continue
            j_best = np.argmax(sim[:, j])
            if j_best == i:
                matches.append([i, int(j)])

        return np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
