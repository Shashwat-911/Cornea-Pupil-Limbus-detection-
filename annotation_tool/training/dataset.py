"""
PyTorch Dataset for iris feature training.

Loads annotated iris images, creates keypoint heatmaps and
class maps from annotations for supervised training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:

    class IrisFeatureDataset(Dataset):
        """PyTorch Dataset for iris feature model training.

        Loads images and annotations, produces:
            - image: [3, H, W] normalised tensor
            - keypoint_heatmap: [1, H, W] Gaussian at each keypoint
            - class_map: [1, H, W] integer class per pixel
        """

        CLASS_MAP = {
            "iris_crypt": 1,
            "crypt": 1,
            "radial_furrow": 2,
            "furrow": 2,
            "vessel_bifurcation": 3,
            "vessel": 3,
            "ink": 4,
            "collarette_junction": 1,
            "pigment_spot": 1,
        }

        def __init__(
            self,
            split_file: Path,
            image_size: int = 224,
            augment: bool = False,
            heatmap_sigma: float = 3.0,
        ):
            self.image_size = image_size
            self.augment = augment
            self.heatmap_sigma = heatmap_sigma

            # Load split manifest
            with open(split_file) as f:
                self.samples: List[Dict[str, str]] = json.load(f)

            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            sample = self.samples[idx]
            image_path = sample["image_path"]
            annotation_path = sample["annotation_path"]

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))

            # Load annotation
            with open(annotation_path) as f:
                annotation = json.load(f)

            # Scale factor for coordinates
            orig_h = annotation.get("image_height", image.shape[0])
            orig_w = annotation.get("image_width", image.shape[1])
            scale_x = self.image_size / max(orig_w, 1)
            scale_y = self.image_size / max(orig_h, 1)

            # Create targets
            heatmap = self._create_heatmap(
                annotation, scale_x, scale_y
            )
            class_map = self._create_class_map(
                annotation, scale_x, scale_y
            )

            # Augmentation
            if self.augment:
                image, heatmap, class_map = self._augment(
                    image, heatmap, class_map
                )

            # To tensor
            image_tensor = self._to_tensor(image)
            heatmap_tensor = torch.from_numpy(heatmap).float().unsqueeze(0)
            class_tensor = torch.from_numpy(class_map).long().unsqueeze(0)

            return {
                "image": image_tensor,
                "heatmap": heatmap_tensor,
                "class_map": class_tensor,
                "image_path": image_path,
            }

        def _create_heatmap(
            self,
            annotation: dict,
            scale_x: float,
            scale_y: float,
        ) -> np.ndarray:
            """Create Gaussian heatmap from keypoints."""
            heatmap = np.zeros(
                (self.image_size, self.image_size), dtype=np.float32
            )

            for kp in annotation.get("keypoints", []):
                x = kp["x"] * scale_x
                y = kp["y"] * scale_y
                self._add_gaussian(heatmap, x, y, self.heatmap_sigma)

            for mark in annotation.get("ink_marks", []):
                x = mark["center_x"] * scale_x
                y = mark["center_y"] * scale_y
                self._add_gaussian(heatmap, x, y, self.heatmap_sigma * 1.5)

            return np.clip(heatmap, 0, 1)

        def _create_class_map(
            self,
            annotation: dict,
            scale_x: float,
            scale_y: float,
        ) -> np.ndarray:
            """Create per-pixel class map from annotations."""
            class_map = np.zeros(
                (self.image_size, self.image_size), dtype=np.int32
            )

            for kp in annotation.get("keypoints", []):
                x = int(kp["x"] * scale_x)
                y = int(kp["y"] * scale_y)
                cls = self.CLASS_MAP.get(kp.get("type", ""), 1)
                r = int(self.heatmap_sigma * 2)
                cv2.circle(class_map, (x, y), r, int(cls), -1)

            for mark in annotation.get("ink_marks", []):
                x = int(mark["center_x"] * scale_x)
                y = int(mark["center_y"] * scale_y)
                r = int(mark.get("radius", 8) * min(scale_x, scale_y))
                cv2.circle(class_map, (x, y), max(r, 3), 4, -1)

            return class_map.astype(np.int64)

        @staticmethod
        def _add_gaussian(heatmap, cx, cy, sigma):
            """Add a 2D Gaussian to the heatmap at (cx, cy)."""
            h, w = heatmap.shape
            x = np.arange(0, w)
            y = np.arange(0, h)
            X, Y = np.meshgrid(x, y)
            gaussian = np.exp(
                -((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)
            )
            heatmap += gaussian.astype(np.float32)

        def _augment(self, image, heatmap, class_map):
            """Apply data augmentation."""
            # Random horizontal flip
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                heatmap = np.fliplr(heatmap).copy()
                class_map = np.fliplr(class_map).copy()

            # Random brightness/contrast
            if np.random.random() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.2, 0.2)
                beta = np.random.uniform(-20, 20)
                image = np.clip(
                    image.astype(np.float32) * alpha + beta, 0, 255
                ).astype(np.uint8)

            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-180, 180)
                h, w = image.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
                heatmap = cv2.warpAffine(heatmap, M, (w, h))
                class_map = cv2.warpAffine(
                    class_map.astype(np.float32), M, (w, h),
                    flags=cv2.INTER_NEAREST,
                ).astype(np.int64)

            return image, heatmap, class_map

        def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
            """Normalise and convert HWC uint8 → CHW float tensor."""
            img = image.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            return torch.from_numpy(img.transpose(2, 0, 1)).float()

else:
    class IrisFeatureDataset:
        """Stub — torch not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for training")
