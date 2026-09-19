"""
Iris-specific data augmentations for training feature detection models.

Preserves geometric validity of keypoint annotations under spatial
transformations including continuous rotation (critical for cyclotorsion),
scaling, cropping, and illumination variation.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class IrisAugmentor:
    """Augmentation pipeline specifically tuned for iris texture and landmarks.

    Handles synchronized image and keypoint transformations.
    """

    def __init__(
        self,
        rotation_range: float = 360.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        brightness: float = 0.2,
        contrast: float = 0.2,
        gaussian_blur_prob: float = 0.3,
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness = brightness
        self.contrast = contrast
        self.gaussian_blur_prob = gaussian_blur_prob

    def __call__(
        self,
        image: np.ndarray,
        keypoints: List[Tuple[float, float]],
        classes: List[int],
    ) -> Tuple[np.ndarray, List[Tuple[float, float]], List[int]]:
        """Apply random augmentations to image and keypoints.

        Args:
            image: [H, W, 3] BGR/RGB image
            keypoints: List of (x, y) coordinates in pixel space
            classes: List of class IDs corresponding to keypoints

        Returns:
            Tuple of (augmented_image, transformed_keypoints, valid_classes)
        """
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)

        # 1. Spatial transforms (rotation + scaling)
        angle = random.uniform(-self.rotation_range / 2.0, self.rotation_range / 2.0)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])

        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        aug_img = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Transform keypoints
        new_kps = []
        new_classes = []
        for (kx, ky), cls_id in zip(keypoints, classes):
            # Homogeneous coordinate transform
            pt = np.array([kx, ky, 1.0], dtype=np.float32)
            transformed = rot_mat @ pt
            tx, ty = float(transformed[0]), float(transformed[1])

            # Keep if within bounds
            if 0 <= tx < w and 0 <= ty < h:
                new_kps.append((tx, ty))
                new_classes.append(cls_id)

        # 2. Photometric transforms
        # Brightness & contrast
        alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
        beta = random.uniform(-self.brightness, self.brightness) * 255.0
        aug_img = np.clip(alpha * aug_img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Random Gaussian blur
        if random.random() < self.gaussian_blur_prob:
            ksize = random.choice([3, 5])
            aug_img = cv2.GaussianBlur(aug_img, (ksize, ksize), 0)

        return aug_img, new_kps, new_classes
