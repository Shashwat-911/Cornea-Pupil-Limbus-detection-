"""
Training configuration for iris feature models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class IrisTrainingConfig:
    """Configuration for iris feature model training.

    Controls all training hyperparameters, data augmentation,
    model architecture, and output paths.
    """

    # Data
    dataset_dir: str = "data/iris_features"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 4

    # Model
    backbone: str = "dinov2_vits14"  # DINOv2 ViT-S/14
    lora_rank: int = 8
    lora_alpha: int = 16
    descriptor_dim: int = 128
    num_classes: int = 5  # bg, crypt, furrow, vessel, ink

    # Training
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5

    # Loss weights
    keypoint_loss_weight: float = 1.0
    descriptor_loss_weight: float = 0.5
    classification_loss_weight: float = 0.3

    # Augmentation
    augment_rotation_range: float = 360.0
    augment_scale_range: Tuple[float, float] = (0.8, 1.2)
    augment_brightness: float = 0.2
    augment_contrast: float = 0.2

    # Output
    output_dir: str = "outputs/iris_training"
    checkpoint_interval: int = 5

    # Export
    export_onnx: bool = True
    export_opset_version: int = 17
