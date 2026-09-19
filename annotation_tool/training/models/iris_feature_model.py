"""
Integrated Iris Feature Model architecture.

Combines DINOv2 backbone, LoRA adapters, and multi-task prediction heads:
- Keypoint heatmap regression
- Dense L2-normalized feature descriptors
- Feature class prediction
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from annotation_tool.training.models.backbone import DINOv2Backbone
from annotation_tool.training.models.heads import KeypointHead, DescriptorHead, ClassificationHead
from annotation_tool.training.models.lora_adapter import apply_lora_to_model

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class IrisFeatureModel(nn.Module):
        """Unified multi-task model for iris landmark detection and descriptor matching."""

        def __init__(
            self,
            backbone_name: str = "dinov2_vits14",
            descriptor_dim: int = 128,
            num_classes: int = 5,
            lora_rank: int = 8,
            lora_alpha: float = 16.0,
            use_lora: bool = True,
            pretrained_backbone: bool = True,
        ):
            super().__init__()
            self.backbone = DINOv2Backbone(model_name=backbone_name, pretrained=pretrained_backbone)
            embed_dim = self.backbone.embed_dim

            if use_lora:
                num_adapted = apply_lora_to_model(self.backbone, rank=lora_rank, alpha=lora_alpha)
                logger.info("Applied LoRA to %d layers in backbone", num_adapted)

            self.keypoint_head = KeypointHead(in_channels=embed_dim, out_channels=1)
            self.descriptor_head = DescriptorHead(in_channels=embed_dim, descriptor_dim=descriptor_dim)
            self.class_head = ClassificationHead(in_channels=embed_dim, num_classes=num_classes)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward pass for multi-task prediction.

            Args:
                x: Input tensor of shape [B, 3, H, W]

            Returns:
                heatmaps: [B, 1, H, W] in [0, 1]
                descriptors: [B, D, H, W] L2 normalized
                class_logits: [B, num_classes, H, W]
            """
            target_size = (x.shape[2], x.shape[3])
            features = self.backbone(x)

            heatmaps = self.keypoint_head(features, target_size=target_size)
            descriptors = self.descriptor_head(features, target_size=target_size)
            class_logits = self.class_head(features, target_size=target_size)

            return heatmaps, descriptors, class_logits

else:
    class IrisFeatureModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for IrisFeatureModel.")
