"""
Backbone feature extractor for iris feature models.

Supports DINOv2 vision transformer with hierarchical / dense feature extraction,
with lightweight CNN fallback for offline or low-compute environments.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class DINOv2Backbone(nn.Module):
        """DINOv2 ViT backbone with feature extraction at patch resolution."""

        def __init__(self, model_name: str = "dinov2_vits14", pretrained: bool = True):
            super().__init__()
            self.model_name = model_name
            self.embed_dim = 384  # ViT-S default
            self.patch_size = 14

            self.backbone = None
            if pretrained:
                try:
                    self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
                    logger.info("Loaded pretrained DINOv2 backbone: %s", model_name)
                except Exception as e:
                    logger.warning("Could not download DINOv2 backbone via torch.hub (%s). Using CNN fallback.", e)

            if self.backbone is None:
                # Lightweight convolutional fallback with matching feature interface
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.embed_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.embed_dim),
                    nn.ReLU(inplace=True),
                )
                self.patch_size = 16

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Extract dense 2D feature map.

            Args:
                x: [B, 3, H, W]

            Returns:
                feature_map: [B, C, H', W']
            """
            b, c, h, w = x.shape
            if isinstance(self.backbone, nn.Sequential):
                return self.backbone(x)

            # DINOv2 hub model
            try:
                features_dict = self.backbone.forward_features(x)
                # patch tokens shape: [B, N, D]
                patch_tokens = features_dict["x_norm_patchtokens"]
                patch_h = h // self.patch_size
                patch_w = w // self.patch_size
                feat = patch_tokens.reshape(b, patch_h, patch_w, self.embed_dim)
                return feat.permute(0, 3, 1, 2).contiguous()
            except Exception:
                # Fallback directly to intermediate layers
                tokens = self.backbone.get_intermediate_layers(x, n=1)[0]
                patch_h = h // self.patch_size
                patch_w = w // self.patch_size
                feat = tokens.reshape(b, patch_h, patch_w, self.embed_dim)
                return feat.permute(0, 3, 1, 2).contiguous()

else:
    class DINOv2Backbone:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DINOv2Backbone.")
