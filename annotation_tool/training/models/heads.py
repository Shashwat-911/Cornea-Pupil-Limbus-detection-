"""
Task-specific prediction heads for the iris feature model.

1. KeypointHead: Regresses sub-pixel Gaussian keypoint heatmaps.
2. DescriptorHead: Generates dense L2-normalized feature descriptors for matching.
3. ClassificationHead: Classifies anatomical landmark type (crypt, furrow, vessel, ink).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class KeypointHead(nn.Module):
        """Heatmap prediction head with transposed convolutions for upsampling."""

        def __init__(self, in_channels: int = 384, out_channels: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
            out = self.net(x)
            if target_size is not None and (out.shape[2:] != target_size):
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            return out


    class DescriptorHead(nn.Module):
        """Dense descriptor head outputting unit-normalized embedding vectors."""

        def __init__(self, in_channels: int = 384, descriptor_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, descriptor_dim, kernel_size=1),
            )

        def forward(self, x: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
            out = self.net(x)
            if target_size is not None and (out.shape[2:] != target_size):
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            return F.normalize(out, p=2, dim=1)


    class ClassificationHead(nn.Module):
        """Pixel-wise or landmark-wise class prediction head."""

        def __init__(self, in_channels: int = 384, num_classes: int = 5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1),
            )

        def forward(self, x: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
            out = self.net(x)
            if target_size is not None and (out.shape[2:] != target_size):
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            return out

else:
    class KeypointHead:
        pass
    class DescriptorHead:
        pass
    class ClassificationHead:
        pass
