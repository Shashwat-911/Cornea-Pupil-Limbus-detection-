"""Differentiable deep learning architectures for iris cyclotorsion registration.

This module provides:
1. CircularConv2d: Circular-padding convolution along the 0-360 deg angular axis.
2. IrisPolarFeatureNet: Multi-scale rotation-equivariant iris feature extractor.
3. IrisPolarSiameseNet: End-to-end trainable Siamese cyclotorsion estimator.
4. Model-to-Matcher adapter providing seamless integration with SittingRegistrationSession.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .sitting import AngularMatch

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


if _HAS_TORCH:

    class CircularConv2d(nn.Module):
        """2D convolution with circular boundary wrapping along the angular (width) axis."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
        ) -> None:
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            if isinstance(stride, int):
                sh, sw = stride, stride
            else:
                sh, sw = stride
            if isinstance(dilation, int):
                dh, dw = dilation, dilation
            else:
                dh, dw = dilation

            self.pad_h = (kh - 1) * dh // 2
            self.pad_w = (kw - 1) * dw // 2
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                (kh, kw),
                stride=(sh, sw),
                dilation=(dh, dw),
                groups=groups,
                bias=bias,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Replicate padding for radial height, circular padding for angular width
            if self.pad_h > 0:
                x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="replicate")
            if self.pad_w > 0:
                x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
            return self.conv(x)

    class IrisPolarFeatureNet(nn.Module):
        """Extracts dense angular feature embeddings from unwrapped polar iris strips."""

        def __init__(
            self,
            in_channels: int = 1,
            embedding_dim: int = 64,
            base_channels: int = 32,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim

            self.stem = nn.Sequential(
                CircularConv2d(in_channels, base_channels, (5, 5)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )

            # Stage 1: Preserve angular resolution, compress radial dimension
            self.stage1 = nn.Sequential(
                CircularConv2d(base_channels, base_channels * 2, (3, 3)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce radius by 2
            )

            # Stage 2: Aggregate mid-frequency crypts & furrows
            self.stage2 = nn.Sequential(
                CircularConv2d(base_channels * 2, base_channels * 4, (3, 3)),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # Reduce radius by 2
            )

            # Stage 3: Pool across radial dimension to produce angular descriptor column
            self.head = nn.Sequential(
                CircularConv2d(base_channels * 4, embedding_dim, (3, 3)),
                nn.BatchNorm2d(embedding_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveAvgPool2d((1, None)),  # (B, embedding_dim, 1, W)
            )

        def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            if mask is not None:
                x = x * mask
            feat = self.stem(x)
            feat = self.stage1(feat)
            feat = self.stage2(feat)
            emb = self.head(feat).squeeze(2)  # (B, embedding_dim, W)
            return F.normalize(emb, p=2, dim=1)

    class IrisPolarSiameseNet(nn.Module):
        """End-to-end Siamese model estimating cyclotorsion between paired iris strips."""

        def __init__(
            self,
            num_angles: int = 720,
            max_degrees: float = 30.0,
            embedding_dim: int = 64,
            temperature: float = 0.05,
        ) -> None:
            super().__init__()
            self.num_angles = num_angles
            self.max_degrees = max_degrees
            self.temperature = temperature
            self.feature_net = IrisPolarFeatureNet(embedding_dim=embedding_dim)

            # Precomputed angular shifts
            max_bins = int(np.floor(max_degrees * num_angles / 360.0))
            self.max_bins = max_bins
            shifts = np.arange(num_angles, dtype=np.float32)
            shifts[shifts > num_angles / 2] -= num_angles
            self.register_buffer("shifts_deg", torch.from_numpy(shifts * 360.0 / num_angles))

        def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self.feature_net(x, mask)

        def correlate(
            self, emb_ref: torch.Tensor, emb_curr: torch.Tensor
        ) -> torch.Tensor:
            """Compute circular cross-correlation between two angular embeddings via 1D FFT."""
            # emb: (B, C, W)
            f_ref = torch.fft.rfft(emb_ref, dim=-1)
            f_curr = torch.fft.rfft(emb_curr, dim=-1)
            # Dot product across channel dimension
            prod = torch.sum(f_ref * torch.conj(f_curr), dim=1)  # (B, W_freq)
            corr = torch.fft.irfft(prod, n=self.num_angles, dim=-1)  # (B, W)
            # Normalization: divide by sequence length W and embedding dimension
            corr = corr / (self.num_angles * max(emb_ref.size(1), 1))
            return torch.clamp(corr, -1.0, 1.0)

        def forward(
            self,
            ref: torch.Tensor,
            curr: torch.Tensor,
            mask_ref: Optional[torch.Tensor] = None,
            mask_curr: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            emb_ref = self.encode(ref, mask_ref)
            emb_curr = self.encode(curr, mask_curr)
            corr = self.correlate(emb_ref, emb_curr)  # (B, W)

            # Mask shifts outside [-max_degrees, +max_degrees]
            allowed_mask = torch.abs(self.shifts_deg) <= self.max_degrees
            corr_masked = torch.where(allowed_mask.unsqueeze(0), corr, torch.tensor(-1.0, device=corr.device))

            # Peak extraction & soft-argmax around peak
            max_scores, peak_idx = torch.max(corr_masked, dim=-1)

            # Differentiable soft-argmax over a localized window around the peak
            weights = F.softmax(corr_masked / self.temperature, dim=-1)
            pred_deg = torch.sum(weights * self.shifts_deg.unsqueeze(0), dim=-1)

            # Peak margin vs secondary peaks
            dist = torch.abs(
                (torch.arange(self.num_angles, device=corr.device).unsqueeze(0) - peak_idx.unsqueeze(1) + self.num_angles // 2)
                % self.num_angles - self.num_angles // 2
            )
            competing_mask = allowed_mask.unsqueeze(0) & (dist > 4)
            secondary_scores = torch.where(
                competing_mask, corr_masked, torch.tensor(-1.0, device=corr.device)
            )
            max_secondary, _ = torch.max(secondary_scores, dim=-1)
            peak_margin = max_scores - max_secondary

            return {
                "pred_deg": pred_deg,
                "peak_idx": peak_idx,
                "score": max_scores,
                "peak_margin": peak_margin,
                "corr": corr_masked,
            }

        def estimate_rotation(
            self,
            ref_strip: np.ndarray,
            curr_strip: np.ndarray,
            mask_ref: Optional[np.ndarray] = None,
            mask_curr: Optional[np.ndarray] = None,
            max_degrees: float = 30.0,
            *,
            reference_cache: Optional[dict] = None,
            **kwargs: Any,
        ) -> AngularMatch:
            """Inference method returning AngularMatch for session integration."""
            device = next(self.parameters()).device
            t_ref = (
                torch.from_numpy(ref_strip.astype(np.float32) / 127.5 - 1.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            t_curr = (
                torch.from_numpy(curr_strip.astype(np.float32) / 127.5 - 1.0)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            t_mr = (
                torch.from_numpy((mask_ref > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                if mask_ref is not None
                else None
            )
            t_mc = (
                torch.from_numpy((mask_curr > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                if mask_curr is not None
                else None
            )

            self.eval()
            with torch.no_grad():
                out = self.forward(t_ref, t_curr, t_mr, t_mc)
                angle = float(out["pred_deg"][0].cpu().item())
                score = float(np.clip(out["score"][0].cpu().item(), -1.0, 1.0))
                margin = float(max(0.0, out["peak_margin"][0].cpu().item()))

            overlap = 1.0
            if mask_ref is not None and mask_curr is not None:
                overlap = float(np.mean((mask_ref > 0) & (mask_curr > 0)))

            valid = abs(angle) < max_degrees and score >= 0.35 and margin >= 0.03 and overlap >= 0.30
            return AngularMatch(
                angle_deg=angle,
                valid=valid,
                score=score,
                overlap=overlap,
                peak_margin=margin,
                band_spread_deg=0.0,
                reason="" if valid else "learned_model_rejection",
            )

        def as_matcher(self, device: str = "cpu"):
            """Return an AngularMatcher callable wrapping this model for session registration."""
            from .matcher import LearnedAngularMatcher

            self.to(device)
            return LearnedAngularMatcher(model=self, name="IrisPolarSiameseMatcher")

        def export_onnx(self, output_path: Union[str, Path]) -> str:
            """Export model to ONNX for CPU / DirectML production deployment."""
            import sys

            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            output_path = Path(output_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval()

            # Export feature extraction backbone for ultra-fast, robust inference
            dummy = torch.randn(1, 1, 64, self.num_angles)
            torch.onnx.export(
                self.feature_net,
                dummy,
                str(output_path),
                input_names=["polar_strip"],
                output_names=["angular_embedding"],
                opset_version=18,
                dynamic_axes={
                    "polar_strip": {0: "batch_size"},
                    "angular_embedding": {0: "batch_size"},
                },
            )
            logger.info("Exported feature net ONNX to %s", output_path)
            return str(output_path)

else:
    # Fallback stubs when PyTorch is not installed
    class IrisPolarSiameseNet:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for IrisPolarSiameseNet. Install torch.")
