"""
Loss functions for iris feature model training.

Implements:
1. KeypointFocalLoss: Modified focal loss for Gaussian peak heatmaps (CenterNet style).
2. DescriptorContrastiveLoss: Contrastive / triplet loss for dense descriptor matching.
3. IrisCompositeLoss: Balanced multi-task loss combining keypoint, descriptor, and class losses.
"""

from __future__ import annotations

from typing import Dict, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class KeypointFocalLoss(nn.Module):
        """Modified focal loss for penalty-reduced Gaussian heatmaps."""

        def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.eps = eps

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute focal loss between predicted and ground-truth heatmaps.

            Args:
                pred: [B, 1, H, W] in [0, 1]
                target: [B, 1, H, W] in [0, 1]
            """
            pred = torch.clamp(pred, self.eps, 1.0 - self.eps)

            pos_mask = target.ge(0.99)
            neg_mask = target.lt(0.99)

            pos_loss = -torch.pow(1.0 - pred, self.alpha) * torch.log(pred) * pos_mask
            neg_loss = (
                -torch.pow(1.0 - target, self.beta)
                * torch.pow(pred, self.alpha)
                * torch.log(1.0 - pred)
                * neg_mask
            )

            num_pos = pos_mask.float().sum()
            loss = pos_loss.sum() + neg_loss.sum()

            if num_pos > 0:
                loss = loss / num_pos
            return loss


    class DescriptorContrastiveLoss(nn.Module):
        """Contrastive loss enforcing similarity between corresponding keypoint descriptors."""

        def __init__(self, margin: float = 0.5):
            super().__init__()
            self.margin = margin

        def forward(
            self,
            desc_a: torch.Tensor,
            desc_b: torch.Tensor,
            matches_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Compute contrastive loss for matched descriptors.

            Args:
                desc_a: [N, D] normalized descriptors from image A
                desc_b: [N, D] normalized descriptors from image B
                matches_mask: [N] 1 for positive pairs, 0 for negative
            """
            if desc_a.numel() == 0 or desc_b.numel() == 0:
                return torch.tensor(0.0, device=desc_a.device)

            dist = torch.norm(desc_a - desc_b, p=2, dim=-1)
            pos_loss = matches_mask * torch.pow(dist, 2)
            neg_loss = (1.0 - matches_mask) * torch.pow(
                torch.clamp(self.margin - dist, min=0.0), 2
            )
            return torch.mean(pos_loss + neg_loss)


    class IrisCompositeLoss(nn.Module):
        """Multi-task loss aggregating keypoint, descriptor, and class predictions."""

        def __init__(
            self,
            keypoint_weight: float = 1.0,
            descriptor_weight: float = 0.5,
            class_weight: float = 0.3,
        ):
            super().__init__()
            self.keypoint_loss_fn = KeypointFocalLoss()
            self.descriptor_loss_fn = DescriptorContrastiveLoss()
            self.class_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

            self.keypoint_weight = keypoint_weight
            self.descriptor_weight = descriptor_weight
            self.class_weight = class_weight

        def forward(
            self,
            pred_heatmap: torch.Tensor,
            pred_desc: torch.Tensor,
            pred_class: torch.Tensor,
            target_heatmap: torch.Tensor,
            target_class: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """Compute composite training loss.

            Returns:
                total_loss: scalar loss for backward pass
                loss_dict: individual loss values for monitoring
            """
            loss_kp = self.keypoint_loss_fn(pred_heatmap, target_heatmap)

            # Squeeze or format target_class for CrossEntropyLoss
            if target_class.dim() == 4 and target_class.shape[1] == 1:
                target_class_flat = target_class.squeeze(1).long()
            else:
                target_class_flat = target_class.long()

            loss_cls = self.class_loss_fn(pred_class, target_class_flat)

            # Regularization on descriptors: unit length variance penalty
            norm = torch.norm(pred_desc, p=2, dim=1)
            loss_desc = torch.mean((norm - 1.0) ** 2)

            total_loss = (
                self.keypoint_weight * loss_kp
                + self.descriptor_weight * loss_desc
                + self.class_weight * loss_cls
            )

            metrics = {
                "loss_total": float(total_loss.item()),
                "loss_keypoint": float(loss_kp.item()),
                "loss_descriptor": float(loss_desc.item()),
                "loss_classification": float(loss_cls.item()),
            }

            return total_loss, metrics

else:
    class KeypointFocalLoss:
        pass
    class DescriptorContrastiveLoss:
        pass
    class IrisCompositeLoss:
        pass
