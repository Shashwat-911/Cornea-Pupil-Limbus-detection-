"""Loss functions for iris CNN models.

Provides Focal+Dice loss for iris segmentation and triplet loss for iris
embedding training. Follows the pattern of pupil_tracking/ml/losses.py.

Usage
-----
>>> from pupil_tracking.ml.iris_losses import IrisSegmentationLoss, IrisTripletLoss
>>> seg_loss = IrisSegmentationLoss(gamma=2.0, dice_weight=0.5)
>>> triplet_loss = IrisTripletLoss(margin=0.3)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False


class IrisSegmentationLoss:
    """Composite Focal + Dice loss for iris texture segmentation.

    Combines focal loss (handles class imbalance between usable iris and
    background) with Dice loss (optimizes region overlap).

    Parameters
    ----------
    gamma : float
        Focal loss focusing parameter. Higher = more focus on hard pixels.
    dice_weight : float
        Weight of Dice loss relative to focal loss.
    smooth : float
        Smoothing constant to avoid division by zero.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.smooth = smooth

        if _HAS_TORCH:
            self._ce = nn.CrossEntropyLoss(reduction="none")

    def __call__(
        self,
        logits: "torch.Tensor",
        targets: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute composite loss.

        Parameters
        ----------
        logits : (B, 2, H, W)
            Raw model output.
        targets : (B, H, W) long
            Binary masks (0 = non_iris, 1 = usable_iris).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for IrisSegmentationLoss")

        ce = self._ce(logits, targets)

        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = ((1 - pt) ** self.gamma) * ce
        focal_loss = focal.mean()

        usable_prob = probs[:, 1]
        target_float = targets.float()
        intersection = (usable_prob * target_float).sum()
        union = usable_prob.sum() + target_float.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        return focal_loss + self.dice_weight * dice_loss


class IrisTripletLoss:
    """Triplet loss with hard negative mining for iris embeddings.

    Given anchor (A), positive (P), and negative (N) embeddings:
        loss = max(0, d(A,P) - d(A,N) + margin)

    where d is squared Euclidean distance.

    Parameters
    ----------
    margin : float
        Minimum separation between positive and negative pairs.
    """

    def __init__(self, margin: float = 0.3):
        self.margin = margin

        if _HAS_TORCH:
            self._loss = nn.TripletMarginLoss(margin=margin, p=2)

    def __call__(
        self,
        anchors: "torch.Tensor",
        positives: "torch.Tensor",
        negatives: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute triplet loss.

        Parameters
        ----------
        anchors : (B, D)
        positives : (B, D)
        negatives : (B, D)

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for IrisTripletLoss")

        return self._loss(anchors, positives, negatives)


class IrisContrastiveLoss:
    """Contrastive loss for iris embeddings (alternative to triplet).

    Pulls same-iris pairs together and pushes different-iris pairs apart.

    Parameters
    ----------
    margin : float
        Minimum distance for negative pairs.
    """

    def __init__(self, margin: float = 1.0):
        self.margin = margin

    def __call__(
        self,
        embeddings_a: "torch.Tensor",
        embeddings_b: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute contrastive loss.

        Parameters
        ----------
        embeddings_a : (B, D)
        embeddings_b : (B, D)
        labels : (B,) float
            1.0 for same-iris pairs, 0.0 for different-iris pairs.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch required for IrisContrastiveLoss")

        dist = torch.norm(embeddings_a - embeddings_b, p=2, dim=1)

        pos_loss = labels * dist.pow(2)
        neg_loss = (1 - labels) * torch.clamp(self.margin - dist, min=0.0).pow(2)

        return (pos_loss + neg_loss).mean()


def hard_negative_mining(
    embeddings: "torch.Tensor",
    labels: "torch.Tensor",
    num_negatives: int = 1,
) -> tuple:
    """Select hard negatives for triplet loss.

    For each anchor, finds the negative sample with highest similarity
    (hardest negative).

    Parameters
    ----------
    embeddings : (N, D)
        All embeddings in the batch.
    labels : (N,)
        Class labels (iris identity).
    num_negatives : int
        Number of hard negatives per anchor.

    Returns
    -------
    tuple (anchors, positives, negatives) each (B, D)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch required for hard_negative_mining")

    n = embeddings.size(0)
    sim_matrix = torch.mm(embeddings, embeddings.t())

    anchors = []
    positives = []
    negatives = []

    for i in range(n):
        same_mask = labels == labels[i]
        diff_mask = labels != labels[i]

        if not diff_mask.any():
            continue

        same_indices = same_mask.nonzero(as_tuple=True)[0]
        pos_idx = same_indices[same_indices != i]
        if len(pos_idx) == 0:
            continue
        pos_idx = pos_idx[0]

        diff_indices = diff_mask.nonzero(as_tuple=True)[0]
        diff_sims = sim_matrix[i, diff_indices]
        _, hard_idx = diff_sims.topk(min(num_negatives, len(diff_indices)))

        anchors.append(embeddings[i])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[diff_indices[hard_idx[0]]])

    if not anchors:
        return (
            embeddings[:1],
            embeddings[:1],
            embeddings[:1],
        )

    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives),
    )
