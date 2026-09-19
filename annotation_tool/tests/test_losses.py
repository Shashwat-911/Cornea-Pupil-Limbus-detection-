"""Tests for iris feature training loss functions."""

import pytest

try:
    import torch
    from annotation_tool.training.losses import (
        KeypointFocalLoss,
        DescriptorContrastiveLoss,
        IrisCompositeLoss,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_keypoint_focal_loss():
    loss_fn = KeypointFocalLoss()
    pred = torch.full((1, 1, 64, 64), 0.5)
    target = torch.zeros((1, 1, 64, 64))
    target[0, 0, 32, 32] = 1.0

    loss = loss_fn(pred, target)
    assert loss.item() > 0.0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_iris_composite_loss():
    loss_fn = IrisCompositeLoss()

    b, c, h, w = 2, 5, 32, 32
    pred_hm = torch.sigmoid(torch.randn(b, 1, h, w))
    pred_desc = torch.randn(b, 128, h, w)
    pred_cls = torch.randn(b, c, h, w)

    target_hm = torch.zeros(b, 1, h, w)
    target_hm[:, :, 16, 16] = 1.0
    target_cls = torch.randint(0, c, (b, 1, h, w))

    total_loss, metrics = loss_fn(pred_hm, pred_desc, pred_cls, target_hm, target_cls)
    assert total_loss.item() > 0.0
    assert "loss_keypoint" in metrics
    assert "loss_descriptor" in metrics
    assert "loss_classification" in metrics
