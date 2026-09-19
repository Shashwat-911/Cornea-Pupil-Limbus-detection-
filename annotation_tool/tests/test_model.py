"""Tests for IrisFeatureModel architecture and forward pass."""

import pytest

try:
    import torch
    from annotation_tool.training.models.iris_feature_model import IrisFeatureModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_iris_feature_model_forward():
    model = IrisFeatureModel(pretrained_backbone=False, use_lora=True)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        heatmaps, descriptors, class_logits = model(dummy_input)

    assert heatmaps.shape == (2, 1, 224, 224)
    assert descriptors.shape == (2, 128, 224, 224)
    assert class_logits.shape == (2, 5, 224, 224)
