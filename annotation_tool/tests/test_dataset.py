"""Tests for IrisFeatureDataset and data loading."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

try:
    import torch
    from annotation_tool.training.dataset import IrisFeatureDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_iris_feature_dataset(tmp_path: Path):
    # Create dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "eye1.png"
    cv2.imwrite(str(img_path), img)

    # Create dummy annotation
    ann = {
        "image_path": str(img_path),
        "image_width": 100,
        "image_height": 100,
        "keypoints": [{"type": "iris_crypt", "x": 50.0, "y": 50.0}],
        "segments": [],
        "ink_marks": [],
    }
    ann_path = tmp_path / "eye1.json"
    with open(ann_path, "w") as f:
        json.dump(ann, f)

    # Create split file
    split_data = [{"image_path": str(img_path), "annotation_path": str(ann_path)}]
    split_file = tmp_path / "train.json"
    with open(split_file, "w") as f:
        json.dump(split_data, f)

    dataset = IrisFeatureDataset(split_file, image_size=128)
    assert len(dataset) == 1

    sample = dataset[0]
    assert "image" in sample
    assert "heatmap" in sample
    assert "class_map" in sample
    assert sample["image"].shape == (3, 128, 128)
    assert sample["heatmap"].shape == (1, 128, 128)
