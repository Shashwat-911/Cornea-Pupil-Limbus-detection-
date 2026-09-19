"""
Training pipeline for iris feature models.
"""

from annotation_tool.training.config import IrisTrainingConfig
from annotation_tool.training.dataset import IrisFeatureDataset
from annotation_tool.training.augmentations import IrisAugmentor
from annotation_tool.training.losses import (
    KeypointFocalLoss,
    DescriptorContrastiveLoss,
    IrisCompositeLoss,
)
from annotation_tool.training.trainer import IrisTrainer
from annotation_tool.training.validator import IrisModelValidator
from annotation_tool.training.export import export_to_onnx, export_to_torchscript

__all__ = [
    "IrisTrainingConfig",
    "IrisFeatureDataset",
    "IrisAugmentor",
    "KeypointFocalLoss",
    "DescriptorContrastiveLoss",
    "IrisCompositeLoss",
    "IrisTrainer",
    "IrisModelValidator",
    "export_to_onnx",
    "export_to_torchscript",
]
