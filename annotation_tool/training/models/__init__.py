"""
Model architectures for iris feature detection.
"""

from annotation_tool.training.models.backbone import DINOv2Backbone
from annotation_tool.training.models.heads import KeypointHead, DescriptorHead, ClassificationHead
from annotation_tool.training.models.lora_adapter import LoRALinear, apply_lora_to_model
from annotation_tool.training.models.iris_feature_model import IrisFeatureModel

__all__ = [
    "DINOv2Backbone",
    "KeypointHead",
    "DescriptorHead",
    "ClassificationHead",
    "LoRALinear",
    "apply_lora_to_model",
    "IrisFeatureModel",
]
