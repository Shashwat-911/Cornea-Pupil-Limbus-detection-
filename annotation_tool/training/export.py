"""
Model exporter for trained iris feature models.

Converts trained PyTorch models to ONNX and TorchScript formats for low-latency
real-time inference in Stream E (`CustomFeatureStream`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: Tuple[int, int] = (224, 224),
    opset_version: int = 17,
    device: str = "cpu",
) -> Path:
    """Export PyTorch IrisFeatureModel to ONNX.

    Args:
        model: Trained IrisFeatureModel instance
        output_path: Path for output .onnx file
        input_size: (H, W) input resolution
        opset_version: ONNX opset version
        device: Device to run dummy inference

    Returns:
        Path to generated ONNX file.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for ONNX export.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    input_names = ["image"]
    output_names = ["heatmaps", "descriptors", "class_logits"]

    dynamic_axes = {
        "image": {0: "batch_size"},
        "heatmaps": {0: "batch_size"},
        "descriptors": {0: "batch_size"},
        "class_logits": {0: "batch_size"},
    }

    logger.info("Exporting model to ONNX format at %s...", out_path)

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("ONNX model successfully exported: %s", out_path)
    return out_path


def export_to_torchscript(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: Tuple[int, int] = (224, 224),
    device: str = "cpu",
) -> Path:
    """Export PyTorch IrisFeatureModel to TorchScript."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for TorchScript export.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)

    logger.info("Tracing model to TorchScript at %s...", out_path)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(str(out_path))

    logger.info("TorchScript model successfully saved: %s", out_path)
    return out_path
