"""
CLI command to export a trained iris feature model to ONNX or TorchScript.

Usage:
    python -m annotation_tool.cli.export --checkpoint outputs/iris_training/best_model.pth --output models/iris_features/iris_feature_model.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from annotation_tool.training.models.iris_feature_model import IrisFeatureModel
from annotation_tool.training.export import export_to_onnx, export_to_torchscript


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained iris model to ONNX/TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output file path (.onnx or .pt)")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--img-size", type=int, default=224, help="Square input size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("export_cli")

    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for exporting.")
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = IrisFeatureModel()
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    output_path = Path(args.output)
    if args.format == "onnx":
        export_to_onnx(model, output_path, input_size=(args.img_size, args.img_size))
    else:
        export_to_torchscript(model, output_path, input_size=(args.img_size, args.img_size))

    logger.info("Successfully exported model to %s", output_path)
    return 0


if __name__ == "__main__":
    main()
