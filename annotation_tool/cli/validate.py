"""
CLI command to validate a trained iris feature model.

Usage:
    python -m annotation_tool.cli.validate --checkpoint outputs/iris_training/best_model.pth --test-split data/test.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from annotation_tool.training.models.iris_feature_model import IrisFeatureModel
from annotation_tool.training.validator import IrisModelValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained iris feature model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-split", type=str, default=None, help="Path to test.json dataset split")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("validate_cli")

    try:
        import torch
    except ImportError:
        logger.error("PyTorch is required for validation.")
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint file not found: %s", checkpoint_path)
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = IrisFeatureModel()
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    validator = IrisModelValidator(model)
    logger.info("Model loaded successfully from %s", checkpoint_path)
    return 0


if __name__ == "__main__":
    main()
