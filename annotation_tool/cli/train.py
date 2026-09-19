"""
CLI command to train the custom iris feature model.

Usage:
    python -m annotation_tool.cli.train --data-dir data/iris_dataset --epochs 50 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from annotation_tool.training.config import IrisTrainingConfig
from annotation_tool.training.dataset import IrisFeatureDataset
from annotation_tool.training.trainer import IrisTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train custom iris feature detection model")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing dataset and splits")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="outputs/iris_training", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("train_cli")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("PyTorch is required for model training. Please install torch and torchvision.")
        return 1

    config = IrisTrainingConfig(
        dataset_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    data_path = Path(args.data_dir)
    train_split = data_path / "train.json"
    val_split = data_path / "val.json"

    if not train_split.exists():
        logger.error("train.json not found in %s", data_path)
        return 1

    train_dataset = IrisFeatureDataset(train_split, image_size=config.image_size, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = None
    if val_split.exists():
        val_dataset = IrisFeatureDataset(val_split, image_size=config.image_size, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = IrisTrainer(config)
    trainer.fit(train_loader, val_loader)

    logger.info("Training finished. Checkpoints saved to %s", config.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
