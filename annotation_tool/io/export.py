"""Export utilities for training datasets."""

from __future__ import annotations

from pathlib import Path
from annotation_tool.io.dataset_manager import DatasetManager


def export_dataset(dataset_manager: DatasetManager, output_dir: Path):
    """Export the current dataset for training."""
    dataset_manager.export_for_training(output_dir)
