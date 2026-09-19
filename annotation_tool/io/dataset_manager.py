"""
Dataset management for iris feature annotations.

Handles image discovery, annotation loading/saving, train/val/test
split creation, statistics computation, and dataset export.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annotation_tool.io.annotation_format import AnnotationFile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class DatasetManager:
    """Manages annotation datasets for iris feature labeling.

    Responsibilities:
        - Discover images in a folder
        - Load/save annotations alongside images
        - Create train/val/test splits
        - Compute statistics
        - Export for training
    """

    def __init__(self):
        self.image_paths: List[Path] = []
        self.annotations: Dict[str, AnnotationFile] = {}
        self.annotation_dir: Optional[Path] = None
        self.splits: Dict[str, List[str]] = {}

    def load_folder(self, folder: Path):
        """Discover images in a folder and load any existing annotations."""
        self.image_paths = sorted([
            p for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        # Look for annotations in a sibling 'annotations' directory
        self.annotation_dir = folder.parent / "annotations"
        self.annotation_dir.mkdir(parents=True, exist_ok=True)

        # Load existing annotations
        self.annotations.clear()
        for img_path in self.image_paths:
            ann_path = self._annotation_path(img_path)
            if ann_path.exists():
                try:
                    self.annotations[img_path.name] = AnnotationFile.load(ann_path)
                except Exception:
                    pass

    def load_annotation(self, img_path: Path) -> AnnotationFile:
        """Load or create annotation for an image."""
        if img_path.name in self.annotations:
            return self.annotations[img_path.name]

        ann_path = self._annotation_path(img_path)
        if ann_path.exists():
            annotation = AnnotationFile.load(ann_path)
        else:
            annotation = AnnotationFile(
                image_path=img_path,
            )

        self.annotations[img_path.name] = annotation
        return annotation

    def save_annotation(self, annotation: AnnotationFile):
        """Save an annotation to disk."""
        if annotation.image_path is None:
            return
        ann_path = self._annotation_path(annotation.image_path)
        annotation.save(ann_path)

    def count_annotated(self) -> int:
        """Count images that have at least one annotation."""
        count = 0
        for img_path in self.image_paths:
            ann = self.annotations.get(img_path.name)
            if ann and (ann.keypoints or ann.segments or ann.ink_marks):
                count += 1
        return count

    def count_keypoints(self) -> int:
        """Count total keypoints across all annotations."""
        total = 0
        for ann in self.annotations.values():
            total += len(ann.keypoints)
        return total

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        stats = {
            "total_images": len(self.image_paths),
            "annotated_images": self.count_annotated(),
            "total_keypoints": self.count_keypoints(),
            "total_segments": sum(len(a.segments) for a in self.annotations.values()),
            "total_ink_marks": sum(len(a.ink_marks) for a in self.annotations.values()),
        }

        # Per-type counts
        type_counts: Dict[str, int] = {}
        for ann in self.annotations.values():
            for kp in ann.keypoints:
                t = kp.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
        stats["keypoint_types"] = type_counts

        return stats

    def create_splits(
        self,
        train: float = 0.7,
        val: float = 0.15,
        test: float = 0.15,
        seed: int = 42,
    ):
        """Create train/val/test splits from annotated images.

        Only includes images that have annotations.
        """
        annotated = [
            p.name for p in self.image_paths
            if p.name in self.annotations and (
                self.annotations[p.name].keypoints or
                self.annotations[p.name].segments or
                self.annotations[p.name].ink_marks
            )
        ]

        random.seed(seed)
        random.shuffle(annotated)

        n = len(annotated)
        n_train = int(n * train)
        n_val = int(n * val)

        self.splits = {
            "train": annotated[:n_train],
            "val": annotated[n_train:n_train + n_val],
            "test": annotated[n_train + n_val:],
        }

        # Save splits
        if self.annotation_dir:
            splits_dir = self.annotation_dir.parent / "splits"
            splits_dir.mkdir(parents=True, exist_ok=True)
            for split_name, items in self.splits.items():
                split_file = splits_dir / f"{split_name}.json"
                with open(split_file, "w") as f:
                    json.dump(items, f, indent=2)

    def validate_all_annotations(self) -> List[str]:
        """Validate all annotations. Returns list of errors."""
        all_errors = []
        for name, ann in self.annotations.items():
            errors = ann.validate()
            for err in errors:
                all_errors.append(f"{name}: {err}")
        return all_errors

    def export_for_training(self, output_dir: Path):
        """Export dataset in a format ready for the training pipeline.

        Creates a structured directory with images, annotations, and
        split files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        images_dir = output_dir / "images"
        annotations_dir = output_dir / "annotations"
        splits_dir = output_dir / "splits"
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        splits_dir.mkdir(exist_ok=True)

        # Copy images and annotations
        import shutil
        for img_path in self.image_paths:
            if img_path.name not in self.annotations:
                continue
            ann = self.annotations[img_path.name]
            if not (ann.keypoints or ann.segments or ann.ink_marks):
                continue

            # Copy image
            shutil.copy2(str(img_path), str(images_dir / img_path.name))

            # Save annotation
            ann_path = annotations_dir / f"{img_path.stem}.json"
            ann.save(ann_path)

        # Create split files with full paths
        for split_name, items in self.splits.items():
            samples = []
            for name in items:
                img_path = images_dir / name
                ann_path = annotations_dir / f"{Path(name).stem}.json"
                samples.append({
                    "image_path": str(img_path),
                    "annotation_path": str(ann_path),
                })
            with open(splits_dir / f"{split_name}.json", "w") as f:
                json.dump(samples, f, indent=2)

    def _annotation_path(self, img_path: Path) -> Path:
        """Get annotation file path for an image."""
        if self.annotation_dir is None:
            return img_path.parent / "annotations" / f"{img_path.stem}.json"
        return self.annotation_dir / f"{img_path.stem}.json"
