"""
CLI command to launch the IXcentai Iris Feature Annotation GUI.

Usage:
    python -m annotation_tool.cli.annotate --image-dir data/raw_images --annotation-dir data/annotations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IXcentai Standalone Iris Feature Annotation Tool")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing iris images to annotate")
    parser.add_argument("--annotation-dir", type=str, default=None, help="Directory to load/save JSON annotations")
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="Optional path to SAM/SAM 2 checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        from PyQt6.QtWidgets import QApplication
        from annotation_tool.gui.main_window import AnnotationMainWindow
    except ImportError as e:
        print(f"Error: PyQt6 is required to run the annotation GUI.\n{e}")
        return 1

    app = QApplication(sys.argv)
    window = AnnotationMainWindow()

    if args.image_dir:
        img_path = Path(args.image_dir)
        if img_path.exists():
            window.dataset_mgr.set_image_dir(img_path)
            window.refresh_image_list()

    if args.annotation_dir:
        ann_path = Path(args.annotation_dir)
        ann_path.mkdir(parents=True, exist_ok=True)
        window.dataset_mgr.set_annotation_dir(ann_path)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
