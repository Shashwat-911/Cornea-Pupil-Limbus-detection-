"""
Main annotation window for the IXcentai Iris Feature Annotator.

Provides a PyQt6-based GUI with image list, annotation canvas,
tool selection, and SAM 2 assisted labeling.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QListWidget, QFileDialog,
        QMessageBox, QStatusBar, QSplitter,
    )
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QShortcut

    from annotation_tool.gui.canvas import AnnotationCanvas
    from annotation_tool.gui.toolbar import AnnotationToolbar
    from annotation_tool.gui.sam_assist import SAMAssistPanel
    from annotation_tool.io.dataset_manager import DatasetManager

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


if HAS_PYQT6:

    class AnnotationMainWindow(QMainWindow):
        """Main window for iris feature annotation.

        Layout: [Image List | Annotation Canvas | Tools + SAM Assist]
        """

        def __init__(self):
            super().__init__()
            self.setWindowTitle("IXcentai Iris Feature Annotator")
            self.setGeometry(100, 100, 1600, 1000)

            self.dataset_manager = DatasetManager()
            self.current_annotation = None

            self._build_ui()
            self._setup_shortcuts()
            self._connect_signals()

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            self._build_menu()

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # LEFT: Image list
            self.image_list = QListWidget()
            self.image_list.setMinimumWidth(200)
            splitter.addWidget(self.image_list)

            # CENTER: Canvas
            self.canvas = AnnotationCanvas()
            splitter.addWidget(self.canvas)

            # RIGHT: Tools + SAM assist
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            right_panel.setMaximumWidth(300)

            self.toolbar = AnnotationToolbar()
            right_layout.addWidget(self.toolbar)

            self.sam_panel = SAMAssistPanel()
            right_layout.addWidget(self.sam_panel)
            right_layout.addStretch()

            splitter.addWidget(right_panel)
            splitter.setSizes([200, 1100, 300])

            layout = QHBoxLayout(central)
            layout.addWidget(splitter)

            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)

        def _build_menu(self):
            menubar = self.menuBar()

            file_menu = menubar.addMenu("&File")
            file_menu.addAction("Load Image Folder...", self.load_folder, "Ctrl+O")
            file_menu.addAction("Save Annotation", self.save_current, "Ctrl+S")
            file_menu.addSeparator()
            file_menu.addAction("Exit", self.close, "Ctrl+Q")

            dataset_menu = menubar.addMenu("&Dataset")
            dataset_menu.addAction("Split Train/Val/Test", self.create_splits)
            dataset_menu.addAction("Dataset Statistics", self.show_stats)
            dataset_menu.addAction("Validate Annotations", self.validate_all)

            export_menu = menubar.addMenu("&Export")
            export_menu.addAction("Export for Training", self.export_dataset)

        def _setup_shortcuts(self):
            QShortcut(QKeySequence("1"), self, lambda: self.toolbar.select_tool("crypt"))
            QShortcut(QKeySequence("2"), self, lambda: self.toolbar.select_tool("furrow"))
            QShortcut(QKeySequence("3"), self, lambda: self.toolbar.select_tool("vessel"))
            QShortcut(QKeySequence("4"), self, lambda: self.toolbar.select_tool("ink"))
            QShortcut(QKeySequence("5"), self, lambda: self.toolbar.select_tool("collarette_junction"))
            QShortcut(QKeySequence("N"), self, self.next_image)
            QShortcut(QKeySequence("P"), self, self.prev_image)
            QShortcut(QKeySequence("Delete"), self, self.canvas.delete_selected)
            QShortcut(QKeySequence("Z"), self, self.canvas.undo)
            QShortcut(QKeySequence("Ctrl+Z"), self, self.canvas.undo)
            QShortcut(QKeySequence("Space"), self, self.trigger_sam_assist)

        def _connect_signals(self):
            self.image_list.currentRowChanged.connect(self.on_image_selected)
            self.canvas.annotation_changed.connect(self.on_annotation_changed)
            self.toolbar.tool_changed.connect(self.canvas.set_tool)
            self.sam_panel.sam_predicted.connect(self.canvas.add_sam_prediction)

        def load_folder(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
            if folder:
                self.dataset_manager.load_folder(Path(folder))
                self.image_list.clear()
                for img_path in self.dataset_manager.image_paths:
                    self.image_list.addItem(img_path.name)
                self._update_status()

        def on_image_selected(self, row):
            if row < 0:
                return
            img_path = self.dataset_manager.image_paths[row]
            if self.current_annotation and self.current_annotation.modified:
                self.save_current()
            self.current_annotation = self.dataset_manager.load_annotation(img_path)
            self.canvas.load_image_and_annotation(img_path, self.current_annotation)

        def on_annotation_changed(self):
            if self.current_annotation:
                self.current_annotation.modified = True
                self._update_status()

        def save_current(self):
            if self.current_annotation:
                self.dataset_manager.save_annotation(self.current_annotation)
                self.current_annotation.modified = False
                self.status_bar.showMessage(
                    f"Saved: {self.current_annotation.image_path.name}", 2000
                )

        def next_image(self):
            row = self.image_list.currentRow()
            if row < self.image_list.count() - 1:
                self.image_list.setCurrentRow(row + 1)

        def prev_image(self):
            row = self.image_list.currentRow()
            if row > 0:
                self.image_list.setCurrentRow(row - 1)

        def trigger_sam_assist(self):
            if self.canvas.current_image is not None:
                self.sam_panel.run_prediction(
                    self.canvas.current_image,
                    self.canvas.last_click_point,
                )

        def create_splits(self):
            self.dataset_manager.create_splits(train=0.7, val=0.15, test=0.15)
            QMessageBox.information(
                self, "Splits Created",
                f"Train: {len(self.dataset_manager.splits.get('train', []))}\n"
                f"Val: {len(self.dataset_manager.splits.get('val', []))}\n"
                f"Test: {len(self.dataset_manager.splits.get('test', []))}",
            )

        def show_stats(self):
            stats = self.dataset_manager.compute_statistics()
            msg = "\n".join([f"{k}: {v}" for k, v in stats.items()])
            QMessageBox.information(self, "Dataset Statistics", msg)

        def validate_all(self):
            errors = self.dataset_manager.validate_all_annotations()
            if errors:
                QMessageBox.warning(
                    self, "Validation Errors",
                    f"Found {len(errors)} issues:\n" + "\n".join(errors[:10]),
                )
            else:
                QMessageBox.information(self, "Validation", "All annotations valid ✓")

        def export_dataset(self):
            out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
            if out_dir:
                self.dataset_manager.export_for_training(Path(out_dir))
                QMessageBox.information(self, "Exported", f"Dataset exported to {out_dir}")

        def _update_status(self):
            n_annotated = self.dataset_manager.count_annotated()
            n_total = len(self.dataset_manager.image_paths)
            n_keypoints = self.dataset_manager.count_keypoints()
            self.status_bar.showMessage(
                f"{n_annotated}/{n_total} images annotated | {n_keypoints} keypoints total"
            )

else:
    class AnnotationMainWindow:
        """Stub — PyQt6 not available."""
        def __init__(self):
            raise ImportError(
                "PyQt6 is required. Install with: pip install PyQt6>=6.7.0"
            )
