"""
Annotation canvas — zoomable, pannable image widget.

Uses QGraphicsView for smooth rendering with annotation overlays.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
    from PyQt6.QtCore import Qt, pyqtSignal, QRectF
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


if HAS_PYQT6:

    class AnnotationCanvas(QGraphicsView):
        """Interactive annotation canvas with zoom, pan, and annotation tools.

        - Scroll wheel to zoom
        - Middle-drag to pan
        - Left-click to add annotation (based on selected tool)
        - Right-click to delete nearest
        """

        annotation_changed = pyqtSignal()

        TOOL_COLOURS = {
            "crypt": QColor(255, 100, 100),
            "furrow": QColor(100, 255, 100),
            "vessel": QColor(100, 100, 255),
            "ink": QColor(200, 100, 200),
            "collarette_junction": QColor(255, 200, 0),
        }

        def __init__(self):
            super().__init__()
            self.scene = QGraphicsScene()
            self.setScene(self.scene)
            self.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setTransformationAnchor(
                QGraphicsView.ViewportAnchor.AnchorUnderMouse
            )

            self.current_image = None
            self.current_annotation = None
            self.current_tool = "crypt"
            self.pixmap_item = None
            self.annotation_items = []
            self.last_click_point = None
            self.undo_stack = []

        def load_image_and_annotation(self, image_path, annotation):
            """Load image and its annotation for editing."""
            img = cv2.imread(str(image_path))
            if img is None:
                return

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img_rgb
            self.current_annotation = annotation

            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(
                img_rgb.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888,
            )
            pixmap = QPixmap.fromImage(qimg)

            self.scene.clear()
            self.annotation_items.clear()
            self.pixmap_item = self.scene.addPixmap(pixmap)
            self.setSceneRect(QRectF(pixmap.rect().toRectF()))
            self.fitInView(
                self.pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio,
            )

            self._render_annotations()

        def set_tool(self, tool_name: str):
            self.current_tool = tool_name

        def wheelEvent(self, event):
            zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(zoom_factor, zoom_factor)

        def mousePressEvent(self, event):
            scene_pos = self.mapToScene(event.pos())
            self.last_click_point = (scene_pos.x(), scene_pos.y())

            if event.button() == Qt.MouseButton.LeftButton:
                self._add_annotation_at(scene_pos.x(), scene_pos.y())
            elif event.button() == Qt.MouseButton.MiddleButton:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            elif event.button() == Qt.MouseButton.RightButton:
                self._select_or_delete_at(scene_pos.x(), scene_pos.y())

            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.MouseButton.MiddleButton:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            super().mouseReleaseEvent(event)

        def _add_annotation_at(self, x, y):
            if self.current_annotation is None:
                return

            self.undo_stack.append(self.current_annotation.copy())
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)

            if self.current_tool in ["crypt", "vessel", "collarette_junction"]:
                self.current_annotation.add_keypoint(
                    type=self.current_tool, x=x, y=y
                )
            elif self.current_tool == "ink":
                self.current_annotation.add_ink_mark(
                    center_x=x, center_y=y, radius=8.0
                )

            self._render_annotations()
            self.annotation_changed.emit()

        def _select_or_delete_at(self, x, y):
            if self.current_annotation is None:
                return
            deleted = self.current_annotation.delete_nearest(x, y, threshold=10.0)
            if deleted:
                self._render_annotations()
                self.annotation_changed.emit()

        def _render_annotations(self):
            for item in self.annotation_items:
                self.scene.removeItem(item)
            self.annotation_items.clear()

            if self.current_annotation is None:
                return

            for kp in self.current_annotation.keypoints:
                colour = self.TOOL_COLOURS.get(
                    kp.get("type", ""), QColor(255, 255, 255)
                )
                pen = QPen(colour, 2)
                brush = QBrush(colour)
                item = self.scene.addEllipse(
                    kp["x"] - 4, kp["y"] - 4, 8, 8, pen, brush
                )
                self.annotation_items.append(item)

            for mark in self.current_annotation.ink_marks:
                pen = QPen(self.TOOL_COLOURS["ink"], 3)
                r = mark.get("radius", 8.0)
                item = self.scene.addEllipse(
                    mark["center_x"] - r,
                    mark["center_y"] - r,
                    r * 2, r * 2, pen,
                )
                self.annotation_items.append(item)

        def add_sam_prediction(self, sam_result: dict):
            for kp in sam_result.get("keypoints", []):
                self._add_annotation_at(kp[0], kp[1])

        def undo(self):
            if self.undo_stack:
                self.current_annotation = self.undo_stack.pop()
                self._render_annotations()
                self.annotation_changed.emit()

        def delete_selected(self):
            if self.last_click_point:
                self._select_or_delete_at(*self.last_click_point)

else:
    # Stub when PyQt6 is not installed
    class AnnotationCanvas:
        """Stub — PyQt6 not available."""
        def __init__(self):
            raise ImportError(
                "PyQt6 is required for the annotation canvas. "
                "Install it with: pip install PyQt6>=6.7.0"
            )
