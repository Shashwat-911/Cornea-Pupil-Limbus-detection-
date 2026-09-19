"""
SAM 2 assisted annotation panel.

Provides one-click feature boundary prediction using Segment Anything
models. Degrades gracefully when SAM models are not installed.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    )
    from PyQt6.QtCore import pyqtSignal, QThread
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

import cv2

logger = logging.getLogger(__name__)


if HAS_PYQT6:

    class SAMAssistPanel(QWidget):
        """SAM 2 integration for one-click feature detection.

        Click a point → SAM predicts the feature boundary → extract centroid.
        """

        sam_predicted = pyqtSignal(dict)

        def __init__(self):
            super().__init__()
            self._build_ui()
            self.sam_model = None

        def _build_ui(self):
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel("<b>SAM 2 Assist</b>"))

            self.model_selector = QComboBox()
            self.model_selector.addItems([
                "EfficientSAM (fast)",
                "SAM 2 Tiny (balanced)",
                "MedSAM-2 (medical)",
            ])
            layout.addWidget(self.model_selector)

            self.load_btn = QPushButton("Load Model")
            self.load_btn.clicked.connect(self._load_sam)
            layout.addWidget(self.load_btn)

            self.status_label = QLabel("Not loaded")
            layout.addWidget(self.status_label)
            layout.addWidget(QLabel("Press [Space] on image to predict"))

        def _load_sam(self):
            model_type = self.model_selector.currentText()
            self.status_label.setText("Loading...")

            # Try to load — will fail gracefully if not installed
            try:
                self.loader = SAMLoaderThread(model_type)
                self.loader.finished_loading.connect(self._on_loaded)
                self.loader.start()
            except Exception as e:
                self.status_label.setText(f"Error: {e}")

        def _on_loaded(self, model):
            if model is not None:
                self.sam_model = model
                self.status_label.setText("✓ Loaded")
            else:
                self.status_label.setText("Failed to load")

        def run_prediction(self, image: np.ndarray,
                           click_point: Optional[tuple] = None):
            if self.sam_model is None:
                self.status_label.setText("Load model first!")
                return

            if click_point is None:
                return

            try:
                mask = self.sam_model.predict(image, point=click_point)
                keypoints = self._mask_to_keypoints(mask)
                self.sam_predicted.emit({
                    "mask": mask,
                    "keypoints": keypoints,
                    "source": "sam_assist",
                })
            except Exception as e:
                logger.error("SAM prediction failed: %s", e)
                self.status_label.setText(f"Prediction failed: {e}")

        @staticmethod
        def _mask_to_keypoints(mask):
            """Extract centroid from binary mask."""
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                return []
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return [(cx, cy)]
            return []


    class SAMLoaderThread(QThread):
        """Background thread for loading SAM models."""

        finished_loading = pyqtSignal(object)

        def __init__(self, model_type: str):
            super().__init__()
            self.model_type = model_type

        def run(self):
            model = None
            try:
                if "EfficientSAM" in self.model_type:
                    try:
                        from efficient_sam import EfficientSAM
                        model = EfficientSAM.load_pretrained()
                    except ImportError:
                        logger.warning("EfficientSAM not installed")
                elif "MedSAM" in self.model_type:
                    try:
                        from medsam2 import MedSAM2
                        model = MedSAM2.load_pretrained()
                    except ImportError:
                        logger.warning("MedSAM-2 not installed")
                else:
                    try:
                        from segment_anything_2 import SAM2Predictor
                        model = SAM2Predictor.load_pretrained("sam2_hiera_tiny")
                    except ImportError:
                        logger.warning("SAM 2 not installed")
            except Exception as e:
                logger.error("Failed to load SAM model: %s", e)

            self.finished_loading.emit(model)

else:
    class SAMAssistPanel:
        """Stub — PyQt6 not available."""
        def __init__(self):
            raise ImportError("PyQt6 required")
