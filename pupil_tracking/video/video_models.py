# pupil_tracking/video/video_models.py
"""Data models, constants, and utility functions extracted from
:class:`OptimizedVideoProcessor` during the Phase-5 refactoring.

These are pure data classes and functions with no dependencies
on the main processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pupil_tracking.utils.types import assign_quality_grade


def annotate_quality(det: Dict[str, Any]) -> Dict[str, Any]:
    """Attach unified confidence + quality labels to flat detection dicts."""
    pupil_detected = bool(det.get("pupil_detected", False))
    limbus_detected = bool(det.get("limbus_detected", False))

    if pupil_detected and limbus_detected:
        overall_conf = float(
            (
                float(det.get("pupil_confidence", 0.0))
                + float(det.get("limbus_confidence", 0.0))
            )
            / 2.0
        )
    elif pupil_detected:
        overall_conf = float(det.get("pupil_confidence", 0.0))
    elif limbus_detected:
        overall_conf = float(det.get("limbus_confidence", 0.0))
    else:
        overall_conf = 0.0

    det["overall_confidence"] = overall_conf
    det["overall_quality"] = assign_quality_grade(overall_conf).value
    return det


@dataclass
class ManualCircularROI:
    """User-defined circular ROI stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height


@dataclass
class ManualRingAnnotation:
    """User-confirmed suction-ring circle stored in source-frame coordinates."""

    center_x: float
    center_y: float
    radius: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    dot_count: int = 12

    def matches_frame(self, frame: np.ndarray) -> bool:
        if self.frame_width is None or self.frame_height is None:
            return True
        h, w = frame.shape[:2]
        return w == self.frame_width and h == self.frame_height


class TrackingQuality:
    """Enum-like class for tracking quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    OK = "ok"
    POOR = "poor"
    LOST = "lost"


class FrameResult(dict):
    """Result from processing a single frame.

    Extends dict with convenient attribute access.
    Contains detection results, metadata, and quality metrics.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"FrameResult has no attribute '{key}'")

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val
