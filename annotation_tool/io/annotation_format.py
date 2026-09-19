"""
JSON annotation format for iris feature labeling.

Defines the AnnotationFile data model, serialisation, and validation
for a single annotated iris image.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Keypoint:
    """A single annotated keypoint."""
    id: str = ""
    type: str = "iris_crypt"
    x: float = 0.0
    y: float = 0.0
    confidence: str = "certain"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Keypoint":
        return cls(
            id=d.get("id", ""),
            type=d.get("type", "iris_crypt"),
            x=float(d.get("x", 0)),
            y=float(d.get("y", 0)),
            confidence=d.get("confidence", "certain"),
        )


@dataclass
class Segment:
    """A polyline segment (furrow, vessel)."""
    id: str = ""
    type: str = "radial_furrow"
    polyline: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "polyline": self.polyline,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(
            id=d.get("id", ""),
            type=d.get("type", "radial_furrow"),
            polyline=d.get("polyline", []),
        )


@dataclass
class InkMark:
    """An annotated ink mark."""
    id: str = ""
    center_x: float = 0.0
    center_y: float = 0.0
    radius: float = 8.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "InkMark":
        return cls(
            id=d.get("id", ""),
            center_x=float(d.get("center_x", 0)),
            center_y=float(d.get("center_y", 0)),
            radius=float(d.get("radius", 8.0)),
        )


@dataclass
class AnnotationFile:
    """Complete annotation for a single image.

    Manages keypoints, segments, and ink marks with undo support.
    """
    image_path: Optional[Path] = None
    image_width: int = 0
    image_height: int = 0
    annotator: str = ""
    annotation_date: str = ""

    iris_bounds: Dict[str, Any] = field(default_factory=dict)
    keypoints: List[dict] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)
    ink_marks: List[dict] = field(default_factory=list)

    modified: bool = False

    def add_keypoint(self, type: str, x: float, y: float,
                     confidence: str = "certain"):
        """Add a keypoint annotation."""
        kp = {
            "id": f"kp_{uuid.uuid4().hex[:8]}",
            "type": type,
            "x": x,
            "y": y,
            "confidence": confidence,
        }
        self.keypoints.append(kp)
        self.modified = True

    def add_segment(self, type: str, polyline: List[List[float]]):
        """Add a polyline segment annotation."""
        seg = {
            "id": f"seg_{uuid.uuid4().hex[:8]}",
            "type": type,
            "polyline": polyline,
        }
        self.segments.append(seg)
        self.modified = True

    def add_ink_mark(self, center_x: float, center_y: float,
                     radius: float = 8.0):
        """Add an ink mark annotation."""
        mark = {
            "id": f"ink_{uuid.uuid4().hex[:8]}",
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
        }
        self.ink_marks.append(mark)
        self.modified = True

    def undo(self) -> bool:
        """Undo the last added annotation (keypoint, segment, or ink mark)."""
        if self.keypoints:
            self.keypoints.pop()
            self.modified = True
            return True
        if self.ink_marks:
            self.ink_marks.pop()
            self.modified = True
            return True
        if self.segments:
            self.segments.pop()
            self.modified = True
            return True
        return False

    def delete_nearest(self, x: float, y: float,
                       threshold: float = 10.0) -> bool:
        """Delete the nearest annotation within threshold distance.

        Returns True if something was deleted.
        """
        # Check keypoints
        best_dist = threshold
        best_idx = -1
        best_type = None

        for i, kp in enumerate(self.keypoints):
            dist = ((kp["x"] - x)**2 + (kp["y"] - y)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_type = "keypoint"

        for i, mark in enumerate(self.ink_marks):
            dist = ((mark["center_x"] - x)**2 + (mark["center_y"] - y)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_type = "ink"

        if best_type == "keypoint":
            self.keypoints.pop(best_idx)
            self.modified = True
            return True
        elif best_type == "ink":
            self.ink_marks.pop(best_idx)
            self.modified = True
            return True

        return False

    def copy(self) -> "AnnotationFile":
        """Create a deep copy for undo support."""
        return deepcopy(self)

    def to_dict(self) -> dict:
        """Serialise to dictionary."""
        return {
            "image_path": str(self.image_path) if self.image_path else "",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "annotator": self.annotator,
            "annotation_date": self.annotation_date or datetime.now().isoformat(),
            "iris_bounds": self.iris_bounds,
            "keypoints": list(self.keypoints),
            "segments": list(self.segments),
            "ink_marks": list(self.ink_marks),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnnotationFile":
        """Deserialise from dictionary."""
        af = cls(
            image_path=Path(d["image_path"]) if d.get("image_path") else None,
            image_width=d.get("image_width", 0),
            image_height=d.get("image_height", 0),
            annotator=d.get("annotator", ""),
            annotation_date=d.get("annotation_date", ""),
            iris_bounds=d.get("iris_bounds", {}),
            keypoints=d.get("keypoints", []),
            segments=d.get("segments", []),
            ink_marks=d.get("ink_marks", []),
        )
        return af

    def save(self, path: Path):
        """Save to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        self.modified = False

    @classmethod
    def load(cls, path: Path) -> "AnnotationFile":
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        af = cls.from_dict(data)
        af.modified = False
        return af

    def validate(self) -> List[str]:
        """Validate annotation integrity. Returns list of error messages."""
        errors = []

        if not self.image_path:
            errors.append("Missing image_path")

        for kp in self.keypoints:
            if "x" not in kp or "y" not in kp:
                errors.append(f"Keypoint {kp.get('id', '?')} missing coordinates")
            if kp.get("type") not in [
                "iris_crypt", "collarette_junction", "vessel_bifurcation",
                "pigment_spot", "crypt", "vessel",
            ]:
                errors.append(f"Keypoint {kp.get('id', '?')} has unknown type: {kp.get('type')}")

        for seg in self.segments:
            if not seg.get("polyline") or len(seg["polyline"]) < 2:
                errors.append(f"Segment {seg.get('id', '?')} needs ≥2 points")

        for mark in self.ink_marks:
            if mark.get("radius", 0) <= 0:
                errors.append(f"Ink mark {mark.get('id', '?')} has invalid radius")

        return errors
