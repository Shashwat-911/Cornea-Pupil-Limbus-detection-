"""Tests for JSON annotation schema and serialisation."""

import json
from pathlib import Path
import pytest
from annotation_tool.io.annotation_format import AnnotationFile, Keypoint, Segment, InkMark


def test_annotation_file_creation(tmp_path: Path):
    ann = AnnotationFile(image_width=640, image_height=480, annotator="test_user")
    ann.add_keypoint("iris_crypt", 150.0, 200.0, "certain")
    ann.add_segment("radial_furrow", [[100.0, 100.0], [110.0, 120.0]])
    ann.add_ink_mark(300.0, 320.0, radius=5.0)

    assert len(ann.keypoints) == 1
    assert len(ann.segments) == 1
    assert len(ann.ink_marks) == 1
    assert ann.modified is True

    # Save to disk
    out_file = tmp_path / "test_annotation.json"
    ann.save(out_file)
    assert out_file.exists()

    # Load back
    loaded = AnnotationFile.load(out_file)
    assert loaded.image_width == 640
    assert loaded.image_height == 480
    assert len(loaded.keypoints) == 1
    assert loaded.keypoints[0]["type"] == "iris_crypt"
    assert loaded.keypoints[0]["x"] == 150.0
    assert len(loaded.segments) == 1
    assert len(loaded.ink_marks) == 1


def test_annotation_file_undo():
    ann = AnnotationFile()
    ann.add_keypoint("iris_crypt", 10.0, 20.0)
    assert len(ann.keypoints) == 1
    ann.undo()
    assert len(ann.keypoints) == 0
