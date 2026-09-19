"""Unit test for concurrent dual-stream video recording (clean raw + annotated overlay)."""

import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

from pupil_tracking.interface.frame_recorder import FrameRecorder


def test_dual_stream_concurrent_recording(tmp_path: Path):
    raw_path = str(tmp_path / "patient_001_raw.mp4")
    overlay_path = str(tmp_path / "patient_001_overlay.mp4")

    raw_recorder = FrameRecorder(name="raw")
    overlay_recorder = FrameRecorder(name="overlay")

    width, height, fps = 320, 240, 30.0
    num_frames = 15

    # Start both recorders simultaneously
    assert raw_recorder.start(raw_path, width, height, fps) is True
    assert overlay_recorder.start(overlay_path, width, height, fps) is True

    assert raw_recorder.is_recording is True
    assert overlay_recorder.is_recording is True

    # Simulate video feed
    for i in range(num_frames):
        # 1. Clean RAW frame (plain background with simple circle)
        raw_frame = np.full((height, width, 3), 100, dtype=np.uint8)
        cv2.circle(raw_frame, (160, 120), 40, (30, 30, 30), -1)

        # 2. Overlay frame (RAW frame + yellow pupil ellipse + green limbus + UI text)
        overlay_frame = raw_frame.copy()
        cv2.ellipse(overlay_frame, (160, 120), (40, 40), 0, 0, 360, (0, 255, 255), 2)
        cv2.putText(overlay_frame, f"REC FRAME {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        raw_recorder.write(raw_frame)
        overlay_recorder.write(overlay_frame)
        time.sleep(0.01)

    time.sleep(0.2)

    # Stop both recorders
    saved_raw = raw_recorder.stop()
    saved_overlay = overlay_recorder.stop()

    assert saved_raw == raw_path
    assert saved_overlay == overlay_path
    assert Path(raw_path).exists()
    assert Path(overlay_path).exists()
    assert Path(raw_path).stat().st_size > 0
    assert Path(overlay_path).stat().st_size > 0

    # Verify video integrity using cv2.VideoCapture
    cap_raw = cv2.VideoCapture(raw_path)
    cap_overlay = cv2.VideoCapture(overlay_path)

    assert cap_raw.isOpened()
    assert cap_overlay.isOpened()

    ret_r, frame_r = cap_raw.read()
    ret_o, frame_o = cap_overlay.read()

    assert ret_r is True
    assert ret_o is True
    assert frame_r.shape == (height, width, 3)
    assert frame_o.shape == (height, width, 3)

    # Verify that raw frame does NOT contain the overlay annotations (e.g. green text at (10, 30))
    # In overlay_frame, pixel (30, 10) is green text (0, 255, 0)
    # In raw_frame, pixel (30, 10) is background (100, 100, 100)
    assert np.all(frame_r[30, 10] != (0, 255, 0))

    cap_raw.release()
    cap_overlay.release()
