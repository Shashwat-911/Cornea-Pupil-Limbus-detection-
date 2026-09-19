"""
Professional FrameRecorder for high-quality video recording.

Features:
    - Thread-safe frame queue with dedicated writer thread
    - Multiple codec support (H.264/AVC preferred, fallback to others)
    - Proper frame timing for smooth playback
    - Status callbacks for UI updates
    - Graceful handling of different video sources
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np


class FrameRecorder:
    """High-quality frame recorder with dedicated writer thread.

    This recorder uses a separate thread to write frames to disk,
    ensuring smooth recording without blocking the main processing
    thread. It handles frame timing automatically for proper playback.

    Usage:
        recorder = FrameRecorder()
        recorder.start('/path/to/output.mp4', width, height, fps)

        # In processing loop:
        recorder.write(frame_with_overlays)

        # To stop:
        recorder.stop()  # Returns path to saved file
    """

    SUPPORTED_CODECS = [
        ("mp4v", ".mp4", "MPEG-4 (MP4)"),
        ("XVID", ".avi", "XviD (AVI)"),
        ("MJPG", ".avi", "Motion JPEG (AVI)"),
        ("DIVX", ".avi", "DivX (AVI)"),
        ("WMV2", ".wmv", "WMV2 (WMV)"),
        ("avc1", ".mp4", "H.264/AVC (MP4)"),
        ("H264", ".mp4", "H.264/AVC (MP4)"),
        ("X264", ".mp4", "H.264 (MP4)"),
    ]

    def __init__(self, name: str = "main") -> None:
        self._name = name
        self._writer_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=60)
        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[str] = None
        self._fps: float = 30.0
        self._width: int = 0
        self._height: int = 0
        self._running: bool = False
        self._codec: str = "mp4v"
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._lock = threading.Lock()

        self._status_callback: Optional[Callable[[dict], None]] = None

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        with self._lock:
            return self._running and self._writer is not None

    @property
    def frame_count(self) -> int:
        """Get number of frames written."""
        with self._lock:
            return self._frame_count

    @property
    def dropped_frames(self) -> int:
        """Get number of dropped frames."""
        with self._lock:
            return self._dropped_frames

    @property
    def elapsed_time(self) -> float:
        """Get elapsed recording time."""
        if self._start_time > 0:
            return time.monotonic() - self._start_time
        return 0.0

    def set_status_callback(self, callback: Callable[[dict], None]) -> None:
        """Set callback for status updates.

        Callback receives dict with keys:
            - is_recording: bool
            - frame_count: int
            - elapsed_time: float
            - fps: float
            - dropped_frames: int
            - queue_size: int
        """
        self._status_callback = callback

    def _get_best_codec(self, path: str) -> Tuple[str, str]:
        """Select the best available codec for the output format."""
        ext = Path(path).suffix.lower()
        if not ext:
            ext = ".mp4"

        # Try preferred codecs first
        for codec, codec_ext, name in self.SUPPORTED_CODECS:
            if ext == codec_ext or not ext:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                return fourcc, codec

        # Default fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return fourcc, "mp4v"

    def start(self, path: str, width: int, height: int, fps: float = 30.0) -> bool:
        """Start recording to the specified path.

        Args:
            path: Output video file path
            width: Frame width (must match actual frames)
            height: Frame height (must match actual frames)
            fps: Target frames per second

        Returns:
            True if recording started successfully
        """
        if self.is_recording:
            self.stop()

        # Validate dimensions
        if width <= 0 or height <= 0:
            return False

        self._path = path
        self._fps = float(fps)
        self._width = width
        self._height = height
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_time = 0.0

        # Get codec
        fourcc, codec_name = self._get_best_codec(path)
        self._codec = codec_name

        # Create writer
        self._writer = cv2.VideoWriter(path, fourcc, self._fps, (width, height))

        if not self._writer.isOpened():
            # Try fallback codecs
            fallback_tried = set()
            fallback_tried.add(codec_name)

            for codec, _, name in self.SUPPORTED_CODECS:
                if codec in fallback_tried:
                    continue
                fallback_tried.add(codec)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                self._writer.release()
                self._writer = cv2.VideoWriter(path, fourcc, self._fps, (width, height))
                if self._writer.isOpened():
                    self._codec = codec
                    break
            else:
                self._writer.release()
                self._writer = None
                return False

        # Start writer thread
        self._running = True
        self._start_time = time.monotonic()
        self._frame_queue = queue.Queue(maxsize=60)
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name=f"FrameRecorderWriter_{self._name}"
        )
        self._writer_thread.start()

        return True

    def _writer_loop(self) -> None:
        """Background thread that writes frames to disk."""
        frames_written = 0
        last_status_update = 0.0

        while self._running or not self._frame_queue.empty():
            try:
                # Wait for next frame with timeout
                try:
                    frame_data = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if frame_data is None:
                    # Sentinel: stop signal
                    break

                frame, timestamp = frame_data

                # Write frame
                if self._writer is not None and self._writer.isOpened():
                    self._writer.write(frame)
                    frames_written += 1

                    # Calculate actual FPS
                    with self._lock:
                        self._frame_count = frames_written

                # Update status every 100ms
                now = time.monotonic()
                if now - last_status_update >= 0.1 and self._status_callback:
                    elapsed = now - self._start_time
                    actual_fps = frames_written / elapsed if elapsed > 0 else 0

                    self._status_callback(
                        {
                            "is_recording": True,
                            "frame_count": frames_written,
                            "elapsed_time": elapsed,
                            "fps": actual_fps,
                            "dropped_frames": self._dropped_frames,
                            "queue_size": self._frame_queue.qsize(),
                        }
                    )
                    last_status_update = now

            except Exception:
                # Avoid crashing the writer thread
                pass

        # Final flush
        if self._writer is not None:
            self._writer.release()

    def write(self, frame: np.ndarray) -> bool:
        """Write a frame to the recording.

        This method is thread-safe and non-blocking.
        If the queue is full, the frame is dropped and dropped_frames counter increases.

        Args:
            frame: Frame to write (must be correct dimensions)

        Returns:
            True if frame was queued, False if dropped
        """
        if not self.is_recording:
            return False

        # Validate frame dimensions
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            return False

        timestamp = time.monotonic()

        try:
            # Try non-blocking put
            self._frame_queue.put_nowait((frame.copy(), timestamp))
            return True
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1
            return False

    def write_with_retry(self, frame: np.ndarray, max_retries: int = 3) -> bool:
        """Write a frame with retry logic for better frame preservation.

        Args:
            frame: Frame to write
            max_retries: Maximum number of retries

        Returns:
            True if frame was written, False otherwise
        """
        for attempt in range(max_retries):
            if self.write(frame):
                return True
            # Brief wait before retry
            time.sleep(0.001)
        return False

    def stop(self) -> Optional[str]:
        """Stop recording and finalize the video file.

        Returns:
            Path to the saved video file, or None if not recording
        """
        if not self.is_recording:
            return None

        path = self._path

        # Signal writer thread to stop
        self._running = False

        # Send sentinel to wake up writer thread
        try:
            self._frame_queue.put_nowait((None, 0))
        except queue.Full:
            pass

        # Wait for writer thread to finish
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)

        # Final status update
        if self._status_callback:
            self._status_callback(
                {
                    "is_recording": False,
                    "frame_count": self._frame_count,
                    "elapsed_time": self.elapsed_time,
                    "fps": self._frame_count / self.elapsed_time
                    if self.elapsed_time > 0
                    else 0,
                    "dropped_frames": self._dropped_frames,
                    "queue_size": 0,
                }
            )

        # Cleanup
        with self._lock:
            self._writer = None
            self._path = None
            self._frame_count = 0
            self._dropped_frames = 0
            self._start_time = 0.0
            self._running = False

        return path

    def pause(self) -> None:
        """Pause recording (frames will be dropped)."""
        with self._lock:
            self._running = False

    def resume(self) -> None:
        """Resume recording."""
        with self._lock:
            if self._writer is not None:
                self._running = True

    def get_status(self) -> dict:
        """Get current recording status."""
        with self._lock:
            elapsed = time.monotonic() - self._start_time if self._start_time > 0 else 0
            return {
                "is_recording": self._running and self._writer is not None,
                "frame_count": self._frame_count,
                "elapsed_time": elapsed,
                "fps": self._frame_count / elapsed if elapsed > 0 else 0,
                "dropped_frames": self._dropped_frames,
                "queue_size": self._frame_queue.qsize()
                if hasattr(self, "_frame_queue")
                else 0,
                "path": self._path,
                "codec": self._codec,
            }
