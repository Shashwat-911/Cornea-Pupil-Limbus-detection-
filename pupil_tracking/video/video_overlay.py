# pupil_tracking/video/video_overlay.py
"""Overlay renderer for video frames extracted from
:class:`OptimizedVideoProcessor` during the Phase-5 refactoring.

Pure visualization utility with no dependencies on the main processor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


class OverlayRenderer:
    """Draws detection results onto video frames."""

    PUPIL_COLOR = (0, 255, 0)
    LIMBUS_COLOR = (255, 180, 0)
    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)

    @classmethod
    def draw(
        cls,
        frame: np.ndarray,
        det: Dict[str, Any],
        frame_idx: int = 0,
        fps: float = 0.0,
        out: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        vis = frame if out is None else out
        if out is not None:
            np.copyto(out, frame)

        if det.get("pupil_detected"):
            cx = int(det.get("pupil_x", 0))
            cy = int(det.get("pupil_y", 0))
            r = int(det.get("pupil_radius", det.get("pupil_r", 0)))
            cv2.circle(vis, (cx, cy), r, cls.PUPIL_COLOR, 2)
            cv2.circle(vis, (cx, cy), 3, cls.PUPIL_COLOR, -1)

            if "pupil_major" in det and "pupil_minor" in det:
                axes = (
                    int(det["pupil_major"] / 2),
                    int(det["pupil_minor"] / 2),
                )
                angle = det.get("pupil_angle", 0)
                cv2.ellipse(
                    vis, (cx, cy), axes, angle, 0, 360,
                    cls.PUPIL_COLOR, 2, cv2.LINE_AA,
                )

        if det.get("limbus_detected"):
            lx = int(det.get("limbus_x", 0))
            ly = int(det.get("limbus_y", 0))
            lr = int(det.get("limbus_radius", det.get("limbus_r", 0)))
            if "limbus_major" in det and "limbus_minor" in det:
                axes = (
                    int(det["limbus_major"] / 2),
                    int(det["limbus_minor"] / 2),
                )
                angle = det.get("limbus_angle", 0)
                cv2.ellipse(
                    vis, (lx, ly), axes, angle, 0, 360,
                    cls.LIMBUS_COLOR, 2, cv2.LINE_AA,
                )
            else:
                cv2.circle(vis, (lx, ly), lr, cls.LIMBUS_COLOR, 2)

        lines = [f"Frame: {frame_idx}"]
        if fps > 0:
            lines.append(f"FPS: {fps:.1f}")
        if det.get("pupil_detected"):
            conf = det.get("pupil_confidence", 0)
            pr = det.get("pupil_radius", det.get("pupil_r", 0))
            lines.append(
                f"Pupil: ({int(det.get('pupil_x', 0))}, "
                f"{int(det.get('pupil_y', 0))})  "
                f"r={int(pr)}  conf={conf:.2f}"
            )
        else:
            lines.append("Pupil: not detected")
        if det.get("limbus_detected"):
            lr = det.get("limbus_radius", det.get("limbus_r", 0))
            lines.append(f"Limbus: r={int(lr)}")

        quality = det.get("overall_quality", det.get("frame_quality", ""))
        if quality:
            lines.append(f"Quality: {quality}")

        latency = det.get("latency_ms", 0)
        if latency > 0:
            lines.append(f"Latency: {latency:.0f} ms")

        y0 = 25
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (8, y0 - th - 4), (14 + tw, y0 + 4), cls.BG_COLOR, -1)
            cv2.putText(
                vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, cls.TEXT_COLOR, 1, cv2.LINE_AA,
            )
            y0 += th + 12

        return vis
