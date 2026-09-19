"""Shared constants and optional fast-pipeline imports for the GUI mixins."""
from __future__ import annotations

try:
    from pupil_tracking.ml.fast_inference import FastInference
    from pupil_tracking.video.optimized_processor import (
        OptimizedVideoProcessor,
        AsyncCapture,
        FrameResult,
        TrackingQuality,
    )

    _FAST_PIPELINE_AVAILABLE = True
except ImportError:
    _FAST_PIPELINE_AVAILABLE = False
    FastInference = None
    OptimizedVideoProcessor = None
    AsyncCapture = None
    FrameResult = None
    TrackingQuality = None


_CORNEAL_DIAMETER_MM = 12.0
_CIRCLE_DRAW_THRESHOLD = 0.95

_QUALITY_COLORS = {
    "SURGICAL": "#00e676",
    "CLINICAL": "#29b6f6",
    "RESEARCH": "#ffa726",
    "INSUFFICIENT": "#ef5350",
    "NO_DETECTION": "#616161",
}

# ══════════════════════════════════════════════════════════════════
# GRAYSCALE GUI 2 of 12 — Grayscale mode display labels & colours
# ══════════════════════════════════════════════════════════════════
_GRAYSCALE_LABELS = {
    "off": "RGB",
    "auto": "AUTO",
    "force": "GRAY",
}
_GRAYSCALE_COLORS = {
    "off": "#aaaaaa",
    "auto": "#00bcd4",
    "force": "#ffeb3b",
}
_GRAYSCALE_CYCLE = ["off", "auto", "force"]
# ══════════════════════════════════════════════════════════════════

_WINDOW_TITLE = "Medevplus IXcentai — Surgical Grade"
_MIN_WIDTH = 1280
_MIN_HEIGHT = 800
_DISPLAY_FPS_CAP = 30.0
