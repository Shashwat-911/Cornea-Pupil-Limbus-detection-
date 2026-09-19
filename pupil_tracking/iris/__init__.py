"""Iris feature detection (Phase I) with CNN + RL extensions (Phase V).

Phase I: classical iris-feature-detection baseline.
Phase V adds:
    - CNN segmentation (IrisSegmentationNet)
    - CNN encoding (IrisEncoderNet)
    - RL parameter tuning (IrisParamAgent)

The iris detector is **disabled by default** in the production pipeline and is
invoked explicitly (e.g. via :func:`detect_iris_features`).

Public API
----------
* ``IrisFeatureDetector`` / ``detect_iris_features`` -- top-level detection
* ``IrisConfig`` -- tunable parameters (including CNN/RL settings)
* ``IrisDetectionResult`` / ``IrisFeatureSet`` / ``IrisFeature`` / ``IrisROI``
  -- result contracts
* ``draw_iris_overlay`` -- optional debug visualisation
* ``IrisParamAgent`` -- RL parameter tuning agent
"""

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.detect import IrisFeatureDetector, detect_iris_features
from pupil_tracking.iris.masking import IrisMasking
from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.roi import IrisROIExtractor
from pupil_tracking.iris.types import (
    IrisDetectionResult,
    IrisFeature,
    IrisFeatureSet,
    IrisFeatureType,
    IrisROI,
    IrisStatus,
)
from pupil_tracking.iris.visualize import draw_iris_overlay

# Phase V: RL agent (lazy import to avoid torch dependency)
try:
    from pupil_tracking.iris.rl_agent import IrisParamAgent
except ImportError:
    IrisParamAgent = None

__all__ = [
    "IrisConfig",
    "IrisFeatureDetector",
    "detect_iris_features",
    "IrisMasking",
    "IrisNormalizer",
    "IrisROIExtractor",
    "IrisDetectionResult",
    "IrisFeature",
    "IrisFeatureSet",
    "IrisFeatureType",
    "IrisROI",
    "IrisStatus",
    "draw_iris_overlay",
    "IrisParamAgent",
]
