"""
Image preprocessing modules for clinical eye images.

Provides:
    - ReflectionRemover      : Specular reflection detection and inpainting (A3)
    - SuctionRingMasker      : Red marker dot detection and masking (A5)
    - ImageNormalizer        : Illumination/contrast normalisation (A6)
    - RingAwarePreprocessor  : Adaptive preprocessing for docked/pre-docked images
    - AdaptiveContourFilter  : Ring-aware contour filtering
    - GrayscaleHandler       : Grayscale detection, enhancement, and model-input
                               preparation (NEW)
    - GrayscaleMode          : Operating mode enum (AUTO / FORCE / OFF) (NEW)
    - GrayscaleInfo          : Diagnostic metadata from grayscale processing (NEW)

All modules are designed to be used independently or chained
together in a preprocessing pipeline before ML inference.

Usage
-----
>>> from pupil_tracking.preprocessing import (
...     ReflectionRemover,
...     SuctionRingMasker,
...     ImageNormalizer,
...     RingAwarePreprocessor,
...     GrayscaleHandler,
...     GrayscaleMode,
... )
>>> remover = ReflectionRemover()
>>> masker = SuctionRingMasker()
>>> normalizer = ImageNormalizer()
>>>
>>> # Full preprocessing pipeline
>>> image, refl_mask = remover.remove(image)
>>> image, ring_mask = masker.remove(image)
>>> image = normalizer.normalize(image)
>>>
>>> # Ring-aware preprocessing (auto-adapts to docked vs pre-docked)
>>> from pupil_tracking.core.deterministic_ring_detector import RingDetector
>>> ring_detector = RingDetector()
>>> ring_result = ring_detector.detect(image)
>>> preprocessor = RingAwarePreprocessor()
>>> prep_result = preprocessor.preprocess(image, ring_result)
>>>
>>> # Grayscale handling (auto-detect or force)
>>> handler = GrayscaleHandler()
>>> model_input, info = handler.to_model_input(image, force_grayscale=False)
>>> print(f"Was grayscale: {info.was_grayscale}, enhanced: {info.conversion_applied}")
"""

from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
from pupil_tracking.preprocessing.suction_ring_masker import (
    SuctionRingMasker,
    SuctionRingResult,
)
from pupil_tracking.preprocessing.normalizer import ImageNormalizer
from pupil_tracking.preprocessing.ring_aware import (
    RingAwarePreprocessor,
    PreprocessingResult,
    AdaptiveContourFilter,
)
from pupil_tracking.preprocessing.grayscale_handler import (
    GrayscaleHandler,
    GrayscaleMode,
    GrayscaleInfo,
)
from pupil_tracking.preprocessing.red_light_filter import (
    RedLightFilter,
    AdaptiveRedLightFilter,
)

__all__ = [
    # Existing modules
    "ReflectionRemover",
    "SuctionRingMasker",
    "SuctionRingResult",
    "ImageNormalizer",
    # Ring-aware modules
    "RingAwarePreprocessor",
    "PreprocessingResult",
    "AdaptiveContourFilter",
    # Grayscale modules (NEW)
    "GrayscaleHandler",
    "GrayscaleMode",
    "GrayscaleInfo",
    # Red light filter (NEW)
    "RedLightFilter",
    "AdaptiveRedLightFilter",
]
