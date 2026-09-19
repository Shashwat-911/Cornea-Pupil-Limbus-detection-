"""
Abstract base class for cyclotorsion detection streams.

Every stream must:
    1. Accept two images + two detection results
    2. Return a StreamResult with torsion angle + confidence
    3. Be independently testable
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from pupil_tracking.utils.types import (
    EyeDetectionResult,
    StreamName,
    StreamResult,
)

logger = logging.getLogger(__name__)


class BaseStream(ABC):
    """Abstract base class for all cyclotorsion detection streams.

    Subclasses implement ``compute()`` which receives reference and
    current images with their detections, and returns a ``StreamResult``.

    The base class provides:
        - Timing wrapper
        - Error handling
        - Result construction helpers
    """

    def __init__(self, name: StreamName):
        self.name = name
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def run(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Execute the stream with timing and error handling.

        Parameters
        ----------
        img_ref : np.ndarray
            Reference (pre-operative) image.
        img_curr : np.ndarray
            Current (intra-operative) image.
        detection_ref : EyeDetectionResult
            Detection result for the reference image.
        detection_curr : EyeDetectionResult
            Detection result for the current image.

        Returns
        -------
        StreamResult
        """
        if not self._enabled:
            return StreamResult(
                stream=self.name,
                metadata={"disabled": True},
            )

        start = time.perf_counter()
        try:
            result = self.compute(
                img_ref, img_curr, detection_ref, detection_curr
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            result.processing_time_ms = elapsed_ms
            result.stream = self.name
            return result

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Stream %s failed: %s", self.name.value, str(e),
                exc_info=True,
            )
            return StreamResult(
                stream=self.name,
                processing_time_ms=elapsed_ms,
                metadata={"error": str(e)},
            )

    @abstractmethod
    def compute(
        self,
        img_ref: np.ndarray,
        img_curr: np.ndarray,
        detection_ref: EyeDetectionResult,
        detection_curr: EyeDetectionResult,
    ) -> StreamResult:
        """Compute cyclotorsion from paired images.

        Must be implemented by each stream.

        Parameters
        ----------
        img_ref, img_curr : np.ndarray
            Reference and current eye images.
        detection_ref, detection_curr : EyeDetectionResult
            Detection results for each image.

        Returns
        -------
        StreamResult
        """
        ...

    def _make_result(
        self,
        torsion_deg: Optional[float] = None,
        confidence: float = 0.0,
        inlier_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StreamResult:
        """Helper to construct a StreamResult."""
        return StreamResult(
            stream=self.name,
            torsion_deg=torsion_deg,
            confidence=confidence,
            inlier_count=inlier_count,
            metadata=metadata or {},
        )
