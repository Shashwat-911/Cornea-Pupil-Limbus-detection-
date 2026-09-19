"""
Multi-stream fusion engine for cyclotorsion estimation.

Combines results from all active detection streams into a single
fused estimate using adaptive weighting based on per-stream
confidence and inter-stream agreement.

Supports three fusion methods:
    - weighted_median: Robust to outliers, recommended default
    - bayesian: Full posterior estimation, best accuracy
    - robust_mean: Fast, good for real-time
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np

from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.types import (
    RegistrationQuality,
    RegistrationResult,
    StreamResult,
    assign_registration_grade,
)

logger = logging.getLogger(__name__)


class FusionEngine:
    """Fuse multiple stream results into a single torsion estimate.

    Parameters
    ----------
    method : str
        Fusion method: 'weighted_median', 'bayesian', 'robust_mean'.
    agreement_threshold_deg : float
        Maximum angular difference for two streams to agree.
    min_streams : int
        Minimum number of valid streams required.
    """

    # Prior weights for each stream based on expected reliability
    STREAM_PRIORS = {
        "phase_correlation": 0.85,
        "deep_matcher": 0.75,
        "ink_markers": 0.95,
        "limbal_vessels": 0.70,
        "custom_feature": 0.80,
    }

    def __init__(
        self,
        method: Optional[str] = None,
        agreement_threshold_deg: Optional[float] = None,
        min_streams: Optional[int] = None,
    ):
        cfg = get_config().registration
        self.method = method or cfg.fusion_method
        self.agreement_threshold_deg = (
            agreement_threshold_deg or cfg.fusion_agreement_threshold_deg
        )
        self.min_streams = min_streams or cfg.fusion_min_streams

    def fuse(
        self,
        stream_results: Dict[str, StreamResult],
    ) -> RegistrationResult:
        """Fuse stream results into a single registration result.

        Parameters
        ----------
        stream_results : dict
            Mapping of stream name → StreamResult.

        Returns
        -------
        RegistrationResult
        """
        # Normalize stream_results if passed as list or sequence
        if isinstance(stream_results, (list, tuple)):
            stream_results = {
                (sr.stream.value if hasattr(sr.stream, "value") else str(sr.stream)): sr
                for sr in stream_results
            }

        # Filter to valid results only
        valid_results = {
            k: v for k, v in stream_results.items() if v.valid
        }

        result = RegistrationResult(
            stream_results=stream_results,
            active_streams=len(valid_results),
        )

        if len(valid_results) < self.min_streams:
            # Not enough streams — check if we have at least one
            if len(valid_results) == 1:
                # Single stream: use it but with reduced confidence
                name, sr = next(iter(valid_results.items()))
                result.valid = True
                result.torsion_deg = sr.torsion_deg
                result.confidence = sr.confidence * 0.7  # Penalty for single stream
                result.quality = assign_registration_grade(result.confidence)
                result.agreeing_streams = 1
                result.torsion_std_deg = 0.0
                result.total_processing_time_ms = sum(
                    v.processing_time_ms for v in stream_results.values()
                )
                return result
            else:
                result.total_processing_time_ms = sum(
                    v.processing_time_ms for v in stream_results.values()
                )
                return result

        # Extract torsion values and weights
        torsion_values = []
        weights = []
        for name, sr in valid_results.items():
            torsion_values.append(sr.torsion_deg)
            prior = self.STREAM_PRIORS.get(name, 0.5)
            weights.append(sr.confidence * prior)

        torsion_values = np.array(torsion_values)
        weights = np.array(weights)

        # Normalise weights
        weight_sum = np.sum(weights)
        if weight_sum < 1e-10:
            result.total_processing_time_ms = sum(
                v.processing_time_ms for v in stream_results.values()
            )
            return result
        weights_norm = weights / weight_sum

        # Fuse based on method
        if self.method == "weighted_median":
            fused_torsion = self._weighted_median(torsion_values, weights_norm)
        elif self.method == "bayesian":
            fused_torsion = self._bayesian_fusion(torsion_values, weights)
        elif self.method == "robust_mean":
            fused_torsion = self._robust_mean(torsion_values, weights_norm)
        else:
            fused_torsion = self._weighted_median(torsion_values, weights_norm)

        # Compute agreement
        agreeing = np.sum(
            np.abs(torsion_values - fused_torsion) < self.agreement_threshold_deg
        )

        # Compute uncertainty
        if len(torsion_values) > 1:
            weighted_residuals = torsion_values - fused_torsion
            torsion_std = float(np.sqrt(
                np.sum(weights_norm * weighted_residuals**2)
            ))
        else:
            torsion_std = 0.0

        # Compute fused confidence
        agreement_ratio = agreeing / len(torsion_values)
        mean_confidence = float(np.mean([
            sr.confidence for sr in valid_results.values()
        ]))
        consistency_bonus = 1.0 + 0.2 * (agreement_ratio - 0.5)
        fused_confidence = float(np.clip(
            mean_confidence * consistency_bonus, 0.0, 1.0
        ))

        # Confidence interval (approximate 95%)
        ci_half = 1.96 * max(torsion_std, 0.05)

        # Populate result
        result.valid = True
        result.torsion_deg = float(fused_torsion)
        result.confidence = fused_confidence
        result.quality = assign_registration_grade(fused_confidence)
        result.agreeing_streams = int(agreeing)
        result.torsion_std_deg = torsion_std
        result.confidence_interval_deg = (
            float(fused_torsion - ci_half),
            float(fused_torsion + ci_half),
        )
        result.total_processing_time_ms = sum(
            v.processing_time_ms for v in stream_results.values()
        )

        return result

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted median — robust to outliers."""
        sorted_idx = np.argsort(values)
        sorted_values = values[sorted_idx]
        sorted_weights = weights[sorted_idx]

        cumsum = np.cumsum(sorted_weights)
        midpoint = 0.5 * cumsum[-1]

        # Find first index where cumulative weight exceeds midpoint
        idx = np.searchsorted(cumsum, midpoint)
        idx = min(idx, len(sorted_values) - 1)

        return float(sorted_values[idx])

    @staticmethod
    def _bayesian_fusion(values: np.ndarray, weights: np.ndarray) -> float:
        """Bayesian fusion treating each stream as a Gaussian observation.

        Assumes each stream's torsion estimate is normally distributed
        with variance proportional to 1/weight.
        """
        precisions = weights  # weight ∝ 1/variance ∝ precision
        precision_sum = np.sum(precisions)
        if precision_sum < 1e-10:
            return float(np.mean(values))
        return float(np.sum(values * precisions) / precision_sum)

    @staticmethod
    def _robust_mean(values: np.ndarray, weights: np.ndarray) -> float:
        """Iteratively re-weighted mean with outlier rejection."""
        estimate = np.sum(values * weights)

        for _ in range(3):
            residuals = np.abs(values - estimate)
            # Huber-like weight adjustment
            adjusted_weights = weights / (1.0 + residuals**2)
            adjusted_weights /= np.sum(adjusted_weights) + 1e-10
            estimate = np.sum(values * adjusted_weights)

        return float(estimate)
