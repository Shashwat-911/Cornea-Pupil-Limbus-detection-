"""Replaceable angular estimator contract; no model framework dependency.

Implement this callable with a trained CPU model to reuse preprocessing,
geometry, masks, session identity checks, and result serialization. Instances
belong to one session and must not share mutable inference caches.
"""
from __future__ import annotations

import abc
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np

from .sitting import AngularMatch, masked_angular_match


class AngularMatcher(Protocol):
    def __call__(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        """Return CCW image rotation in degrees; masks are nonzero for valid pixels.

        Inputs are grayscale polar strips [radius, angle]. Do not mutate inputs.
        Scores describe match quality, not calibrated clinical probabilities.
        Return valid=False for out-of-distribution or ambiguous inputs.
        """
        ...


class BaseAngularMatcher(abc.ABC):
    """Abstract base class for stateful or configurable cyclotorsion matchers."""

    def __init__(self, name: str = "BaseMatcher") -> None:
        self.name = name
        self.total_calls = 0
        self.accepted_calls = 0
        self.total_time_ms = 0.0

    @abc.abstractmethod
    def _match(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        """Subclasses implement specific estimation logic."""
        raise NotImplementedError

    def __call__(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        t0 = time.perf_counter()
        self.total_calls += 1

        # Common input sanity checks
        if not isinstance(ref, np.ndarray) or not isinstance(curr, np.ndarray):
            return AngularMatch(reason="invalid_input_type")
        if ref.ndim != 2 or curr.ndim != 2 or ref.shape != curr.shape:
            return AngularMatch(reason="invalid_shape")

        result = self._match(
            ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
        )
        dt = (time.perf_counter() - t0) * 1000.0
        self.total_time_ms += dt
        if result.valid:
            self.accepted_calls += 1
        return result

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_calls / max(1, self.total_calls)

    @property
    def average_latency_ms(self) -> float:
        return self.total_time_ms / max(1, self.total_calls)


class ClassicalFFTMatcher(BaseAngularMatcher):
    """Classical shift-dependent masked ZNCC evaluated via circular FFTs."""

    def __init__(
        self,
        min_score: float = 0.35,
        min_overlap: float = 0.30,
        min_peak_margin: float = 0.035,
        max_band_spread_deg: float = 2.0,
    ) -> None:
        super().__init__(name="ClassicalFFTMatcher")
        self.min_score = min_score
        self.min_overlap = min_overlap
        self.min_peak_margin = min_peak_margin
        self.max_band_spread_deg = max_band_spread_deg

    def _match(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        return masked_angular_match(
            ref,
            curr,
            mask_ref=mask_ref,
            mask_curr=mask_curr,
            max_degrees=max_degrees,
            reference_cache=reference_cache,
        )


class LearnedAngularMatcher(BaseAngularMatcher):
    """Adapter for trained deep learning models or ONNX inference sessions.

    Accepts any callable or model object providing an `estimate_rotation` or
    `forward` method. Automatically falls back to classical matching if the model
    fails or produces an out-of-bounds result.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        *,
        fallback_matcher: Optional[AngularMatcher] = None,
        confidence_threshold: float = 0.40,
        enable_fallback: bool = True,
        name: str = "LearnedAngularMatcher",
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self.fallback = fallback_matcher or ClassicalFFTMatcher()
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback

    def _match(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        if self.model is None:
            if self.enable_fallback and self.fallback is not None:
                return self.fallback(
                    ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
                )
            return AngularMatch(reason="no_trained_model_provided")

        try:
            # Check if model has a dedicated estimate_rotation method
            if hasattr(self.model, "estimate_rotation"):
                res = self.model.estimate_rotation(
                    ref, curr, mask_ref, mask_curr, max_degrees=max_degrees
                )
                if isinstance(res, AngularMatch):
                    if res.valid:
                        return res
                    elif self.enable_fallback and self.fallback is not None:
                        return self.fallback(
                            ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
                        )
                    return res
                if isinstance(res, tuple) and len(res) >= 2:
                    angle_deg, score = float(res[0]), float(res[1])
                    valid = abs(angle_deg) < max_degrees and score >= self.confidence_threshold
                    if valid:
                        return AngularMatch(
                            angle_deg=angle_deg,
                            valid=True,
                            score=score,
                            overlap=1.0,
                            peak_margin=score,
                            band_spread_deg=0.0,
                            reason="",
                        )
                    elif self.enable_fallback and self.fallback is not None:
                        return self.fallback(
                            ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
                        )
                    return AngularMatch(
                        angle_deg=angle_deg,
                        valid=False,
                        score=score,
                        overlap=1.0,
                        peak_margin=score,
                        band_spread_deg=0.0,
                        reason="learned_model_low_confidence",
                    )

            # Fallback to classical matcher if model did not return a valid result
            if self.enable_fallback and self.fallback is not None:
                return self.fallback(
                    ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
                )
            return AngularMatch(reason="unsupported_model_interface")
        except Exception as exc:
            if self.enable_fallback and self.fallback is not None:
                return self.fallback(
                    ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
                )
            return AngularMatch(reason=f"model_inference_error: {exc}")


class EnsembleMatcher(BaseAngularMatcher):
    """Ensemble combining classical FFT with a learned model for corroborative consensus."""

    def __init__(
        self,
        matchers: Optional[List[AngularMatcher]] = None,
        max_discrepancy_deg: float = 1.0,
        consensus_threshold: float = 0.5,
    ) -> None:
        super().__init__(name="EnsembleMatcher")
        self.matchers = matchers or [ClassicalFFTMatcher()]
        self.max_discrepancy_deg = max_discrepancy_deg
        self.consensus_threshold = consensus_threshold

    def _match(
        self,
        ref: np.ndarray,
        curr: np.ndarray,
        mask_ref: np.ndarray,
        mask_curr: np.ndarray,
        max_degrees: float,
        *,
        reference_cache: dict,
    ) -> AngularMatch:
        results = [
            m(ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache)
            for m in self.matchers
        ]
        valid_results = [r for r in results if r.valid]

        if not valid_results:
            first_fail = results[0] if results else AngularMatch(reason="empty_ensemble")
            return first_fail

        if len(valid_results) == 1:
            # Single surviving matcher
            return valid_results[0]

        # Check angular agreement between valid estimators
        angles = [r.angle_deg for r in valid_results]
        spread = float(np.ptp(angles))
        if spread > self.max_discrepancy_deg:
            return AngularMatch(
                valid=False,
                reason=f"ensemble_discrepancy_{spread:.2f}deg_exceeds_{self.max_discrepancy_deg}deg",
                band_spread_deg=spread,
            )

        # Variance/score-weighted fusion
        scores = [max(r.score, 0.01) for r in valid_results]
        total_score = sum(scores)
        fused_angle = sum(a * s for a, s in zip(angles, scores)) / total_score
        fused_score = max(r.score for r in valid_results)
        fused_overlap = min(r.overlap for r in valid_results)

        return AngularMatch(
            angle_deg=float(fused_angle),
            valid=True,
            score=float(fused_score),
            overlap=float(fused_overlap),
            peak_margin=min(r.peak_margin for r in valid_results),
            band_spread_deg=spread,
            reason="",
        )


def checked_match(
    matcher: AngularMatcher,
    ref: np.ndarray,
    curr: np.ndarray,
    mask_ref: Optional[np.ndarray],
    mask_curr: Optional[np.ndarray],
    max_degrees: float,
    *,
    reference_cache: dict,
) -> AngularMatch:
    """Reject malformed backend measurements before they enter temporal state."""
    result = matcher(
        ref, curr, mask_ref, mask_curr, max_degrees, reference_cache=reference_cache
    )
    if not isinstance(result, AngularMatch):
        raise TypeError("Angular matcher must return AngularMatch")
    values = (
        result.angle_deg,
        result.score,
        result.overlap,
        result.peak_margin,
        result.band_spread_deg,
    )
    if (
        not np.isfinite(values).all()
        or abs(result.angle_deg) >= max_degrees
        or not -1 <= result.score <= 1
        or not 0 <= result.overlap <= 1
        or (result.valid and (result.score <= 0 or result.overlap < 0.30))
    ):
        return AngularMatch(reason="invalid_backend_measurement")
    return result


def create_matcher(
    kind: str = "classical",
    *,
    model: Optional[Any] = None,
    **kwargs: Any,
) -> AngularMatcher:
    """Factory helper to instantiate cyclotorsion matchers."""
    k = kind.lower().strip()
    if k in ("classical", "fft", "zncc"):
        return ClassicalFFTMatcher(**kwargs)
    elif k in ("learned", "nn", "deep"):
        return LearnedAngularMatcher(model=model, **kwargs)
    elif k in ("ensemble", "hybrid"):
        return EnsembleMatcher(**kwargs)
    else:
        raise ValueError(f"Unknown matcher kind: '{kind}'. Choose from: classical, learned, ensemble.")


DEFAULT_MATCHER: AngularMatcher = masked_angular_match
