"""Top-level iris-feature detection orchestration (Phase I + Phase V).

This is the public entry point that wires together:

    ROI construction -> masking -> normalization -> extraction -> result

Phase V adds:
    CNN segmentation (replaces classical masking when available)
    CNN encoding (replaces 16-bin histogram descriptors)
    RL parameter tuning (adaptive thresholds per frame)

It consumes the existing pupil/limbus geometry (``EllipseParams``) and does
**not** re-run pupil/limbus detection. It is safe to call with missing or
invalid geometry (returns a non-crashing ``IrisDetectionResult`` with
``status=NO_ROI``).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from pupil_tracking.iris.config import IrisConfig
from pupil_tracking.iris.extraction import IrisFeatureExtractor
from pupil_tracking.iris.masking import IrisMasking, mask_stats, roi_iris_stats
from pupil_tracking.iris.roi import IrisROIExtractor
from pupil_tracking.iris.types import IrisDetectionResult, IrisFeatureSet, IrisStatus
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
from pupil_tracking.utils.types import EllipseParams

logger = logging.getLogger(__name__)


class IrisFeatureDetector:
    """Stateful detector wrapping ROI, masking and extraction.

    Instantiate once and call :meth:`detect` per image. All parameters are read
    from :class:`IrisConfig`.

    Phase V additions:
        - CNN segmentation (optional, replaces classical masking)
        - CNN encoding (optional, replaces 16-bin histograms)
        - RL parameter tuning (optional, adjusts thresholds per frame)
    """

    def __init__(
        self,
        config: Optional[IrisConfig] = None,
        reflection_remover: Optional[ReflectionRemover] = None,
    ) -> None:
        self.config = config or IrisConfig()
        self.extractor = IrisFeatureExtractor(
            num_angles=self.config.num_angles,
            num_radii=self.config.num_radii,
            radius_px=self.config.radius_px,
            min_contrast=self.config.min_contrast,
            texture_floor=self.config.texture_floor,
            texture_rel_frac=self.config.texture_rel_frac,
            max_features=self.config.max_features,
            min_angular_sep_deg=self.config.min_angular_sep_deg,
            min_patch_valid_fraction=self.config.min_patch_valid_fraction,
            use_roi_percentiles=self.config.use_roi_percentiles,
            roi_p05=self.config.roi_p05,
            roi_p95=self.config.roi_p95,
            intensity_low_frac=self.config.intensity_low_frac,
            intensity_high_frac=self.config.intensity_high_frac,
            line_suppression_threshold=self.config.line_suppression_threshold,
            line_suppression_window_r=self.config.line_suppression_window_r,
        )
        self.roi_extractor = IrisROIExtractor(
            inner_inset_frac=self.config.inner_inset_frac,
            outer_inset_frac=self.config.outer_inset_frac,
        )
        self.masking = IrisMasking(
            reflection_remover=reflection_remover,
            only_within_roi=self.config.only_within_roi,
            eyelid_method=self.config.eyelid_method,
            eyelid_edge_threshold=self.config.eyelid_edge_threshold,
            eyelid_dilate_px=self.config.eyelid_dilate_px,
            saturation_threshold=self.config.saturation_threshold,
            suppress_blue_ring_lines=self.config.suppress_blue_ring_lines,
            blue_line_blueness_threshold=self.config.blue_line_blueness_threshold,
            blue_line_min_length_px=self.config.blue_line_min_length_px,
            blue_line_max_local_density=self.config.blue_line_max_local_density,
            blue_line_dilate_px=self.config.blue_line_dilate_px,
        )

        # Phase V: CNN components (lazy-loaded)
        self._cnn_segmentor = None
        self._cnn_encoder = None
        self._rl_agent = None
        self._prev_result: Optional[IrisDetectionResult] = None

        self._init_cnn_components()
        self._init_rl_agent()

    def _init_cnn_components(self) -> None:
        """Lazy-load CNN segmentation and encoding models."""
        if self.config.use_cnn_segmentation:
            try:
                from pupil_tracking.ml.iris_segmentation import IrisSegmentationNet

                self._cnn_segmentor = IrisSegmentationNet(
                    model_path=self.config.cnn_segmentation_model_path or None,
                    threshold=self.config.cnn_segmentation_threshold,
                )
                if self._cnn_segmentor.available:
                    logger.info("CNN iris segmentation enabled")
                else:
                    self._cnn_segmentor = None
                    logger.warning("CNN iris segmentation model not found; using classical")
            except ImportError:
                logger.warning("IrisSegmentationNet not available")

        if self.config.use_cnn_encoding:
            try:
                from pupil_tracking.ml.iris_encoder import IrisEncoderNet

                self._cnn_encoder = IrisEncoderNet(
                    model_path=self.config.cnn_encoder_model_path or None,
                )
                if self._cnn_encoder.available:
                    logger.info("CNN iris encoding enabled (dim=%d)", self._cnn_encoder.embedding_dim)
                else:
                    self._cnn_encoder = None
                    logger.warning("CNN iris encoder model not found; using classical")
            except ImportError:
                logger.warning("IrisEncoderNet not available")

    def _init_rl_agent(self) -> None:
        """Lazy-load RL parameter tuning agent."""
        if self.config.rl_enabled:
            try:
                from pupil_tracking.iris.rl_agent import IrisParamAgent

                self._rl_agent = IrisParamAgent(
                    model_path=self.config.rl_agent_model_path or None,
                )
                if self._rl_agent.available:
                    logger.info("RL parameter tuning enabled")
                else:
                    self._rl_agent = None
                    logger.warning("RL agent model not found; using static config")
            except ImportError:
                logger.warning("IrisParamAgent not available")

    def detect(
        self,
        image: np.ndarray,
        pupil: Optional[EllipseParams],
        limbus: Optional[EllipseParams],
        *,
        external_occlusion: Optional[np.ndarray] = None,
    ) -> IrisDetectionResult:
        """Run iris-feature detection.

        Parameters
        ----------
        image : np.ndarray  BGR (H, W, 3) or grayscale (H, W)
        pupil : EllipseParams | None
        limbus : EllipseParams | None
        external_occlusion : np.ndarray | None
            Optional boolean (H, W) mask of occluded pixels (True = occluded).

        Returns
        -------
        IrisDetectionResult
        """
        start = time.perf_counter()

        roi = self.roi_extractor.build(pupil, limbus)
        if not roi.valid:
            return self._finish(
                IrisDetectionResult(
                    valid=False,
                    status=IrisStatus.NO_ROI,
                    feature_set=IrisFeatureSet(roi=roi),
                ),
                start,
            )

        # Phase V: RL parameter adjustment (before detection)
        if self._rl_agent is not None and self._prev_result is not None:
            self._apply_rl_tuning()

        # Phase V: CNN segmentation or classical masking
        if self._cnn_segmentor is not None and self._cnn_segmentor.available:
            usable = self._cnn_segmentor.predict(image, roi)
        else:
            usable = self.masking.build(image, roi, external_occlusion=external_occlusion)

        iris_stats = roi_iris_stats(image, usable, roi)

        feature_set = self.extractor.extract(
            image,
            roi,
            usable,
            pupil=pupil,
            limbus=limbus,
            roi_stats=iris_stats,
        )
        feature_set.usable_fraction = mask_stats(usable, roi).get(
            "usable_fraction", 0.0
        )
        feature_set.region_coverage = self._coverage(feature_set, roi)

        # Phase V: CNN encoding (replace 16-bin histograms)
        if self._cnn_encoder is not None and self._cnn_encoder.available:
            self._apply_cnn_encoding(image, feature_set)

        n_features = len(feature_set.features)
        status = (
            IrisStatus.OK
            if n_features > 0
            else IrisStatus.NO_FEATURES
        )

        result = IrisDetectionResult(
            valid=n_features > 0,
            status=status,
            feature_set=feature_set,
            mask_stats=mask_stats(usable, roi),
        )
        result.mask_stats.update(iris_stats)

        # Store for RL agent
        self._prev_result = result

        return self._finish(result, start)

    def _apply_rl_tuning(self) -> None:
        """Apply RL agent's parameter adjustments to the config."""
        if self._rl_agent is None or self._prev_result is None:
            return

        try:
            from pupil_tracking.iris.rl_agent import IrisParamAgent

            state = IrisParamAgent.state_from_result(
                self._prev_result, self.config, self._prev_result
            )
            action = self._rl_agent.select_action(state)

            import copy

            adjusted = copy.copy(self.config)
            self._rl_agent.apply_action(adjusted, action)

            # Update extractor with adjusted parameters
            self.extractor.min_contrast = adjusted.min_contrast
            self.extractor.texture_floor = adjusted.texture_floor
            self.extractor.max_features = adjusted.max_features

        except Exception as e:
            logger.debug("RL tuning failed (using defaults): %s", e)

    def _apply_cnn_encoding(
        self,
        image: np.ndarray,
        feature_set: IrisFeatureSet,
    ) -> None:
        """Replace classical 16-bin descriptors with CNN embeddings."""
        import cv2

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        for feature in feature_set.features:
            try:
                from pupil_tracking.iris.extraction import _safe_patch

                patch = _safe_patch(gray, feature.x, feature.y, self.config.radius_px)
                feature.descriptor = self._cnn_encoder.encode_patch(patch)
                feature.cnn_descriptor = True
            except Exception as e:
                logger.debug("CNN encoding failed for feature %d: %s", feature.id, e)

    def detect_from_ellipses(self, image, pupil, limbus, **kwargs):
        """Alias for :meth:`detect` (explicit geometry entry point)."""
        return self.detect(image, pupil, limbus, **kwargs)

    @staticmethod
    def _coverage(feature_set: IrisFeatureSet, roi) -> float:
        """Fraction of the annulus (coarsely) 'covered' by accepted features.

        Cover is estimated as the summed area of the accepted feature patches
        relative to the annulus area. It is an upper-bound style indicator, not
        a precise geometric coverage; interpreted only as a coarse metric.
        """
        if not roi.valid or roi.limbus_radius_px <= roi.pupil_radius_px:
            return 0.0
        annulus_area = np.pi * (
            roi.limbus_radius_px ** 2 - roi.pupil_radius_px ** 2
        )
        if annulus_area <= 0:
            return 0.0
        n = len(feature_set.features)
        # Approximate each accepted feature as covering a small disc of radius
        # 3 px. Not a precise geometric coverage; used only as a coarse
        # distribution indicator.
        r = 3.0
        area_sum = n * (np.pi * r * r)
        return float(min(area_sum / annulus_area, 1.0))

    @staticmethod
    def _finish(result: IrisDetectionResult, start: float) -> IrisDetectionResult:
        result.processing_time_ms = (time.perf_counter() - start) * 1000.0
        return result


def detect_iris_features(
    image: np.ndarray,
    pupil: Optional[EllipseParams],
    limbus: Optional[EllipseParams],
    *,
    config: Optional[IrisConfig] = None,
    reflection_remover: Optional[ReflectionRemover] = None,
    external_occlusion: Optional[np.ndarray] = None,
) -> IrisDetectionResult:
    """Convenience one-shot iris-feature detection.

    Equivalent to constructing an :class:`IrisFeatureDetector` and calling
    :meth:`IrisFeatureDetector.detect`.
    """
    detector = IrisFeatureDetector(
        config=config,
        reflection_remover=reflection_remover,
    )
    return detector.detect(
        image,
        pupil,
        limbus,
        external_occlusion=external_occlusion,
    )
