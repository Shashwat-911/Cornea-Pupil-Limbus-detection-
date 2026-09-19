"""
Central configuration — single source of truth.

Provides a clean, validated, dataclass-based configuration for the
entire pupil-tracking system.  Every subsystem (detection, fitting,
video, calibration, training, ring detection, paths) draws its
parameters from here.

Usage
-----
    from pupil_tracking.utils.config import get_config, set_config

    # Use global defaults
    cfg = get_config()

    # Or load from file
    cfg = PupilTrackingConfig.load("my_config.json")
    set_config(cfg)

    # Video mode relaxes detection thresholds
    cfg.apply_video_mode()

    # Access ring configuration
    print(cfg.ring.classifier_path)
    print(cfg.ring.docked_threshold_value)

Notes
-----
*   ``apply_video_mode()`` must be called before video processing
    begins — it relaxes detection thresholds because the Kalman
    filter provides temporal smoothing that compensates for noisier
    per-frame detections.

*   JSON round-trip via ``save()`` / ``load()`` silently ignores
    unknown keys, making configs forward-compatible.

*   Ring configuration (``RingConfig``) was added to support the
    adaptive pipeline that handles both docked (suction ring present)
    and pre-docked (no ring) eye images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ════════════════════════════════════════════════════════════════
# Sub-configuration dataclasses
# ════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    """ML model parameters.

    Attributes
    ----------
    encoder : str
        Backbone encoder architecture (e.g. ``"resnet34"``).
    input_size : int
        Input image resolution (square).  512 for single-image,
        256 typical for video mode via ``FastInference``.
    num_classes : int
        Segmentation classes: background (0), pupil (1), iris (2),
        and optionally suction_ring (3) when using ring-aware models.
    pretrained : bool
        Whether to use ImageNet-pretrained encoder weights.
    model_path : str
        Default path to the trained model checkpoint.
    confidence_threshold : float
        Minimum segmentation confidence for mask extraction.
    device : str
        Compute device: ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``.
    """

    encoder: str = "resnet34"
    input_size: int = 512
    num_classes: int = 3
    pretrained: bool = True
    model_path: str = "models/best_model.pth"
    confidence_threshold: float = 0.5
    device: str = "auto"
    enable_multiscale: bool = True
    multiscale_sizes: tuple = (448, 512, 640)


@dataclass
class DetectionConfig:
    """Detection thresholds.

    These control the minimum quality required for a detection to
    be accepted.  In video mode, ``min_pupil_confidence``,
    ``min_limbus_confidence``, and ``min_contour_points`` are
    relaxed by ``apply_video_mode()``.

    Attributes
    ----------
    min_pupil_confidence : float
        Minimum confidence to accept a pupil detection.
        Default 0.25 (image) → 0.15 (video).
    min_limbus_confidence : float
        Minimum confidence to accept a limbus detection.
        Default 0.25 (image) → 0.15 (video).
    min_pupil_area : int
        Minimum contour area in pixels for pupil candidates.
    min_limbus_area : int
        Minimum contour area in pixels for limbus candidates.
    min_contour_points : int
        Minimum contour points for ellipse fitting.
        Default 30 (image) → 20 (video).
    morph_kernel_size : int
        Morphological operation kernel size for mask cleanup.
    morph_iterations : int
        Number of morphological open/close iterations.
    enable_classical_fallback : bool
        Whether to try classical (non-ML) detection when ML fails.
    classical_confidence_penalty : float
        Multiplicative penalty applied to classical detection
        confidence (0.85 = 15% penalty).
    """

    min_pupil_confidence: float = 0.25
    min_limbus_confidence: float = 0.25
    min_pupil_area: int = 200
    min_limbus_area: int = 3000
    min_contour_points: int = 30
    morph_kernel_size: int = 5
    morph_iterations: int = 2
    enable_classical_fallback: bool = True
    classical_confidence_penalty: float = 0.85
    pre_docked_limbus_shrink_factor: float = 0.93


@dataclass
class FittingConfig:
    """Ellipse fitting parameters.

    Controls RANSAC-based robust ellipse fitting and Huber-weighted
    refinement used by the ``EllipseFitter``.

    Attributes
    ----------
    ransac_iterations : int
        Maximum RANSAC iterations.
    ransac_threshold : float
        Inlier distance threshold in pixels.
    ransac_min_inlier_ratio : float
        Minimum fraction of inlier points for a valid fit.
    min_fit_quality : float
        Minimum overall fit quality score (0–1).
    max_rms_residual : float
        Maximum RMS residual in pixels for acceptance.
    huber_delta : float
        Huber loss transition point for robust refinement.
    huber_max_iters : int
        Maximum iterations for Huber-weighted least squares.
    """

    ransac_iterations: int = 500
    ransac_threshold: float = 1.5
    ransac_min_inlier_ratio: float = 0.60
    min_fit_quality: float = 0.30
    max_rms_residual: float = 5.0
    huber_delta: float = 1.5
    huber_max_iters: int = 15


@dataclass
class VideoConfig:
    """Video and real-time processing parameters.

    Controls temporal smoothing (Kalman filter), blink detection,
    carry-forward behaviour, and frame-rate management.

    Attributes
    ----------
    enable_kalman : bool
        Whether to apply Kalman smoothing to detections.
    kalman_process_noise : float
        Kalman filter process noise (Q diagonal).  Lower values
        produce smoother output but slower response to changes.
    kalman_measurement_noise : float
        Kalman filter measurement noise (R diagonal).  Higher
        values trust predictions more than measurements.
    buffer_size : int
        Temporal buffer depth for multi-frame consensus.
    consensus_threshold : float
        Minimum agreement ratio across buffer for consensus.
    enable_blink_detection : bool
        Whether to classify low-confidence frames as blinks.
    blink_confidence_threshold : float
        Pupil confidence below this triggers blink classification.
    target_fps : float
        Target processing frame rate.
    skip_frames : int
        Number of frames to skip between processed frames.
        0 = process every frame.
    max_carry_forward_frames : int
        Maximum consecutive frames to coast on Kalman prediction
        when detection fails.  After this limit, returns empty.
    carry_forward_decay : float
        Confidence multiplier per carry-forward frame.
        0.85^5 ≈ 0.44, so confidence halves after ~4 frames.
    """

    enable_kalman: bool = True
    kalman_process_noise: float = 0.1
    kalman_measurement_noise: float = 1.0
    buffer_size: int = 5
    consensus_threshold: float = 0.6
    enable_blink_detection: bool = True
    blink_confidence_threshold: float = 0.3
    target_fps: float = 30.0
    skip_frames: int = 0
    max_carry_forward_frames: int = 5
    carry_forward_decay: float = 0.85


from enum import Enum


class CalibrationMode(str, Enum):
    """Supported calibration modes."""
    ANATOMICAL_ANCHOR = "ANATOMICAL_ANCHOR"  # Default: 12.0mm horizontal corneal standard
    FIXED_PIXEL_SCALE = "FIXED_PIXEL_SCALE"  # External/fixed px_per_mm scale
    RING_REFLECTION = "RING_REFLECTION"      # Known LED/suction ring dimension


@dataclass
class CalibrationConfig:
    """Spatial calibration parameters.

    Used to convert pixel measurements to millimetres using
    known physical dimensions (suction ring, corneal diameter, or fixed scale).

    Attributes
    ----------
    mode : str
        Active calibration mode: 'ANATOMICAL_ANCHOR', 'FIXED_PIXEL_SCALE', or 'RING_REFLECTION'.
    suction_ring_diameter_mm : float
        Known diameter of the suction ring or placido ring in mm.
        Used for auto-calibration when detected in frame.
    corneal_diameter_mm : float
        Average horizontal visible iris diameter (HVID) in mm.
        Used as reference anchor in ANATOMICAL_ANCHOR mode.
    manual_px_per_mm : float or None
        Fixed scale ratio for FIXED_PIXEL_SCALE mode.
    manual_mm_per_px : float or None
        Alternative fixed scale ratio (mm per pixel).
    enable_auto_calibration : bool
        Whether to attempt auto-calibration from detected landmarks.
    """

    mode: str = "FIXED_PIXEL_SCALE"
    suction_ring_diameter_mm: float = 9.4
    corneal_diameter_mm: float = 12.0
    manual_px_per_mm: Optional[float] = 58.2
    manual_mm_per_px: Optional[float] = None
    enable_auto_calibration: bool = True


@dataclass
class PathConfig:
    """Filesystem paths.

    Attributes
    ----------
    model_dir : str
        Directory containing model checkpoints.
    data_dir : str
        Root directory for clinical data.
    annotation_dir : str
        Directory containing annotation files.
    output_dir : str
        Directory for pipeline output (overlays, CSVs, etc.).
    log_dir : str
        Directory for log files.
    ring_labels_path : str
        Path to the ring presence labels JSON file.
    ring_classifier_path : str
        Path to the ring classifier model checkpoint.
    """

    model_dir: str = "models"
    data_dir: str = "clinical_data"
    annotation_dir: str = "clinical_data/annotations"
    output_dir: str = "clinical_data/output"
    log_dir: str = "logs"
    ring_labels_path: str = "clinical_data/ring_labels.json"
    ring_classifier_path: str = "models/ring_classifier.pth"


@dataclass
class TrainingConfig:
    """Model training hyper-parameters.

    Attributes
    ----------
    epochs : int
        Maximum training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularisation weight.
    early_stopping_patience : int
        Epochs without improvement before stopping.
    val_ratio : float
        Fraction of images reserved for validation.
    augmentations_per_image : int
        Number of augmented copies per training image.
    num_workers : int
        DataLoader worker processes.
    use_amp : bool
        Whether to use automatic mixed precision (FP16).
    """

    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30
    val_ratio: float = 0.2
    augmentations_per_image: int = 50
    num_workers: int = 2
    use_amp: bool = True


@dataclass
class RingClassifierConfig:
    """Ring presence classifier parameters.

    Controls the lightweight CNN (MobileNetV2) that classifies
    images as ring-present or ring-absent before the main
    detection pipeline runs.

    Attributes
    ----------
    enabled : bool
        Whether to run the ring classifier as the first
        pipeline stage.  When ``False``, heuristic-only
        detection is used.
    classifier_path : str
        Path to the trained ring classifier checkpoint.
    confidence_threshold : float
        Minimum classifier confidence to accept without
        falling back to the heuristic detector.
    input_size : int
        Input resolution for the classifier (224 for MobileNetV2).
    """

    enabled: bool = True
    classifier_path: str = "models/ring_classifier.pth"
    confidence_threshold: float = 0.70
    input_size: int = 224


@dataclass
class RingHeuristicConfig:
    """Heuristic (traditional CV) ring detection parameters.

    Controls the Hough-circle and contour-based ring detection
    that serves as fallback when no trained classifier is
    available, or as geometry enrichment when the classifier
    detects a ring but cannot provide centre / radius.

    Attributes
    ----------
    enabled : bool
        Whether to use the heuristic detector as a fallback
        or validation signal.
    canny_low : int
        Lower Canny edge threshold.
    canny_high : int
        Upper Canny edge threshold.
    hough_dp : float
        Inverse ratio of accumulator resolution for HoughCircles.
    hough_min_dist : int
        Minimum pixel distance between detected circle centres.
    hough_param1 : int
        Canny upper threshold used internally by HoughCircles.
    hough_param2 : int
        Accumulator threshold for HoughCircles.
    radius_min_frac : float
        Minimum ring radius as a fraction of the smaller image
        dimension (0.25 = 25 %).
    radius_max_frac : float
        Maximum ring radius as a fraction of the smaller image
        dimension (0.48 = 48 %).
    min_circularity : float
        Minimum circularity for contour-based ring candidates.
    edge_density_threshold : float
        Edge-pixel fraction in the ring band above which a
        Hough candidate scores well.
    """

    enabled: bool = True
    canny_low: int = 30
    canny_high: int = 100
    hough_dp: float = 1.2
    hough_min_dist: int = 100
    hough_param1: int = 80
    hough_param2: int = 40
    radius_min_frac: float = 0.25
    radius_max_frac: float = 0.48
    min_circularity: float = 0.70
    edge_density_threshold: float = 0.12


@dataclass
class RingPreprocessingConfig:
    """Ring-aware preprocessing parameters.

    Controls how the adaptive preprocessor handles docked
    images differently from pre-docked images.

    Attributes
    ----------
    inner_margin : int
        Pixels to shrink inward from the detected ring radius
        when building the inner ROI mask.
    clahe_clip_multiplier : float
        CLAHE clip limit multiplier for docked images (applied
        on top of the base CLAHE clip).
    clahe_grid_docked : tuple of int
        CLAHE tile grid for the smaller ROI inside the ring.
    reflection_percentile : float
        Percentile threshold for specular reflection detection
        inside the ring opening.
    eyelid_suppression : bool
        Whether to apply soft eyelid suppression on pre-docked
        images (fade top / bottom 15 % of frame).
    eyelid_margin_frac : float
        Fraction of image height to suppress at top and bottom
        when ``eyelid_suppression`` is enabled.
    """

    inner_margin: int = 15
    clahe_clip_multiplier: float = 1.2
    clahe_grid_docked: tuple = (4, 4)
    reflection_percentile: float = 97.0
    eyelid_suppression: bool = True
    eyelid_margin_frac: float = 0.15


@dataclass
class DockedDetectionConfig:
    """Detection parameter overrides for docked (ring-present) images.

    When a suction ring is detected, the pipeline switches to
    these tighter thresholds because the ring constrains the
    search area and reduces ambiguity.

    Attributes
    ----------
    threshold_value : int
        Global binary threshold for pupil extraction.
    adaptive_block_size : int
        Adaptive threshold block size (must be odd, ≥ 3).
    adaptive_c : float
        Constant subtracted from the adaptive threshold mean.
    min_contour_area : int
        Minimum contour area for pupil candidates (pixels).
    max_contour_area : int
        Maximum contour area for pupil candidates (pixels).
    min_circularity : float
        Minimum circularity for pupil contour candidates.
    max_pupil_ring_ratio : float
        Maximum ratio of pupil equivalent radius to ring radius.
        Contours larger than this are assumed to be the ring.
    max_center_offset_ratio : float
        Maximum distance from pupil centre to ring centre,
        expressed as a fraction of the ring radius.
    """

    threshold_value: int = 35
    adaptive_block_size: int = 31
    adaptive_c: float = 10.0
    min_contour_area: int = 100
    max_contour_area: int = 30000
    min_circularity: float = 0.35
    max_pupil_ring_ratio: float = 0.60
    max_center_offset_ratio: float = 0.75


@dataclass
class PreDockedDetectionConfig:
    """Detection parameter overrides for pre-docked (no ring) images.

    Natural eye images have a wider field of view and may include
    eyelids, eyelashes, and variable lighting.  These thresholds
    are therefore more permissive than the docked-mode counterparts.

    Attributes
    ----------
    threshold_value : int
        Global binary threshold for pupil extraction.
    adaptive_block_size : int
        Adaptive threshold block size (must be odd, ≥ 3).
    adaptive_c : float
        Constant subtracted from the adaptive threshold mean.
    min_contour_area : int
        Minimum contour area for pupil candidates (pixels).
    max_contour_area : int
        Maximum contour area for pupil candidates (pixels).
    min_circularity : float
        Minimum circularity for pupil contour candidates.
    """

    threshold_value: int = 40
    adaptive_block_size: int = 51
    adaptive_c: float = 12.0
    min_contour_area: int = 150
    max_contour_area: int = 50000
    min_circularity: float = 0.30


@dataclass
class GrayscaleConfig:
    """Configuration for grayscale-optimized preprocessing.

    When native grayscale images are passed to the pipeline, this
    config controls an optimized path that avoids lossy BGR round-trips
    and applies multi-scale CLAHE with unsharp masking for enhanced
    boundary detection.

    Attributes
    ----------
    enabled : bool
        Use the grayscale-optimized path when input is single-channel.
    clahe_clip_low : float
        CLAHE clip limit for the detail-preserving pass.
    clahe_clip_high : float
        CLAHE clip limit for the boundary-revealing pass.
    clahe_grid_size : int
        CLAHE tile grid size (shared by both passes).
    clahe_merge_weight : float
        Weight for the low-clip result in the merge (0–1).
    unsharp_sigma : float
        Gaussian sigma for the unsharp mask blur.
    unsharp_amount : float
        Sharpening strength (0 = off).
    unsharp_threshold : int
        Minimum pixel difference to apply sharpening (avoids
        amplifying noise in flat regions).
    gaussian_kernel_size : int
        Fallback Gaussian blur kernel size (must be odd).
    use_bilateral : bool
        Use bilateral filter instead of Gaussian for denoising.
    """

    enabled: bool = True
    clahe_clip_low: float = 1.5
    clahe_clip_high: float = 4.0
    clahe_grid_size: int = 8
    clahe_merge_weight: float = 0.6
    unsharp_sigma: float = 2.0
    unsharp_amount: float = 0.5
    unsharp_threshold: int = 5
    gaussian_kernel_size: int = 5
    use_bilateral: bool = True


@dataclass
class MeasurementStabilizationConfig:
    """Calibration stabilization for consistent mm measurements.

    Controls EMA (exponential moving average) smoothing and outlier
    rejection on the px-to-mm calibration ratio.  This prevents a
    single noisy limbus detection from shifting all downstream mm
    measurements.

    Attributes
    ----------
    enable_ema_smoothing : bool
        Enable EMA smoothing on calibration values.
    ema_alpha : float
        EMA smoothing factor (0–1).  Lower = smoother, slower response.
    outlier_sigma : float
        Reject new calibration if it deviates more than this many
        standard deviations from the running mean.
    min_samples_for_rejection : int
        Minimum calibration history before outlier rejection activates.
    max_calibration_history : int
        Maximum number of calibration entries to keep.
    """

    enable_ema_smoothing: bool = True
    ema_alpha: float = 0.15
    outlier_sigma: float = 2.0
    min_samples_for_rejection: int = 5
    max_calibration_history: int = 50


@dataclass
class SubPixelConfig:
    """Sub-pixel refinement configuration.

    Controls the precision of contour point localization used by
    the SmartContourFitter for achieving surgical-grade accuracy.

    Attributes
    ----------
    use_scharr : bool
        Use Scharr operator (better rotational symmetry) instead of Sobel.
    interpolation_step : float
        Sampling step along gradient normal in pixels.
    use_parabolic_peak : bool
        Fit a parabola to the 3 points around the gradient maximum
        for true sub-pixel peak localization.
    use_multiscale_gradient : bool
        Fuse gradient magnitudes from multiple scales.
    gradient_scales : tuple
        Gaussian sigma values for each scale.
    gradient_scale_weights : tuple
        Weights for each scale in the fusion.
    use_weighted_fit : bool
        Use gradient-magnitude-weighted circle fitting.
    multi_pass_ransac : bool
        Run a second RANSAC pass with tightened threshold.
    ransac_tighten_factor : float
        Second-pass threshold = first-pass × this factor.
    bootstrap_uncertainty : bool
        Use bootstrap resampling for uncertainty estimation.
    bootstrap_n_samples : int
        Number of bootstrap resamples.
    """

    use_scharr: bool = True
    interpolation_step: float = 0.25
    use_parabolic_peak: bool = True
    use_multiscale_gradient: bool = True
    gradient_scales: tuple = (1, 3)
    gradient_scale_weights: tuple = (0.6, 0.4)
    use_weighted_fit: bool = True
    multi_pass_ransac: bool = True
    ransac_tighten_factor: float = 0.5
    bootstrap_uncertainty: bool = True
    bootstrap_n_samples: int = 50


@dataclass
class RingConfig:
    """Top-level ring detection and adaptive pipeline configuration.

    Aggregates all ring-related sub-configurations into a single
    section.  Accessed via ``cfg.ring``.

    Attributes
    ----------
    classifier : RingClassifierConfig
        CNN classifier parameters.
    heuristic : RingHeuristicConfig
        Traditional CV ring detection parameters.
    preprocessing : RingPreprocessingConfig
        Adaptive preprocessing parameters.
    docked : DockedDetectionConfig
        Detection overrides for docked images.
    pre_docked : PreDockedDetectionConfig
        Detection overrides for pre-docked images.
    ring_class_weight : float
        Loss weight for the ring segmentation class (class 3)
        during 4-class training.  Higher values penalise
        ring mis-segmentation more.
    default_mode : str
        Default ring detection mode: ``"auto"`` (classify each
        image), ``"docked"`` (assume ring), or ``"pre_docked"``
        (assume no ring).
    merge_weight_classifier : float
        Weight of the CNN classifier signal when merging with
        the heuristic signal (0–1).  The heuristic weight is
        ``1 − merge_weight_classifier``.
    agreement_bonus : float
        Multiplicative confidence bonus when classifier and
        heuristic agree on ring status.
    disagreement_penalty : float
        Multiplicative confidence penalty when classifier and
        heuristic disagree.
    """

    classifier: RingClassifierConfig = field(
        default_factory=RingClassifierConfig,
    )
    heuristic: RingHeuristicConfig = field(
        default_factory=RingHeuristicConfig,
    )
    preprocessing: RingPreprocessingConfig = field(
        default_factory=RingPreprocessingConfig,
    )
    docked: DockedDetectionConfig = field(
        default_factory=DockedDetectionConfig,
    )
    pre_docked: PreDockedDetectionConfig = field(
        default_factory=PreDockedDetectionConfig,
    )

    ring_class_weight: float = 1.2
    default_mode: str = "auto"
    merge_weight_classifier: float = 0.65
    agreement_bonus: float = 1.15
    disagreement_penalty: float = 0.80


@dataclass
class RegistrationConfig:
    """Cyclotorsion detection and iris registration parameters.

    Controls polar unwrapping, enhancement, multi-stream detection,
    and fusion for the registration pipeline.

    Attributes
    ----------
    enabled : bool
        Whether registration/cyclotorsion detection is available.
    polar_num_angles : int
        Number of angular samples in polar unwrapping (360 = 1°/bin).
    polar_num_radial : int
        Number of radial samples between pupil and limbus.
    polar_inner_margin : float
        Fractional inset from pupil boundary (0–1).
    polar_outer_margin : float
        Fractional inset from limbus boundary (0–1).
    enhance_clahe_clip : float
        CLAHE clip limit for iris enhancement.
    enhance_clahe_grid : int
        CLAHE tile grid size.
    enhance_gabor_wavelengths : tuple
        Gabor filter wavelengths for texture enhancement.
    enable_phase_correlation : bool
        Enable Stream A: phase-only correlation.
    enable_deep_matcher : bool
        Enable Stream B: deep feature matching.
    enable_ink_tracker : bool
        Enable Stream C: surgical ink marker tracking.
    enable_vessel_tracker : bool
        Enable Stream D: limbal vessel tracking.
    enable_custom_feature : bool
        Enable Stream E: custom-trained iris feature model.
    custom_feature_model_path : str
        Path to ONNX model for Stream E.
    fusion_method : str
        Fusion method: 'weighted_median', 'bayesian', 'robust_mean'.
    fusion_agreement_threshold_deg : float
        Maximum angular disagreement (degrees) for streams to be
        considered in agreement.
    fusion_min_streams : int
        Minimum number of valid streams for a fused result.
    poc_upsample_factor : int
        Sub-pixel upsample factor for phase correlation.
    deep_matcher_max_keypoints : int
        Maximum keypoints for the deep matcher stream.
    ink_min_markers : int
        Minimum ink markers required for torsion estimation.
    ink_hsv_lower : tuple
        Lower HSV bound for purple/blue ink detection.
    ink_hsv_upper : tuple
        Upper HSV bound for purple/blue ink detection.
    vessel_min_bifurcations : int
        Minimum vessel bifurcation points for tracking.
    """

    enabled: bool = True

    # Polar unwrapping
    polar_num_angles: int = 360
    polar_num_radial: int = 64
    polar_inner_margin: float = 0.10
    polar_outer_margin: float = 0.10

    # Enhancement
    enhance_clahe_clip: float = 3.0
    enhance_clahe_grid: int = 8
    enhance_gabor_wavelengths: tuple = (4, 8, 16)

    # Stream enables
    enable_phase_correlation: bool = True
    enable_deep_matcher: bool = True
    enable_ink_tracker: bool = True
    enable_vessel_tracker: bool = True
    enable_custom_feature: bool = True

    # Custom feature model
    custom_feature_model_path: str = "models/iris_features/iris_feature_model.onnx"

    # Fusion
    fusion_method: str = "weighted_median"
    fusion_agreement_threshold_deg: float = 1.0
    fusion_min_streams: int = 2

    # Stream-specific parameters
    poc_upsample_factor: int = 100

    deep_matcher_max_keypoints: int = 500

    ink_min_markers: int = 2
    ink_hsv_lower: tuple = (120, 50, 50)
    ink_hsv_upper: tuple = (160, 255, 255)

    vessel_min_bifurcations: int = 3


# ════════════════════════════════════════════════════════════════
# Top-level configuration container
# ════════════════════════════════════════════════════════════════


@dataclass
class PupilTrackingConfig:
    """Top-level configuration container.

    Aggregates all sub-configurations into a single object that
    serves as the single source of truth for the entire system.

    Attributes
    ----------
    model : ModelConfig
        ML model parameters.
    detection : DetectionConfig
        Detection thresholds.
    fitting : FittingConfig
        Ellipse fitting parameters.
    video : VideoConfig
        Video / real-time processing parameters.
    calibration : CalibrationConfig
        Spatial calibration parameters.
    paths : PathConfig
        Filesystem paths.
    training : TrainingConfig
        Model training hyper-parameters.
    ring : RingConfig
        Ring detection and adaptive pipeline parameters.
    grayscale : GrayscaleConfig
        Grayscale-optimized preprocessing parameters.
    measurement_stabilization : MeasurementStabilizationConfig
        Calibration EMA smoothing and outlier rejection.
    subpixel : SubPixelConfig
        Sub-pixel refinement precision controls.
    registration : RegistrationConfig
        Cyclotorsion detection and iris registration parameters.
    video_mode : bool
        Whether video-mode relaxations have been applied.
        Set automatically by ``apply_video_mode()``.
    debug : bool
        Enable verbose debug logging and visualisation.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    fitting: FittingConfig = field(default_factory=FittingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    grayscale: GrayscaleConfig = field(default_factory=GrayscaleConfig)
    measurement_stabilization: MeasurementStabilizationConfig = field(
        default_factory=MeasurementStabilizationConfig,
    )
    subpixel: SubPixelConfig = field(default_factory=SubPixelConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)

    video_mode: bool = False
    debug: bool = False

    # ── video mode preset ───────────────────────────────────────

    def apply_video_mode(self) -> None:
        """Relax thresholds for real-time video processing.

        Must be called before video processing begins (typically
        inside ``init_video_mode()``).

        Changes applied::

            detection.min_pupil_confidence   : 0.25 → 0.15
            detection.min_limbus_confidence  : 0.25 → 0.15
            detection.min_contour_points     : 30   → 20
            video.enable_kalman              : True  (ensured)
            video_mode                       : True  (flag set)

        The rationale is that Kalman temporal smoothing compensates
        for the noisier per-frame detections that result from lower
        thresholds, yielding smoother output with fewer dropped frames.
        """
        self.video_mode = True
        self.detection.min_pupil_confidence = 0.15
        self.detection.min_limbus_confidence = 0.15
        self.detection.min_contour_points = 20
        self.video.enable_kalman = True

    # ── ring mode helpers ───────────────────────────────────────

    def get_ring_detection_params(self, mode: str) -> dict:
        """Return detection parameters for the given ring mode.

        Parameters
        ----------
        mode : str
            ``"docked"`` or ``"pre_docked"``.

        Returns
        -------
        dict
            Flat dictionary of detection parameters suitable for
            passing to the adaptive pipeline.

        Raises
        ------
        ValueError
            If *mode* is not ``"docked"`` or ``"pre_docked"``.
        """
        if mode == "docked":
            src = self.ring.docked
        elif mode == "pre_docked":
            src = self.ring.pre_docked
        else:
            raise ValueError(
                f"Unknown ring mode {mode!r}; expected 'docked' or 'pre_docked'"
            )

        return {
            "threshold_value": src.threshold_value,
            "adaptive_block_size": src.adaptive_block_size,
            "adaptive_c": src.adaptive_c,
            "min_contour_area": src.min_contour_area,
            "max_contour_area": src.max_contour_area,
            "min_circularity": src.min_circularity,
            "morph_kernel_size": self.detection.morph_kernel_size,
        }

    def get_num_segmentation_classes(self) -> int:
        """Return the effective number of segmentation classes.

        Returns 4 when ring-aware training is configured, otherwise 3.

        Returns
        -------
        int
            3 or 4.
        """
        return self.model.num_classes

    # ── persistence ─────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save configuration to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.  Parent directories are created
            automatically if they do not exist.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        # Convert tuples to lists for JSON serialisation
        _convert_tuples(data)

        with open(out, "w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "PupilTrackingConfig":
        """Load configuration from a JSON file.

        Creates a default config first, then patches it with values
        from the file.  Unknown keys are silently ignored, making
        configs forward-compatible across versions.

        Parameters
        ----------
        path : str
            Path to JSON config file.

        Returns
        -------
        PupilTrackingConfig
            Loaded and patched configuration.

        Raises
        ------
        FileNotFoundError
            If the config file does not exist.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        with open(path) as fh:
            data = json.load(fh)
        cfg = cls()
        _update_dc(cfg, data)
        return cfg


# ════════════════════════════════════════════════════════════════
# Helpers — recursive dataclass patching and serialisation
# ════════════════════════════════════════════════════════════════


def _update_dc(obj: object, data: dict) -> None:
    """Recursively patch a dataclass instance from a dictionary.

    Walks the dict tree and sets matching attributes on the
    dataclass.  Sub-dataclasses are patched recursively rather
    than replaced, preserving any defaults not present in *data*.

    Unknown keys (not present as attributes on *obj*) are
    silently skipped for forward-compatibility.

    Parameters
    ----------
    obj : object
        Dataclass instance to patch.
    data : dict
        Dictionary of values to apply.
    """
    import dataclasses as dc

    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if dc.is_dataclass(current) and isinstance(value, dict):
            _update_dc(current, value)
        else:
            # Convert lists back to tuples where the field type is tuple
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)


def _convert_tuples(data: dict) -> None:
    """Recursively convert tuple values to lists for JSON serialisation.

    Modifies *data* in place.

    Parameters
    ----------
    data : dict
        Dictionary (typically from ``dataclasses.asdict``) that may
        contain tuple values which are not natively JSON-serialisable.
    """
    for key, value in data.items():
        if isinstance(value, tuple):
            data[key] = list(value)
        elif isinstance(value, dict):
            _convert_tuples(value)


# ════════════════════════════════════════════════════════════════
# Global singleton
# ════════════════════════════════════════════════════════════════

_global_config: Optional[PupilTrackingConfig] = None


def get_config() -> PupilTrackingConfig:
    """Get the global configuration singleton.

    Creates a default ``PupilTrackingConfig`` on first call.
    Subsequent calls return the same instance.

    Returns
    -------
    PupilTrackingConfig
        The global configuration object.
    """
    global _global_config
    if _global_config is None:
        _global_config = PupilTrackingConfig()
    return _global_config


def set_config(config: PupilTrackingConfig) -> None:
    """Replace the global configuration singleton.

    Parameters
    ----------
    config : PupilTrackingConfig
        New configuration to install as the global singleton.
    """
    global _global_config
    _global_config = config


def reset_config() -> None:
    """Reset the global configuration to defaults.

    Useful in tests to ensure a clean state between test cases.
    """
    global _global_config
    _global_config = None