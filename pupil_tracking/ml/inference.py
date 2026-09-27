"""
Single-image and batch inference engine.

Converts raw images -> segmentation masks -> contours -> ellipse fits
in a single, auditable pipeline.

Fixes:
    - Validates anatomy at MODEL resolution (512x512), not original
      image resolution.
    - Detects border-touching contours (incomplete segmentation).
    - Better limbus extraction from iris mask outer boundary.
    - Diagnostic logging for failed detections.
    - ROBUST model loading: handles raw state_dict, wrapped checkpoint,
      and architecture mismatches gracefully.
    - PyTorch is optional — gracefully degrades when not installed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# PyTorch is optional in production — we prefer ONNX Runtime
# This module is only used as fallback when ONNX models aren't available
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False

# Lazy import — architecture.py requires torch
if _HAS_TORCH:
    from pupil_tracking.ml.architecture import EyeSegmentationModel, get_device
else:
    EyeSegmentationModel = None
    get_device = lambda x: None

from pupil_tracking.core.ellipse_fitter import EllipseFitter
from pupil_tracking.utils.types import (
    EllipseParams,
    EyeDetectionResult,
    PupilDetection,
    LimbusDetection,
    DetectionQuality,
    DetectionMethod,
    QualityFlag,
    FrameMetadata,
    FitResult,
    ANATOMICAL_LIMITS,
    assign_quality_grade,
    quality_to_flag,
)
from pupil_tracking.utils.config import get_config
from pupil_tracking.utils.logger import get_logger
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
from pupil_tracking.preprocessing.suction_ring_masker import SuctionRingMasker


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PUPIL_MIN_FRAC = 0.012
_PUPIL_MAX_FRAC = 0.42
_LIMBUS_MIN_FRAC = 0.04
_LIMBUS_MAX_FRAC = 0.49
_MIN_MASK_FRAC = 0.0005
_MAX_MASK_FRAC = 0.90
_BORDER_MARGIN = 3

# Sub-pixel refinement constants
_SUBPIXEL_SEARCH_RADIUS = 3
_SUBPIXEL_STEP = 0.25  # finer than default 0.5 for surgical precision


class SegmentationInference:
    """Production inference engine with robust model loading.

    When PyTorch is not installed, the instance is created but
    ``available`` returns ``False`` and ``detect()`` returns empty
    results.  This allows the rest of the application (e.g. classical
    fallback detectors) to function without torch.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config=None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = config or get_config()
        self.logger = get_logger()

        model_path = model_path or self.cfg.model.model_path
        self.model_path = model_path
        self.model = None
        self._model_loaded = False

        # Initialise input size, fitter, and preprocessors regardless
        # of torch availability — they are used by other code paths.
        self.input_size = self.cfg.model.input_size
        self.fitter = EllipseFitter()

        self._reflection_remover = ReflectionRemover(
            brightness_threshold=220,
            min_reflection_area=15,
            inpaint_radius=5,
            detect_red_highlights=True,
            red_threshold_offset=25,
        )
        self._ring_masker = SuctionRingMasker()
        self._red_light_filter = None  # Lazy initialization

        # ── PyTorch gate ──────────────────────────────────────────
        if not _HAS_TORCH:
            self.device = None
            self.logger.warning(
                "PyTorch not installed — ML segmentation inference "
                "disabled.  Install with: pip install torch"
            )
            return

        dev_str = device or self.cfg.model.device
        self.device = get_device(dev_str)

        if Path(model_path).exists():
            self._load_model_robust(model_path)
        else:
            self.logger.warning(
                "Model not found at %s -- ML detection unavailable",
                model_path,
            )

    # ================================================================
    # Red Light Filter Control
    # ================================================================

    def _get_red_light_filter(self):
        """Lazy import and return red light filter."""
        from pupil_tracking.preprocessing.red_light_filter import RedLightFilter

        return RedLightFilter(
            red_threshold=200,
            dominance_offset=30,
            min_area=5,
            enable_inpaint=True,
            inpaint_radius=3,
            enable_temporal=False,
        )

    def set_red_light_filter_enabled(self, enabled: bool) -> None:
        """Enable or disable the red light filter."""
        if enabled:
            self._red_light_filter = self._get_red_light_filter()
        else:
            from pupil_tracking.preprocessing.red_light_filter import RedLightFilter

            self._red_light_filter = RedLightFilter(
                red_threshold=255,
                dominance_offset=1000,
                min_area=100000,
                enable_inpaint=False,
                enable_temporal=False,
            )
        self.logger.info("Red light filter enabled=%s", enabled)

    def set_red_light_temporal_mode(self, enabled: bool) -> None:
        """Enable temporal mode for red light filtering (for video)."""
        if not hasattr(self, "_red_light_filter") or self._red_light_filter is None:
            self._red_light_filter = self._get_red_light_filter()
        self._red_light_filter.enable_temporal = enabled
        if not enabled:
            self._red_light_filter.reset_temporal()
        self.logger.info("Red light temporal mode enabled=%s", enabled)

    def reset_red_light_temporal(self) -> None:
        """Reset temporal tracking state for red light filter."""
        if hasattr(self, "_red_light_filter") and self._red_light_filter is not None:
            self._red_light_filter.reset_temporal()

    # ================================================================
    # Robust Model Loading
    # ================================================================

    def _load_model_robust(self, model_path: str) -> None:
        """Try multiple strategies to load the model checkpoint.

        Handles:
        1. EyeSegmentationModel.load() classmethod (standard)
        2. Raw state_dict saved via torch.save(model.state_dict(), path)
        3. Wrapped checkpoint {'state_dict': ..., ...}
        4. Wrapped checkpoint {'model_state_dict': ..., ...}
        5. Architecture mismatch (strict=False)

        Requires PyTorch — callers must check ``_HAS_TORCH`` first.
        """
        if not _HAS_TORCH:
            self.logger.error("Cannot load model: PyTorch is not installed")
            return

        # ── Strategy 1: Standard .load() classmethod ──
        try:
            self.model = EyeSegmentationModel.load(
                model_path, device=str(self.device)
            ).to(self.device)
            self.model.eval()
            self._model_loaded = True
            self.logger.info(
                "Model loaded (standard) from %s (device=%s)",
                model_path,
                self.device,
            )
            return
        except Exception as e1:
            self.logger.warning(
                "Standard model load failed: %s — trying fallbacks…",
                e1,
            )

        # ── Load raw checkpoint ──
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False,
            )
        except Exception as e_load:
            self.logger.error("Cannot load checkpoint file at all: %s", e_load)
            return

        # ── Extract state_dict from various formats ──
        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            self.logger.error(
                "Cannot extract state_dict from checkpoint (type=%s, keys=%s)",
                type(checkpoint).__name__,
                list(checkpoint.keys())[:5] if isinstance(checkpoint, dict) else "N/A",
            )
            return

        # ── Strategy 2: Create model then load state_dict ──
        # Try different num_classes since we don't know what was trained
        num_classes_options = [3, 2, 4]

        # Also try to infer num_classes from state_dict
        inferred_nc = self._infer_num_classes(state_dict)
        if inferred_nc is not None and inferred_nc not in num_classes_options:
            num_classes_options.insert(0, inferred_nc)
        elif inferred_nc is not None:
            num_classes_options.remove(inferred_nc)
            num_classes_options.insert(0, inferred_nc)

        for nc in num_classes_options:
            try:
                model = EyeSegmentationModel(num_classes=nc)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    self.logger.debug(
                        "Loaded with %d missing keys (nc=%d): %s",
                        len(missing),
                        nc,
                        missing[:3],
                    )
                if unexpected:
                    self.logger.debug(
                        "Loaded with %d unexpected keys (nc=%d): %s",
                        len(unexpected),
                        nc,
                        unexpected[:3],
                    )
                self.model = model.to(self.device)
                self.model.eval()
                self._model_loaded = True
                self.logger.info(
                    "Model loaded (fallback, num_classes=%d) from %s "
                    "(missing=%d, unexpected=%d)",
                    nc,
                    model_path,
                    len(missing),
                    len(unexpected),
                )
                return
            except TypeError:
                # EyeSegmentationModel might not accept num_classes
                try:
                    model = EyeSegmentationModel()
                    model.load_state_dict(state_dict, strict=False)
                    self.model = model.to(self.device)
                    self.model.eval()
                    self._model_loaded = True
                    self.logger.info(
                        "Model loaded (fallback, default init) from %s",
                        model_path,
                    )
                    return
                except Exception:
                    pass
            except Exception as e_nc:
                self.logger.debug("Fallback load with nc=%d failed: %s", nc, e_nc)
                continue

        # ── Strategy 3: Try PupilSegmentationModel from training ──
        try:
            from pupil_tracking.ml.model import PupilSegmentationModel

            for nc in num_classes_options:
                try:
                    model = PupilSegmentationModel(num_classes=nc)
                    model.load_state_dict(state_dict, strict=False)
                    self.model = model.to(self.device)
                    self.model.eval()
                    self._model_loaded = True
                    self.logger.info(
                        "Model loaded (PupilSegmentationModel, nc=%d) from %s",
                        nc,
                        model_path,
                    )
                    return
                except Exception:
                    continue
        except ImportError:
            pass

        self.logger.error("ALL model loading strategies failed for %s", model_path)

    @staticmethod
    def _extract_state_dict(checkpoint) -> Optional[Dict]:
        """Extract the actual state_dict from various checkpoint formats."""
        if not isinstance(checkpoint, dict):
            return None

        # Direct key lookup
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # Check if the dict itself IS a state_dict
        # (keys look like 'encoder.conv1.weight', etc.)
        sample_keys = list(checkpoint.keys())[:10]
        if sample_keys and all(
            isinstance(k, str) and ("." in k or "weight" in k or "bias" in k)
            for k in sample_keys
        ):
            return checkpoint

        # Last resort: find the first dict value that looks like a state_dict
        for key, val in checkpoint.items():
            if isinstance(val, dict):
                sub_keys = list(val.keys())[:5]
                if sub_keys and any("." in str(k) for k in sub_keys):
                    return val

        # Maybe it's a raw state_dict with short keys
        if sample_keys:
            return checkpoint

        return None

    @staticmethod
    def _infer_num_classes(state_dict: Dict) -> Optional[int]:
        """Try to infer num_classes from the final layer shape."""
        # Look for the output layer (usually the last conv with
        # 'out', 'final', 'head', 'classifier', or 'o.' in the name)
        for key in reversed(list(state_dict.keys())):
            if any(
                pat in key.lower()
                for pat in ("out", "final", "head", "classifier", "o.")
            ):
                tensor = state_dict[key]
                if hasattr(tensor, "shape") and len(tensor.shape) >= 1:
                    return int(tensor.shape[0])
        return None

    @property
    def available(self) -> bool:
        """Whether the ML inference engine is ready to use."""
        return self._model_loaded and _HAS_TORCH

    # ================================================================
    # Grayscale safety
    # ================================================================

    @staticmethod
    def _ensure_bgr(image: np.ndarray) -> np.ndarray:
        """Normalise any input image to BGR uint8.

        Handles grayscale (2D), single-channel (H,W,1), BGRA (H,W,4),
        and RGB/BGR (H,W,3) inputs.  Returns a (H,W,3) BGR array.
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3:
            c = image.shape[2]
            if c == 1:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if c == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    # ================================================================
    # Main entry point
    # ================================================================

    def detect(
        self,
        image: np.ndarray,
        frame_number: int = -1,
        source: str = "",
    ) -> EyeDetectionResult:
        t0 = time.time()

        # Phase 1: normalise input to BGR
        image = self._ensure_bgr(image)

        result = EyeDetectionResult()
        result.metadata = FrameMetadata(
            frame_number=frame_number,
            source=source,
            image_height=image.shape[0],
            image_width=image.shape[1],
        )

        if not self._model_loaded or self.model is None or not _HAS_TORCH:
            result.metadata.processing_time_ms = (time.time() - t0) * 1000.0
            return result

        h_orig, w_orig = image.shape[:2]
        scale_x = w_orig / self.input_size
        scale_y = h_orig / self.input_size

        tensor, gray_resized = self._preprocess(image, frame_number=frame_number)

        with torch.no_grad():
            # Phase 3: Multi-scale inference (single-image only)
            probs_np = self._multiscale_segment(tensor, gray_resized)

        pred_mask = probs_np.argmax(axis=0).astype(np.uint8)

        pupil_conf = self._class_confidence(probs_np, pred_mask, class_id=1)
        iris_conf = self._class_confidence(probs_np, pred_mask, class_id=2)

        # Phase 5: Adaptive morphological kernel per structure
        pupil_mask = self._postprocess_mask(
            pred_mask,
            class_id=1,
            min_area=max(
                50,
                int(self.input_size * self.input_size * _MIN_MASK_FRAC),
            ),
            structure="pupil",
        )
        iris_mask = self._postprocess_mask(
            pred_mask,
            class_id=2,
            min_area=max(
                200,
                int(self.input_size * self.input_size * _MIN_MASK_FRAC * 3),
            ),
            structure="limbus",
        )

        # Phase 5.1: Gradient-guided mask boundary refinement
        pupil_mask = self._refine_mask_boundary(pupil_mask, gray_resized)
        iris_mask = self._refine_mask_boundary(iris_mask, gray_resized)

        result.pupil = self._detect_structure(
            pupil_mask,
            confidence=pupil_conf,
            scale_x=scale_x,
            scale_y=scale_y,
            structure="pupil",
            gray_model=gray_resized,
        )

        result.limbus = self._detect_limbus_from_iris(
            iris_mask,
            pupil_mask,
            confidence=iris_conf,
            scale_x=scale_x,
            scale_y=scale_y,
            gray_model=gray_resized,
            pupil_hint=result.pupil,
        )

        # Store raw mask for downstream SmartContourFitter
        result._raw_mask = pred_mask

        confs = []
        if result.pupil.detected:
            confs.append(result.pupil.confidence)
        if result.limbus.detected:
            confs.append(result.limbus.confidence)

        if confs:
            result.overall_confidence = float(np.mean(confs))
        result.overall_quality = assign_quality_grade(result.overall_confidence)

        result.metadata.processing_time_ms = (time.time() - t0) * 1000.0
        return result

    # ================================================================
    # Preprocessing
    # ================================================================

    def _preprocess(
        self, image_bgr: np.ndarray, frame_number: int = -1
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """Preprocess image for model inference.

        Returns the normalised tensor AND a grayscale copy at model
        resolution for sub-pixel contour refinement.

        Requires PyTorch — only called when ``_HAS_TORCH`` is True.
        """
        # Try to obtain ring diagnostics to build an ROI for reflection removal
        roi_mask = None
        if self._ring_masker is not None:
            try:
                clean_bgr, marker_mask, ring_result = self._ring_masker.remove_with_diagnostics(image_bgr)
                if getattr(ring_result, "ring_centre", None) is not None:
                    cx, cy = int(round(ring_result.ring_centre[0])), int(round(ring_result.ring_centre[1]))
                    inner_r = int(round(ring_result.ring_inner_radius)) if getattr(ring_result, "ring_inner_radius", None) is not None else None
                    if inner_r is not None and inner_r > 4:
                        h, w = clean_bgr.shape[:2]
                        roi_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.circle(roi_mask, (cx, cy), inner_r, 255, -1)
            except Exception:
                clean_bgr, _ = self._ring_masker.remove(image_bgr)

        # Reflection removal with ROI when possible
        clean_bgr, _ = self._reflection_remover.remove(clean_bgr, roi_mask=roi_mask)

        # Lazy initialize red light filter if not already done
        if self._red_light_filter is None:
            self._red_light_filter = self._get_red_light_filter()

        if self._red_light_filter is not None:
            clean_bgr, _ = self._red_light_filter.apply(
                clean_bgr, frame_number=frame_number
            )

        rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )

        # Store grayscale at model resolution for sub-pixel refinement
        gray_resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        img_f = resized.astype(np.float32) / 255.0
        img_f = (img_f - _MEAN) / _STD
        tensor = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, gray_resized

    # ================================================================
    # Multi-scale inference (Phase 3)
    # ================================================================

    def _multiscale_segment(
        self,
        tensor_base: "torch.Tensor",
        gray_resized: np.ndarray,
    ) -> np.ndarray:
        """Run inference at multiple scales and average probabilities.

        For single-image mode: averages predictions at 448, 512, 640
        to reduce boundary noise.  Returns averaged probability map
        at the base resolution.

        Requires PyTorch — only called when ``_HAS_TORCH`` is True.
        """
        cfg = self.cfg
        enable_ms = getattr(getattr(cfg, "model", None), "enable_multiscale", True)
        ms_sizes = getattr(
            getattr(cfg, "model", None), "multiscale_sizes", [448, 512, 640]
        )

        # On CPU (e.g. low-end testing PCs without discrete GPU), running 3 full ResNet passes
        # adds 2,000-3,500ms latency. Auto-gate to single-scale on CPU unless explicitly forced.
        is_cpu = getattr(self.device, "type", str(self.device)) == "cpu"
        force_cpu_ms = getattr(getattr(cfg, "model", None), "force_multiscale_on_cpu", False)
        if is_cpu and not force_cpu_ms:
            enable_ms = False

        if not enable_ms or len(ms_sizes) <= 1:
            # Single-scale: just run the base tensor
            logits = self.model(tensor_base)
            probs = torch.softmax(logits, dim=1)
            return probs[0].cpu().numpy()

        base_size = self.input_size
        all_probs = []

        for size in ms_sizes:
            if size == base_size:
                logits = self.model(tensor_base)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs[0].cpu().numpy())
            else:
                # Resize the base tensor to this scale
                scaled = F.interpolate(
                    tensor_base,
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                )
                logits = self.model(scaled)
                probs = torch.softmax(logits, dim=1)
                # Resize probabilities back to base resolution
                probs_resized = F.interpolate(
                    probs,
                    size=(base_size, base_size),
                    mode="bilinear",
                    align_corners=False,
                )
                all_probs.append(probs_resized[0].cpu().numpy())

        # Average probability maps across scales
        avg_probs = np.mean(all_probs, axis=0).astype(np.float32)
        return avg_probs

    # ================================================================
    # Gradient-guided mask boundary refinement (Phase 5.1)
    # ================================================================

    @staticmethod
    def _refine_mask_boundary(
        binary_mask: np.ndarray,
        gray_image: np.ndarray,
        band_width: int = 3,
        gradient_threshold: float = 0.3,
    ) -> np.ndarray:
        """Refine mask boundaries using grayscale gradient information.

        Dilates and erodes the mask to create a boundary band,
        then within that band reclassifies pixels based on whether
        they sit on a strong gradient (boundary) or not.
        """
        if binary_mask.sum() == 0:
            return binary_mask

        # Compute normalised gradient magnitude
        grad_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_max = grad_mag.max()
        if grad_max > 0:
            grad_mag = grad_mag / grad_max

        # Create dilated and eroded versions
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1)
        )
        dilated = cv2.dilate(binary_mask, kernel, iterations=1)
        eroded = cv2.erode(binary_mask, kernel, iterations=1)

        # Boundary band = dilated - eroded
        band = (dilated > 127) & ~(eroded > 127)
        if not np.any(band):
            return binary_mask

        # In the boundary band, pixels with strong gradient get
        # assigned based on which side of the boundary they are on
        refined = binary_mask.copy()

        # Pixels in band on the dilated side but not original mask:
        # include them only if gradient is strong (real edge nearby)
        outer_band = band & ~(binary_mask > 127)
        inner_band = band & (binary_mask > 127)

        # Add outer pixels where gradient is strong
        strong_gradient = grad_mag > gradient_threshold
        refined[outer_band & strong_gradient] = 255

        # Remove inner pixels where gradient is weak (noise)
        weak_gradient = grad_mag < (gradient_threshold * 0.5)
        # Don't remove too aggressively — only at the very edge
        edge_inner = inner_band & weak_gradient
        refined[edge_inner] = 0

        return refined

    # ================================================================
    # Sub-pixel contour refinement (Phase 2)
    # ================================================================

    @staticmethod
    def _refine_contour_subpixel(
        gray_image: np.ndarray,
        contour: np.ndarray,
    ) -> np.ndarray:
        """Refine contour points to sub-pixel accuracy using gradient.

        Uses Scharr operator (more rotationally symmetric than Sobel)
        and searches along gradient normals with 0.25px steps for
        surgical-grade precision.  Applies quadratic interpolation
        around the gradient peak for true sub-pixel positioning.
        """
        h, w = gray_image.shape[:2]
        pts = contour.reshape(-1, 2).astype(np.float64)
        refined = pts.copy()

        # Scharr gradients (more accurate than Sobel)
        grad_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        for i in range(len(pts)):
            px, py = pts[i]
            ix, iy = int(round(px)), int(round(py))

            if not (1 <= ix < w - 1 and 1 <= iy < h - 1):
                continue

            gx = grad_x[iy, ix]
            gy = grad_y[iy, ix]
            g_len = math.sqrt(gx * gx + gy * gy)

            if g_len < 1e-6:
                continue

            nx, ny = gx / g_len, gy / g_len

            # Search along the normal with fine steps
            best_val = 0.0
            best_t = 0.0
            search_vals = []
            search_ts = []

            num_steps = int(_SUBPIXEL_SEARCH_RADIUS / _SUBPIXEL_STEP)
            for t_idx in range(-num_steps, num_steps + 1):
                t = t_idx * _SUBPIXEL_STEP
                sx = px + nx * t
                sy = py + ny * t

                six, siy = int(round(sx)), int(round(sy))
                if 0 <= siy < h and 0 <= six < w:
                    val = grad_mag[siy, six]
                    search_vals.append(val)
                    search_ts.append(t)
                    if val > best_val:
                        best_val = val
                        best_t = t

            # Quadratic interpolation around the peak for true sub-pixel
            if len(search_vals) >= 3 and best_val > 0:
                peak_idx = search_ts.index(best_t)
                if 0 < peak_idx < len(search_vals) - 1:
                    v_prev = search_vals[peak_idx - 1]
                    v_peak = search_vals[peak_idx]
                    v_next = search_vals[peak_idx + 1]
                    denom = 2.0 * (2.0 * v_peak - v_prev - v_next)
                    if abs(denom) > 1e-9:
                        delta = (v_prev - v_next) / denom
                        best_t += delta * _SUBPIXEL_STEP

            refined[i, 0] = px + nx * best_t
            refined[i, 1] = py + ny * best_t

        return refined.reshape(-1, 1, 2)

    # ================================================================
    # Edge alignment score (Phase 7)
    # ================================================================

    @staticmethod
    def _compute_edge_alignment(
        gray_image: np.ndarray,
        contour: np.ndarray,
        gradient_threshold_percentile: float = 70.0,
    ) -> float:
        """Compute fraction of contour points sitting on strong edges.

        A high alignment score means the fitted contour matches real
        anatomical boundaries, boosting confidence.  Returns 0-1.
        """
        h, w = gray_image.shape[:2]
        pts = contour.reshape(-1, 2)

        if len(pts) < 5:
            return 0.0

        grad_x = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Dynamic threshold based on image gradient distribution
        threshold = np.percentile(grad_mag, gradient_threshold_percentile)
        if threshold < 1.0:
            return 0.5  # Very low gradient image — neutral score

        aligned = 0
        total = 0
        for px, py in pts:
            ix, iy = int(round(px)), int(round(py))
            if 0 <= iy < h and 0 <= ix < w:
                total += 1
                if grad_mag[iy, ix] >= threshold:
                    aligned += 1

        if total == 0:
            return 0.0

        return float(aligned) / float(total)

    # ================================================================
    # Mask post-processing
    # ================================================================

    @staticmethod
    def _postprocess_mask(
        pred_mask: np.ndarray,
        class_id: int,
        min_area: int = 200,
        structure: str = "pupil",
    ) -> np.ndarray:
        binary = ((pred_mask == class_id) * 255).astype(np.uint8)

        # Phase 5: Adaptive morphological kernel per structure
        if structure == "pupil":
            # Smaller kernel for pupil — prevents over-smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        else:
            # Larger kernel for limbus — needs more smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if n_labels <= 1:
            return np.zeros_like(binary)

        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(areas[largest_label - 1])

        if largest_area < min_area:
            return np.zeros_like(binary)

        result = np.zeros_like(binary)
        result[labels == largest_label] = 255
        return result

    # ================================================================
    # Structure detection
    # ================================================================

    def _detect_structure(
        self,
        binary_mask: np.ndarray,
        confidence: float,
        scale_x: float,
        scale_y: float,
        structure: str = "pupil",
        gray_model: Optional[np.ndarray] = None,
    ) -> PupilDetection:
        detection = PupilDetection()
        detection.method = DetectionMethod.ML

        if binary_mask.sum() == 0:
            return detection

        total_pixels = self.input_size * self.input_size
        mask_pixels = np.count_nonzero(binary_mask > 127)
        mask_frac = mask_pixels / total_pixels

        if mask_frac < _MIN_MASK_FRAC:
            self.logger.debug(
                "%s mask too small: %.4f%% of image",
                structure,
                mask_frac * 100,
            )
            return detection

        if mask_frac > _MAX_MASK_FRAC:
            self.logger.debug(
                "%s mask covers %.1f%% of image -- penalising",
                structure,
                mask_frac * 100,
            )
            confidence *= 0.3

        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return detection

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < max(5, self.cfg.detection.min_contour_points):
            return detection

        # Phase 2: Sub-pixel contour refinement using gradient
        if gray_model is not None:
            contour = self._refine_contour_subpixel(gray_model, contour)

        if _touches_border(contour, self.input_size, _BORDER_MARGIN):
            confidence *= 0.6
            self.logger.debug("%s contour touches image border", structure)

        fit = self.fitter.fit(contour, prefer_ellipse=True, config=self.cfg)
        if not fit.valid:
            return detection

        model_r = fit.radius

        if structure == "pupil":
            min_r = self.input_size * _PUPIL_MIN_FRAC
            max_r = self.input_size * _PUPIL_MAX_FRAC
        else:
            min_r = self.input_size * _LIMBUS_MIN_FRAC
            max_r = self.input_size * _LIMBUS_MAX_FRAC

        if model_r < min_r:
            self.logger.debug(
                "%s radius %.1f too small at model res (min=%.1f)",
                structure,
                model_r,
                min_r,
            )
            return detection

        if model_r > max_r:
            confidence *= 0.5
            self.logger.debug(
                "%s radius %.1f exceeds soft max (max=%.1f)",
                structure,
                model_r,
                max_r,
            )

        aspect = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0
        if aspect < 0.35:
            confidence *= 0.4
            self.logger.debug("%s very elongated: aspect=%.2f", structure, aspect)
        elif aspect < 0.50:
            confidence *= 0.7

        ellipse = self._scale_fit(fit, scale_x, scale_y)

        # Phase 7: Edge alignment score boosts/penalises confidence
        edge_alignment = 0.5
        if gray_model is not None:
            edge_alignment = self._compute_edge_alignment(gray_model, contour)

        combined_conf = (
            0.35 * confidence
            + 0.30 * fit.fit_quality_score
            + 0.20 * min(1.0, aspect / 0.7)
            + 0.15 * edge_alignment
        )
        combined_conf = float(np.clip(combined_conf, 0.0, 1.0))

        detection.detected = True
        detection.ellipse = ellipse
        detection.confidence = combined_conf
        detection.quality = assign_quality_grade(combined_conf)
        detection.contour_points = contour

        return detection

    # ================================================================
    # Limbus from iris
    # ================================================================

    def _detect_limbus_from_iris(
        self,
        iris_mask: np.ndarray,
        pupil_mask: np.ndarray,
        confidence: float,
        scale_x: float,
        scale_y: float,
        gray_model: Optional[np.ndarray] = None,
        pupil_hint: Optional[PupilDetection] = None,
    ) -> LimbusDetection:
        detection = LimbusDetection()
        detection.method = DetectionMethod.ML

        iris_pixels = np.count_nonzero(iris_mask > 127)
        if iris_pixels == 0:
            self.logger.debug("Empty iris mask -- no limbus detection")
            return detection

        combined = np.maximum(iris_mask, pupil_mask)

        total_pixels = self.input_size * self.input_size
        mask_frac = np.count_nonzero(combined > 127) / total_pixels

        if mask_frac < _MIN_MASK_FRAC * 3:
            self.logger.debug(
                "Combined iris+pupil mask too small: %.2f%%",
                mask_frac * 100,
            )
            return detection

        if mask_frac > _MAX_MASK_FRAC:
            confidence *= 0.4
            self.logger.debug(
                "Combined mask covers %.1f%% of image",
                mask_frac * 100,
            )

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return detection

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < max(5, self.cfg.detection.min_contour_points):
            return detection

        # --- Pupil-Guided Limbus Contour Filtering ---
        if pupil_hint is not None and pupil_hint.detected and pupil_hint.ellipse is not None:
            # We need the pupil center in the model's unscaled coordinate space
            model_cx = pupil_hint.ellipse.center_x / scale_x
            model_cy = pupil_hint.ellipse.center_y / scale_y

            pts = contour.reshape(-1, 2).astype(np.float64)
            dx = pts[:, 0] - model_cx
            dy = pts[:, 1] - model_cy
            dists = np.sqrt(dx**2 + dy**2)

            # The limbus radius should be relatively constant around the pupil
            r_med = np.median(dists)

            # Normal limbus radius is very constant. Allow +/- 15% variation.
            keep = (dists >= r_med * 0.85) & (dists <= r_med * 1.15)
            kept_pts = pts[keep]

            if len(kept_pts) >= max(5, self.cfg.detection.min_contour_points) and len(kept_pts) > len(pts) * 0.25:
                contour = kept_pts.reshape(-1, 1, 2).astype(np.int32)
                self.logger.debug(
                    "Filtered limbus contour using pupil anchor: kept %d/%d points",
                    len(kept_pts), len(pts)
                )

        border_touching = _touches_border(contour, self.input_size, _BORDER_MARGIN)
        if border_touching:
            confidence *= 0.5
            self.logger.debug("Limbus contour touches image border")

            interior_pts = _remove_border_points(
                contour, self.input_size, margin=_BORDER_MARGIN + 2
            )
            if len(interior_pts) >= 20:
                contour = interior_pts

        # Phase 2: Sub-pixel contour refinement
        if gray_model is not None:
            contour = self._refine_contour_subpixel(gray_model, contour)

        fit = self.fitter.fit(contour, prefer_ellipse=True, config=self.cfg)
        if not fit.valid:
            return detection

        model_r = fit.radius
        min_r = self.input_size * _LIMBUS_MIN_FRAC
        max_r = self.input_size * _LIMBUS_MAX_FRAC

        if model_r < min_r:
            self.logger.debug("Limbus radius %.1f too small at model res", model_r)
            return detection

        if model_r > max_r:
            confidence *= 0.5

        aspect = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0
        if aspect < 0.35:
            confidence *= 0.4
        elif aspect < 0.50:
            confidence *= 0.7

        ellipse = self._scale_fit(fit, scale_x, scale_y)

        # Phase 7: Edge alignment score
        edge_alignment = 0.5
        if gray_model is not None:
            edge_alignment = self._compute_edge_alignment(gray_model, contour)

        combined_conf = (
            0.35 * confidence
            + 0.30 * fit.fit_quality_score
            + 0.20 * min(1.0, aspect / 0.7)
            + 0.15 * edge_alignment
        )
        combined_conf = float(np.clip(combined_conf, 0.0, 1.0))

        detection.detected = True
        detection.ellipse = ellipse
        detection.confidence = combined_conf
        detection.quality = assign_quality_grade(combined_conf)
        detection.contour_points = contour

        return detection

    # ================================================================
    # Scaling helper
    # ================================================================

    def _scale_fit(
        self, fit: FitResult, scale_x: float, scale_y: float
    ) -> EllipseParams:
        scale_major = max(scale_x, scale_y)
        scale_minor = min(scale_x, scale_y)

        circ = fit.semi_minor / fit.semi_major if fit.semi_major > 0 else 0.0

        return EllipseParams(
            center_x=fit.center_x * scale_x,
            center_y=fit.center_y * scale_y,
            semi_major=fit.semi_major * scale_major,
            semi_minor=fit.semi_minor * scale_minor,
            angle_deg=fit.angle_deg,
            uncertainty_center_x=fit.uncertainty_center[0] * scale_x,
            uncertainty_center_y=fit.uncertainty_center[1] * scale_y,
            uncertainty_semi_major=fit.uncertainty_radius * scale_major,
            uncertainty_semi_minor=fit.uncertainty_radius * scale_minor,
            fit_quality=fit.fit_quality_score,
            fit_rms_residual=fit.rms_residual * scale_major,
            num_contour_points=fit.num_points,
            eccentricity=fit.eccentricity,
            circularity=circ,
        )

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def _class_confidence(
        probs: np.ndarray,
        pred_mask: np.ndarray,
        class_id: int,
    ) -> float:
        region = pred_mask == class_id
        if region.sum() == 0:
            return 0.0
        return float(np.mean(probs[class_id][region]))

    def get_raw_mask(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self._model_loaded or self.model is None or not _HAS_TORCH:
            h, w = self.input_size, self.input_size
            return (
                np.zeros((h, w), dtype=np.uint8),
                np.zeros((3, h, w), dtype=np.float32),
            )

        image_bgr = self._ensure_bgr(image_bgr)
        tensor, _gray = self._preprocess(image_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs[0].cpu().numpy()
        pred_mask = probs_np.argmax(axis=0).astype(np.uint8)
        return pred_mask, probs_np


# ======================================================================
# Module-level helpers
# ======================================================================


def _touches_border(
    contour: np.ndarray,
    image_size: int,
    margin: int = 3,
) -> bool:
    pts = contour.reshape(-1, 2)
    if len(pts) == 0:
        return False

    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    return (
        x_min <= margin
        or y_min <= margin
        or x_max >= image_size - 1 - margin
        or y_max >= image_size - 1 - margin
    )


def _remove_border_points(
    contour: np.ndarray,
    image_size: int,
    margin: int = 5,
) -> np.ndarray:
    pts = contour.reshape(-1, 2)
    keep = (
        (pts[:, 0] > margin)
        & (pts[:, 1] > margin)
        & (pts[:, 0] < image_size - 1 - margin)
        & (pts[:, 1] < image_size - 1 - margin)
    )
    kept = pts[keep]
    if len(kept) == 0:
        return contour
    return kept.reshape(-1, 1, 2)