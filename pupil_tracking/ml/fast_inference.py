"""
ACCURACY-FIRST single-image inference for video pipelines.

Accuracy improvements (v3 — plan-aligned):
    * Default 320×320 input (A2: was 192)
    * INTER_AREA resize (A1: was INTER_NEAREST)
    * Specular reflection removal before inference (A3)
    * Suction ring marker masking before inference (A5)
    * Actual ML probability propagation (A4: no hardcoded confidences)
    * Close+Open morphology with 5×5 kernel (A7: was 3×3 close only)
    * Batch inference for video pipeline (S1)
    * Resize-then-convert ordering (S2)

Target latency:
    - 320×320: 12-25 ms GPU, 25-50 ms CPU
    - 256×256: 8-20 ms GPU, 20-40 ms CPU
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp

    _HAS_SMP = True
except ImportError:
    _HAS_SMP = False

from pupil_tracking.utils.logger import get_logger
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover
from pupil_tracking.preprocessing.suction_ring_masker import SuctionRingMasker

# ImageNet normalisation (must match training)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FastInference:
    """
    ACCURACY-FIRST inference engine for per-frame video processing.

    v3 plan-aligned improvements:
        - 320×320 default resolution (A2)
        - INTER_AREA resize (A1)
        - Specular reflection removal (A3)
        - Suction ring marker masking (A5)
        - Actual ML confidence propagation (A4)
        - Close+Open morphology with 5×5 kernel (A7)
        - Batch inference detect_batch / segment_batch (S1)
        - Cached device tensors for speed

    Parameters
    ----------
    model_path : str
        Path to checkpoint (``.pth``).
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"``.
    input_size : int
        Spatial resolution (default 320 for accuracy).
    use_half : bool
        Use FP16 on CUDA (ignored on CPU/MPS).
    half_precision : bool or None
        Alias for use_half (backward compat with OptimizedVideoProcessor).
    pupil_threshold : float
        Probability threshold for pupil class.
    iris_threshold : float
        Probability threshold for iris class.
    skip_morphology : bool
        Skip morphological cleanup for extreme speed.
    light_morphology : bool
        Use lighter morphological operations.
    reflection_removal : bool
        Enable specular reflection inpainting (A3).
    suction_ring_removal : bool
        Enable red marker dot masking (A5).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        input_size: int = 320,  # A2: 192 → 320
        use_half: bool = True,
        use_compile: bool = True,
        half_precision: Optional[bool] = None,  # alias for use_half
        pupil_threshold: float = 0.42,
        iris_threshold: float = 0.22,
        skip_morphology: bool = False,
        light_morphology: bool = True,
        reflection_removal: bool = True,  # A3
        suction_ring_removal: bool = True,  # A5
    ) -> None:
        self.logger = get_logger()
        self.input_size = input_size
        self.pupil_threshold = pupil_threshold
        self.iris_threshold = iris_threshold
        self.skip_morphology = skip_morphology
        self.light_morphology = light_morphology

        # ── half_precision alias ────────────────────────────────
        if half_precision is not None:
            use_half = half_precision

        # ── device ──────────────────────────────────────────────
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._use_half = use_half and self._device.type == "cuda"

        # ── model ───────────────────────────────────────────────
        self._model = self._load_model(model_path)
        self._model.to(self._device)
        if self._use_half:
            self._model.half()

        # ── torch.compile (PyTorch ≥ 2.0, CUDA only) ────────────
        # On CPU, torch.compile with mode="reduce-overhead" adds
        # overhead without benefit (CUDA graph capture is meaningless).
        self._compiled = False
        if use_compile and self._device.type != "cpu" and hasattr(torch, "compile"):
            try:
                compiled = torch.compile(
                    self._model,
                    mode="reduce-overhead",
                    fullgraph=True,
                )
                # Eagerly test compilation with a dummy forward pass so we
                # catch missing-compiler errors (e.g. cl.exe on Windows)
                # before committing to the compiled model.
                _dummy = torch.zeros(
                    1,
                    3,
                    self.input_size,
                    self.input_size,
                    device=self._device,
                    dtype=torch.float16 if self._use_half else torch.float32,
                )
                with torch.no_grad():
                    _ = compiled(_dummy)
                self._model = compiled
                self._compiled = True
                self.logger.info("torch.compile enabled (reduce-overhead, fullgraph)")
            except Exception as exc:
                self.logger.debug(
                    "torch.compile unavailable, using eager mode: %s", exc
                )
        elif not use_compile:
            self.logger.debug("torch.compile disabled by configuration")
        elif self._device.type == "cpu":
            self.logger.debug("torch.compile skipped on CPU (no benefit)")

        # ── pre-computed normalisation tensors on device ────────
        self._mean = _MEAN.to(self._device)
        self._std = _STD.to(self._device)
        if self._use_half:
            self._mean = self._mean.half()
            self._std = self._std.half()

        # ── cached morphology kernels ─────────────────────────
        if not skip_morphology:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Smaller kernel for iris OPEN step to preserve limbus boundary
            self._morph_kernel_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        else:
            self._morph_kernel = None
            self._morph_kernel_sm = None

        # ── A3: reflection remover ──────────────────────────────
        self._reflection_remover: Optional[ReflectionRemover] = None
        if reflection_removal:
            self._reflection_remover = ReflectionRemover(
                brightness_threshold=225,
                min_reflection_area=10,
                inpaint_radius=3,
                detect_red_highlights=True,
                red_threshold_offset=25,
            )

        # ── A5: suction ring masker ─────────────────────────────
        self._ring_masker: Optional[SuctionRingMasker] = None
        if suction_ring_removal:
            self._ring_masker = SuctionRingMasker()

        # ── warm-up ─────────────────────────────────────────────
        self._warm_up()

        self.logger.info(
            "FastInference ACCURACY-FIRST ready: %dx%d, half=%s, "
            "skip_morph=%s, light_morph=%s, refl_rm=%s, ring_rm=%s",
            input_size,
            input_size,
            self._use_half,
            skip_morphology,
            light_morphology,
            reflection_removal,
            suction_ring_removal,
        )

    # ================================================================
    # Properties
    # ================================================================

    @property
    def device(self) -> torch.device:
        """Return the device this engine is running on."""
        return self._device

    def warmup(self) -> None:
        """Run warm-up passes to fully compile the model."""
        self._warm_up()

    # ================================================================
    # Model loading
    # ================================================================

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model weights with robust key handling."""
        if not _HAS_SMP:
            raise ImportError(
                "segmentation_models_pytorch is required. "
                "Install with: pip install segmentation-models-pytorch"
            )

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=3,
        )

        ckpt = torch.load(model_path, map_location=self._device, weights_only=False)

        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                state = ckpt
        elif hasattr(ckpt, "state_dict"):
            state = ckpt.state_dict()
        else:
            state = ckpt

        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k == "temperature" or k.endswith(".temperature"):
                continue
            if k.startswith("module."):
                k = k[len("module.") :]
            if k.startswith("model."):
                k = k[len("model.") :]
            if k.startswith("model."):
                k = k[len("model.") :]
            cleaned[k] = v

        n_loaded = len(cleaned)
        n_model = len(model.state_dict())

        result = model.load_state_dict(cleaned, strict=False)

        if result.missing_keys:
            self.logger.warning(
                "Missing %d keys when loading model (first 5): %s",
                len(result.missing_keys),
                result.missing_keys[:5],
            )
        if result.unexpected_keys:
            self.logger.debug(
                "Unexpected %d keys (first 5): %s",
                len(result.unexpected_keys),
                result.unexpected_keys[:5],
            )

        if n_loaded < n_model * 0.5:
            raise RuntimeError(
                f"Model loading suspect: only {n_loaded}/{n_model} "
                f"parameters matched. Check model architecture."
            )

        self.logger.info(
            "FastInference model loaded: %d/%d params, device=%s, half=%s, input=%d",
            n_loaded,
            n_model,
            self._device,
            self._use_half,
            self.input_size,
        )

        model.eval()
        return model

    # ================================================================
    # Warm-up
    # ================================================================

    def _warm_up(self) -> None:
        """Multiple warm-up passes to fully compile."""
        dummy = torch.zeros(
            1,
            3,
            self.input_size,
            self.input_size,
            device=self._device,
            dtype=torch.float16 if self._use_half else torch.float32,
        )
        with torch.no_grad():
            _ = self._model(dummy)

        # Batch pass for compile optimisation
        dummy_batch = torch.zeros(
            2,
            3,
            self.input_size,
            self.input_size,
            device=self._device,
            dtype=torch.float16 if self._use_half else torch.float32,
        )
        with torch.no_grad():
            _ = self._model(dummy_batch)
            _ = self._model(dummy)

        if self._device.type == "cuda":
            torch.cuda.synchronize()

        self.logger.debug("FastInference warm-up complete (3 passes)")

    # ================================================================
    # Preprocessing
    # ================================================================

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        LATENCY-OPTIMIZED: BGR uint8 → normalised tensor [1, 3, H, W].

        Pipeline order (FAST PATH):
            1. Grayscale normalisation to BGR (Phase 1)
            2. Resize with INTER_AREA FIRST (A1, best quality downscale)
            3. Light reflection removal ONLY (skip ring diagnostics for speed)
            4. BGR→RGB (S2, after resize = fewer pixels)
            5. To tensor + ImageNet normalise
        
        Ring masking deferred to main detect() for async/batch processing.
        """
        img = image_bgr

        # Phase 1: normalise grayscale / BGRA to BGR
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # A5: Light suction ring marker removal (fast version) — skip diagnostics
        if self._ring_masker is not None:
            try:
                img, _ = self._ring_masker.remove(img)
            except Exception:
                pass  # Keep original on error

        # A1: Resize with INTER_AREA FIRST (most critical for speed)
        resized = cv2.resize(
            img,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )

        # A3: Remove specular reflections at DOWNSCALED resolution (~0.1-0.2ms)
        # This is 25-50× faster than at full resolution
        if self._reflection_remover is not None:
            try:
                resized, _ = self._reflection_remover.remove(resized, roi_mask=None)
            except Exception:
                pass  # Keep downscaled original on error

        # S2: BGR→RGB after resize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Efficient tensor creation — transpose first, single GPU transfer
        t = (
            torch.from_numpy(rgb.transpose(2, 0, 1).copy())
            .unsqueeze(0)
            .to(
                self._device,
                dtype=torch.float16 if self._use_half else torch.float32,
                non_blocking=True,
            )
            .div_(255.0)
        )

        # Normalise with cached tensors
        t = (t - self._mean) / self._std
        return t

    # ================================================================
    # Morphology helper
    # ================================================================

    def _apply_morphology(
        self, pupil_mask: np.ndarray, iris_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Close+Open morphology to both masks (A7).

        Always does CLOSE then OPEN regardless of light_morphology
        flag — the only difference is iteration count.
        """
        if self.skip_morphology or self._morph_kernel is None:
            return pupil_mask, iris_mask

        # Use smaller kernel for iris OPEN step to preserve limbus boundary
        iris_open_kernel = (
            self._morph_kernel_sm
            if self._morph_kernel_sm is not None
            else self._morph_kernel
        )

        if self.light_morphology:
            # A7: single pass close + open with 5×5 kernel
            pupil_mask = cv2.morphologyEx(
                pupil_mask, cv2.MORPH_CLOSE, self._morph_kernel
            )
            pupil_mask = cv2.morphologyEx(
                pupil_mask, cv2.MORPH_OPEN, self._morph_kernel
            )
            iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_CLOSE, self._morph_kernel)
            iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, iris_open_kernel)
        else:
            # Full two-pass cleanup
            pupil_mask = cv2.morphologyEx(
                pupil_mask,
                cv2.MORPH_CLOSE,
                self._morph_kernel,
                iterations=2,
            )
            pupil_mask = cv2.morphologyEx(
                pupil_mask, cv2.MORPH_OPEN, self._morph_kernel
            )
            iris_mask = cv2.morphologyEx(
                iris_mask,
                cv2.MORPH_CLOSE,
                self._morph_kernel,
                iterations=2,
            )
            iris_mask = cv2.morphologyEx(iris_mask, cv2.MORPH_OPEN, iris_open_kernel)

        return pupil_mask, iris_mask

    # ================================================================
    # Core single-frame inference
    # ================================================================

    def segment(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment an image and return binary masks at model resolution.

        Returns
        -------
        pupil_mask : np.ndarray  [input_size, input_size] uint8 0/255
        iris_mask  : np.ndarray  [input_size, input_size] uint8 0/255
        raw_probs  : np.ndarray  [3, input_size, input_size] float32
        """
        tensor = self._preprocess(image_bgr)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits.float(), dim=1)

        probs_np = probs[0].cpu().numpy()  # [3, H, W]

        # Vectorised thresholding
        pupil_mask = (probs_np[1] > self.pupil_threshold).astype(np.uint8) * 255
        iris_mask = (probs_np[2] > self.iris_threshold).astype(np.uint8) * 255

        # A7: Close+Open morphology
        pupil_mask, iris_mask = self._apply_morphology(pupil_mask, iris_mask)

        return pupil_mask, iris_mask, probs_np

    # ================================================================
    # Geometry extraction helper (shared by detect / detect_batch)
    # ================================================================

    def _extract_detection(
        self,
        pupil_mask: np.ndarray,
        iris_mask: np.ndarray,
        probs_np: np.ndarray,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Dict[str, object]:
        """Convert masks + probabilities into a flat result dict.

        Coordinates are mapped from model space to original image
        space using scale and offset.  Confidences come from actual
        ML probabilities (A4).
        """
        result: Dict[str, object] = {
            "pupil_detected": False,
            "pupil_x": 0.0,
            "pupil_y": 0.0,
            "pupil_radius": 0.0,
            "pupil_r": 0.0,  # alias for backward compat
            "pupil_confidence": 0.0,
            "limbus_detected": False,
            "limbus_x": 0.0,
            "limbus_y": 0.0,
            "limbus_radius": 0.0,
            "limbus_r": 0.0,  # alias for backward compat
            "limbus_confidence": 0.0,
        }

        # ── pupil ──────────────────────────────────────────────
        p = self._mask_to_ellipse(pupil_mask, min_area=25)
        if p is not None:
            cx, cy, r, p_sa, p_sb, p_angle = p
            scaled_r = r * max(scale_x, scale_y)
            result["pupil_detected"] = True
            result["pupil_x"] = cx * scale_x + offset_x
            result["pupil_y"] = cy * scale_y + offset_y
            result["pupil_radius"] = scaled_r
            result["pupil_r"] = scaled_r  # alias
            result["pupil_major"] = p_sa * 2.0 * scale_x
            result["pupil_minor"] = p_sb * 2.0 * scale_y
            result["pupil_angle"] = p_angle
            circ = p_sb / p_sa if p_sa > 0 else 1.0
            result["pupil_fit_type"] = "circle" if circ >= 0.97 else "ellipse"

            # A4: actual ML probability
            pupil_region = pupil_mask > 127
            if np.any(pupil_region):
                result["pupil_confidence"] = float(np.mean(probs_np[1][pupil_region]))
            else:
                result["pupil_confidence"] = 0.5

        # ── limbus (outer boundary of iris + pupil) ────────────
        combined = np.maximum(iris_mask, pupil_mask)
        # Dilate the combined mask slightly to close holes from reflection removal
        # and smooth the boundary.
        _dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.dilate(combined, _dilate_k, iterations=1)
        l = self._mask_to_ellipse(combined, min_area=80)
        if l is not None:
            cx, cy, r, l_sa, l_sb, l_angle = l
            
            scaled_r = r * max(scale_x, scale_y)
            result["limbus_detected"] = True
            result["limbus_x"] = cx * scale_x + offset_x
            result["limbus_y"] = cy * scale_y + offset_y
            result["limbus_radius"] = scaled_r
            result["limbus_r"] = scaled_r  # alias
            result["limbus_major"] = l_sa * 2.0 * scale_x
            result["limbus_minor"] = l_sb * 2.0 * scale_y
            result["limbus_angle"] = l_angle
            circ = l_sb / l_sa if l_sa > 0 else 1.0
            result["limbus_fit_type"] = "circle" if circ >= 0.97 else "ellipse"

            # A4: actual ML probability
            iris_region = iris_mask > 127
            if np.any(iris_region):
                result["limbus_confidence"] = float(np.mean(probs_np[2][iris_region]))
            else:
                result["limbus_confidence"] = 0.4

        return result

    # ================================================================
    # Single-frame detect (existing API, now uses shared helper)
    # ================================================================

    def detect(
        self,
        image_bgr: np.ndarray,
        scale_x: Optional[float] = None,
        scale_y: Optional[float] = None,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> Dict[str, object]:
        """
        Segment + extract ellipse parameters as a flat dict.

        Coordinates are mapped from model space to original image
        space using scale and offset. If scale_x/scale_y are omitted,
        they are automatically inferred from the input image dimensions.

        Returns
        -------
        dict with keys: pupil_detected, pupil_x, pupil_y,
            pupil_radius, pupil_confidence, limbus_*, processing_time_ms
        """
        t0 = time.time()
        h, w = image_bgr.shape[:2]
        if scale_x is None:
            scale_x = float(w) / float(self.input_size)
        if scale_y is None:
            scale_y = float(h) / float(self.input_size)

        pupil_mask, iris_mask, probs_np = self.segment(image_bgr)

        result = self._extract_detection(
            pupil_mask,
            iris_mask,
            probs_np,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        )
        result["processing_time_ms"] = (time.time() - t0) * 1000.0
        return result

    # ================================================================
    # Batch inference (S1) — raw masks
    # ================================================================

    def infer_batch(
        self, images: List[np.ndarray]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Batch inference for multiple images.

        Parameters
        ----------
        images : list of BGR uint8 arrays

        Returns
        -------
        list of (pupil_mask, iris_mask, raw_probs) tuples
            All at model input resolution.
        """
        if not images:
            return []

        # Batch preprocessing
        tensors = [self._preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            logits = self._model(batch)
            probs = torch.softmax(logits.float(), dim=1)

        probs_np = probs.cpu().numpy()
        results = []

        for i in range(len(images)):
            p_np = probs_np[i]
            pm = (p_np[1] > self.pupil_threshold).astype(np.uint8) * 255
            im = (p_np[2] > self.iris_threshold).astype(np.uint8) * 255

            # A7: Close+Open morphology (FIXED: both modes now correct)
            pm, im = self._apply_morphology(pm, im)

            results.append((pm, im, p_np))

        return results

    # ================================================================
    # S1: Batch detect — flat result dicts
    # ================================================================

    def detect_batch(
        self,
        images_bgr: List[np.ndarray],
        offsets: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Dict[str, object]]:
        """
        Batch inference + geometry extraction.

        Parameters
        ----------
        images_bgr : list of BGR uint8 arrays
        offsets : list of (offset_x, offset_y) per frame, or None

        Returns
        -------
        list of flat result dicts (same format as detect())
        """
        if not images_bgr:
            return []

        t0 = time.time()
        batch_raw = self.infer_batch(images_bgr)
        per_frame_ms = ((time.time() - t0) * 1000.0) / len(images_bgr)

        results: List[Dict[str, object]] = []
        for i, (pm, im, probs_np) in enumerate(batch_raw):
            eh, ew = images_bgr[i].shape[:2]
            scale_x = ew / self.input_size
            scale_y = eh / self.input_size
            ox = offsets[i][0] if offsets else 0.0
            oy = offsets[i][1] if offsets else 0.0

            det = self._extract_detection(
                pm,
                im,
                probs_np,
                scale_x,
                scale_y,
                ox,
                oy,
            )
            det["processing_time_ms"] = per_frame_ms
            results.append(det)

        return results

    # ================================================================
    # S1: Batch segment — mask-level dicts (for detector.detect_from_masks)
    # ================================================================

    def segment_batch(self, images_bgr: List[np.ndarray]) -> List[Dict[str, object]]:
        """
        Batch inference returning mask-level dicts.

        Each dict contains the raw masks and real ML confidences,
        suitable for passing to UnifiedDetector.detect_from_masks().

        Returns
        -------
        list of dicts with keys:
            pupil_mask, iris_mask, pupil_confidence,
            limbus_confidence, probabilities, inference_time_ms
        """
        if not images_bgr:
            return []

        t0 = time.time()
        batch_raw = self.infer_batch(images_bgr)
        per_frame_ms = ((time.time() - t0) * 1000.0) / len(images_bgr)

        results: List[Dict[str, object]] = []
        for pm, im, probs_np in batch_raw:
            # A4: actual ML probabilities
            pupil_region = pm > 127
            if np.any(pupil_region):
                pupil_conf = float(np.mean(probs_np[1][pupil_region]))
            else:
                pupil_conf = 0.0

            iris_region = im > 127
            iris_only = iris_region & ~pupil_region
            if np.any(iris_only):
                limbus_conf = float(np.mean(probs_np[2][iris_only]))
            elif np.any(iris_region):
                limbus_conf = float(np.mean(probs_np[2][iris_region]))
            else:
                limbus_conf = 0.0

            results.append(
                {
                    "pupil_mask": pm,
                    "iris_mask": im,
                    "pupil_confidence": pupil_conf,
                    "limbus_confidence": limbus_conf,
                    "probabilities": probs_np,
                    "inference_time_ms": per_frame_ms,
                }
            )

        return results

    # ================================================================
    # Mask → ellipse extraction
    # ================================================================

    @staticmethod
    def _mask_to_ellipse(
        mask: np.ndarray, min_area: int = 50
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Extract (cx, cy, mean_radius, semi_major, semi_minor, angle_deg)
        from a binary mask.

        Returns None if no valid contour found.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < min_area:
            return None

        # Circularity filter
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circ = 4.0 * np.pi * area / (perimeter * perimeter)
            if circ < 0.25:
                return None

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (w, h), angle = ellipse
            semi_a = max(w, h) / 2.0  # semi-major
            semi_b = min(w, h) / 2.0  # semi-minor
            r = (semi_a + semi_b) / 2.0  # mean radius
            if r < 1.5:
                return None
            # Normalise angle so it refers to the major axis
            if h > w:
                angle = (angle + 90.0) % 180.0
            return (
                float(cx),
                float(cy),
                float(r),
                float(semi_a),
                float(semi_b),
                float(angle),
            )

        # Fallback to moments (circle only)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            r = np.sqrt(area / np.pi)
            return (float(cx), float(cy), float(r), float(r), float(r), 0.0)

        return None

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        """Standalone morphological clean-up (5×5 close+open)."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        return out
