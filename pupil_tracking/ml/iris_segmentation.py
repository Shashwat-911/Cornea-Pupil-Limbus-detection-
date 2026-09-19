"""Lightweight U-Net for iris texture segmentation.

Segmentation of usable iris texture regions from eye images. Binary output:
usable_iris (1) vs non_iris (0). Uses a MobileNet encoder (prefers
``mobilenet_v3_small``, falls back to ``mobilenet_v2`` depending on the
installed segmentation_models_pytorch) for CPU deployment via ONNX Runtime.

The model crops to the iris ROI annulus (between pupil and limbus) and
produces a per-pixel mask of usable iris texture, handling reflections,
eyelid occlusion, and saturation.

Usage
-----
>>> from pupil_tracking.ml.iris_segmentation import IrisSegmentationNet
>>> model = IrisSegmentationNet.load("models/iris_segmentation.onnx")
>>> mask = model.predict(image, roi)  # bool (H, W) mask
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = False

try:
    import segmentation_models_pytorch as smp

    _HAS_SMP = True
except ImportError:
    smp = None
    _HAS_SMP = False

logger = logging.getLogger(__name__)

# ImageNet normalization for RGB input
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default model search paths
_MODEL_SEARCH_PATHS = [
    "models/iris_segmentation_quantized.onnx",
    "models/iris_segmentation.onnx",
    "models/iris_segmentation.pth",
]


_ENCODER_PREFERRED = "mobilenet_v3_small"
_ENCODER_FALLBACK = "mobilenet_v2"


def _resolve_encoder_name() -> str:
    """Return a MobileNet encoder supported by the installed smp version.

    Prefers ``mobilenet_v3_small``; falls back to ``mobilenet_v2`` on older
    segmentation_models_pytorch builds that do not ship the v3 encoder.
    """
    try:
        from segmentation_models_pytorch.encoders import encoders

        if _ENCODER_PREFERRED in encoders:
            return _ENCODER_PREFERRED
    except Exception:
        pass
    return _ENCODER_FALLBACK


def _find_model_path() -> Optional[str]:
    """Locate the iris segmentation model file."""
    for p in _MODEL_SEARCH_PATHS:
        if Path(p).is_file():
            return str(Path(p).resolve())
    return None


def _crop_to_roi(
    image: np.ndarray, roi, target_size: int = 256
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crop image to ROI bounding box and resize to target_size.

    Returns (cropped_resized, (x_off, y_off, orig_w, orig_h)) so the
    output mask can be mapped back to original coordinates.
    """
    h, w = image.shape[:2]
    cx = int(round(roi.center_x))
    cy = int(round(roi.center_y))
    r = int(round(roi.limbus_radius_px * 1.1))

    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(w, cx + r)
    y1 = min(h, cy + r)

    cropped = image[y0:y1, x0:x1]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized, (x0, y0, w, h)


def _preprocess(
    cropped_bgr: np.ndarray,
) -> np.ndarray:
    """BGR -> RGB, ImageNet normalize, HWC -> NCHW float32 tensor shape (1,3,H,W)."""
    rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]
    return tensor.astype(np.float32)


def _postprocess(
    logits: np.ndarray,
    orig_shape: Tuple[int, int, int, int],
    roi,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert model output to binary mask in original image coordinates.

    Parameters
    ----------
    logits : np.ndarray (1, 2, H, W) or (2, H, W)
        Raw model output (class 0 = non_iris, class 1 = usable_iris).
    orig_shape : tuple
        (x_off, y_off, orig_w, orig_h) from _crop_to_roi.
    roi : IrisROI
        Iris region of interest for annulus masking.
    threshold : float
        Decision threshold for usable_iris class.

    Returns
    -------
    np.ndarray (orig_h, orig_w) bool
        Binary mask in original image coordinates.
    """
    if logits.ndim == 4:
        logits = logits[0]

    probs = _softmax(logits)
    usable_prob = probs[1]

    binary = (usable_prob >= threshold).astype(np.uint8)

    x_off, y_off, orig_w, orig_h = orig_shape
    mask_full = np.zeros((orig_h, orig_w), dtype=np.uint8)

    crop_h, crop_w = binary.shape
    mask_full[y_off : y_off + crop_h, x_off : x_off + crop_w] = binary

    mask_bool = mask_full.astype(bool)

    if roi.valid:
        h, w = mask_full.shape
        Y, X = np.mgrid[0:h, 0:w]
        dx = (X - roi.center_x) / max(roi.limbus_radius_px, 1)
        dy = (Y - roi.center_y) / max(roi.limbus_radius_px, 1)
        dist = np.sqrt(dx * dx + dy * dy)
        annulus = (dist >= roi.pupil_radius_px / roi.limbus_radius_px * 0.85) & (
            dist <= 1.05
        )
        mask_bool = mask_bool & annulus

    return mask_bool


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along axis=0."""
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / (exp.sum(axis=0, keepdims=True) + 1e-8)


class IrisSegmentationNet:
    """Inference wrapper for iris texture segmentation.

    Works with both PyTorch and ONNX Runtime backends. ONNX is preferred
    for production (zero PyTorch dependency).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: int = 256,
        threshold: float = 0.5,
        device: str = "auto",
    ):
        self.input_size = input_size
        self.threshold = threshold
        self._backend = None
        self._model = None
        self._session = None

        if model_path is None:
            model_path = _find_model_path()

        if model_path is None:
            logger.warning("No iris segmentation model found; CNN mask disabled")
            return

        path = str(model_path)
        if path.endswith(".onnx"):
            self._load_onnx(path)
        elif path.endswith(".pth"):
            self._load_pytorch(path, device)
        else:
            logger.warning("Unknown model format: %s", path)

    def _load_onnx(self, path: str) -> None:
        """Load ONNX Runtime session."""
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            preferred = []
            for p in ["CUDAExecutionProvider", "DirectMLExecutionProvider", "CPUExecutionProvider"]:
                if p in providers:
                    preferred.append(p)
            if not preferred:
                preferred = ["CPUExecutionProvider"]

            self._session = ort.InferenceSession(path, providers=preferred)
            self._backend = "onnx"
            logger.info("Iris segmentation ONNX loaded: %s", path)
        except ImportError:
            logger.warning("onnxruntime not available; cannot load %s", path)
        except Exception as e:
            logger.warning("Cannot load ONNX model %s: %s", path, e)

    def _load_pytorch(self, path: str, device: str) -> None:
        """Load PyTorch checkpoint."""
        if not _HAS_TORCH:
            logger.warning("PyTorch not available; cannot load %s", path)
            return

        if not _HAS_SMP:
            logger.warning("segmentation_models_pytorch not available")
            return

        try:
            model = smp.Unet(
                encoder_name=_resolve_encoder_name(),
                encoder_weights=None,
                in_channels=3,
                classes=2,
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(device)
            self._backend = "pytorch"
            logger.info("Iris segmentation PyTorch loaded: %s", path)
        except Exception as e:
            logger.warning("Cannot load PyTorch model %s: %s", path, e)

    @property
    def available(self) -> bool:
        return self._backend is not None

    def predict(
        self,
        image: np.ndarray,
        roi,
    ) -> np.ndarray:
        """Segment usable iris texture.

        Parameters
        ----------
        image : np.ndarray
            BGR eye image (H, W, 3).
        roi : IrisROI
            Iris region of interest.

        Returns
        -------
        np.ndarray (H, W) bool
            Binary mask of usable iris texture in original coordinates.
        """
        if not self.available:
            return np.ones(image.shape[:2], dtype=bool)

        cropped, meta = _crop_to_roi(image, roi, self.input_size)
        tensor = _preprocess(cropped)

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            logits = self._session.run(None, {input_name: tensor})[0]
        elif self._backend == "pytorch":
            with torch.no_grad():
                t = torch.from_numpy(tensor).to(next(self._model.parameters()).device)
                logits = self._model(t).cpu().numpy()
        else:
            return np.ones(image.shape[:2], dtype=bool)

        return _postprocess(logits, meta, roi, self.threshold)

    def predict_batch(
        self,
        images: list,
        rois: list,
    ) -> list:
        """Batch prediction for video processing."""
        return [self.predict(img, roi) for img, roi in zip(images, rois)]


def create_model(
    pretrained: bool = True,
    device: str = "auto",
) -> "nn.Module":
    """Create a new IrisSegmentationNet PyTorch model.

    Parameters
    ----------
    pretrained : bool
        Use ImageNet-pretrained encoder weights.
    device : str
        Target device.

    Returns
    -------
    torch.nn.Module
        U-Net with MobileNetV3-Small encoder, 2-class output.
    """
    if not _HAS_SMP:
        raise ImportError("segmentation_models_pytorch required to create model")

    model = smp.Unet(
        encoder_name=_resolve_encoder_name(),
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=2,
    )

    if device != "cpu" and _HAS_TORCH:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    return model
