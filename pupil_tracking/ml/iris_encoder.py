"""CNN-based iris texture encoder for discriminative feature embeddings.

Replaces the classical 16-bin histogram descriptors with learned 128-d
embedding vectors. Designed for correspondence and cyclotorsion estimation.

The encoder takes a normalized polar iris strip (64x512 grayscale) and
produces an L2-normalized 128-d embedding. Pairs of same-iris images
produce similar embeddings; different irises produce distant embeddings.

Usage
-----
>>> from pupil_tracking.ml.iris_encoder import IrisEncoderNet
>>> encoder = IrisEncoderNet.load("models/iris_encoder.onnx")
>>> embedding = encoder.encode_patch(iris_patch)  # (128,) float32
>>> sim = encoder.similarity(emb_a, emb_b)  # cosine similarity
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

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

logger = logging.getLogger(__name__)

_MODEL_SEARCH_PATHS = [
    "models/iris_encoder_quantized.onnx",
    "models/iris_encoder.onnx",
    "models/iris_encoder.pth",
]

_INPUT_HEIGHT = 64
_INPUT_WIDTH = 512
_EMBEDDING_DIM = 128


def _find_model_path() -> Optional[str]:
    for p in _MODEL_SEARCH_PATHS:
        if Path(p).is_file():
            return str(Path(p).resolve())
    return None


def _normalize_iris_strip(
    gray: np.ndarray,
    pupil_center: Tuple[float, float],
    pupil_radius: float,
    limbus_radius: float,
    angle_deg: float = 0.0,
    output_size: Tuple[int, int] = (_INPUT_HEIGHT, _INPUT_WIDTH),
) -> np.ndarray:
    """Unwrap iris to polar-rectangular strip (Daugman rubber-sheet).

    Parameters
    ----------
    gray : np.ndarray
        Grayscale eye image.
    pupil_center : tuple
        (cx, cy) of pupil.
    pupil_radius : float
        Pupil semi-axis (px).
    limbus_radius : float
        Limbus semi-axis (px).
    angle_deg : float
        Rotation angle (degrees, for alignment).
    output_size : tuple
        (height, width) of output strip.

    Returns
    -------
    np.ndarray (H, W) uint8
        Normalized iris strip.
    """
    h, w = gray.shape[:2]
    out_h, out_w = output_size
    cx, cy = pupil_center

    strip = np.zeros((out_h, out_w), dtype=np.float32)

    for row in range(out_h):
        r_norm = row / max(out_h - 1, 1)
        r = pupil_radius + r_norm * (limbus_radius - pupil_radius)

        for col in range(out_w):
            theta = (col / max(out_w - 1, 1)) * 2 * np.pi
            theta_shifted = theta + np.deg2rad(angle_deg)

            x = cx + r * np.cos(theta_shifted)
            y = cy + r * np.sin(theta_shifted)

            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                strip[row, col] = gray[yi, xi]

    p2, p98 = np.percentile(strip[strip > 0], [2, 98]) if np.any(strip > 0) else (0, 255)
    if p98 > p2:
        strip = np.clip((strip - p2) / (p98 - p2) * 255, 0, 255)
    return strip.astype(np.uint8)


def _preprocess_strip(strip: np.ndarray) -> np.ndarray:
    """HWC uint8 -> NCHW float32 tensor with ImageNet-style normalization."""
    float_img = strip.astype(np.float32) / 255.0
    mean = float_img.mean()
    std = max(float_img.std(), 1e-6)
    normalized = (float_img - mean) / std
    tensor = normalized[np.newaxis, np.newaxis, ...]
    return tensor.astype(np.float32)


class _IrisEncoderModel(nn.Module):
    """PyTorch model for iris embedding extraction.

    Architecture:
        Conv2d(1,32,3,padding=1) -> BN -> ReLU -> MaxPool(2)    # 32x32x256
        Conv2d(32,64,3,padding=1) -> BN -> ReLU -> MaxPool(2)   # 64x16x128
        Conv2d(64,128,3,padding=1) -> BN -> ReLU -> MaxPool(2)  # 128x8x64
        Conv2d(128,256,3,padding=1) -> BN -> ReLU -> MaxPool(2) # 256x4x32
        AdaptiveAvgPool2d(1) -> Flatten -> FC(256,128) -> L2norm
    """

    def __init__(self, embedding_dim: int = _EMBEDDING_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class IrisEncoderNet:
    """Inference wrapper for iris texture encoding.

    Works with both PyTorch and ONNX Runtime backends.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        self._backend = None
        self._model = None
        self._session = None
        self._device = device

        if model_path is None:
            model_path = _find_model_path()

        if model_path is None:
            logger.warning("No iris encoder model found; CNN encoding disabled")
            return

        path = str(model_path)
        if path.endswith(".onnx"):
            self._load_onnx(path)
        elif path.endswith(".pth"):
            self._load_pytorch(path, device)
        else:
            logger.warning("Unknown model format: %s", path)

    def _load_onnx(self, path: str) -> None:
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
            logger.info("Iris encoder ONNX loaded: %s", path)
        except ImportError:
            logger.warning("onnxruntime not available; cannot load %s", path)
        except Exception as e:
            logger.warning("Cannot load ONNX model %s: %s", path, e)

    def _load_pytorch(self, path: str, device: str) -> None:
        if not _HAS_TORCH:
            logger.warning("PyTorch not available; cannot load %s", path)
            return

        try:
            model = _IrisEncoderModel()
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            model.eval()

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = model.to(device)
            self._backend = "pytorch"
            logger.info("Iris encoder PyTorch loaded: %s", path)
        except Exception as e:
            logger.warning("Cannot load PyTorch model %s: %s", path, e)

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    def encode_strip(self, strip: np.ndarray) -> np.ndarray:
        """Encode a normalized iris strip into an embedding.

        Parameters
        ----------
        strip : np.ndarray (H, W) uint8
            Normalized iris strip (polar unwrapped).

        Returns
        -------
        np.ndarray (_EMBEDDING_DIM,) float32
            L2-normalized embedding vector.
        """
        if not self.available:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        resized = cv2.resize(strip, (_INPUT_WIDTH, _INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        tensor = _preprocess_strip(resized)

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            emb = self._session.run(None, {input_name: tensor})[0]
            return emb.flatten().astype(np.float32)
        elif self._backend == "pytorch":
            with torch.no_grad():
                t = torch.from_numpy(tensor).to(next(self._model.parameters()).device)
                emb = self._model(t).cpu().numpy()
            return emb.flatten().astype(np.float32)

        return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

    def encode_patch(self, patch: np.ndarray) -> np.ndarray:
        """Encode a grayscale patch (any size) into an embedding.

        Resizes the patch to the standard input size before encoding.
        """
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return self.encode_strip(patch)

    def encode_batch(self, strips: list) -> np.ndarray:
        """Encode multiple strips. Returns (N, embedding_dim) array."""
        return np.stack([self.encode_strip(s) for s in strips], axis=0)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity matrix.

        Parameters
        ----------
        embeddings_a : (N, D)
        embeddings_b : (M, D)

        Returns
        -------
        np.ndarray (N, M)
        """
        a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
        b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
        return a_norm @ b_norm.T

    def save(self, path: str) -> None:
        """Save PyTorch model checkpoint."""
        if self._backend != "pytorch" or self._model is None:
            raise RuntimeError("No PyTorch model to save")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "embedding_dim": _EMBEDDING_DIM,
                "input_height": _INPUT_HEIGHT,
                "input_width": _INPUT_WIDTH,
            },
            path,
        )
        logger.info("Iris encoder saved: %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "IrisEncoderNet":
        """Load from a saved checkpoint or ONNX model."""
        return cls(model_path=path, device=device)


def create_model(
    embedding_dim: int = _EMBEDDING_DIM,
    device: str = "auto",
) -> "_IrisEncoderModel":
    """Create a new IrisEncoderNet PyTorch model."""
    model = _IrisEncoderModel(embedding_dim=embedding_dim)

    if device != "cpu" and _HAS_TORCH:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

    return model
