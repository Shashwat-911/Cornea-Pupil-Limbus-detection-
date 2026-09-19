"""
Iris image enhancement for cyclotorsion detection.

Applies CLAHE, Gabor filtering, and normalisation to improve iris
texture visibility and feature detectability in both Cartesian and
polar-unwrapped representations.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.config import get_config

logger = logging.getLogger(__name__)


class IrisEnhancer:
    """Multi-stage iris image enhancer.

    Enhancement pipeline:
        1. Adaptive histogram equalisation (CLAHE)
        2. Multi-scale Gabor filtering (optional)
        3. Illumination normalisation
        4. Noise reduction

    Parameters
    ----------
    clahe_clip : float
        CLAHE clip limit.
    clahe_grid : int
        CLAHE tile grid size.
    gabor_wavelengths : tuple
        Gabor filter wavelengths for texture enhancement.
    """

    def __init__(
        self,
        clahe_clip: Optional[float] = None,
        clahe_grid: Optional[int] = None,
        gabor_wavelengths: Optional[tuple] = None,
    ):
        cfg = get_config().registration
        self.clahe_clip = clahe_clip or cfg.enhance_clahe_clip
        self.clahe_grid = clahe_grid or cfg.enhance_clahe_grid
        self.gabor_wavelengths = gabor_wavelengths or cfg.enhance_gabor_wavelengths

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )

    def enhance(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        apply_gabor: bool = False,
    ) -> np.ndarray:
        """Apply full enhancement pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale uint8 or float32).
        mask : np.ndarray, optional
            Valid pixel mask (255 = valid).
        apply_gabor : bool
            Whether to apply Gabor filtering.

        Returns
        -------
        np.ndarray
            Enhanced image (float32, range [0, 1]).
        """
        # Ensure grayscale uint8
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.dtype == np.float32 or gray.dtype == np.float64:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Step 1: CLAHE
        enhanced = self._clahe.apply(gray)

        # Step 2: Illumination normalisation (local mean subtraction)
        enhanced = self._normalize_illumination(enhanced, mask)

        # Step 3: Gabor filtering (optional)
        if apply_gabor:
            enhanced = self._apply_gabor_bank(enhanced)

        # Step 4: Final normalisation to [0, 1]
        result = enhanced.astype(np.float32) / 255.0

        # Apply mask
        if mask is not None:
            result = result * (mask.astype(np.float32) / 255.0)

        return result

    def enhance_polar(
        self,
        polar_image: np.ndarray,
        polar_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enhance a polar-unwrapped iris image.

        Applies column-wise (angle-wise) normalisation in addition
        to the standard pipeline, to compensate for radial intensity
        gradients in the iris.

        Parameters
        ----------
        polar_image : np.ndarray
            Polar unwrapped image [num_radial, num_angles].
        polar_mask : np.ndarray, optional
            Valid pixel mask.

        Returns
        -------
        np.ndarray
            Enhanced polar image (float32, [0, 1]).
        """
        if polar_image is None or polar_image.size == 0:
            return polar_image

        # Convert to uint8 if needed
        if polar_image.dtype == np.float32 or polar_image.dtype == np.float64:
            img = np.clip(polar_image, 0, 255).astype(np.uint8)
        else:
            img = polar_image.copy()

        # CLAHE
        enhanced = self._clahe.apply(img)

        # Column-wise normalisation to remove radial gradient
        enhanced_f = enhanced.astype(np.float32)
        for col in range(enhanced_f.shape[1]):
            column = enhanced_f[:, col]
            if polar_mask is not None:
                valid = polar_mask[:, col] > 0
                if np.sum(valid) < 3:
                    continue
                col_vals = column[valid]
            else:
                col_vals = column

            col_mean = np.mean(col_vals)
            col_std = np.std(col_vals) + 1e-6
            enhanced_f[:, col] = (column - col_mean) / col_std

        # Rescale to [0, 1]
        vmin, vmax = np.percentile(enhanced_f[enhanced_f != 0], [2, 98]) if np.any(enhanced_f != 0) else (0, 1)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1.0
        result = np.clip((enhanced_f - vmin) / (vmax - vmin), 0, 1).astype(np.float32)

        if polar_mask is not None:
            result = result * (polar_mask.astype(np.float32) / 255.0)

        return result

    def _normalize_illumination(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Remove uneven illumination via large-kernel blur subtraction."""
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=31)
        # Subtract low-frequency background, shift to mean 128
        diff = image.astype(np.float32) - blur.astype(np.float32) + 128.0
        return np.clip(diff, 0, 255).astype(np.uint8)

    def _apply_gabor_bank(self, image: np.ndarray) -> np.ndarray:
        """Apply multi-scale, multi-orientation Gabor filter bank.

        Returns the maximum response across all filters at each pixel.
        """
        responses = []
        orientations = np.arange(0, np.pi, np.pi / 8)  # 8 orientations

        for wavelength in self.gabor_wavelengths:
            for theta in orientations:
                kernel = cv2.getGaborKernel(
                    ksize=(0, 0),
                    sigma=wavelength * 0.56,
                    theta=theta,
                    lambd=wavelength,
                    gamma=0.5,
                    psi=0,
                )
                filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
                responses.append(np.abs(filtered))

        # Maximum response across all filters
        max_response = np.max(responses, axis=0)
        # Normalise to uint8
        max_response = cv2.normalize(max_response, None, 0, 255, cv2.NORM_MINMAX)
        return max_response.astype(np.uint8)

    def extract_iris_crop(
        self,
        image: np.ndarray,
        center: Tuple[float, float],
        radius: float,
        padding: float = 1.1,
        target_size: int = 224,
    ) -> dict:
        """Extract a square crop centred on the iris.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        center : (float, float)
            Iris centre (x, y).
        radius : float
            Iris radius in pixels.
        padding : float
            Padding factor (1.1 = 10% padding).
        target_size : int
            Output crop size.

        Returns
        -------
        dict
            {'image': np.ndarray, 'crop_offset': (x1, y1),
             'crop_size': (w, h), 'target_size': int, 'scale': float}
        """
        cx, cy = int(center[0]), int(center[1])
        r = int(radius * padding)
        h, w = image.shape[:2]

        x1, y1 = max(0, cx - r), max(0, cy - r)
        x2, y2 = min(w, cx + r), min(h, cy + r)

        crop = image[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        if crop_h < 4 or crop_w < 4:
            return {
                'image': np.zeros((target_size, target_size, 3), dtype=np.uint8),
                'crop_offset': (x1, y1),
                'crop_size': (crop_w, crop_h),
                'target_size': target_size,
                'scale': 1.0,
            }

        scale = target_size / max(crop_w, crop_h)
        resized = cv2.resize(crop, (target_size, target_size))

        return {
            'image': resized,
            'crop_offset': (x1, y1),
            'crop_size': (crop_w, crop_h),
            'target_size': target_size,
            'scale': scale,
        }
