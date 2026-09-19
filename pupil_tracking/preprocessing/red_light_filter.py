"""
Dedicated red light filter for surgical eye imaging.

This module specifically targets blinking red LED/surgical lights that can
temporarily distort pupil detection. Unlike general specular reflection
removal, this filter is designed to:

1. Detect bright red light spots (typically R > G + threshold, R > B + threshold)
2. Identify blinking patterns (temporal consistency checks)
3. Mask out these regions before ML segmentation
4. Optionally inpaint or neutralize the red light contribution

The key difference from ReflectionRemover:
- RedLightFilter focuses specifically on red-dominated bright spots
- Uses color-space analysis (RGB ratios) rather than just intensity
- Can handle the specific case of surgical red illumination lights
- Designed to be conservative to not affect normal iris color

Plan-aligned changes:
    - Initial implementation for red light filtering
    - Integration with preprocessing pipeline
    - Optional temporal tracking for blink detection
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.logger import get_logger


class RedLightFilter:
    """Filter out bright red lights that interfere with pupil detection.

    This filter specifically targets surgical illumination lights that
    emit bright red spots. These can temporarily shift or distort the
    detected pupil position when they appear on the pupil region.

    The filter works by:
    1. Detecting red-dominant bright pixels using color ratio analysis
    2. Applying morphological operations to get clean blobs
    3. Optionally inpainting or masking the detected regions

    Usage
    -----
    >>> filter = RedLightFilter()
    >>> clean_image, red_mask = filter.apply(image_bgr)

    For video with temporal tracking:
    >>> filter = RedLightFilter(enable_temporal=True)
    >>> clean_image, red_mask = filter.apply(image_bgr, frame_number=0)

    Parameters
    ----------
    red_threshold : int
        Minimum red channel value to consider (0-255). Default 200.
    dominance_offset : int
        Red must be at least this much greater than G and B. Default 30.
    min_area : int
        Minimum blob area in pixels. Default 5.
    max_area_frac : float
        Maximum fraction of image area for a single blob. Default 0.1 (10%)
    enable_inpaint : bool
        If True, inpaint the detected red regions. If False, just return mask.
    inpaint_radius : int
        Radius for cv2.inpaint(). Default 3.
    enable_temporal : bool
        If True, track red light position across frames for better detection.
    temporal_confidence : float
        Weight for temporal information (0-1). Higher = more stable.
    """

    def __init__(
        self,
        red_threshold: int = 180,
        dominance_offset: int = 20,
        min_area: int = 3,
        max_area_frac: float = 0.15,
        enable_inpaint: bool = True,
        inpaint_radius: int = 5,
        enable_temporal: bool = False,
        temporal_confidence: float = 0.7,
        dilation_size: int = 5,
    ) -> None:
        self.red_threshold = red_threshold
        self.dominance_offset = dominance_offset
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.enable_inpaint = enable_inpaint
        self.inpaint_radius = inpaint_radius
        self.enable_temporal = enable_temporal
        self.temporal_confidence = temporal_confidence
        self.dilation_size = dilation_size
        self.logger = get_logger()

        if dilation_size > 0:
            self._dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_size, dilation_size),
            )
        else:
            self._dilate_kernel = None

        self._temporal_mask: Optional[np.ndarray] = None
        self._temporal_count: int = 0

    def apply(
        self,
        image: np.ndarray,
        frame_number: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply red light filtering to an image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image of shape (H, W, 3).
        frame_number : int
            Frame number for temporal tracking (-1 for single images).

        Returns
        -------
        (filtered_image, red_mask)
            filtered_image : Same format as input, with red regions handled
            red_mask : Binary uint8 mask (0/255) showing detected red regions
        """
        if image is None or image.size == 0:
            h, w = image.shape[:2] if image is not None else (0, 0)
            return image, np.zeros((h, w), dtype=np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        mask = self._detect_red_lights(image)

        if self.enable_temporal and frame_number >= 0:
            mask = self._apply_temporal(mask, frame_number)

        if not np.any(mask):
            return image, mask

        h, w = image.shape[:2]
        total_area = h * w

        filtered = image.copy()
        if self.enable_inpaint:
            filtered = cv2.inpaint(
                filtered, mask, self.inpaint_radius, cv2.INPAINT_TELEA
            )

        n_red = int(np.count_nonzero(mask))
        pct = 100.0 * n_red / total_area

        self.logger.debug(
            "RedLightFilter: %d pixels (%.2f%% of image)",
            n_red,
            pct,
        )

        return filtered, mask

    def _detect_red_lights(self, image: np.ndarray) -> np.ndarray:
        """Detect bright red light spots using color analysis.

        Strategy:
        1. Split into BGR channels
        2. Find pixels where R is dominant (R > G + offset AND R > B + offset)
        3. Also check for high absolute red value (very bright red lights)
        4. Combine with morphological operations for clean blobs
        5. Filter by area constraints
        """
        h, w = image.shape[:2]
        total_area = h * w
        max_blob_area = int(total_area * self.max_area_frac)

        b, g, r = cv2.split(image)

        red_high = (r >= self.red_threshold).astype(np.uint8) * 255

        red_dominant = (
            (r > g + self.dominance_offset) & (r > b + self.dominance_offset)
        ).astype(np.uint8) * 255

        very_bright_red = (r >= 240).astype(np.uint8) * 255

        candidates = cv2.bitwise_or(red_high, red_dominant)
        candidates = cv2.bitwise_or(candidates, very_bright_red)

        # For very bright red lights, don't require low saturation
        # as LED lights can have high saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, sat, _ = cv2.split(hsv)

        # Apply saturation filter only to non-very-bright red
        low_sat = (sat <= 25).astype(np.uint8) * 255
        bright_enough = (r >= 200).astype(np.uint8) * 255

        # Combine: (very bright) OR (bright AND low saturation)
        sat_filtered = cv2.bitwise_and(bright_enough, low_sat)
        candidates = cv2.bitwise_or(candidates, sat_filtered)

        # Also detect pink/magenta hues (common in surgical lights)
        # Pink: high R, moderate-high G, high B
        pink_mask = ((r >= 180) & (g >= 80) & (b >= 150) & (r > g) & (b > g)).astype(
            np.uint8
        ) * 255
        candidates = cv2.bitwise_or(candidates, pink_mask)

        # Detect any very bright reddish region regardless of other channels
        # This catches saturated red LEDs
        bright_saturated = ((r >= 220) & (sat >= 50)).astype(np.uint8) * 255
        candidates = cv2.bitwise_or(candidates, bright_saturated)

        if self._dilate_kernel is not None:
            candidates = cv2.dilate(candidates, self._dilate_kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidates, connectivity=8
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            if area > max_blob_area:
                continue
            mask[labels == i] = 255

        return mask

    def get_red_light_regions(self, image: np.ndarray) -> list:
        """Return list of (x, y, radius) for detected red light regions.

        Can be used for post-processing to ignore detections near red lights.

        Returns
        -------
        List of (center_x, center_y, radius) tuples
        """
        mask = self._detect_red_lights(image)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        regions = []
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue
            cx, cy = centroids[i]
            radius = int(np.sqrt(area / np.pi)) * 2  # Approximate radius
            regions.append((int(cx), int(cy), radius))

        return regions

    def _apply_temporal(
        self,
        current_mask: np.ndarray,
        frame_number: int,
    ) -> np.ndarray:
        """Apply temporal smoothing to red light detection.

        If enabled, this tracks red light positions across frames and
        provides more stable detection for blinking lights.
        """
        if self._temporal_mask is None or frame_number == 0:
            self._temporal_mask = current_mask.copy()
            self._temporal_count = 1
            return current_mask

        if current_mask.shape != self._temporal_mask.shape:
            self._temporal_mask = current_mask.copy()
            self._temporal_count = 1
            return current_mask

        if current_mask.sum() == 0 and self._temporal_mask.sum() > 0:
            confidence = min(self.temporal_confidence, self._temporal_count / 10.0)
            if confidence > 0.3:
                faded = (self._temporal_mask * (1.0 - confidence)).astype(np.uint8)
                _, faded = cv2.threshold(faded, 127, 255, cv2.THRESH_BINARY)
                return faded

        if current_mask.sum() > 0:
            alpha = self.temporal_confidence
            combined = cv2.addWeighted(
                current_mask, alpha, self._temporal_mask, 1 - alpha, 0
            )
            _, combined = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)

            self._temporal_mask = combined
            self._temporal_count += 1
            return combined

        return current_mask

    def detect_only(self, image: np.ndarray) -> np.ndarray:
        """Return the red light mask without filtering."""
        return self._detect_red_lights(image)

    def reset_temporal(self) -> None:
        """Reset temporal tracking state."""
        self._temporal_mask = None
        self._temporal_count = 0

    def get_red_light_stats(self, image: np.ndarray) -> dict:
        """Return statistics about detected red lights."""
        mask = self._detect_red_lights(image)
        total = image.shape[0] * image.shape[1]
        n_red = int(np.count_nonzero(mask))

        n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        blobs = []
        for i in range(1, n_labels):
            blobs.append(
                {
                    "area": int(stats[i, cv2.CC_STAT_AREA]),
                    "center": (
                        float(centroids[i, 0]),
                        float(centroids[i, 1]),
                    ),
                }
            )

        return {
            "total_red_pixels": n_red,
            "red_fraction": n_red / max(total, 1),
            "num_blobs": len(blobs),
            "blobs": blobs,
        }


class AdaptiveRedLightFilter(RedLightFilter):
    """Adaptive version that auto-tunes based on image content.

    This version automatically adjusts thresholds based on:
    - Image brightness levels
    - Presence of red-dominant regions
    - Overall color distribution
    """

    def __init__(
        self, base_threshold: int = 200, base_dominance: int = 30, **kwargs
    ) -> None:
        super().__init__(
            red_threshold=base_threshold, dominance_offset=base_dominance, **kwargs
        )
        self._base_threshold = base_threshold
        self._base_dominance = base_dominance

    def _auto_adjust(self, image: np.ndarray) -> None:
        """Automatically adjust thresholds based on image content."""
        b, g, r = cv2.split(image)

        r_mean = float(r.mean())
        g_mean = float(g.mean())
        b_mean = float(b.mean())

        overall_brightness = (r_mean + g_mean + b_mean) / 3.0

        if overall_brightness < 80:
            self.red_threshold = max(150, self._base_threshold - 30)
            self.dominance_offset = max(20, self._base_dominance - 10)
        elif overall_brightness > 180:
            self.red_threshold = min(230, self._base_threshold + 20)
            self.dominance_offset = min(50, self._base_dominance + 15)
        else:
            self.red_threshold = self._base_threshold
            self.dominance_offset = self._base_dominance

    def apply(
        self,
        image: np.ndarray,
        frame_number: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if image is not None and image.size > 0:
            self._auto_adjust(image)
        return super().apply(image, frame_number)
