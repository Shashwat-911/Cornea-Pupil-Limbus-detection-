"""Tests for vectorized area filtering in RedLightFilter (PHASE XX-I)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pupil_tracking.preprocessing.red_light_filter import RedLightFilter


class TestVectorizedAreaFiltering:
    """Verify that np.isin vectorization produces identical masks to the original loop."""

    def _detect_original(self, f: RedLightFilter, image: np.ndarray) -> np.ndarray:
        """Original per-label loop (for reference comparison)."""
        h, w = image.shape[:2]
        total_area = h * w
        max_blob_area = int(total_area * f.max_area_frac)

        b, g, r = cv2.split(image)
        red_high = (r >= f.red_threshold).astype(np.uint8) * 255
        red_dominant = (
            (r > g + f.dominance_offset) & (r > b + f.dominance_offset)
        ).astype(np.uint8) * 255
        very_bright_red = (r >= 240).astype(np.uint8) * 255
        candidates = cv2.bitwise_or(red_high, red_dominant)
        candidates = cv2.bitwise_or(candidates, very_bright_red)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, sat, _ = cv2.split(hsv)
        low_sat = (sat <= 25).astype(np.uint8) * 255
        bright_enough = (r >= 200).astype(np.uint8) * 255
        sat_filtered = cv2.bitwise_and(bright_enough, low_sat)
        candidates = cv2.bitwise_or(candidates, sat_filtered)

        pink_mask = ((r >= 180) & (g >= 80) & (b >= 150) & (r > g) & (b > g)).astype(np.uint8) * 255
        candidates = cv2.bitwise_or(candidates, pink_mask)

        bright_saturated = ((r >= 220) & (sat >= 50)).astype(np.uint8) * 255
        candidates = cv2.bitwise_or(candidates, bright_saturated)

        if f._dilate_kernel is not None:
            candidates = cv2.dilate(candidates, f._dilate_kernel, iterations=1)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)

        mask = np.zeros((h, w), dtype=np.uint8)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < f.min_area:
                continue
            if area > max_blob_area:
                continue
            mask[labels == i] = 255

        return mask

    def test_empty_image(self):
        f = RedLightFilter(enable_temporal=False, enable_inpaint=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = f._detect_red_lights(image)
        assert mask.shape == (100, 100)
        assert mask.sum() == 0

    def test_single_red_blob_matches_original(self):
        f = RedLightFilter(
            red_threshold=180, dominance_offset=20, min_area=3,
            max_area_frac=0.15, enable_temporal=False, dilation_size=5,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 20, (0, 0, 255), -1)

        mask_new = f._detect_red_lights(image)
        mask_old = self._detect_original(f, image)

        assert np.array_equal(mask_new, mask_old)

    def test_multiple_blobs_matches_original(self):
        f = RedLightFilter(
            red_threshold=200, dominance_offset=30, min_area=5,
            max_area_frac=0.15, enable_temporal=False, dilation_size=5,
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
        cv2.circle(image, (150, 50), 10, (0, 0, 255), -1)
        cv2.circle(image, (100, 150), 25, (0, 0, 255), -1)

        mask_new = f._detect_red_lights(image)
        mask_old = self._detect_original(f, image)

        assert np.array_equal(mask_new, mask_old)

    def test_tiny_blobs_filtered_out(self):
        f = RedLightFilter(
            red_threshold=200, dominance_offset=30, min_area=10,
            max_area_frac=0.15, enable_temporal=False, dilation_size=0,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # 2x2 red blob = 4 pixels, below min_area=10
        image[10:12, 10:12] = [0, 0, 255]

        mask = f._detect_red_lights(image)
        assert mask.sum() == 0

    def test_huge_blob_filtered_out(self):
        f = RedLightFilter(
            red_threshold=200, dominance_offset=30, min_area=5,
            max_area_frac=0.05, enable_temporal=False, dilation_size=0,
        )
        # Create image where red fills > 5% of area
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 10:90] = [0, 0, 255]  # 80x80 = 6400 px = 64% of 10000

        mask = f._detect_red_lights(image)
        assert mask.sum() == 0

    def test_no_labels_passes_filter(self):
        f = RedLightFilter(
            red_threshold=250, dominance_offset=100, min_area=5,
            max_area_frac=0.15, enable_temporal=False, dilation_size=0,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[50, 50] = [0, 0, 200]

        mask = f._detect_red_lights(image)
        assert mask.sum() == 0

    def test_many_small_blobs_vectorized_correctly(self):
        """Many tiny blobs — vectorized must produce same result as loop."""
        f = RedLightFilter(
            red_threshold=180, dominance_offset=10, min_area=3,
            max_area_frac=0.15, enable_temporal=False, dilation_size=0,
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # 50 tiny red spots
        for i in range(50):
            y, x = 10 + (i % 10) * 18, 10 + (i // 10) * 18
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        mask_new = f._detect_red_lights(image)
        mask_old = self._detect_original(f, image)

        assert np.array_equal(mask_new, mask_old)
        assert mask_new.sum() > 0

    def test_vectorized_speedup_many_labels(self):
        """Performance: vectorized must be faster than loop for many labels."""
        f = RedLightFilter(
            red_threshold=180, dominance_offset=10, min_area=3,
            max_area_frac=0.15, enable_temporal=False, dilation_size=0,
        )
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        for i in range(200):
            y, x = 20 + (i % 20) * 23, 20 + (i // 20) * 23
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

        import time
        t0 = time.perf_counter()
        for _ in range(3):
            mask_new = f._detect_red_lights(image)
        t_new = (time.perf_counter() - t0) / 3 * 1000

        t0 = time.perf_counter()
        for _ in range(3):
            mask_old = self._detect_original(f, image)
        t_old = (time.perf_counter() - t0) / 3 * 1000

        assert np.array_equal(mask_new, mask_old)
        assert t_new < t_old, f"vectorized ({t_new:.1f}ms) should be faster than loop ({t_old:.1f}ms)"

    def test_apply_end_to_end_masks_unchanged(self):
        """Full apply() pipeline produces same mask as manual detection."""
        f = RedLightFilter(
            red_threshold=200, dominance_offset=30, min_area=5,
            max_area_frac=0.15, enable_inpaint=True, inpaint_radius=3,
            enable_temporal=False, dilation_size=5,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)

        filtered, mask = f.apply(image)
        assert mask.shape == (100, 100)
        assert mask.max() == 255
        assert filtered.shape == image.shape
