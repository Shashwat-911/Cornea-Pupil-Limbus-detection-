"""Phase XX-E: Unit tests for SmartContourFitter gradient caching."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pupil_tracking.core.smart_fitter import (
    SmartContourFitter,
    _compute_multiscale_gradient,
    _refine_contour_subpixel,
    _compute_gradient_weights,
)


class TestGradientCache:
    """Tests for gradient caching in SmartContourFitter."""

    def test_first_fit_populates_cache(self):
        """First fit with gray image should compute and cache gradients."""
        fitter = SmartContourFitter()
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 80, 255, -1)

        assert fitter._cached_grad_mag is None
        assert fitter._cache_gray_id is None

        fitter.fit(mask, gray_image=gray)

        assert fitter._cached_grad_mag is not None
        assert fitter._cached_grad_x is not None
        assert fitter._cached_grad_y is not None
        assert fitter._cache_gray_id == id(gray)

    def test_second_fit_reuses_cache(self):
        """Second fit with same gray image should reuse cached gradients."""
        fitter = SmartContourFitter()
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mask1 = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask1, (200, 240), 50, 255, -1)
        mask2 = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask2, (400, 240), 50, 255, -1)

        fitter.fit(mask1, gray_image=gray)
        grad_id_after_first = id(fitter._cached_grad_mag)

        fitter.fit(mask2, gray_image=gray)
        grad_id_after_second = id(fitter._cached_grad_mag)

        # Same object, not recomputed
        assert grad_id_after_first == grad_id_after_second

    def test_cache_invalidated_on_new_image(self):
        """Cache should be invalidated when gray image changes."""
        fitter = SmartContourFitter()
        gray1 = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        gray2 = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 80, 255, -1)

        fitter.fit(mask, gray_image=gray1)
        grad_id1 = id(fitter._cached_grad_mag)
        gray_id1 = fitter._cache_gray_id

        fitter.fit(mask, gray_image=gray2)
        grad_id2 = id(fitter._cached_grad_mag)
        gray_id2 = fitter._cache_gray_id

        # Different image → different cache
        assert gray_id1 != gray_id2
        # Gradient arrays should be different objects (recomputed)
        assert grad_id1 != grad_id2

    def test_cache_invalidated_on_dimension_change(self):
        """Cache should be invalidated when image dimensions change."""
        fitter = SmartContourFitter()
        gray1 = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        gray2 = np.random.randint(0, 256, (500, 700), dtype=np.uint8)
        mask1 = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask1, (320, 240), 80, 255, -1)
        mask2 = np.zeros((500, 700), dtype=np.uint8)
        cv2.circle(mask2, (350, 250), 80, 255, -1)

        fitter.fit(mask1, gray_image=gray1)
        assert fitter._cached_grad_mag.shape == (480, 640)

        fitter.fit(mask2, gray_image=gray2)
        assert fitter._cached_grad_mag.shape == (500, 700)

    def test_cached_and_uncached_produce_equivalent_results(self):
        """Cached and uncached paths should produce equivalent results."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([
            [300, 200], [320, 210], [340, 240], [330, 270],
            [300, 280], [270, 270], [260, 240], [270, 210],
        ], dtype=np.float64)

        # Uncached
        result_uncached = _refine_contour_subpixel(gray, contour)

        # Cached
        grad_mag, grad_x, grad_y = _compute_multiscale_gradient(gray)
        result_cached = _refine_contour_subpixel(
            gray, contour,
            cached_grad_mag=grad_mag,
            cached_grad_x=grad_x,
            cached_grad_y=grad_y,
        )

        np.testing.assert_array_almost_equal(result_cached, result_uncached, decimal=10)

    def test_gradient_weights_still_computed_independently(self):
        """Gradient weights should be computed independently (not cached)."""
        fitter = SmartContourFitter()
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 80, 255, -1)

        # Fit to populate gradient cache
        fitter.fit(mask, gray_image=gray)
        assert fitter._cached_grad_mag is not None

        # Gradient weights use single-scale Scharr, not cached multiscale
        pts = np.array([[320, 240], [330, 250], [340, 240]], dtype=np.float64)
        weights = _compute_gradient_weights(gray, pts)
        assert weights is not None
        assert len(weights) == 3

    def test_no_stale_cache_across_fitters(self):
        """Different fitter instances should not share cache state."""
        fitter1 = SmartContourFitter()
        fitter2 = SmartContourFitter()
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 80, 255, -1)

        fitter1.fit(mask, gray_image=gray)
        assert fitter1._cached_grad_mag is not None
        assert fitter2._cached_grad_mag is None

    def test_cache_cleared_when_no_gray(self):
        """Cache should not be used when no gray image is provided."""
        fitter = SmartContourFitter()
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 240), 80, 255, -1)

        fitter.fit(mask, gray_image=None)
        assert fitter._cached_grad_mag is None
        assert fitter._cache_gray_id is None
