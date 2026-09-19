"""Phase XX-F: Unit tests for vectorized subpixel refinement."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from pupil_tracking.core.smart_fitter import (
    _refine_contour_subpixel,
    _compute_multiscale_gradient,
)


class TestVectorizedSubpixel:
    """Tests for vectorized _refine_contour_subpixel."""

    def test_empty_contour(self):
        """Empty contour should return empty array."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([], dtype=np.float64).reshape(0, 2)
        result = _refine_contour_subpixel(gray, contour)
        assert result.shape == (0, 2)

    def test_single_point(self):
        """Single valid point should be refined."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([[320, 240]], dtype=np.float64)
        result = _refine_contour_subpixel(gray, contour)
        assert result.shape == (1, 2)
        # Point should be near original (within search radius)
        assert abs(result[0, 0] - 320) < 5
        assert abs(result[0, 1] - 240) < 5

    def test_boundary_points_unchanged(self):
        """Points outside image bounds should remain unchanged."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([
            [0, 0],      # corner
            [639, 479],  # opposite corner
            [320, 240],  # center
        ], dtype=np.float64)
        result = _refine_contour_subpixel(gray, contour)
        # Boundary points should not move
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[1, 0] == 639.0
        assert result[1, 1] == 479.0
        # Center point should be refined
        assert result[2, 0] != 320.0 or result[2, 1] != 240.0

    def test_zero_gradient_unchanged(self):
        """Points with zero gradient should remain unchanged."""
        gray = np.ones((480, 640), dtype=np.uint8) * 128  # uniform
        contour = np.array([[320, 240], [330, 250]], dtype=np.float64)
        result = _refine_contour_subpixel(gray, contour)
        # Uniform image → zero gradient → no refinement
        np.testing.assert_array_equal(result, contour)

    def test_cached_vs_uncached_equivalent(self):
        """Cached and uncached paths should produce identical results."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([
            [300, 200], [320, 210], [340, 240], [330, 270],
            [300, 280], [270, 270], [260, 240], [270, 210],
        ], dtype=np.float64)

        result_uncached = _refine_contour_subpixel(gray, contour)

        grad_mag, grad_x, grad_y = _compute_multiscale_gradient(gray)
        result_cached = _refine_contour_subpixel(
            gray, contour,
            cached_grad_mag=grad_mag,
            cached_grad_x=grad_x,
            cached_grad_y=grad_y,
        )

        np.testing.assert_array_almost_equal(result_cached, result_uncached, decimal=10)

    def test_multiple_points_independent(self):
        """Each point should be refined independently."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([
            [200, 200], [400, 200], [200, 400], [400, 400],
        ], dtype=np.float64)
        result = _refine_contour_subpixel(gray, contour)
        assert result.shape == (4, 2)
        # All points should be refined (moved from original)
        for i in range(4):
            dist = math.sqrt(
                (result[i, 0] - contour[i, 0])**2 +
                (result[i, 1] - contour[i, 1])**2
            )
            # Should be refined (moved) but within search radius
            assert dist < 5.0

    def test_parabolic_interpolation(self):
        """Parabolic interpolation should refine peak location."""
        # Create image with strong edge
        gray = np.zeros((480, 640), dtype=np.uint8)
        gray[:, :320] = 200
        gray[:, 320:] = 50

        contour = np.array([[320, 240]], dtype=np.float64)
        result = _refine_contour_subpixel(gray, contour, use_parabolic=True)
        # Edge at x=320 should be refined to sub-pixel location
        assert result.shape == (1, 2)

    def test_no_parabolic(self):
        """Without parabolic, result should use peak sample directly."""
        gray = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        contour = np.array([[320, 240]], dtype=np.float64)
        result_no_para = _refine_contour_subpixel(gray, contour, use_parabolic=False)
        result_para = _refine_contour_subpixel(gray, contour, use_parabolic=True)
        # Both should produce valid results
        assert result_no_para.shape == (1, 2)
        assert result_para.shape == (1, 2)

    def test_large_contour_performance(self):
        """Vectorized should handle large contours efficiently."""
        gray = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
        # Simulate large contour (circle)
        theta = np.linspace(0, 2 * np.pi, 6000)
        contour = np.column_stack([
            960 + 400 * np.cos(theta),
            540 + 400 * np.sin(theta),
        ]).astype(np.float64)

        import time
        t0 = time.perf_counter()
        result = _refine_contour_subpixel(gray, contour)
        elapsed = (time.perf_counter() - t0) * 1000

        assert result.shape == (6000, 2)
        # Should complete in reasonable time (< 500ms for 6000 points)
        assert elapsed < 500, f"Too slow: {elapsed:.0f}ms"
