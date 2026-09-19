"""Tests for RedLightFilter temporal smoothing stability.

Phase 27 regression: ROI crop shape changes between frames must not
crash cv2.addWeighted inside _apply_temporal().
"""

import numpy as np
import pytest

from pupil_tracking.preprocessing.red_light_filter import RedLightFilter


def _make_mask(h, w, nonzero=False):
    mask = np.zeros((h, w), dtype=np.uint8)
    if nonzero:
        mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
    return mask


class TestTemporalShapeGuard:
    """Shape mismatch between consecutive frames must not crash."""

    def test_shape_mismatch_returns_current_mask(self):
        f = RedLightFilter(enable_temporal=True, enable_inpaint=False)
        mask_a = _make_mask(480, 640, nonzero=True)
        mask_b = _make_mask(512, 512, nonzero=True)

        # Frame 0 establishes temporal state with shape (480, 640)
        result_a = f._apply_temporal(mask_a, 0)
        assert result_a.shape == (480, 640)
        assert f._temporal_mask.shape == (480, 640)

        # Frame 1 has different shape — must not raise
        result_b = f._apply_temporal(mask_b, 1)
        assert result_b.shape == (512, 512)
        # Temporal state is reset to current mask
        assert f._temporal_mask.shape == (512, 512)
        assert f._temporal_count == 1

    def test_shape_mismatch_no_cv2_error(self):
        f = RedLightFilter(enable_temporal=True, enable_inpaint=False)
        mask_small = _make_mask(100, 200, nonzero=True)
        mask_large = _make_mask(300, 400, nonzero=True)

        f._apply_temporal(mask_small, 0)
        # Must not raise cv2.error
        result = f._apply_temporal(mask_large, 1)
        assert result.shape == (300, 400)

    def test_matching_shapes_blend_normally(self):
        f = RedLightFilter(
            enable_temporal=True, enable_inpaint=False, temporal_confidence=0.7
        )
        mask_a = _make_mask(200, 200, nonzero=True)
        mask_b = _make_mask(200, 200, nonzero=True)
        mask_b[:] = 0
        mask_b[50:150, 50:150] = 255

        f._apply_temporal(mask_a, 0)
        result = f._apply_temporal(mask_b, 1)

        assert result.shape == (200, 200)
        assert f._temporal_count == 2
        # Result should be binary (0 or 255) after threshold
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_zero_mask_fade_with_shape_change(self):
        """Shape change during zero-current-mask path must also be safe."""
        f = RedLightFilter(
            enable_temporal=True, enable_inpaint=False, temporal_confidence=0.7
        )
        mask_a = _make_mask(480, 640, nonzero=True)
        f._apply_temporal(mask_a, 0)
        assert f._temporal_count == 1

        # Different shape + zero mask → must reset, not fade stale
        mask_zero = _make_mask(512, 512, nonzero=False)
        result = f._apply_temporal(mask_zero, 1)
        assert result.shape == (512, 512)
        assert f._temporal_mask.shape == (512, 512)
        assert f._temporal_count == 1

    def test_reset_clears_temporal_state(self):
        f = RedLightFilter(enable_temporal=True, enable_inpaint=False)
        mask = _make_mask(200, 200, nonzero=True)
        f._apply_temporal(mask, 0)
        assert f._temporal_mask is not None

        f.reset_temporal()
        assert f._temporal_mask is None
        assert f._temporal_count == 0


class TestTemporalNormalBehavior:
    """Verify temporal smoothing works correctly with matching shapes."""

    def test_first_frame_initializes_state(self):
        f = RedLightFilter(enable_temporal=True, enable_inpaint=False)
        mask = _make_mask(100, 100, nonzero=True)
        result = f._apply_temporal(mask, 0)

        np.testing.assert_array_equal(result, mask)
        assert f._temporal_count == 1

    def test_blending_produces_binary_output(self):
        f = RedLightFilter(
            enable_temporal=True, enable_inpaint=False, temporal_confidence=0.5
        )
        mask_a = _make_mask(100, 100, nonzero=True)
        mask_b = _make_mask(100, 100, nonzero=True)
        mask_b[:30, :] = 0

        f._apply_temporal(mask_a, 0)
        result = f._apply_temporal(mask_b, 1)

        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_temporal_count_increments(self):
        f = RedLightFilter(enable_temporal=True, enable_inpaint=False)
        mask = _make_mask(100, 100, nonzero=True)

        f._apply_temporal(mask, 0)
        assert f._temporal_count == 1
        f._apply_temporal(mask, 1)
        assert f._temporal_count == 2
        f._apply_temporal(mask, 2)
        assert f._temporal_count == 3
