"""Visualisation / debug overlays for iris-feature detection.

This is an **optional**, diagnostic-only path. It is never enabled in the
default production GUI and does not alter production overlay behaviour. It is
used for manual validation and debugging.

Colour scheme (consistent with the existing project overlays where possible):
    * pupil / limbus ellipses drawn in the standard green / orange tones
    * iris ROI (annulus) drawn as concentric ring contours
    * rejected or masked regions optionally drawn in dark red
    * accepted features drawn as points with a brightness/confidence scale
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from pupil_tracking.iris.types import IrisDetectionResult, IrisFeatureSet, IrisROI
from pupil_tracking.utils.types import EllipseParams

_PUPIL_COLOR = (0, 255, 0)
_LIMBUS_COLOR = (255, 180, 0)
_ROI_COLOR = (200, 200, 0)
_MASKED_COLOR = (0, 0, 150)


def _draw_ellipse(img, ellipse: Optional[EllipseParams], color) -> None:
    if ellipse is None or not ellipse.is_valid:
        return
    center = (int(round(ellipse.center_x)), int(round(ellipse.center_y)))
    axes = (int(round(ellipse.semi_major)), int(round(ellipse.semi_minor)))
    cv2.ellipse(img, center, axes, ellipse.angle_deg, 0.0, 360.0, color, 1)


def _draw_rings(img, roi: IrisROI) -> None:
    if not roi.valid:
        return
    center = (int(round(roi.center_x)), int(round(roi.center_y)))
    inner = int(round(roi.pupil_radius_px * (1.0 + roi.inner_inset_frac)))
    outer = int(round(roi.limbus_radius_px * (1.0 - roi.outer_inset_frac)))
    cv2.circle(img, center, inner, _ROI_COLOR, 1)
    cv2.circle(img, center, outer, _ROI_COLOR, 1)


def _draw_feature(img, feat, color) -> None:
    pt = (int(round(feat.x)), int(round(feat.y)))
    cv2.circle(img, pt, 2, color, -1)
    ang = np.deg2rad(feat.orientation_deg)
    x2 = int(round(feat.x + 4.0 * np.cos(ang)))
    y2 = int(round(feat.y + 4.0 * np.sin(ang)))
    cv2.line(img, pt, (x2, y2), color, 1)


def draw_iris_overlay(
    image_bgr: np.ndarray,
    result: IrisDetectionResult,
    *,
    pupil: Optional[EllipseParams] = None,
    limbus: Optional[EllipseParams] = None,
    draw_masked_region: bool = False,
    usable_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a copy of ``image_bgr`` with the iris debug overlay drawn.

    Parameters
    ----------
    image_bgr : (H, W, 3) BGR image to annotate (not modified in place).
    result : IrisDetectionResult from the detector.
    pupil/limbus : optional existing ellipses to draw as reference.
    draw_masked_region : if True, colour the masked (non-usable) iris pixels.
    usable_mask : boolean (H, W) mask; required when ``draw_masked_region``.
    """
    out = image_bgr.copy()
    if draw_masked_region and usable_mask is not None:
        overlay = np.zeros_like(out)
        mask_bgr = np.where(usable_mask[..., None], 0, _MASKED_COLOR)
        out = cv2.addWeighted(out, 1.0, mask_bgr.astype(np.uint8), 0.4, 0)

    if pupil is not None:
        _draw_ellipse(out, pupil, _PUPIL_COLOR)
    if limbus is not None:
        _draw_ellipse(out, limbus, _LIMBUS_COLOR)

    roi = result.feature_set.roi
    if roi.valid:
        _draw_rings(out, roi)

    for feat in result.feature_set.features:
        conf = float(feat.confidence)
        # map confidence in [0,1] to a blue->green->red-ish scale
        blue = int(np.clip(255 * (1.0 - conf), 0, 255))
        green = int(np.clip(255 * conf, 0, 255))
        _draw_feature(out, feat, (0, green, blue))

    return out
