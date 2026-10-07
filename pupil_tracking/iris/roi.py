"""Iris annular ROI construction from existing pupil/limbus geometry.

The ROI is the ring of iris tissue between the pupil ellipse and the limbus
ellipse. This module consumes the existing detection geometry (``EllipseParams``
from ``pupil_tracking``) and does **not** run its own pupil/limbus detector.
"""

from __future__ import annotations

import math

import numpy as np

from pupil_tracking.iris.types import IrisROI
from pupil_tracking.utils.types import EllipseParams

# Sanity limits for the pupil/limbus radius ratio (mean radii). These guard
# against degenerate or implausible geometry before we build a ring.
MIN_PUPIL_LIMBUS_RATIO = 0.10
MAX_PUPIL_LIMBUS_RATIO = 0.95

# Minimum limbus size (px) below which the annulus is too thin to be useful.
MIN_LIMBUS_RADIUS_PX = 10.0


def _ellipse_mean_radius(ellipse: EllipseParams) -> float:
    if ellipse is None:
        return 0.0
    return (ellipse.semi_major + ellipse.semi_minor) / 2.0


class IrisROIExtractor:
    """Build an annular iris ROI from pupil/limbus ellipse geometry.

    Parameters
    ----------
    inner_inset_frac : float
        Fraction of the local pupil radius to back away from the pupil edge.
        A value of 0 keeps the ROI boundary on the pupil ellipse; 0.1 moves the
        inner boundary 10% of the pupil radius outward into the iris.
    outer_inset_frac : float
        Fraction of the local limbus radius to back away from the limbus edge
        toward the pupil. Avoids limbus-ambiguous and sclera-adjacent pixels.
    """

    def __init__(
        self,
        inner_inset_frac: float = 0.10,
        outer_inset_frac: float = 0.10,
    ) -> None:
        self.inner_inset_frac = float(inner_inset_frac)
        self.outer_inset_frac = float(outer_inset_frac)

    def build(
        self,
        pupil: EllipseParams,
        limbus: EllipseParams,
    ) -> IrisROI:
        """Construct the iris ROI from existing geometry.

        Returns an :class:`IrisROI` with ``valid=False`` (and a ``reason``)
        when the geometry is missing or implausible; it never raises.
        """
        roi = IrisROI(
            inner_inset_frac=self.inner_inset_frac,
            outer_inset_frac=self.outer_inset_frac,
        )

        if pupil is None or limbus is None:
            roi.valid = False
            roi.reason = "missing pupil or limbus geometry"
            return roi
        if not pupil.is_valid or not limbus.is_valid:
            roi.valid = False
            roi.reason = "invalid pupil or limbus ellipse"
            return roi

        p_r = _ellipse_mean_radius(pupil)
        l_r = _ellipse_mean_radius(limbus)
        if l_r < MIN_LIMBUS_RADIUS_PX:
            roi.valid = False
            roi.reason = "limbus radius below minimum"
            return roi
        ratio = p_r / l_r if l_r > 0 else 0.0
        if ratio < MIN_PUPIL_LIMBUS_RATIO or ratio > MAX_PUPIL_LIMBUS_RATIO:
            roi.valid = False
            roi.reason = f"pupil/limbus radius ratio out of range: {ratio:.2f}"
            return roi

        # Reference for pixel-space sanity checks: the absolute mean radii.
        roi.center_x = float(limbus.center_x)
        roi.center_y = float(limbus.center_y)
        roi.pupil_center_x = float(pupil.center_x)
        roi.pupil_center_y = float(pupil.center_y)
        roi.pupil_semi_major = float(pupil.semi_major)
        roi.pupil_semi_minor = float(pupil.semi_minor)
        roi.pupil_angle_deg = float(pupil.angle_deg)
        roi.limbus_semi_major = float(limbus.semi_major)
        roi.limbus_semi_minor = float(limbus.semi_minor)
        roi.limbus_angle_deg = float(limbus.angle_deg)
        roi.pupil_radius_px = p_r
        roi.limbus_radius_px = l_r

        roi.valid = True
        roi.reason = "ok"
        return roi

    def build_from_detection(
        self,
        detection,
        *,
        use_pupil: bool = True,
    ) -> IrisROI:
        """Convenience: build ROI from a result object exposing ``pupil`` and
        ``limbus`` attributes (e.g. an ``EyeDetectionResult``).

        When ``use_pupil`` is True (default), both detections must be flagged
        detected; otherwise the returned ROI is invalid.
        """
        pupil_ellipse = None
        limbus_ellipse = None
        try:
            if use_pupil and getattr(detection.pupil, "detected", False):
                pupil_ellipse = detection.pupil.ellipse
            elif not use_pupil:
                pupil_ellipse = None
            if getattr(detection.limbus, "detected", False):
                limbus_ellipse = detection.limbus.ellipse
        except AttributeError:
            return IrisROI(valid=False, reason="detection object missing attributes")

        if pupil_ellipse is None and use_pupil:
            return IrisROI(
                valid=False,
                reason="pupil not detected",
                inner_inset_frac=self.inner_inset_frac,
                outer_inset_frac=self.outer_inset_frac,
            )
        return self.build(pupil_ellipse, limbus_ellipse)


def point_in_roi_annulus(x: float, y: float, roi: IrisROI) -> bool:
    """Return True if a point lies within the iris annulus with insets applied.

    Evaluated in each ellipse's respective coordinate frame (handling decentered
    and eccentric geometry exactly).
    """
    if not roi.valid:
        return False

    # Check inside limbus (with outer inset)
    l_smaj = roi.limbus_semi_major * (1.0 - roi.outer_inset_frac)
    l_smin = roi.limbus_semi_minor * (1.0 - roi.outer_inset_frac)
    if l_smaj <= 0 or l_smin <= 0:
        return False
    phi_l = math.radians(roi.limbus_angle_deg)
    dx_l = x - roi.center_x
    dy_l = y - roi.center_y
    xr_l = dx_l * math.cos(phi_l) + dy_l * math.sin(phi_l)
    yr_l = -dx_l * math.sin(phi_l) + dy_l * math.cos(phi_l)
    if (xr_l / l_smaj) ** 2 + (yr_l / l_smin) ** 2 > 1.0:
        return False

    # Check outside pupil (with inner inset)
    p_cx = roi.pupil_center_x if roi.pupil_center_x != 0.0 else roi.center_x
    p_cy = roi.pupil_center_y if roi.pupil_center_y != 0.0 else roi.center_y
    p_smaj = roi.pupil_semi_major * (1.0 + roi.inner_inset_frac)
    p_smin = roi.pupil_semi_minor * (1.0 + roi.inner_inset_frac)
    if p_smaj <= 0 or p_smin <= 0:
        return True
    phi_p = math.radians(roi.pupil_angle_deg)
    dx_p = x - p_cx
    dy_p = y - p_cy
    xr_p = dx_p * math.cos(phi_p) + dy_p * math.sin(phi_p)
    yr_p = -dx_p * math.sin(phi_p) + dy_p * math.cos(phi_p)
    if (xr_p / p_smaj) ** 2 + (yr_p / p_smin) ** 2 < 1.0:
        return False

    return True


def sample_annulus_mask(shape, roi: IrisROI) -> np.ndarray:
    """Return a boolean (H, W) mask marking iris-annulus pixels.

    The annulus is defined by the actual pupil and limbus ellipse boundaries,
    evaluated in each ellipse's respective coordinate frame with insets applied.
    This guarantees exact handling of non-concentric and rotated geometry with
    zero leakage into the sclera or inner pupil.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if not roi.valid:
        return mask

    l_cx = roi.center_x
    l_cy = roi.center_y
    l_smaj = max(roi.limbus_semi_major * (1.0 - roi.outer_inset_frac), 1e-6)
    l_smin = max(roi.limbus_semi_minor * (1.0 - roi.outer_inset_frac), 1e-6)

    # Restrict computation to bounding box of the limbus ellipse
    max_r = max(roi.limbus_semi_major, roi.limbus_semi_minor) + 2.0
    x0 = max(0, int(np.floor(l_cx - max_r)))
    x1 = min(w, int(np.ceil(l_cx + max_r)) + 1)
    y0 = max(0, int(np.floor(l_cy - max_r)))
    y1 = min(h, int(np.ceil(l_cy + max_r)) + 1)

    if x1 <= x0 or y1 <= y0:
        return mask

    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = xx.astype(np.float64) + 0.5
    py = yy.astype(np.float64) + 0.5

    # Limbus boundary: strictly inside scaled limbus ellipse
    phi_l = np.radians(roi.limbus_angle_deg)
    cos_l, sin_l = np.cos(phi_l), np.sin(phi_l)
    dx_l = px - l_cx
    dy_l = py - l_cy
    xr_l = dx_l * cos_l + dy_l * sin_l
    yr_l = -dx_l * sin_l + dy_l * cos_l
    inside_limbus = ((xr_l / l_smaj) ** 2 + (yr_l / l_smin) ** 2) <= 1.0

    # Pupil boundary: strictly outside scaled pupil ellipse
    p_cx = roi.pupil_center_x if roi.pupil_center_x != 0.0 else roi.center_x
    p_cy = roi.pupil_center_y if roi.pupil_center_y != 0.0 else roi.center_y
    p_smaj = max(roi.pupil_semi_major * (1.0 + roi.inner_inset_frac), 1e-6)
    p_smin = max(roi.pupil_semi_minor * (1.0 + roi.inner_inset_frac), 1e-6)
    phi_p = np.radians(roi.pupil_angle_deg)
    cos_p, sin_p = np.cos(phi_p), np.sin(phi_p)
    dx_p = px - p_cx
    dy_p = py - p_cy
    xr_p = dx_p * cos_p + dy_p * sin_p
    yr_p = -dx_p * sin_p + dy_p * cos_p
    outside_pupil = ((xr_p / p_smaj) ** 2 + (yr_p / p_smin) ** 2) >= 1.0

    mask[y0:y1, x0:x1] = (inside_limbus & outside_pupil)
    return mask
