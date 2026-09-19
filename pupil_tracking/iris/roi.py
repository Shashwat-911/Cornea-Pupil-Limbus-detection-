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
    """Return True if a point lies within the iris annulus (ignoring insets).

    Uses a ray-from-centre test: the point is accepted when its distance from
    the centre is between the inner (pupil) and outer (limbus) radii of the
    ellipse along that angle. The inset fractions are applied.
    """
    if not roi.valid:
        return False
    dx = x - roi.center_x
    dy = y - roi.center_y
    angle_deg = math.degrees(math.atan2(dy, dx)) % 360.0

    inner_r = roi.pupil_radius_px * (1.0 + roi.inner_inset_frac)
    outer_r = roi.limbus_radius_px * (1.0 - roi.outer_inset_frac)
    dist = math.hypot(dx, dy)
    return inner_r < dist < outer_r


def sample_annulus_mask(shape, roi: IrisROI) -> np.ndarray:
    """Return a boolean (H, W) mask marking iris-annulus pixels.

    The annulus is defined by the actual pupil and limbus ellipse boundaries
    at each angle, not by a circular mean-radius approximation.  This
    correctly handles non-concentric and non-circular geometry.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if not roi.valid:
        return mask

    yy, xx = np.mgrid[0:h, 0:w]
    dx = (xx - roi.center_x).astype(np.float64)
    dy = (yy - roi.center_y).astype(np.float64)
    dist = np.sqrt(dx * dx + dy * dy)
    angles_rad = np.arctan2(dy, dx)

    def _ellipse_r_at_angles(semi_a, semi_b, ell_angle_deg, angles):
        phi = np.radians(ell_angle_deg)
        diff = angles - phi
        c = np.cos(diff)
        s = np.sin(diff)
        denom = (s / semi_a) ** 2 + (c / semi_b) ** 2
        denom = np.maximum(denom, 1e-12)
        return 1.0 / np.sqrt(denom)

    inner = _ellipse_r_at_angles(
        roi.pupil_semi_major, roi.pupil_semi_minor,
        roi.pupil_angle_deg, angles_rad,
    ) * (1.0 + roi.inner_inset_frac)
    outer = _ellipse_r_at_angles(
        roi.limbus_semi_major, roi.limbus_semi_minor,
        roi.limbus_angle_deg, angles_rad,
    ) * (1.0 - roi.outer_inset_frac)

    mask = (dist >= inner) & (dist <= outer)
    return mask
