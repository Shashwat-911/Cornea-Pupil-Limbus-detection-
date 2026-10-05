"""
Polar unwrapping of the iris annulus.

Converts a Cartesian iris image into a polar (angle × radius) representation
suitable for rotation estimation via 1-D shift detection.

The unwrapping maps the annular iris region (between pupil and limbus
boundaries) to a rectangular strip where:
    - Horizontal axis = angle (0–360°)
    - Vertical axis = normalised radial distance (pupil → limbus)

A rotation of the eye in Cartesian space corresponds to a horizontal
(angular) shift in the polar image, which can be measured efficiently
via phase correlation or other 1-D matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.utils.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class PolarImage:
    """Result of polar unwrapping."""
    image: Optional[np.ndarray] = None      # [num_radial, num_angles]
    mask: Optional[np.ndarray] = None       # valid-pixel mask
    center: Tuple[float, float] = (0.0, 0.0)
    inner_radius: float = 0.0
    outer_radius: float = 0.0
    num_angles: int = 360
    num_radial: int = 64
    valid: bool = False


class PolarUnwrapper:
    """Converts an iris annulus to polar coordinates.

    Handles elliptical pupil/limbus boundaries by interpolating
    the radial extent at each angle using the parametric ellipse
    equation.

    Parameters
    ----------
    num_angles : int
        Angular resolution (default from config: 360).
    num_radial : int
        Radial resolution (default from config: 64).
    inner_margin : float
        Fractional inset from pupil boundary to avoid pupil edge.
    outer_margin : float
        Fractional inset from limbus boundary to avoid limbus edge.
    """

    def __init__(
        self,
        num_angles: Optional[int] = None,
        num_radial: Optional[int] = None,
        inner_margin: Optional[float] = None,
        outer_margin: Optional[float] = None,
    ):
        cfg = get_config().registration
        self.num_angles = num_angles or cfg.polar_num_angles
        self.num_radial = num_radial or cfg.polar_num_radial
        self.inner_margin = inner_margin if inner_margin is not None else cfg.polar_inner_margin
        self.outer_margin = outer_margin if outer_margin is not None else cfg.polar_outer_margin

    def unwrap(
        self,
        image: np.ndarray,
        pupil_center: Tuple[float, float],
        pupil_axes: Tuple[float, float],
        pupil_angle_deg: float,
        limbus_center: Tuple[float, float],
        limbus_axes: Tuple[float, float],
        limbus_angle_deg: float,
    ) -> PolarImage:
        """Unwrap the iris annulus into a polar image.

        Parameters
        ----------
        image : np.ndarray
            Source image (grayscale or BGR).
        pupil_center, pupil_axes, pupil_angle_deg
            Pupil ellipse parameters. axes = (semi_major, semi_minor).
        limbus_center, limbus_axes, limbus_angle_deg
            Limbus ellipse parameters.

        Returns
        -------
        PolarImage
            Unwrapped polar image with metadata.
        """
        result = PolarImage(
            center=limbus_center,
            num_angles=self.num_angles,
            num_radial=self.num_radial,
        )

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape[:2]
        angles = np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)
        # Use Daugman rubber-sheet model: interpolate between pupil and limbus margins
        # Parametric representation of pupil ellipse at all angles
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        theta_p = np.radians(pupil_angle_deg)
        cos_tp, sin_tp = np.cos(theta_p), np.sin(theta_p)
        ap, bp = max(pupil_axes[0], 1.0), max(pupil_axes[1], 1.0)
        px = pupil_center[0] + (ap * cos_a * cos_tp - bp * sin_a * sin_tp)
        py = pupil_center[1] + (ap * cos_a * sin_tp + bp * sin_a * cos_tp)

        theta_l = np.radians(limbus_angle_deg)
        cos_tl, sin_tl = np.cos(theta_l), np.sin(theta_l)
        al, bl = max(limbus_axes[0], 1.0), max(limbus_axes[1], 1.0)
        lx = limbus_center[0] + (al * cos_a * cos_tl - bl * sin_a * sin_tl)
        ly = limbus_center[1] + (al * cos_a * sin_tl + bl * sin_a * cos_tl)

        # Normalized radial coordinates with margins
        r_inner = float(np.clip(self.inner_margin, 0.0, 0.4))
        r_outer = float(np.clip(self.outer_margin, 0.0, 0.4))
        r_steps = np.linspace(r_inner, 1.0 - r_outer, self.num_radial, dtype=np.float32)[:, None]

        # Vectorized rubber sheet coordinate grid: (num_radial, num_angles)
        x_coords = ((1.0 - r_steps) * px + r_steps * lx).astype(np.float32)
        y_coords = ((1.0 - r_steps) * py + r_steps * ly).astype(np.float32)

        in_bounds = (
            (x_coords >= 0.0)
            & (x_coords < float(w - 1))
            & (y_coords >= 0.0)
            & (y_coords < float(h - 1))
        )

        polar_img = cv2.remap(
            gray,
            x_coords,
            y_coords,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(np.float32, copy=False)

        # Specular reflection & eyelid occlusion masking
        valid_texture = in_bounds & (polar_img < 250.0) & (polar_img > 5.0)
        polar_mask = valid_texture.astype(np.uint8) * 255

        result.image = polar_img
        result.mask = polar_mask
        result.inner_radius = float(ap)
        result.outer_radius = float(al)
        result.valid = bool(np.mean(valid_texture) > 0.3)

        return result

    def unwrap_from_detection(self, image: np.ndarray, detection) -> PolarImage:
        """Convenience method using an EyeDetectionResult.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        detection : EyeDetectionResult
            Detection result with pupil and limbus ellipses.

        Returns
        -------
        PolarImage
        """
        if not detection.has_both:
            return PolarImage()

        pe = detection.pupil.ellipse
        le = detection.limbus.ellipse

        return self.unwrap(
            image=image,
            pupil_center=(pe.center_x, pe.center_y),
            pupil_axes=(pe.semi_major, pe.semi_minor),
            pupil_angle_deg=pe.angle_deg,
            limbus_center=(le.center_x, le.center_y),
            limbus_axes=(le.semi_major, le.semi_minor),
            limbus_angle_deg=le.angle_deg,
        )

    @staticmethod
    def _ellipse_radii_at_angles(
        ellipse_center: Tuple[float, float],
        ellipse_axes: Tuple[float, float],
        ellipse_angle_deg: float,
        query_center: Tuple[float, float],
        angles: np.ndarray,
    ) -> np.ndarray:
        """Compute distance from query_center to ellipse boundary at each angle.

        Uses the parametric ellipse equation with rotation.

        Parameters
        ----------
        ellipse_center : (float, float)
            Centre of the ellipse.
        ellipse_axes : (float, float)
            (semi_major, semi_minor) of the ellipse.
        ellipse_angle_deg : float
            Rotation angle of the ellipse (degrees).
        query_center : (float, float)
            Centre from which distances are measured.
        angles : np.ndarray
            Array of angles (radians) at which to compute distances.

        Returns
        -------
        np.ndarray
            Array of radial distances.
        """
        a, b = ellipse_axes
        if a <= 0 or b <= 0:
            return np.zeros_like(angles)

        # Offset between centres
        dx = ellipse_center[0] - query_center[0]
        dy = ellipse_center[1] - query_center[1]

        # Rotation of ellipse
        theta = np.radians(ellipse_angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # For each query angle, find the intersection of the ray from
        # query_center with the ellipse.  Use the simplified formula
        # when centres coincide; otherwise offset the parametric form.
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            # Coincident centres — simple formula
            cos_a = np.cos(angles - theta)
            sin_a = np.sin(angles - theta)
            # r(angle) = a*b / sqrt((b*cos)^2 + (a*sin)^2)
            denom = np.sqrt((b * cos_a) ** 2 + (a * sin_a) ** 2)
            denom = np.maximum(denom, 1e-8)
            radii = (a * b) / denom
        else:
            # Non-coincident: approximate by shifting the angle-dependent
            # radius by the centre offset projected onto each ray direction
            cos_a = np.cos(angles - theta)
            sin_a = np.sin(angles - theta)
            denom = np.sqrt((b * cos_a) ** 2 + (a * sin_a) ** 2)
            denom = np.maximum(denom, 1e-8)
            base_radii = (a * b) / denom

            # Project offset onto each ray
            offset_proj = dx * np.cos(angles) + dy * np.sin(angles)
            radii = np.maximum(base_radii + offset_proj, 1.0)

        return radii
