"""Occlusion / validity masking for the iris region.

This module builds the *valid iris mask*:

    valid_iris_mask =
        annulus_mask
        & ~eyelid_mask
        & ~eyelash_mask
        & ~reflection_mask
        & ~saturated_mask
        & ~low_snr_mask

Each component is derived where the architecture supports it.  Incomplete
masks are made explicit (an unavailable mask is simply not applied) rather
than fabricated.  The result is a boolean usable-pixel mask that the feature
extractor must respect both at feature centres and across feature patches.

Reflection detection reuses the existing ``ReflectionRemover`` so we do not
introduce a redundant reflection implementation.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from pupil_tracking.iris.roi import sample_annulus_mask
from pupil_tracking.iris.types import IrisROI
from pupil_tracking.preprocessing.reflection_removal import ReflectionRemover


class IrisMasking:
    """Build a usable-pixels mask for the iris annulus.

    Parameters
    ----------
    reflection_remover : ReflectionRemover | None
        Existing reflection detector. If None, a default ``ReflectionRemover``
        is constructed (using the existing implementation).
    only_within_roi : bool
        If True, reflections outside the annulus are ignored (they do not
        affect the iris region).
    eyelid_method : str
        "gradient" enables horizontal-edge eyelid/eyelash detection inside the
        annulus; "none" disables it (e.g. when geometry/lighting is trusted).
    eyelid_edge_threshold : float
        Sobel horizontal gradient magnitude above which a pixel is an eyelid /
        eyelash candidate (eyelids have sharp vertical edges).
    eyelid_dilate_px : int
        Dilate the eyelid mask this many pixels to cover the eyelid margin.
    saturation_threshold : int
        Pixels at/above this brightness are treated as saturated/overexposed
        and excluded.
    suppress_blue_ring_lines : bool
        Erase thin saturated-blue streaks (printed docking-ring quadrant
        marks) from the usable mask so the extractor cannot report features
        on them. A broadly blue iris is preserved (see
        :meth:`_blue_ring_mask`).
    blue_line_blueness_threshold : int
        Minimum ``B - max(R, G)`` for a pixel to count as blue.
    blue_line_min_length_px : int
        Morphological opening size used to drop isolated blue specks; streaks
        thinner than this (in px) are not treated as printed lines.
    blue_line_max_local_density : float
        A blue pixel is erased only when its local blue density is at or below
        this fraction, i.e. it belongs to a thin streak rather than a solid
        blue region (a genuinely blue iris).
    blue_line_dilate_px : int
        Erase a margin of this many pixels around the printed-blue line so
        feature patches straddling the line edge are not accepted either.
    """

    def __init__(
        self,
        reflection_remover: Optional[ReflectionRemover] = None,
        only_within_roi: bool = True,
        eyelid_method: str = "gradient",
        eyelid_edge_threshold: float = 60.0,
        eyelid_dilate_px: int = 6,
        saturation_threshold: int = 250,
        suppress_blue_ring_lines: bool = True,
        blue_line_blueness_threshold: int = 45,
        blue_line_min_length_px: int = 3,
        blue_line_max_local_density: float = 0.7,
        blue_line_dilate_px: int = 4,
    ) -> None:
        self.reflection_remover = reflection_remover or ReflectionRemover()
        self.only_within_roi = only_within_roi
        self.eyelid_method = eyelid_method
        self.eyelid_edge_threshold = float(eyelid_edge_threshold)
        self.eyelid_dilate_px = int(eyelid_dilate_px)
        self.saturation_threshold = int(saturation_threshold)
        self.suppress_blue_ring_lines = bool(suppress_blue_ring_lines)
        self.blue_line_blueness_threshold = int(blue_line_blueness_threshold)
        self.blue_line_min_length_px = int(blue_line_min_length_px)
        self.blue_line_max_local_density = float(blue_line_max_local_density)
        self.blue_line_dilate_px = int(blue_line_dilate_px)

    def build(
        self,
        image_bgr: np.ndarray,
        roi: IrisROI,
        *,
        external_occlusion: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a boolean (H, W) mask of *usable* iris pixels.

        A pixel is usable when it lies inside the annulus, is not a specular
        reflection, not saturated/overexposed, not an eyelid/eyelash, and not
        excluded by an external occlusion mask.

        Parameters
        ----------
        image_bgr : np.ndarray  (H, W, 3) BGR image
        roi : IrisROI
        external_occlusion : np.ndarray | None
            Optional boolean (H, W) mask where True = occluded (e.g. eyelids).
        """
        if image_bgr is None:
            return np.zeros((0, 0), dtype=bool)
        h, w = image_bgr.shape[:2]

        annulus = sample_annulus_mask((h, w), roi)

        # Reflection detection, restricted to the annulus when requested.
        grain, reflection = self._reflection_mask(image_bgr, annulus)
        del grain
        reflection = reflection > 0

        usable = annulus & ~reflection

        # Saturation / overexposure exclusion.
        gray = self._as_gray(image_bgr)
        saturated = gray >= self.saturation_threshold
        usable = usable & ~saturated

        # Eyelid / eyelash exclusion.
        if self.eyelid_method == "gradient":
            eyelid = self._eyelid_mask(gray, annulus)
            usable = usable & ~eyelid

        # Printed blue docking-ring marks overlay the iris annulus (their
        # quadrant crosshair). Erase them so the extractor cannot report
        # features on them.
        if self.suppress_blue_ring_lines:
            blue_ring = self._blue_ring_mask(image_bgr, annulus)
            usable = usable & ~blue_ring

        if external_occlusion is not None and external_occlusion.shape == (h, w):
            usable = usable & ~external_occlusion.astype(bool)

        return usable

    # ---- component constructors ----------------------------------------- #

    def _as_gray(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr.ndim == 3:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return image_bgr.astype(np.uint8, copy=False)

    def _blue_ring_mask(
        self,
        image_bgr: np.ndarray,
        annulus: np.ndarray,
    ) -> np.ndarray:
        """Mask of thin, saturated-blue streaks (printed docking-ring marks).

        The suction-docking ring prints thin saturated-blue quadrant marks
        that overlay the iris annulus. Their pixels are returned as True so
        they are excluded from the usable mask.

        Discriminant: a printed blue line is strongly blue relative to its
        full-colour surround, and sits as a *thin streak* (low local blue
        density). A broadly blue iris -- where every pixel is blue -- has high
        local blue density and is therefore preserved even though its
        individual pixels pass the blueness test.

        Notes
        -----
        The blueness and local-density tests are computed on the *full* blue
        mask (not pre-restricted to ``annulus``): computing density inside the
        annulus band would push the local density below threshold near the
        inner/outer annulus borders, where the neighbourhood window pokes into
        the non-blue pupil or background, erasing genuine iris. The final
        streak mask is intersected with the annulus afterwards.

        Returns
        -------
        np.ndarray (H, W) bool
            True where a printed-blue-line pixel sits inside the annulus.
        """
        if image_bgr.ndim != 3:
            return np.zeros(annulus.shape, dtype=bool)

        b = image_bgr[..., 0].astype(np.int16)
        g = image_bgr[..., 1].astype(np.int16)
        r = image_bgr[..., 2].astype(np.int16)
        blueness = b - np.maximum(g, r)

        blue = blueness >= self.blue_line_blueness_threshold

        # Drop isolated specks / single blue flecks, then relax back so the
        # eroded part of a genuine streak is recovered.
        if self.blue_line_min_length_px > 1:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)
            )
            blue = cv2.morphologyEx(
                blue.astype(np.uint8), cv2.MORPH_OPEN, k
            ).astype(bool)

        # Thin-streak test: local blue density must be low enough that the
        # blue pixels belong to a line, not a blue region.
        if self.blue_line_max_local_density < 1.0:
            win = max(7, 3 * max(self.blue_line_min_length_px, 1))
            density = cv2.boxFilter(
                blue.astype(np.float32), -1, (win, win), normalize=True
            )
            blue = blue & (density <= self.blue_line_max_local_density)

        # Erase a margin around the streak too, so feature patches straddling
        # the line's edge cannot be accepted on the blue/iris contrast edge.
        if self.blue_line_dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.blue_line_dilate_px + 1,
                 2 * self.blue_line_dilate_px + 1),
            )
            blue = cv2.dilate(blue.astype(np.uint8), k).astype(bool)

        return blue & annulus

    def _reflection_mask(
        self, image_bgr: np.ndarray, annulus: np.ndarray
    ) -> tuple:
        # Configure reflection detector with our thresholds.
        remover = ReflectionRemover(
            brightness_threshold=self.reflection_remover.brightness_threshold,
            saturation_threshold=self.reflection_remover.saturation_threshold,
            min_reflection_area=self.reflection_remover.min_reflection_area,
            max_reflection_area_frac=self.reflection_remover.max_reflection_area_frac,
            dilation_size=self.reflection_remover.dilation_size,
            inpaint_radius=self.reflection_remover.inpaint_radius,
        )
        if self.only_within_roi:
            roi_255 = annulus.astype(np.uint8) * 255
            _, reflection = remover.remove(image_bgr, roi_mask=roi_255)
        else:
            _, reflection = remover.remove(image_bgr)
        return None, reflection

    def _eyelid_mask(self, gray: np.ndarray, annulus: np.ndarray) -> np.ndarray:
        """Detect eyelid/eyelash occluding pixels inside the annulus.

        Eyelids sweep across the top/bottom of the iris as horizontally
        oriented edges, so a strong horizontal Sobel gradient marks them.
        The mask is restricted to the annulus and dilated to cover the eyelid
        margin.  This is a heuristic; it cannot distinguish a lower eyelid
        from an upper one but is adequate for masking.
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        mag = np.abs(sobel_x)
        edge = mag >= self.eyelid_edge_threshold
        edge = edge & annulus
        if self.eyelid_dilate_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.eyelid_dilate_px + 1, 2 * self.eyelid_dilate_px + 1),
            )
            edge = cv2.dilate(
                edge.astype(np.uint8), k, iterations=1
            ).astype(bool) & annulus
        return edge


def roi_iris_stats(image_bgr: np.ndarray, usable: np.ndarray, roi: IrisROI) -> dict:
    """Compute intensity / texture statistics from the ISOLATED iris ROI.

    The stats are computed only over the usable iris pixels, NOT the whole
    camera frame, so thresholds derived from them reflect actual iris tissue
    rather than background / UI / Pentacam chrome.  Values:

      intensity_p05 / p50 / p95  : percentile of usable-iris grayscale
      mean_intensity / std_intensity
      local_contrast_mean        : mean |center - surround| over the ROI
      texture_response_mean      : mean |Laplacian| over the ROI
      valid_iris_pixels / valid_iris_area_frac
      saturation_px / saturation_fraction
    """
    if (
        image_bgr is None or usable is None or usable.size == 0
        or not roi.valid or not np.any(usable)
    ):
        return {
            "intensity_p05": 0.0, "intensity_p50": 0.0, "intensity_p95": 0.0,
            "mean_intensity": 0.0, "std_intensity": 0.0,
            "local_contrast_mean": 0.0, "texture_response_mean": 0.0,
            "valid_iris_pixels": 0, "valid_iris_area_frac": 0.0,
            "saturation_px": 0, "saturation_fraction": 0.0,
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 \
        else image_bgr.astype(np.float32, copy=False)
    vals = gray[usable]
    p05 = float(np.percentile(vals, 5))
    p50 = float(np.percentile(vals, 50))
    p95 = float(np.percentile(vals, 95))

    n_usable = int(np.count_nonzero(usable))
    annulus_area = np.pi * (roi.limbus_radius_px ** 2 - roi.pupil_radius_px ** 2)
    annulus_area = max(annulus_area, 1.0)

    # Local contrast / texture over the usable iris pixels, vectorised.
    pad = 1
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 2 and g.shape[0] > 2 and g.shape[1] > 2:
        bright = g >= 250
        # 3x3 local patch statistics guarded by a 1px interior margin.
        interior_usable = usable.copy()
        interior_usable[:pad, :] = False
        interior_usable[-pad:, :] = False
        interior_usable[:, :pad] = False
        interior_usable[:, -pad:] = False
        ys, xs = np.nonzero(interior_usable)
        if ys.size:
            # Sample to bound cost on huge frames.
            step = max(1, ys.size // 4000)
            ys = ys[::step]
            xs = xs[::step]
            g_ = g[ys, xs]
            surround = (
                g[ys - 1, xs - 1] + g[ys - 1, xs] + g[ys - 1, xs + 1]
                + g[ys, xs - 1] + g[ys, xs + 1]
                + g[ys + 1, xs - 1] + g[ys + 1, xs] + g[ys + 1, xs + 1]
            ) / 8.0
            center = g_
            diff = np.abs(center - surround)
            local_min = np.minimum.reduce([
                g[ys - 1, xs - 1], g[ys - 1, xs], g[ys - 1, xs + 1],
                g[ys, xs - 1], center, g[ys, xs + 1],
                g[ys + 1, xs - 1], g[ys + 1, xs], g[ys + 1, xs + 1],
            ])
            local_max = np.maximum.reduce([
                g[ys - 1, xs - 1], g[ys - 1, xs], g[ys - 1, xs + 1],
                g[ys, xs - 1], center, g[ys, xs + 1],
                g[ys + 1, xs - 1], g[ys + 1, xs], g[ys + 1, xs + 1],
            ])
            local_range = np.maximum(local_max - local_min, 1e-6)
            contrast = np.mean(diff / local_range)
            # Laplacian via 3x3 kernel (approx) for texture response.
            lap = np.abs(
                4.0 * center
                - (g[ys - 1, xs] + g[ys + 1, xs] + g[ys, xs - 1] + g[ys, xs + 1])
            )
            texture = float(np.mean(lap))
            local_contrast_mean = float(contrast)
        else:
            texture = 0.0
            local_contrast_mean = 0.0
        sat_pixels = int(np.count_nonzero(bright[usable]))
    else:
        local_contrast_mean = 0.0
        texture = 0.0
        sat_pixels = int(np.count_nonzero(vals >= 250))

    return {
        "intensity_p05": p05,
        "intensity_p50": p50,
        "intensity_p95": p95,
        "mean_intensity": float(vals.mean()),
        "std_intensity": float(vals.std()),
        "local_contrast_mean": float(local_contrast_mean),
        "texture_response_mean": float(texture),
        "valid_iris_pixels": n_usable,
        "valid_iris_area_frac": float(n_usable / annulus_area),
        "saturation_px": sat_pixels,
        "saturation_fraction": float(sat_pixels / vals.size) if vals.size else 0.0,
    }


def mask_stats(usable: np.ndarray, roi: IrisROI) -> dict:
    """Return compact statistics about the usable mask.

    Values are reported honestly: when the mask is empty or the ROI invalid,
    the fractions are 0 rather than inflated.
    """
    if usable is None or usable.size == 0 or not roi.valid:
        return {
            "usable_iris_pixels": 0,
            "usable_fraction": 0.0,
            "annulus_area_px": 0.0,
            "reflection_estimated": False,
        }
    n_usable = int(np.count_nonzero(usable))
    annulus_area = (
        np.pi * (roi.limbus_radius_px ** 2 - roi.pupil_radius_px ** 2)
    )
    annulus_area = max(annulus_area, 1.0)
    return {
        "usable_iris_pixels": n_usable,
        "usable_fraction": float(n_usable / annulus_area),
        "annulus_area_px": float(annulus_area),
        "reflection_estimated": True,
    }
