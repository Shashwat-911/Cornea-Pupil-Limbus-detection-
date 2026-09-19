"""Classical iris feature extraction (Phase I baseline).

The extractor samples candidate locations on an iris-relative lattice (angle x
    normalized radial distance), then accepts only candidates that:

    * lie inside the usable iris mask (not occluded / not reflective),
    * are not too close to the pupil or limbus boundary,
    * have local textural response above ``min_contrast`` (i.e. are not flat
      iris),
    * are not on a long straight printed accessory line (e.g. the docked
      ring's quadrant crosshair) when ``line_suppression_threshold`` > 0, and
    * are spatially separated (angular suppression) so features are
      distributed rather than clumped.

Every accepted feature is assigned a deterministic, patch-based descriptor and
a composite confidence. The pipeline is fully classical and CPU-friendly; it
adds no neural-network or heavyweight dependency. It is deterministic: the same
input yields the same output. No matching/registration logic is included.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import (
    IrisFeature,
    IrisFeatureType,
    IrisFeatureSet,
    IrisROI,
)
from pupil_tracking.utils.types import EllipseParams


def _safe_patch(gray: np.ndarray, x: float, y: float, radius: int) -> np.ndarray:
    """Extract a square grayscale patch centred at (x, y), clamped to bounds.

    Returns a float32 patch of size (2*radius+1) x (2*radius+1).
    """
    h, w = gray.shape[:2]
    r = int(radius)
    x0 = int(round(x))
    y0 = int(round(y))

    # Build an index range and crop the source accordingly.
    y_start = max(0, y0 - r)
    y_end = min(h, y0 + r + 1)
    x_start = max(0, x0 - r)
    x_end = min(w, x0 + r + 1)
    patch = gray[y_start:y_end, x_start:x_end]
    if patch.size == 0:
        return np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)

    # Pad to full size with the edge value so off-image samples are stable.
    top = r - (y0 - y_start)
    bottom = (y0 + r + 1) - y_end
    left = r - (x0 - x_start)
    right = (x0 + r + 1) - x_end
    patch = cv2.copyMakeBorder(
        patch, max(0, top), max(0, bottom), max(0, left), max(0, right),
        cv2.BORDER_REPLICATE,
    )
    return patch.astype(np.float32, copy=False)


class IrisFeatureExtractor:
    """Classical iris feature extractor.

    Parameters
    ----------
    num_angles : int
        Angular lattice resolution.
    num_radii : int
        Radial lattice resolution.
    radius_px : int
        Patch radius (px) used for texture response and the descriptor.
    min_contrast : float
        Minimum mean patch texture-energy for a candidate to be accepted.
    max_features : int
        Hard cap on accepted features (quality sorted).
    min_angular_sep_deg : float
        Enforce roughly this angular separation between accepted features.
    min_patch_valid_fraction : float
        Fraction of the feature's local patch that must lie inside the valid
        iris mask for the feature to be accepted (default 0.7).
    use_roi_percentiles : bool
        When True, intensity acceptance uses the ROI-derived percentile span
        instead of a fixed global band; this adapts to ELITA RGB vs Pentacam
        illumination.
    roi_p05 / roi_p95 : float
        Percentile ranks used to derive the adaptive intensity span when
        ``use_roi_percentiles``.
    intensity_low_frac : float
        Acceptable patch-mean intensity range expressed as a fraction of the
        ROI p05..p95 span (adaptive), so it is never a hard-coded universal
        intensity band.
    line_suppression_threshold : float
        Anisotropy above which a candidate is rejected as a long straight
        printed accessory line (e.g. the docked ring's quadrant crosshair).
        Only active when > 0.0.
    line_suppression_window_r : int
        Window radius (px) used for the line-suppression anisotropy test. It
        must be larger than ``radius_px`` so short curved furrows lose their
        anisotropy while long straight printed lines keep it ~1.0. Empirically
        needs to be ~3x ``radius_px`` (e.g. 15 for radius_px=5) so the printed
        ring lines become clearly anisotropic.
    """

    def __init__(
        self,
        num_angles: int = 72,
        num_radii: int = 8,
        radius_px: int = 5,
        min_contrast: float = 4.0,
        texture_floor: float = 2.5,
        texture_rel_frac: float = 0.5,
        max_features: int = 120,
        min_angular_sep_deg: float = 5.0,
        min_patch_valid_fraction: float = 0.7,
        use_roi_percentiles: bool = True,
        roi_p05: float = 0.05,
        roi_p95: float = 0.95,
        intensity_low_frac: float = 0.30,
        intensity_high_frac: float = 0.80,
        line_suppression_threshold: float = 0.6,
        line_suppression_window_r: int = 15,
    ) -> None:
        self.num_angles = int(num_angles)
        self.num_radii = int(num_radii)
        self.radius_px = int(radius_px)
        self.min_contrast = float(min_contrast)
        self.texture_floor = float(texture_floor)
        self.texture_rel_frac = float(texture_rel_frac)
        self.max_features = int(max_features)
        self.min_angular_sep_deg = float(min_angular_sep_deg)
        self.min_patch_valid_fraction = float(min_patch_valid_fraction)
        self.use_roi_percentiles = bool(use_roi_percentiles)
        self.roi_p05 = float(roi_p05)
        self.roi_p95 = float(roi_p95)
        self.intensity_low_frac = float(intensity_low_frac)
        self.intensity_high_frac = float(intensity_high_frac)
        self.line_suppression_threshold = float(line_suppression_threshold)
        self.line_suppression_window_r = int(line_suppression_window_r)
        self._normalizer = IrisNormalizer()

    def _local_measures(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
    ) -> Tuple[float, float, float, float]:
        """Return (mean_intensity, local_contrast, response, patch_std).

        ``local_contrast`` is the difference between the local mean intensity
        and the mean intensity of the surrounding annulus, normalised to a
        (0, ~1) scale by the dynamic range of the local window.
        """
        patch_full = _safe_patch(gray, x, y, self.radius_px)
        if patch_full.size == 0:
            return 0.0, 0.0, 0.0, 0.0

        center = patch_full[
            self.radius_px - 1:self.radius_px + 2,
            self.radius_px - 1:self.radius_px + 2,
        ]
        mean_c = float(center.mean())
        patch_std = float(patch_full.std())

        # Surrounding ring mean = outer part of the patch.
        outer = patch_full.copy()
        outer[self.radius_px - 1:self.radius_px + 2,
              self.radius_px - 1:self.radius_px + 2] = -1
        outer_vals = outer[outer >= 0]
        mean_s = float(outer_vals.mean()) if outer_vals.size else mean_c

        # Contrast relative to the local dynamic range.
        local_range = max(float(patch_full.max()) - float(patch_full.min()), 1e-6)
        local_contrast = abs(mean_c - mean_s) / local_range

        # Texture response: mean absolute Laplacian of the patch.
        lap = np.abs(
            cv2.Laplacian(
                patch_full,
                cv2.CV_32F,
            )
        )
        response = float(lap.mean())

        return mean_c, local_contrast, response, patch_std

    def _anchored_tensor(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
        radius: int,
    ) -> Tuple[float, float]:
        """Structure-tensor (anisotropy, gradient-orientation) of a window.

        Returns (anisotropy, theta) where anisotropy is 1 - lmin/lmax from the
        eigenvalues of the 2x2 structure tensor, and theta is the dominant
        gradient orientation in radians. ``radius`` is the window radius in px.
        """
        patch = _safe_patch(gray, x, y, radius)
        if patch.size == 0:
            return 0.0, 0.0
        patch = cv2.GaussianBlur(patch, (3, 3), 0.8)
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
        gx = np.asarray(gx, dtype=np.float64)
        gy = np.asarray(gy, dtype=np.float64)
        sxx = float(np.sum(gx * gx))
        syy = float(np.sum(gy * gy))
        sxy = float(np.sum(gx * gy))
        tr = sxx + syy
        if tr <= 0.0:
            return 0.0, 0.0
        disc = np.sqrt(max(0.0, tr * tr * 0.25 - (sxx * syy - sxy * sxy)))
        lmax = 0.5 * tr + disc
        if lmax <= 0.0:
            return 0.0, 0.0
        lmin = max(0.5 * tr - disc, 0.0)
        aniso = float(1.0 - lmin / lmax)
        theta = float(0.5 * np.arctan2(2.0 * sxy, sxx - syy))
        return aniso, theta

    def _anchored_anisotropy(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
        radius: int,
    ) -> float:
        """Anisotropy of a centred window (see :meth:`_anchored_tensor`)."""
        return self._anchored_tensor(gray, x, y, radius)[0]

    def _line_direction(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
        radius: int,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Principal line direction at (x, y).

        Returns (anisotropy, theta) only when the window is clearly
        anisotropic; theta is the dominant gradient orientation, so the line
        itself runs perpendicular. ``radius`` is the window radius in px.
        """
        aniso, theta = self._anchored_tensor(gray, x, y, radius)
        if aniso < 0.6:
            return None, None
        return aniso, theta

    def _is_straight_line(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
        radius: int,
    ) -> bool:
        """True when (x, y) lies on a long, straight, high-contrast line.

        A printed accessory line (e.g. the docked ring's quadrant crosshair)
        is long, near-perfectly straight, and high-contrast: its dominant
        orientation is stable both within the centre window and at several
        probe points offset along the line direction. A curved iris ridge or
        furrow is indistinguishable from a straight line over one small window
        (locally every curve looks straight), but its tangent direction
        rotates steadily with distance from the centre point -- far probes
        therefore disagree with the centre orientation, while for a genuinely
        straight line all probes agree. We require:
          * the centre window is strongly anisotropic (>= threshold), and
          * a contiguous run of >= 2 agreeing probes along the line direction
            on one side, reaching a distance of at least the second probe
            (18 px); probes that land on flat/no-texture patches or disagree
            terminate the run. A true printed line stays aligned across the
            whole run (0.2 deg drift over 24 px here); a circular ridge curve
            rotates by several degrees even at the first probe distance.
        """
        aniso, theta = self._line_direction(gray, x, y, radius)
        if aniso is None or aniso < self.line_suppression_threshold:
            return False

        lx = -np.sin(theta)
        ly = np.cos(theta)
        probe_d = (12, 18, 24)
        tol = np.deg2rad(5.0)
        best_run = 0
        for sign in (-1.0, 1.0):
            run = 0
            for d in probe_d:
                _, t2 = self._line_direction(
                    gray,
                    x + sign * d * lx,
                    y + sign * d * ly,
                    self.radius_px,
                )
                if t2 is None:
                    break  # off-line / flat: end of the aligned run
                gap = abs(t2 - theta) % np.pi
                if min(gap, np.pi - gap) <= tol:
                    run += 1
                else:
                    break
            best_run = max(best_run, run)
        return best_run >= 2

    def _suppressed_by_printed_line(
        self,
        gray: np.ndarray,
        x: float,
        y: float,
    ) -> bool:
        """True when a candidate sits on a long straight printed line.

        A printed accessory line (e.g. the docked ring's quadrant crosshair)
        is a long, near-perfectly straight, high-contrast mark over the iris.
        Locally it looks like a high-anisotropy ridge, but so does a curved
        iris furrow; the discriminant is straightness over distance -- the
        dominant orientation must stay aligned at probe points offset along
        the line (see :meth:`_is_straight_line`). Run only when the threshold
        is enabled.
        """
        if self.line_suppression_threshold <= 0.0:
            return False
        return self._is_straight_line(
            gray, x, y, self.line_suppression_window_r
        )

    def _classify(self, patch: np.ndarray) -> IrisFeatureType:
        """Coarse anatomical category from a patch (heuristic, honest).

        - If the central region is substantially darker than its surround and
          the patch is fairly isotropic -> CRYPT.
        - If the patch is strongly anisotropic (elongated) -> FURROW.
        - Otherwise -> TEXTURE.
        """
        # Centre window is the middle (2*radius_px+1) patch pixel, matching the
        # indexing used in _local_measures (not a hard-coded top-left offset).
        c0 = self.radius_px - 1
        c1 = self.radius_px + 2
        center = patch[c0:c1, c0:c1]
        c_mean = float(center.mean())
        surround = patch.copy()
        surround[c0:c1, c0:c1] = -1
        s_vals = surround[surround >= 0]
        s_mean = float(s_vals.mean()) if s_vals.size else c_mean

        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
        sx = float(np.sum(gx * gx))
        sy = float(np.sum(gy * gy))
        anisotropy = 1.0 - min(sx, sy) / (max(sx, sy) + 1e-6)

        if anisotropy < 0.4 and (c_mean - s_mean) < -5.0:
            return IrisFeatureType.CRYPT
        if anisotropy > 0.7 and (sx + sy) > 0.0:
            return IrisFeatureType.FURROW
        return IrisFeatureType.TEXTURE

    def _descriptor(self, gray: np.ndarray, x: float, y: float) -> np.ndarray:
        """Deterministic patch descriptor: normalised intensity histogram.

        A 16-bin histogram of the patch intensities, normalised to unit L1
        norm. This is illumination-roughness-invariant enough for a concept
        model and is fully deterministic.
        """
        patch = _safe_patch(gray, x, y, self.radius_px)
        if patch.size == 0:
            return np.zeros(16, dtype=np.float32)
        hist, _ = np.histogram(
            patch.ravel(), bins=16, range=(0.0, 255.0)
        )
        hist = hist.astype(np.float32)
        n = hist.sum()
        if n > 0:
            hist = hist / n
        return hist

    def _local_visibility(
        self,
        usable_mask: np.ndarray,
        x: float,
        y: float,
    ) -> float:
        """Fraction of usable (non-occluded / non-reflective) pixels in the
        feature's local patch neighbourhood, in ``[0, 1]``.

        The feature's own center pixel is guaranteed usable by the extraction
        gate, but its surrounding patch may be partially occluded or specular;
        this reports how much of that patch is actually visible.
        """
        h, w = usable_mask.shape[:2]
        x0 = int(round(x))
        y0 = int(round(y))
        r = int(self.radius_px)
        ys = max(0, y0 - r)
        ye = min(h, y0 + r + 1)
        xs = max(0, x0 - r)
        xe = min(w, x0 + r + 1)
        if ys >= ye or xs >= xe:
            return 1.0
        patch = usable_mask[ys:ye, xs:xe]
        total = patch.size
        if total == 0:
            return 1.0
        return float(np.count_nonzero(patch) / total)

    def _patch_valid_fraction(
        self,
        usable_mask: np.ndarray,
        x: float,
        y: float,
    ) -> float:
        """Fraction of the feature's local patch that lies in the valid mask.

        Validates the *support region*, not just the centre pixel, so a feature
        whose patch straddles an occlusion / reflection / sclera / pupil edge
        is rejected.
        """
        return self._local_visibility(usable_mask, x, y)

    def _intensity_range_ok(
        self,
        patch_mean: float,
        roi_stats: dict,
    ) -> bool:
        """Check a patch's mean intensity against the ROI-derived range.

        When ``use_roi_percentiles`` the bounds are a fraction of the ROI
        p05..p95 span, so they adapt to the actual frame illumination.  The
        patch mean must fall within ``[p05 + low_frac*span, p05 + high_frac*span]``.
        Without percentiles, a permissive fixed band is used.
        """
        if not self.use_roi_percentiles:
            return 0.0 <= patch_mean <= 255.0
        p05 = roi_stats.get("intensity_p05", 0.0)
        p95 = roi_stats.get("intensity_p95", 0.0)
        span = max(p95 - p05, 1e-6)
        lo = p05 + self.intensity_low_frac * span
        hi = p05 + self.intensity_high_frac * span
        return lo <= patch_mean <= hi

    def extract(
        self,
        image: np.ndarray,
        roi: IrisROI,
        usable_mask: np.ndarray,
        *,
        pupil: Optional[EllipseParams] = None,
        limbus: Optional[EllipseParams] = None,
        roi_stats: Optional[dict] = None,
    ) -> IrisFeatureSet:
        """Extract iris features from an image given the ROI and usable mask.

        ``image`` may be BGR (H, W, 3) or grayscale (H, W); luminance is used.
        ``usable_mask`` is the boolean mask produced by ``IrisMasking``.
        ``roi_stats`` optionally carries the isolated-ROI intensity/texture
        statistics used for adaptive acceptance; when absent a fixed band is
        assumed.
        """
        h, w = roi_valid_shape(image, roi)
        if not roi.valid or h is None:
            return IrisFeatureSet(roi=roi)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.astype(np.float32, copy=False)

        candidates: List[IrisFeature] = []
        rejection_reasons: Counter[str] = Counter()

        for ai in range(self.num_angles):
            angle_deg = (360.0 * ai) / self.num_angles
            inner, outer = self._normalizer.radial_bounds(roi, angle_deg)

            for ri in range(1, self.num_radii + 1):
                radial_norm = ri / (self.num_radii + 1.0)
                radius = inner + radial_norm * (outer - inner)

                ang = math.radians(angle_deg)
                x = roi.center_x + radius * math.cos(ang)
                y = roi.center_y + radius * math.sin(ang)

                xi = int(round(x))
                yi = int(round(y))
                if xi < 0 or yi < 0 or xi >= w or yi >= h:
                    continue
                if not usable_mask[yi, xi]:
                    rejection_reasons["outside_valid_mask"] += 1
                    continue

                mean_c, local_contrast, response, _ = self._local_measures(
                    gray, x, y
                )
                # Adaptive texture gate: relative to ROI texture level with an
                # absolute floor.  The floor protects against flat/noise iris
                # (texture ~0) and degenerate synthetic test fixtures; the
                # relative term normalises for camera gain / illumination
                # (Daugman 1993: raw amplitude is gain-sensitive).
                roi_tex = (roi_stats or {}).get("texture_response_mean", 0.0)
                adaptive_min = max(self.texture_floor, self.texture_rel_frac * roi_tex)
                if response < adaptive_min:
                    rejection_reasons["low_texture"] += 1
                    continue

                if self._suppressed_by_printed_line(gray, x, y):
                    rejection_reasons["printed_line"] += 1
                    continue

                patch = _safe_patch(gray, x, y, self.radius_px)
                feat_type = self._classify(patch)

                if not self._intensity_range_ok(mean_c, roi_stats or {}):
                    rejection_reasons["low_contrast"] += 1
                    continue
                pvf = self._patch_valid_fraction(usable_mask, x, y)
                if pvf < self.min_patch_valid_fraction:
                    rejection_reasons["patch_outside_iris"] += 1
                    continue

                descriptor = self._descriptor(gray, x, y)
                clearance = min(radial_norm, 1.0 - radial_norm) * 2.0
                confidence = self._confidence(
                    response, local_contrast, clearance
                )

                candidates.append(
                    IrisFeature(
                        id=len(candidates),
                        x=float(x),
                        y=float(y),
                        angle_deg=angle_deg,
                        radial_norm=float(radial_norm),
                        scale=float(radius_px_to_scale(self.radius_px, outer)),
                        orientation_deg=angle_deg,
                        feature_type=feat_type,
                        response=float(response),
                        local_contrast=float(local_contrast),
                        visibility=self._local_visibility(usable_mask, x, y),
                        confidence=float(confidence),
                        valid=True,
                        descriptor=descriptor,
                    )
                )

        return self._filter(roi, candidates, rejection_reasons)

    def _confidence(self, response, local_contrast, clearance):
        """Blend texture, contrast and boundary-clearance into [0, 1]."""
        resp = min(response / (self.min_contrast * 4.0 + 1e-6), 1.0)
        return float(np.clip(0.5 * resp + 0.3 * local_contrast + 0.2 * clearance, 0.0, 1.0))

    def _filter(
        self,
        roi: IrisROI,
        candidates: Sequence[IrisFeature],
        rejection_reasons: Optional[Counter[str]] = None,
    ) -> IrisFeatureSet:
        """Angular suppression + quality sort + cap, and set ROI/coverage."""
        rejection_reasons = rejection_reasons or Counter()
        n_cand = len(candidates)
        if n_cand == 0:
            return IrisFeatureSet(
                roi=roi,
                num_candidates=0,
                num_accepted=0,
                rejection_reasons=dict(rejection_reasons),
            )

        # Sort by confidence descending, then greedily accept while enforcing
        # a minimum angular separation (deterministic).
        ordered = sorted(candidates, key=lambda f: (-f.confidence, f.x, f.y))
        accepted: List[IrisFeature] = []
        for feat in ordered:
            if len(accepted) >= self.max_features:
                rejection_reasons["max_features_cap"] += 1
                continue
            if all(
                self._angular_gap(feat.angle_deg, a.angle_deg)
                >= self.min_angular_sep_deg
                for a in accepted
            ):
                accepted.append(feat)
            else:
                rejection_reasons["angular_suppression"] += 1

        accepted.sort(key=lambda f: f.angle_deg)
        for i, f in enumerate(accepted):
            f.id = i

        return IrisFeatureSet(
            roi=roi,
            features=list(accepted),
            num_candidates=n_cand,
            num_accepted=len(accepted),
            rejection_reasons=dict(rejection_reasons),
        )

    @staticmethod
    def _angular_gap(a: float, b: float) -> float:
        gap = abs(a - b) % 360.0
        return min(gap, 360.0 - gap)


def radius_px_to_scale(radius_px: int, outer_radius: float) -> float:
    """Characteristic scale of a feature relative to the iris outer radius."""
    if outer_radius <= 0:
        return float(radius_px)
    return float(radius_px) / outer_radius


def roi_valid_shape(image: np.ndarray, roi: IrisROI):
    """Return (h, w) if the image and ROI are usable, else (None, None)."""
    if image is None or image.size == 0 or not roi.valid:
        return None, None
    h, w = image.shape[:2]
    if w < 2 or h < 2:
        return None, None
    return h, w
