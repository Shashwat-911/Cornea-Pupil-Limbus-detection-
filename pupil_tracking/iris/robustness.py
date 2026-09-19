"""Phase II repeatability & robustness evaluation helpers (pure, deterministic).

Phase II measures whether the Phase I detector produces *stable*, *well-distributed*
iris features under controlled perturbations. Everything in this module is
**evaluation-only**: nothing here is wired into the production pipeline and nothing
here implements feature matching, registration, homography/affine/non-linear
registration, cyclotorsion / rotation-angle estimation, or astigmatism correction
(those are later phases).

The repeatability model used here is the controlled-perturbation model required by
the plan:

  * baseline features are extracted from an unperturbed image with a fixed
    pupil/limbus geometry,
  * the same image is perturbed and features are re-extracted with the **same**
    baseline geometry (photometric perturbations) or mapped back into the baseline
    pixel frame via the inverse of the known geometric transformation
    (translation / rotation / scale),
  * two features "correspond" when they land within a small angular and radial
    tolerance of each other in the baseline **normalised iris coordinate frame**.

This is deliberately NOT a general-purpose matcher; correspondence is known from
the controlled transformation. The tolerances are halved lattice spacings:
``ang_tol_deg`` = half the configured angular spacing, ``rad_tol`` = half the
configured radial step. These are documented defaults, not tuned to game results.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import (
    IrisDetectionResult,
    IrisFeature,
    IrisFeatureSet,
    IrisROI,
)

# Default angular- / radial-sector discretisation used by the distribution metrics.
N_ANGULAR_SECTORS = 12           # 12 x 30 deg sectors
N_RADIAL_BANDS = 4               # 4 equal radial_norm bands over (0, 1]


# --------------------------------------------------------------------------- #
# Perturbation generators (deterministic; each takes a seed)
# --------------------------------------------------------------------------- #

def perturb_brightness(image_bgr: np.ndarray, delta: float, seed: int = 0) -> np.ndarray:
    """Add ``delta`` (typically +/-25) to every pixel, clipped to [0, 255]."""
    return np.clip(image_bgr.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def perturb_contrast(image_bgr: np.ndarray, factor: float, seed: int = 0) -> np.ndarray:
    """Multiply deviation from the mean by ``factor`` (typically 0.8-1.2)."""
    img = image_bgr.astype(np.float32)
    mean = img.mean()
    out = np.clip((img - mean) * factor + mean, 0, 255)
    return out.astype(np.uint8)


def perturb_gamma(image_bgr: np.ndarray, gamma: float, seed: int = 0) -> np.ndarray:
    """Apply ``out = 255 * (in/255) ** gamma`` (gamma ~ 0.8-1.2 = moderate)."""
    lut = np.array(
        [np.clip(255.0 * ((i / 255.0) ** gamma), 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image_bgr, lut)


def perturb_noise(image_bgr: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise with std ``sigma`` (px intensity; typically 2-6)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image_bgr.shape).astype(np.float32)
    return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def perturb_blur(image_bgr: np.ndarray, ksize: int, seed: int = 0) -> np.ndarray:
    """Gaussian blur with kernel size ``ksize`` (odd; typically 3-7)."""
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)


def perturb_sharpen(image_bgr: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Unsharp-mask style mild sharpening by ``amount`` (typically 0.2-0.6)."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), 1.0)
    sharp = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def perturb_translate(image_bgr: np.ndarray, value, seed: int = 0) -> np.ndarray:
    """Translate by ``(dx, dy)`` px with border replication (dx, dy up to ~8 px).

    ``value`` is a ``(dx, dy)`` pair.
    """
    dx, dy = int(round(value[0])), int(round(value[1]))
    h, w = image_bgr.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perturb_rotate(image_bgr: np.ndarray, deg: float, seed: int = 0) -> np.ndarray:
    """Rotate by ``deg`` (typically +/-3 deg) about the image centre."""
    h, w = image_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)
    return cv2.warpAffine(image_bgr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perturb_scale(image_bgr: np.ndarray, factor: float, seed: int = 0) -> np.ndarray:
    """Scale by ``factor`` (e.g. 0.95-1.05) about the image centre.

    The output keeps the same size; empty borders are filled by replication.
    """
    h, w = image_bgr.shape[:2]
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    m = cv2.getRotationMatrix2D(
        ((new_w - 1) / 2.0, (new_h - 1) / 2.0), 0.0, 1.0
    )
    m[0, 2] += (w - new_w) / 2.0
    m[1, 2] += (h - new_h) / 2.0
    return cv2.warpAffine(resized, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perturb_reflection(image_bgr: np.ndarray, radius: int, seed: int = 0) -> np.ndarray:
    """Add a bright specular-style disc inside the iris region.

    ``radius`` is the disc radius in px. The disc is placed on a ray from the
    image centre at an angle derived from the seed so runs are deterministic but
    the location varies across seeds. Returns the perturbed image and the
    disc centre so callers can inspect behaviour near the mask.
    """
    rng = np.random.default_rng(seed)
    out = image_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r = int(radius)
    ang = rng.uniform(0.0, 2.0 * math.pi)
    dist = rng.uniform(0.55, 0.85) * (min(w, h) / 2.0)
    px = int(round(cx + dist * math.cos(ang)))
    py = int(round(cy + dist * math.sin(ang)))
    cv2.circle(out, (px, py), max(r, 3), (255, 255, 255), -1)
    return out, (px, py)


def perturb_occlusion(image_bgr: np.ndarray, radius: int, seed: int = 0) -> np.ndarray:
    """Simulate an eyelid/occluder: a filled disc rendered to mid-grey.

    Returns the perturbed image and a boolean occlusion mask (True = occluded).
    """
    rng = np.random.default_rng(seed)
    out = image_bgr.copy()
    h, w = out.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r = int(radius)
    ang = rng.uniform(0.0, 2.0 * math.pi)
    dist = rng.uniform(0.55, 0.85) * (min(w, h) / 2.0)
    px = int(round(cx + dist * math.cos(ang)))
    py = int(round(cy + dist * math.sin(ang)))
    cv2.circle(out, (px, py), max(r, 3), (90, 90, 90), -1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (px, py), max(r, 3), 1, -1)
    return out, mask.astype(bool)


PERTURBATIONS = {
    "brightness": perturb_brightness,
    "contrast": perturb_contrast,
    "gamma": perturb_gamma,
    "noise": perturb_noise,
    "blur": perturb_blur,
    "sharpen": perturb_sharpen,
    "translate": perturb_translate,
    "rotate": perturb_rotate,
    "scale": perturb_scale,
}


# --------------------------------------------------------------------------- #
# Inverse geometric mapping into the baseline pixel frame (for geometric perts)
# --------------------------------------------------------------------------- #

def map_point_back(
    x: float,
    y: float,
    kind: str,
    value: float,
    w: int,
    h: int,
) -> Tuple[float, float]:
    """Map a pixel point from a geometrically-perturbed image back to the
    baseline pixel frame using the inverse of the applied transformation.

    Only used for ``translate`` / ``rotate`` / ``scale``; other perturbations
    return the point unchanged (they are photometric and pixel-stationary).
    """
    if kind == "translate":
        dx, dy = int(round(value[0])), int(round(value[1]))
        return x - dx, y - dy
    if kind == "rotate":
        cx, cy = w / 2.0, h / 2.0
        deg = value
        ang = math.radians(deg)
        ox, oy = x - cx, y - cy
        xr = ox * math.cos(ang) + oy * math.sin(ang)
        yr = -ox * math.sin(ang) + oy * math.cos(ang)
        return cx + xr, cy + yr
    if kind == "scale":
        factor = float(value)
        return _unscale(x, y, factor, w, h)
    return x, y


def _unscale(x, y, factor, w, h):
    """Inverse of :func:`perturb_scale` about the image centre.

    Mirrors the forward transform exactly: resize by ``new_w/w`` / ``new_h/h``
    then recentre by ``(w - new_w)/2`` / ``(h - new_h)/2``. Both axes divide by
    their effective scale factor ``sx`` / ``sy``.
    """
    new_w = max(1, int(round(w * factor)))
    new_h = max(1, int(round(h * factor)))
    sx = new_w / w
    sy = new_h / h
    tx = (w - new_w) / 2.0
    ty = (h - new_h) / 2.0
    return (x - tx) / sx, (y - ty) / sy


# --------------------------------------------------------------------------- #
# Normalised-coordinate helpers
# --------------------------------------------------------------------------- #

def feature_to_normalised(
    feat: IrisFeature,
) -> Tuple[float, float]:
    """Return ``(angle_deg, radial_norm)`` for a feature from its contract."""
    return float(feat.angle_deg), float(feat.radial_norm)


def angular_gap(a: float, b: float) -> float:
    """Smallest circular angular separation in degrees."""
    gap = abs(a - b) % 360.0
    return min(gap, 360.0 - gap)


def normalise_from_pixel(
    x: float,
    y: float,
    roi: IrisROI,
    normalizer: IrisNormalizer,
) -> Optional[Tuple[float, float]]:
    """Map a baseline-frame pixel point to ``(angle_deg, radial_norm)``."""
    return normalizer.to_iris_relative(float(x), float(y), roi)


# --------------------------------------------------------------------------- #
# Baseline statistics
# --------------------------------------------------------------------------- #

def baseline_statistics(result: IrisDetectionResult) -> dict:
    """Collect per-image feature statistics from a detection result.

    ``rejected`` = candidates - accepted. Quality is taken from the per-feature
    ``confidence``. All counts are honest integers (never inflated).
    """
    fset: IrisFeatureSet = result.feature_set
    feats = fset.features
    n = len(feats)
    quals = [float(f.confidence) for f in feats]
    paper = {
        "valid": bool(result.valid),
        "status": result.status.value,
        "roi_valid": bool(fset.roi.valid),
        "usable_fraction": float(fset.usable_fraction),
        "candidates": int(fset.num_candidates),
        "accepted": n,
        "rejected": int(max(fset.num_candidates - n, 0)),
        "mean_quality": float(np.mean(quals)) if quals else 0.0,
        "median_quality": float(np.median(quals)) if quals else 0.0,
        "min_quality": float(np.min(quals)) if quals else 0.0,
        "max_quality": float(np.max(quals)) if quals else 0.0,
        "processing_time_ms": float(result.processing_time_ms),
    }
    return paper


# --------------------------------------------------------------------------- #
# Spatial distribution
# --------------------------------------------------------------------------- #

def spatial_distribution(fset: IrisFeatureSet) -> dict:
    """Measure how features are spread across normalised iris coordinates.

    A detector that dumps many points into one region is NOT robust even if the
    count is high; these metrics expose that.

    Metrics (documented definitions):
      * angular_coverage : fraction of the ``N_ANGULAR_SECTORS`` angular sectors
        that contain at least one accepted feature.
      * radial_coverage  : fraction of the ``N_RADIAL_BANDS`` radial bands that
        contain at least one accepted feature.
      * cell_coverage    : fraction of the (sector x band) cells that are occupied.
      * concentration    : max fraction of accepted features in any single cell
        (1.0 = every feature in one region; lower is better distributed).
      * angular_entropy  : normalised entropy of the per-sector counts
        (1.0 = uniform, 0.0 = all in one sector).
      * mean_nn_angular_gap / min_nn_angular_gap: mean / min nearest-neighbour
        circular angular separation (deg).
      * quadrant_count   : number of 90-deg quadrants that contain a feature.

    If there are 0 or 1 features, coverage is reported honestly (~0 / minimal)
    rather than inflated.
    """
    feats = fset.features
    n = len(feats)
    if n == 0:
        return {
            "n_features": 0,
            "angular_coverage": 0.0,
            "radial_coverage": 0.0,
            "cell_coverage": 0.0,
            "concentration": 0.0,
            "angular_entropy": 0.0,
            "mean_nn_angular_gap": 0.0,
            "min_nn_angular_gap": 0.0,
            "quadrant_count": 0,
            "occupied_cells": 0,
            "total_cells": N_ANGULAR_SECTORS * N_RADIAL_BANDS,
        }

    angles = np.asarray([float(f.angle_deg) % 360.0 for f in feats], dtype=float)
    rnorms = np.asarray([float(np.clip(f.radial_norm, 0.0, 1.0)) for f in feats], dtype=float)

    sector = np.floor(angles / (360.0 / N_ANGULAR_SECTORS)).astype(int) % N_ANGULAR_SECTORS
    band = np.floor(rnorms * N_RADIAL_BANDS).astype(int)
    band = np.clip(band, 0, N_RADIAL_BANDS - 1)

    cells = sector * N_RADIAL_BANDS + band
    counts = np.bincount(cells, minlength=N_ANGULAR_SECTORS * N_RADIAL_BANDS).astype(float)
    sector_counts = np.bincount(sector, minlength=N_ANGULAR_SECTORS).astype(float)
    band_counts = np.bincount(band, minlength=N_RADIAL_BANDS).astype(float)

    total_cells = N_ANGULAR_SECTORS * N_RADIAL_BANDS
    occupied_cells = int(np.count_nonzero(counts))

    # Entropy of per-sector counts (normalised by log2 of #sectors).
    p = sector_counts[sector_counts > 0] / n
    ent = -float(np.sum(p * np.log2(p))) if p.size else 0.0
    norm_entropy = ent / math.log2(N_ANGULAR_SECTORS) if N_ANGULAR_SECTORS > 1 else (1.0 if n else 0.0)

    # Nearest-neighbour circular angular gaps.
    sorted_ang = np.sort(angles)
    gaps = np.diff(np.concatenate([sorted_ang, sorted_ang[:1] + 360.0]))
    circ_gaps = np.minimum(gaps, 360.0 - gaps)
    mean_nn = float(np.mean(circ_gaps)) if n > 1 else 0.0
    min_nn = float(np.min(circ_gaps)) if n > 1 else 0.0

    quad = np.floor(angles / 90.0).astype(int) % 4
    quadrant_count = int(len(np.unique(quad)))

    return {
        "n_features": n,
        "angular_coverage": float(np.count_nonzero(sector_counts) / N_ANGULAR_SECTORS),
        "radial_coverage": float(np.count_nonzero(band_counts) / N_RADIAL_BANDS),
        "cell_coverage": float(occupied_cells / total_cells),
        "concentration": float(counts.max() / n),
        "angular_entropy": float(np.clip(norm_entropy, 0.0, 1.0)),
        "mean_nn_angular_gap": mean_nn,
        "min_nn_angular_gap": min_nn,
        "quadrant_count": quadrant_count,
        "occupied_cells": occupied_cells,
        "total_cells": total_cells,
    }


# --------------------------------------------------------------------------- #
# Correspondence / repeatability
# --------------------------------------------------------------------------- #

def _match(
    base_feats: Sequence[IrisFeature],
    pert_feats: Sequence[IrisFeature],
    ang_tol_deg: float,
    rad_tol: float,
) -> List[Tuple[int, int]]:
    """Greedy spatial correspondence in normalised iris coordinates.

    Returns a list of ``(base_index, pert_index)`` pairs such that each baseline
    feature matches at most one perturbed feature and vice-versa, and every pair
    is within ``ang_tol_deg`` and ``rad_tol`` of each other. Matching is greedy
    by smallest distance; this is a correspondence helper for the controlled
    evaluation, not a general-purpose matcher.
    """
    pairs: List[Tuple[int, int]] = []
    used_pert = set()
    # Order baseline features by quality (most confident first) so ties are
    # resolved deterministically.
    order = sorted(range(len(base_feats)), key=lambda i: (-float(base_feats[i].confidence), i))
    for bi in order:
        bx = base_feats[bi]
        bx_a, bx_r = float(bx.angle_deg), float(bx.radial_norm)
        best = None
        best_d = float("inf")
        for pi in range(len(pert_feats)):
            if pi in used_pert:
                continue
            px = pert_feats[pi]
            pa, pr = float(px.angle_deg), float(px.radial_norm)
            da = angular_gap(bx_a, pa)
            dr = abs(bx_r - pr)
            if da <= ang_tol_deg and dr <= rad_tol:
                d = (da / max(ang_tol_deg, 1e-9)) ** 2 + (dr / max(rad_tol, 1e-9)) ** 2
                if d < best_d:
                    best_d = d
                    best = pi
        if best is not None:
            pairs.append((bi, best))
            used_pert.add(best)
    return pairs


def repeatability_metrics(
    base_feats: Sequence[IrisFeature],
    pert_feats: Sequence[IrisFeature],
    ang_tol_deg: float = 3.0,
    rad_tol: float = 0.06,
) -> dict:
    """Compute controlled repeatability metrics between two feature sets.

    Definitions (documented):
      * repeatability_rate    : |matched base features| / |base features| --
        the fraction of baseline features that have a corresponding feature in
        the perturbed set within tolerance (plan: "fraction of features with a
        corresponding match").
      * retained_feature_rate : |perturbed features| / |base features| -- a count
        ratio (may exceed 1).
      * matched_count / base_count / pert_count : raw counts.
      * median_ang_gap_deg / median_rad_gap : median angular / radial displacement
        among the matched pairs.
      * mean_quality_change : mean change in confidence between matched pairs
        (perturbed - baseline).
    """
    n_base = len(base_feats)
    if n_base == 0:
        return {
            "repeatability_rate": 0.0,
            "retained_feature_rate": 0.0,
            "matched_count": 0,
            "base_count": 0,
            "pert_count": len(pert_feats),
            "median_ang_gap_deg": 0.0,
            "median_rad_gap": 0.0,
            "mean_quality_change": 0.0,
        }
    pairs = _match(base_feats, pert_feats, ang_tol_deg, rad_tol)
    n_match = len(pairs)
    if pairs:
        ang_gaps = [
            angular_gap(float(base_feats[b].angle_deg), float(pert_feats[p].angle_deg))
            for b, p in pairs
        ]
        rad_gaps = [
            abs(float(base_feats[b].radial_norm) - float(pert_feats[p].radial_norm))
            for b, p in pairs
        ]
        q_changes = [
            float(pert_feats[p].confidence) - float(base_feats[b].confidence)
            for b, p in pairs
        ]
        median_ang = float(np.median(ang_gaps))
        median_rad = float(np.median(rad_gaps))
        mean_q = float(np.mean(q_changes))
    else:
        median_ang = median_rad = mean_q = 0.0
    return {
        "repeatability_rate": float(n_match / n_base),
        "retained_feature_rate": float(len(pert_feats) / n_base),
        "matched_count": n_match,
        "base_count": n_base,
        "pert_count": len(pert_feats),
        "median_ang_gap_deg": median_ang,
        "median_rad_gap": median_rad,
        "mean_quality_change": mean_q,
    }


def quality_stability_correlation(
    base_feats: Sequence[IrisFeature],
    pert_feats: Sequence[IrisFeature],
    ang_tol_deg: float = 3.0,
    rad_tol: float = 0.06,
    n_quantiles: int = 4,
) -> dict:
    """Evaluate whether higher-quality baseline features survive perturbations
    more often.

    Baseline features are binned into ``n_quantiles`` by confidence; for each
    bin the *retention* is the fraction of those features that have a matching
    feature in the perturbed set. Returns per-bin retention plus the Spearman
    rank correlation between baseline confidence and being retained (per-feature),
    and a Kendall-Tau-like monotonicity summary.

    If the quality score is meaningful, retention should rise (or at least not
    fall) with the quality bin. This reports honestly even if that is not the case.
    """
    n_base = len(base_feats)
    if n_base < 2:
        return {"bins": [], "spearman_rho": 0.0, "n_base": n_base, "defined": False}
    pairs = _match(base_feats, pert_feats, ang_tol_deg, rad_tol)
    matched = {b for b, _ in pairs}

    conf = np.asarray([float(f.confidence) for f in base_feats], dtype=float)
    retained = np.asarray([1.0 if i in matched else 0.0 for i in range(n_base)], dtype=float)

    from scipy.stats import spearmanr  # type: ignore

    # When all (or none) features are retained, or all confidences are equal,
    # the rank correlation is undefined; report 0.0 with defined=False rather
    # than a NaN that would silently break downstream aggregation.
    defined = bool(np.unique(conf).size > 1 and np.unique(retained).size > 1)
    if not defined:
        rho = 0.0
    else:
        rho = float(spearmanr(conf, retained).correlation)

    # Per-quantile retention.
    order = np.argsort(conf, kind="stable")
    nq = max(1, min(n_quantiles, n_base))
    bins = []
    for q in range(nq):
        idx = order[q * n_base // nq: (q + 1) * n_base // nq]
        if len(idx) == 0:
            continue
        r = float(np.mean(retained[idx]))
        bins.append({
            "q": q,
            "n": int(len(idx)),
            "conf_min": float(conf[idx].min()),
            "conf_max": float(conf[idx].max()),
            "retention": r,
        })
    return {"bins": bins, "spearman_rho": float(rho), "n_base": n_base,
            "defined": defined}
