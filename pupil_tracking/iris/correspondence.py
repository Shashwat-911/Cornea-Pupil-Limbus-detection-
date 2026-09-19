"""Phase IV correspondence & rotation-recovery prototype (data-free, pure).

This module implements the *correspondence* and *rotation/scale recovery*
layer for the Phase IV closed-loop evaluation: given two ``IrisFeatureSet``
objects (from IMAGE A and IMAGE B of a synthetic pair produced by
``pupil_tracking/iris/paired.py``), it

  * matches features on the 5-deg angular lattice using the geometric
    (lattice) coordinates plus an optional intensity-histogram descriptor,
  * estimates the rotation as a *minimal circular difference* problem,
  * refines the estimate below the lattice resolution with normalized
    cross-correlation (NCC) re-sampling of the iris annulus,
  * estimates the relative scale from per-match pixel radii and the ROI
    geometry, and
  * classifies *why* recovery succeeded or failed.

Semantics / scope
-----------------
* **Evaluation-only.** Nothing here is wired into production detection; no
  clinical claim is made. Ground truth is never consulted inside the
  estimators: ``gt_*`` only enters via :func:`evaluate_pair`, which computes
  the minimal circular error and success flags.
* **Rotation convention**: a feature at iris angle ``phi`` in IMAGE A appears
  at iris angle ``phi - rot`` in IMAGE B when the applied OpenCV rotation is
  ``+rot`` (positive = clockwise on screen, OpenCV's convention). Hence for a
  matched pair ``theta_hat_i = (angle_a - angle_b + s*) mod 360`` where ``s*``
  is the sub-lattice NCC shift. Tests verify this convention empirically.
* **Synthetic-only validation.** Real paired ELITA images do not yet exist;
  this layer is validated against :func:`make_synthetic_pair`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pupil_tracking.iris.normalization import IrisNormalizer
from pupil_tracking.iris.types import IrisFeature, IrisFeatureSet, IrisROI


# --------------------------------------------------------------------------- #
# Public configuration / enums
# --------------------------------------------------------------------------- #

class MatchingBaseline(Enum):
    """Similarity baseline used for matching weights.

    GEOMETRIC only uses the (angle, radial_norm) lattice coordinates and the
    feature confidence. GEOMETRIC_DESCRIPTOR additionally multiplies in a
    descriptor similarity from the deterministic intensity histogram.
    """
    GEOMETRIC = "geometric"
    GEOMETRIC_DESCRIPTOR = "geometric_descriptor"


class FailureKind(Enum):
    """Why rotation/scale recovery of a pair did not reach OK.

    Precedence when more than one condition holds (documented):
    DEGENERATE -> LOW_NCC -> LOW_SIMILARITY -> HIGH_RESIDUAL -> AMBIGUOUS -> OK
    """
    OK = "OK"
    DEGENERATE = "DEGENERATE"          # too few matches to estimate anything
    LOW_NCC = "LOW_NCC"                # most refined NCC scores below ncc_min
    LOW_SIMILARITY = "LOW_SIMILARITY"  # descriptor baseline: matches too dissimilar
    HIGH_RESIDUAL = "HIGH_RESIDUAL"    # per-pair rotation estimates inconsistent
    AMBIGUOUS = "AMBIGUOUS"            # periodic / duplicate texture ambiguity
    LOW_EVIDENCE = "LOW_EVIDENCE"      # insufficient spatial evidence (sparse features)


@dataclass(frozen=True)
class CorrespondenceConfig:
    """Tunables for matching and rotation/scale recovery.

    Default tolerances are *half* the configured lattice spacings: the feature
    lattice uses 5-deg angular steps (``num_angles=72``) and radial steps of
    ``1/(num_radii+1) = 1/9``, so the corridors are ``5/2`` deg and ``1/18``.
    These are documented defaults, consistent with Phase II's tolerances, and
    are not tuned to game the closed-loop numbers.
    """

    # matching corridors (half lattice spacing)
    ang_tol_deg: float = 2.5
    rad_tol: float = 1.0 / (2.0 * 9.0)        # half of 1/9 radial step

    # coarse cyclic lattice search
    coarse_step_deg: float = 5.0

    # one-to-one matching
    min_matches: int = 4
    min_matches_for_scale: int = 3

    # sub-lattice NCC refinement
    refine: bool = True
    refine_max_shift_deg: float = 2.5         # search within +/- half lattice step
    refine_n_steps: int = 21                  # samples over [-max,+max]
    refine_lattice_deg: float = 5.0           # also try +/- these neighbours
    refine_edge_gate_deg: float = 0.6         # peaks near grid end are rejected
    refine_half_ang_deg: float = 1.5          # half width of the angular window
    refine_n_ang: int = 11                    # samples across the angular window
    refine_half_rad: float = 0.09             # half width of the radial window
    refine_n_rad: int = 5                     # samples across the radial window
    ncc_min: float = 0.42                     # gate for a refined estimate
    ncc_flat_peak_reject_denom: float = 0.002  # |denom| below this → skip parabolic interp (flat peak)

    # multi-hypothesis coarse search: the coarse lattice is periodically
    # ambiguous on low-information textures (e.g. an angularly symmetric iris),
    # so a single score-argmax can lock the NCC refinement into the wrong
    # basin. When ``refine`` is on, the top-K coarse alignments are refined
    # independently and the candidate with the most self-consistent refined
    # evidence (inlier mass at the NCC gate) is kept.
    top_k_coarse: int = 3

    # global spatial consistency
    global_consistency_inlier_tol_deg: float = 1.5   # inlier window for global voting
    global_consistency_min_inlier_frac: float = 0.40  # minimum inlier fraction to accept
    global_consistency_min_inlier_count: int = 3      # absolute minimum inlier count

    # sparse evidence gate (disabled by default; enable for clinical use)
    evidence_gate: bool = False                   # enable honest-refusal gate
    evidence_min_features: int = 4               # minimum features for any estimate
    evidence_min_angular_coverage: float = 0.20  # minimum angular coverage ratio
    evidence_min_occupied_bins: int = 3           # minimum 30° angular bins occupied

    # estimator thresholds used by failure classification
    ransac_tol_deg: float = 1.5
    residual_std_max_deg: float = 2.0
    min_consensus_fraction: float = 0.5
    ambiguity_ratio_max: float = 0.5
    low_ncc_ratio_max: float = 0.5
    low_similarity_ratio_max: float = 0.5


# --------------------------------------------------------------------------- #
# Circular statistics
# --------------------------------------------------------------------------- #

def wrap_deg(angle_deg: float) -> float:
    """Wrap an angle to ``[0, 360)`` degrees."""
    return float(angle_deg % 360.0)


def circular_distance(a_deg: float, b_deg: float) -> float:
    """Smallest circular angular separation in degrees, in ``[0, 180]``."""
    gap = abs(a_deg - b_deg) % 360.0
    return float(min(gap, 360.0 - gap))


def circular_signed_difference(a_deg: float, b_deg: float) -> float:
    """Signed minimal difference ``(b - a)`` normalised to ``(-180, 180]``."""
    return float((b_deg - a_deg + 540.0) % 360.0 - 180.0)


def circular_mean(angles_deg: Sequence[float],
                  weights: Optional[Sequence[float]] = None) -> float:
    """Weighted circular mean of angles (default: uniform weights)."""
    angles = np.asarray([float(a) for a in angles_deg], dtype=float)
    if angles.size == 0:
        return 0.0
    n = angles.size
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.maximum(np.asarray([float(x) for x in weights], dtype=float), 0.0)
    s = float(np.sum(w * np.sin(np.radians(angles))))
    c = float(np.sum(w * np.cos(np.radians(angles))))
    if abs(s) + abs(c) < 1e-12:
        return 0.0
    return wrap_deg(math.degrees(math.atan2(s, c)))


def circular_std(angles_deg: Sequence[float],
                 weights: Optional[Sequence[float]] = None) -> float:
    """Circular standard deviation in degrees (0 when degenerate)."""
    angles = np.asarray([float(a) for a in angles_deg], dtype=float)
    if angles.size == 0:
        return 0.0
    n = angles.size
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.maximum(np.asarray([float(x) for x in weights], dtype=float), 0.0)
    total = float(np.sum(w))
    if total < 1e-12:
        return 0.0
    s = float(np.sum(w * np.sin(np.radians(angles)))) / total
    c = float(np.sum(w * np.cos(np.radians(angles)))) / total
    r = min(math.hypot(s, c), 1.0)
    if r >= 1.0:
        return 0.0
    return float(math.degrees(math.sqrt(-2.0 * math.log(r))))


def angular_span(angles_deg: Sequence[float]) -> float:
    """Circular span of angles: smallest arc that contains every angle."""
    angles = sorted(float(a) % 360.0 for a in angles_deg)
    if not angles:
        return 0.0
    if len(angles) == 1:
        return 0.0
    gaps = [
        angles[i + 1] - angles[i] for i in range(len(angles) - 1)
    ]
    gaps.append(angles[0] + 360.0 - angles[-1])
    return float(360.0 - max(gaps))


# --------------------------------------------------------------------------- #
# Sparse-feature coverage metrics
# --------------------------------------------------------------------------- #

def compute_feature_metrics(
    features: Sequence[IrisFeature],
) -> Dict[str, float]:
    """Compute sparse-feature coverage metrics from a list of IrisFeature.

    Returns a dict with engineering measurements (not clinical interpretations):
    - feature_count: number of features
    - angular_span: smallest arc containing all features (degrees)
    - largest_angular_gap: largest angular gap between adjacent features
    - angular_coverage_ratio: angular_span / 360
    - occupied_angular_bins_30: number of distinct 30° bins with features
    """
    if not features:
        return {
            "feature_count": 0,
            "angular_span": 0.0,
            "largest_angular_gap": 360.0,
            "angular_coverage_ratio": 0.0,
            "occupied_angular_bins_30": 0,
        }

    angles = sorted(float(f.angle_deg) % 360.0 for f in features)
    n = len(angles)

    if n < 2:
        span = 0.0
        largest_gap = 360.0
    else:
        gaps = [angles[i + 1] - angles[i] for i in range(n - 1)]
        gaps.append(angles[0] + 360.0 - angles[-1])
        largest_gap = max(gaps)
        span = 360.0 - largest_gap

    bins = set()
    for a in angles:
        bins.add(int(a // 30) % 12)

    return {
        "feature_count": n,
        "angular_span": span,
        "largest_angular_gap": largest_gap,
        "angular_coverage_ratio": span / 360.0,
        "occupied_angular_bins_30": len(bins),
    }


# --------------------------------------------------------------------------- #
# Descriptor similarity
# --------------------------------------------------------------------------- #

def descriptor_distance(da: Optional[np.ndarray],
                        db: Optional[np.ndarray]) -> Optional[float]:
    """L1 distance of two unit-L1 16-bin intensity histograms, or None.

    A missing descriptor (or a shape mismatch) yields None -- treated as
    "unknown, neutral" (similarity 1.0) rather than a penalty, because a
    feature can be reliably located purely geometrically.
    """
    if da is None or db is None or da.size == 0 or db.size == 0:
        return None
    da = np.asarray(da, dtype=np.float32).ravel()
    db = np.asarray(db, dtype=np.float32).ravel()
    if da.shape != db.shape:
        return None
    return float(np.abs(da - db).sum())


def descriptor_similarity(da: Optional[np.ndarray],
                          db: Optional[np.ndarray]) -> float:
    """Map a descriptor distance to a similarity in (0, 1]; neutral = 1.0."""
    d = descriptor_distance(da, db)
    if d is None:
        return 1.0
    return float(1.0 / (1.0 + d))


# --------------------------------------------------------------------------- #
# Match records
# --------------------------------------------------------------------------- #

@dataclass
class Correspondence:
    """A single one-to-one matched pair with its geometry and quality.

    ``refined_shift_deg`` / ``ncc`` / ``rotation_estimate_i`` are filled in
    after construction by the refinement / estimation stages.
    """
    index_a: int
    index_b: int
    angle_a: float
    angle_b: float
    radial_a: float
    radial_b: float
    confidence_a: float
    confidence_b: float
    weight_geometric: float
    weight_descriptor: float
    descriptor_distance: Optional[float]
    coarse_residual_deg: float
    refined_shift_deg: Optional[float] = None
    ncc: Optional[float] = None
    rotation_estimate_i: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "index_a": int(self.index_a),
            "index_b": int(self.index_b),
            "angle_a": float(self.angle_a),
            "angle_b": float(self.angle_b),
            "radial_a": float(self.radial_a),
            "radial_b": float(self.radial_b),
            "confidence_a": float(self.confidence_a),
            "confidence_b": float(self.confidence_b),
            "weight_geometric": float(self.weight_geometric),
            "weight_descriptor": float(self.weight_descriptor),
            "descriptor_distance": self.descriptor_distance,
            "coarse_residual_deg": float(self.coarse_residual_deg),
            "refined_shift_deg": self.refined_shift_deg,
            "ncc": self.ncc,
            "rotation_estimate_i": self.rotation_estimate_i,
        }


@dataclass
class CorrespondenceResult:
    """Full estimation output for one IMAGE A -> IMAGE B pair."""
    valid: bool = False
    failure: FailureKind = FailureKind.DEGENERATE
    failure_reason: str = ""
    baseline: MatchingBaseline = MatchingBaseline.GEOMETRIC_DESCRIPTOR
    rotation_method: str = "consensus"

    n_matches: int = 0
    matched: List[Correspondence] = field(default_factory=list)

    coarse_rotation_deg: float = 0.0
    coarse_score: float = 0.0
    coarse_matches: int = 0
    ambiguity_ratio: float = 0.0

    rotation_estimates: Dict[str, float] = field(default_factory=dict)
    estimated_rotation_deg: float = 0.0
    refined_used: int = 0
    ncc_below_gate: int = 0
    mean_ncc: float = 0.0
    min_ncc: float = 0.0
    circular_std_deg: float = 0.0
    consensus_fraction: float = 0.0
    consensus_inlier_std_deg: float = 999.0

    estimated_scale: float = 1.0
    geometry_scale: float = 1.0
    pupil_scale: float = 1.0
    scale_matches_used: int = 0
    scale_valid: bool = False

    processing_time_ms: float = 0.0

    # Global consistency diagnostics (populated when method="global_consistency")
    global_inlier_count: int = 0
    global_inlier_frac: float = 0.0
    global_inlier_std_deg: float = 0.0

    # Sparse evidence metrics (populated by estimate_correspondence)
    feature_count: int = 0
    angular_span: float = 0.0
    angular_coverage_ratio: float = 0.0
    largest_angular_gap: float = 0.0
    occupied_angular_bins: int = 0

    def to_dict(self) -> dict:
        d = {
            "valid": bool(self.valid),
            "failure": self.failure.value,
            "failure_reason": str(self.failure_reason),
            "baseline": self.baseline.value,
            "rotation_method": self.rotation_method,
            "n_matches": int(self.n_matches),
            "coarse_rotation_deg": float(self.coarse_rotation_deg),
            "coarse_score": float(self.coarse_score),
            "coarse_matches": int(self.coarse_matches),
            "ambiguity_ratio": float(self.ambiguity_ratio),
            "rotation_estimates": {k: float(v) for k, v in self.rotation_estimates.items()},
            "estimated_rotation_deg": float(self.estimated_rotation_deg),
            "refined_used": int(self.refined_used),
            "ncc_below_gate": int(self.ncc_below_gate),
            "mean_ncc": float(self.mean_ncc),
            "min_ncc": float(self.min_ncc),
            "circular_std_deg": float(self.circular_std_deg),
            "consensus_fraction": float(self.consensus_fraction),
            "consensus_inlier_std_deg": float(self.consensus_inlier_std_deg),
            "estimated_scale": float(self.estimated_scale),
            "geometry_scale": float(self.geometry_scale),
            "pupil_scale": float(self.pupil_scale),
            "scale_matches_used": int(self.scale_matches_used),
            "scale_valid": bool(self.scale_valid),
            "processing_time_ms": float(self.processing_time_ms),
            "global_inlier_count": int(self.global_inlier_count),
            "global_inlier_frac": float(self.global_inlier_frac),
            "global_inlier_std_deg": float(self.global_inlier_std_deg),
            "feature_count": int(self.feature_count),
            "angular_span": float(self.angular_span),
            "angular_coverage_ratio": float(self.angular_coverage_ratio),
            "largest_angular_gap": float(self.largest_angular_gap),
            "occupied_angular_bins": int(self.occupied_angular_bins),
        }
        return d


# --------------------------------------------------------------------------- #
# Coarse cyclic lattice matching
# --------------------------------------------------------------------------- #

def _pair_weight(fa: IrisFeature, fb: IrisFeature,
                 baseline: MatchingBaseline) -> Tuple[float, float]:
    """Return ``(weight_geometric, weight_descriptor)`` for a candidate pair."""
    conf = float(min(fa.confidence, fb.confidence))
    sim = descriptor_similarity(fa.descriptor, fb.descriptor)
    return conf, conf * sim


def _similarity_matrix(fa: Sequence[IrisFeature],
                       fb: Sequence[IrisFeature]) -> np.ndarray:
    """Precompute descriptor similarities ``(n_a x n_b)`` (neutral 1.0 default)."""
    n_a, n_b = len(fa), len(fb)
    mat = np.ones((n_a, n_b), dtype=np.float64)
    a_idx = [i for i, f in enumerate(fa)
             if f.descriptor is not None and f.descriptor.size > 0]
    b_idx = [j for j, f in enumerate(fb)
             if f.descriptor is not None and f.descriptor.size > 0]
    if a_idx and b_idx:
        A = np.stack([
            np.asarray(fa[i].descriptor, dtype=np.float32).ravel() for i in a_idx
        ])
        B = np.stack([
            np.asarray(fb[j].descriptor, dtype=np.float32).ravel() for j in b_idx
        ])
        if A.shape[1] == B.shape[1] and A.shape[1] > 0:
            dist = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
            mat[np.ix_(a_idx, b_idx)] = 1.0 / (1.0 + dist)
    return mat


def _coarse_alignments(
    fa: Sequence[IrisFeature],
    fb: Sequence[IrisFeature],
    baseline: MatchingBaseline,
    config: CorrespondenceConfig,
    weight_matrix: np.ndarray,
) -> List[Dict]:
    """Exhaustive cyclic alignment over the coarse lattice.

    Returns one dict per candidate rotation ``d`` (0..355 step 5) with keys
    ``d``, ``matches`` [(ia, ib), ...], ``score``, ``ambiguous_b``.
    """
    out: List[Dict] = []
    step = float(config.coarse_step_deg)
    n_a, n_b = len(fa), len(fb)
    if n_a == 0 or n_b == 0:
        return [{"d": float(d % 360.0), "matches": [], "score": 0.0,
                 "ambiguous_b": 0, "n_matches": 0}
                for d in np.arange(0.0, 360.0, step)]

    a_ang = np.asarray([wrap_deg(f.angle_deg) for f in fa], dtype=float)
    b_ang = np.asarray([wrap_deg(f.angle_deg) for f in fb], dtype=float)
    base = (a_ang[:, None] - b_ang[None, :]) % 360.0  # angular diff before shift
    rad = np.abs(
        np.asarray([f.radial_norm for f in fa], dtype=float)[:, None]
        - np.asarray([f.radial_norm for f in fb], dtype=float)[None, :]
    )
    rad_ok = rad <= config.rad_tol
    atol = max(config.ang_tol_deg, 1e-9)
    rtol = max(config.rad_tol, 1e-9)
    d2_full = (rad / rtol) ** 2                       # radial term (angle added per d)

    # A feature order: highest confidence first (ties by id) -> deterministic.
    order = sorted(range(n_a),
                   key=lambda i: (-float(fa[i].confidence), int(fa[i].id)))

    for d in np.arange(0.0, 360.0, step):
        dd = (base - d) % 360.0
        dist_ang = np.minimum(dd, 360.0 - dd)
        ang_ok = dist_ang <= config.ang_tol_deg
        mask = ang_ok & rad_ok
        d2 = (dist_ang / atol) ** 2 + d2_full

        matches: List[Tuple[int, int]] = []
        score = 0.0
        used_b = set()
        for ia in order:
            row_ibs = np.nonzero(mask[ia])[0]
            row_ibs = [int(ib) for ib in row_ibs if int(ib) not in used_b]
            if not row_ibs:
                continue
            best_ib = min(row_ibs, key=lambda ib: float(d2[ia, ib]))
            matches.append((ia, best_ib))
            score += float(weight_matrix[ia, best_ib])
            used_b.add(best_ib)
        ambiguous = int((mask.sum(axis=0) > 1).sum())
        out.append({
            "d": float(d % 360.0),
            "matches": matches,
            "score": score,
            "ambiguous_b": int(min(ambiguous, len(matches))),
            "n_matches": len(matches),
        })
    return out


# --------------------------------------------------------------------------- #
# Correspondence record construction
# --------------------------------------------------------------------------- #

def _build_correspondences(
    fa: Sequence[IrisFeature],
    fb: Sequence[IrisFeature],
    alignment: Dict,
    baseline: MatchingBaseline,
    config: CorrespondenceConfig,
) -> List[Correspondence]:
    """Build one-to-one :class:`Correspondence` records for one aggregate.

    ``alignment`` is a coarse candidate dict from :func:`_coarse_alignments`
    with keys ``d`` and ``matches``. The angular residual is measured against
    the candidate's shift ``d``.
    """
    out: List[Correspondence] = []
    for ia, ib in alignment["matches"]:
        a, b = fa[ia], fb[ib]
        wg, wd = _pair_weight(a, b, baseline)
        ddesc = descriptor_distance(a.descriptor, b.descriptor)
        residual = wrap_deg((a.angle_deg - b.angle_deg) - float(alignment["d"]))
        if residual > 180.0:
            residual -= 360.0
        out.append(Correspondence(
            index_a=ia, index_b=ib,
            angle_a=float(a.angle_deg), angle_b=float(b.angle_deg),
            radial_a=float(a.radial_norm), radial_b=float(b.radial_norm),
            confidence_a=float(a.confidence), confidence_b=float(b.confidence),
            weight_geometric=wg, weight_descriptor=wd,
            descriptor_distance=ddesc,
            coarse_residual_deg=float(residual),
        ))
    return out


def _candidate_evidence(
    matched: Sequence[Correspondence],
    baseline: MatchingBaseline,
    config: CorrespondenceConfig,
) -> float:
    """Weighted refined-inlier mass of a candidate alignment.

    Only NCC-gated (refined, ``ncc >= ncc_min``) matches vote. The dominant
    consensus of those per-match rotation estimates is found, and the returned
    score is the total weight of the estimates within the inlier tolerance of
    that consensus. On a wrong coarse basin the refined peaks scatter and a
    small fraction of the mass lands inside the inlier window; on the correct
    basin nearly all refined estimates agree (fraction ~1.0). This is what
    lets multi-hypothesis selection escape a score-argmax that locked onto a
    periodic (wrong) basin.
    """
    thetas: List[float] = []
    weights: List[float] = []
    for m in matched:
        if (m.refined_shift_deg is None or m.ncc is None
                or m.ncc < config.ncc_min):
            continue
        thetas.append(wrap_deg(m.angle_a - m.angle_b + m.refined_shift_deg))
        w = (m.weight_descriptor
             if baseline == MatchingBaseline.GEOMETRIC_DESCRIPTOR
             else m.weight_geometric)
        weights.append(w)
    if not thetas:
        return 0.0
    t = np.asarray(thetas, dtype=float)
    w = np.asarray(weights, dtype=float)
    center = _estimate_consensus(t, w)
    tol = max(float(config.ransac_tol_deg),
              float(config.global_consistency_inlier_tol_deg))
    inl = np.asarray([circular_distance(float(center), float(x)) <= tol
                      for x in t], dtype=bool)
    return float(np.sum(w[inl]))


# --------------------------------------------------------------------------- #
# Sub-lattice NCC refinement
# --------------------------------------------------------------------------- #

def _radial_bounds_np(roi: IrisROI, angles_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised ``radial_bounds`` for arrays of angles (px radii)."""
    a = max(float(roi.pupil_semi_major), 1e-9)
    b = max(float(roi.pupil_semi_minor), 1e-9)
    L = max(float(roi.limbus_semi_major), 1e-9)
    B = max(float(roi.limbus_semi_minor), 1e-9)
    ang = np.radians(np.asarray(angles_deg, dtype=float))

    def _ell(phi_ell_deg, smaj, smin):
        phi = np.radians(float(phi_ell_deg))
        cost = np.cos(ang - phi)
        sint = np.sin(ang - phi)
        denom = (sint / smaj) ** 2 + (cost / smin) ** 2
        r = np.where(denom > 1e-12, 1.0 / np.sqrt(np.maximum(denom, 1e-12)),
                     max(smaj, smin))
        return r

    inner = _ell(roi.pupil_angle_deg, a, b) * (1.0 + float(roi.inner_inset_frac))
    outer = _ell(roi.limbus_angle_deg, L, B) * (1.0 - float(roi.outer_inset_frac))
    return inner, outer


def _bilinear_sample(gray: np.ndarray,
                     xs: np.ndarray,
                     ys: np.ndarray) -> np.ndarray:
    """Vectorised bilinear sampling with border replication (clamp)."""
    h, w = gray.shape
    xc = np.clip(xs, 0.0, w - 1.0)
    yc = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xc - x0).astype(np.float32)
    fy = (yc - y0).astype(np.float32)
    ia = gray[y0, x0]
    ib = gray[y0, x1]
    ic = gray[y1, x0]
    id_ = gray[y1, x1]
    out = (ia * (1 - fx) * (1 - fy) + ib * fx * (1 - fy)
           + ic * (1 - fx) * fy + id_ * fx * fy)
    return out.astype(np.float32)


def _sample_windows_many(
    gray: np.ndarray, roi: IrisROI,
    center_angles_deg: np.ndarray,
    center_radials: np.ndarray,
    config: CorrespondenceConfig,
) -> np.ndarray:
    """Vectorised window sampling for many matches and angular shifts.

    ``center_angles_deg`` has shape ``(M, n_basis, n_shifts)`` and is the sole
    differentiated axis (used for the sub-lattice shift search on the A side).
    ``center_radials`` has shape ``(M,)``. Returns intensity windows with shape
    ``(M, n_basis, n_shifts, n_ang, n_rad)``.
    """
    M, n_basis, n_shifts = center_angles_deg.shape
    ang_off = np.linspace(-config.refine_half_ang_deg, config.refine_half_ang_deg,
                          config.refine_n_ang)
    rad_off = np.linspace(-config.refine_half_rad, config.refine_half_rad,
                          config.refine_n_rad)
    angles = np.mod(
        np.asarray(center_angles_deg, dtype=float)[..., None, None]
        + ang_off[None, None, None, :, None],
        360.0,
    )
    radials = (
        np.asarray(center_radials, dtype=float)[:, None, None, None, None]
        + rad_off[None, None, None, None, :]
    )
    angles = np.broadcast_to(angles, (M, n_basis, n_shifts,
                                      config.refine_n_ang, config.refine_n_rad))
    radials = np.broadcast_to(radials, angles.shape)
    inner, outer = _radial_bounds_np(roi, angles.reshape(-1))
    radius = (inner + np.clip(radials.reshape(-1), 0.0, 1.0) * (outer - inner))
    ang = np.radians(angles.reshape(-1))
    xs = roi.center_x + radius * np.cos(ang)
    ys = roi.center_y + radius * np.sin(ang)
    vals = _bilinear_sample(gray, xs, ys)
    return vals.reshape(angles.shape)


def _ncc_many(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised NCC over the trailing ``(n_ang, n_rad)`` axes.

    ``a``: ``(M, n_basis, n_shifts, n_ang, n_rad)``,
    ``b``: ``(M, 1, 1, n_ang, n_rad)`` -> ``(M, n_basis, n_shifts)``.
    """
    ac = a - a.mean(axis=(3, 4), keepdims=True)
    bc = b - b.mean(axis=(3, 4), keepdims=True)
    num = (ac * bc).sum(axis=(3, 4))
    den = np.sqrt((ac * ac).sum(axis=(3, 4)) * (bc * bc).sum(axis=(3, 4)))
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def _refine_batch(
    gray_a: np.ndarray, roi_a: IrisROI,
    gray_b: np.ndarray, roi_b: IrisROI,
    matched: Sequence[Correspondence],
    config: CorrespondenceConfig,
    coarse_rotation_deg: float = 0.0,
) -> None:
    """Refine every match in one vectorised pass (mutates the records).

    For each match the A-side window is sampled over one *contiguous* grid of
    angular offsets ``t`` in ``[-L - m, +L + m]`` (``L`` = lattice step, ``m`` =
    ``refine_max_shift_deg``), covering the pair's own lattice slot plus its two
    lattice neighbours. The best offset by NCC is kept and the per-pair rotation
    estimate becomes ``angle_a - angle_b + t``. A peak that sits within
    ``refine_edge_gate_deg`` of the grid's absolute end is unreliable (the true
    correlation lies farther away than the searched window) and is rejected by
    forcing NCC to 0 -- a real alignment therefore always lands in the interior.

    When ``coarse_rotation_deg`` is provided the A-side search centre is
    shifted by the coarse residual for each match so that NCC refinement
    operates in the basin of the coarse estimate rather than around zero
    offset.

    A *flat-peak gate* is applied to parabolic refinement: when the absolute
    curvature of the NCC parabola (``|denom|``) falls below
    ``config.ncc_flat_peak_reject_denom``, the parabolic interpolation is
    skipped and the raw grid-argmax offset is used instead.  On flat peaks the
    parabola is unreliable and its correction can inject spurious sub-degree
    bias; the raw argmax is more conservative and avoids this.
    """
    M = len(matched)
    if M == 0:
        return
    if not roi_a.valid or not roi_b.valid:
        for m in matched:
            m.refined_shift_deg = None
            m.ncc = None
        return
    L = float(config.refine_lattice_deg)
    m = float(config.refine_max_shift_deg)
    s_grid = np.linspace(-m, m, config.refine_n_steps)
    offs = np.unique(np.concatenate([s_grid - L, s_grid, s_grid + L]))

    a_angles = np.asarray([wrap_deg(mt.angle_a) for mt in matched], dtype=float)
    a_rad = np.asarray([mt.radial_a for mt in matched], dtype=float)
    b_angles = np.asarray([wrap_deg(mt.angle_b) for mt in matched], dtype=float)
    b_rad = np.asarray([mt.radial_b for mt in matched], dtype=float)

    # Compute coarse residual per match and shift A-side search centre.
    if coarse_rotation_deg != 0.0:
        raw_diff = (a_angles - b_angles) % 360.0
        residuals = (raw_diff - coarse_rotation_deg) % 360.0
        residuals[residuals > 180.0] -= 360.0
        a_angles = (a_angles - residuals) % 360.0

    centers = a_angles[:, None, None] + offs[None, None, :]     # (M, 1, N)
    wins_a = _sample_windows_many(gray_a, roi_a, centers, a_rad, config)
    wins_b = _sample_windows_many(gray_b, roi_b, b_angles[:, None, None], b_rad,
                                  config)
    ncc = _ncc_many(wins_a, wins_b)[:, 0, :]                     # (M, N)
    best_j = np.argmax(ncc, axis=1)

    edge_guard = (L + m) - config.refine_edge_gate_deg
    step = offs[1] - offs[0]
    for i, mt in enumerate(matched):
        j = int(best_j[i])
        t_star = float(offs[j])
        if 0 < j < ncc.shape[1] - 1:
            a0 = float(ncc[i, j - 1])
            a1 = float(ncc[i, j])
            a2 = float(ncc[i, j + 1])
            denom = a0 - 2.0 * a1 + a2
            if abs(denom) > config.ncc_flat_peak_reject_denom:
                delta = 0.5 * (a0 - a2) / denom
                if abs(delta) <= 1.0:
                    t_star = float(offs[j] + delta * step)
        ncc_val = float(ncc[i, j])
        if abs(t_star) >= edge_guard:
            ncc_val = 0.0
        mt.refined_shift_deg = t_star
        mt.ncc = ncc_val


# --------------------------------------------------------------------------- #
# Rotation estimators
# --------------------------------------------------------------------------- #

def _estimates_from_matches(matched: Sequence[Correspondence],
                            baseline: MatchingBaseline,
                            config: CorrespondenceConfig) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return ``(thetas_deg, weights, n_reliable)`` over the accepted matches.

    A match is *reliable* when the refined estimate passed the NCC gate; when
    refinement is disabled every match is reliable using its coarse (lattice)
    estimate. Gated matches fall back to the coarse estimate in the returned
    arrays but are not counted as reliable.
    """
    thetas = []
    weights = []
    n_reliable = 0
    for m in matched:
        used_refined = (
            config.refine
            and m.refined_shift_deg is not None
            and m.ncc is not None
            and m.ncc >= config.ncc_min
        )
        if used_refined:
            th = wrap_deg(m.angle_a - m.angle_b + m.refined_shift_deg)
            n_reliable += 1
        else:
            th = wrap_deg(m.angle_a - m.angle_b)
        w = m.weight_descriptor if baseline == MatchingBaseline.GEOMETRIC_DESCRIPTOR \
            else m.weight_geometric
        thetas.append(th)
        weights.append(w)
    return np.asarray(thetas, dtype=float), np.asarray(weights, dtype=float), n_reliable


def _estimate_consensus(thetas: np.ndarray, weights: np.ndarray) -> float:
    """Exhaustive angular-consensus estimate.

    Bin the per-pair estimates into 0.5-deg circular bins, take the modal bin,
    then return the weight-weighted circular mean of the estimates within +/-1
    deg of the modal bin centre. Deterministic (no random sampling).
    """
    n = thetas.size
    if n == 0:
        return 0.0
    if n == 1:
        return wrap_deg(float(thetas[0]))
    bin_w = 0.5
    nbins = int(round(360.0 / bin_w))
    edges = np.arange(0.0, 360.0, bin_w)
    idx = np.clip(np.floor(np.mod(thetas, 360.0) / bin_w).astype(int), 0, nbins - 1)
    bin_w_sum = np.zeros(nbins, dtype=float)
    np.add.at(bin_w_sum, idx, weights)
    mode = int(np.argmax(bin_w_sum))
    center = bin_w * (mode + 0.5)
    mask = np.abs(np.mod(thetas - center + 180.0, 360.0) - 180.0) <= 1.0
    if not mask.any():
        return wrap_deg(float(thetas[int(np.argmax(weights))]))
    return circular_mean(thetas[mask].tolist(), weights[mask].tolist())


def _estimate_weighted_circular(thetas: np.ndarray, weights: np.ndarray) -> float:
    if thetas.size == 0:
        return 0.0
    return circular_mean(thetas.tolist(), weights.tolist())


def _estimate_ransac_exhaustive(thetas: np.ndarray, weights: np.ndarray,
                                tol_deg: float) -> float:
    """Exhaustive two-point RANSAC-style inlier consensus (deterministic)."""
    n = thetas.size
    if n == 0:
        return 0.0
    if n == 1:
        return wrap_deg(float(thetas[0]))
    if n == 2:
        return circular_mean(thetas.tolist(), weights.tolist())
    best_base = None
    best_score = -1.0
    best_inliers = None
    for i in range(n):
        for j in range(i + 1, n):
            base = circular_mean([thetas[i], thetas[j]], [weights[i], weights[j]])
            mask = np.asarray([circular_distance(base, float(t)) <= tol_deg
                               for t in thetas], dtype=bool)
            score = float(np.sum(weights[mask]))
            # primary: score (weighted inlier mass); tie-break: count then base
            if (score > best_score + 1e-12 or
                (abs(score - best_score) <= 1e-12 and
                 (int(mask.sum()) > int(best_inliers.sum()) if best_inliers is not None else True))):
                best_score = score
                best_inliers = mask
                best_base = base
    if best_inliers is None or not best_inliers.any():
        return circular_mean(thetas.tolist(), weights.tolist())
    return circular_mean(thetas[best_inliers].tolist(), weights[best_inliers].tolist())


def _estimate_global_consistency(
    thetas: np.ndarray,
    weights: np.ndarray,
    config: CorrespondenceConfig,
    matched: Sequence[Correspondence] = (),
) -> Tuple[float, Dict]:
    """Global spatial consistency estimator.

    Builds a weighted circular histogram of per-pair rotation estimates,
    finds the dominant peak, and verifies that multiple spatially
    distributed correspondences agree with the hypothesis.

    Returns (theta_hat, info_dict) where info_dict contains diagnostic
    fields for the caller.
    """
    info: Dict = {
        "global_inlier_count": 0,
        "global_inlier_frac": 0.0,
        "global_inlier_std_deg": 0.0,
        "global_peak_weight": 0.0,
        "global_reliable_count": int(thetas.size),
    }

    n = thetas.size
    if n == 0:
        return 0.0, info
    if n == 1:
        return wrap_deg(float(thetas[0])), info

    # Build weighted circular histogram (1.0-deg bins for robustness)
    bin_w = 1.0
    nbins = int(round(360.0 / bin_w))
    idx = np.clip(np.floor(np.mod(thetas, 360.0) / bin_w).astype(int), 0, nbins - 1)
    bin_w_sum = np.zeros(nbins, dtype=float)
    np.add.at(bin_w_sum, idx, weights)
    mode = int(np.argmax(bin_w_sum))
    center = bin_w * (mode + 0.5)

    # Inliers: estimates within tolerance of the dominant peak
    tol = float(config.global_consistency_inlier_tol_deg)
    inlier_mask = np.asarray(
        [circular_distance(float(thetas[i]), center) <= tol
         for i in range(n)], dtype=bool
    )
    n_inlier = int(inlier_mask.sum())
    inlier_frac = float(n_inlier / n) if n > 0 else 0.0
    total_weight = float(np.sum(weights))
    inlier_weight = float(np.sum(weights[inlier_mask])) if n_inlier > 0 else 0.0
    peak_weight = inlier_weight / total_weight if total_weight > 1e-12 else 0.0

    # Weighted circular mean of inlier estimates
    if n_inlier > 0:
        theta_hat = circular_mean(
            thetas[inlier_mask].tolist(), weights[inlier_mask].tolist()
        )
        inlier_std = float(circular_std(
            thetas[inlier_mask].tolist(), weights[inlier_mask].tolist()
        ))
    else:
        theta_hat = center
        inlier_std = 999.0

    info.update({
        "global_inlier_count": n_inlier,
        "global_inlier_frac": inlier_frac,
        "global_inlier_std_deg": inlier_std,
        "global_peak_weight": peak_weight,
    })

    return wrap_deg(theta_hat), info


# --------------------------------------------------------------------------- #
# Scale estimation
# --------------------------------------------------------------------------- #

def _feature_px_radius(feat: IrisFeature, roi: IrisROI,
                       normalizer: IrisNormalizer) -> float:
    inner, outer = normalizer.radial_bounds(roi, float(feat.angle_deg))
    return float(inner + float(feat.radial_norm) * (outer - inner))


# --------------------------------------------------------------------------- #
# Top-level orchestrators
# --------------------------------------------------------------------------- #

def _classify_failure(matched: Sequence[Correspondence],
                      result: CorrespondenceResult,
                      config: CorrespondenceConfig) -> None:
    """Assign ``failure`` / ``failure_reason``/``valid`` per documented order."""
    n = len(matched)
    if n < config.min_matches:
        result.failure = FailureKind.DEGENERATE
        result.failure_reason = f"only {n} matches (< {config.min_matches})"
        result.valid = False
        return

    # Sparse evidence gate: insufficient spatial evidence for reliable rotation
    if (config.evidence_gate
            and (result.feature_count < config.evidence_min_features
                 or result.angular_coverage_ratio < config.evidence_min_angular_coverage
                 or result.occupied_angular_bins < config.evidence_min_occupied_bins)):
        result.failure = FailureKind.LOW_EVIDENCE
        result.failure_reason = (
            f"sparse evidence: {result.feature_count} features, "
            f"{result.angular_coverage_ratio:.2f} angular coverage, "
            f"{result.occupied_angular_bins} bins"
        )
        result.valid = False
        return

    refined = [m for m in matched if m.ncc is not None]
    if config.refine and refined:
        low_ncc = sum(1 for m in refined if m.ncc < config.ncc_min)
        if low_ncc / max(len(refined), 1) > config.low_ncc_ratio_max:
            result.failure = FailureKind.LOW_NCC
            result.failure_reason = (
                f"{low_ncc}/{len(refined)} refined NCC scores below {config.ncc_min}"
            )
            result.valid = False
            return

        if (result.consensus_fraction < config.min_consensus_fraction
                or result.consensus_inlier_std_deg > config.residual_std_max_deg):
            result.failure = FailureKind.HIGH_RESIDUAL
            result.failure_reason = (
                f"consensus fraction {result.consensus_fraction:.2f} < "
                f"{config.min_consensus_fraction} or inlier std "
                f"{result.consensus_inlier_std_deg:.2f} deg > "
                f"{config.residual_std_max_deg} deg"
            )
            result.valid = False
            return

    if result.ambiguity_ratio > config.ambiguity_ratio_max:
        result.failure = FailureKind.AMBIGUOUS
        result.failure_reason = (
            f"ambiguity ratio {result.ambiguity_ratio:.2f} > "
            f"{config.ambiguity_ratio_max}"
        )
        result.valid = False
        return

    if result.baseline == MatchingBaseline.GEOMETRIC_DESCRIPTOR and n > 0:
        with_desc = [m for m in matched if m.descriptor_distance is not None
                     and m.weight_geometric > 1e-9]
        if with_desc:
            # weight_descriptor = min(conf) * (1/(1+d)), weight_geometric = min(conf),
            # so their ratio is the descriptor similarity.
            sims = [m.weight_descriptor / m.weight_geometric for m in with_desc]
            low_sim = sum(1 for s in sims if s < 0.5)
            if low_sim / len(sims) > config.low_similarity_ratio_max:
                result.failure = FailureKind.LOW_SIMILARITY
                result.failure_reason = (
                    f"{low_sim}/{len(sims)} matched descriptors below 0.5 similarity"
                )
                result.valid = False
                return

    result.failure = FailureKind.OK
    result.failure_reason = ""
    result.valid = True


def estimate_correspondence(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_set_a: IrisFeatureSet,
    feature_set_b: IrisFeatureSet,
    baseline: MatchingBaseline = MatchingBaseline.GEOMETRIC_DESCRIPTOR,
    rotation_method: str = "consensus",
    config: Optional[CorrespondenceConfig] = None,
) -> CorrespondenceResult:
    """Estimate rotation/scale between two feature sets (no ground truth used)."""
    import time
    t0 = time.perf_counter()
    cfg = config if config is not None else CorrespondenceConfig()

    res = CorrespondenceResult(
        baseline=baseline,
        rotation_method=rotation_method,
    )

    roi_a, roi_b = feature_set_a.roi, feature_set_b.roi
    fa, fb = feature_set_a.features, feature_set_b.features

    if not roi_a.valid or not roi_b.valid or not fa or not fb:
        res.failure = FailureKind.DEGENERATE
        res.failure_reason = "invalid ROI or empty feature set"
        res.valid = False
        res.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # Compute sparse-feature coverage metrics from the A-side feature set
    fm = compute_feature_metrics(fa)
    res.feature_count = fm["feature_count"]
    res.angular_span = fm["angular_span"]
    res.angular_coverage_ratio = fm["angular_coverage_ratio"]
    res.largest_angular_gap = fm["largest_angular_gap"]
    res.occupied_angular_bins = fm["occupied_angular_bins_30"]

    gray_a = gray_b = None
    if cfg.refine:
        gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if image_a.ndim == 3 \
            else image_a.astype(np.float32, copy=False)
        gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if image_b.ndim == 3 \
            else image_b.astype(np.float32, copy=False)

    # coarse cyclic search
    conf_a = np.asarray([float(f.confidence) for f in fa], dtype=float)
    conf_b = np.asarray([float(f.confidence) for f in fb], dtype=float)
    conf = np.minimum(conf_a[:, None], conf_b[None, :])
    if baseline == MatchingBaseline.GEOMETRIC_DESCRIPTOR:
        weight_matrix = conf * _similarity_matrix(fa, fb)
    else:
        weight_matrix = conf
    aligns = _coarse_alignments(fa, fb, baseline, cfg, weight_matrix)

    if cfg.refine and cfg.top_k_coarse > 1:
        # Multi-hypothesis refinement: the score-argmax coarse baseline is
        # periodically ambiguous on low-information textures (an angularly
        # symmetric iris is a deliberately hard case). Refine the top-K
        # candidates independently and keep the one whose refined estimates
        # form the most self-consistent NCC-gated cluster. Each candidate is
        # refined inside its own basin; the winning candidate's records are
        # already refined, so the single-basin refinement below is skipped.
        candidates = sorted(
            aligns, key=lambda x: (x["score"], x["n_matches"]), reverse=True
        )[: max(1, int(cfg.top_k_coarse))]
        best_evidence: float = -1.0
        best_candidate: Optional[Dict] = None
        best_matched: Optional[List[Correspondence]] = None
        for cand in candidates:
            cand_matched = _build_correspondences(fa, fb, cand, baseline, cfg)
            _refine_batch(gray_a, roi_a, gray_b, roi_b, cand_matched, cfg,
                          coarse_rotation_deg=float(cand["d"]))
            ev = _candidate_evidence(cand_matched, baseline, cfg)
            if best_candidate is None or ev > best_evidence:
                best_evidence = ev
                best_candidate = cand
                best_matched = cand_matched
        best = best_candidate if best_candidate is not None else max(
            aligns, key=lambda x: (x["score"], x["n_matches"]))
        matched = best_matched if best_matched is not None else []
    else:
        best = max(aligns, key=lambda x: (x["score"], x["n_matches"]))
        matched = _build_correspondences(fa, fb, best, baseline, cfg)
        if cfg.refine:
            _refine_batch(gray_a, roi_a, gray_b, roi_b, matched, cfg,
                          coarse_rotation_deg=float(best["d"]))

    res.coarse_rotation_deg = best["d"]
    res.coarse_score = best["score"]
    res.coarse_matches = best["n_matches"]
    res.n_matches = len(matched)
    res.matched = matched
    res.ambiguity_ratio = (
        best["ambiguous_b"] / max(len(matched), 1) if matched else 0.0
    )

    na = IrisNormalizer()
    nb = IrisNormalizer()

    # sub-lattice refinement summary
    if cfg.refine:
        refined = [m for m in matched if m.ncc is not None]
        res.refined_used = len(refined)
        if refined:
            nccs = np.asarray([float(m.ncc) for m in refined], dtype=float)
            res.mean_ncc = float(nccs.mean())
            res.min_ncc = float(nccs.min())
            res.ncc_below_gate = int((nccs < cfg.ncc_min).sum())

    # per-pair rotation estimates + estimators (reliable pairs only)
    thetas, weights, n_reliable = _estimates_from_matches(matched, baseline, cfg)
    est_thetas = thetas
    est_weights = weights
    if cfg.refine and 0 < n_reliable < len(matched):
        reliable: List[Tuple[float, float]] = []
        for m in matched:
            if (m.refined_shift_deg is not None and m.ncc is not None
                    and m.ncc >= cfg.ncc_min):
                th = wrap_deg(m.angle_a - m.angle_b + m.refined_shift_deg)
            else:
                continue
            w = (m.weight_descriptor
                 if baseline == MatchingBaseline.GEOMETRIC_DESCRIPTOR
                 else m.weight_geometric)
            reliable.append((th, w))
        if reliable:
            est_thetas = np.asarray([t for t, _ in reliable], dtype=float)
            est_weights = np.asarray([w for _, w in reliable], dtype=float)
    res.rotation_estimates = {
        "consensus": _estimate_consensus(est_thetas, est_weights),
        "weighted_circular": _estimate_weighted_circular(est_thetas, est_weights),
        "ransac": _estimate_ransac_exhaustive(est_thetas, est_weights, cfg.ransac_tol_deg),
    }
    # Global spatial consistency: build circular histogram, find peak,
    # verify multi-feature agreement.
    gc_theta, gc_info = _estimate_global_consistency(
        est_thetas, est_weights, cfg, matched,
    )
    res.rotation_estimates["global_consistency"] = gc_theta
    res._global_consistency_info = gc_info
    res.global_inlier_count = gc_info.get("global_inlier_count", 0)
    res.global_inlier_frac = gc_info.get("global_inlier_frac", 0.0)
    res.global_inlier_std_deg = gc_info.get("global_inlier_std_deg", 0.0)

    # Hybrid: use global consistency when it has sufficient support,
    # otherwise fall back to consensus. This prevents sparse/ambiguous
    # cases from being mis-estimated by global voting alone.
    gc_ok = (
        res.global_inlier_count >= cfg.global_consistency_min_inlier_count
        and res.global_inlier_frac >= cfg.global_consistency_min_inlier_frac
    )
    res.rotation_estimates["global_hybrid"] = (
        gc_theta if gc_ok
        else res.rotation_estimates["consensus"]
    )
    if rotation_method not in res.rotation_estimates:
        rotation_method = "consensus"
    res.rotation_method = rotation_method
    res.estimated_rotation_deg = float(res.rotation_estimates[rotation_method])

    # residual consistency over the estimates that actually contributed
    res.circular_std_deg = float(circular_std(est_thetas.tolist(),
                                              est_weights.tolist()))

    # consensus-cluster statistics used by failure classification: the spread
    # *within* the modal neighbourhood (not the global std) is the meaningful
    # residual indicator, because a handful of near-tie NCC peaks on an
    # otherwise clean lattice rotation inflate the global std without harming
    # the estimate.
    if est_thetas.size:
        win = max(float(cfg.ransac_tol_deg), 1.0)
        theta_hat = float(res.estimated_rotation_deg)
        inlier = [i for i in range(est_thetas.size)
                  if circular_distance(est_thetas[i], theta_hat) <= win]
        res.consensus_fraction = float(len(inlier) / est_thetas.size)
        if inlier:
            in_t = [float(est_thetas[i]) for i in inlier]
            in_w = [float(est_weights[i]) for i in inlier]
            res.consensus_inlier_std_deg = float(circular_std(in_t, in_w))

    # scale estimation: median per-match pixel-radius ratio
    radii = []
    for m in matched:
        ra = _feature_px_radius(fa[m.index_a], roi_a, na)
        rb = _feature_px_radius(fb[m.index_b], roi_b, nb)
        if ra > 1e-9:
            radii.append(rb / ra)
    if len(radii) >= cfg.min_matches_for_scale:
        res.estimated_scale = float(np.median(radii))
        res.scale_matches_used = len(radii)
        res.scale_valid = True
    if roi_a.limbus_radius_px > 1e-9 and roi_b.limbus_radius_px > 1e-9:
        res.geometry_scale = float(roi_b.limbus_radius_px / roi_a.limbus_radius_px)
    if roi_a.pupil_radius_px > 1e-9 and roi_b.pupil_radius_px > 1e-9:
        res.pupil_scale = float(roi_b.pupil_radius_px / roi_a.pupil_radius_px)

    _classify_failure(matched, res, cfg)
    res.processing_time_ms = (time.perf_counter() - t0) * 1000.0
    return res


def evaluate_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    feature_set_a: IrisFeatureSet,
    feature_set_b: IrisFeatureSet,
    gt_rotation_deg: float,
    gt_scale: float,
    baseline: MatchingBaseline = MatchingBaseline.GEOMETRIC_DESCRIPTOR,
    rotation_method: str = "consensus",
    config: Optional[CorrespondenceConfig] = None,
) -> Dict:
    """Run :func:`estimate_correspondence` and compare against ground truth.

    The minimal-circular-difference metric is used for rotation error (wraps
    correctly at 0/360); success is reported at sub-0.5/1.0/2.0 deg. This is
    the *only* function that may inspect ``gt_*``.
    """
    res = estimate_correspondence(
        image_a, image_b, feature_set_a, feature_set_b,
        baseline=baseline, rotation_method=rotation_method, config=config,
    )
    theta = res.estimated_rotation_deg
    out = res.to_dict()
    out.update({
        "gt_rotation_deg": float(gt_rotation_deg),
        "gt_scale": float(gt_scale),
        "min_circular_diff_deg": float(circular_distance(theta, float(gt_rotation_deg))),
        "rotation_error_signed_deg": float(
            circular_signed_difference(float(gt_rotation_deg), theta)
        ),
        "scale_error_ratio": float(res.estimated_scale / gt_scale),
        "success_0_5_deg": bool(
            circular_distance(theta, float(gt_rotation_deg)) <= 0.5
        ),
        "success_1_0_deg": bool(
            circular_distance(theta, float(gt_rotation_deg)) <= 1.0
        ),
        "success_2_0_deg": bool(
            circular_distance(theta, float(gt_rotation_deg)) <= 2.0
        ),
        "failure": res.failure.value,
        "failure_reason": res.failure_reason,
    })
    return out