"""Phase IV synthetic-pair generator (data-free, deterministic).

Creates IMAGE A -> IMAGE B pairs whose geometric transform is *exactly known*
and recorded as ground truth, so an evaluation layer can measure how well the
correspondence/rotation-recovery stage recovers the applied rotation, scale and
translation without real paired ELITA data (which does not yet exist).

Semantics
---------
* IMAGE A is the source image, returned unmodified.
* IMAGE B is generated in memory by applying, about a chosen centre:

      rotation_deg  (OpenCV convention: positive = clockwise on screen)
      scale         (isotropic; > 1 magnifies)
      translation_px

  followed by an optional perturbation applied **after the warp, in IMAGE B's
  pixel frame**. The occlusion mask therefore aligns directly with IMAGE B and
  can be passed to the iris detector's ``external_occlusion`` argument.

* Determinism: the same ``(source, config)`` yields pixel-identical pairs; the
  noise/reflection/occlusion perturbations draw from
  ``np.random.default_rng(config.seed)``. No source file is ever written and no
  source image is modified.

The perturbed copies reuse the existing deterministic perturbation helpers in
``pupil_tracking/iris/robustness.py`` (no new dependency). Scaling is applied
about the same centre, so the iris content grows/shrinks in place -- mirroring
the dock-magnification change a real pre/post-dock pair would show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from pupil_tracking.iris import robustness as R


VALID_PERTURBATIONS: Tuple[str, ...] = (
    "brightness", "contrast", "gamma", "noise", "blur", "sharpen",
    "reflection", "occlusion",
)

# Sensible defaults for each perturbation kind when ``perturbation_params`` is
# empty (mirroring the Phase II harness strengths).
_DEFAULT_PARAMS: Mapping[str, Mapping[str, float]] = {
    "brightness": {"delta": 25.0},
    "contrast": {"factor": 1.2},
    "gamma": {"gamma": 0.8},
    "noise": {"sigma": 6.0},
    "blur": {"ksize": 7.0},
    "sharpen": {"amount": 0.6},
    "reflection": {"radius": 14.0},
    "occlusion": {"radius": 40.0},
}


@dataclass(frozen=True)
class PairConfig:
    """Transformation applied to the source to produce IMAGE B.

    ``center`` is the pixel ``(x, y)`` used as the rotation/scale centre.
    When None, the image centre is used. ``translation_px`` is applied after
    rotation/scale about that centre.
    """

    rotation_deg: float = 0.0
    scale: float = 1.0
    translation_px: Tuple[float, float] = (0.0, 0.0)
    center: Optional[Tuple[float, float]] = None
    perturbation: Optional[str] = None
    perturbation_params: Mapping[str, float] = field(default_factory=dict)
    seed: int = 0


@dataclass(frozen=True)
class SyntheticPair:
    """A generated IMAGE A -> IMAGE B pair with exact ground truth."""

    name: str
    image_a: np.ndarray               # reference image (source, unmodified)
    image_b: np.ndarray               # warped (+ optionally perturbed) image
    gt_rotation_deg: float            # rotation applied to A to yield B
    gt_scale: float                   # scale applied to A to yield B
    gt_translation_px: Tuple[float, float]
    perturbation: Optional[str]
    perturbation_params: Mapping[str, float]
    seed: int
    rotation_center: Optional[Tuple[float, float]]
    occlusion_mask: Optional[np.ndarray]  # in IMAGE B space, iff occlusion
    transform_description: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gt_rotation_deg": float(self.gt_rotation_deg),
            "gt_scale": float(self.gt_scale),
            "gt_translation_px": list(self.gt_translation_px),
            "perturbation": self.perturbation,
            "perturbation_params": dict(self.perturbation_params),
            "seed": int(self.seed),
            "rotation_center": self.rotation_center,
            "transform_description": self.transform_description,
            "image_a_shape": list(self.image_a.shape),
            "image_b_shape": list(self.image_b.shape),
        }


def _post_warp_perturbation(
    image_bgr: np.ndarray,
    kind: str,
    params: Mapping[str, float],
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply one deterministic perturbation to IMAGE B; returns (image, mask)."""
    p = dict(_DEFAULT_PARAMS[kind])
    p.update({k: float(v) for k, v in params.items()})

    if kind == "brightness":
        return R.perturb_brightness(image_bgr, p["delta"], seed), None
    if kind == "contrast":
        return R.perturb_contrast(image_bgr, p["factor"], seed), None
    if kind == "gamma":
        return R.perturb_gamma(image_bgr, p["gamma"], seed), None
    if kind == "noise":
        return R.perturb_noise(image_bgr, p["sigma"], seed), None
    if kind == "blur":
        return R.perturb_blur(image_bgr, int(round(p["ksize"])), seed), None
    if kind == "sharpen":
        return R.perturb_sharpen(image_bgr, p["amount"], seed), None
    if kind == "reflection":
        out, _ = R.perturb_reflection(image_bgr, int(round(p["radius"])), seed)
        return out, None
    if kind == "occlusion":
        return R.perturb_occlusion(image_bgr, int(round(p["radius"])), seed)
    raise ValueError(f"unknown perturbation kind: {kind!r}")


def make_synthetic_pair(
    source_bgr: np.ndarray,
    config: PairConfig,
    name: str = "pair",
) -> SyntheticPair:
    """Build one deterministic synthetic pair.

    Rotation and scale are applied about ``config.center`` (default: the image
    centre) using a single OpenCV rotation+scale matrix, then translation, then
    (optionally) the perturbation. GT values are recorded as applied.
    """
    if source_bgr is None or source_bgr.size == 0:
        raise ValueError("source_bgr must be a non-empty image")
    if config.perturbation is not None and config.perturbation not in VALID_PERTURBATIONS:
        raise ValueError(
            f"perturbation must be one of {VALID_PERTURBATIONS}, "
            f"got {config.perturbation!r}"
        )

    h, w = source_bgr.shape[:2]
    cx, cy = config.center if config.center is not None else (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D((cx, cy), config.rotation_deg, config.scale)
    matrix[0, 2] += float(config.translation_px[0])
    matrix[1, 2] += float(config.translation_px[1])
    image_b = cv2.warpAffine(
        source_bgr, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    occlusion_mask = None
    if config.perturbation is not None:
        image_b, occlusion_mask = _post_warp_perturbation(
            image_b, config.perturbation, config.perturbation_params, config.seed,
        )

    desc = (
        f"rot {config.rotation_deg:g} deg, scale {config.scale:g}, "
        f"trans {config.translation_px}"
        + (f", perturb={config.perturbation}" if config.perturbation else "")
    )
    return SyntheticPair(
        name=name,
        image_a=source_bgr.copy(),
        image_b=image_b,
        gt_rotation_deg=float(config.rotation_deg),
        gt_scale=float(config.scale),
        gt_translation_px=(
            float(config.translation_px[0]), float(config.translation_px[1]),
        ),
        perturbation=config.perturbation,
        perturbation_params=dict(config.perturbation_params),
        seed=int(config.seed),
        rotation_center=config.center,
        occlusion_mask=occlusion_mask,
        transform_description=desc,
    )