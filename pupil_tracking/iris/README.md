# Iris Feature Detection (Phase I concept model)

This package implements the **classical (non-learned) iris feature detection
baseline** for the Pupil-Limbus detector. It is a self-contained, CPU-friendly,
deterministic module. It is **disabled by default** in the production pipeline:
you opt in by calling its public API explicitly.

> **Important scope note.** Phase I only *detects* iris features. It does **not**
> implement feature matching, image registration, rotation/cyclotorsion
> estimation, or astigmatism-axis correction. Those are later phases and depend
> on this detector producing a stable, normalised feature set.

> **No clinical accuracy claims.** The thresholds (e.g. `min_contrast`) are
> conservative placeholders to be validated against real ELITA paired data when
> it becomes available. The available surgical clinical imagery is used only as
> a dark, low-texture proxy during development, not as clinical ground truth.

---

## What it detects

Within the annular iris region (between the pupil and limbus boundaries), the
detector samples candidate locations on a polar lattice and accepts those that
are:

* inside the **usable iris mask** (not occluded, not specular reflection),
* not too close to the pupil or limbus boundary,
* not **flat** — they have local textural response above `min_contrast`,
* **spatially separated** (angular suppression) so features are distributed
  rather than clumped.

Each accepted feature describes one iris landmark: crypts, furrows, or generic
texture. Every feature carries:

| field            | meaning                                              |
|------------------|------------------------------------------------------|
| `x`, `y`         | pixel coordinates                                   |
| `angle_deg`      | polar angle relative to the iris center [0, 360)     |
| `radial_norm`    | normalised radial position (0 = pupil, 1 = limbus)   |
| `scale`          | patch scale relative to the iris outer radius        |
| `orientation_deg`| local orientation (rotationally informative later)   |
| `feature_type`   | `CRYPT` / `FURROW` / `TEXTURE` (heuristic)           |
| `response`       | local texture energy (mean |Laplacian| of the patch) |
| `local_contrast` | center-vs-surround contrast in [0, ~1]              |
| `visibility`     | from the usable mask (0/1)                           |
| `confidence`     | composite quality score in [0, 1]                   |
| `descriptor`     | deterministic 16-bin intensity-histogram (float32)   |

The radial/angular normalisation is what later phases need for **rotation
invariance and registration**: the same anatomical feature should map to a
stable `(angle_deg, radial_norm)` across images regardless of scale or pupil
dilation.

---

## Pipeline

```
pupil/limbus EllipseParams (from UnifiedDetector or any source)
        │
        ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
│ ROI build   │ → │ Masking      │ → │ Normalization    │ → │ Extraction   │
│ iris/roi.py │   │ iris/masking │   │ iris/normalization│  │ iris/extract │
└─────────────┘   └──────────────┘   └──────────────────┘   └──────────────┘
        │                │                    │                    │
        └────────────────┴────────────────────┴────────────────────┘
                                  │
                                  ▼
                        IrisDetectionResult (iris/detect.py)
```

### 1. ROI — `IrisROIExtractor` (`roi.py`)
Constructs the annular region from the pupil and limbus ellipses, insetting away
from each boundary. If geometry is missing or implausible (e.g. pupil almost as
large as limbus), it returns an invalid `IrisROI` with a `reason` rather than
raising — the pipeline degrades gracefully to `NO_ROI`.

Currently the annulus uses *circular* rings centered on the iris center. The
polar sampling uses the *ellipse-aware* radius from `iris/normalization.py`.

### 2. Masking — `IrisMasking` (`masking.py`)
Builds a boolean "usable" mask = annulus ∩ ¬reflection ∩ ¬external occlusion.
Reflections are detected by reusing the existing `ReflectionRemover` (restricted
to the ROI). External occlusion can be supplied by the caller.

### 3. Normalization — `IrisNormalizer` (`normalization.py`)
Maps between image pixels and iris-relative coordinates `(angle_deg, radial_norm)`.
`radial_bounds(angle)` returns the pupil/limbus boundary radii *at that angle*,
so sampling follows the elliptical iris boundary rather than assuming a circle.

### 4. Extraction — `IrisFeatureExtractor` (`extraction.py`)
Samples the polar lattice, computes texture response / contrast / descriptor,
classifies coarsely, filters by quality and angular separation, and builds the
`IrisFeatureSet` with candidate/accepted counts, region coverage and usable
fraction.

### 5. Orchestration — `detect.py`
`IrisFeatureDetector` / `detect_iris_features(...)` wire the stages together,
measure processing time, and return an `IrisDetectionResult`.

---

## Quick start

```python
import cv2
from pupil_tracking.core.detector import UnifiedDetector
from pupil_tracking.iris import detect_iris_features, draw_iris_overlay

det = UnifiedDetector()
img = cv2.imread("clinical_data/clean/eye_01.jpeg")
dr = det.detect(img, frame_number=0)

pupil = dr.pupil.ellipse if dr.has_pupil else None
limbus = dr.limbus.ellipse if dr.has_limbus else None

result = detect_iris_features(img, pupil, limbus)
print(result.status, result.feature_set.num_accepted)

overlay = draw_iris_overlay(img, result, pupil=pupil, limbus=limbus)
```

Run end-to-end on the clinical proxy set:

```bash
python scripts/iris_feature_smoke.py
```

---

## Integration & safety

* **Disabled by default.** This module is not wired into `UnifiedDetector`'s
  default path. It is invoked explicitly and returns its own result type, so it
  cannot alter existing pupil/limbus/quality outputs.
* **Non-mutating.** It does not modify the input image nor the `EllipseParams`
  it is given. It only reads them.
* **Isolation.** `pupil_tracking/iris/` is a standalone sub-package. Its only
  dependencies on the rest of the repo are `ReflectionRemover` (masking) and
  the shared `types.py` dataclasses (`EllipseParams`), both reused, not
  duplicated, and never modified here.

---

## Limitations (honest)

* **Low texture on real data.** Surgical and clinical iris regions are often
  dark and low-texture; measured mean |Laplacian| is mostly 1.2–3.3 on the proxy
  images. Expect modest accepted-feature counts (order 1–100) and do not inflate
  results. `min_contrast` thresholds must be tuned against real ELITA data.
* **Heuristic classification.** `CRYPT`/`FURROW`/`TEXTURE` are coarse and not
  validated against anatomical ground truth; treat as informal.
* **Circular annulus basis.** The usable mask currently assumes a circular
  annulus; elliptical pupil/limbus are handled in the *sampling* radius but not
  in the mask geometry.
* **No invariance guarantees.** Illumination invariance of the histogram
  descriptor is limited; later phases should normalise intensity before use.
* **Not validated clinically.** No clinical repeatability/reliability study has
  been performed; this is a feature-detection baseline only.
* **Performance.** A full run including reflection removal is ~0.2–0.6 s/image
  on the small clinical proxy images (CPU). Not yet optimised.

---

## Testing

* `pupil_tracking/tests/test_iris_features.py` — deterministic, synthetic
  fixtures; runs in the normal pytest suite without the ML model or clinical
  data.
* `scripts/iris_feature_smoke.py` — clinical-proxy smoke validation that
  requires the production model and clinical imagery (gitignored); run manually.

---

## Roadmap (later phases, not implemented here)

1. Feature **matching** / correspondence between images.
2. **Registration** and **rotation/cyclotorsion** estimation from matched
   features.
3. **Astigmatism-axis** correction.
4. Threshold validation and calibration against **ELITA paired data**.
