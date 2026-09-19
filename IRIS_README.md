# Iris Detection & Cyclotorsion

## 1. Objective

The goal is to reliably detect and represent iris structures in ELITA RGB
pre-/post-docking imagery, and eventually estimate **cyclotorsion** to support
the surgical workflow.

Iris functionality is **ADDITIVE**: it runs alongside the existing
pupil/limbus/centration pipeline and never replaces it. The iris module is a
separate sub-package (`pupil_tracking/iris/`) that consumes validated
pupil+limbus geometry and produces its own result type.

## 2. Existing Centrations/Clinical Functions

The following clinical functions are preserved and **out of scope for
modification** by any iris work:

- pupil detection
- limbus detection
- corneal centre
- offset
- offset angle
- calibration
- WTW (white-to-white)
- centration

Iris detection must not bypass, alter, or degrade any of these.

## 3. Current Pipeline

The implemented pipeline (as of the latest commits) is:

```
ELITA RGB frame
→ pupil/limbus geometry (UnifiedDetector)
→ validated iris ROI (valid iris annulus)
→ masking (reflection / occlusion inside the annulus)
→ texture candidate extraction (polar lattice)
→ feature rejection / acceptance (contrast + texture gate + angular suppression)
→ IrisFeatureSet (accepted features with angle / radial_norm / descriptor)
→ correspondence / rotation estimate when a reference image exists
```

Key modules:

| stage          | module                    |
|----------------|---------------------------|
| ROI build      | `pupil_tracking/iris/roi.py` |
| Masking        | `pupil_tracking/iris/masking.py` |
| Normalization  | `pupil_tracking/iris/normalization.py` |
| Extraction     | `pupil_tracking/iris/extraction.py` |
| Orchestration  | `pupil_tracking/iris/detect.py` |
| Correspondence | `pupil_tracking/iris/correspondence.py` |

See `pupil_tracking/iris/README.md` for the module-level design.

Further detail in the phase reports under `_phase_artifacts/`:
`PHASE_9_PENTACAM_ISOLATION.md`, `PHASE_10_EYELID_MASKING.md`,
`PHASE_11_LOWSNR.md`, `PHASE_13_RUNTIME_VERIFY.md`.

## 4. Daugman-Inspired Direction

**Implemented today** — classical (non-learned) iris *feature detection*:
polar sampling inside the annulus, texture/contrast response, deterministic
histogram descriptors, and masking of reflections/occlusion. This is the
baseline that later phases build on.

**Planned (not yet implemented as a production system)** — a Daugman-style
normalized iris representation:

- iris boundary normalization (pupil ↔ limbus)
- rubber-sheet / polar (unwrapped) representation
- normalized iris texture
- phase / Gabor-style representation
- masking
- angular cyclic shifts for rotational comparison

Do **not** treat the current implementation as a production Daugman
IrisCode, and do **not** assume the IrisCode representation is inherently
rotation-invariant — rotation handling has to be implemented explicitly
(e.g. cyclic shifts / correspondence search over angular offsets).

## 5. Proven Validation

- **Pentacam isolation (Phase 9):** on 5 valid Pentacam captures,
  **289/289 (100%)** accepted features lie strictly inside the validated iris
  annulus. UI/background/sclera regions produced no accepted features, and
  captures without stable limbus geometry were honestly refused.
- **Eyelid investigation (Phase 10):** eyelid occlusion was **not** the
  primary cause of `NO_FEATURES`; the important failure mode was low-SNR iris
  texture on dim ELITA RGB frames.
- **Phase 11 (adaptive, ROI-relative texture gate):** the previous absolute
  texture threshold killed candidates on dim/low-SNR frames. Known rescue
  results (pre → post, accepted features):
  - `232912A #0`:      0 → 10
  - `232912A #5769`:  29 → 68
  - `232912A #6346`:  12 → 71
  - `233210A #0`:      1 → 6
  - good frames remained around 54–72
  - flat/noise frames continued to refuse honestly (0 features kept)
- **Tests:** the iris + Pentacam suite
  (`test_iris_features.py`, `test_iris_correspondence.py`,
  `test_iris_paired.py`, `test_iris_robustness.py`,
  `test_pentacam_types.py`) is **140/140 green**.
- **Live GUI evidence:** the running application displayed a state of
  `Status: Valid, Features: 70, Coverage: 0.6%`, proving the detector
  generates accepted features on real input at runtime.

## 6. Current Application Behavior

The application currently shows in the surgery/rings UI:

```
CYCLOTORSION / IRIS
Status:            Valid
Features:          70
Coverage:          0.6%
Rotation Angle:    ---
Confidence:        ---
Evidence:          Single image
```

This demonstrates **successful runtime iris-feature generation** but
**incomplete visualization**: the 70 accepted features are counted, but the
feature coordinates are not visibly rendered over the eye image. This is a
detection-vs-visualization gap, not an absence of detection.

## 7. Where We Are Stuck

### Blocker #1 — Feature Visualization

The detector produces accepted iris features, but the GUI does not currently
render those feature coordinates over the eye. Until that works, feature
quality cannot be visually validated. **This is the immediate blocker.**

### Blocker #2 — Reference Correspondence

A single image cannot establish cyclotorsion. A valid rotation estimate
requires reliable correspondence between the pre-dock reference iris and the
post-dock iris sequence, with known laterality.

### Blocker #3 — Upstream Geometry Gating

Iris detection runs only when valid pupil **and** limbus geometry exist
(`result.has_both`). Real ELITA frames can contain visible iris texture but
still produce no iris features because the upstream geometry is unavailable.
This is a known gating limitation — it is not proof that iris texture is
absent.

### Blocker #4 — Clinical Validation

Properly paired/labeled clinical pre/post data with known laterality and
trustworthy ground truth is required before treating any rotation result as
clinically validated.

## 8. What Is NOT Blocking Us

- Iris features **do exist** in real data.
- The iris detector is **not** universally returning zero.
- Pentacam isolation is **already proven** (289/289 in-iris).
- Low-SNR handling **already improved** real ELITA frames.
- The `70 features` GUI state **proves runtime feature generation**.

## 9. Exact Next Step

The next implementation phase is **not** another performance optimization.
It is:

```
Live Iris Feature Visualization & Runtime Trace
```

Goal: render the existing accepted features at their true image coordinates
and trace the data path
(`IrisFeatureDetector → IrisDetectionResult → GUI state →
_refresh_display() → _draw_overlay_scaled() → displayed image`) to find why
the accepted coordinates are not drawn.

## 10. Future Roadmap

1. Feature visualization / runtime trace
2. Validate feature quality across pre-dock ELITA RGB frames
3. Validate post-dock features
4. Pre/post feature correspondence
5. Robust cyclotorsion estimation
6. Pentacam ↔ ELITA cross-modality validation
7. Clinical validation
8. Surgical workflow integration with strict safety gating

## 11. Safety Constraints

- No iris work may bypass or alter the existing centration/clinical
  detection pipeline.
- Rotation must never be automatically injected into surgical control or
  planning without validated evidence and explicit safety gating.

## 12. Current Known Performance

Phase 12 addressed GUI/CLI startup responsiveness (details in
`_phase_artifacts/PHASE_12_STARTUP.md`):

- Startup was dominated by an eager `torch` import.
- Lazy import reduced CLI startup dramatically (`launch_gui.py --help`:
  ~31 000 ms → ~115 ms).
- Background detector initialization removed Tk event-loop blocking
  (window appears immediately; model loads in a background thread).
- ONNX inference remains CPU-bound (~1–2 s/frame); video frame processing
  is well below real-time on CPU.
- Performance optimization is currently **secondary** to proving iris
  feature correctness.

## 13. Phase History

| Phase | Scope | Commit |
|-------|-------|--------|
| Phase 9  | Pentacam isolation (289/289 in-iris) | (see `PHASE_9_PENTACAM_ISOLATION.md`) |
| Phase 10 | Eyelid masking investigation | (see `PHASE_10_EYELID_MASKING.md`) |
| Phase 11 | Low-SNR adaptive texture gate | `d809692` |
| Phase 12 | Startup responsiveness (lazy imports, background detector init) | `41f7da1`, `2ba5665` |

Commit descriptions verified from `git log`. Earlier-phase evidence is
documented in the referenced reports under `_phase_artifacts/`.