# Cyclotorsion Phase 2 architecture

The implementation follows the Phase 2 requirement: a seated Pentacam iris
image is the reference and a focused ELITA image is the current image. A
single image can establish pupil and limbus geometry, but a cyclotorsion angle
requires the reference/current pair. The static path is the clinical default;
the dynamic path reuses the same registration core while updating the current
frame.

## Pipeline

1. `UnifiedDetector` or `PentacamIrisDetector` validates the input as grayscale
   or BGR uint8 and rejects empty, malformed, and UI-only images.
2. The detector builds a UI/glint mask, finds the pupil, and estimates the
   limbus. The measured limbus fit uses supported radial transitions where
   available. Occluded sectors remain inferred and are reported separately.
3. The pupil and limbus define the fixed annular ROI. Polar unwrapping maps the
   pupil-to-limbus band to a rectangular representation while retaining a
   validity mask. No features outside the annulus enter registration.
4. The registration engine estimates an angular shift with masked normalized
   correlation across radial bands. It rejects weak, boundary, low-overlap,
   and inconsistent matches rather than returning a guessed angle.
5. The result contains the angle, confidence, quality state, diagnostics, and
   processing time. A caller can safely abstain when `valid` is false.

The legacy multi-stream `RegistrationEngine` also requires the configured
minimum number of corroborating streams. A single surviving stream is an
abstention, never a clinical answer. Its broad clinical-image audit remains an
exploratory compatibility check; the seated Phase 2 engine and its paired
geometry tests are the release path.

## Feature review output

Texture keypoints are spatially separated local structures used for inspection
and future training. They are not anatomical labels. The optional review mode
also emits `crypt_like_candidate` and `furrow_like_candidate` hypotheses with
`anatomy_verified=false`; a clinician must label these before they can become
training targets. This distinction prevents a detector response from being
presented as a confirmed crypt, collarette, furrow, freckle, or pigment spot.

`scripts/review_sitting.py` produces a three-panel image showing the source,
the pupil/limbus and texture points, and the unverified anatomy hypotheses.
It marks observed limbus support in cyan and inferred/occluded boundary in
gray. The JSON beside the image is the machine-readable contract for later
annotation tooling.

## Static and dynamic use

```powershell
# One seated reference and one current image
python scripts/register_sitting.py --reference seated.bmp --current elita.jpeg

# Local review output; anatomy labels remain hypotheses until expert-labelled
python scripts/review_sitting.py --input seated.bmp --output-dir output/review

# Video or camera tracking, with frame zero as the explicit baseline
python scripts/live_sitting.py --video eye.mp4 --headless
python scripts/live_sitting.py --camera
```

The live tracker should reset its baseline only on an explicit operator action
and should display `NO RELIABLE MATCH` for an invalid frame. Capture, display,
and file-transfer latency are outside the registration timing measurement.

## Integration boundaries

The centation and cyclotorsion toggles are independent in the GUI. Centration
continues to expose pupil, limbus, corneal centre, and offset. Cyclotorsion
consumes validated geometry and can be disabled without changing centration.
The ink-marker stream remains available as a separate fallback/verification
stream and does not alter the iris registration result.

The four controls (master cyclotorsion, iris features, ink markers, and phase
correlation) are persisted in the local application settings file and restored
when the GUI reopens. Disabled modules are not executed by the registration
engine. `scripts/static_cyclotorsion_review.py` creates the requested static
review artifact: side-by-side unwrapped strips, angle, confidence, diagnostics,
and a JSON record marked `doctor_validation: PENDING`. The operator validates
the result before any treatment-axis action.

The detector API is intentionally small so a later model can replace only the
feature proposal stage:

```python
result = PentacamIrisDetector().detect(image, review_anatomy=True)
result.geometry.pupil
result.geometry.limbus
result.geometry.refined_limbus
result.feature_set.features
result.anatomy_candidates
```

For future training, retain the original image, eye/laterality metadata,
boundary annotations, feature-point annotations, posture/device pair ID, and
expert confidence. Do not train on the current unverified hypotheses as
ground truth. The Phase 2 cutoff is an engineering gate: target absolute
registration error at or below 1.5 degrees and processing below 150 ms; it is
not a clinical validation claim.

## Release audit

### Replaceable matcher and future training

### Replaceable matcher and future training

The CPU implementation remains the default. `pentacam/matcher.py` provides an extensible, modular architecture:
1. `AngularMatcher(Protocol)`: callable contract `(ref, curr, mask_ref, mask_curr, max_degrees, reference_cache) -> AngularMatch`.
2. `BaseAngularMatcher(ABC)`: abstract base class providing common input sanitization, checked output verification, and latency/acceptance rate metrics.
3. `ClassicalFFTMatcher`: shift-dependent masked ZNCC with sub-pixel multi-band refinement and angular uncertainty estimation.
4. `LearnedAngularMatcher`: deep learning adapter for PyTorch / ONNX models with automatic graceful fallback to classical matching.
5. `EnsembleMatcher`: dual-system consensus matcher combining classical and neural estimators with tolerance-based divergence rejection.
6. `create_matcher(kind="classical|learned|ensemble", ...)`: factory function for dynamic configuration.

Pass a matcher instance through `SittingRegistrationSession(matcher=m)` or
`CrossModalityRegistrationEngine(matcher=m)`. Geometry, image metadata,
unwrapping, failure handling, and JSON output stay in the existing pipeline.
Adapters must use positive counter-clockwise image angles, return overlap and
quality diagnostics, and abstain on unsupported input. A common output check
rejects nonfinite/out-of-range angles and malformed accepted scores.

### Neural model architecture and training pipeline

`pentacam/models.py` implements a differentiable, rotation-equivariant Siamese architecture:
- `CircularConv2d`: 2D convolutions with circular boundary padding along the angular (0°–360°) axis, avoiding seam artifacts.
- `IrisPolarFeatureNet`: convolutional backbone extracting dense rotation-equivariant descriptors `(B, D, W)`.
- `IrisPolarSiameseNet`: end-to-end Siamese network computing circular cross-correlation via 1D FFT and differentiable soft-argmax over candidate angles.
- Zero-dependency deployment via `model.export_onnx(path)` and direct session integration via `model.as_matcher()`.

`pentacam/training_data.py` loads framework-independent JSONL pairs and provides:
- `CyclotorsionAugmentor`: physiologically plausible clinical augmentations (random circular roll, gamma/contrast variations, pupil dilation stretch, eyelid/lash occlusions, sensor noise).
- `get_pytorch_dataset(...)`: PyTorch Dataset loaded lazily on demand, keeping runtime session imports 100% free of torch dependencies.
- `scripts/train_cyclotorsion_model.py`: end-to-end training script supporting both clinical JSONL manifests and synthetic smoke tests, checkpointing, and ONNX export.

Example manifest format:
```json
{"patient_id":"subject_001","eye_id":"subject_001_OD","laterality":"OD","reference":"reference.png","current":"current.png","rotation_deg":3.5,"split":"train","label_source":"expert"}
```

```python
from pupil_tracking.pentacam.training_data import load_pair_manifest, get_pytorch_dataset
from pupil_tracking.pentacam.matcher import create_matcher

# 1. Classical CPU matcher (default)
matcher = create_matcher("classical")

# 2. Or load neural model
dataset = get_pytorch_dataset("local_data/pairs.jsonl", split="train", augment=True)
```

Allowed splits are `train`, `validation`, and `test`; label sources are
`expert` or `synthetic`. The loader rejects missing files, invalid labels,
conflicting eye metadata, patients crossing splits, and byte-identical images
crossing splits (including renamed copies).

### Sub-pixel accuracy improvements

The annulus unwrapping solves exact ray/ellipse intersections for decentered pupils.
The matcher evaluates three radial bands with continuous parabolic peak interpolation,
eliminating integer bin quantization (reducing band spread on rigid shifts from 0.50° to <0.01°).
`AngularMatch` reports individual continuous `band_angles` and curvature-based `uncertainty_deg`.
Geometry validation rejects crossing pupil/limbus boundaries, and tracking loss
clears temporal smoothing so reacquisition starts fresh.

### Release audit results

The release audit runs the complete Python test suite (622 passed, 14 skipped, 0 failed) and the seated benchmark across 11 real Pentacam iris screenshots with 154 trials per mode:
- **Fixed geometry**: 154/154 accepted (100%), median error 0.0009°, p95 error 0.010°, max error 0.034°, median latency 32.4 ms.
- **Automatic geometry**: 136/154 accepted (88.3%), median error 0.0083°, p95 error 0.044°, max error 0.194°, median detection+registration latency 103.7 ms (<150 ms target).
- **Bounded session**: 150/154 accepted (97.4%), median error 0.0093°, p95 error 0.087°, max error 0.225°, median latency 51.0 ms (~20 FPS).
- **Specificity / False Acceptance**: 0/55 cross-eye unconfirmed pairs accepted (0.0% FAR, 100% true rejection).
- **Accepted cases > 1.0°**: 0 across all modes.

