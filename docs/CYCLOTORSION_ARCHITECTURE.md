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

The release audit runs the complete Python test suite, the seated benchmark,
RGB/grayscale and image-size combinations, malformed input cases, repeated
determinism checks, metadata mismatch checks, and integrated GUI/stream tests.
The latest run passed **555 tests with 14 skips**. The production
cross-modality smoke case recovered 2.504 degrees from a 2.500 degree textured
synthetic rotation (0.004 degree error, 0.982 confidence). The legacy
multi-stream compatibility sweep had 16/20 cases within 1.5 degrees; its four
weak cases remain visible audit failures and are not used to claim Phase 2
performance.
