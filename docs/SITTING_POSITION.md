# Seated iris detection and CPU cyclotorsion registration

## Scope and status

Implemented against the nine supplied Pentacam BMP screenshots and two seated
JPEG screenshots. The ZIP contains duplicates of the nine BMPs, so it does not
increase the independent sample count. Eleven additional ELITA screenshots
were inspected and included as unpaired difficult inputs. Their rendered
overlays are not expert ground-truth annotations, and no confirmed same-eye
cross-device pairs or labelled clinical rotation angles were supplied.

This is an engineering improvement with reproducible numerical validation,
not a clinically validated or manufacturer-approved system. Digital image
rotation cannot measure clinical accuracy, segmentation Dice, sensitivity, or
real sitting-to-supine performance. No models were trained on these eleven
images and no external images or weights were downloaded.
These supplied images were used during development; they are not an independent
held-out validation cohort.

## Installation and use

From the repository root, in a working Python environment:

```powershell
python -m pip install -r requirements-sitting.txt
python -m scripts.register_sitting reference.bmp current.bmp --eye-id eye_001 --laterality OD
```

The command writes `output/sitting/registration.json`; exit status 2 means no
accepted registration. `--threads 1` is the default. It is a two-seated-image
workflow, separate from the existing GUI and live camera pipeline.

An independent CPU camera/video runner is also available:

```powershell
python -m scripts.live_sitting --reference reference.bmp --source eye_video.mp4 --eye-id eye_001 --laterality OD --headless
python -m scripts.live_sitting --reference reference.bmp --source 0 --eye-id eye_001 --laterality OD
```

It expects a seated ocular camera view, not a full face or an ELITA surgical
ring view. It writes per-frame JSONL; rejected frames expose no accepted angle.
The preview displays `NO RELIABLE MATCH` on rejection. Escape stops the preview;
`--max-frames` bounds test runs. Camera buffer reduction is best effort and
actual camera/display latency still needs measurement on the target device.

For applications and live processing, keep one session per stream:

```python
import cv2
from pupil_tracking.pentacam import SittingRegistrationSession

session = SittingRegistrationSession(max_size=640, num_angles=720)
session.set_reference(reference_bgr, eye_id="eye_001", laterality="OD")
result = session.register(current_bgr, eye_id="eye_001", laterality="OD")
if result.valid:
    angle = result.rotation_deg
else:
    reason = result.failure_reason
```

For a current ELITA image, pass `detection=current_eye_detection` using the
existing `EyeDetectionResult` contract, with coordinates in the original
current image. The session scales that geometry without mutating it. Automatic
current geometry is intended for seated images; ELITA ring/surgical images
must use the existing modality-specific detector. Session output timing includes
current preprocessing, detection when needed, and registration, excluding
reference setup, file/camera I/O, display, and recording.

`dynamic=True` enables smoothing of accepted measurements. Rejected frames
do not update the temporal estimate and do not return a final accepted angle.
Replace the reference to start a new acquisition; reference replacement clears
filter state. Do not share a session between threads. Identity fields are
caller-supplied checks, not biometric identity verification.

Positive rotation is counter-clockwise in image coordinates, matching
`cv2.getRotationMatrix2D`. Both camera views must be unmirrored and consistently
oriented. Laterality alone cannot resolve a mirrored image or head/camera roll.
Interpretation as physical ocular torsion requires acquisition calibration.

## What changed

* Detect a large ocular screenshot panel and bound working resolution; map
  pupil, limbus, and feature coordinates back to source pixels.
* Use several dark thresholds, ellipse-shape checks, and measured contours.
  Remove the invented central-pupil fallback.
* Search lateral limbal arcs with robust radial gradients. Remove the upper
  limbus-radius constraint tied to pupil size, which failed on constricted pupils.
  This remains a circular limbus approximation, not full off-axis gaze correction.
* Correct the OpenCV ellipse-axis angle convention and batch feature patches
  and orientation histograms instead of hundreds of Python histogram calls.
* Compare radial iris texture with shift-dependent masked normalized
  correlation. The FFT computes candidate-specific overlap, means, variances,
  and covariance, so exclusion masks do not create a stationary false peak.
* Exclude glints, black margins, and vivid rendered overlays. Use circular
  illumination normalization, bilinear sampling, sub-sample peak interpolation,
  competitor separation, minimum overlap, and agreement among radial bands.
* Cache the reference polar strip by image content, full ellipse geometry,
  sampling settings, and eye side. In-place buffer changes invalidate the cache.
* Remove landmark-histogram confidence inflation and fabricated geometric
  residuals. Results identify a rotation-only model and expose angular diagnostics.
* Skip unused current-frame descriptors in the bounded session; the matcher
  still validates current texture and radial consistency on every frame.
* Decouple CPU registration type imports from optional iris/ML components.
* Batch legacy contour-refinement sampling in bounded chunks while retaining
  every contour point and float64 bilinear weights. Fix the parabolic peak's
  interpolation sign, covered by an analytic quadratic-peak test.

The correlation score and existing geometry confidence values are heuristic
quality indicators, not calibrated probabilities. Thresholds are engineering
defaults, not clinically established tolerances. Always check `valid`; raw
candidate angles in diagnostics must never be treated as accepted measurements.

## Reproduce validation

```powershell
python -m pytest pupil_tracking/tests/test_sitting_registration.py pupil_tracking/tests/test_pentacam_detector.py pupil_tracking/tests/test_cross_modality_registration.py pupil_tracking/tests/test_registration_polar.py pupil_tracking/tests/test_pentacam_types.py -q
python -m scripts.benchmark_sitting --data-root . --baseline-ref e06599afa5977916279d619fee578a22ff7cd40c --report output/sitting/benchmark.json
```

The benchmark deduplicates source files by SHA-256, uses a fixed noise seed,
and reports anonymous image indices. Reference images and screenshots stay
local. No patient filenames, image content, or screenshot thumbnails are
included in the committed report. New source-data ignore rules prevent
accidental staging of the supplied root-level files.

For every seated input, tests apply seven known rotations (-12, -5, -0.75,
0, 0.75, 5, 12 degrees), both rigid-only and with translation, gamma/intensity
changes, and noise. There are 154 trials per mode. The three modes are:

1. Fixed, analytically transformed detected geometry: isolates registration.
2. Automatically redetected geometry: includes segmentation variation.
3. Bounded session: includes screenshot preparation and current detection.

The historical matcher is evaluated using the **same new geometry** to isolate
registration changes. Its timing is not a complete original-application FPS
measurement. Original and new detector timings are separately recorded. Cold
startup and reference setup are excluded from steady-state timing. Report both
acceptance coverage and errors; rejected cases must not disappear from accuracy
claims. Percentile errors are conditional on accepted trials. Timings depend on
hardware, image size, OpenCV build, thermal state, and background workloads.

The 55 comparisons between distinct supplied seated images are **unconfirmed
pair controls**, not a representative biometric false-accept dataset. Rejection
of them does not establish a clinical false-accept rate.

### Measured results, 6 October 2026

Final measurements used Windows, Python 3.14.2, OpenCV 4.13.0, NumPy 2.4.4,
and one OpenCV CPU thread. No concurrent test run was active during this final
benchmark. All 11 seated source images returned detected geometry; this is
detection coverage, not labelled boundary accuracy.

| Workflow | Accepted / attempted | Accepted error p95 | Accepted maximum error | Median current-frame processing |
| --- | ---: | ---: | ---: | ---: |
| Fixed geometry, registration only | 154 / 154 | 0.011 degrees | 0.025 degrees | 38.05 ms |
| Automatic geometry, full-resolution registration | 142 / 154 | 0.071 degrees | 0.475 degrees | 87.62 ms, including detection |
| Bounded CPU session | 151 / 154 | 0.096 degrees | 0.301 degrees | 58.50 ms, including preparation and detection |

Bounded-session p95 processing time was 66.89 ms. Its median corresponds to
approximately **17.1 processed frames/second**, excluding capture, display, and
recording. This is not a measured camera-to-display FPS guarantee. Three trials
were rejected; no accepted bounded-session trial exceeded 1 degree error.

The historical matcher accepted all 154 automatic-geometry trials, including
a maximum error of 16.109 degrees. With identical geometry, median registration
time decreased from 146.62 ms to 36.21 ms. Detector-only median-of-image-medians
decreased from 256.33 ms to 52.67 ms. The 55 unconfirmed cross-image controls
were all rejected. All 11 unpaired ELITA screenshots were rejected by the seated
detector; they require the ELITA-specific geometry path rather than automatic
seated detection.

Final full regression run: **546 passed, 14 skipped**, one warning, no failures
or errors. Initial sandbox temporary-directory failures passed after rerunning
with normal temporary-file access. The legacy contour timing failure was fixed
by batched sampling; the analytic peak-sign test and the complete suite passed.
A six-frame MJPEG headless smoke test completed with six accepted frames and
raw estimated angles approximately -0.006, 1.008, 2.009, 2.999, 3.990, 4.994
degrees for generated targets 0 through 5 degrees. Live camera hardware was
not tested.

See [the full anonymized report](sitting_validation.json) for every trial and
runtime details. These are development-set digital-rotation measurements, not
clinical performance or a held-out validation claim.

## CPU latency update (2026-10-06)

A sequential comparison against an isolated checkout of commit `4aa1d86`
used the same 11 local images, three digital rotations per image, ten repeats,
one OpenCV thread, automatic geometry, 640-pixel processing limit and 720
angular samples. Each implementation processed 330 frames after warm-up.

| Session processing | Previous version | Optimized version |
| --- | ---: | ---: |
| Median latency | 41.92 ms | 36.76 ms |
| P95 latency | 48.76 ms | 41.06 ms |
| Median-equivalent processing throughput | 23.86 FPS | 27.20 FPS |

Median latency fell 12.3%; p95 fell 15.8%. These comparable measurements
supersede the historical timing above for this optimization comparison.
They exclude capture, file decoding, display and initial reference setup;
they are not camera-to-display latency or a guarantee for other laptops.

The matcher sums spectra within radial bands before inverse FFT, reducing
384 inverse transforms to 18 without discarding pixels. It caches the reference
spectrum with content-based invalidation, including mask changes. Pupil
thresholding uses an exact uint8 histogram percentile and OpenCV operations.
Resolution, current-frame geometry detection and rejection thresholds remain
unchanged. One CPU thread remains the recommended default: additional threads
did not consistently improve this workload.

All 462 digital-rotation acceptance decisions matched the previous report;
the largest accepted angle change was below 0.000003 degrees. The bounded
session still accepted 151/154 trials with maximum accepted error 0.301 degrees;
all 55 unconfirmed cross-image controls were rejected. These remain synthetic
development tests, not clinical accuracy measurements.

Verification: full regression suite 546 passed / 14 skipped; subsequent seated
suite 23 passed, including three new cache-mutation cases. Sixty additional
masked numerical comparisons against the archived matcher agreed within 1e-4.
See [the anonymous comparison report](sitting_latency_validation.json).

Reproduce the latency run with:

```powershell
python scripts/benchmark_sitting_latency.py --report output/sitting/latency.json
# Compare against an extracted git-archive checkout containing pupil_tracking:
python scripts/benchmark_sitting_latency.py --implementation-root output/latency_baseline --report output/sitting/latency_before.json
```

Run the implementations sequentially on an otherwise idle computer. The
benchmark reads local data and publishes only anonymous IDs and measurements.

## Detection quality update: clinical review and boundary refinement (2026-10-06)

Following feedback requesting enhanced detection quality for clinical coordinator review,
the detection pipeline was expanded with independent boundary refinement, repeatable 2D
texture keypoints, and explicit anatomical morphology hypotheses:

### 1. Image-supported limbus boundary refinement

* **Beyond concentric circles**: The standard initial limbus estimate assumes an approximately
  circular boundary centered on the pupil. While sufficient for a stable registration envelope,
  real human eyes exhibit limbus decentration and slight ellipticity.
* **Radial transition sampling**: Evaluates radial intensity gradients along 180 radial rays
  spanning the iris-sclera transition, weighted with a gaussian proximity prior.
* **Occlusion awareness**: The superior sector (where upper eyelids and eyelashes obscure the
  limbus) is automatically excluded, preventing false fits to eyelid margins.
* **Constrained ellipse fit**: RANSAC-like iterative ellipse fitting requires bilateral and
  inferior arc support (at least 45 valid support points with residual < 2.5% of radius).
* **Supported vs. Inferred arcs**: Where direct gradient evidence exists, boundaries are
  recorded as observed (`supported_ellipse`). Occluded or missing arcs are clearly marked as
  inferred/extrapolated, never claimed as directly observed edges.

### 2. Repeatable 2D iris texture keypoints

* Rather than arbitrary lattice grid sampling, the detector now uses `cv2.goodFeaturesToTrack`
  to identify actual 2D local texture extrema (eigenvalue response) within a safe, eroded
  annulus mask.
* Safety margins exclude specular Purkinje glints, pupil border artifacts, and uncertain limbal
  transitions.
* Spatial minimum distance constraints ensure well-distributed coverage across visible iris
  quadrants.
* In a known-warp repeatability test across all 11 seated patient images under rotations and
  contrast/brightness changes (`scripts/benchmark_sitting_features.py`), top keypoints achieved
  **63.7% one-to-one spatial repeatability** within a 3-pixel tolerance at 640px width.
  See [feature validation report](sitting_features_validation.json).

### 3. Morphology hypotheses for expert review (crypts and furrows)

* Multi-scale Hessian analysis (`sigma = 1.5, 3.0, 4.5`) identifies local morphological patterns:
  - **Crypt-like candidates (`C?`)**: Dark, compact depressions with positive Hessian eigenvalues
    in the inner-to-mid iris zone.
  - **Furrow-like candidates (`F?`)**: Peripheral dark ridges oriented tangentially along iris
    circumference in the outer third of the iris.
* **Strict clinical boundary**: Near-infrared image appearance alone cannot conclusively verify
  anatomical crypts or furrows without histological or multi-angle slit-lamp confirmation.
  These points are exposed with `anatomy_verified = False` and `confidence = 0.0` strictly as
  unverified hypotheses for expert ophthalmic review, and are deliberately excluded from
  affecting cyclotorsion registration.

### 4. Architectural separation: Review geometry vs. Registration envelope

* Experimental evaluation revealed that allowing single-frame refined ellipse parameters to
  directly distort the polar unwrapping grid increased angular measurement jitter.
* To preserve high registration accuracy, cyclotorsion matching retains the validated, robust
  circular geometry envelope, while the refined boundary and feature points are provided for
  visualization, coordinate reporting, and physician review.

### 5. Physician review visualization

A dedicated review renderer generates a standardized 3-panel comparative diagnostic card:

```powershell
python -m scripts.review_sitting --input "sitting 1.jpeg" --output-dir output/doctor_review/image_1
python -m scripts.review_sitting --input "sitting 2.jpeg" --output-dir output/doctor_review/image_2
```

Outputs:
* **Panel 1 (Original)**: Unmodified input crop.
* **Panel 2 (Boundary support + keypoints)**: Green pupil ellipse, cyan solid supported limbus,
  dashed gray inferred limbus, amber local texture keypoints.
* **Panel 3 (Anatomical review)**: Blue `C?` crypt-like candidates and magenta `F?` furrow-like
  candidates for ophthalmic inspection.
* Formatted JSON diagnostics detailing support point counts, fit residual in pixels, and keypoint
  coordinates.

## Remaining validation required

Acquire confirmed same-eye seated/current pairs with laterality, camera
orientation, expert pupil/limbus contours, and rotation labels with inter-rater
uncertainty. Keep patients separated across tuning and held-out evaluation.
Include real dilation, off-axis gaze, blinks, eyelid occlusion, contact/suction
deformation, motion blur, weak iris texture, spectral changes, and multiple
devices. Measure boundary errors, angular bias and absolute error, accepted
coverage, false acceptance, abstention, and confidence calibration by subgroup.
Use target CPU-only laptops and capture-to-display p50/p95/p99 latency and frame
drops. Agree acceptance criteria with the intended clinical/manufacturer team
before describing the system as production-ready for treatment.

Research context: seated-to-supine cyclotorsion is estimated by comparing
registered images, not by assigning an angle to an isolated seated image.
See [posture-related ocular cyclotorsion study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7005750/)
and [OpenCV motion-analysis documentation](https://docs.opencv.org/4.5.1/d7/df3/group__imgproc__motion.html).

