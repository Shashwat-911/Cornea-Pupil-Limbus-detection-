# Pupil & Limbus Detector — Medevplus IXcentai (Surgical Grade)

A deep-learning + classical-computer-vision system that detects and measures the
**pupil** (dark central aperture) and the **limbus** (iris–sclera boundary) in eye
images and video. It reports geometry in pixels and millimetres, tracks smoothly
across video, adapts to grayscale/IR sources, and handles the femtosecond-laser
**suction ring** used in eye surgery.

- **Package version:** 2.0.0 · **GUI version:** 2.3
- **Primary model:** U-Net with a ResNet-34 encoder (3-class: background / pupil /
  iris; optional 4-class adds the suction ring)
- **Runtime:** ONNX Runtime in production (no CUDA toolkit needed), PyTorch in
  development

> **Two companion documents:**
> - This **README** — install, run, train, and use the system.
> - **`PROJECT_COMPLETE_ANALYSIS.md`** — exhaustive, code-verified breakdown of
>   every module (read that for internals).

---
##Real time Recording of Application


https://github.com/user-attachments/assets/135ec84c-2733-41a3-b81c-079041c7823a


---

## Table of Contents

1. [What it does](#what-it-does)
2. [Why the hybrid design](#why-the-hybrid-design)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [The GUI](#the-gui)
7. [Command-line reference](#command-line-reference)
8. [How detection works](#how-detection-works)
9. [Outputs & result schema](#outputs--result-schema)
10. [Training your own model](#training-your-own-model)
11. [Exporting to ONNX](#exporting-to-onnx)
12. [Annotation workflow](#annotation-workflow)
13. [Project structure](#project-structure)
14. [Configuration](#configuration)
15. [Testing](#testing)
16. [Troubleshooting](#troubleshooting)
17. [Known limitations](#known-limitations)

---

## What it does

Given an eye image or a video/camera stream, the system produces — per frame — a
structured result containing:

- **Pupil** — center, radius, full ellipse (semi-axes + angle), confidence, quality.
- **Limbus** — same geometry; the outer iris boundary and the anatomical reference.
- **Corneal center & offset** — the pupil-to-limbus decentration (magnitude +
  angle), clinically important for centering surgical treatments.
- **Suction ring** — whether a femto-laser ring is docked, plus its geometry.
- **Calibration** — pixel↔millimetre scale, auto-derived from the limbus (≈ 11.5 mm)
  or a known ring diameter, so measurements are in real units.
- **Quality grade** — `SURGICAL` / `CLINICAL` / `RESEARCH` / `INSUFFICIENT` /
  `NO_DETECTION`, telling you how much to trust each frame.

Typical uses: cataract/refractive surgery guidance, pupillometry, eye-tracking
research, and dataset annotation.

---

## Why the hybrid design

Surgical eye imagery is hard: reflections, red LED markers, suction rings, IR
illumination, and motion blur all break naive detectors. Instead of one method,
the system layers several so it **degrades gracefully** rather than failing hard:

1. **U-Net segmentation** (learned, appearance-robust) — the primary detector.
2. **Classical CV fallback** (thresholding + contours) — when ML output is weak.
3. **Robust geometric fitting** (RANSAC circle/ellipse + sub-pixel refinement) —
   turns noisy masks into precise geometry (~0.05 px claimed accuracy).
4. **Temporal smoothing** (Kalman) — removes jitter across video.
5. **Ring-aware preprocessing** — detects the suction ring first, then runs a
   pipeline tuned for a **docked** vs **pre-docked** eye (they look completely
   different).

---

## Requirements

- **OS:** Windows 10/11, macOS 10.15+, or Linux
- **Python:** 3.8+
- **GPU:** optional (NVIDIA CUDA or Apple MPS). CPU works, just slower.
- **RAM:** 8 GB minimum; 16 GB+ recommended for training

**Python dependencies** (`requirements.txt`): `torch`, `torchvision`,
`segmentation-models-pytorch`, `opencv-python`, `Pillow`, `albumentations`,
`scipy`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`, `pytest`.

> **Also needed for ONNX / production inference (not yet in `requirements.txt`):**
> `pip install onnx onnxruntime` (or `onnxruntime-gpu`). The optimized/production
> path and `convert_to_onnx.py` require these.

Tkinter (for the GUI) ships with standard Python on Windows/macOS; on Linux install
`python3-tk`.

---

## Installation

```bash
git clone <your-repo-url>
cd Pupil-Limbus-detector-main

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install onnx onnxruntime        # for ONNX/production inference

# Verify
python -c "import torch, cv2, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Place model weights under `models/` (they are git-ignored):
`models/best_model.pth` for PyTorch, and/or `models/onnx/segmentation.onnx`
(+ `segmentation_quantized.onnx`) for ONNX. If no model is present, the app still
launches but returns empty detections.

---

## Quick start

```bash
# Launch the interactive GUI (default mode)
python launch_gui.py

# Analyze a single image (console report + overlay)
python launch_gui.py image -i path/to/eye.jpg

# Force a grayscale/IR look
python launch_gui.py image -i eye.jpg --grayscale force

# Process a video file and save an annotated output
python launch_gui.py video -i clip.mp4 -o clip_tracked.mp4

# Live webcam
python launch_gui.py camera
python launch_gui.py camera --camera-id 1
```

Everything runs through **`launch_gui.py`** — the single entry point. The first
line of output reports the auto-detected runtime profile (which sets sensible
FP16 / compile / resolution defaults for your machine).

---

## The GUI

`python launch_gui.py` opens a dark-themed Tkinter application.

**Modes:** open a single image, a folder of images, a video file, or a live camera
— each in a *classic* (full-accuracy) or *optimized* (real-time) pipeline.

**Right-panel tabs:**
- **Measurements** — live cards for Quality, Tracking, Latency, Pipeline, plus
  Pupil, Limbus, Corneal Centre & Offset, and Calibration values.
- **Details** — the raw result dump.
- **⚙ Settings** — grayscale mode, pipeline preset (max_accuracy / balanced /
  low_latency), FP16 / compile / optimized toggles, resolution / stride / FPS,
  ROI and Kalman-noise controls, and a **Rebuild Engine** button.

**Manual overrides:** you can draw a manual ROI (region of interest) or a manual
suction ring on the canvas with the mouse; arrow keys nudge the ROI.

**Recording & export:** record annotated video to disk, and export results as CSV,
JSON, or a snapshot image.

**Keyboard shortcuts:**

| Shortcut | Action | Shortcut | Action |
|----------|--------|----------|--------|
| `Ctrl+O` | Open image | `Ctrl+R` | Start recording |
| `Ctrl+V` | Open video | `Ctrl+Shift+R` | Toggle recording |
| `Ctrl+Q` | Quit | `G` / `Shift+G` | Cycle grayscale mode |
| `Space` | Pause / resume | `Return` / `Esc` | Confirm / cancel manual selection |
| arrows | Nudge manual ROI | | |

---

## Command-line reference

```
python launch_gui.py [gui|image|video|camera] [options]
```

Positional `mode` defaults to `gui`.

**Input / output**

| Flag | Default | Meaning |
|------|---------|---------|
| `--input, -i PATH` | – | input image or video |
| `--output, -o PATH` | – | output path |
| `--model, -m PATH` | – | model weights (`.pth`) |
| `--camera-id ID` | `0` | camera device index |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

**Ring detection**

| Flag | Default | Meaning |
|------|---------|---------|
| `--ring-mode` | `auto` | `auto` / `docked` / `pre_docked` |
| `--ring-classifier PATH` | `models/ring_classifier.pth` | ring model |
| `--show-ring / --no-show-ring` | on | draw the ring outline |

**Grayscale**

| Flag | Default | Meaning |
|------|---------|---------|
| `--grayscale MODE` | `off` | `off` = RGB passthrough · `auto` = detect & enhance · `force` = always IR look |

**Pipeline**

| Flag | Default | Meaning |
|------|---------|---------|
| `--optimized / --no-optimized` | on | use the fast real-time pipeline |
| `--fp16 / --no-fp16` | machine profile | FP16 half precision |
| `--compile / --no-compile` | machine profile | `torch.compile` JIT |

**Video**

| Flag | Default | Meaning |
|------|---------|---------|
| `--stride N` | `1` | process every Nth frame |
| `--resolution PX` | profile (320) | inference resolution |
| `--target-fps FPS` | profile | target processing FPS |

**ROI & tracking**

| Flag | Default | Meaning |
|------|---------|---------|
| `--roi / --no-roi` | on | ROI tracking |
| `--roi-cache N` | `5` | ROI cache lifetime (frames) |
| `--kalman-process-noise F` | `0.03` | Kalman process noise |
| `--kalman-measure-noise F` | `0.1` | Kalman measurement noise |

Defaults marked "profile" are chosen by `runtime_profile.py` from your hardware:
on a CUDA machine, FP16 + compile default **on**; on CPU they default **off**.

**Headless video** (bypasses the GUI, direct `OptimizedVideoProcessor`):

```bash
python scripts/process_video.py -i clip.mp4 -o out.mp4 --csv out.csv
python scripts/process_video.py -i clip.mp4 --benchmark
python scripts/process_video.py --camera 0 --preview
```

---

## How detection works

The classic path is `pupil_tracking.core.UnifiedDetector.detect()`. Each frame goes
through nine stages (details in `PROJECT_COMPLETE_ANALYSIS.md` §6.1):

1. **Format + grayscale normalisation** — everything becomes 3-channel BGR;
   grayscale/IR handling per the `--grayscale` mode.
2. **Ring detection** — decide docked vs pre-docked (drives everything after it).
3. **Ring-aware preprocessing** — pipeline tuned to the scene type.
4. **ML segmentation** — U-Net produces a per-pixel label map
   (background / pupil / iris / optional ring).
5. **Smart fitting** — `SmartContourFitter` fits a circle or ellipse (chosen
   automatically) with RANSAC + sub-pixel refinement.
6. **Classical fallback** — thresholding + contours if ML missed pupil or limbus.
7. **Cross-validation** — reject anatomically impossible pupil/limbus pairs.
8. **Calibration + mm values + corneal center/offset**.
9. **Quality grading** — combine confidences into a clinical tier.

The real-time path (`OptimizedVideoProcessor` + `FastInference`) is a leaner,
faster version (FP16, single-scale 320 px, threaded decode-ahead, batching, ROI
tracking, overload protection) for video and camera.

**Backend selection is automatic:** ONNX Runtime first (small, fast, portable),
PyTorch as the development fallback, and a dummy engine so the app never crashes
just because a model is missing.

---

## Outputs & result schema

Every detection returns an `EyeDetectionResult` (`pupil_tracking/utils/types.py`):

```python
result.pupil.detected          # bool
result.pupil.ellipse.center_x  # float (px)
result.pupil.ellipse.center_y
result.pupil.ellipse.radius    # read-only property (mean of semi-axes)
result.pupil.radius_mm         # if calibrated
result.pupil.confidence        # 0..1
result.pupil.quality           # DetectionQuality enum

result.limbus.*                # same shape as pupil
result.corneal_center.offset_magnitude_mm   # decentration
result.corneal_center.offset_angle_deg
result.calibration.mm_per_px
result.overall_quality         # SURGICAL / CLINICAL / RESEARCH / …
result.overall_confidence
```

Ring information (`ring_status`, `ring_center`, `ring_radius`, `ring_dot_count`,
`image_category`, …) is attached dynamically and included in `to_dict()`.

**Programmatic use:**

```python
import cv2
from pupil_tracking.core.detector import UnifiedDetector

detector = UnifiedDetector(model_path="models/best_model.pth")
image = cv2.imread("eye.jpg")
result = detector.detect(image, source="eye.jpg")

if result.has_both:
    p = result.pupil.ellipse
    print(f"Pupil center=({p.center_x:.1f},{p.center_y:.1f}) r={p.radius:.1f}px")
    if result.calibration.calibrated:
        print(f"Pupil diameter = {result.pupil.radius_mm*2:.2f} mm")
    print(f"Decentration = {result.corneal_center.offset_magnitude_mm:.2f} mm")
    print(f"Quality: {result.overall_quality.value}")
```

**CSV/JSON exports** flatten this schema (via `result_to_dict()`), zero-filling
fields when nothing is detected, and include mm conversions when calibrated.

---

## Training your own model

The model is a U-Net (ResNet-34 encoder), 3-class (background/pupil/iris) or
4-class (adds suction_ring). Training uses AMP mixed precision, cosine-annealing
LR, early stopping on validation IoU, a composite loss (CE + Dice + Boundary),
and heavy augmentation.

```bash
# 1) Check your data first
python scripts/check_training_data.py

# 2) Train (3-class)
python scripts/train_model.py \
    --epochs 200 --batch-size 4 --lr 1e-4 \
    --input-size 512 --device cuda \
    --annotation-path clinical_data/annotations/annotations.json \
    --image-dir clinical_data/training_data/images \
    --mask-dir  clinical_data/training_data/masks

# 4-class (ring-aware) — requires ring labels
python scripts/train_model.py --num-classes 4 --ring-labels clinical_data/ring_labels.json

# Quick single-epoch sanity check
python scripts/run_epoch.py
```

Outputs: `models/best_model.pth` + `models/checkpoint_meta.json` (epoch, best val
IoU), plus an audit log. The bundled reference checkpoint reached **val IoU
0.943** at epoch 30.

**Loss options:** `--loss-type {composite,ce_dice,focal_dice,ce}`, `--use-focal`,
`--focal-gamma`, `--class-weights`.

**Alternative entry** — `train_production.py` bridges the annotation format from
`annotate_live_video.py` into the training schema and trains in one step. No local
GPU? Use `train_colab.ipynb` (generated by `gen_notebook.py`).

**Grayscale robustness fine-tune** — `scripts/finetune_grayscale.py` fine-tunes an
existing model for grayscale/IR inputs with a safety gate (only saves if RGB
accuracy doesn't regress and grayscale Dice clears `--min-gray-dice`).

---

## Exporting to ONNX

Production inference uses ONNX Runtime (no PyTorch/CUDA toolkit required).

```bash
# Simple: export just the segmentation model
python scripts/export_onnx.py --model models/best_model.pth --resolution 320 --verify

# Full: segmentation + ring classifier, with INT8 quantization + manifest
python scripts/convert_to_onnx.py
```

`convert_to_onnx.py` writes `models/onnx/segmentation.onnx` (~93 MB),
`segmentation_quantized.onnx` (~23 MB), and `manifest.json` (sizes + sha256). The
backend then prefers the quantized model automatically.

**Build a distributable app (Windows):** `build_app.bat` runs PyInstaller (needs
the ONNX models present first).

---

## Annotation workflow

To build training data (image + segmentation mask pairs):

```bash
# Point-and-click annotation tool (pupil/limbus/ring points → ellipse → JSON)
python scripts/annotate_data.py

# Advanced live-video annotator with edge-snapping + incremental retraining
python scripts/annotate_live_video.py annotate clip.mp4

# Label ring presence per image (R=present, N=absent, P=partial)
python scripts/annotate_ring_data.py --image-dir clinical_data/training_data/images

# Rasterise annotations into training masks
python scripts/generate_masks.py

# Validate the dataset before training
python scripts/verify_data.py
python scripts/check_training_data.py
```

Annotations are stored as JSON (boundary points + ellipse parameters). Masks are
PNGs where pixel labels are 0=background, 1=pupil, 2=iris (3=suction_ring for
4-class). Train/val splitting is done at the **image level** so augmented copies of
the same eye never leak across the split.

---

## Project structure

```
launch_gui.py              Single entry point (CLI + GUI)
train_production.py        Annotation→training bridge + train
gen_notebook.py            Generates train_colab.ipynb
build_app.bat              PyInstaller build (Windows)
requirements.txt           Dependencies
manual_ring_priors.json    Runtime cache of learned manual-ring priors

pupil_tracking/            Main package
├── core/                  Detection engine (UnifiedDetector, SmartContourFitter,
│                          ring detector, corneal center, ROI, ellipse fitter)
├── ml/                    Model, training, inference (PyTorch + ONNX), postprocess
├── preprocessing/         Grayscale, normalize, reflection/ring/red-light filters
├── video/                 OptimizedVideoProcessor, Kalman tracker, smoother
├── interface/             Tkinter GUI, frame recorder, theme
├── calibration/           Pixel↔mm calibration
├── annotation/            Annotation tool
├── utils/                 config, types (result schema), logger, runtime_profile
└── tests/                 pytest suite

scripts/                   Training, export, benchmarking, annotation, diagnostics
models/                    Weights (git-ignored) + onnx/ + checkpoint_meta.json
clinical_data/             Sample images + annotations + masks
```

A complete, per-file breakdown with line counts and roles is in
`PROJECT_COMPLETE_ANALYSIS.md` §4.

---

## Configuration

Central configuration is a tree of dataclasses in
`pupil_tracking/utils/config.py`, accessed through a global singleton:

```python
from pupil_tracking.utils.config import get_config, set_config

cfg = get_config()
cfg.model.input_size          # 512
cfg.model.device = "cuda"
cfg.detection.min_pupil_confidence = 0.30
cfg.video.enable_kalman = True
set_config(cfg)

cfg.save("my_config.json")    # persist
```

Key groups: `model`, `detection`, `fitting`, `video`, `calibration`, `paths`,
`training`, `ring`, `grayscale`, `measurement_stabilization`, `subpixel`. Call
`cfg.apply_video_mode()` before video processing (it relaxes thresholds and
enables Kalman). Full field list in `PROJECT_COMPLETE_ANALYSIS.md` §13.1.

Most CLI flags override the corresponding config values for that run.

---

## Testing

```bash
python -m pytest pupil_tracking/tests/ -v
```

| Test | Needs a model? | Needs data? |
|------|----------------|-------------|
| `test_grayscale.py` | no | no |
| `test_deterministic_ring_detector.py` | no | no (synthetic) |
| `test_manual_roi.py` | no | no |
| `test_video_pipeline.py` | yes (else skips) | no |
| `test_clinical_accuracy.py` | yes (else skips) | yes: `clinical_data/clean/` |

Tests that need a model or data skip cleanly when they're absent, so the suite runs
in any environment.

---

## Troubleshooting

- **GPU not used?** `python -c "import torch; print(torch.cuda.is_available())"`.
  If `False`, install a CUDA build of PyTorch or run on CPU (slower).
- **"No inference backend available"** — no model found. Provide
  `models/best_model.pth` or `models/onnx/*.onnx`, and `pip install onnxruntime`.
- **ONNX errors / `convert_to_onnx.py` fails** — `pip install onnx onnxruntime`;
  on torch ≥ 2.6 the script deliberately uses the legacy exporter.
- **`torch.compile` errors on Windows** — needs a C compiler; run
  `--no-compile` (`FastInference` also falls back to eager automatically).
- **GUI won't start on Linux** — install Tkinter: `sudo apt install python3-tk`.
- **Grayscale/IR footage looks wrong** — try `--grayscale auto` or `force`, or
  press `G` in the GUI to cycle modes.
- **`scripts/debug_single_image.py` crashes on import** — it references modules
  that no longer exist; use `scripts/diagnose_detection.py` instead.

---

## Known limitations

- `onnx` / `onnxruntime` are required for production inference but not yet listed
  in `requirements.txt` — install them manually.
- `scripts/debug_single_image.py` is broken (stale imports).
- A few clinical constants are hardcoded rather than configurable (e.g. the
  pre-docked limbus-radius correction and the 11.5 mm corneal-diameter assumption).
- See `PROJECT_COMPLETE_ANALYSIS.md` §18 for the full, code-verified list of known
  issues and dead code.

---

*For internals — module-by-module algorithms, the full config and result schema,
data flow, and the known-issues list — see `PROJECT_COMPLETE_ANALYSIS.md`.*




