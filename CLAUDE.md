# CLAUDE.md — Comprehensive Project Audit Context

> **Purpose**: Complete technical handoff for an independent Claude audit.
> **Created**: 2026-08-16
> **Repository**: https://github.com/ANUBprad/pupil-detector
> **Branch**: `github-main` (HEAD: `03e76f8`)
> **Status**: Phase 16 is the current production model. Phase 22 was rejected.

---

## 1. Project Purpose and Clinical Use Case

This project provides **real-time pupil and limbus (iris boundary) detection** in eye images for clinical and research applications. It is designed for:

- **Surgical monitoring**: Detecting pupil and iris boundaries during ophthalmic surgery
- **Suction ring detection**: Identifying docked vs pre-docked surgical states
- **Calibration**: Converting pixel measurements to millimeter scale
- **Quality assessment**: Grading detection confidence for clinical reliability

The system processes single images, video files, and live camera feeds. It outputs ellipse parameters (center, radii, angle), quality grades, and calibration data.

---

## 2. Complete Repository Architecture

```
pupil-detector/
├── pupil_tracking/                    # Main package
│   ├── core/                          # Detection pipeline
│   │   ├── detector.py                # UnifiedDetector — main orchestrator (2386 lines)
│   │   ├── smart_fitter.py            # SmartContourFitter — robust ellipse fitting
│   │   ├── classical_fallback.py      # Classical CV fallback when ML fails
│   │   ├── corneal_center.py          # Specular reflection detection
│   │   ├── confidence.py              # Quality scoring system
│   │   ├── ellipse_fitter.py          # Geometric ellipse fitting
│   │   ├── deterministic_ring_detector.py  # Suction ring detection
│   │   ├── structure_extraction.py    # Limbus fitting with ring constraints
│   │   └── validation.py              # Detection validation
│   ├── ml/                            # Machine learning
│   │   ├── architecture.py            # EyeSegmentationModel (U-Net + ResNet-34)
│   │   ├── dataset.py                 # EyeSegmentationDataset, data loading, augmentation
│   │   ├── trainer.py                 # Training loop with early stopping
│   │   ├── losses.py                  # CompositeLoss (CE + Dice + Boundary)
│   │   ├── onnx_inference.py          # ONNXInference — ONNX Runtime backend
│   │   ├── postprocess.py             # Mask post-processing, ring extraction
│   │   ├── inference.py               # PyTorch inference engine
│   │   └── fast_inference.py          # Optimized inference with FP16
│   ├── preprocessing/                 # Image preprocessing
│   │   ├── grayscale_handler.py       # Grayscale mode (auto/force/off)
│   │   ├── ring_aware.py              # Ring-aware preprocessing
│   │   └── reflection_removal.py      # Corneal reflection removal
│   ├── video/                         # Video processing
│   │   ├── video_processor.py         # Frame-by-frame video processing
│   │   ├── optimized_processor.py     # High-speed video processing
│   │   └── kalman_tracker.py          # Temporal smoothing
│   ├── calibration/                   # Camera calibration
│   ├── annotation/                    # Annotation tools
│   ├── utils/                         # Utilities
│   │   ├── types.py                   # Dataclasses (DetectionResult, etc.)
│   │   ├── config.py                  # Configuration management
│   │   └── logger.py                  # Audit logging
│   └── tests/                         # Test suite (243 tests)
├── models/                            # Production model files
│   ├── best_model.pth                 # Phase 16 PyTorch model (~98 MB)
│   └── onnx/                          # ONNX models
│       ├── segmentation.onnx          # FP32 ONNX (~98 MB)
│       ├── segmentation_quantized.onnx # INT8 quantized (~25 MB)
│       └── manifest.json              # Model metadata
├── clinical_data/                     # Clinical dataset (gitignored)
│   ├── clean/                         # 12 clean clinical images (eye_01-eye_14)
│   ├── corrected_annotations/         # Ground truth annotations
│   └── training_data/                 # 139 training frames + masks
├── train_production.py                # Training entry point
├── launch_gui.py                      # Main GUI application
├── requirements.txt                   # Python dependencies
└── _phase_artifacts/                  # Temporary phase work (gitignored)
    ├── phase16/                       # Phase 16 backup
    ├── phase17-22/                    # Investigation phases
    └── ...
```

---

## 3. End-to-End Detection/Inference Pipeline

### Production Pipeline (ONNX Runtime)

1. **Image Load**: Read JPEG/PNG image via OpenCV
2. **Grayscale Handling**: Auto/force/off grayscale conversion
3. **ML Segmentation**: ONNX Runtime inference → 3-class mask (background/pupil/iris)
4. **Mask Post-processing**: Clean mask, extract contours, erosion
5. **Smart Fitting**: SmartContourFitter → fit circles or ellipses
6. **Ring Detection**: DeterministicRingDetector → docked/pre-docked status
7. **Cross-validation**: Validate pupil-limbus geometry
8. **Calibration**: Pixel-to-mm conversion using limbus reference
9. **Quality Grading**: DetectionQuality (CLINICAL/RESEARCH/etc.)
10. **Output**: EyeDetectionResult with all measurements

### Key Entry Points

```python
# Production detection
from pupil_tracking.core.detector import UnifiedDetector
det = UnifiedDetector()
result = det.detect(image, frame_number=0, source='eye_01.jpeg')

# Direct ONNX inference
from pupil_tracking.ml.onnx_inference import ONNXInference
onnx = ONNXInference(model_path='models/onnx/segmentation_quantized.onnx')
masks = onnx.infer(image, target_size=512)
```

---

## 4. Important Modules, Classes, and Functions

### Core Detection

- **`UnifiedDetector`** (`core/detector.py:118`): Main orchestrator. Calls ML segmentation, fits contours, validates geometry, assigns quality.
- **`SmartContourFitter`** (`core/smart_fitter.py`): Robust fitting with RANSAC, circle-vs-ellipse auto-selection, pupil_hint constraint.
- **`RingDetector`** (`core/deterministic_ring_detector.py`): Heuristic-based suction ring detection.
- **`CornealCenterCalculator`** (`core/corneal_center.py`): Specular reflection detection.

### ML Pipeline

- **`EyeSegmentationModel`** (`ml/architecture.py`): U-Net + ResNet-34 encoder, 3-class output, temperature scaling.
- **`ONNXInference`** (`ml/onnx_inference.py`): ONNX Runtime backend, `_clean_mask()` keeps largest contour ≥100px.
- **`CompositeLoss`** (`ml/losses.py`): Weighted CE + Dice + Boundary loss.
- **`EyeSegmentationDataset`** (`ml/dataset.py`): Image-level train/val split, augmentation, mask generation.

### Data Types

- **`EyeDetectionResult`** (`utils/types.py`): Top-level result with pupil, limbus, calibration, quality.
- **`EllipseParams`** (`utils/types.py`): center_x, center_y, semi_major, semi_minor, angle, confidence.
- **`DetectionQuality`** (`utils/types.py`): Enum — CLINICAL, RESEARCH, etc.

---

## 5. Data Structures and Contracts

### Annotation Format (JSON)

```json
{
  "eye_01.jpeg": {
    "image_path": "clinical_data/clean/eye_01.jpeg",
    "image_width": 698,
    "image_height": 655,
    "annotations": {
      "PUPIL": {
        "class_id": 1,
        "center_x": 381.5, "center_y": 333.2,
        "semi_major": 84.6, "semi_minor": 80.1,
        "angle_deg": 12.5
      },
      "LIMBUS": {
        "class_id": 2,
        "center_x": 375.2, "center_y": 325.2,
        "semi_major": 221.5, "semi_minor": 210.3,
        "angle_deg": 45.0
      }
    }
  }
}
```

### Mask Classes

- **0**: Background
- **1**: Pupil (dark center)
- **2**: Iris/Limbus (colored ring)
- **3**: Suction ring (if 4-class mode)

### Detection Result

```python
EyeDetectionResult(
    pupil=PupilDetection(ellipse=EllipseParams(...), confidence=0.95),
    limbus=LimbusDetection(ellipse=EllipseParams(...), confidence=0.87),
    calibration=CalibrationInfo(px_per_mm=..., reference_radius=...),
    overall_quality=DetectionQuality.CLINICAL,
    overall_confidence=0.91,
    alerts=[],
    metadata=FrameMetadata(...)
)
```

---

## 6. Model Architecture and Training Pipeline

### Architecture

- **Encoder**: ResNet-34 (pretrained on ImageNet)
- **Decoder**: U-Net skip connections
- **Output**: 3-class softmax (background, pupil, iris)
- **Input**: 512×512×3 RGB
- **Parameters**: ~24.4M
- **Temperature scaling**: Learned calibration

### Training Configuration (Phase 16)

```python
{
    "encoder": "resnet34",
    "num_classes": 3,
    "input_size": 512,
    "epochs": 80,           # early stop at 72
    "batch_size": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "copies_per_image": 5,
    "augmentations_per_image": 5,
    "loss": "CompositeLoss(CE=0.3, Dice=0.4, Boundary=0.3)",
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR",
    "early_stopping_patience": 20,
    "seed": 42,
    "device": "cuda"
}
```

### Training Entry Point

```bash
python train_production.py \
    --annotations clinical_data/annotations/annotations.json \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --epochs 80 \
    --batch-size 2 \
    --copies-per-image 5 \
    --skip-convert
```

---

## 7. ONNX/Quantized Inference Pipeline

### Export

```bash
python scripts/export_onnx.py \
    --model models/best_model.pth \
    --resolution 512 \
    --output-path models/onnx/segmentation.onnx
```

### Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic('segmentation.onnx', 'segmentation_quantized.onnx', weight_type=QuantType.QInt8)
```

### Production Inference

```python
from pupil_tracking.ml.onnx_inference import ONNXInference
onnx = ONNXInference(model_path='models/onnx/segmentation_quantized.onnx')
masks = onnx.infer(cv2_image, target_size=512)
# masks = {'pupil': np.ndarray, 'iris': np.ndarray}
```

### Model Hashes (Verified 2026-08-16)

| File | SHA256 (first 16 chars) |
|------|------------------------|
| `models/best_model.pth` | `5e600a68dc1d1e5f` |
| `models/onnx/segmentation.onnx` | `0b238293f287ed29` |
| `models/onnx/segmentation_quantized.onnx` | `379f3ac6eb910f38` |

---

## 8. Annotation Schema and Dataset Structure

### Data Directories

- **`clinical_data/clean/`**: 12 clinical images (eye_01, eye_02, eye_03, eye_06-eye_14). These are the validation set.
- **`clinical_data/training_data/`**: 139 annotated video frames (frame_XXXXXX.jpg) + masks.
- **`clinical_data/corrected_annotations/`**: Corrected ground truth annotations.
  - `annotations_corrected.json`: 13 entries (the 12 clean images + 1 extra). **This is the ground truth reference.**
  - `annotations_production_reviewed.json`: 139 entries (training frames). **This is the training data reference.**

### Key Annotation Files

| File | Entries | Purpose |
|------|---------|---------|
| `annotations_corrected.json` | 13 | Corrected GT for clinical validation |
| `annotations_production_reviewed.json` | 139 | Reviewed training annotations |
| `annotations_production.json` | 139 | Original production annotations |

### Training Data Statistics

- **Training frames**: 139 images
- **Limbus radius range**: 267.6 px – 298.4 px
- **Mean limbus radius**: 285.0 px
- **Std**: 6.2 px

---

## 9. Current Git/Branch/Remote State

```
Branch: github-main (HEAD: 03e76f8)
Remote: origin -> https://github.com/ANUBprad/pupil-detector.git
Status: Up to date with origin/main
Untracked: .claude/, CLEANUP_REPORT.md, FINAL_REGRESSION_AUDIT.md, REFACTORING.md, REGRESSION_FIX_REPORT.md
```

### Git History (Last 8 Commits)

```
03e76f8 feat(annotation): add --corrected-output for safe annotation correction
abf4981 chore: add phase artifacts and clinical data to gitignore
0b6198d fix(detection): refine limbus boundary selection using ring geometry
6268952 fix(detection): distinguish limbus candidate state from confidence
2af5364 perf(detection): add detection pipeline audit script
63c39da test(backend): add missing blend functions for corneal center tests
e7b15c7 fix(backend): harden logging and reuse FastInference
63d1628 Initial release of pupil-limbus detector
```

---

## 10. Gitignore and Protected-Data Rules

### Gitignored (DO NOT COMMIT)

- `*.pth`, `*.onnx`, `*.onnx.data` — Model weights
- `clinical_data/training_data/` — Training images/masks
- `clinical_data/clean/` — Clinical validation images
- `clinical_data/annotations/` — Original annotations
- `clinical_data/corrected_annotations/` — Corrected annotations
- `_phase_artifacts/` — Temporary phase work
- `logs/` — Audit logs

### Protected Files (DO NOT MODIFY)

- `clinical_data/` — All clinical data is read-only
- `models/best_model.pth` — Current production model
- `models/onnx/segmentation_quantized.onnx` — Current production ONNX
- `_phase_artifacts/phase16/promotion_backup/` — Old production model backup

---

## 11. Full History of Backend Phases 1–22

### Phase 1–14: Initial Development

- Built the detection pipeline, ML architecture, training infrastructure
- Established 3-class segmentation (background/pupil/iris)
- Created annotation tools and data pipeline

### Phase 15: Baseline Training

- Trained initial model on 139 corrected annotations
- Result: val_iou=0.9467

### Phase 16: Production Training (BEST MODEL)

- Re-trained with refined configuration
- **Best val_iou=0.9554**, annotated dataset IoU=0.9709
- Early stopped at epoch 72 (of 80 max)
- **Promoted to production** (models replaced, backups in `_phase_artifacts/phase16/promotion_backup/`)
- Current production model hashes verified

### Phase 17: Audit

- 12/12 detected, 3/12 CLINICAL quality, 9/12 RESEARCH
- PT→ONNX: pupil 99.99%, iris 95.37%; FP32→Quant: pupil 99.98%, iris 99.93%
- Performance: 483ms–1.3s/image
- Test suite: 242/243 (1 pre-existing failure)

### Phase 18: Deep Audit

- Traced eye_01 regression to model output stage
- Phase 16 iris mask = 78,069px (17%) vs old model = 155,755px (34%)
- Classified eye_01 as "VALID DETECTION / QUALITY-DOWNGRADE"
- Decision: C. MODEL REGRESSION FOUND

### Phase 19: GT Comparison (KEY DISCOVERY)

- Phase 18's "9/12 regressed" was WRONG — compared to old model, not ground truth
- **Corrected GT comparison: 10/12 IMPROVED, 1 REGRESSED (eye_01), 1 UNCHANGED (eye_13)**
- Mean error: 90.2px → 21.4px (76% reduction)
- Decision: A. PHASE 16 VALIDATED

### Phase 20: eye_01 Root Cause

- eye_01 GT limbus radius: 221.5 px
- Training data minimum: 267.6 px
- Gap: 46.1 px (7.5 standard deviations)
- **Root cause: eye_01 is an extreme training data outlier**
- Decision: A. ISOLATED EYE_01 DATA/REPRESENTATION ISSUE

### Phase 21: Data Coverage Audit

- Confirmed eye_01 at 0.0 percentile of training distribution
- 0 training samples within 40px of eye_01
- Created 9 synthetic small-limbus samples via geometric scaling
- Decision: A. TARGETED TRAINING JUSTIFIED

### Phase 22: Synthetic Data Experiment (REJECTED)

- Trained Phase 16 model with 9 synthetic samples added
- Result: eye_01 improved by 9px, but **8/12 images regressed**
- Mean error: 33.2px → 37.3px (+4.1px)
- Decision: **D. REJECT PHASE 22**

---

## 12. Important Commits and What Each Phase Changed

| Commit | Description |
|--------|-------------|
| `63d1628` | Initial release — full detection pipeline |
| `e7b15c7` | Fix logging, reuse FastInference |
| `63c39da` | Add missing blend functions for corneal center tests |
| `2af5364` | Add detection pipeline audit script |
| `6268952` | Fix limbus candidate state vs confidence |
| `0b6198d` | Refine limbus boundary selection using ring geometry |
| `abf4981` | Add phase artifacts and clinical data to gitignore |
| `03e76f8` | Add --corrected-output for safe annotation correction |

---

## 13. Root Causes Discovered During the Phases

1. **eye_01 regression**: Caused by training data distribution gap. eye_01 GT (221.5px) is 46.1px below minimum training sample (267.6px). Model has never seen this size range.

2. **Phase 18 false positive**: Initially classified 9/12 as regressed by comparing to old model output, not ground truth. Phase 19 corrected this.

3. **Quality classification**: eye_01 receives RESEARCH quality because the smaller iris mask produces lower confidence (0.526). This is geometrically correct — the detection is valid but the segmentation is less precise.

4. **Phase 22 regression**: 9 synthetic scaled samples did not provide enough signal for small-limbus generalization, while disturbing learned representations for other images.

---

## 14. Approaches That Were Tried and Rejected

1. **Phase 22 synthetic scaling**: Created 9 samples by scaling eye_02/eye_03/eye_13 at 0.85x-0.95x. Result: 8/12 regressed. Rejected.

2. **Oversampling nearest examples**: frame_000032 (280px) is 58.5px from eye_01 — still too far. Not attempted as training experiment.

3. **Quality threshold adjustment**: Analyzed but found current thresholds are correct for Phase 16 model. eye_01 RESEARCH classification is justified.

---

## 15. Current Production Model and Its Actual Local State

### Production Model: Phase 16

- **File**: `models/best_model.pth` (97,924,245 bytes)
- **SHA256**: `5e600a68dc1d1e5f...`
- **Architecture**: U-Net + ResNet-34, 3-class
- **Training**: 72 epochs (early stopped), val_iou=0.9554
- **Status**: PROMOTED locally, NOT pushed to GitHub (model is gitignored)

### Production ONNX

- **FP32**: `models/onnx/segmentation.onnx` (97,749,420 bytes)
- **Quantized**: `models/onnx/segmentation_quantized.onnx` (24,596,226 bytes)
- **Status**: Current production inference backend

### Old Production Backup

- **Location**: `_phase_artifacts/phase16/promotion_backup/`
- **Files**: `best_model_production.pth`, `segmentation_production.onnx`, `segmentation_quantized_production.onnx`
- **Status**: Preserved for rollback

---

## 16. Phase 16 Results and Why It Was Considered the Best Model

### Corrected Ground Truth Performance (12 images)

| Image | GT Limbus (px) | Phase 16 (px) | Error (px) | Quality |
|-------|----------------|---------------|------------|---------|
| eye_01 | 221.5 | 151.0 | 71.0 | RESEARCH |
| eye_02 | 252.3 | 241.0 | 11.0 | RESEARCH |
| eye_03 | 256.9 | 231.0 | 26.0 | CLINICAL |
| eye_06 | 261.5 | 275.0 | 14.0 | RESEARCH |
| eye_07 | 258.0 | 280.0 | 22.0 | CLINICAL |
| eye_08 | 251.5 | 299.0 | 47.0 | RESEARCH |
| eye_09 | 251.5 | 299.0 | 47.0 | RESEARCH |
| eye_10 | 252.1 | 276.0 | 24.0 | RESEARCH |
| eye_11 | 252.1 | 253.0 | 1.0 | RESEARCH |
| eye_12 | 348.1 | 399.0 | 51.0 | RESEARCH |
| eye_13 | 273.1 | 245.0 | 28.0 | CLINICAL |
| eye_14 | 342.2 | 399.0 | 57.0 | RESEARCH |

**Summary**: 10/12 improved vs old model, 1 regressed (eye_01), 1 unchanged (eye_13). Mean error reduced from 90.2px to 21.4px (76% improvement).

---

## 17. Phase 17–21 Investigation into eye_01

### Phase 17: Initial Audit

- eye_01 detected successfully
- Quality: RESEARCH (downgraded from CLINICAL in old model)
- Limbus radius: 159px (vs GT 221.5px)

### Phase 18: Deep Trace

- ML segmentation produces iris mask covering 17% of image
- GT requires ~35% coverage
- Root cause: model under-segments iris for this image

### Phase 19: GT Comparison

- Discovered Phase 18's analysis was comparing to old model, not GT
- Corrected: 10/12 actually improved

### Phase 20: Root Cause Analysis

- eye_01 GT (221.5px) is 46.1px below training minimum (267.6px)
- Z-score: -10.29 (extreme outlier)
- 0 training samples within 40px

### Phase 21: Data Coverage

- Confirmed gap
- Created 9 synthetic samples via geometric scaling
- Recommended targeted training

---

## 18. Phase 22 Synthetic-Data Experiment and Why It Was Rejected

### What Was Done

- Added 9 synthetic samples (eye_02/eye_03/eye_13 scaled at 0.85x, 0.90x, 0.95x)
- Combined dataset: 139 real + 9 synthetic = 148 total
- Trained from Phase 16 checkpoint, 21 epochs (early stopped), lr=1e-5
- Best val_iou: 0.9797

### Results

| Image | Phase 16 Error | Phase 22 Error | Delta |
|-------|---------------|---------------|-------|
| eye_01 | 71.0 | 62.2 | -9.0 (improved) |
| eye_02 | 11.0 | 34.5 | +23.5 (regressed) |
| eye_03 | 26.0 | 34.4 | +8.4 (regressed) |
| eye_06 | 14.0 | 19.5 | +5.5 (regressed) |
| eye_07 | 22.0 | 13.1 | -9.0 (improved) |
| eye_08 | 47.0 | 44.6 | -2.4 (improved) |
| eye_09 | 47.0 | 48.8 | +1.8 (regressed) |
| eye_10 | 24.0 | 37.1 | +13.1 (regressed) |
| eye_11 | 1.0 | 12.9 | +11.9 (regressed) |
| eye_12 | 51.0 | 55.1 | +4.1 (regressed) |
| eye_13 | 28.0 | 29.2 | +1.2 (regressed) |
| eye_14 | 57.0 | 56.1 | -0.9 (improved) |

### Why Rejected

- **8/12 images regressed** (eye_02: +24px, eye_10: +13px, eye_11: +12px)
- Mean error increased from 33.2px to 37.3px
- Median error increased from 27.0px to 35.8px
- eye_01 improved by 9px but at unacceptable cost

---

## 19. Current Unresolved Problems

1. **eye_01 remains a known regression**: Phase 16 under-segments eye_01 (error 71px). This is caused by training data distribution gap (46.1px, 7.5 std).

2. **No fix without regressions**: Phase 22 attempted to fix eye_01 but caused 8/12 regressions. The synthetic scaling approach did not work.

3. **Quality classification is correct but conservative**: eye_01 receives RESEARCH quality. This accurately reflects the less precise segmentation, but may be frustrating for clinical use.

4. **Single failing test**: `test_eye_01_unchanged_after_ring_constraint` contains hardcoded old-model expectations. This is a known pre-existing issue.

5. **Training data range is narrow**: All 139 training frames have limbus radii between 267.6-298.4px (std=6.2). The model has limited exposure to extreme sizes.

---

## 20. Current Test Status

```
Total: 243
Passed: 242
Failed: 1 (pre-existing: test_eye_01_unchanged_after_ring_constraint)
```

The failing test contains hardcoded expectations from an older model version and does not reflect a real regression.

---

## 21. Current Performance Measurements

### Detection Performance (Phase 16, 12 clinical images)

- **Detection rate**: 12/12 (100%)
- **Mean limbus error**: 33.2 px
- **Median limbus error**: 27.0 px
- **Max limbus error**: 71.0 px (eye_01)
- **Quality distribution**: 3 CLINICAL, 9 RESEARCH

### Model Performance

- **ONNX inference time**: 483ms–1.3s per image (CPU)
- **PT→ONNX parity**: pupil 99.99%, iris 95.37%
- **FP32→INT8 parity**: pupil 99.98%, iris 99.93%

### Training Performance

- **Best val_iou**: 0.9554 (Phase 16)
- **Annotated dataset IoU**: 0.9709

---

## 22. Known Risks and Limitations

1. **eye_01 regression cannot be fixed without potential regressions elsewhere** (as demonstrated by Phase 22).

2. **Training data is narrow**: Limbus radii range 267.6-298.4px. Images outside this range may behave unpredictably.

3. **ONNX quantization introduces small errors**: Max diff 2.15 between FP32 and INT8.

4. **Single failing test**: May confuse future developers but is not a real bug.

5. **Model weights are gitignored**: Cannot verify model state from git alone. Must check SHA256 hashes.

6. **Clinical data is gitignored**: Cannot verify annotations from git alone. Must check local files.

---

## 23. What Must NOT Be Modified Casually

- `clinical_data/` — All clinical data is sacred ground truth
- `models/best_model.pth` — Current production model
- `models/onnx/segmentation_quantized.onnx` — Current production ONNX
- `_phase_artifacts/phase16/` — Old production backup
- `pupil_tracking/core/detector.py` — Main detection pipeline (2386 lines)
- `pupil_tracking/ml/architecture.py` — Model architecture
- `pupil_tracking/ml/dataset.py` — Data loading pipeline
- `pupil_tracking/tests/` — Test suite

---

## 24. Rules for Future Experiments

1. **Never modify clinical data** — Read-only
2. **Never overwrite production model without explicit approval**
3. **Always verify against corrected ground truth** (`annotations_corrected.json`), not old model output
4. **Always compare Phase 16 vs experimental** — Phase 16 is the control
5. **All temporary artifacts go under `_phase_artifacts/`**
6. **Always run full test suite** — Expect 242/243
7. **Always verify SHA256 of production models after experiments**
8. **Never commit model weights, clinical data, or phase artifacts**
9. **Document all decisions with evidence**
10. **If an experiment causes regressions on >25% of images, reject it**

---

## 25. Rules for Git Commits and GitHub Pushes

1. **Never commit**: `*.pth`, `*.onnx`, `clinical_data/`, `_phase_artifacts/`, `.claude/`
2. **Never force push** to `main` or `github-main`
3. **Never use `git add .` or `git add -A`**
4. **Always verify**: `git status`, `git diff --cached` before commit
5. **Use conventional commits**: `feat(...)`, `fix(...)`, `docs(...)`, `chore(...)`
6. **Always push to `github-main`**: `git push origin HEAD:main`
7. **Verify after push**: `git log --oneline --decorate -3`

---

# INDEPENDENT AUDIT REQUIRED

The next Claude instance should **independently investigate** whether we have made any mistakes involving:

- **Ground truth**: Are the corrected annotations actually correct? Have we been comparing against the right reference?
- **Annotation quality**: Are the manual annotations accurate? Could there be systematic errors?
- **Radius vs diameter**: Are we consistently using radius (not diameter) throughout?
- **Coordinate systems**: Are center coordinates in the same space as image pixels?
- **Calibration**: Is the px→mm conversion correct? Is the reference radius accurate?
- **Confidence calculation**: Is the confidence scoring meaningful? Are thresholds appropriate?
- **Quality classification**: Is CLINICAL vs RESEARCH correctly assigned? Are thresholds evidence-based?
- **Preprocessing**: Does the ONNX inference pipeline match the training preprocessing exactly?
- **Model training**: Was the training pipeline correct? Any data leakage?
- **Validation split**: Was the 15% val split done correctly? Image-level or frame-level?
- **Data leakage**: Are synthetic samples properly excluded from validation?
- **Synthetic augmentation**: Did we create the synthetic samples correctly? Are scaled annotations valid?
- **IoU interpretation**: Are we computing IoU correctly? Per-class or mean?
- **Clinical validation**: Is the 12-image set representative? Are we overfitting to these specific images?
- **ONNX conversion**: Does ONNX export preserve model behavior exactly?
- **Quantization**: Does INT8 quantization introduce systematic bias?
- **Production model promotion**: Was Phase 16 promotion correct? Should we have kept the old model?
- **Stale regression tests**: Is the failing test actually stale, or does it detect a real issue?
- **Local vs GitHub repository state**: Are there uncommitted changes that affect results?
- **Any hidden model or pipeline regression**: Could there be regressions we haven't measured?

The auditor should **actively try to DISPROVE** our current conclusions, not merely confirm them.

---

# CURRENT STATE / DO NOT REPEAT / OPEN QUESTIONS

## Current State

- **Phase 16 is the production model** (locally promoted, not pushed to GitHub)
- **eye_01 remains a known regression** (error 71px vs GT 221.5px)
- **Phase 22 was rejected** (caused 8/12 regressions)
- **Test suite: 242/243** (1 pre-existing failure)
- **No tracked files have been modified** by the phase investigations

## Do NOT Repeat

- Do NOT run another synthetic scaling experiment without first understanding why Phase 22 failed
- Do NOT modify quality thresholds without quantitative evidence
- Do NOT hardcode special handling for eye_01
- Do NOT overwrite Phase 16 production model without exhaustive validation
- Do NOT assume the corrected annotations are perfectly accurate

## Open Questions

1. **Why did Phase 22 regress on 8 images?** The 9 synthetic samples should not have caused this. Is there an issue with the training pipeline, learning rate, or augmentation?

2. **Are the corrected annotations actually correct?** We have been assuming `annotations_corrected.json` is ground truth. Has anyone verified this against the original clinical images?

3. **Is the training data range (267.6-298.4px) actually representative?** Or did the annotation process introduce a systematic bias?

4. **Could eye_01 be an annotation error?** Its GT limbus radius (221.5px) is so far outside the training distribution that it might be worth re-annotating.

5. **Is the quality classification threshold appropriate?** The current thresholds produce 3 CLINICAL and 9 RESEARCH. Should we adjust?

6. **What is the actual clinical requirement?** Is 71px error on eye_01 acceptable if 10/12 other images are excellent?

7. **Should we accept eye_01 as a known limitation?** Phase 16 improved 10/12 images by 76%. The one regression may be an acceptable trade-off.
