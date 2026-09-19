# IXcentai Standalone Iris Feature Annotation Tool & Custom Training Pipeline

An independent, clinical-grade PyQt6 workstation application for labeling anatomical iris features and surgical ink marks, coupled with an end-to-end PyTorch training pipeline for iris feature detection and cyclotorsion registration.

---

## 🚀 Quick Start

### 1. Installation
Install annotation & training dependencies:
```bash
pip install -r annotation_tool/requirements.txt
```

### 2. Launch Annotation GUI
```bash
python -m annotation_tool.main
# or
python -m annotation_tool.cli.annotate --image-dir path/to/images --annotation-dir path/to/annotations
```

### 3. Training Custom Model
```bash
python -m annotation_tool.cli.train --data-dir path/to/dataset --epochs 50 --batch-size 8
```

### 4. Validation & Export
```bash
python -m annotation_tool.cli.validate --checkpoint checkpoints/best_model.pth --data-dir path/to/dataset
python -m annotation_tool.cli.export --checkpoint checkpoints/best_model.pth --output models/iris_features/iris_feature_model.onnx
```

---

## 🎨 Supported Annotations
- **Iris Crypts**: Point landmarks on distinct iris trabecular openings.
- **Contraction Furrows**: Line segments / contours along concentric iris furrows.
- **Limbal Vessels**: Scleral and limbal vessel bifurcation branch points.
- **Surgical Ink Marks**: Pre-operative ink marks (gentian violet / blue) on sclera/limbus.
- **Collarette**: Minor circle dividing pupillary and ciliary zones.

## ⌨️ Default Keyboard Shortcuts
- `1`: Iris Crypt tool
- `2`: Furrow tool
- `3`: Limbal Vessel tool
- `4`: Ink Mark tool
- `5`: Collarette tool
- `N`: Next image (auto-save)
- `P`: Previous image
- `S`: Save annotations
- `Z`: Undo last point
- `+/-`: Zoom in / Zoom out
- `R`: Reset view
