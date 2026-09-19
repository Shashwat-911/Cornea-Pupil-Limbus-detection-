# Models

This directory holds trained model weights. The weight files themselves are **not
committed to Git** because they are large binary artefacts that change frequently
during training.

## Required files

| File | Purpose | Size (approx.) |
|------|---------|----------------|
| `best_model.pth` | PyTorch checkpoint (development fallback) | ~93 MB |
| `onnx/segmentation.onnx` + `onnx/segmentation.onnx.data` | Full-precision ONNX export | ~93 MB |
| `onnx/segmentation_quantized.onnx` | INT8-quantized ONNX (preferred for production) | ~24 MB |
| `onnx/segmentation_optimized.onnx` | Optimized ONNX export | ~93 MB |

The application automatically selects the best available backend:

1. `segmentation_quantized.onnx` (preferred — smallest, fastest)
2. `segmentation.onnx` (full ONNX)
3. `best_model.pth` (PyTorch fallback)
4. Dummy engine (detection returns empty results)

## How to obtain the models

Model weights are **not included** in this repository and must be obtained or
built separately:

1. **Train from scratch** — place your training data in `clinical_data/`
   (properly de-identified and authorized) and run:

   ```bash
   python scripts/train_model.py --epochs 200 --batch-size 4
   ```

2. **Export to ONNX** — after training:

   ```bash
   python scripts/convert_to_onnx.py
   ```

3. **Obtain pre-trained weights** — if you have been provided with model
   weights separately, place them in the paths listed above.

## Production recommendation

For production deployment, use the ONNX quantized model
(`onnx/segmentation_quantized.onnx`). It runs on CPU without a CUDA toolkit
and is the fastest option.

## Distribution

Model weights trained on clinical data are subject to separate distribution
rights. Contact the project owner for authorization before redistributing any
model files.
