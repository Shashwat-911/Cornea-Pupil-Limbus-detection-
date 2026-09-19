#!/usr/bin/env python3
"""Export all iris CNN and RL models to ONNX with INT8 quantization.

Usage
-----
python scripts/export_iris_onnx.py \
    --model-dir models \
    --input-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export iris models to ONNX")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--input-size", type=int, default=256, help="Segmentation CNN input size")
    p.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    return p.parse_args()


def _export_onnx(model, dummy, path, input_names, output_names, dynamic_axes):
    """Export using the legacy ONNX exporter (PyTorch 2.6+ dynamo default
    requires onnxscript and external weight files)."""
    try:
        torch.onnx.export(
            model=model,
            args=dummy,
            f=str(path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model=model,
            args=dummy,
            f=str(path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )


def export_segmentation(model_dir: Path, input_size: int, quantize: bool) -> Path:
    pth = model_dir / "iris_segmentation.pth"
    if not pth.exists():
        print(f"  ! {pth} not found; skipping segmentation export")
        return None

    import torch

    from pupil_tracking.ml.iris_segmentation import create_model

    model = create_model(pretrained=False)
    ckpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    onnx_path = model_dir / "iris_segmentation.onnx"
    dummy = torch.randn(1, 3, input_size, input_size)
    _export_onnx(
        model, dummy, onnx_path,
        ["input"], ["logits"],
        {"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"  ✓ {onnx_path}")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            qpath = model_dir / "iris_segmentation_quantized.onnx"
            quantize_dynamic(str(onnx_path), str(qpath), weight_type=QuantType.QInt8)
            print(f"  ✓ {qpath}")
        except ImportError:
            print("  ! onnxruntime.quantization not available; skipping INT8")

    return onnx_path


def export_encoder(model_dir: Path, quantize: bool) -> Path:
    pth = model_dir / "iris_encoder.pth"
    if not pth.exists():
        print(f"  ! {pth} not found; skipping encoder export")
        return None

    import torch

    from pupil_tracking.ml.iris_encoder import create_model

    model = create_model()
    ckpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    onnx_path = model_dir / "iris_encoder.onnx"
    dummy = torch.randn(1, 1, 64, 512)
    _export_onnx(
        model, dummy, onnx_path,
        ["iris_strip"], ["embedding"],
        {"iris_strip": {0: "batch"}, "embedding": {0: "batch"}},
    )
    print(f"  ✓ {onnx_path}")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            qpath = model_dir / "iris_encoder_quantized.onnx"
            quantize_dynamic(str(onnx_path), str(qpath), weight_type=QuantType.QInt8)
            print(f"  ✓ {qpath}")
        except ImportError:
            print("  ! onnxruntime.quantization not available; skipping INT8")

    return onnx_path


def export_rl_agent(model_dir: Path, quantize: bool) -> Path:
    pth = model_dir / "iris_rl_agent.pth"
    if not pth.exists():
        print(f"  ! {pth} not found; skipping RL agent export")
        return None

    import torch

    from pupil_tracking.iris.rl_agent import _QNetwork, _STATE_DIM

    model = _QNetwork()
    ckpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["q_net_state_dict"])
    model.eval()

    onnx_path = model_dir / "iris_rl_agent.onnx"
    dummy = torch.randn(1, _STATE_DIM)
    _export_onnx(
        model, dummy, onnx_path,
        ["state"], ["q_values"],
        {"state": {0: "batch"}, "q_values": {0: "batch"}},
    )
    print(f"  ✓ {onnx_path}")

    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            qpath = model_dir / "iris_rl_agent_quantized.onnx"
            quantize_dynamic(str(onnx_path), str(qpath), weight_type=QuantType.QInt8)
            print(f"  ✓ {qpath}")
        except ImportError:
            print("  ! onnxruntime.quantization not available; skipping INT8")

    return onnx_path


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Iris Model ONNX Export")
    print("=" * 50)

    results = {}

    p = export_segmentation(model_dir, args.input_size, not args.no_quantize)
    if p:
        results["iris_segmentation.onnx"] = {
            "size_bytes": p.stat().st_size,
            "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()[:16],
        }

    p = export_encoder(model_dir, not args.no_quantize)
    if p:
        results["iris_encoder.onnx"] = {
            "size_bytes": p.stat().st_size,
            "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()[:16],
        }

    p = export_rl_agent(model_dir, not args.no_quantize)
    if p:
        results["iris_rl_agent.onnx"] = {
            "size_bytes": p.stat().st_size,
            "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()[:16],
        }

    manifest_path = model_dir / "iris_onnx_manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()