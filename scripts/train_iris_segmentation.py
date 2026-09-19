#!/usr/bin/env python3
"""Train the iris texture segmentation CNN (MobileNetV3-Small U-Net).

Produces models/iris_segmentation.pth, models/iris_segmentation.onnx, and
models/iris_segmentation_quantized.onnx

Usage
-----
python scripts/train_iris_segmentation.py \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/iris_masks \
    --epochs 80 --batch-size 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.ml.iris_losses import IrisSegmentationLoss
from pupil_tracking.ml.iris_segmentation import create_model
from pupil_tracking.utils.logger import get_logger

logger = get_logger()

_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train iris segmentation CNN")
    p.add_argument("--image-dir", required=True, help="Directory of eye images")
    p.add_argument("--mask-dir", required=True, help="Directory of iris masks (usable=255, non-iris=0)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma")
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="models")
    return p.parse_args()


def _load_pairs(image_dir: str, mask_dir: str) -> list:
    img_dir = Path(image_dir)
    msk_dir = Path(mask_dir)
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        mask_path = msk_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = msk_dir / (img_path.stem + "_mask.png")
        if mask_path.exists():
            pairs.append((str(img_path), str(mask_path)))
    return pairs


def _prepare_batch(pairs, indices, input_size: int):
    import torch

    images = []
    masks = []
    for idx in indices:
        img = cv2.imread(pairs[idx][0])
        msk = cv2.imread(pairs[idx][1], cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.int64)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = (img_rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_t = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)))
        msk_t = torch.from_numpy(msk)
        images.append(img_t)
        masks.append(msk_t)

    if not images:
        return None, None
    return torch.stack(images), torch.stack(masks)


def main():
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    pairs = _load_pairs(args.image_dir, args.mask_dir)
    if len(pairs) < 4:
        logger.error("Need >= 4 image/mask pairs, found %d", len(pairs))
        sys.exit(1)

    n_val = max(1, int(len(pairs) * args.val_split))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(pairs))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    logger.info("Training: %d images, Validation: %d images", len(train_idx), len(val_idx))

    model = create_model(pretrained=True, device=args.device)
    loss_fn = IrisSegmentationLoss(gamma=args.gamma, dice_weight=args.dice_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        bs = args.batch_size
        for start in range(0, len(train_idx), bs):
            chunk = train_idx[start : start + bs]
            images_t, masks_t = _prepare_batch(pairs, chunk, args.input_size)
            if images_t is None:
                continue
            images_t, masks_t = images_t.to(args.device), masks_t.to(args.device)

            optimizer.zero_grad()
            logits = model(images_t)
            loss = loss_fn(logits, masks_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(chunk)

        scheduler.step()
        train_loss /= max(len(train_idx), 1)

        dice = _evaluate(model, pairs, val_idx, args.input_size, args.device)
        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_dice=%.4f  lr=%.2e",
            epoch + 1, args.epochs, train_loss, dice, scheduler.get_last_lr()[0],
        )

        if dice > best_dice:
            best_dice = dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    if best_state is None:
        logger.error("No improvement; nothing saved")
        sys.exit(1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(best_state)
    pth_path = save_dir / "iris_segmentation.pth"
    torch.save({"model_state_dict": best_state, "best_val_dice": best_dice}, pth_path)

    meta = {"best_val_dice": float(best_dice), "epochs": args.epochs, "input_size": args.input_size}
    (save_dir / "iris_segmentation_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved %s (val_dice=%.4f)", pth_path, best_dice)

    # ONNX export
    try:
        model.eval()
        model.to("cpu")
        dummy = torch.randn(1, 3, args.input_size, args.input_size)
        onnx_path = save_dir / "iris_segmentation.onnx"
        _export_onnx(
            model, dummy, onnx_path,
            ["input"], ["logits"],
            {"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        logger.info("Exported %s", onnx_path)

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            qpath = save_dir / "iris_segmentation_quantized.onnx"
            quantize_dynamic(str(onnx_path), str(qpath), weight_type=QuantType.QInt8)
            logger.info("Exported %s", qpath)
        except ImportError:
            logger.warning("onnxruntime.quantization not available; skipping INT8")
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)


def _export_onnx(model, dummy, path, input_names, output_names, dynamic_axes):
    """Export to ONNX using the legacy exporter (PyTorch 2.6+ defaults to the
    dynamo exporter, which needs onnxscript and splits weights into external
    files). Falls back gracefully if ``dynamo`` is not accepted."""
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


def _evaluate(model, pairs, indices, input_size: int, device: str) -> float:
    import torch

    model.eval()
    dice_total = 0.0
    count = 0
    for idx in indices:
        images_t, masks_t = _prepare_batch(pairs, [idx], input_size)
        if images_t is None:
            continue
        with torch.no_grad():
            logits = model(images_t.to(device))
            preds = torch.argmax(logits, dim=1).cpu()
            m = masks_t.cpu()
            inter = ((preds == 1) & (m == 1)).sum().float()
            union = ((preds == 1) | (m == 1)).sum().float()
            dice = (2 * inter + 1.0) / (union + 1.0)
            dice_total += dice.item()
            count += 1
    return dice_total / max(count, 1)


if __name__ == "__main__":
    main()