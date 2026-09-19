#!/usr/bin/env python3
"""Train the iris texture encoder CNN with triplet loss.

Expects a manifest JSON mapping iris identity -> list of image paths:

    {
      "eye_01": ["clinical_data/clean/eye_01.jpeg"],
      "eye_02": ["clinical_data/clean/eye_02.jpeg"],
      ...
    }

Produces models/iris_encoder.pth and models/iris_encoder.onnx.

Usage
-----
python scripts/train_iris_encoder.py \
    --manifest iris_manifest.json \
    --epochs 100 --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from pupil_tracking.ml.iris_encoder import _IrisEncoderModel, create_model
from pupil_tracking.ml.iris_losses import hard_negative_mining
from pupil_tracking.utils.logger import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train iris encoder CNN")
    p.add_argument("--manifest", required=True, help="JSON: {identity: [image paths]}")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--save-dir", type=str, default="models")
    return p.parse_args()


def _load_images(paths) -> np.ndarray:
    """Load grayscale versions of the given image paths, resized to standard input."""
    from pupil_tracking.ml.iris_encoder import _INPUT_HEIGHT, _INPUT_WIDTH

    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (_INPUT_WIDTH, _INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    if not imgs:
        return np.zeros((0, _INPUT_HEIGHT, _INPUT_WIDTH), dtype=np.uint8)
    return np.stack(imgs)


def main():
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    with open(args.manifest) as f:
        manifest = json.load(f)

    identities = list(manifest.keys())
    if len(identities) < 2:
        logger.error("Need >= 2 identities for triplet training")
        sys.exit(1)

    n_val = max(1, int(len(identities) * args.val_split))
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(identities))
    val_ids = [identities[i] for i in perm[:n_val]]
    train_ids = [identities[i] for i in perm[n_val:]]

    logger.info("Train identities: %d, Val identities: %d", len(train_ids), len(val_ids))

    model = create_model(device=args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # sample one anchor per identity per batch
        for _ in range(max(4, len(train_ids) // args.batch_size)):
            ids = rng.choice(train_ids, size=args.batch_size, replace=False)
            anchors = []
            positives = []
            negatives = []

            for iid in ids:
                paths = manifest[iid]
                if len(paths) < 2:
                    continue
                idx_a, idx_p = rng.choice(len(paths), size=2, replace=False)
                neg_id = rng.choice([o for o in train_ids if o != iid])
                neg_paths = manifest[neg_id]
                img_a = _load_images([paths[idx_a]])
                img_p = _load_images([paths[idx_p]])
                img_n = _load_images([neg_paths[rng.randint(len(neg_paths))]])

                if img_a.shape[0] == 0 or img_p.shape[0] == 0 or img_n.shape[0] == 0:
                    continue

                anchors.append(_to_tensor(img_a[0], args.device))
                positives.append(_to_tensor(img_p[0], args.device))
                negatives.append(_to_tensor(img_n[0], args.device))

            if len(anchors) < 2:
                continue

            a_t = torch.stack(anchors)
            p_t = torch.stack(positives)
            n_t = torch.stack(negatives)

            emb_a = model(a_t)
            emb_p = model(p_t)
            emb_n = model(n_t)

            from pupil_tracking.ml.iris_losses import IrisTripletLoss

            loss_fn = IrisTripletLoss(margin=args.margin)
            loss = loss_fn(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        val_loss = _evaluate_triplet(model, manifest, val_ids, args.device, args.margin)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch + 1, args.epochs, avg_loss, val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
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
    pth_path = save_dir / "iris_encoder.pth"
    torch.save({"model_state_dict": best_state, "best_val_loss": best_loss}, pth_path)
    logger.info("Saved %s (val_loss=%.4f)", pth_path, best_loss)

    try:
        model.eval()
        model.to("cpu")
        dummy = torch.randn(1, 1, 64, 512)
        onnx_path = save_dir / "iris_encoder.onnx"
        _export_onnx(
            model, dummy, onnx_path,
            ["iris_strip"], ["embedding"],
            {"iris_strip": {0: "batch"}, "embedding": {0: "batch"}},
        )
        logger.info("Exported %s", onnx_path)

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            qpath = save_dir / "iris_encoder_quantized.onnx"
            quantize_dynamic(str(onnx_path), str(qpath), weight_type=QuantType.QInt8)
            logger.info("Exported %s", qpath)
        except ImportError:
            logger.warning("onnxruntime.quantization not available; skipping INT8")
    except Exception as e:
        logger.warning("ONNX export failed: %s", e)


def _export_onnx(model, dummy, path, input_names, output_names, dynamic_axes):
    """Export using the legacy ONNX exporter (linear-mode)."""
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


def _to_tensor(gray: np.ndarray, device: str) -> "torch.Tensor":
    import torch

    img = gray.astype(np.float32) / 255.0
    img = (img - img.mean()) / max(img.std(), 1e-6)
    t = torch.from_numpy(img[np.newaxis]).unsqueeze(0)
    return t.to(device)


def _evaluate_triplet(model, manifest, val_ids, device: str, margin: float) -> float:
    import torch

    from pupil_tracking.ml.iris_losses import IrisTripletLoss

    model.eval()
    loss_fn = IrisTripletLoss(margin=margin)
    total = 0.0
    count = 0

    rng = np.random.RandomState(0)
    with torch.no_grad():
        for _ in range(10):
            ids = rng.choice(val_ids, size=min(8, len(val_ids)), replace=False)
            anchors, positives, negatives = [], [], []
            for iid in ids:
                paths = manifest[iid]
                if len(paths) < 2:
                    continue
                idx_a, idx_p = rng.choice(len(paths), size=2, replace=False)
                neg_id = rng.choice([o for o in val_ids if o != iid])
                neg_paths = manifest[neg_id]
                a = _load_images([paths[idx_a]])
                p = _load_images([paths[idx_p]])
                n = _load_images([neg_paths[rng.randint(len(neg_paths))]])
                if a.shape[0] == 0 or p.shape[0] == 0 or n.shape[0] == 0:
                    continue
                anchors.append(_to_tensor(a[0], device))
                positives.append(_to_tensor(p[0], device))
                negatives.append(_to_tensor(n[0], device))
            if len(anchors) < 2:
                continue

            loss = loss_fn(
                model(torch.stack(anchors)),
                model(torch.stack(positives)),
                model(torch.stack(negatives)),
            )
            total += loss.item()
            count += 1

    return total / max(count, 1)


if __name__ == "__main__":
    main()