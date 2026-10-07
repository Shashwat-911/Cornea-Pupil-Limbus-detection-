"""End-to-end training and evaluation script for deep learning cyclotorsion models.

Usage:
------
# 1. Self-test smoke run on synthetic paired data:
python scripts/train_cyclotorsion_model.py --smoke-test

# 2. Train on patient JSONL manifest:
python scripts/train_cyclotorsion_model.py --manifest data/pairs.jsonl --epochs 20 --output-dir models/cyclotorsion
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainCyclotorsion")


def generate_synthetic_training_data(
    num_samples: int = 40,
    num_angles: int = 720,
    num_radial: int = 64,
    seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Synthesize realistic iris polar strips with known ground-truth angular shifts."""
    rng = np.random.default_rng(seed)
    samples = []
    freq = np.fft.rfftfreq(num_angles)

    for i in range(num_samples):
        # Generate base iris texture with Gaussian smoothing
        base = rng.normal(120, 35, (num_radial, num_angles)).astype(np.float32)
        base = cv2.GaussianBlur(base, (7, 3), 0)
        base = np.clip(base, 0, 255).astype(np.uint8)

        # Ground truth rotation in [-15, +15] degrees
        angle_deg = float(rng.uniform(-15.0, 15.0))
        shift_bins = angle_deg * num_angles / 360.0

        # Sub-pixel shift via Fourier transform
        f_base = np.fft.rfft(base.astype(np.float32), axis=1)
        f_shifted = f_base * np.exp(-2j * np.pi * freq * shift_bins)
        curr = np.fft.irfft(f_shifted, n=num_angles, axis=1)

        # Apply realistic illumination and noise
        curr = np.clip(curr * rng.uniform(0.85, 1.15) + rng.normal(0, 2.0, curr.shape), 0, 255).astype(np.uint8)

        mask_ref = np.ones((num_radial, num_angles), dtype=np.uint8) * 255
        mask_curr = np.ones((num_radial, num_angles), dtype=np.uint8) * 255

        # Sector occlusion simulation (eyelashes/eyelids)
        if rng.random() > 0.5:
            occ_len = int(rng.uniform(0.1, 0.25) * num_angles)
            occ_start = int(rng.integers(0, num_angles))
            idx = (np.arange(occ_start, occ_start + occ_len)) % num_angles
            mask_curr[:, idx] = 0

        samples.append({
            "ref": base,
            "curr": curr,
            "mask_ref": mask_ref,
            "mask_curr": mask_curr,
            "angle_deg": angle_deg,
            "patient_id": f"syn_subject_{i // 2:03d}",
        })

    return samples


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from pupil_tracking.pentacam.models import IrisPolarSiameseNet
    from pupil_tracking.pentacam.matcher import ClassicalFFTMatcher

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Dataset setup
    if args.smoke_test or not args.manifest:
        logger.info("Running with synthetic dataset (40 pairs, 10 val pairs)")
        train_raw = generate_synthetic_training_data(num_samples=32, num_angles=args.num_angles, seed=42)
        val_raw = generate_synthetic_training_data(num_samples=16, num_angles=args.num_angles, seed=101)
    else:
        from pupil_tracking.pentacam.training_data import get_pytorch_dataset
        logger.info("Loading patient manifest from %s", args.manifest)
        train_dataset = get_pytorch_dataset(args.manifest, split="train", augment=True, num_angles=args.num_angles)
        val_dataset = get_pytorch_dataset(args.manifest, split="validation", augment=False, num_angles=args.num_angles)

    class RawDataset(Dataset):
        def __init__(self, raw_list):
            self.items = raw_list

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            item = self.items[idx]
            ref_t = torch.from_numpy(item["ref"].astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
            curr_t = torch.from_numpy(item["curr"].astype(np.float32) / 127.5 - 1.0).unsqueeze(0)
            mr_t = torch.from_numpy((item["mask_ref"] > 0).astype(np.float32)).unsqueeze(0)
            mc_t = torch.from_numpy((item["mask_curr"] > 0).astype(np.float32)).unsqueeze(0)
            return {
                "ref": ref_t,
                "curr": curr_t,
                "mask_ref": mr_t,
                "mask_curr": mc_t,
                "target_deg": torch.tensor(item["angle_deg"], dtype=torch.float32),
                "raw_ref": item["ref"],
                "raw_curr": item["curr"],
                "raw_mr": item["mask_ref"],
                "raw_mc": item["mask_curr"],
            }

    if args.smoke_test or not args.manifest:
        train_dataset = RawDataset(train_raw)
        val_dataset = RawDataset(val_raw)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Model initialization
    model = IrisPolarSiameseNet(
        num_angles=args.num_angles,
        max_degrees=args.max_degrees,
        embedding_dim=args.embedding_dim,
        temperature=0.04,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    classical_matcher = ClassicalFFTMatcher()
    best_val_mae = float("inf")
    history = []

    # 3. Training Loop
    logger.info("Beginning training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_errors = []

        for batch in train_loader:
            ref = batch["ref"].to(device)
            curr = batch["curr"].to(device)
            mr = batch["mask_ref"].to(device)
            mc = batch["mask_curr"].to(device)
            target_deg = batch["target_deg"].to(device)

            optimizer.zero_grad()
            out = model(ref, curr, mr, mc)
            pred_deg = out["pred_deg"]

            # Geodesic circular angular difference in [-180, +180]
            diff = (pred_deg - target_deg + 180.0) % 360.0 - 180.0
            angle_loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.5)

            # Contrastive margin loss: encourage confident sharp peaks
            margin_loss = F.relu(0.10 - out["peak_margin"]).mean()

            total_loss = angle_loss + 0.2 * margin_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * len(target_deg)
            train_errors.extend(torch.abs(diff).detach().cpu().numpy().tolist())

        scheduler.step()
        train_mae = float(np.mean(train_errors)) if train_errors else 0.0

        # 4. Validation step
        model.eval()
        val_errors = []
        classical_errors = []
        with torch.no_grad():
            for batch in val_loader:
                ref = batch["ref"].to(device)
                curr = batch["curr"].to(device)
                mr = batch["mask_ref"].to(device)
                mc = batch["mask_curr"].to(device)
                target_deg = batch["target_deg"].to(device)

                out = model(ref, curr, mr, mc)
                pred_deg = out["pred_deg"]
                diff = (pred_deg - target_deg + 180.0) % 360.0 - 180.0
                val_errors.extend(torch.abs(diff).cpu().numpy().tolist())

                # Classical matcher baseline comparison
                if "raw_ref" in batch:
                    for i in range(len(target_deg)):
                        r_ref = batch["raw_ref"][i].numpy() if hasattr(batch["raw_ref"][i], "numpy") else np.array(batch["raw_ref"][i])
                        r_curr = batch["raw_curr"][i].numpy() if hasattr(batch["raw_curr"][i], "numpy") else np.array(batch["raw_curr"][i])
                        r_mr = batch["raw_mr"][i].numpy() if hasattr(batch["raw_mr"][i], "numpy") else np.array(batch["raw_mr"][i])
                        r_mc = batch["raw_mc"][i].numpy() if hasattr(batch["raw_mc"][i], "numpy") else np.array(batch["raw_mc"][i])
                        c_res = classical_matcher(r_ref, r_curr, r_mr, r_mc, max_degrees=args.max_degrees, reference_cache={})
                        if c_res.valid:
                            classical_errors.append(abs(c_res.angle_deg - target_deg[i].item()))

        val_mae = float(np.mean(val_errors)) if val_errors else 0.0
        val_med = float(np.median(val_errors)) if val_errors else 0.0
        c_mae = float(np.mean(classical_errors)) if classical_errors else float("nan")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_dataset),
            "train_mae_deg": train_mae,
            "val_mae_deg": val_mae,
            "val_med_deg": val_med,
            "classical_mae_deg": c_mae,
        }
        history.append(epoch_stat)
        logger.info(
            "Epoch %02d/%02d | Loss: %.4f | Train MAE: %.3f deg | Val MAE: %.3f deg | Classical MAE: %.3f deg",
            epoch, args.epochs, epoch_stat["train_loss"], train_mae, val_mae, c_mae,
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_dir / "best_model.pth"))

    # 5. Export artifacts
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "iris_polar_siamese.onnx"
    try:
        model.export_onnx(onnx_path)
    except Exception as e:
        logger.warning("ONNX export skipped: %s", e)

    report = {
        "best_val_mae_deg": best_val_mae,
        "history": history,
        "config": vars(args),
    }
    report_file = out_dir / "training_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Training complete. Report written to %s", report_file)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Path to pairs.jsonl manifest")
    parser.add_argument("--output-dir", default="output/trained_model")
    parser.add_argument("--num-angles", type=int, default=720)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-degrees", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run_training(args)
