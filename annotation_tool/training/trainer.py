"""
Training pipeline for iris feature models.

Handles training loop, optimization, learning rate scheduling, validation,
and model checkpointing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from annotation_tool.training.config import IrisTrainingConfig
from annotation_tool.training.dataset import IrisFeatureDataset
from annotation_tool.training.models.iris_feature_model import IrisFeatureModel
from annotation_tool.training.losses import IrisCompositeLoss

logger = logging.getLogger(__name__)


class IrisTrainer:
    """End-to-end trainer for iris landmark and descriptor models."""

    def __init__(self, config: IrisTrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else "cpu"
        self.model: Optional[IrisFeatureModel] = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self._init_components()

    def _init_components(self) -> None:
        if not HAS_TORCH:
            logger.warning("PyTorch not installed. Trainer operates in stub mode.")
            return

        self.model = IrisFeatureModel(
            backbone_name=self.config.backbone,
            descriptor_dim=self.config.descriptor_dim,
            num_classes=self.config.num_classes,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            use_lora=True,
        ).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )

        self.criterion = IrisCompositeLoss(
            keypoint_weight=self.config.keypoint_loss_weight,
            descriptor_weight=self.config.descriptor_loss_weight,
            class_weight=self.config.classification_loss_weight,
        ).to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            target_heatmaps = batch["heatmap"].to(self.device)
            target_classes = batch["class_map"].to(self.device)

            self.optimizer.zero_grad()
            pred_hm, pred_desc, pred_cls = self.model(images)

            loss, _ = self.criterion(
                pred_hm, pred_desc, pred_cls,
                target_heatmaps, target_classes,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            batches += 1

        return {"train_loss": total_loss / max(1, batches)}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Execute validation over dataset."""
        self.model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                target_heatmaps = batch["heatmap"].to(self.device)
                target_classes = batch["class_map"].to(self.device)

                pred_hm, pred_desc, pred_cls = self.model(images)
                loss, _ = self.criterion(
                    pred_hm, pred_desc, pred_cls,
                    target_heatmaps, target_classes,
                )
                total_loss += loss.item()
                batches += 1

        return {"val_loss": total_loss / max(1, batches)}

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Run complete training cycle."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required to run fit().")

        best_val_loss = float("inf")
        history = []

        logger.info("Starting training for %d epochs on device: %s", self.config.epochs, self.device)

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader) if val_loader else {}
            self.scheduler.step()

            current_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            epoch_log = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_log)

            logger.info(
                "Epoch [%d/%d] - train_loss: %.4f%s",
                epoch, self.config.epochs, train_metrics["train_loss"],
                f" - val_loss: {val_metrics['val_loss']:.4f}" if val_loader else "",
            )

            # Checkpoint best model
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                best_path = self.output_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.__dict__,
                    "val_loss": best_val_loss,
                }, best_path)
                logger.info("Saved new best model checkpoint to %s", best_path)

        # Save history log
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        return {"history": history, "best_val_loss": best_val_loss}
