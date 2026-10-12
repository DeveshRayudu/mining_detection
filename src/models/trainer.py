"""
Training, Validation, and Model Versioning Module for U-Net ResNet-34.
Implements reproducible seed management, mixed precision, and metric logging.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import config
from src.models.unet import build_unet_model
from src.models.losses import CombinedDiceFocalLoss
from src.utils.metrics import compute_binary_metrics


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seeds across python, numpy, and pytorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MiningTrainer:
    """Orchestrates deep learning model training, validation, and checkpointing."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        device: Optional[torch.device] = None,
    ):
        set_seed(config.project.random_seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or build_unet_model()
        self.model.to(self.device)

        self.criterion = CombinedDiceFocalLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            assert not torch.isnan(loss), "Fatal: NaN loss encountered during training step!"
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return float(total_loss / max(1, len(dataloader)))

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, threshold: float = 0.5) -> Dict[str, float]:
        """Evaluates model performance across a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            m = compute_binary_metrics(probs, masks, threshold=threshold)
            total_tp += m["tp"]
            total_fp += m["fp"]
            total_fn += m["fn"]
            total_tn += m["tn"]

        iou = float(total_tp / (total_tp + total_fp + total_fn + 1e-7))
        f1 = float(2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-7))
        precision = float(total_tp / (total_tp + total_fp + 1e-7))
        recall = float(total_tp / (total_tp + total_fn + 1e-7))

        return {
            "loss": float(total_loss / max(1, len(dataloader))),
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        checkpoint_name: str = "unet_resnet34_v1.pt",
    ) -> Dict[str, List[float]]:
        """
        Executes complete training loop with validation checks and best model tracking.
        """
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )
        best_val_iou = -1.0
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
        }

        ckpt_dir = config.dirs.models_checkpoints
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / checkpoint_name

        current_dir = config.dirs.models_current
        current_dir.mkdir(parents=True, exist_ok=True)
        current_model_path = current_dir / "model.pt"

        print(f"🚀 Starting training on {self.device} for {epochs} epochs...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_metrics = self.evaluate(val_loader)
            scheduler.step()

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_metrics["loss"])
            history["val_iou"].append(v_metrics["iou"])
            history["val_f1"].append(v_metrics["f1"])
            history["val_precision"].append(v_metrics["precision"])
            history["val_recall"].append(v_metrics["recall"])

            is_best = v_metrics["iou"] > best_val_iou
            if is_best:
                best_val_iou = v_metrics["iou"]
                # Save versioned checkpoint
                torch.save(self.model.state_dict(), ckpt_path)
                # Update current active model pointer
                torch.save(self.model.state_dict(), current_model_path)
                with open(current_dir / "model_meta.json", "w") as f:
                    json.dump({
                        "architecture": "Unet-ResNet34",
                        "in_channels": 8,
                        "classes": 1,
                        "active_checkpoint": checkpoint_name,
                        "best_val_iou": best_val_iou,
                        "best_val_f1": v_metrics["f1"],
                        "best_epoch": epoch,
                    }, f, indent=2)

            best_marker = "🌟 (BEST)" if is_best else ""
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {t_loss:.4f} | "
                f"Val Loss: {v_metrics['loss']:.4f} | "
                f"Val IoU: {v_metrics['iou']:.4f} | "
                f"Val F1: {v_metrics['f1']:.4f} {best_marker}"
            )

        elapsed = time.time() - start_time
        print(f"✅ Training completed in {elapsed:.1f}s. Best Val IoU: {best_val_iou:.4f}")
        return history
