"""
CLI script to train U-Net ResNet-34 on Sentinel-2 8-channel mining data.
Runs on CPU or GPU automatically, logs metrics, and version-controls checkpoints.
"""

import argparse
import json
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.preprocessing import DatasetNormalizer
from src.dataset import MiningDataset
from src.dataset.mining_dataset import get_training_augmentations
from src.models import build_unet_model
from src.models.trainer import MiningTrainer, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net ResNet-34 Mining Detector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint-name", type=str, default="unet_resnet34_v1.pt", help="Checkpoint name")
    args = parser.parse_args()

    print("=" * 70)
    print("🛰️  PHASE 5: U-NET RESNET-34 TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Split Manifest
    manifest_path = config.dirs.data_dir / "split_manifest.json"
    if not manifest_path.exists():
        print(f"❌ Split manifest missing at {manifest_path}. Run Phase 3 first.")
        sys.exit(1)

    with open(manifest_path) as f:
        split = json.load(f)

    train_ids = split["train"]
    val_ids = split["val"]
    scene_id = "S2A_45QUE_20240102_0_L2A"

    print(f"Dataset Configuration:")
    print(f"  - Training tiles:   {len(train_ids)} (with paired flips/rotations)")
    print(f"  - Validation tiles: {len(val_ids)}")
    print(f"  - Input Channels:   8 (B02, B03, B04, B08, B11, B12, NDVI, BSI)")
    print(f"  - Backbone:         ResNet-34 (Pretrained ImageNet)")
    print(f"  - Loss Function:    Combined Dice + Focal Loss")

    # 2. Setup Normalizer & Datasets
    norm_path = config.dirs.models_dir / "norm_stats.json"
    normalizer = DatasetNormalizer(stats_path=norm_path)

    img_dir = config.dirs.data_tiles / scene_id
    mask_dir = config.dirs.data_labels / "tiles" / scene_id

    train_dataset = MiningDataset(
        tile_ids=train_ids,
        images_dir=img_dir,
        masks_dir=mask_dir,
        normalizer=normalizer,
        transform=get_training_augmentations(),
    )
    val_dataset = MiningDataset(
        tile_ids=val_ids,
        images_dir=img_dir,
        masks_dir=mask_dir,
        normalizer=normalizer,
        transform=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Initialize Model & Trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Hardware Device:  {device}")

    model = build_unet_model(
        in_channels=8,
        classes=1,
        encoder_name="resnet34",
        encoder_weights="imagenet",
    )

    trainer = MiningTrainer(model=model, lr=args.lr, device=device)

    # 4. Train Model
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_name=args.checkpoint_name,
    )

    # Save training metrics history
    history_path = config.dirs.models_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 PHASE 5 TRAINING COMPLETE: Model Checkpointed to models/current/model.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
