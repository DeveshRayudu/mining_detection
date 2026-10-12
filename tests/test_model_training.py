"""
Unit tests and verification gates for U-Net ResNet-34 model and training pipeline.
Covers:
1. One-batch shape test
2. Combined loss stability (no NaNs)
3. Fixed random seeds reproducibility
4. Overfit-8-tiles gradient flow and memorization gate
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import build_unet_model, CombinedDiceFocalLoss
from src.models.trainer import MiningTrainer, set_seed
from src.utils.metrics import compute_binary_metrics


def test_one_batch_shape_and_no_nans():
    """Gate check: one-batch shape test (B=2, C=8, H=256, W=256) -> (B=2, C=1, H=256, W=256)."""
    model = build_unet_model(in_channels=8, classes=1, encoder_weights=None)
    model.eval()

    dummy_input = torch.randn(2, 8, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 1, 256, 256)
    assert not torch.isnan(out).any()


def test_combined_loss_stability():
    """Gate check: Combined Dice + Focal loss produces valid non-negative scalar without NaNs."""
    loss_fn = CombinedDiceFocalLoss()
    logits = torch.randn(4, 1, 64, 64)
    targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0


def test_fixed_seed_reproducibility():
    """Gate check: fixed seeds produce identical random initialization."""
    set_seed(42)
    t1 = torch.randn(10, 10)

    set_seed(42)
    t2 = torch.randn(10, 10)

    assert torch.equal(t1, t2)


def test_overfit_8_tiles_gate():
    """
    Gate check: overfit-8-tiles test.
    Trains on 8 tiles for 25 steps; validates that loss drops by >60%
    and train IoU exceeds 0.80, proving gradient flow and architecture integrity.
    """
    set_seed(42)
    device = torch.device("cpu")

    # Construct 8 synthetic tiles (8, 64, 64 for fast CPU unit test)
    # Target mask: center 20x20 box is mining class (1)
    images = torch.randn(8, 8, 64, 64)
    masks = torch.zeros(8, 1, 64, 64)
    masks[:, :, 20:44, 20:44] = 1.0

    # Model for test
    model = build_unet_model(in_channels=8, classes=1, encoder_weights=None)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CombinedDiceFocalLoss()

    initial_loss = None
    final_loss = None

    model.train()
    for step in range(25):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        assert not torch.isnan(loss), "NaN loss encountered!"
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    # Check loss drop
    assert final_loss < initial_loss * 0.40, (
        f"Loss failed to drop by >60%: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )

    # Check prediction IoU on the overfit batch
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(images))
        m = compute_binary_metrics(preds, masks, threshold=0.5)

    assert m["iou"] >= 0.80, f"Overfit IoU too low: {m['iou']:.4f} (expected >= 0.80)"
