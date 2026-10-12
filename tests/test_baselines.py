"""
Unit tests for baseline models (Index Thresholding and Random Forest).
"""

from pathlib import Path
import numpy as np
import pytest

from src.models import ThresholdBaseline, RandomForestBaseline
from src.utils.metrics import compute_binary_metrics


def test_threshold_baseline_prediction_and_optimization():
    """Verify ThresholdBaseline predicts binary masks and optimizes thresholds."""
    # Create synthetic 8-channel patch: channel 1=Green, 4=SWIR1, 6=NDVI, 7=BSI
    patch = np.zeros((8, 64, 64), dtype=np.float32)
    patch[1] = 0.1  # Green
    patch[4] = 0.4  # SWIR1
    patch[6] = 0.15 # Low NDVI (bare)
    patch[7] = 0.10 # High BSI (bare soil)

    tb = ThresholdBaseline(ndvi_thresh=0.20, bsi_thresh=0.05)
    pred = tb.predict(patch)

    assert pred.shape == (64, 64)
    assert set(np.unique(pred)).issubset({0, 1})
    assert (pred == 1).all(), "All pixels should trigger mine condition"

    # Optimization test
    val_images = [patch]
    val_masks = [np.ones((64, 64), dtype=np.uint8)]
    res = tb.optimize(val_images, val_masks)
    assert "best_f1" in res
    assert res["best_f1"] > 0.0


def test_random_forest_fit_predict_save_load(tmp_path):
    """Verify RandomForestBaseline trains, predicts, and serializes properly."""
    rng = np.random.RandomState(42)
    # Synthetic training patches
    train_imgs = [rng.uniform(0.0, 1.0, (8, 32, 32)).astype(np.float32) for _ in range(2)]
    # Target masks: center is mine (1), borders are background (0)
    train_masks = []
    for _ in range(2):
        m = np.zeros((32, 32), dtype=np.uint8)
        m[10:22, 10:22] = 1
        train_masks.append(m)

    rf = RandomForestBaseline(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(train_imgs, train_masks, max_samples_per_class=500)
    assert rf.is_fitted

    # Predict
    test_img = train_imgs[0]
    pred = rf.predict(test_img, threshold=0.5)
    assert pred.shape == (32, 32)
    assert set(np.unique(pred)).issubset({0, 1})

    # Save and load
    model_file = tmp_path / "rf_test.joblib"
    rf.save(model_file)
    assert model_file.exists()

    loaded_rf = RandomForestBaseline()
    loaded_rf.load(model_file)
    assert loaded_rf.is_fitted

    pred_loaded = loaded_rf.predict(test_img, threshold=0.5)
    assert np.array_equal(pred, pred_loaded)
