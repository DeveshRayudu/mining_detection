"""
Unit tests for segmentation metrics on toy masks with analytically known answers.
"""

import numpy as np
import torch
import pytest

from src.utils.metrics import compute_binary_metrics


def test_metrics_perfect_overlap():
    """Verify metrics return 1.0 on identical ground truth and prediction."""
    y_true = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=np.uint8)
    y_pred = y_true.copy()

    m = compute_binary_metrics(y_pred, y_true)
    assert m["tp"] == 3
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["tn"] == 6
    assert abs(m["iou"] - 1.0) < 1e-6
    assert abs(m["f1"] - 1.0) < 1e-6
    assert abs(m["precision"] - 1.0) < 1e-6
    assert abs(m["recall"] - 1.0) < 1e-6


def test_metrics_zero_overlap():
    """Verify metrics return 0.0 when predictions have no true positive overlap."""
    y_true = np.array([
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=np.uint8)
    y_pred = np.array([
        [0, 0, 1],
        [1, 0, 0],
    ], dtype=np.uint8)

    m = compute_binary_metrics(y_pred, y_true)
    assert m["tp"] == 0
    assert m["fp"] == 2
    assert m["fn"] == 2
    assert m["tn"] == 2
    assert abs(m["iou"] - 0.0) < 1e-6
    assert abs(m["f1"] - 0.0) < 1e-6
    assert abs(m["precision"] - 0.0) < 1e-6
    assert abs(m["recall"] - 0.0) < 1e-6


def test_metrics_known_analytical_fractions():
    """
    Test exact analytical fractions:
    True:  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  (6 positives)
    Pred:  [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]  (5 positives)
    TP = 4, FP = 1, FN = 2, TN = 3
    IoU = 4 / (4 + 1 + 2) = 4/7 ~ 0.57142857
    Precision = 4 / (4 + 1) = 0.8
    Recall = 4 / (4 + 2) = 4/6 ~ 0.66666667
    F1 = 8 / (8 + 1 + 2) = 8/11 ~ 0.72727273
    """
    y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

    m = compute_binary_metrics(y_pred, y_true)
    assert m["tp"] == 4
    assert m["fp"] == 1
    assert m["fn"] == 2
    assert m["tn"] == 3

    assert abs(m["iou"] - (4.0 / 7.0)) < 1e-5
    assert abs(m["precision"] - 0.8) < 1e-5
    assert abs(m["recall"] - (4.0 / 6.0)) < 1e-5
    assert abs(m["f1"] - (8.0 / 11.0)) < 1e-5


def test_metrics_torch_tensor_compatibility():
    """Verify PyTorch tensor inputs are processed correctly."""
    t_true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8]])  # Probabilities above 0.5 threshold

    m = compute_binary_metrics(t_pred, t_true, threshold=0.5)
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert abs(m["iou"] - 1.0) < 1e-6
