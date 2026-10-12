"""
Geospatial Segmentation Metrics Module.
Calculates IoU, F1-Score (Dice), Precision, and Recall for the mining target class.
"""

from typing import Dict, Union
import numpy as np
import torch


def compute_binary_metrics(
    y_pred: Union[np.ndarray, torch.Tensor],
    y_true: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Computes confusion matrix and metrics on the positive (mining) class.
    
    Args:
        y_pred: Predicted probabilities or binary values (Numpy or Torch tensor)
        y_true: Ground truth binary mask {0, 1}
        threshold: Binarization threshold for probability inputs
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Dict containing iou, f1, precision, recall, tp, fp, fn, tn
    """
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Binarize
    pred_bin = (y_pred >= threshold).astype(bool)
    true_bin = (y_true >= 0.5).astype(bool)

    # Flatten arrays
    pred_flat = pred_bin.ravel()
    true_flat = true_bin.ravel()

    tp = int(np.logical_and(pred_flat, true_flat).sum())
    fp = int(np.logical_and(pred_flat, ~true_flat).sum())
    fn = int(np.logical_and(~pred_flat, true_flat).sum())
    tn = int(np.logical_and(~pred_flat, ~true_flat).sum())

    # Metrics
    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2 * tp / (2 * tp + fp + fn + eps))
    iou = float(tp / (tp + fp + fn + eps))

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
