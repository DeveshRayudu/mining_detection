"""
Loss Functions for Geospatial Mining Segmentation.
Combined Binary Dice Loss + Focal Loss for handling severe class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Numerically stable Soft-Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized network predictions of shape (B, 1, H, W)
            targets: Binary ground truth labels of shape (B, 1, H, W) with values {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy background pixels and focuses gradient on hard mining boundary transitions.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedDiceFocalLoss(nn.Module):
    """Unified loss combining Dice Loss and Focal Loss."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = BinaryDiceLoss()
        self.focal_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        return (self.dice_weight * d_loss) + (self.focal_weight * f_loss)
