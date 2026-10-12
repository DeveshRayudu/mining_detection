"""
Dataset Normalization Module.
Computes channel-wise mean and standard deviation strictly from training samples
to prevent test set data leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class DatasetNormalizer:
    """Normalizes 8-channel satellite inputs using strictly training-set statistics."""

    def __init__(self, stats_path: Optional[Path] = None):
        self.stats_path = stats_path
        self.means: Optional[np.ndarray] = None  # Shape: (8, 1, 1)
        self.stds: Optional[np.ndarray] = None   # Shape: (8, 1, 1)
        self.channel_names = ["B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "BSI"]

        if stats_path and stats_path.exists():
            self.load(stats_path)

    def fit(self, training_arrays: List[np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculates channel-wise mean and std from a list of training arrays (C=8, H, W).
        """
        assert len(training_arrays) > 0, "No training samples provided to fit normalizer"
        assert training_arrays[0].shape[0] == 8, "Expected 8 channels per sample"

        # Accumulate sums and squared sums for numerical stability
        total_pixels = 0
        sum_channels = np.zeros(8, dtype=np.float64)
        sum_sq_channels = np.zeros(8, dtype=np.float64)

        for arr in training_arrays:
            # Flatten spatial dimensions per channel
            c, h, w = arr.shape
            pixels = h * w
            total_pixels += pixels
            for i in range(8):
                ch_data = arr[i].astype(np.float64)
                sum_channels[i] += np.sum(ch_data)
                sum_sq_channels[i] += np.sum(ch_data ** 2)

        mean_vals = sum_channels / total_pixels
        variance = (sum_sq_channels / total_pixels) - (mean_vals ** 2)
        variance = np.maximum(variance, 1e-6)  # Avoid division by zero
        std_vals = np.sqrt(variance)

        self.means = mean_vals.astype(np.float32).reshape(8, 1, 1)
        self.stds = std_vals.astype(np.float32).reshape(8, 1, 1)

        stats_dict = {
            "channel_names": self.channel_names,
            "means": [float(m) for m in mean_vals],
            "stds": [float(s) for s in std_vals],
        }

        if self.stats_path:
            self.save(self.stats_path)

        return stats_dict

    def transform(self, sample: np.ndarray) -> np.ndarray:
        """Standardizes an 8-channel array: (x - mean) / std."""
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer has not been fitted or loaded yet.")
        assert sample.shape[0] == 8, f"Expected 8 channels, got {sample.shape[0]}"
        return (sample - self.means) / (self.stds + 1e-7)

    def inverse_transform(self, sample: np.ndarray) -> np.ndarray:
        """Reverses standardization: x * std + mean."""
        if self.means is None or self.stds is None:
            raise RuntimeError("Normalizer has not been fitted or loaded yet.")
        return (sample * (self.stds + 1e-7)) + self.means

    def save(self, path: Path) -> None:
        """Save statistics to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "channel_names": self.channel_names,
            "means": [float(m) for m in self.means.flatten()],
            "stds": [float(s) for s in self.stds.flatten()],
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def load(self, path: Path) -> None:
        """Load statistics from JSON."""
        with open(path, "r") as f:
            stats = json.load(f)
        self.means = np.array(stats["means"], dtype=np.float32).reshape(8, 1, 1)
        self.stds = np.array(stats["stds"], dtype=np.float32).reshape(8, 1, 1)
