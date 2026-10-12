"""
PyTorch Mining Dataset implementation.
Loads 8-channel normalized satellite patches and binary masks,
with paired spatial augmentations.
"""

from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from src.config import config
from src.preprocessing import DatasetNormalizer


def get_training_augmentations() -> A.Compose:
    """Paired spatial augmentations applied to satellite image and mask together."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


class MiningDataset(Dataset):
    """PyTorch Dataset for Sentinel-2 Mining Segmentation."""

    def __init__(
        self,
        tile_ids: List[str],
        images_dir: Path,
        masks_dir: Path,
        normalizer: Optional[DatasetNormalizer] = None,
        transform: Optional[Callable] = None,
    ):
        self.tile_ids = tile_ids
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.normalizer = normalizer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int):
        tid = self.tile_ids[idx]
        img_path = self.images_dir / f"{tid}.npy"
        mask_path = self.masks_dir / f"{tid}.npy"

        # Load raw numpy arrays: img is (8, 256, 256), mask is (256, 256)
        img_np = np.load(img_path).astype(np.float32)
        mask_np = np.load(mask_path).astype(np.float32)

        # Normalize 8-channel features if normalizer is provided
        if self.normalizer is not None:
            img_np = self.normalizer.transform(img_np)

        # Albumentations expects (H, W, C) for multi-channel images
        if self.transform is not None:
            img_hwc = np.transpose(img_np, (1, 2, 0))  # (256, 256, 8)
            augmented = self.transform(image=img_hwc, mask=mask_np)
            img_hwc = augmented["image"]
            mask_np = augmented["mask"]
            img_np = np.transpose(img_hwc, (2, 0, 1))  # Back to (8, 256, 256)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(img_np).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # (1, 256, 256)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "tile_id": tid,
        }
