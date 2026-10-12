"""
Overlapping Tiler and Seamless Stitching Module.
Tiles large scenes into 256x256 patches and seamlessly stitches predictions or features back.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np
from affine import Affine


@dataclass
class Patch:
    """Container for an extracted raster patch."""
    data: np.ndarray  # Shape: (C, tile_size, tile_size) or (tile_size, tile_size)
    y_start: int
    x_start: int
    tile_size: int
    transform: Affine
    patch_id: str


class PatchTiler:
    """Slices large geospatial rasters into overlapping patches and stitches them back."""

    def __init__(self, tile_size: int = 256, overlap: int = 32):
        assert overlap < tile_size, "Overlap must be less than tile_size"
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self._weight_mask = self._create_2d_weight_mask()

    def _create_2d_weight_mask(self) -> np.ndarray:
        """
        Creates a 2D linear blending weight mask with smooth ramps along the overlapping borders.
        Ensures smooth transition between adjacent overlapping tiles.
        """
        mask_1d = np.ones(self.tile_size, dtype=np.float32)
        ramp = np.linspace(0.01, 1.0, self.overlap, dtype=np.float32)
        mask_1d[: self.overlap] = ramp
        mask_1d[-self.overlap :] = ramp[::-1]
        mask_2d = np.outer(mask_1d, mask_1d)
        return mask_2d

    def get_padded_shape(self, height: int, width: int) -> Tuple[int, int]:
        """Calculates padded dimensions so stride steps cover the entire raster."""
        pad_h = (
            height
            if (height - self.tile_size) % self.stride == 0
            else height + self.stride - ((height - self.tile_size) % self.stride)
        )
        pad_w = (
            width
            if (width - self.tile_size) % self.stride == 0
            else width + self.stride - ((width - self.tile_size) % self.stride)
        )
        pad_h = max(pad_h, self.tile_size)
        pad_w = max(pad_w, self.tile_size)
        return pad_h, pad_w

    def tile_raster(
        self,
        raster: np.ndarray,
        base_transform: Affine,
        scene_id: str = "scene",
    ) -> Tuple[List[Patch], Tuple[int, int]]:
        """
        Extracts overlapping patches from a raster array of shape (C, H, W) or (H, W).
        
        Returns:
            (list_of_patches, (original_height, original_width))
        """
        is_2d = raster.ndim == 2
        if is_2d:
            c, h, w = 1, raster.shape[0], raster.shape[1]
            data = raster[np.newaxis, ...]
        else:
            c, h, w = raster.shape[0], raster.shape[1], raster.shape[2]
            data = raster

        pad_h, pad_w = self.get_padded_shape(h, w)
        padded = np.pad(
            data,
            ((0, 0), (0, pad_h - h), (0, pad_w - w)),
            mode="reflect" if (pad_h - h < h and pad_w - w < w) else "edge",
        )

        patches: List[Patch] = []
        y_steps = range(0, pad_h - self.tile_size + 1, self.stride)
        x_steps = range(0, pad_w - self.tile_size + 1, self.stride)

        for y in y_steps:
            for x in x_steps:
                patch_data = padded[:, y : y + self.tile_size, x : x + self.tile_size]
                if is_2d:
                    patch_data = patch_data[0]

                patch_transform = base_transform * Affine.translation(x, y)
                patch_id = f"{scene_id}_y{y}_x{x}"
                patches.append(
                    Patch(
                        data=patch_data,
                        y_start=y,
                        x_start=x,
                        tile_size=self.tile_size,
                        transform=patch_transform,
                        patch_id=patch_id,
                    )
                )

        return patches, (h, w)

    def stitch_raster(
        self,
        patches: List[Patch],
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Reconstructs the original raster from overlapping patches using weighted blending.
        
        Args:
            patches: List of Patch objects
            original_shape: (height, width)
            
        Returns:
            Reconstructed array matching original dimensions.
        """
        h, w = original_shape
        pad_h, pad_w = self.get_padded_shape(h, w)

        sample = patches[0].data
        is_2d = sample.ndim == 2
        c = 1 if is_2d else sample.shape[0]

        weight_2d = self._weight_mask
        accum_data = np.zeros((c, pad_h, pad_w), dtype=np.float32)
        accum_weights = np.zeros((pad_h, pad_w), dtype=np.float32)

        for patch in patches:
            y, x = patch.y_start, patch.x_start
            pdata = patch.data[np.newaxis, ...] if is_2d else patch.data

            for ci in range(c):
                accum_data[ci, y : y + self.tile_size, x : x + self.tile_size] += (
                    pdata[ci] * weight_2d
                )
            accum_weights[y : y + self.tile_size, x : x + self.tile_size] += weight_2d

        # Normalize by blended weights
        mask = accum_weights > 1e-7
        for ci in range(c):
            accum_data[ci, mask] /= accum_weights[mask]

        # Crop back to original dimensions
        reconstructed = accum_data[:, :h, :w]
        return reconstructed[0] if is_2d else reconstructed
