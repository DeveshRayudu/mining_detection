"""
Spectral index calculation (NDVI, BSI) and multi-channel feature preparation.
"""

from typing import Tuple
import numpy as np


class SpectralIndices:
    """Computes specialized remote sensing indices for mine and vegetation segmentation."""

    # Expected index mapping in the 7-band raw GeoTIFF:
    # 0: B02, 1: B03, 2: B04, 3: B08, 4: B11, 5: B12, 6: SCL
    IDX_BLUE = 0
    IDX_GREEN = 1
    IDX_RED = 2
    IDX_NIR = 3
    IDX_SWIR1 = 4
    IDX_SWIR2 = 5
    IDX_SCL = 6

    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)
        Dense vegetation (forests) ~ 0.5 to 0.85
        Open mines, bare soil, rock ~ -0.1 to 0.2
        Water ~ negative
        """
        denom = nir + red
        ndvi = np.where(denom > eps, (nir - red) / (denom + eps), 0.0)
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def compute_bsi(
        blue: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir1: np.ndarray,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Bare Soil Index: ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        High values indicate exposed mineral soil, open-cast pits, and quarry faces.
        """
        num = (swir1 + red) - (nir + blue)
        denom = (swir1 + red) + (nir + blue)
        bsi = np.where(denom > eps, num / (denom + eps), 0.0)
        return np.clip(bsi, -1.0, 1.0)

    @classmethod
    def prepare_8channel_stack(cls, raw_7band_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts raw 7-band uint16 GeoTIFF into:
        1. Float32 8-channel array [B02, B03, B04, B08, B11, B12, NDVI, BSI]
           where raw bands are scaled to [0, 1] reflectance (DN / 10000.0).
        2. 2D SCL array (uint8).
        
        Args:
            raw_7band_data: Shape (7, H, W)
            
        Returns:
            (feature_stack [8, H, W], scl_mask [H, W])
        """
        h, w = raw_7band_data.shape[1], raw_7band_data.shape[2]
        
        # Scale reflectance to [0, 1]
        refl = raw_7band_data[:6].astype(np.float32) / 10000.0
        refl = np.clip(refl, 0.0, 1.0)

        blue = refl[cls.IDX_BLUE]
        green = refl[cls.IDX_GREEN]
        red = refl[cls.IDX_RED]
        nir = refl[cls.IDX_NIR]
        swir1 = refl[cls.IDX_SWIR1]
        swir2 = refl[cls.IDX_SWIR2]
        scl = raw_7band_data[cls.IDX_SCL].astype(np.uint8)

        # Compute spectral indices
        ndvi = cls.compute_ndvi(nir, red)
        bsi = cls.compute_bsi(blue, red, nir, swir1)

        # Stack into 8 channels: 6 spectral bands + 2 indices
        features = np.zeros((8, h, w), dtype=np.float32)
        features[0:6] = refl
        features[6] = ndvi
        features[7] = bsi

        return features, scl
