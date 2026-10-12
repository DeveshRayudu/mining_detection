"""
Quality and Cloud Masking for Sentinel-2 Scenes using SCL (Scene Classification Layer).
"""

import numpy as np


class QualityMasker:
    """Extracts valid data masks from Sentinel-2 SCL band."""

    # SCL Class identifiers
    NO_DATA = 0
    SATURATED = 1
    DARK_AREA = 2
    CLOUD_SHADOW = 3
    VEGETATION = 4
    NOT_VEGETATED = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROB = 8
    CLOUD_HIGH_PROB = 9
    THIN_CIRRUS = 10
    SNOW = 11

    # Standard mask: exclude clouds, thin cirrus, cloud shadows, saturated, and nodata
    INVALID_CLASSES = {NO_DATA, SATURATED, CLOUD_SHADOW, CLOUD_MEDIUM_PROB, CLOUD_HIGH_PROB, THIN_CIRRUS}

    @classmethod
    def get_valid_mask(cls, scl: np.ndarray) -> np.ndarray:
        """
        Generate a boolean mask where True indicates clean, valid land/water surface.
        
        Args:
            scl: 2D numpy array of SCL classes.
            
        Returns:
            Boolean array of shape (H, W). True = valid pixel, False = cloud/shadow/nodata.
        """
        valid = np.ones(scl.shape, dtype=bool)
        for invalid_class in cls.INVALID_CLASSES:
            valid &= (scl != invalid_class)
        return valid

    @classmethod
    def apply_mask(cls, image_stack: np.ndarray, valid_mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
        """
        Zero out (or fill) invalid pixels across all channels of an image stack.
        
        Args:
            image_stack: Array of shape (C, H, W)
            valid_mask: Boolean array of shape (H, W)
            fill_value: Value to set for masked pixels
            
        Returns:
            Masked array of shape (C, H, W)
        """
        masked = image_stack.copy()
        inv_mask = ~valid_mask
        for c in range(masked.shape[0]):
            masked[c, inv_mask] = fill_value
        return masked
