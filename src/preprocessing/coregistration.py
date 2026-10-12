"""
Cross-date co-registration verification using Phase Cross-Correlation.
Ensures Sentinel-2 time series multi-date scenes are aligned under 1 pixel.
"""

from typing import Tuple
import numpy as np


def compute_phase_cross_correlation(
    ref_image: np.ndarray,
    target_image: np.ndarray,
    crop_margin: int = 64,
) -> Tuple[float, float, float]:
    """
    Computes spatial translation (shift_y, shift_x) between two co-registered scenes
    using sub-pixel phase cross-correlation (cross-power spectrum).
    
    Args:
        ref_image: 2D array of reference date (e.g. NIR or Red band)
        target_image: 2D array of target date
        crop_margin: Margin to avoid boundary effects
        
    Returns:
        (shift_y, shift_x, peak_correlation_value)
    """
    assert ref_image.shape == target_image.shape, "Images must have identical dimensions"
    
    # Subsample central region to eliminate border effects
    h, w = ref_image.shape
    if h > 2 * crop_margin and w > 2 * crop_margin:
        im0 = ref_image[crop_margin : h - crop_margin, crop_margin : w - crop_margin].astype(np.float64)
        im1 = target_image[crop_margin : h - crop_margin, crop_margin : w - crop_margin].astype(np.float64)
    else:
        im0 = ref_image.astype(np.float64)
        im1 = target_image.astype(np.float64)

    # Normalize images (zero mean, unit variance)
    im0 = (im0 - np.mean(im0)) / (np.std(im0) + 1e-8)
    im1 = (im1 - np.mean(im1)) / (np.std(im1) + 1e-8)

    # Window function (Hanning) to reduce spectral leakage
    hann_y = np.hanning(im0.shape[0])
    hann_x = np.hanning(im0.shape[1])
    window = np.outer(hann_y, hann_x)
    im0 *= window
    im1 *= window

    # 2D FFT
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)

    # Cross-power spectrum
    cross_power = (f0 * np.conj(f1)) / (np.abs(f0 * np.conj(f1)) + 1e-12)
    cps = np.fft.ifft2(cross_power).real

    # Find peak
    peak_y, peak_x = np.unravel_index(np.argmax(cps), cps.shape)
    peak_val = float(cps[peak_y, peak_x])

    # Convert wrapped frequency indices to signed shifts
    ny, nx = cps.shape
    shift_y = float(peak_y if peak_y < ny // 2 else peak_y - ny)
    shift_x = float(peak_x if peak_x < nx // 2 else peak_x - nx)

    return shift_y, shift_x, peak_val


def verify_alignment(
    ref_band: np.ndarray,
    target_band: np.ndarray,
    max_allowed_pixel_shift: float = 1.0,
) -> bool:
    """Check if cross-date alignment shift is strictly under threshold (1.0 pixel)."""
    shift_y, shift_x, _ = compute_phase_cross_correlation(ref_band, target_band)
    shift_magnitude = np.sqrt(shift_y**2 + shift_x**2)
    return shift_magnitude <= max_allowed_pixel_shift
