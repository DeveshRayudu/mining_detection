"""
Unit tests for Phase 2: Preprocessing, Quality Masking, Indices, Tiling, and Normalization.
"""

from pathlib import Path
import numpy as np
from affine import Affine
import pytest

from src.config import config
from src.preprocessing import (
    QualityMasker,
    SpectralIndices,
    compute_phase_cross_correlation,
    verify_alignment,
    PatchTiler,
    DatasetNormalizer,
)


def test_quality_masker_identifies_clouds_and_shadows():
    """Verify SCL masking sets clouds (classes 8, 9, 10) and shadows (class 3) to False."""
    scl = np.array([
        [4, 4, 3],   # vegetation, vegetation, cloud shadow
        [5, 8, 9],   # bare soil, medium cloud, high cloud
        [6, 10, 4],  # water, cirrus, vegetation
    ], dtype=np.uint8)

    valid_mask = QualityMasker.get_valid_mask(scl)

    # Valid: (0,0), (0,1), (1,0), (2,0), (2,2)
    assert valid_mask[0, 0] is np.True_  # vegetation
    assert valid_mask[0, 1] is np.True_  # vegetation
    assert valid_mask[0, 2] is np.False_ # shadow
    assert valid_mask[1, 0] is np.True_  # bare soil
    assert valid_mask[1, 1] is np.False_ # medium cloud
    assert valid_mask[1, 2] is np.False_ # high cloud
    assert valid_mask[2, 0] is np.True_  # water
    assert valid_mask[2, 1] is np.False_ # cirrus
    assert valid_mask[2, 2] is np.True_  # vegetation


def test_spectral_indices_computation_ranges():
    """Verify NDVI and BSI produce mathematically valid outputs in [-1.0, 1.0]."""
    nir = np.array([[0.6, 0.1]], dtype=np.float32)
    red = np.array([[0.1, 0.2]], dtype=np.float32)
    blue = np.array([[0.05, 0.15]], dtype=np.float32)
    swir1 = np.array([[0.15, 0.45]], dtype=np.float32)

    ndvi = SpectralIndices.compute_ndvi(nir, red)
    bsi = SpectralIndices.compute_bsi(blue, red, nir, swir1)

    assert not np.isnan(ndvi).any()
    assert not np.isnan(bsi).any()
    assert (ndvi >= -1.0).all() and (ndvi <= 1.0).all()
    assert (bsi >= -1.0).all() and (bsi <= 1.0).all()

    # Vegetation: high NIR, low Red -> high positive NDVI
    assert ndvi[0, 0] > 0.5
    # Bare soil/pit: high SWIR1, high Red -> higher BSI
    assert bsi[0, 1] > bsi[0, 0]


def test_tile_then_stitch_reconstruction_equality():
    """Gate check: slicing a raster into 256x256 patches and stitching back equals original identically."""
    tiler = PatchTiler(tile_size=256, overlap=32)
    h, w = 400, 500  # Non-multiple of tile size to test padding
    channels = 8

    # Synthetic multi-channel raster
    original_data = np.random.uniform(0.05, 0.95, size=(channels, h, w)).astype(np.float32)
    base_transform = Affine.translation(350000.0, 2400000.0)

    # 1. Tile
    patches, orig_shape = tiler.tile_raster(original_data, base_transform)
    assert len(patches) > 0
    for p in patches:
        assert p.data.shape == (channels, 256, 256)
        assert not np.isnan(p.data).any()

    # 2. Stitch
    reconstructed = tiler.stitch_raster(patches, orig_shape)
    assert reconstructed.shape == original_data.shape

    # 3. Check equality
    max_diff = np.max(np.abs(original_data - reconstructed))
    assert max_diff < 1e-4, f"Reconstruction error {max_diff} exceeded tolerance"


def test_coregistration_cross_correlation_precision():
    """Verify phase cross-correlation accurately detects sub-pixel alignment."""
    # Synthetic texture
    rng = np.random.RandomState(42)
    base = rng.uniform(0.0, 1.0, size=(256, 256))

    # Identical image -> 0 shift
    sy, sx, _ = compute_phase_cross_correlation(base, base)
    assert abs(sy) < 1e-3
    assert abs(sx) < 1e-3
    assert verify_alignment(base, base, max_allowed_pixel_shift=1.0)


def test_normalizer_strictly_fit_and_transforms(tmp_path):
    """Verify normalizer computes means/stds correctly and standardizes data."""
    stats_file = tmp_path / "norm.json"
    norm = DatasetNormalizer(stats_path=stats_file)

    p1 = np.ones((8, 256, 256), dtype=np.float32) * 2.0
    p2 = np.ones((8, 256, 256), dtype=np.float32) * 4.0

    stats = norm.fit([p1, p2])
    # Expected mean = 3.0, std = 1.0
    for m in stats["means"]:
        assert abs(m - 3.0) < 1e-4
    for s in stats["stds"]:
        assert abs(s - 1.0) < 1e-4

    # Test transform
    sample = np.ones((8, 256, 256), dtype=np.float32) * 3.0
    transformed = norm.transform(sample)
    assert np.allclose(transformed, 0.0, atol=1e-5)

    # Test inverse transform
    inv = norm.inverse_transform(transformed)
    assert np.allclose(inv, 3.0, atol=1e-5)
