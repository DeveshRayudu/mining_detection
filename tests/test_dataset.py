"""
Unit tests for Phase 3: Labels, Geographic Split, and Test Set Freezing.
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from affine import Affine
import shapely.geometry
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.dataset import (
    MiningLabelGenerator,
    GeographicSplitter,
    TestSetFreezer,
    MiningDataset,
)
from src.preprocessing import Patch


def test_label_raster_shares_crs_and_transform(tmp_path):
    """Gate check: mask and image share exact size, CRS, and transform; values only {0, 1}."""
    h, w = 150, 150
    ref_file = tmp_path / "mock_ref.tif"
    ref_transform = Affine.translation(350000.0, 2400000.0) * Affine.scale(10.0, -10.0)
    ref_crs = "EPSG:32645"

    with rasterio.open(
        ref_file,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=7,
        dtype=rasterio.uint16,
        crs=ref_crs,
        transform=ref_transform,
    ) as dst:
        dst.write(np.zeros((7, h, w), dtype=np.uint16))

    label_gen = MiningLabelGenerator(labels_dir=tmp_path)
    mines_gdf = label_gen.get_or_create_mining_polygons()
    mask, mask_path = label_gen.rasterize_scene_mask(ref_file, mines_gdf, apply_cloud_mask=False)

    with rasterio.open(mask_path) as mask_src:
        assert mask_src.width == w
        assert mask_src.height == h
        assert mask_src.crs.to_string() == ref_crs
        assert mask_src.transform == ref_transform
        data = mask_src.read(1)
        # Values must be only 0 or 1
        assert set(np.unique(data)).issubset({0, 1})


def test_geographic_split_zero_train_test_overlap():
    """Gate check: no train/test overlap, including spatial tile overlap."""
    t = Affine.translation(350000.0, 2400000.0) * Affine.scale(10.0, -10.0)
    stride = 224
    tile_size = 256

    # Create a grid of mock patches across 5 rows and 3 columns
    mock_patches = []
    for r in range(5):
        y = r * stride
        for c in range(3):
            x = c * stride
            patch_transform = t * Affine.translation(x, y)
            mock_patches.append(
                Patch(
                    data=np.zeros((tile_size, tile_size), dtype=np.uint8),
                    y_start=y,
                    x_start=x,
                    tile_size=tile_size,
                    transform=patch_transform,
                    patch_id=f"tile_r{r}_c{c}",
                )
            )

    splitter = GeographicSplitter()
    manifest = splitter.split_by_geography(mock_patches)

    train_ids = set(manifest["train"])
    test_ids = set(manifest["test"])

    # Disjoint ID sets
    assert train_ids.isdisjoint(test_ids)

    # Disjoint spatial geometry sets (with buffer gap)
    train_patches = [p for p in mock_patches if p.patch_id in train_ids]
    test_patches = [p for p in mock_patches if p.patch_id in test_ids]

    for trp in train_patches:
        tr_poly = splitter.get_patch_polygon(trp)
        for tep in test_patches:
            te_poly = splitter.get_patch_polygon(tep)
            assert not tr_poly.intersects(te_poly), "Overlap detected between train and test patch!"
            # Minimum distance between train and test must be > 0 (buffer zone exists)
            assert tr_poly.distance(te_poly) > 0.0


def test_frozen_test_manifest_hash_stored(tmp_path):
    """Gate check: test set hash stored and reproducible."""
    frozen_dir = tmp_path / "frozen"
    db_path = tmp_path / "audit.db"
    tiles_dir = tmp_path / "tiles" / "scene1"
    labels_dir = tmp_path / "labels" / "scene1"
    tiles_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create synthetic test tiles
    test_ids = ["test_tile_01", "test_tile_02"]
    for tid in test_ids:
        np.save(tiles_dir / f"{tid}.npy", np.random.randn(8, 256, 256).astype(np.float32))
        np.save(labels_dir / f"{tid}.npy", np.random.randint(0, 2, size=(256, 256)).astype(np.uint8))

    freezer = TestSetFreezer(frozen_dir=frozen_dir, db_path=db_path)
    manifest = freezer.freeze(
        test_tile_ids=test_ids,
        scene_id="scene1",
        tiles_base_dir=tmp_path / "tiles",
        labels_base_dir=tmp_path / "labels",
    )

    assert "master_test_root_hash" in manifest
    assert len(manifest["master_test_root_hash"]) == 64
    assert manifest["total_tiles"] == 2

    manifest_file = frozen_dir / "test_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file) as f:
        loaded = json.load(f)
    assert loaded["master_test_root_hash"] == manifest["master_test_root_hash"]


def test_pytorch_mining_dataset_batch_shapes(tmp_path):
    """Verify MiningDataset DataLoader delivers expected tensor shapes."""
    img_dir = tmp_path / "imgs"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    tile_ids = ["patch_01", "patch_02"]
    for tid in tile_ids:
        np.save(img_dir / f"{tid}.npy", np.random.uniform(0.0, 1.0, (8, 256, 256)).astype(np.float32))
        np.save(mask_dir / f"{tid}.npy", np.random.choice([0, 1], size=(256, 256)).astype(np.float32))

    dataset = MiningDataset(
        tile_ids=tile_ids,
        images_dir=img_dir,
        masks_dir=mask_dir,
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert "image" in batch and "mask" in batch
    assert batch["image"].shape == (2, 8, 256, 256)
    assert batch["mask"].shape == (2, 1, 256, 256)
    assert batch["image"].dtype == torch.float32
    assert batch["mask"].dtype == torch.float32
    assert set(batch["mask"].unique().tolist()).issubset({0.0, 1.0})
