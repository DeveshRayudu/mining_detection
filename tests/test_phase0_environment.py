"""
Unit tests for Phase 0 environment and baseline assumptions.
"""

import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import shapely.geometry
import geopandas as gpd
import torch
import segmentation_models_pytorch as smp

from src.config import config


def test_directories_exist():
    config.ensure_directories()
    assert config.dirs.inbox_images.exists()
    assert config.dirs.inbox_training.exists()
    assert config.dirs.models_checkpoints.exists()
    assert config.dirs.models_current.exists()
    assert config.dirs.data_raw.exists()
    assert config.dirs.data_tiles.exists()
    assert config.dirs.data_labels.exists()
    assert config.dirs.data_frozen_test.exists()
    assert config.dirs.outputs_masks.exists()
    assert config.dirs.outputs_geojson.exists()
    assert config.dirs.outputs_reports.exists()


def test_rasterio_multichannel_io():
    channels = 8
    h, w = config.sensor.tile_size_px, config.sensor.tile_size_px
    res = config.sensor.target_resolution_m
    crs = config.project.target_crs
    transform = from_origin(350000.0, 2400000.0, res, res)
    data = np.random.uniform(0.0, 1.0, size=(channels, h, w)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tif = Path(tmpdir) / "test_patch.tif"
        with rasterio.open(
            out_tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=channels,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

        with rasterio.open(out_tif) as src:
            assert src.count == 8
            assert src.crs.to_string() == crs
            assert src.res == (res, res)
            arr = src.read()
            assert arr.shape == (8, h, w)
            assert not np.isnan(arr).any()


def test_shapely_geopandas_metric_crs():
    poly = shapely.geometry.Polygon([
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
        (0.0, 0.0)
    ])
    assert poly.is_valid
    assert abs(poly.area - 10000.0) < 1e-4

    gdf = gpd.GeoDataFrame([{"geometry": poly}], crs="EPSG:32645")
    assert gdf.crs.to_epsg() == 32645


def test_unet_resnet34_architecture():
    # 8-channel input (B2, B3, B4, B8, B11, B12, NDVI, BSI)
    x = torch.randn(2, 8, 256, 256)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=8,
        classes=1,
    )
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 256, 256)
    assert not torch.isnan(out).any()
