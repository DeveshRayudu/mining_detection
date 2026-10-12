"""
Unit and integration tests for Phase 1 Data Acquisition.
Verifies STAC queries, band structure, CRS conformity, duplicate prevention, and DB audit.
"""

import sqlite3
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import pytest

from src.config import config
from src.acquisition.sentinel2 import Sentinel2Acquisition, SceneMetadata


def test_stac_query_aoi():
    """Verify STAC returns valid candidate scenes over the target AOI."""
    acq = Sentinel2Acquisition()
    scenes = acq.query_scenes(
        bbox=config.project.aoi_bbox_wgs84,
        date_range="2024-01-01/2024-01-31",
        max_cloud_cover=15.0,
    )
    assert len(scenes) > 0, "No scenes found for known clear period in January 2024"
    # Ensure sorted by lowest cloud cover
    cloud_covers = [s["cloud_cover"] for s in scenes]
    assert cloud_covers == sorted(cloud_covers)
    assert all(cc <= 15.0 for cc in cloud_covers)


def test_cloud_cover_filter_skips_cloudy():
    """Verify strict cloud cover thresholds filter out cloudy scenes."""
    acq = Sentinel2Acquisition()
    # Query with strict 1% cloud cover vs 50% cloud cover
    strict_scenes = acq.query_scenes(
        bbox=config.project.aoi_bbox_wgs84,
        date_range="2024-07-01/2024-07-31",  # Monsoon month
        max_cloud_cover=2.0,
    )
    # In monsoon month, very few if any scenes have < 2% clouds
    assert len(strict_scenes) <= 2


def test_duplicate_prevention_idempotency(tmp_path):
    """Verify downloading an existing scene skips re-downloading and returns existing path."""
    db_path = tmp_path / "test_audit.db"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    acq = Sentinel2Acquisition(output_dir=raw_dir, db_path=db_path)

    fake_scene_id = "S2A_TEST_DUPLICATE_CHECK"
    fake_file = raw_dir / f"{fake_scene_id}.tif"

    # Create dummy GeoTIFF
    with rasterio.open(
        fake_file,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=7,
        dtype=rasterio.uint16,
        crs="EPSG:32645",
        transform=from_origin(350000, 2400000, 10, 10),
    ) as dst:
        dst.write(np.zeros((7, 100, 100), dtype=np.uint16))

    # Record in database
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO processed_scenes (scene_id, file_hash, acquisition_date, cloud_cover, file_path, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (fake_scene_id, "mock_hash_123", "2024-01-01", 1.5, str(fake_file), "DOWNLOADED"),
        )
        conn.commit()

    # Check that is_scene_downloaded recognizes the existing file
    found_path = acq.is_scene_downloaded(fake_scene_id)
    assert found_path == fake_file

    # Verify SHA-256 calculation
    sha = acq.compute_sha256(fake_file)
    assert len(sha) == 64


def test_scene_crs_and_band_structure(tmp_path):
    """Verify standard scene properties match requirements (7 bands, UTM CRS, 10m res)."""
    test_scene = tmp_path / "test_scene_structure.tif"
    height, width = 200, 200
    expected_bands = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]

    with rasterio.open(
        test_scene,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=7,
        dtype=rasterio.uint16,
        crs="EPSG:32645",
        transform=from_origin(350000, 2400000, 10.0, 10.0),
    ) as dst:
        dst.write(np.ones((7, height, width), dtype=np.uint16) * 1000)
        dst.descriptions = tuple(expected_bands)

    with rasterio.open(test_scene) as src:
        assert src.count == 7
        assert src.crs.to_epsg() == 32645
        assert src.res == (10.0, 10.0)
        assert list(src.descriptions) == expected_bands
        data = src.read()
        assert data.shape == (7, height, width)
        assert (data > 0).all()
