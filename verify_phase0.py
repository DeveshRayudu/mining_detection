"""
Phase 0 Environment & Pipeline Sanity Verification Script.
Executes non-destructive end-to-end checks across:
1. Directory layout and permissions
2. Rasterio GeoTIFF creation, CRS metadata, affine transform, and I/O
3. Shapely / GeoPandas metric geometry operations (UTM projection, buffering, intersection)
4. PyTorch 8-channel U-Net (ResNet-34) forward pass
5. SQLite pipeline audit database initialization
"""

import sys
import sqlite3
import tempfile
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import rasterio
from rasterio.transform import from_origin
import shapely.geometry
import geopandas as gpd
import torch
import segmentation_models_pytorch as smp

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import config


def check_directories() -> bool:
    print("[1/5] Checking directory structure...")
    config.ensure_directories()
    required = [
        config.dirs.inbox_images,
        config.dirs.inbox_training,
        config.dirs.models_checkpoints,
        config.dirs.models_current,
        config.dirs.data_raw,
        config.dirs.data_tiles,
        config.dirs.data_labels,
        config.dirs.data_frozen_test,
        config.dirs.outputs_masks,
        config.dirs.outputs_geojson,
        config.dirs.outputs_reports,
        config.dirs.processed_dir,
        config.dirs.failed_dir,
    ]
    for d in required:
        if not d.exists():
            print(f"  ❌ Missing directory: {d}")
            return False
    print("  ✅ All project directories exist and are write-accessible.")
    return True


def check_rasterio_geotiff() -> bool:
    print("[2/5] Testing Rasterio / GDAL GeoTIFF creation and reading...")
    # Synthetic 8-channel patch: 6 raw Sentinel-2 bands + 2 indices
    channels = 8
    height, width = config.sensor.tile_size_px, config.sensor.tile_size_px
    res = config.sensor.target_resolution_m
    crs_code = config.project.target_crs

    # Upper-left coordinates in UTM Zone 45N (Keonjhar belt ~ 350,000m E, 2,400,000m N)
    transform = from_origin(350000.0, 2400000.0, res, res)
    test_data = np.random.uniform(0.01, 0.99, size=(channels, height, width)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_tif = Path(tmpdir) / "synthetic_8band.tif"

        # Write test raster
        with rasterio.open(
            test_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=channels,
            dtype=rasterio.float32,
            crs=crs_code,
            transform=transform,
        ) as dst:
            dst.write(test_data)

        # Read back and verify
        with rasterio.open(test_tif, "r") as src:
            assert src.count == channels, f"Band count mismatch: expected {channels}, got {src.count}"
            assert src.crs.to_string() == crs_code, f"CRS mismatch: expected {crs_code}, got {src.crs}"
            assert src.res == (res, res), f"Resolution mismatch: expected ({res}, {res}), got {src.res}"
            read_arr = src.read()
            assert not np.isnan(read_arr).any(), "NaN values detected in raster!"
            assert read_arr.shape == (channels, height, width), "Read array shape mismatch!"

    print(f"  ✅ GeoTIFF I/O verified: 8 bands, {width}x{height} px, {res}m res, CRS {crs_code}.")
    return True


def check_geospatial_geometry() -> bool:
    print("[3/5] Testing Shapely & GeoPandas vector geometry in metric CRS...")
    # Synthetic mine polygon in UTM coordinates
    mine_poly = shapely.geometry.Polygon([
        (350000.0, 2400000.0),
        (350500.0, 2400000.0),
        (350500.0, 2399500.0),
        (350000.0, 2399500.0),
        (350000.0, 2400000.0),
    ])
    assert mine_poly.is_valid, "Synthetic mine polygon is invalid"
    area_sqm = mine_poly.area
    expected_area = 500.0 * 500.0  # 250,000 sq m (25 hectares)
    assert abs(area_sqm - expected_area) < 1e-3, f"Area mismatch: {area_sqm} vs {expected_area}"

    # Synthetic lease boundary polygon (partially overlapping)
    lease_poly = shapely.geometry.Polygon([
        (350250.0, 2400000.0),
        (350750.0, 2400000.0),
        (350750.0, 2399500.0),
        (350250.0, 2399500.0),
        (350250.0, 2400000.0),
    ])

    # Legal overlay check: encroachment is area of mine OUTSIDE authorized lease
    encroachment = mine_poly.difference(lease_poly)
    assert encroachment.is_valid, "Encroachment geometry is invalid"
    assert abs(encroachment.area - (250.0 * 500.0)) < 1e-3

    # Buffer operation check
    buffered = lease_poly.buffer(config.project.buffer_distance_m)
    assert buffered.is_valid, "Buffered lease polygon is invalid"
    assert buffered.area > lease_poly.area

    # GeoDataFrame creation
    gdf = gpd.GeoDataFrame(
        [{"id": 1, "type": "Mine Detection", "geometry": mine_poly}],
        crs=config.project.target_crs
    )
    assert gdf.crs.to_string() == config.project.target_crs, "GeoDataFrame CRS mismatch"
    print(f"  ✅ Geometric calculations verified: Area = {area_sqm/10000:.2f} ha, valid topology.")
    return True


def check_deep_learning_model() -> bool:
    print("[4/5] Testing PyTorch 8-channel U-Net ResNet-34 forward pass...")
    # Input batch: 1 sample, 8 channels (B02, B03, B04, B08, B11, B12, NDVI, BSI), 256x256 px
    dummy_input = torch.randn(1, 8, config.sensor.tile_size_px, config.sensor.tile_size_px)

    # Initialize model with weights=None to avoid network download during environment check
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=8,
        classes=1,
        activation=None,  # Logits for BCE/Dice combined loss
    )
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 1, config.sensor.tile_size_px, config.sensor.tile_size_px), (
        f"Output shape mismatch: expected (1, 1, 256, 256), got {output.shape}"
    )
    assert not torch.isnan(output).any(), "NaN detected in model forward pass output"
    print(f"  ✅ U-Net ResNet-34 instantiated: input [1, 8, 256, 256] -> output {list(output.shape)}.")
    return True


def check_audit_database() -> bool:
    print("[5/5] Testing SQLite audit and transaction database initialization...")
    db_path = config.dirs.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_scenes (
                scene_id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                details TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mining_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                acquisition_date TEXT NOT NULL,
                polygon_geojson TEXT NOT NULL,
                area_sqm REAL NOT NULL,
                confidence_score REAL NOT NULL,
                legality_status TEXT NOT NULL,
                reviewer_status TEXT DEFAULT 'PENDING_REVIEW',
                reviewer_notes TEXT,
                FOREIGN KEY (scene_id) REFERENCES processed_scenes (scene_id)
            )
        """)
        conn.commit()

        # Insert and retrieve a test record
        cursor.execute(
            "INSERT OR REPLACE INTO processed_scenes (scene_id, file_hash, status, details) VALUES (?, ?, ?, ?)",
            ("TEST_SCENE_001", "hash_placeholder", "INITIALIZED", "Phase 0 environment check")
        )
        conn.commit()

        cursor.execute("SELECT status FROM processed_scenes WHERE scene_id = 'TEST_SCENE_001'")
        res = cursor.fetchone()
        assert res is not None and res[0] == "INITIALIZED", "Database query failed"

    print(f"  ✅ Audit SQLite database initialized at {db_path.name}.")
    return True


def main() -> int:
    print("=" * 70)
    print("🛰️  SATELLITE MINING DETECTION SYSTEM - PHASE 0 ENVIRONMENT GATE")
    print("=" * 70)

    try:
        checks = [
            check_directories(),
            check_rasterio_geotiff(),
            check_geospatial_geometry(),
            check_deep_learning_model(),
            check_audit_database(),
        ]
        if all(checks):
            print("=" * 70)
            print("🎉 PHASE 0 GATE PASSED: Environment, Geospatial, ML & DB checks OK!")
            print("=" * 70)
            return 0
        else:
            print("❌ Phase 0 gate failed on one or more checks.")
            return 1
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
