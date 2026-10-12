"""
Sentinel-2 L2A Data Acquisition Module.
Handles STAC queries, cloud filtering, band-level Cloud-Optimized GeoTIFF (COG)
windowed reading, native 10m grid alignment, and audit ledger tracking.
"""

import hashlib
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from src.config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class SceneMetadata:
    """Metadata container for an acquired Sentinel-2 scene."""
    scene_id: str
    acquisition_date: str
    cloud_cover: float
    file_path: Path
    file_hash: str
    width: int
    height: int
    crs: str
    bands: List[str]


class Sentinel2Acquisition:
    """Repeatable Sentinel-2 downloader for the target Area of Interest (AOI)."""

    # Asset mapping between STAC keys and standardized Sentinel-2 band names
    BAND_MAP = {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "nir": "B08",
        "swir16": "B11",
        "swir22": "B12",
        "scl": "SCL",
    }

    def __init__(
        self,
        stac_url: Optional[str] = None,
        output_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        self.stac_url = stac_url or config.project.stac_api_url
        self.output_dir = output_dir or config.dirs.data_raw
        self.db_path = db_path or config.dirs.db_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Ensure the audit tables and required columns exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_scenes (
                    scene_id TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    acquisition_date TEXT,
                    cloud_cover REAL,
                    file_path TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL,
                    details TEXT
                )
            """)
            # Check existing columns and migrate if necessary
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(processed_scenes)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            for col, col_type in [
                ("file_path", "TEXT"),
                ("acquisition_date", "TEXT"),
                ("cloud_cover", "REAL"),
            ]:
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE processed_scenes ADD COLUMN {col} {col_type}")
            conn.commit()

    def query_scenes(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: str,
        max_cloud_cover: Optional[float] = None,
    ) -> List[Dict]:
        """
        Query open STAC catalog for Sentinel-2 L2A scenes over the AOI.
        
        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84
            date_range: e.g. "2024-01-01/2024-01-31"
            max_cloud_cover: Max allowed cloud cover percentage (default from config)
            
        Returns:
            List of scene summary dictionaries sorted by cloud cover ascending.
        """
        max_cloud = max_cloud_cover if max_cloud_cover is not None else config.project.cloud_cover_max_percent
        logger.info(f"Querying STAC at {self.stac_url} for bbox={bbox}, date={date_range}, cloud<={max_cloud}%")

        client = pystac_client.Client.open(self.stac_url)
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=list(bbox),
            datetime=date_range,
            query={"eo:cloud_cover": {"lte": max_cloud}},
        )

        items = list(search.items())
        logger.info(f"STAC search returned {len(items)} scenes below {max_cloud}% cloud cover.")

        results = []
        for item in items:
            cc = item.properties.get("eo:cloud_cover", 100.0)
            date_str = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"
            results.append({
                "id": item.id,
                "date": date_str,
                "cloud_cover": float(cc),
                "item_obj": item,
            })

        # Sort by lowest cloud cover first
        results.sort(key=lambda x: x["cloud_cover"])
        return results

    def is_scene_downloaded(self, scene_id: str) -> Optional[Path]:
        """Check if scene is already downloaded and recorded in DB (idempotency check)."""
        expected_path = self.output_dir / f"{scene_id}.tif"
        if expected_path.exists():
            return expected_path
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path FROM processed_scenes WHERE scene_id = ? AND status = 'DOWNLOADED'",
                (scene_id,)
            )
            row = cursor.fetchone()
            if row and Path(row[0]).exists():
                return Path(row[0])
        return None

    def download_scene(
        self,
        item,
        aoi_bbox: Tuple[float, float, float, float],
    ) -> SceneMetadata:
        """
        Download and composite the 6 spectral bands + 1 SCL layer into a single GeoTIFF.
        Uses windowed HTTP reads on Cloud-Optimized GeoTIFFs (COGs) to crop only the AOI.
        Resamples 20m bands (B11, B12, SCL) to the exact 10m grid of B02.
        """
        scene_id = item.id
        existing_file = self.is_scene_downloaded(scene_id)
        if existing_file:
            logger.info(f"Scene {scene_id} already exists at {existing_file}. Skipping duplicate download.")
            return self._load_metadata(existing_file, scene_id, item)

        out_path = self.output_dir / f"{scene_id}.tif"
        logger.info(f"Downloading and compositing scene {scene_id} -> {out_path.name}")

        # Reference 10m band: blue (B02)
        blue_asset = item.assets.get("blue")
        if not blue_asset:
            raise ValueError(f"Required 'blue' band asset missing in STAC item {scene_id}")

        blue_url = blue_asset.href

        # 1. Determine native CRS and AOI window in native coordinates using reference band
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(blue_url) as ref_src:
                ref_crs = ref_src.crs
                # Reproject WGS84 bbox into native raster CRS (e.g., UTM Zone 45N)
                native_minx, native_miny, native_maxx, native_maxy = transform_bounds(
                    "EPSG:4326", ref_crs, *aoi_bbox
                )
                
                raw_window = from_bounds(native_minx, native_miny, native_maxx, native_maxy, ref_src.transform)
                # Clamp to raster extents
                scene_window = raw_window.intersection(
                    rasterio.windows.Window(0, 0, ref_src.width, ref_src.height)
                ).round_shape()

                target_transform = rasterio.windows.transform(scene_window, ref_src.transform)
                target_height = int(scene_window.height)
                target_width = int(scene_window.width)

                if target_height <= 0 or target_width <= 0:
                    raise ValueError(f"AOI bounding box does not intersect scene {scene_id}")

                logger.info(f"Grid dimensions: {target_width}x{target_height} px at 10m resolution (CRS: {ref_crs})")

                # Prepare stack array: 7 bands (B02, B03, B04, B08, B11, B12, SCL)
                band_order = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]
                stacked_data = np.zeros((len(band_order), target_height, target_width), dtype=np.uint16)

                # 2. Extract and resample each band
                for idx, asset_key in enumerate(band_order):
                    asset = item.assets.get(asset_key)
                    if not asset:
                        raise ValueError(f"Band '{asset_key}' missing in STAC item {scene_id}")
                    
                    band_url = asset.href
                    is_20m = asset_key in ["swir16", "swir22", "scl"]
                    resampling_method = Resampling.nearest if asset_key == "scl" else Resampling.bilinear

                    with rasterio.open(band_url) as band_src:
                        if is_20m:
                            # Use WarpedVRT to precisely warp/resample 20m grid onto 10m target grid
                            vrt_options = {
                                "resampling": resampling_method,
                                "crs": ref_crs,
                                "transform": target_transform,
                                "height": target_height,
                                "width": target_width,
                            }
                            with WarpedVRT(band_src, **vrt_options) as vrt:
                                data = vrt.read(1)
                        else:
                            # 10m band shares reference grid exactly
                            data = band_src.read(1, window=scene_window)

                        stacked_data[idx] = data.astype(np.uint16)

        # 3. Write multi-band GeoTIFF
        band_names = [self.BAND_MAP[k] for k in band_order]
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=target_height,
            width=target_width,
            count=len(band_order),
            dtype=rasterio.uint16,
            crs=ref_crs,
            transform=target_transform,
            compress="deflate",
        ) as dst:
            dst.write(stacked_data)
            dst.descriptions = tuple(band_names)

        # 4. Compute file hash for integrity tracking
        file_hash = self.compute_sha256(out_path)
        cc = float(item.properties.get("eo:cloud_cover", 0.0))
        acq_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"

        # 5. Log in SQLite audit database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_scenes 
                (scene_id, file_hash, acquisition_date, cloud_cover, file_path, status, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scene_id,
                    file_hash,
                    acq_date,
                    cc,
                    str(out_path),
                    "DOWNLOADED",
                    f"Acquired via EarthSearch STAC. Bands: {','.join(band_names)}"
                ),
            )
            conn.commit()

        logger.info(f"Successfully saved and hashed scene: {scene_id} (SHA256: {file_hash[:12]}...)")
        return SceneMetadata(
            scene_id=scene_id,
            acquisition_date=acq_date,
            cloud_cover=cc,
            file_path=out_path,
            file_hash=file_hash,
            width=target_width,
            height=target_height,
            crs=str(ref_crs),
            bands=band_names,
        )

    def _load_metadata(self, file_path: Path, scene_id: str, item) -> SceneMetadata:
        """Construct metadata for an already downloaded file."""
        with rasterio.open(file_path) as src:
            width, height = src.width, src.height
            crs_str = src.crs.to_string()
            bands = list(src.descriptions) if src.descriptions else list(self.BAND_MAP.values())
        
        file_hash = self.compute_sha256(file_path)
        cc = float(item.properties.get("eo:cloud_cover", 0.0))
        acq_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"

        return SceneMetadata(
            scene_id=scene_id,
            acquisition_date=acq_date,
            cloud_cover=cc,
            file_path=file_path,
            file_hash=file_hash,
            width=width,
            height=height,
            crs=crs_str,
            bands=bands,
        )

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
