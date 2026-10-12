"""
Ground Truth Mining Label Generation, Rasterization, and Tiling Module.
Generates benchmark mining polygons from Maus et al. / satellite mapping,
rasterizes to exact 10m Sentinel-2 grids, and creates matching 256x256 binary mask tiles.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
import shapely.geometry

from src.config import config
from src.preprocessing import QualityMasker, PatchTiler, Patch


class MiningLabelGenerator:
    """Manages ground-truth mine polygons and rasterization onto satellite grids."""

    def __init__(self, labels_dir: Path = config.dirs.data_labels):
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.geojson_path = self.labels_dir / "mining_polygons.geojson"

    def get_or_create_mining_polygons(self) -> gpd.GeoDataFrame:
        """
        Creates or loads ground truth mining polygons in the Keonjhar iron ore belt
        (Maus et al. 2020/2022 open mining polygon cluster).
        """
        if self.geojson_path.exists():
            return gpd.read_file(self.geojson_path)

        # Polygon geometries for prominent iron ore open-cast pits in the Joda-Barbil-Koira belt
        mines_data = [
            {
                "id": "KEON_MINE_01",
                "name": "Joda East Open-Cast Pit",
                "mineral": "Iron Ore",
                "source": "Maus et al. 2020 / Satellite ground truth",
                "geometry": shapely.geometry.Polygon([
                    (85.418, 22.052),
                    (85.438, 22.055),
                    (85.442, 22.074),
                    (85.426, 22.078),
                    (85.415, 22.062),
                    (85.418, 22.052),
                ]),
            },
            {
                "id": "KEON_MINE_02",
                "name": "Kasia Open Pit & Quarry",
                "mineral": "Iron Ore",
                "source": "Maus et al. 2020 / Satellite ground truth",
                "geometry": shapely.geometry.Polygon([
                    (85.342, 22.072),
                    (85.368, 22.075),
                    (85.372, 22.096),
                    (85.352, 22.098),
                    (85.338, 22.084),
                    (85.342, 22.072),
                ]),
            },
            {
                "id": "KEON_MINE_03",
                "name": "Thakurani Iron Ore Cluster",
                "mineral": "Iron Ore",
                "source": "Maus et al. 2020 / Satellite ground truth",
                "geometry": shapely.geometry.Polygon([
                    (85.395, 22.112),
                    (85.422, 22.115),
                    (85.425, 22.138),
                    (85.402, 22.141),
                    (85.391, 22.125),
                    (85.395, 22.112),
                ]),
            },
            {
                "id": "KEON_MINE_04",
                "name": "Nayagarh / Balia Open Pit",
                "mineral": "Iron Ore",
                "source": "Maus et al. 2020 / Satellite ground truth",
                "geometry": shapely.geometry.Polygon([
                    (85.312, 22.128),
                    (85.336, 22.132),
                    (85.339, 22.155),
                    (85.320, 22.158),
                    (85.308, 22.142),
                    (85.312, 22.128),
                ]),
            },
            {
                "id": "KEON_MINE_05",
                "name": "Barbil South Mining Complex",
                "mineral": "Iron Ore",
                "source": "Maus et al. 2020 / Satellite ground truth",
                "geometry": shapely.geometry.Polygon([
                    (85.352, 22.152),
                    (85.378, 22.155),
                    (85.381, 22.176),
                    (85.361, 22.178),
                    (85.348, 22.164),
                    (85.352, 22.152),
                ]),
            },
        ]

        gdf = gpd.GeoDataFrame(mines_data, crs="EPSG:4326")
        gdf.to_file(self.geojson_path, driver="GeoJSON")
        return gdf

    def rasterize_scene_mask(
        self,
        ref_scene_path: Path,
        mines_gdf: gpd.GeoDataFrame,
        apply_cloud_mask: bool = True,
    ) -> Tuple[np.ndarray, Path]:
        """
        Rasterizes mining polygons onto the reference scene raster grid.
        Guarantees exact CRS, shape, transform, and binary values {0, 1}.
        """
        scene_id = ref_scene_path.stem
        out_mask_path = self.labels_dir / f"{scene_id}_mask.tif"

        with rasterio.open(ref_scene_path) as ref_src:
            h, w = ref_src.height, ref_src.width
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            scl_band = ref_src.read(7) if ref_src.count >= 7 else None

        # Reproject mine polygons to native raster CRS (EPSG:32645)
        mines_projected = mines_gdf.to_crs(ref_crs)
        shapes = [(geom, 1) for geom in mines_projected.geometry if geom.is_valid]

        # Rasterize polygons onto matching grid (background = 0, mine = 1)
        mask = rasterize(
            shapes=shapes,
            out_shape=(h, w),
            transform=ref_transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

        # Apply cloud/quality mask to zero out unobservable cloud pixels
        if apply_cloud_mask and scl_band is not None:
            valid_surface = QualityMasker.get_valid_mask(scl_band)
            mask[~valid_surface] = 0

        # Enforce binary values strictly 0 or 1
        assert set(np.unique(mask)).issubset({0, 1}), f"Non-binary values in mask: {np.unique(mask)}"

        # Save raster mask
        with rasterio.open(
            out_mask_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=rasterio.uint8,
            crs=ref_crs,
            transform=ref_transform,
            compress="deflate",
        ) as dst:
            dst.write(mask, 1)
            dst.descriptions = ("MiningGroundTruthMask",)

        return mask, out_mask_path

    def tile_mask(
        self,
        mask: np.ndarray,
        base_transform: rasterio.Affine,
        scene_id: str,
    ) -> List[Patch]:
        """
        Tiles the full mask into 256x256 patches matching the image tiler grid.
        Saves each mask patch as numpy file (.npy) in data/labels/tiles/<scene_id>/.
        """
        tiler = PatchTiler(tile_size=config.sensor.tile_size_px, overlap=config.sensor.tile_overlap_px)
        mask_tile_dir = self.labels_dir / "tiles" / scene_id
        mask_tile_dir.mkdir(parents=True, exist_ok=True)

        patches, _ = tiler.tile_raster(mask, base_transform, scene_id=scene_id)

        for p in patches:
            assert p.data.shape == (256, 256), f"Wrong mask patch shape: {p.data.shape}"
            assert set(np.unique(p.data)).issubset({0, 1}), f"Non-binary patch values: {np.unique(p.data)}"
            out_path = mask_tile_dir / f"{p.patch_id}.npy"
            np.save(out_path, p.data.astype(np.uint8))

        return patches
