"""
Geographic Train/Validation/Test Splitter.
Splits patches strictly by spatial geographical clusters with a spatial buffer zone,
guaranteeing ZERO pixel and spatial overlap between training and test sets.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import shapely.geometry
from affine import Affine

from src.config import config
from src.preprocessing import Patch


class GeographicSplitter:
    """Partitions spatial patches into Train, Validation, and Test by geographical blocks."""

    def __init__(self, output_manifest_path: Path = config.dirs.data_dir / "split_manifest.json"):
        self.manifest_path = output_manifest_path

    @staticmethod
    def get_patch_polygon(patch: Patch) -> shapely.geometry.Polygon:
        """Returns the spatial bounding polygon in projected metric coordinates."""
        # Patch bounds via transform
        t = patch.transform
        minx = t.c
        maxy = t.f
        maxx = minx + patch.tile_size * t.a
        miny = maxy + patch.tile_size * t.e  # t.e is negative (-10)
        return shapely.geometry.box(minx, miny, maxx, maxy)

    def split_by_geography(
        self,
        patches: List[Patch],
        buffer_margin_px: int = 32,
    ) -> Dict[str, List[str]]:
        """
        Geographic 3-way split using Northing (Y coordinate) bands with an exclusion buffer.
        
        Zones:
        - Southern Zone (Joda East / Kasia): Training set (~60% of tiles)
        - Buffer Zone: Excluded to prevent edge leakage
        - Central-North Zone: Validation set (~20% of tiles)
        - Northern Zone (Barbil / Thakurani North): Frozen Test set (~20% of tiles)
        
        Returns:
            Dictionary with 'train', 'val', 'test' lists of patch IDs.
        """
        assert len(patches) > 0, "No patches provided for splitting"

        # Extract unique y_start coordinates to divide into spatial latitude bands
        y_coords = sorted(list(set(p.y_start for p in patches)))
        n_bands = len(y_coords)
        assert n_bands >= 3, f"Need at least 3 spatial bands, got {n_bands}"

        # Assign spatial bands:
        # Train: bottom ~55% of bands
        train_cutoff_idx = int(n_bands * 0.50)
        # Buffer: 1 band between train and val
        val_start_idx = train_cutoff_idx + 1
        val_end_idx = val_start_idx + max(1, int(n_bands * 0.20))
        # Buffer: 1 band between val and test
        test_start_idx = val_end_idx + 1

        train_y_max = y_coords[train_cutoff_idx - 1]
        val_y_min = y_coords[val_start_idx]
        val_y_max = y_coords[val_end_idx - 1]
        test_y_min = y_coords[min(test_start_idx, n_bands - 1)]

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []
        excluded_buffer_ids: List[str] = []

        train_polys = []
        test_polys = []

        for p in patches:
            poly = self.get_patch_polygon(p)
            if p.y_start <= train_y_max:
                train_ids.append(p.patch_id)
                train_polys.append(poly)
            elif val_y_min <= p.y_start <= val_y_max:
                val_ids.append(p.patch_id)
            elif p.y_start >= test_y_min:
                test_ids.append(p.patch_id)
                test_polys.append(poly)
            else:
                excluded_buffer_ids.append(p.patch_id)

        # Mathematical verification: Assert zero intersection between any train and test tile
        for tr_poly in train_polys:
            for te_poly in test_polys:
                assert not tr_poly.intersects(te_poly), (
                    f"CRITICAL: Spatial intersection detected between train and test tiles!"
                )

        manifest = {
            "strategy": "Geographic Latitude/Northing Spatial Block Partition",
            "counts": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "buffer_excluded": len(excluded_buffer_ids),
                "total": len(patches),
            },
            "train": sorted(train_ids),
            "val": sorted(val_ids),
            "test": sorted(test_ids),
            "buffer_excluded": sorted(excluded_buffer_ids),
        }

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
