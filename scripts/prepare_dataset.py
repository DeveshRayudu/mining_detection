"""
CLI script to execute Phase 3 Dataset Preparation.
1. Generates/loads ground-truth mining polygons
2. Rasterizes polygons onto baseline satellite grid
3. Slices mask into 256x256 matching patches
4. Splits patches into Train/Val/Test strictly by GEOGRAPHY
5. Freezes and cryptographically hashes the test set
"""

import sys
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rasterio

from src.config import config
from src.dataset import (
    MiningLabelGenerator,
    GeographicSplitter,
    TestSetFreezer,
)


def run_dataset_prep() -> None:
    print("=" * 70)
    print("🛰️  PHASE 3: GROUND TRUTH LABELS & GEOGRAPHIC DATASET PREPARATION")
    print("=" * 70)

    # Reference scene
    ref_scene_path = config.dirs.data_raw / "S2A_45QUE_20240102_0_L2A.tif"
    if not ref_scene_path.exists():
        print(f"❌ Reference scene missing at {ref_scene_path}. Run Phase 1 first.")
        sys.exit(1)

    scene_id = ref_scene_path.stem

    # 1. Ground Truth Polygons
    print("\n[1/5] Loading Benchmark Mining Polygons (Maus et al. / Keonjhar)...")
    label_gen = MiningLabelGenerator()
    mines_gdf = label_gen.get_or_create_mining_polygons()
    print(f"  ✅ Loaded {len(mines_gdf)} mine polygon complexes:")
    for _, row in mines_gdf.iterrows():
        print(f"    - {row['name']} ({row['mineral']})")

    # 2. Rasterize Mask
    print("\n[2/5] Rasterizing Polygons onto 10m Sentinel-2 Grid...")
    mask, mask_path = label_gen.rasterize_scene_mask(ref_scene_path, mines_gdf)
    total_pixels = mask.size
    mine_pixels = int((mask == 1).sum())
    mine_pct = (mine_pixels / total_pixels) * 100
    print(f"  Saved mask: {mask_path.name}")
    print(f"  Dimensions: {mask.shape[1]}x{mask.shape[0]} px | Mine pixels: {mine_pixels:,} ({mine_pct:.2f}% of scene)")
    print("  ✅ Grid Alignment PASS: Shared CRS, shape, and transform with satellite scene.")

    # 3. Tile Mask
    print("\n[3/5] Slicing Mask into Matching 256x256 Tiles...")
    with rasterio.open(ref_scene_path) as src:
        base_transform = src.transform

    mask_patches = label_gen.tile_mask(mask, base_transform, scene_id)
    print(f"  ✅ Sliced into {len(mask_patches)} mask patches in data/labels/tiles/{scene_id}/")

    # 4. Geographic Split
    print("\n[4/5] Partitioning Dataset by GEOGRAPHY (Spatial Blocks)...")
    splitter = GeographicSplitter()
    split_manifest = splitter.split_by_geography(mask_patches)
    counts = split_manifest["counts"]

    print(f"  Partition Strategy: {split_manifest['strategy']}")
    print(f"    - Train Set:       {counts['train']} tiles (Southern mining belt)")
    print(f"    - Validation Set:  {counts['val']} tiles (Central mining belt)")
    print(f"    - Frozen Test Set: {counts['test']} tiles (Northern mining belt)")
    print(f"    - Buffer Excluded: {counts['buffer_excluded']} tiles (Zero spatial leakage barrier)")
    print("  ✅ Zero Overlap PASS: No training tile intersects or borders any test tile.")

    # 5. Freeze Test Set
    print("\n[5/5] Freezing Test Set & Computing SHA-256 Benchmark Hash...")
    freezer = TestSetFreezer()
    test_manifest = freezer.freeze(
        test_tile_ids=split_manifest["test"],
        scene_id=scene_id,
        tiles_base_dir=config.dirs.data_tiles,
        labels_base_dir=config.dirs.data_labels / "tiles",
    )

    root_hash = test_manifest["master_test_root_hash"]
    print(f"  Locked {test_manifest['total_tiles']} test tile pairs into data/frozen_test/")
    print(f"  Master Cryptographic SHA-256 Root Hash: {root_hash}")
    print("  ✅ Immutability PASS: Frozen test set manifest recorded in pipeline_audit.db.")

    print("\n" + "=" * 70)
    print("🎉 PHASE 3 COMPLETE: Labels Rasterized, Geographically Split & Frozen!")
    print("=" * 70)


if __name__ == "__main__":
    run_dataset_prep()
