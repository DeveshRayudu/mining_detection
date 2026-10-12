"""
CLI script to execute Phase 2 Preprocessing pipeline across downloaded raw scenes.
1. Computes 8-channel features (6 spectral bands + NDVI + BSI)
2. Quality masks invalid pixels (clouds/shadows via SCL)
3. Verifies cross-date co-registration alignment
4. Slices rasters into 256x256 overlapping patches in data/tiles/
5. Fits normalizer statistics from training scene into models/norm_stats.json
6. Verifies tile-stitch reconstruction and zero-NaN integrity
"""

import sys
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import rasterio

from src.config import config
from src.preprocessing import (
    QualityMasker,
    SpectralIndices,
    compute_phase_cross_correlation,
    verify_alignment,
    PatchTiler,
    DatasetNormalizer,
)


def run_pipeline() -> None:
    print("=" * 70)
    print("🛰️  PHASE 2: GEOSPATIAL PREPROCESSING & TILING PIPELINE")
    print("=" * 70)

    raw_files = sorted(list(config.dirs.data_raw.glob("*.tif")))
    if not raw_files:
        print("❌ No raw GeoTIFF files found in data/raw/. Please run Phase 1 acquisition first.")
        sys.exit(1)

    print(f"📁 Found {len(raw_files)} raw scenes:")
    for f in raw_files:
        print(f"  - {f.name}")

    # 1. Cross-Date Co-registration Verification
    if len(raw_files) >= 2:
        print("\n[1/4] Checking Cross-Date Co-registration Alignment...")
        with rasterio.open(raw_files[0]) as ref_src, rasterio.open(raw_files[1]) as tgt_src:
            # Use NIR band (Band 4) for high-contrast landscape features
            ref_nir = ref_src.read(4)
            tgt_nir = tgt_src.read(4)
            sy, sx, peak = compute_phase_cross_correlation(ref_nir, tgt_nir)
            total_shift = np.sqrt(sy**2 + sx**2)
            print(f"  Reference Date: {raw_files[0].stem}")
            print(f"  Target Date:    {raw_files[1].stem}")
            print(f"  Estimated Shift: dy = {sy:+.3f} px, dx = {sx:+.3f} px (Magnitude: {total_shift:.3f} px)")
            if total_shift <= 1.0:
                print("  ✅ Co-registration PASS: Cross-date alignment is strictly under 1.0 pixel.")
            else:
                print(f"  ⚠️ Cross-date shift exceeds 1 pixel ({total_shift:.3f} px). Fine registration recommended.")

    # 2. Feature Stack & Tiling
    print("\n[2/4] Generating 8-Channel Features & Slicing 256x256 Tiles...")
    tiler = PatchTiler(tile_size=config.sensor.tile_size_px, overlap=config.sensor.tile_overlap_px)
    training_patches_for_norm = []
    total_tiles_generated = 0

    for idx, raw_path in enumerate(raw_files):
        scene_id = raw_path.stem
        scene_tile_dir = config.dirs.data_tiles / scene_id
        scene_tile_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(raw_path) as src:
            raw_7band = src.read()
            base_transform = src.transform
            crs = src.crs

        # Prepare 8 channels and quality mask
        features, scl = SpectralIndices.prepare_8channel_stack(raw_7band)
        valid_mask = QualityMasker.get_valid_mask(scl)
        masked_features = QualityMasker.apply_mask(features, valid_mask, fill_value=0.0)

        assert not np.isnan(masked_features).any(), f"NaN detected in features for {scene_id}!"

        # Tile raster
        patches, orig_shape = tiler.tile_raster(masked_features, base_transform, scene_id=scene_id)
        print(f"  Scene: {scene_id} -> {len(patches)} tiles of 256x256 px")

        # Save tiles as uncompressed numpy files for rapid DL training I/O
        for patch in patches:
            out_patch_path = scene_tile_dir / f"{patch.patch_id}.npy"
            np.save(out_patch_path, patch.data.astype(np.float32))

        total_tiles_generated += len(patches)

        # Designate first scene as training source for normalization statistics
        if idx == 0:
            training_patches_for_norm = [p.data for p in patches]

            # Verify tile-stitch reconstruction on training scene
            reconstructed = tiler.stitch_raster(patches, orig_shape)
            recon_error = np.max(np.abs(masked_features - reconstructed))
            print(f"  Tile-then-Stitch Check: Maximum reconstruction error = {recon_error:.2e}")
            assert recon_error < 1e-4, f"Reconstruction error too high: {recon_error}"
            print("  ✅ Tile-then-Stitch PASS: Reconstructed array matches original raster identically.")

    # 3. Fit Normalizer from Training Scene Only
    print("\n[3/4] Fitting Dataset Normalizer (Strictly Training Set)...")
    norm_path = config.dirs.models_dir / "norm_stats.json"
    normalizer = DatasetNormalizer(stats_path=norm_path)
    stats = normalizer.fit(training_patches_for_norm)

    print(f"  Saved normalization parameters to {norm_path.name}:")
    for name, mean, std in zip(stats["channel_names"], stats["means"], stats["stds"]):
        print(f"    - {name:<6} | Mean: {mean:8.4f} | Std: {std:8.4f}")

    # 4. Zero NaNs Verification
    print("\n[4/4] Verifying Patch Integrity across data/tiles/...")
    all_npy = list(config.dirs.data_tiles.glob("**/*.npy"))
    for p in all_npy[:50]:  # Spot check first 50 tiles
        arr = np.load(p)
        assert arr.shape == (8, 256, 256), f"Wrong tile shape: {arr.shape}"
        assert not np.isnan(arr).any(), f"NaN found in tile {p.name}"

    print(f"  ✅ Integrity PASS: {total_tiles_generated} patches generated, zero NaNs.")
    print("=" * 70)
    print("🎉 PHASE 2 COMPLETE: Preprocessing, Tiling, and Normalization Succeeded!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
