"""
Test Set Freezing and Cryptographic Integrity Hashing Module.
Freezes designated geographic test set tiles into data/frozen_test/
and generates an immutable SHA-256 hash manifest.
"""

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List

from src.config import config


class TestSetFreezer:
    """Copies and cryptographically locks the evaluation test set."""
    __test__ = False  # Tell pytest this is a domain class, not a test case

    def __init__(
        self,
        frozen_dir: Path = config.dirs.data_frozen_test,
        db_path: Path = config.dirs.db_path,
    ):
        self.frozen_dir = frozen_dir
        self.db_path = db_path
        self.images_dir = self.frozen_dir / "images"
        self.masks_dir = self.frozen_dir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sha256_file(path: Path) -> str:
        """Computes SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def freeze(
        self,
        test_tile_ids: List[str],
        scene_id: str,
        tiles_base_dir: Path = config.dirs.data_tiles,
        labels_base_dir: Path = config.dirs.data_labels / "tiles",
    ) -> Dict:
        """
        Locks test tiles, copies them into data/frozen_test/,
        and creates an immutable test manifest.
        """
        assert len(test_tile_ids) > 0, "No test tiles provided to freeze"
        file_hashes = []
        manifest_tiles = []

        for tid in sorted(test_tile_ids):
            src_img = tiles_base_dir / scene_id / f"{tid}.npy"
            src_mask = labels_base_dir / scene_id / f"{tid}.npy"

            assert src_img.exists(), f"Image tile missing: {src_img}"
            assert src_mask.exists(), f"Mask tile missing: {src_mask}"

            dst_img = self.images_dir / f"{tid}.npy"
            dst_mask = self.masks_dir / f"{tid}.npy"

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_mask, dst_mask)

            img_hash = self.sha256_file(dst_img)
            mask_hash = self.sha256_file(dst_mask)

            file_hashes.append(img_hash)
            file_hashes.append(mask_hash)

            manifest_tiles.append({
                "tile_id": tid,
                "image_sha256": img_hash,
                "mask_sha256": mask_hash,
            })

        # Calculate combined root hash
        combined_hasher = hashlib.sha256()
        for h in sorted(file_hashes):
            combined_hasher.update(h.encode("utf-8"))
        root_hash = combined_hasher.hexdigest()

        manifest = {
            "dataset": "Sentinel-2 Mining Detection Frozen Test Benchmark",
            "scene_id": scene_id,
            "total_tiles": len(test_tile_ids),
            "master_test_root_hash": root_hash,
            "tiles": manifest_tiles,
        }

        manifest_path = self.frozen_dir / "test_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Record in SQLite audit database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_manifests (
                    manifest_id TEXT PRIMARY KEY,
                    root_hash TEXT NOT NULL,
                    tile_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    manifest_path TEXT NOT NULL
                )
            """)
            conn.execute(
                """
                INSERT OR REPLACE INTO benchmark_manifests
                (manifest_id, root_hash, tile_count, manifest_path)
                VALUES (?, ?, ?, ?)
                """,
                (f"FROZEN_TEST_{scene_id}", root_hash, len(test_tile_ids), str(manifest_path)),
            )
            conn.commit()

        return manifest
