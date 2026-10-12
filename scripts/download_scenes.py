"""
CLI script to query and download multi-temporal Sentinel-2 L2A scenes for the target AOI.
"""

import sys
from pathlib import Path

# Force UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.acquisition.sentinel2 import Sentinel2Acquisition


def run_acquisition(max_scenes_per_window: int = 1) -> None:
    print("=" * 70)
    print(f"🛰️  SENTINEL-2 ACQUISITION: {config.project.district.upper()} ({config.project.mineral})")
    print(f"📍 AOI WGS84 BBox: {config.project.aoi_bbox_wgs84}")
    print(f"☁️  Max Cloud Cover: {config.project.cloud_cover_max_percent}%")
    print("=" * 70)

    acq = Sentinel2Acquisition()
    downloaded_scenes = []

    for start_date, end_date in config.project.default_acquisition_dates:
        date_range = f"{start_date}/{end_date}"
        print(f"\n🔍 Searching temporal window: {date_range}...")
        
        candidates = acq.query_scenes(
            bbox=config.project.aoi_bbox_wgs84,
            date_range=date_range,
            max_cloud_cover=config.project.cloud_cover_max_percent,
        )

        if not candidates:
            print(f"  ⚠️ No scenes found with cloud cover <= {config.project.cloud_cover_max_percent}% for {date_range}")
            continue

        print(f"  Found {len(candidates)} candidates. Best match:")
        best = candidates[0]
        print(f"  👉 Scene ID: {best['id']} | Date: {best['date']} | Cloud Cover: {best['cloud_cover']:.2f}%")

        for cand in candidates[:max_scenes_per_window]:
            meta = acq.download_scene(cand["item_obj"], config.project.aoi_bbox_wgs84)
            downloaded_scenes.append(meta)

    print("\n" + "=" * 70)
    print(f"✅ ACQUISITION COMPLETE: {len(downloaded_scenes)} scene(s) verified in data/raw/")
    for s in downloaded_scenes:
        print(f"  📁 {s.scene_id} | Date: {s.acquisition_date} | Cloud: {s.cloud_cover:.1f}% | Size: {s.width}x{s.height} px | Hash: {s.file_hash[:12]}...")
    print("=" * 70)


if __name__ == "__main__":
    run_acquisition()
