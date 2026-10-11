"""
Configuration module for the Satellite-based Illegal Mining Detection System.
Provides typed project settings, directory paths, and geospatial constants.
"""

from pathlib import Path
from typing import List, Tuple
from pydantic import BaseModel, Field

# Base Directory
BASE_DIR: Path = Path(__file__).resolve().parent.parent

class DirectoryConfig(BaseModel):
    """Core directory locations for ingestion, processing, models, and outputs."""
    root_dir: Path = BASE_DIR
    inbox_images: Path = BASE_DIR / "inbox" / "images"
    inbox_training: Path = BASE_DIR / "inbox" / "training_data"
    
    models_dir: Path = BASE_DIR / "models"
    models_checkpoints: Path = BASE_DIR / "models" / "checkpoints"
    models_current: Path = BASE_DIR / "models" / "current"
    
    data_dir: Path = BASE_DIR / "data"
    data_raw: Path = BASE_DIR / "data" / "raw"
    data_tiles: Path = BASE_DIR / "data" / "tiles"
    data_labels: Path = BASE_DIR / "data" / "labels"
    data_frozen_test: Path = BASE_DIR / "data" / "frozen_test"
    
    outputs_dir: Path = BASE_DIR / "outputs"
    outputs_masks: Path = BASE_DIR / "outputs" / "masks"
    outputs_geojson: Path = BASE_DIR / "outputs" / "geojson"
    outputs_reports: Path = BASE_DIR / "outputs" / "reports"
    
    processed_dir: Path = BASE_DIR / "processed"
    failed_dir: Path = BASE_DIR / "failed"
    
    db_path: Path = BASE_DIR / "outputs" / "pipeline_audit.db"

class SensorConfig(BaseModel):
    """Satellite band and resolution configurations."""
    # Sentinel-2 Bands to extract (6 raw bands + derived indices in pipeline)
    raw_bands: List[str] = Field(
        default=["B02", "B03", "B04", "B08", "B11", "B12"],
        description="Blue, Green, Red, NIR, SWIR1, SWIR2"
    )
    derived_indices: List[str] = Field(
        default=["NDVI", "BSI"],
        description="Normalized Difference Vegetation Index, Bare Soil Index"
    )
    target_resolution_m: float = 10.0
    tile_size_px: int = 256
    tile_overlap_px: int = 32

class ProjectConfig(BaseModel):
    """Regional, legal, and operational project settings."""
    district: str = Field(default="Keonjhar", description="Target administrative district")
    state: str = Field(default="Odisha", description="Target state")
    mineral: str = Field(default="Iron Ore", description="Primary mineral target")
    
    # EPSG:32645 corresponds to WGS 84 / UTM zone 45N (Covers Odisha, Jharkhand, West Bengal)
    target_crs: str = Field(default="EPSG:32645", description="Projected metric CRS for geometry & area calculations")
    
    # Geometric and operational thresholds
    min_mining_area_sqm: float = Field(
        default=1000.0,
        description="Minimum polygon area in square meters (~10 contiguous 10m pixels) to filter sensor noise"
    )
    buffer_distance_m: float = Field(
        default=50.0,
        description="Buffer distance in meters around lease boundaries for 'Needs Review' classification"
    )
    cloud_cover_max_percent: float = Field(
        default=15.0,
        description="Maximum scene cloud cover percentage allowed during acquisition"
    )
    random_seed: int = 42

class Config(BaseModel):
    """Unified configuration instance."""
    dirs: DirectoryConfig = DirectoryConfig()
    sensor: SensorConfig = SensorConfig()
    project: ProjectConfig = ProjectConfig()

    def ensure_directories(self) -> None:
        """Create all required project directories if they do not exist."""
        for path in [
            self.dirs.inbox_images,
            self.dirs.inbox_training,
            self.dirs.models_checkpoints,
            self.dirs.models_current,
            self.dirs.data_raw,
            self.dirs.data_tiles,
            self.dirs.data_labels,
            self.dirs.data_frozen_test,
            self.dirs.outputs_masks,
            self.dirs.outputs_geojson,
            self.dirs.outputs_reports,
            self.dirs.processed_dir,
            self.dirs.failed_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

# Default singleton instance
config = Config()
