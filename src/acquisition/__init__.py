"""
Satellite data acquisition module.
"""

from .sentinel2 import Sentinel2Acquisition, SceneMetadata

__all__ = ["Sentinel2Acquisition", "SceneMetadata"]
