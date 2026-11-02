"""photomosaic — an independent Python implementation of classic photo-mosaic
rendering: 3x3 sub-block LAB feature matching with near-best random selection.

Clean-room implementation of well-known photomosaic techniques.
"""

__version__ = "1.0.0"

from .features import subblock_lab, average_rgb, crop_to_ratio, FEATURE_DIM
from .db import TileDB
from .library import ingest, VARIANTS
from .match import Matcher
from .render import make_mosaic, master_cell_features

__all__ = [
    "__version__",
    "subblock_lab",
    "average_rgb",
    "crop_to_ratio",
    "FEATURE_DIM",
    "TileDB",
    "ingest",
    "VARIANTS",
    "Matcher",
    "make_mosaic",
    "master_cell_features",
]
