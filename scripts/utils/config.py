from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data folders
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# Region used in the paper
LAT_MIN = -30
LAT_MAX = 30
LON_MIN = 160
LON_MAX = 240