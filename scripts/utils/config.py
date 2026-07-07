from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# ==================================================
# STUDY REGION (Paper)
# ==================================================

LAT_MIN = -30
LAT_MAX = 30

LON_MIN = 160
LON_MAX = 240

# ==================================================
# TRAINING YEARS
# ==================================================

START_YEAR = 2010
END_YEAR = 2020

# ==================================================
# DATASETS
# ==================================================

DATASETS = {

    "sst": {
        "folder": RAW_DATA / "SST",
        "lat": "lat",
        "lon": "lon",
        "time": "time"
    },

    "ssh": {
        "folder": RAW_DATA / "SSH",
        "lat": "latitude",
        "lon": "longitude",
        "time": "time"
    },

    "sss": {
        "folder": RAW_DATA / "SSS",
        "lat": "lat",
        "lon": "lon",
        "time": "time"
    },

    "ssw": {
        "folder": RAW_DATA / "SSW",
        "lat": "latitude",
        "lon": "longitude",
        "time": "time"
    },

    "argo": {
        "folder": RAW_DATA / "ARGO",
        "lat": "lat",
        "lon": "lon",
        "depth": "pres",
        "time": "time"
    }
}