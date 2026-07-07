from pathlib import Path

# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
NORMALIZED_DATA = PROCESSED_DATA / "normalized"
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"

MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ==================================================
# STUDY REGION (Paper)
# ==================================================

START_YEAR = 2010
END_YEAR = 2020

LAT_MIN = -29
LAT_MAX = 29

LON_MIN = 161
LON_MAX = 239

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

# ==================================================
# DATASET CREATION
# ==================================================

SEQUENCE_LENGTH = 3

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 2

TRAIN_END_YEAR = 2017
VAL_END_YEAR = 2019

# ==================================================
# CONVFORMER
# ==================================================

PATCH_SIZE = 4

EMBED_DIM = 128

NUM_HEADS = 8

NUM_LAYERS = 6

CONVLSTM_HIDDEN = 64

MLP_RATIO = 4

DROPOUT = 0.1

# ==================================================
# TRAINING
# ==================================================

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EPOCHS = 100

NUM_WORKERS = 4

PIN_MEMORY = True

# ==================================================
# RANDOMNESS
# ==================================================

RANDOM_SEED = 42
# ==================================================
# EXPERIMENT TRACKING
# ==================================================

EXPERIMENT_NAME = "baseline"

RANDOM_SEED = 42

# ==================================================
# DATASET
# ==================================================

SEQUENCE_LENGTH = 3

INPUT_CHANNELS = 5
OUTPUT_CHANNELS = 2

TRAIN_END_YEAR = 2017
VAL_END_YEAR = 2019

# ==================================================
# CONVFORMER
# ==================================================

PATCH_SIZE = 4

EMBED_DIM = 128

NUM_HEADS = 8

NUM_LAYERS = 6

CONVLSTM_HIDDEN = 64

MLP_RATIO = 4

DROPOUT = 0.1

# ==================================================
# TRAINING
# ==================================================

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

EPOCHS = 100

NUM_WORKERS = 4

PIN_MEMORY = True


def get_training_config():
    """
    Returns every tunable hyperparameter.
    This will be automatically saved for every experiment.
    """

    return {

        "experiment_name": EXPERIMENT_NAME,

        "sequence_length": SEQUENCE_LENGTH,

        "patch_size": PATCH_SIZE,

        "embedding_dimension": EMBED_DIM,

        "num_heads": NUM_HEADS,

        "num_layers": NUM_LAYERS,

        "convlstm_hidden": CONVLSTM_HIDDEN,

        "mlp_ratio": MLP_RATIO,

        "dropout": DROPOUT,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "epochs": EPOCHS,

        "random_seed": RANDOM_SEED
    }