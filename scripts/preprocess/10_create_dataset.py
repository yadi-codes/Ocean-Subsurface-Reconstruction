from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("CREATING TRAINING DATASET")
print("=" * 60)

# -------------------------------------------------------
# Load normalized datasets
# -------------------------------------------------------

norm = config.NORMALIZED_DATA

sst = xr.open_dataset(norm / "sst_aligned.nc")
sss = xr.open_dataset(norm / "sss.nc")
ssh = xr.open_dataset(norm / "ssh.nc")
ssw = xr.open_dataset(norm / "ssw.nc")
argo = xr.open_dataset(norm / "argo_aligned.nc")

# -------------------------------------------------------
# Build input tensor
# Shape -> (time, 5, lat, lon)
# -------------------------------------------------------

print("\nCreating input tensor...")

X = np.stack(
    [
        sst["sst"].values,
        sss["sss"].values,
        ssh["sla"].values,
        ssw["u"].values,
        ssw["v"].values,
    ],
    axis=1,
)

print("Input shape:", X.shape)

# -------------------------------------------------------
# Build target tensor
# Shape -> (time, 2, depth, lat, lon)
# -------------------------------------------------------

print("\nCreating target tensor...")

Y = np.stack(
    [
        argo["temp"].values,
        argo["salt"].values,
    ],
    axis=1,
)

print("Target shape:", Y.shape)

# -------------------------------------------------------
# Sliding windows
# -------------------------------------------------------

T = config.SEQUENCE_LENGTH

print(f"\nSequence Length = {T}")

X_windows = []
Y_windows = []

for i in range(len(X) - T + 1):

    # T months of surface data
    X_windows.append(X[i:i + T])

    # Predict final month
    Y_windows.append(Y[i + T - 1])

X_windows = np.array(X_windows, dtype=np.float32)
Y_windows = np.array(Y_windows, dtype=np.float32)

print("\nWindowed shapes")

print("X:", X_windows.shape)
print("Y:", Y_windows.shape)

# -------------------------------------------------------
# Split
# -------------------------------------------------------

times = sst.time.values

sample_times = times[T - 1:]

years = np.array(
    [int(str(t)[:4]) for t in sample_times]
)

train_mask = years <= config.TRAIN_END_YEAR
val_mask = (years > config.TRAIN_END_YEAR) & (
    years <= config.VAL_END_YEAR
)
test_mask = years > config.VAL_END_YEAR

train_X = X_windows[train_mask]
train_Y = Y_windows[train_mask]

val_X = X_windows[val_mask]
val_Y = Y_windows[val_mask]

test_X = X_windows[test_mask]
test_Y = Y_windows[test_mask]

print("\n===================================")
print("Dataset split")
print("===================================")

print("Train :", train_X.shape[0])
print("Val   :", val_X.shape[0])
print("Test  :", test_X.shape[0])

# -------------------------------------------------------
# Save
# -------------------------------------------------------

config.DATASET_DIR.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    config.DATASET_DIR / "train.npz",
    X=train_X,
    Y=train_Y,
)

np.savez_compressed(
    config.DATASET_DIR / "val.npz",
    X=val_X,
    Y=val_Y,
)

np.savez_compressed(
    config.DATASET_DIR / "test.npz",
    X=test_X,
    Y=test_Y,
)

print("\nSaved datasets to:")
print(config.DATASET_DIR)

# -------------------------------------------------------
# Final summary
# -------------------------------------------------------

print("\n===================================")
print("SUMMARY")
print("===================================")

print("Train X:", train_X.shape)
print("Train Y:", train_Y.shape)

print("Val X:", val_X.shape)
print("Val Y:", val_Y.shape)

print("Test X:", test_X.shape)
print("Test Y:", test_Y.shape)

print("\nDataset creation complete.")