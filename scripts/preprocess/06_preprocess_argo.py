from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("PREPROCESSING ARGO")
print("=" * 60)

# --------------------------------------------------
# Locate files
# --------------------------------------------------

argo_folder = config.RAW_DATA / "ARGO"

files = sorted(argo_folder.glob("*.nc"))

print("Files found:", len(files))

datasets = []

# --------------------------------------------------
# Process monthly files
# --------------------------------------------------

for file in files:

    stem = file.stem
    parts = stem.split("_")

    # Skip annual or unexpected files
    if len(parts) != 4:
        print("Skipping:", file.name)
        continue

    try:
        year = int(parts[2])
        month = int(parts[3])
    except ValueError:
        print("Skipping:", file.name)
        continue

    # Keep only 2010–2020
    if year < config.START_YEAR or year > config.END_YEAR:
        continue

    print(f"Processing {year}-{month:02d}")

    try:

        ds = xr.open_dataset(file, decode_times=False)

        # Keep only target variables
        ds = ds[["temp", "salt"]]

        # Crop to study region
        ds = ds.sel(
            lat=slice(config.LAT_MIN, config.LAT_MAX),
            lon=slice(config.LON_MIN, config.LON_MAX)
        )

        # Replace invalid BOA time with correct timestamp
        timestamp = np.datetime64(f"{year:04d}-{month:02d}-15")
        ds = ds.assign_coords(time=[timestamp])

        datasets.append(ds)

        ds.close()

    except Exception as e:
        print("Failed:", file.name)
        print(e)

print("\nLoaded months:", len(datasets))

if len(datasets) == 0:
    raise RuntimeError("No ARGO datasets were loaded.")

# --------------------------------------------------
# Combine
# --------------------------------------------------

print("\nCombining datasets...")

argo = xr.concat(datasets, dim="time")

# --------------------------------------------------
# Save
# --------------------------------------------------

config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

output = config.PROCESSED_DATA / "argo.nc"

argo.to_netcdf(output)

print("\nSaved to:")
print(output)

print("\nDataset summary:")
print(argo)

print("\nTemperature NaNs:", int(argo.temp.isnull().sum()))
print("Salinity NaNs:", int(argo.salt.isnull().sum()))