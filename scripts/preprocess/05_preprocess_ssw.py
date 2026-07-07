from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("PREPROCESSING SSW")
print("=" * 60)

ssw_folder = config.RAW_DATA / "SSW"

files = sorted(ssw_folder.glob("*.nc"))

print("Files found:", len(files))

datasets = []

for file in files:

    print(file.name)

    try:

        ds = xr.open_dataset(file)

        # -----------------------------------------
        # Keep only U and V
        # -----------------------------------------

        ds = ds[["u", "v"]]

        # -----------------------------------------
        # Convert longitude to 0-360
        # -----------------------------------------

        ds = ds.assign_coords(
            longitude=((ds.longitude + 360) % 360)
        ).sortby("longitude")

        # -----------------------------------------
        # Rename coordinates
        # -----------------------------------------

        ds = ds.rename({
            "latitude": "lat",
            "longitude": "lon"
        })

        # -----------------------------------------
        # Crop
        # -----------------------------------------

        ds = ds.sel(
            lat=slice(config.LAT_MIN, config.LAT_MAX),
            lon=slice(config.LON_MIN, config.LON_MAX)
        )

        datasets.append(ds)

    except Exception as e:

        print("Skipped:", file.name)
        print(e)

print("\nLoaded months:", len(datasets))

# ---------------------------------------------------
# Combine
# ---------------------------------------------------

ssw = xr.concat(datasets, dim="time")

# ---------------------------------------------------
# Build common 1° grid
# ---------------------------------------------------

lat_min = int(np.ceil(float(ssw.lat.min())))
lat_max = int(np.floor(float(ssw.lat.max())))

lon_min = int(np.ceil(float(ssw.lon.min())))
lon_max = int(np.floor(float(ssw.lon.max())))

new_lat = np.arange(lat_min, lat_max + 1, 1.0)
new_lon = np.arange(lon_min, lon_max + 1, 1.0)

print("\nRegridding...")

ssw = ssw.interp(
    lat=new_lat,
    lon=new_lon,
    method="linear"
)

ssw = ssw.dropna(dim="lat", how="all")
ssw = ssw.dropna(dim="lon", how="all")

# ---------------------------------------------------
# Save
# ---------------------------------------------------

output = config.PROCESSED_DATA / "ssw.nc"

ssw.to_netcdf(output)

print("\nSaved to:")
print(output)

print("\nDataset summary:\n")
print(ssw)

print("\nNaNs in U:", int(ssw.u.isnull().sum()))
print("NaNs in V:", int(ssw.v.isnull().sum()))