from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("PREPROCESSING SSS")
print("=" * 60)

sss_folder = config.RAW_DATA / "SSS"

# --------------------------------------------------
# Collect files
# --------------------------------------------------

files = sorted(sss_folder.glob("*.nc"))

print("Total files:", len(files))

monthly_files = defaultdict(list)

for file in files:

    date = file.name.split("-")[-2][:6]
    year = int(date[:4])

    if config.START_YEAR <= year <= config.END_YEAR:
        monthly_files[date].append(file)

print("Months found:", len(monthly_files))

datasets = []

# --------------------------------------------------
# Process month by month
# --------------------------------------------------

for month in sorted(monthly_files):

    print(f"\nProcessing {month}")

    month_ds = []

    for file in sorted(monthly_files[month]):

        try:

            ds = xr.open_dataset(file)

            ds = ds[["sss"]]

            # Convert longitude to 0-360
            ds = ds.assign_coords(
                lon=((ds.lon + 360) % 360)
            ).sortby("lon")

            # Crop
            ds = ds.sel(
                lat=slice(config.LAT_MIN, config.LAT_MAX),
                lon=slice(config.LON_MIN, config.LON_MAX)
            )

            month_ds.append(ds)

        except Exception as e:
            print(f"Skipping {file.name}")
            print(e)

    if len(month_ds) == 0:
        print("No valid files for this month.")
        continue

    elif len(month_ds) == 1:
        print("Using single observation.")
        ds = month_ds[0].squeeze()

    else:
        ds = xr.concat(month_ds, dim="time").mean(
            dim="time",
            keep_attrs=True
        )

    # Proper timestamp
    year = int(month[:4])
    mon = int(month[4:6])

    timestamp = np.datetime64(f"{year:04d}-{mon:02d}-15")

    ds = ds.expand_dims(dim={"time": [timestamp]})

    datasets.append(ds)

    for d in month_ds:
        d.close()

# --------------------------------------------------
# Combine all months
# --------------------------------------------------

print("\nCombining all months...")

sss = xr.concat(datasets, dim="time")

# --------------------------------------------------
# Regrid
# --------------------------------------------------

lat_min = int(np.ceil(float(sss.lat.min())))
lat_max = int(np.floor(float(sss.lat.max())))

lon_min = int(np.ceil(float(sss.lon.min())))
lon_max = int(np.floor(float(sss.lon.max())))

new_lat = np.arange(lat_min, lat_max + 1, 1.0)
new_lon = np.arange(lon_min, lon_max + 1, 1.0)

print("Regridding...")

sss = sss.interp(
    lat=new_lat,
    lon=new_lon,
    method="linear"
)

# Remove fully-empty rows/columns
sss = sss.dropna(dim="lat", how="all")
sss = sss.dropna(dim="lon", how="all")

# --------------------------------------------------
# Save
# --------------------------------------------------

config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

output = config.PROCESSED_DATA / "sss.nc"

sss.to_netcdf(output)

print("\nSaved to:")
print(output)

print("\nDataset summary:")
print(sss)

print("\nNaN count:", int(sss.sss.isnull().sum()))