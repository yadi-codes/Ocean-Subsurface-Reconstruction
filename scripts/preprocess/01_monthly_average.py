from pathlib import Path
import sys

import pandas as pd
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("CREATING MONTHLY SST")
print("=" * 60)

BAD_FILES = {
    "oisst-avhrr-v02r01.20100419.nc",
    "oisst-avhrr-v02r01.20100421.nc",
    "oisst-avhrr-v02r01.20100422.nc",
    "oisst-avhrr-v02r01.20100801.nc",
    "oisst-avhrr-v02r01.20141222.nc",
    "oisst-avhrr-v02r01.20181010.nc",
}

raw_sst = config.RAW_DATA / "SST"

monthly_datasets = []

for month_folder in sorted(raw_sst.iterdir()):

    if not month_folder.is_dir():
        continue

    year = int(month_folder.name[:4])
    month = int(month_folder.name[4:])

    if not (config.START_YEAR <= year <= config.END_YEAR):
        continue

    print(f"\nProcessing {month_folder.name}")

    daily_datasets = []

    for file in sorted(month_folder.glob("*.nc")):

        if file.name in BAD_FILES:
            print("Skipping:", file.name)
            continue

        try:
            ds = xr.open_dataset(file)

            ds = ds.sel(
                lat=slice(config.LAT_MIN, config.LAT_MAX),
                lon=slice(config.LON_MIN, config.LON_MAX)
            )

            daily_datasets.append(ds)

        except Exception as e:
            print(f"Error opening {file.name}: {e}")

    if len(daily_datasets) == 0:
        print("No valid files found!")
        continue

    print(f"Loaded {len(daily_datasets)} daily files")

    # Combine all days
    month_ds = xr.concat(daily_datasets, dim="time")

    # Average over days
    month_ds = month_ds.mean(dim="time")

    # Remove unnecessary variables
    month_ds = month_ds[["sst"]]

    # Remove zlev dimension if present
    if "zlev" in month_ds.dims:
        month_ds = month_ds.squeeze("zlev", drop=True)

    # Assign correct monthly timestamp
    timestamp = pd.Timestamp(year=year, month=month, day=15)

    month_ds = month_ds.expand_dims(time=[timestamp])

    print(f"Added month: {timestamp.strftime('%Y-%m')}")

    monthly_datasets.append(month_ds)

    for ds in daily_datasets:
        ds.close()

print("\nCombining monthly datasets...")

sst_monthly = xr.concat(monthly_datasets, dim="time")

config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

output = config.PROCESSED_DATA / "sst_monthly.nc"

sst_monthly.to_netcdf(output)

print("\nSaved:", output)

print("\nFinal Dataset")
print(sst_monthly)

print("\nNumber of months:", sst_monthly.sizes["time"])