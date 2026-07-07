from pathlib import Path
import xarray as xr
import pandas as pd

file = Path("data/processed/sst_monthly.nc")

ds = xr.open_dataset(file)

print("Time dimension:", ds.sizes["time"])

print("\nAll months in file:\n")

months = pd.to_datetime(ds.time.values).strftime("%Y-%m")

for m in months:
    print(m)