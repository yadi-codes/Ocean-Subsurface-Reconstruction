from pathlib import Path
import xarray as xr

sst_path = Path("data/raw/SST")

datasets = []

folders = sorted(sst_path.iterdir())

for folder in folders:
    if not folder.is_dir():
        continue

    files = sorted(folder.glob("*.nc"))

    for file in files:

        ds = xr.open_dataset(file)

        ds = ds.sel(
            lat=slice(-30,30),
            lon=slice(160,240)
        )

        datasets.append(ds)

print("Loaded",len(datasets),"months")

merged = xr.concat(datasets,dim="time")

merged.to_netcdf("data/processed/sst.nc")

print(merged)