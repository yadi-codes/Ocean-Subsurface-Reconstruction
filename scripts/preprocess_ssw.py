from pathlib import Path
import xarray as xr

path = Path("data/raw/SSW")

datasets=[]

for file in sorted(path.glob("*.nc")):

    ds=xr.open_dataset(file)

    ds=ds.sel(
        latitude=slice(-30,30),
        longitude=slice(160,240)
    )

    datasets.append(ds)

merged=xr.concat(datasets,dim="time")

merged.to_netcdf("data/processed/ssw.nc")

print(merged)