from pathlib import Path
import xarray as xr

path=Path("data/raw/SSS")

datasets=[]

for file in sorted(path.glob("*.nc")):

    ds=xr.open_dataset(file)

    ds=ds.sel(
        lat=slice(-30,30),
        lon=slice(160,240)
    )

    datasets.append(ds)

merged=xr.concat(datasets,dim="time")

merged.to_netcdf("data/processed/sss.nc")

print(merged)