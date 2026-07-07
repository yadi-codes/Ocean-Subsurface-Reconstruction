import xarray as xr

file = "data/raw/SSH/your_file.nc"

ds = xr.open_dataset(file)

ds = ds.sel(
    latitude=slice(-30,30),
    longitude=slice(160,240)
)

ds.to_netcdf("data/processed/ssh.nc")

print(ds)