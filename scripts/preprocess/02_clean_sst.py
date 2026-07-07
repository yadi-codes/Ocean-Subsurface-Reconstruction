from pathlib import Path
import xarray as xr

input_file = Path("data/processed/sst_monthly.nc")
output_file = Path("data/processed/sst.nc")

ds = xr.open_dataset(input_file)

# Keep only SST
ds = ds[["sst"]]

# Remove zlev if it exists
if "zlev" in ds.dims:
    ds = ds.squeeze("zlev", drop=True)

ds.to_netcdf(output_file)

print(ds)
print("\nSaved:", output_file)