from pathlib import Path
import xarray as xr

ssw_folder = Path("data/raw/SSW")

# Find the first NetCDF file
file = next(ssw_folder.glob("*.nc"))

print(f"Opening: {file.name}\n")

ds = xr.open_dataset(file)

print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))