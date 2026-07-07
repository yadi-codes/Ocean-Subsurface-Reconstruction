from pathlib import Path
import xarray as xr

file = next(Path("data/raw/SSS").glob("*.nc"))

print("Opening:", file.name)

ds = xr.open_dataset(file)

print(ds)
print("\nVariables:", list(ds.data_vars))
print("\nCoordinates:", list(ds.coords))