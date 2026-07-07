from pathlib import Path
import sys

import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils.config as config

print("=" * 60)
print("ALIGNING ARGO")
print("=" * 60)

argo = xr.open_dataset(config.PROCESSED_DATA / "argo.nc")

lat = list(range(-29, 30))
lon = list(range(161, 240))

print("Interpolating...")

argo = argo.interp(
    lat=lat,
    lon=lon,
    method="linear"
)

output = config.PROCESSED_DATA / "argo_aligned.nc"

argo.to_netcdf(output)

print("\nSaved:", output)

print(argo)