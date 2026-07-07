from pathlib import Path
import sys

import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))
import utils.config as config

print("=" * 60)
print("ALIGNING SST")
print("=" * 60)

# ----------------------------
# Open SST
# ----------------------------

sst = xr.open_dataset(config.PROCESSED_DATA / "sst.nc")

# ----------------------------
# Target grid
# ----------------------------

lat = list(range(-29, 30))
lon = list(range(161, 240))

print("Interpolating...")

sst = sst.interp(
    lat=lat,
    lon=lon,
    method="linear"
)

output = config.PROCESSED_DATA / "sst_aligned.nc"

sst.to_netcdf(output)

print("\nSaved:", output)

print(sst)