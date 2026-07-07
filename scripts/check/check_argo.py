from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config import *
import xarray as xr

file = RAW_DATA / "ARGO" / "BOA_Argo_2010_01.nc"

print("Opening:", file)

ds = xr.open_dataset(
    file,
    decode_times=False
)

print(ds)

print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))