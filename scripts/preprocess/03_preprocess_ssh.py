from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("PREPROCESSING SSH")
print("=" * 60)

# ---------------------------------------------------
# Locate SSH file
# ---------------------------------------------------

ssh_folder = config.RAW_DATA / "SSH"

files = list(ssh_folder.glob("*.nc"))

if len(files) != 1:
    raise Exception("Expected exactly one SSH file.")

ssh_file = files[0]

print("Opening:")
print(ssh_file.name)

# ---------------------------------------------------
# Open
# ---------------------------------------------------

ds = xr.open_dataset(ssh_file)

# Keep only SLA
ds = ds[["sla"]]

# ---------------------------------------------------
# Rename coordinates (only if needed)
# ---------------------------------------------------

rename_dict = {}

if "latitude" in ds.coords:
    rename_dict["latitude"] = "lat"

if "longitude" in ds.coords:
    rename_dict["longitude"] = "lon"

if rename_dict:
    ds = ds.rename(rename_dict)

# ---------------------------------------------------
# Build target grid
# ---------------------------------------------------

lat_min = int(np.ceil(float(ds.lat.min())))
lat_max = int(np.floor(float(ds.lat.max())))

lon_min = int(np.ceil(float(ds.lon.min())))
lon_max = int(np.floor(float(ds.lon.max())))

new_lat = np.arange(lat_min, lat_max + 1, 1.0)
new_lon = np.arange(lon_min, lon_max + 1, 1.0)

print(f"\nTarget grid:")
print(f"Latitude : {lat_min} to {lat_max}")
print(f"Longitude: {lon_min} to {lon_max}")

# ---------------------------------------------------
# Regrid
# ---------------------------------------------------

print("\nRegridding...")

ds = ds.interp(
    lat=new_lat,
    lon=new_lon,
    method="linear"
)

# ---------------------------------------------------
# Remove rows/columns that are entirely NaN
# ---------------------------------------------------

ds = ds.dropna(dim="lat", how="all")
ds = ds.dropna(dim="lon", how="all")

# ---------------------------------------------------
# Save
# ---------------------------------------------------

config.PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

output = config.PROCESSED_DATA / "ssh.nc"

ds.to_netcdf(output)

print("\nSaved to:")
print(output)

print("\nDataset summary:\n")
print(ds)

print("\nNaN count:", int(ds.sla.isnull().sum()))