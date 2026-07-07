from pathlib import Path
import sys
import json

import numpy as np
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils.config as config

print("=" * 60)
print("NORMALIZING DATASETS")
print("=" * 60)

# ----------------------------------------------------
# Files to normalize
# ----------------------------------------------------

datasets = {
    "sst": ("sst_aligned.nc", ["sst"]),
    "ssh": ("ssh.nc", ["sla"]),
    "sss": ("sss.nc", ["sss"]),
    "ssw": ("ssw.nc", ["u", "v"]),
    "argo": ("argo_aligned.nc", ["temp", "salt"]),
}

stats = {}

output_dir = config.PROCESSED_DATA / "normalized"
output_dir.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------
# Normalize
# ----------------------------------------------------

for name, (filename, variables) in datasets.items():

    print(f"\n{name.upper()}")

    ds = xr.open_dataset(config.PROCESSED_DATA / filename)

    for var in variables:

        print(f"Normalizing {var}")

        minimum = float(ds[var].min(skipna=True))
        maximum = float(ds[var].max(skipna=True))

        ds[var] = (ds[var] - minimum) / (maximum - minimum)

        stats[var] = {
            "min": minimum,
            "max": maximum
        }

        print(f"Min: {minimum:.5f}")
        print(f"Max: {maximum:.5f}")

    output_file = output_dir / filename

    ds.to_netcdf(output_file)

    print("Saved:", output_file.name)

    ds.close()

# ----------------------------------------------------
# Save normalization statistics
# ----------------------------------------------------

stats_file = output_dir / "normalization.json"

with open(stats_file, "w") as f:
    json.dump(stats, f, indent=4)

print("\nSaved normalization statistics:")
print(stats_file)

print("\nDONE!")