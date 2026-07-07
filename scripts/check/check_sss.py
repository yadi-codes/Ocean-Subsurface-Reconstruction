from pathlib import Path
import xarray as xr

folder = Path("data/raw/SSS")

bad = []

for file in sorted(folder.glob("*.nc")):

    try:
        ds = xr.open_dataset(file)
        ds.close()

    except Exception:
        print("BAD:", file.name)
        bad.append(file)

print("\nTotal bad files:", len(bad))

for f in bad:
    print(f)