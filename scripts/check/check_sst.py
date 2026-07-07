from pathlib import Path
import xarray as xr

sst_path = Path("data/raw/SST")

bad = []

for folder in sorted(sst_path.iterdir()):

    if not folder.is_dir():
        continue

    for file in sorted(folder.glob("*.nc")):

        try:
            xr.open_dataset(file).close()

        except Exception as e:
            print("BAD:", file.name)
            bad.append(file)

print("\nTotal bad files:", len(bad))

for f in bad:
    print(f)