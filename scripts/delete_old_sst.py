from pathlib import Path
import shutil

sst_folder = Path("data/raw/SST")

for folder in sst_folder.iterdir():
    if folder.is_dir():
        year = int(folder.name[:4])

        if year < 2010:
            print("Deleting", folder)
            shutil.rmtree(folder)

print("Done!")