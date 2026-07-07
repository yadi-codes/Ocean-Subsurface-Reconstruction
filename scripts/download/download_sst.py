from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# =====================================================
# NOAA OISST Downloader
# Repairs missing folders and corrupted files
# =====================================================

BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr"
)

SAVE_DIR = Path("data/raw/SST")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()

# -----------------------------------------------------
# Months to repair
# -----------------------------------------------------

MONTHS_TO_DOWNLOAD = [
    "201111",
    "201401",
    "201907",
]

# -----------------------------------------------------
# Corrupted files discovered during preprocessing
# -----------------------------------------------------

BAD_FILES = {
    "201004": [
        "oisst-avhrr-v02r01.20100419.nc",
        "oisst-avhrr-v02r01.20100421.nc",
        "oisst-avhrr-v02r01.20100422.nc",
    ],
    "201008": [
        "oisst-avhrr-v02r01.20100801.nc",
    ],
    "201412": [
        "oisst-avhrr-v02r01.20141222.nc",
    ],
    "201810": [
        "oisst-avhrr-v02r01.20181010.nc",
    ],
}

# Add months containing corrupted files
for month in BAD_FILES.keys():
    if month not in MONTHS_TO_DOWNLOAD:
        MONTHS_TO_DOWNLOAD.append(month)

MONTHS_TO_DOWNLOAD.sort()

print("=" * 60)
print("NOAA OISST REPAIR DOWNLOADER")
print("=" * 60)

for folder in MONTHS_TO_DOWNLOAD:

    print(f"\nProcessing {folder}")

    url = f"{BASE_URL}/{folder}/"

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print("Could not access folder.")
        print(e)
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    nc_files = []

    for link in soup.find_all("a"):
        href = link.get("href")
        if href and href.endswith(".nc"):
            nc_files.append(href)

    if not nc_files:
        print("No NetCDF files found.")
        continue

    month_dir = SAVE_DIR / folder
    month_dir.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(nc_files, desc=folder):

        save_path = month_dir / filename

        # ---------------------------------------------
        # Existing good file
        # ---------------------------------------------

        if (
            save_path.exists()
            and folder not in BAD_FILES
        ):
            continue

        if (
            folder in BAD_FILES
            and filename not in BAD_FILES[folder]
            and save_path.exists()
        ):
            continue

        # ---------------------------------------------
        # Delete corrupted file
        # ---------------------------------------------

        if save_path.exists():
            print(f"\nReplacing corrupted file: {filename}")
            save_path.unlink()

        download_url = f"{url}{filename}"

        try:

            with session.get(download_url, stream=True, timeout=60) as r:

                r.raise_for_status()

                with open(save_path, "wb") as f:

                    for chunk in r.iter_content(chunk_size=8192):

                        if chunk:
                            f.write(chunk)

        except Exception as e:

            print(f"\nFailed: {filename}")
            print(e)

print("\n" + "=" * 60)
print("Repair completed!")
print("=" * 60)