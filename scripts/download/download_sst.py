import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"

START_YEAR = 2019
END_YEAR = 2020

SAVE_DIR = Path("data/raw/SST")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()

for year in range(START_YEAR, END_YEAR + 1):
    for month in range(1, 13):

        folder = f"{year}{month:02d}"
        url = f"{BASE_URL}/{folder}/"

        print(f"\n===== {folder} =====")

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Could not access {folder}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        files = []

        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith(".nc"):
                files.append(href)

        if not files:
            print("No NetCDF files found.")
            continue

        month_dir = SAVE_DIR / folder
        month_dir.mkdir(exist_ok=True)

        for file in tqdm(files, desc=folder):

            save_path = month_dir / file

            if save_path.exists():
                continue

            download_url = f"{url}{file}"

            try:
                with session.get(download_url, stream=True, timeout=60) as r:
                    r.raise_for_status()

                    with open(save_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

            except Exception as e:
                print(f"\nFailed: {file}")
                print(e)

print("\nAll downloads completed!")