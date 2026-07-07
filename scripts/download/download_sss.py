import requests
from pathlib import Path
from tqdm import tqdm

BASE = "https://dap.ceda.ac.uk/neodc/esacci/sea_surface_salinity/data/v05.5/GLOBALv5.5/30days"

SAVE_DIR = Path("data/raw/SSS")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()

for year in range(2010, 2021):

    print(f"\n===== {year} =====")

    for month in range(1, 13):

        # Monthly centred dates
        dates = [
            f"{year}{month:02d}01",
            f"{year}{month:02d}15"
        ]

        for date in dates:

            filename = (
                f"ESACCI-SEASURFACESALINITY-L4-SSS-"
                f"GLOBAL-MERGED_OI_Monthly_CENTRED_15Day_0.25deg-"
                f"{date}-fv5.5.nc"
            )

            url = f"{BASE}/{year}/{filename}?download=1"

            outfile = SAVE_DIR / filename

            if outfile.exists():
                continue

            try:

                r = session.get(url, stream=True, timeout=60)

                if r.status_code != 200:
                    continue

                total = int(r.headers.get("content-length", 0))

                with open(outfile, "wb") as f:

                    for chunk in tqdm(
                        r.iter_content(8192),
                        total=max(total // 8192, 1),
                        unit="chunks",
                        leave=False,
                        desc=filename[-20:]
                    ):
                        if chunk:
                            f.write(chunk)

            except Exception as e:

                print(f"Failed: {filename}")
                print(e)

print("\nDone!")