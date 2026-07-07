from pathlib import Path
import re
from collections import Counter

folder = Path("data/raw/SSS")

files = sorted(folder.glob("*.nc"))

print("Total files:", len(files))

pattern = re.compile(r"(\d{8})")

months = []

for f in files:
    m = pattern.search(f.name)
    if m:
        months.append(m.group(1)[:6])

counts = Counter(months)

print("\nMonths that don't have exactly 2 files:\n")

for month in sorted(counts):
    if counts[month] != 2:
        print(month, "->", counts[month], "file(s)")