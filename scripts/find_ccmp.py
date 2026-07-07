import earthaccess

earthaccess.login()

results = earthaccess.search_data(
    short_name="CCMP_WINDS_10MMONTHLY_L4_V3.1",
    temporal=("2010-01-01", "2020-12-31")
)

print(f"Found {len(results)} monthly files.")

earthaccess.download(
    results,
    local_path="data/raw/SSW"
)