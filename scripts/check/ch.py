import xarray as xr

files = [
    "sst_aligned.nc",
    "ssh.nc",
    "sss.nc",
    "ssw.nc",
    "argo_aligned.nc"
]

for f in files:

    ds = xr.open_dataset("data/processed/normalized/" + f)

    print("=" * 50)
    print(f)

    for var in ds.data_vars:
        print(
            var,
            float(ds[var].min(skipna=True)),
            float(ds[var].max(skipna=True))
        )