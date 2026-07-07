import xarray as xr

files = [
    "sst.nc",
    "sss.nc",
    "ssh.nc",
    "ssw.nc",
    "argo.nc"
]

for f in files:

    print("="*60)
    print(f)

    ds = xr.open_dataset("data/processed/" + f)

    print(ds.sizes)

    print("\nLatitude:")
    print(ds.lat.values[:5], "...", ds.lat.values[-5:])

    print("\nLongitude:")
    print(ds.lon.values[:5], "...", ds.lon.values[-5:])

    print("\nTime:")
    print(ds.time.values[:3], "...", ds.time.values[-3:])

    ds.close()