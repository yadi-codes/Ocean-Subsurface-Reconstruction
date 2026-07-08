import xarray as xr


def crop_dataset(
    ds,
    lat_name,
    lon_name,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
):
    """
    Crop a dataset to the study region.
    """

    return ds.sel(
        {
            lat_name: slice(lat_min, lat_max),
            lon_name: slice(lon_min, lon_max),
        }
    )


def save_dataset(ds, output_path):
    """
    Save dataset to NetCDF.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_netcdf(output_path)

    print(f"Saved -> {output_path}")