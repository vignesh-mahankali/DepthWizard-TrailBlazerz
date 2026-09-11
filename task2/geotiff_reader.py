import rasterio


def open_geotiff(tif_path):
    """
    Open a GeoTIFF file and return the dataset.
    """
    return rasterio.open(tif_path)


def get_metadata(tif_path):
    """
    Extract important metadata from a GeoTIFF.
    """

    with rasterio.open(tif_path) as dataset:

        metadata = {
            "crs": str(dataset.crs),
            "width": dataset.width,
            "height": dataset.height,
            "bands": dataset.count,
            "dtype": str(dataset.dtypes[0]),
            "bounds": {
                "left": dataset.bounds.left,
                "bottom": dataset.bounds.bottom,
                "right": dataset.bounds.right,
                "top": dataset.bounds.top
            },
            "transform": str(dataset.transform)
        }

        return metadata


def read_raster_data(tif_path):
    """
    Read all raster bands from the GeoTIFF.
    """

    with rasterio.open(tif_path) as dataset:

        data = dataset.read()

        return data


def read_band(tif_path, band_number):
    """
    Read a specific band from the GeoTIFF.
    """

    with rasterio.open(tif_path) as dataset:

        band = dataset.read(band_number)

        return band


def main():

    tif_path = input("Enter GeoTIFF path: ").strip()

    try:

        metadata = get_metadata(tif_path)

        print("\n===== GEOTIFF METADATA =====")

        for key, value in metadata.items():
            print(f"{key}: {value}")

        data = read_raster_data(tif_path)

        print("\n===== RASTER DATA =====")
        print("Raster shape:", data.shape)

    except FileNotFoundError:
        print("ERROR: GeoTIFF file not found.")

    except rasterio.errors.RasterioIOError:
        print("ERROR: Unable to open GeoTIFF.")

    except Exception as e:
        print("ERROR:", e)


if __name__ == "__main__":
    main()