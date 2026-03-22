import argparse
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
import os
import numpy as np
import imageio.v3 as iio


def sample_images(
    input_raster,
    points_shp,
    output_folder,
    buffer_size,
    output_tif,
    output_jpeg,
    skip_all_no_data,
    skip_some_no_data,
):
    output_folder_tif = os.path.join(output_folder, "tif")
    output_folder_jpeg = os.path.join(output_folder, "jpeg")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_tif, exist_ok=True)
    os.makedirs(output_folder_jpeg, exist_ok=True)

    gdf = gpd.read_file(points_shp)
    with rasterio.open(input_raster) as src:
        raster_crs = src.crs

        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for idx, row in gdf.iterrows():
            point = row.geometry
            x, y = point.x, point.y

            if not (
                src.bounds.left <= x <= src.bounds.right
                and src.bounds.bottom <= y <= src.bounds.top
            ):
                continue

            minx, maxx = x - buffer_size / 2, x + buffer_size / 2
            miny, maxy = y - buffer_size / 2, y + buffer_size / 2
            window = from_bounds(minx, miny, maxx, maxy, src.transform)

            data = src.read(window=window)
            transform = src.window_transform(window)

            no_data_mask = (
                (data[0] == 255) & (data[1] == 255) & (data[2] == 255) & (data[3] == 0)
            )

            if skip_all_no_data and np.all(no_data_mask):
                print(f"Tile {idx} is all NoData, skipping")
                continue

            if skip_some_no_data and np.any(no_data_mask):
                print(f"Tile {idx} includes NoData, skipping")
                continue

            if output_tif:
                output_path_tif = os.path.join(output_folder_tif, f"tile_{idx}.tif")

                profile = src.profile.copy()
                profile.update(
                    {
                        "height": data.shape[1],
                        "width": data.shape[2],
                        "transform": transform,
                        "driver": "GTiff",
                        "compress": "lzw",
                    }
                )
                with rasterio.open(output_path_tif, "w", **profile) as dst:
                    dst.write(data)
                print(f"Saved tif: {output_path_tif}")

            if output_jpeg:
                jpg_out = np.transpose(data[:3], (1, 2, 0))
                jpg_out = np.clip(jpg_out, 0, 255).astype(np.uint8)

                output_path_jpeg = os.path.join(output_folder_jpeg, f"tile_{idx}.jpg")
                iio.imwrite(output_path_jpeg, jpg_out)
                print(f"Saved JPEG: {output_path_jpeg}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_raster")
    parser.add_argument("--points_shp")
    parser.add_argument("--output_folder")
    parser.add_argument("-buffer_size", type=float, default=1.0)
    parser.add_argument("-output_tif", type=int, default=1)
    parser.add_argument("-output_jpeg", type=int, default=1)
    parser.add_argument("-skip_all_no_data", type=int, default=1)
    parser.add_argument("-skip_some_no_data", type=int, default=1)

    args = parser.parse_args()

    sample_images(
        args.input_raster,
        args.points_shp,
        args.output_folder,
        args.buffer_size,
        args.output_tif,
        args.output_jpeg,
        args.skip_all_no_data,
        args.skip_some_no_data,
    )
