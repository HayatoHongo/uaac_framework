import argparse
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from shapely.geometry import shape, Point
from rasterio.features import shapes
import rasterio
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max


def local_maxima_points(mask, min_distance=10, threshold_rel=0.2):
    dist = distance_transform_edt(mask)

    coords = peak_local_max(
        dist,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False,
        labels=mask,
    )

    return coords


def boundary_distant_points(array, min_area, min_distance, nodata=None):
    results = []
    cluster_ids = np.unique(array)
    if nodata is not None:
        cluster_ids = cluster_ids[cluster_ids != nodata]

    for cid in cluster_ids:
        mask = array == cid
        if not np.any(mask):
            continue

        labeled, n = ndimage.label(mask)
        regions_all = []

        for i in range(1, n + 1):
            comp = labeled == i
            for geom, val in shapes(comp.astype(np.uint8), mask=comp):
                if val != 1:
                    continue

                poly = shape(geom)
                if not poly.is_empty:
                    regions_all.append(
                        {"id": i, "poly": poly, "area": poly.area, "mask": comp}
                    )

        if not regions_all:
            continue
        regions_all.sort(key=lambda r: r["area"], reverse=True)
        filtered_regions = [r for r in regions_all if r["area"] >= min_area]

        if not filtered_regions:
            regions = [regions_all[0]]
        else:
            regions = filtered_regions

        for region in regions:
            comp = region["mask"]

            coords = local_maxima_points(
                comp, min_distance=min_distance, threshold_rel=0.2
            )

            for r, c in coords:
                x, y = (c, r)
                result = {
                    "cluster": int(cid),
                    "blob": int(i),
                    "x": x,
                    "y": y,
                }
                print(result)
                results.append(result)

    return results


def preview_points(array, points):
    unique_vals = np.unique(array)
    n_classes = len(unique_vals)

    base_cmap = plt.colormaps.get_cmap("prism")
    colors = base_cmap(np.arange(n_classes))
    cmap = ListedColormap(colors)

    bounds = np.concatenate([unique_vals.astype(float), [float(unique_vals[-1]) + 1.0]])
    norm = BoundaryNorm(bounds, cmap.N)

    _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(array, cmap=cmap, norm=norm, interpolation="none")
    ax.set_title("Clusters with sample points")
    ax.set_axis_off()

    for p in points:
        col, row = (p["x"], p["y"])
        ax.plot(col, row, "ro", markersize=6, markeredgecolor="k")
        ax.text(
            col + 2, row, str(p["cluster"]), color="white", fontsize=16, weight="bold"
        )

    plt.show()


def save_points_to_shp(points, crs, output_path):
    geometries = []
    for p in points:
        geo_x, geo_y = transform * (p["x"], p["y"])
        geometries.append(Point(geo_x, geo_y))

    gdf = gpd.GeoDataFrame(points, geometry=geometries, crs=crs)

    gdf["Id"] = range(0, len(gdf))
    gdf = gdf.rename(
        columns={
            "cluster": "RASTERVALU",
            "blob": "region_id",
        }
    )

    cols = ["Id", "RASTERVALU", "region_id", "geometry"]
    gdf = gdf[cols]

    gdf.to_file(output_path, driver="ESRI Shapefile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_raster_path")
    parser.add_argument("-output_shp_path")
    parser.add_argument("--min_area_ratio", type=float, default=0.001)
    parser.add_argument("--min_distance", type=int, default=300)

    args = parser.parse_args()

    input_raster_path = args.input_raster_path
    output_shp_path = args.output_shp_path
    min_area_ratio = args.min_area_ratio
    min_distance = args.min_distance

    with rasterio.open(input_raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata
        crs = src.crs

    if nodata is not None:
        valid_mask = arr != nodata
    else:
        valid_mask = ~np.isnan(arr)

    valid_pixels = np.count_nonzero(valid_mask)

    points = boundary_distant_points(
        arr, valid_pixels * min_area_ratio, min_distance, nodata
    )
    print(f"{str(len(points))} points")
    preview_points(arr, points)
    save_points_to_shp(points, crs, output_shp_path)
