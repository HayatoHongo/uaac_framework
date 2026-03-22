import argparse

from matplotlib import pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage


def merge_by_center(labeled, n, threshold):
    sizes = ndimage.sum(1, labeled, range(n + 1))

    is_large = sizes >= threshold
    is_large[0] = True

    large_labeled = np.where(is_large[labeled], labeled, 0)

    _, indices = ndimage.distance_transform_edt(large_labeled == 0, return_indices=True)
    territory_map = large_labeled[tuple(indices)]

    weights = np.ones_like(labeled, dtype=np.float64)

    centers = ndimage.center_of_mass(weights, labeled, range(1, n + 1))

    map_array = np.arange(n + 1)

    for i, center in enumerate(centers):
        label_id = i + 1

        if np.any(np.isnan(center)):
            continue

        if not is_large[label_id]:
            cy, cx = map(int, center)

            try:
                map_array[label_id] = territory_map[cy, cx]
            except IndexError:
                pass

    return map_array[labeled]


def divide_clusters_spatial(array, merge_threshold, nodata=None):
    results = []
    cluster_ids = np.unique(array)
    if nodata is not None:
        cluster_ids = cluster_ids[cluster_ids != nodata]

    for cid in cluster_ids:
        mask = array == cid
        if not np.any(mask):
            continue

        labeled, n = ndimage.label(mask)
        new_labeled = merge_by_center(labeled, n, merge_threshold)

        unique_labels = np.unique(new_labeled)
        unique_labels = unique_labels[unique_labels != 0]

        for label_id in unique_labels:
            current_mask = new_labeled == label_id
            results.append(current_mask)

    return results


def merge_channels_with_unique_ids(masks, nodata):
    C, H, W = masks.shape

    temp_merged = np.zeros((H, W), dtype=np.int64)
    offset_base = masks.max() + 1

    for ch in range(C):
        current_mask = masks[ch]
        active_pixels = current_mask > 0
        temp_merged[active_pixels] = current_mask[active_pixels] + (ch * offset_base)

    _, indices = np.unique(temp_merged, return_inverse=True)
    sequential_mask = indices.reshape((H, W))

    final_dtype = np.int32
    if isinstance(nodata, float):
        final_dtype = np.float32

    final_result = np.full((H, W), nodata, dtype=final_dtype)

    valid_pixels = sequential_mask > 0
    final_result[valid_pixels] = sequential_mask[valid_pixels]

    return final_result


def preview(mask, vor_array, title):
    plot_data = np.where(mask, vor_array, np.nan)

    plt.figure(figsize=(8, 6))
    plt.imshow(plot_data, cmap="hsv", interpolation="nearest")
    plt.colorbar(label="Divided cluster index")

    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_raster_path")
    parser.add_argument("--output_raster_path", default="")
    parser.add_argument("--min_area_ratio", type=float, default=0.01)

    args = parser.parse_args()

    input_raster_path = args.input_raster_path
    output_raster_path = args.output_raster_path
    min_area_ratio = args.min_area_ratio

    with rasterio.open(input_raster_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        crs = src.crs
        profile = src.profile

    if nodata is not None:
        valid_mask = arr != nodata
    else:
        valid_mask = ~np.isnan(arr)

    valid_pixels = np.count_nonzero(valid_mask)
    merge_threshold = valid_pixels * min_area_ratio

    results = divide_clusters_spatial(
        arr,
        merge_threshold,
        nodata,
    )
    merged_mask = merge_channels_with_unique_ids(np.array(results), nodata)
    preview(valid_mask, merged_mask, title="Divided clusters")

    if output_raster_path != "":
        profile.update(
            {"count": 1, "dtype": "uint8", "compress": "lzw", "nodata": nodata}
        )

        with rasterio.open(output_raster_path, "w", **profile) as dst:
            dst.write(merged_mask, 1)
