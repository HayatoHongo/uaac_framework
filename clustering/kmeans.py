import argparse
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import img_as_float
from skimage import exposure
from PIL import Image


def reduce_image(im, resize_ratio):
    image_pil = Image.fromarray(im)
    width, height = image_pil.size

    resized_image = image_pil.resize(
        (width // resize_ratio, height // resize_ratio),
        resample=Image.Resampling.BILINEAR,
    )

    im = np.array(resized_image)
    return im


def kmeans_segmentation(
    image_path: str,
    k: int,
    gaussian_sigma: float,
    clip_limit: float,
    show_preprocessed: int,
    resize_ratio: int,
):
    k = int(k)

    with rasterio.open(image_path) as src:
        image = src.read()  # (C, H, W)
        profile = src.profile

    image = np.moveaxis(image, 0, -1)

    if image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    image = reduce_image(image, resize_ratio)

    clahe_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    clahe_blurred_image = gaussian(clahe_image, sigma=gaussian_sigma, channel_axis=-1)
    preprocessed_image = img_as_float(clahe_blurred_image)

    if show_preprocessed:
        plt.imshow(preprocessed_image)
        plt.title("Preprocessed Image")
        plt.axis("off")
        plt.show()

    h, w, c = preprocessed_image.shape
    pixels = preprocessed_image.reshape(-1, c)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixels)

    segmented = labels.reshape(h, w).astype(np.uint8)

    plt.imshow(segmented, cmap="nipy_spectral")
    plt.title(f"KMeans Clustering Result (k={k})")
    plt.axis("off")
    plt.show()

    profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})

    output_path = f"segmented_output_kmeans_k_{k}.tif"
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(segmented, 1)
    print(f"clustering result saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("k")
    parser.add_argument("--input")
    parser.add_argument(
        "--gaussian_sigma",
        default=4.0,
        type=float,
    )
    parser.add_argument(
        "--clip_limit",
        default=0.03,
        type=float,
    )
    parser.add_argument(
        "--show_preprocessed", default=0, type=int, help="show_preprocessed flag"
    )
    parser.add_argument(
        "--resize_ratio",
        default=5,
        type=int,
    )

    args = parser.parse_args()

    kmeans_segmentation(
        args.input,
        args.k,
        args.gaussian_sigma,
        args.clip_limit,
        args.show_preprocessed,
        args.resize_ratio,
    )
