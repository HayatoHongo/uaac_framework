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

    print("current file path", "clustering/kmeans.py")
    print("def reduce_image(im, resize_ratio)")

    print("im(type)\n", type(im))
    print(f"【COND】 isinstance(im, np.ndarray)={isinstance(im, np.ndarray)}")
    if isinstance(im, np.ndarray):
        print("【ENTER】if isinstance(im, np.ndarray):")
        print("im.shape\n", im.shape)
        print("im.dtype\n", im.dtype)
        print("【EXIT】if isinstance(im, np.ndarray):")
    else:
        print("im(type)\n", type(im))
        print("print(risk): print(im) disabled for safety")
    print("resize_ratio\n", resize_ratio)

    image_pil = Image.fromarray(im)
    print("image_pil\n", type(image_pil))
    width, height = image_pil.size
    print("width\n", width)
    print("height\n", height)

    resized_image = image_pil.resize(
        (width // resize_ratio, height // resize_ratio),
        resample=Image.Resampling.BILINEAR,
    )
    print("resized_image\n", type(resized_image))

    im = np.array(resized_image)

    print("im(shape)\n", im.shape)
    print("im(dtype)\n", im.dtype)
    return im


def kmeans_segmentation(
    image_path: str,
    k: int,
    gaussian_sigma: float,
    clip_limit: float,
    show_preprocessed: int,
    resize_ratio: int,
):

    print("current file path", "clustering/kmeans.py")
    print(
        "def kmeans_segmentation(image_path: str, k: int, gaussian_sigma: float, clip_limit: float, show_preprocessed: int, resize_ratio: int)"
    )

    print("image_path\n", image_path)
    print("k\n", k)
    print("gaussian_sigma\n", gaussian_sigma)
    print("clip_limit\n", clip_limit)
    print("show_preprocessed\n", show_preprocessed)
    print("resize_ratio\n", resize_ratio)

    k = int(k)
    print("k\n", k)

    with rasterio.open(image_path) as src:
        print("src(type)\n", type(src))
        image = src.read()  # (C, H, W)
        profile = src.profile

    print("image(shape)\n", image.shape)
    print("image(dtype)\n", image.dtype)
    print("profile(keys)\n", list(profile.keys()))

    image = np.moveaxis(image, 0, -1)
    print("image(shape)\n", image.shape)
    print("image(dtype)\n", image.dtype)

    print(f"【COND】 image.shape[2] == 4 -> {image.shape[2] == 4}, image.shape[2]={image.shape[2]}")
    if image.shape[2] == 4:
        print("【ENTER】if image.shape[2] == 4:")
        image = image[:, :, :3]
        print("image(shape)\n", image.shape)
        print("image(dtype)\n", image.dtype)
        print("【EXIT】if image.shape[2] == 4:")
    else:
        print(f"【COND】 image.shape[2] == 1 -> {image.shape[2] == 1}, image.shape[2]={image.shape[2]}")
        if image.shape[2] == 1:
            print("【ENTER】if image.shape[2] == 1:")
            image = np.repeat(image, 3, axis=2)
            print("image(shape)\n", image.shape)
            print("image(dtype)\n", image.dtype)
            print("【EXIT】if image.shape[2] == 1:")

    image = reduce_image(image, resize_ratio)
    print("image(shape)\n", image.shape)
    print("image(dtype)\n", image.dtype)

    clahe_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    print("clahe_image(type)\n", type(clahe_image))
    print("clahe_image(shape)\n", clahe_image.shape)
    clahe_blurred_image = gaussian(clahe_image, sigma=gaussian_sigma, channel_axis=-1)
    print("clahe_blurred_image(type)\n", type(clahe_blurred_image))
    print("clahe_blurred_image(shape)\n", clahe_blurred_image.shape)
    preprocessed_image = img_as_float(clahe_blurred_image)
    print("preprocessed_image(type)\n", type(preprocessed_image))
    print("preprocessed_image(shape)\n", preprocessed_image.shape)
    print("preprocessed_image(dtype)\n", preprocessed_image.dtype)

    print(f"【COND】 show_preprocessed={show_preprocessed}")
    if show_preprocessed:
        print("【ENTER】if show_preprocessed:")
        plt.imshow(preprocessed_image)
        plt.title("Preprocessed Image")
        plt.axis("off")
        pre_vis_output_path = f"preprocessed_kmeans_k_{k}.png"
        plt.savefig(pre_vis_output_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.show()
        plt.close()
        print(f"preprocessed visualization saved: {pre_vis_output_path}")
        print("【EXIT】if show_preprocessed:")

    h, w, c = preprocessed_image.shape
    print("h\n", h)
    print("w\n", w)
    print("c\n", c)
    pixels = preprocessed_image.reshape(-1, c)
    print("pixels(shape)\n", pixels.shape)
    print("pixels(dtype)\n", pixels.dtype)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    print("kmeans\n", type(kmeans))
    labels = kmeans.fit_predict(pixels)
    print("labels(shape)\n", labels.shape)
    print("labels(dtype)\n", labels.dtype)

    segmented = labels.reshape(h, w).astype(np.uint8)
    print("segmented(shape)\n", segmented.shape)
    print("segmented(dtype)\n", segmented.dtype)

    plt.imshow(segmented, cmap="nipy_spectral")
    plt.title(f"KMeans Clustering Result (k={k})")
    plt.axis("off")
    vis_output_path = f"segmented_output_kmeans_k_{k}.png"
    print("vis_output_path\n", vis_output_path)
    plt.savefig(vis_output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.show()
    plt.close()

    profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})
    print("profile(keys)\n", list(profile.keys()))

    output_path = f"segmented_output_kmeans_k_{k}.tif"
    print("output_path\n", output_path)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(segmented, 1)

    print(f"clustering result saved: {output_path}")
    print(f"visualization saved: {vis_output_path}")


print(f"【COND】 __name__ == '__main__' -> {__name__ == '__main__'}, __name__={__name__}")
if __name__ == "__main__":
    print("【ENTER】if __name__ == '__main__':")
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
    print("args(type)\n", type(args))
    print("args(vars)\n", vars(args))

    kmeans_segmentation(
        args.input,
        args.k,
        args.gaussian_sigma,
        args.clip_limit,
        args.show_preprocessed,
        args.resize_ratio,
    )
    print("【EXIT】if __name__ == '__main__':")
