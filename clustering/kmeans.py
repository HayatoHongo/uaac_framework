import argparse
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
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

    # --- 変更前（skimage CLAHE / メモリピークでプロセスダウンしやすい） ---
    # clahe_image = exposure.equalize_adapthist(image, clip_limit=clip_limit)
    # print("clahe_image(type)\n", type(clahe_image))
    # print("clahe_image(shape)\n", clahe_image.shape)
    # print("clahe_image(dtype)\n", clahe_image.dtype)

    # --- 変更後（OpenCV CLAHE / 低メモリ） ---
    # OpenCV CLAHE は 8-bit single-channel を想定
    if image.dtype == np.uint8:
        image_u8 = image
    else:
        # rasterio 由来の uint16 等にも対応して uint8 にスケール
        image_u8 = np.empty(image.shape, dtype=np.uint8)
        if np.issubdtype(image.dtype, np.integer):
            maxv = float(np.iinfo(image.dtype).max)
            scale = 255.0 / maxv if maxv > 0 else 1.0
            for ch in range(image.shape[2]):
                image_u8[..., ch] = (image[..., ch].astype(np.float64) * scale).astype(
                    np.uint8
                )
        else:
            # float の場合は 0..1 を想定（必要ならデータに合わせて調整）
            for ch in range(image.shape[2]):
                image_u8[..., ch] = (
                    np.clip(image[..., ch], 0.0, 1.0) * 255.0
                ).astype(np.uint8)

    # skimage clip_limit(例:0.03) と OpenCV clipLimit はスケールが異なるため倍率を掛ける
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit) * 100.0,
        tileGridSize=(8, 8),
    )

    # 中間の uint8 画像を別に持たず、直接 float32 (0..1) を作る
    clahe_image = np.empty(image_u8.shape, dtype=np.float64)
    for ch in range(image_u8.shape[2]):
        clahe_image[..., ch] = clahe.apply(image_u8[..., ch]).astype(np.float64) / 255.0

    print("clahe_image(type)\n", type(clahe_image))
    print("clahe_image(shape)\n", clahe_image.shape)
    print("clahe_image(dtype)\n", clahe_image.dtype)

    clahe_blurred_image = gaussian(clahe_image, sigma=gaussian_sigma, channel_axis=-1) #gaussian(clahe_image, sigma=gaussian_sigma, channel_axis=-1)
    # gaussian() は float64 を返しやすいので、巨大画像では float32 に落としてメモリピークを抑える
    clahe_blurred_image = clahe_blurred_image.astype(np.float64, copy=False)
    print("clahe_blurred_image(type)\n", type(clahe_blurred_image))
    print("clahe_blurred_image(shape)\n", clahe_blurred_image.shape)
    # --- 変更前 ---
    # preprocessed_image = img_as_float(clahe_blurred_image)
    # --- 変更後（float64化を避ける） ---
    preprocessed_image = clahe_blurred_image
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

    # --- 変更前 ---
    # profile.update({"count": 1, "dtype": "uint8", "compress": "lzw"})

    # --- 変更後（resize_ratio != 1 でも正しく保存できるようサイズを反映） ---
    profile.update({
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
        "width": w,
        "height": h,
    })
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
        default=1,
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
