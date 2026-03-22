import argparse
import os
import cv2
import numpy as np


def resize_img(img):
    resized_img = cv2.resize(img, (512, 512))
    return resized_img


def gray_world_white_balance(img):
    img = img.astype(np.float32)
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    img[:, :, 0] *= scale_b
    img[:, :, 1] *= scale_g
    img[:, :, 2] *= scale_r

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return img_clahe


def preprocess_underwater(img_path, save_path=None):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("img not found.")

    resized_img = resize_img(img)

    wb_img = gray_world_white_balance(resized_img)
    clahe_img = apply_clahe(wb_img)

    if save_path:
        cv2.imwrite(save_path, clahe_img)

    return clahe_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--output_folder")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    extensions = [".jpg", ".jpeg", ".png"]

    for fname in os.listdir(input_folder):
        if any(fname.lower().endswith(ext) for ext in extensions):
            img_path = os.path.join(input_folder, fname)
            save_path = os.path.join(output_folder, fname)

            preprocess_underwater(img_path, save_path)
