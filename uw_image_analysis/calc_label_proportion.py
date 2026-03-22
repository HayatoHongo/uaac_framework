import argparse
import os
import re
import numpy as np
import pandas as pd

pattern = r"tile_?\d+\.npy"


def calc_label_proportion(input_folder, output_path, class_num):
    proportion_vectors = []
    labels = np.arange(class_num)
    csv_columns = ["id"] + labels.tolist()

    for fname in os.listdir(input_folder):
        if re.fullmatch(pattern, fname):
            tile_id = int(fname.replace("tile_", "").replace(".npy", ""))
            npy_path = os.path.join(input_folder, fname)

            mask = np.load(npy_path)

            no_data_num = (mask == 255).sum()

            if no_data_num > 0:
                print(f"tile_{str(tile_id)}: {str(no_data_num)}")

            proportion = [
                (mask == label).sum() / (mask.size - no_data_num) for label in labels
            ]
            proportion_vector = [tile_id] + proportion

            proportion_vectors.append(proportion_vector)

    df = pd.DataFrame(proportion_vectors, columns=csv_columns).sort_values(by="id")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--output_path")
    parser.add_argument("--class_num", type=int, default=6)

    args = parser.parse_args()

    input_folder = args.input_folder
    output_path = args.output_path
    class_num = args.class_num

    calc_label_proportion(input_folder, output_path, class_num)
