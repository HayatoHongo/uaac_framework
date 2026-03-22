import argparse
import csv
import numpy as np
import pandas as pd


def compare_cluster_proportions(
    n_classes,
    ref_means_csv_file,
    pred_means_csv_file,
    means_error_output_csv_file,
):
    df_means_ref = pd.read_csv(ref_means_csv_file)
    df_means_pred = pd.read_csv(pred_means_csv_file)

    merged_means = pd.merge(
        df_means_ref,
        df_means_pred,
        on="RASTERVALU",
        how="inner",
        suffixes=("_ref", "_pred"),
    )
    num_predicted_clusters = len(np.unique(df_means_pred["RASTERVALU"]))
    print(f"num of predicted clusters: {num_predicted_clusters}")

    df_means_error = pd.DataFrame()
    for c in range(n_classes):
        df_means_error[f"{str(c)}_error"] = abs(
            merged_means[f"{str(c)}_mean_ref"] - merged_means[f"{str(c)}_mean_pred"]
        )

    error_c = df_means_error.mean(axis=1)
    print("Error_c:")
    print(error_c)

    df_means_error_for_output = df_means_error.copy()
    df_means_error_for_output["RASTERVALU"] = merged_means["RASTERVALU"]
    df_means_error_for_output["count_pred"] = merged_means["count_pred"]
    df_means_error_for_output = df_means_error_for_output[
        ["RASTERVALU", "count_pred"]
        + [
            c
            for c in df_means_error_for_output.columns
            if (c != "RASTERVALU" and c != "count_pred")
        ]
    ]

    if means_error_output_csv_file != "":
        df_means_error_for_output.to_csv(means_error_output_csv_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("-ref_means_csv_file")
    parser.add_argument("-pred_means_csv_file")
    parser.add_argument("--means_error_output_csv_file", default="")

    args = parser.parse_args()

    n_classes = args.n_classes
    ref_means_csv_file = args.ref_means_csv_file
    pred_means_csv_file = args.pred_means_csv_file
    means_error_output_csv_file = args.means_error_output_csv_file

    compare_cluster_proportions(
        n_classes,
        ref_means_csv_file,
        pred_means_csv_file,
        means_error_output_csv_file,
    )
