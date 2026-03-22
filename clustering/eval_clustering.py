import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def calc_intra_cluster_variation_score(df_stds, class_num):
    sigma_2s = df_stds[[f"{str(class_id)}_std" for class_id in range(class_num)]] ** 2
    return np.sqrt(sigma_2s.mean().mean())


def calc_inter_cluster_dissimilarity_score(df_means, class_num):
    vectors = df_means[
        [f"{str(class_id)}_mean" for class_id in range(class_num)]
    ].to_numpy()

    dist_vec = pdist(vectors, metric="euclidean")
    dist_mat = squareform(dist_vec)
    np.fill_diagonal(dist_mat, np.inf)
    min_distances = np.min(dist_mat, axis=1)

    return min_distances.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ref_means_csv_file")
    parser.add_argument("-ref_stds_csv_file")
    parser.add_argument("--class_num", type=int, default=6)

    args = parser.parse_args()

    df_means = pd.read_csv(args.ref_means_csv_file)
    df_stds = pd.read_csv(args.ref_stds_csv_file)
    class_num = args.class_num

    intra_cluster_variation_score = calc_intra_cluster_variation_score(
        df_stds, class_num
    )
    inter_cluster_dissimilarity_score = calc_inter_cluster_dissimilarity_score(
        df_means, class_num
    )

    print(f"Intra-cluster variation score: {intra_cluster_variation_score}")
    print(f"Inter-cluster dissimilarity score: {inter_cluster_dissimilarity_score}")
