import argparse
import pandas as pd
import geopandas as gpd


def proportion_by_cluster(
    proportions_csv_file,
    shp_file,
    area_csv_file,
    means_output_csv,
    stds_output_csv,
    vectors_output_csv,
    coverage_output_csv,
    class_num,
):
    CLASS_NUM = class_num
    CLASSES = [str(i) for i in range(CLASS_NUM)]

    df_csv = pd.read_csv(proportions_csv_file)
    gdf = gpd.read_file(shp_file)
    df_area = pd.read_csv(area_csv_file) if area_csv_file != "" else None

    gdf["Id"] = gdf["Id"].astype(int)
    gdf = gdf.merge(df_csv, left_on="Id", right_on="id")

    vectors_by_cid = gdf.groupby("RASTERVALU")

    cluster_mean_stds = (
        vectors_by_cid[["RASTERVALU", "id"] + CLASSES]
        .agg(
            count=("id", "size"),
            **{f"{c}_mean": (c, "mean") for c in (CLASSES)},
            **{f"{c}_std": (c, "std") for c in (CLASSES)},
        )
        .reset_index()
    )

    if means_output_csv != "":
        cluster_mean_stds[
            ["RASTERVALU", "count"] + [f"{c}_mean" for c in CLASSES]
        ].to_csv(means_output_csv, index=False)
    if stds_output_csv != "":
        cluster_mean_stds[
            ["RASTERVALU", "count"] + [f"{c}_std" for c in CLASSES]
        ].to_csv(stds_output_csv, index=False)
    if vectors_output_csv != "":
        cols = ["RASTERVALU", "id"] + CLASSES
        vectors_by_cid_for_csv = (
            gdf[cols].sort_values(by=["RASTERVALU", "id"]).reset_index(drop=True)
        )
        vectors_by_cid_for_csv.to_csv(vectors_output_csv, index=False)

    if df_area is not None:
        coverage_df = cluster_mean_stds.merge(
            df_area, "inner", left_on="RASTERVALU", right_on="VALUE"
        )
        for c in range(CLASS_NUM):
            coverage_df[f"{str(c)}_coverage"] = (
                coverage_df[f"{str(c)}_mean"] * coverage_df["AREA"]
            )
        if coverage_output_csv != "":
            coverage_df[[f"{str(c)}_coverage" for c in range(CLASS_NUM)]].sum(
                axis=0
            ).to_csv(coverage_output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-proportion_csv_file")
    parser.add_argument("-shp_file")
    parser.add_argument("--area_csv_file", default="")
    parser.add_argument("--means_output_csv", default="")
    parser.add_argument("--stds_output_csv", default="")
    parser.add_argument("--vectors_output_csv", default="")
    parser.add_argument("--coverage_output_csv", default="")
    parser.add_argument("--class_num", type=int, default=6)

    args = parser.parse_args()

    proportion_by_cluster(
        args.proportion_csv_file,
        args.shp_file,
        args.area_csv_file,
        args.means_output_csv,
        args.stds_output_csv,
        args.vectors_output_csv,
        args.coverage_output_csv,
        args.class_num,
    )
