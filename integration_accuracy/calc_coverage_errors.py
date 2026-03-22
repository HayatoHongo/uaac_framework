import argparse

import pandas as pd


def calc_coverage_errors(ref_coverage_csv_file, pred_coverage_csv_file):
    df_coverage_ref = pd.read_csv(ref_coverage_csv_file)
    df_coverage_pred = pd.read_csv(pred_coverage_csv_file)

    total_coverage = float(df_coverage_ref.sum(axis=1).iloc[0])

    errors = abs(df_coverage_pred - df_coverage_ref)
    overall_coverage_error = errors.sum(axis=1).iloc[0] / total_coverage * 100

    print(f"Error_overall-coverage: {overall_coverage_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ref_coverage_csv_file")
    parser.add_argument("-pred_coverage_csv_file")

    args = parser.parse_args()

    ref_coverage_csv_file = args.ref_coverage_csv_file
    pred_coverage_csv_file = args.pred_coverage_csv_file

    calc_coverage_errors(
        ref_coverage_csv_file,
        pred_coverage_csv_file,
    )
