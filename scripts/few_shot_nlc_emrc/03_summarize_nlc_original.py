#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default="outputs/nlc_emrc/original_runs/*.csv")
    ap.add_argument("--out_dir", default="summary_tables/nlc_emrc/original")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise FileNotFoundError(args.input_glob)

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    os.makedirs(args.out_dir, exist_ok=True)

    raw_csv = os.path.join(args.out_dir, "nlc_original_raw.csv")
    summary_csv = os.path.join(args.out_dir, "nlc_original_summary.csv")
    summary_md = os.path.join(args.out_dir, "nlc_original_summary.md")
    per_dataset_csv = os.path.join(args.out_dir, "nlc_original_per_dataset.csv")
    per_dataset_md = os.path.join(args.out_dir, "nlc_original_per_dataset.md")

    df.to_csv(raw_csv, index=False)

    summary = (
        df.groupby(["shots", "method"], as_index=False)
        .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"), n=("acc", "count"))
        .sort_values(["shots", "mean_acc"], ascending=[True, False])
    )
    summary.to_csv(summary_csv, index=False)

    per_dataset = (
        df.groupby(["shots", "dataset", "method"], as_index=False)
        .agg(mean_acc=("acc", "mean"), std_acc=("acc", "std"), n=("acc", "count"))
        .sort_values(["shots", "dataset", "mean_acc"], ascending=[True, True, False])
    )
    per_dataset.to_csv(per_dataset_csv, index=False)

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# NLC Original Reproduction Summary\n\n")
        f.write(summary.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    with open(per_dataset_md, "w", encoding="utf-8") as f:
        f.write("# NLC Original Reproduction Per-dataset Results\n\n")
        f.write(per_dataset.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(summary.to_string(index=False))
    print("[saved]", summary_md)

if __name__ == "__main__":
    main()
