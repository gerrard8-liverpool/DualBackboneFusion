#!/usr/bin/env python3
import os
import pandas as pd

IN_CSV = "summary_tables/reliability_prior_cache_hier/paper_ready/table1_main_results_prob_emrc.csv"
OUT_DIR = "summary_tables/reliability_prior_cache_hier/paper_ready_clean"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

# Clean main table: remove oracle-related columns from the main result.
drop_cols = [
    "Dataset-level Scalar Oracle",
    "Δ EMRC vs Scalar Oracle",
]
clean = df.drop(columns=[c for c in drop_cols if c in df.columns])

clean_csv = os.path.join(OUT_DIR, "table1_main_results_prob_emrc_clean_no_oracle.csv")
clean_md = os.path.join(OUT_DIR, "table1_main_results_prob_emrc_clean_no_oracle.md")
clean.to_csv(clean_csv, index=False)

# Recompute clean paired delta summary without oracle.
non_avg = clean[clean["dataset"] != "Average"].copy()
baselines = [
    ("BSS-ZSEn", "BSS-ZSEn"),
    ("ImageNet Dataset Cache", "ImageNet Dataset Cache"),
    ("Fixed Raw w0.50", "Fixed Raw w0.50"),
]

rows = []
for display, col in baselines:
    delta = non_avg["EMRC-TopK"] - non_avg[col]
    wins = int((delta > 1e-12).sum())
    ties = int((delta.abs() <= 1e-12).sum())
    losses = int((delta < -1e-12).sum())
    rows.append({
        "baseline": display,
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "wins/ties/losses": f"{wins}/{ties}/{losses}",
    })

delta_df = pd.DataFrame(rows)
delta_csv = os.path.join(OUT_DIR, "table1_delta_summary_prob_emrc_clean_no_oracle.csv")
delta_md = os.path.join(OUT_DIR, "table1_delta_summary_prob_emrc_clean_no_oracle.md")
delta_df.to_csv(delta_csv, index=False)

with open(clean_md, "w") as f:
    f.write("# Main Table: Zero-shot EMRC, excluding ImageNet\n\n")
    f.write(
        "Protocol: 10 non-ImageNet target datasets, 3 target seeds, "
        "5 k-means meta-cache seeds. ImageNet is used only as the source/cache dataset "
        "and is excluded from the target average. EMRC setting: k=100, topk=2, temperature=0.07.\n\n"
    )
    f.write("Oracle-based diagnostic results are excluded from this main table.\n\n")
    f.write("## Per-dataset Results\n\n")
    f.write(clean.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n\n## Paired Delta Summary\n\n")
    f.write(delta_df.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n")

with open(delta_md, "w") as f:
    f.write("# Paired Delta Summary: EMRC-TopK vs Fair Zero-shot Baselines\n\n")
    f.write(delta_df.to_markdown(index=False, floatfmt=".4f"))
    f.write("\n")

# Appendix oracle diagnostic table.
if "Dataset-level Scalar Oracle" in df.columns:
    oracle = df[[
        "dataset",
        "EMRC-TopK",
        "Dataset-level Scalar Oracle",
        "Δ EMRC vs Scalar Oracle",
    ]].copy()
    oracle = oracle.rename(columns={
        "Dataset-level Scalar Oracle": "Scalar α Oracle (diagnostic)",
        "Δ EMRC vs Scalar Oracle": "Δ EMRC vs Scalar α Oracle",
    })

    oracle_csv = os.path.join(OUT_DIR, "appendix_scalar_alpha_oracle_diagnostic.csv")
    oracle_md = os.path.join(OUT_DIR, "appendix_scalar_alpha_oracle_diagnostic.md")
    oracle.to_csv(oracle_csv, index=False)

    with open(oracle_md, "w") as f:
        f.write("# Appendix: Scalar α Oracle Diagnostic\n\n")
        f.write(
            "This table is a diagnostic reference only. "
            "The scalar α oracle uses target labels to select a dataset-level scalar fusion weight, "
            "so it is not a fair zero-shot method and should not be included as a main baseline.\n\n"
        )
        f.write(oracle.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

print("[saved]", clean_csv)
print("[saved]", clean_md)
print("[saved]", delta_csv)
print("[saved]", delta_md)
print("[saved appendix oracle diagnostic]")
