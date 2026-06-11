#!/usr/bin/env python3
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ORDER = [
    "caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101",
    "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101",
]

PRETTY = {
    "caltech101": "Caltech101",
    "dtd": "DTD",
    "eurosat": "EuroSAT",
    "fgvc_aircraft": "FGVC",
    "food101": "Food101",
    "oxford_flowers": "Flowers",
    "oxford_pets": "Pets",
    "stanford_cars": "Cars",
    "sun397": "SUN397",
    "ucf101": "UCF101",
}

SHOTS = [1, 2, 4, 8]
BETAS = ["0.02", "0.05", "0.10", "0.20", "0.30"]

BASE_METHOD = "nlc_original_train_aligned_beta0.00"
BETA_METHOD = {b: f"nlc_emrc_train_aligned_beta{b}" for b in BETAS}


def load_raw():
    raw_dir = ROOT / "outputs" / "few_shot_raw"
    paths = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw few-shot CSV files found under {raw_dir}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    required = {"dataset", "seed", "shots", "method", "acc"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in few-shot raw CSVs: {missing}")

    df = df[df["shots"].isin(SHOTS)].copy()
    return df


def build_dataset_delta(df):
    records = []

    for shots in SHOTS:
        sub = df[df["shots"] == shots].copy()

        base = (
            sub[sub["method"] == BASE_METHOD]
            .groupby("dataset")["acc"]
            .mean()
            .reindex(DATASET_ORDER)
        )

        for beta in BETAS:
            method = BETA_METHOD[beta]
            cur = (
                sub[sub["method"] == method]
                .groupby("dataset")["acc"]
                .mean()
                .reindex(DATASET_ORDER)
            )

            delta = cur - base

            for dataset, value in delta.items():
                records.append({
                    "shots": shots,
                    "dataset": dataset,
                    "beta": beta,
                    "delta": float(value),
                })

    return pd.DataFrame(records)


def save_tables(delta_df):
    out_dir = ROOT / "summary_tables" / "few_shot"
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_csv = out_dir / "few_shot_beta_delta_by_dataset.csv"
    delta_md = out_dir / "few_shot_beta_delta_by_dataset.md"
    summary_csv = out_dir / "few_shot_shot_beta_summary.csv"
    best_csv = out_dir / "few_shot_best_beta_per_dataset.csv"

    delta_df.to_csv(delta_csv, index=False)

    summary_rows = []
    for shots in SHOTS:
        for beta in BETAS:
            sub = delta_df[(delta_df["shots"] == shots) & (delta_df["beta"] == beta)]
            vals = sub["delta"].values
            summary_rows.append({
                "shots": shots,
                "beta": beta,
                "mean_delta": float(np.mean(vals)),
                "median_delta": float(np.median(vals)),
                "positive_datasets": int((vals > 0).sum()),
                "negative_datasets": int((vals < 0).sum()),
                "num_datasets": int(len(vals)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_csv, index=False)

    best_rows = []
    for shots in SHOTS:
        for dataset in DATASET_ORDER:
            sub = delta_df[(delta_df["shots"] == shots) & (delta_df["dataset"] == dataset)].copy()
            best = sub.sort_values("delta", ascending=False).iloc[0]
            best_rows.append({
                "shots": shots,
                "dataset": dataset,
                "best_beta": best["beta"],
                "best_delta": float(best["delta"]),
            })
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(best_csv, index=False)

    with open(delta_md, "w", encoding="utf-8") as f:
        f.write("# Few-shot EMRC beta delta by dataset\n\n")
        f.write("Cell value is accuracy difference against NLC-equivalent beta=0.\n\n")
        for shots in SHOTS:
            sub = delta_df[delta_df["shots"] == shots]
            mat = (
                sub.pivot(index="dataset", columns="beta", values="delta")
                .reindex(DATASET_ORDER)
                .reindex(columns=BETAS)
            )
            mat.index = [PRETTY[d] for d in mat.index]
            mat.columns = [f"β={b}" for b in mat.columns]
            f.write(f"## {shots}-shot\n\n")
            f.write(mat.to_markdown(floatfmt="+.4f"))
            f.write("\n\n")

    print("[saved]", delta_csv)
    print("[saved]", delta_md)
    print("[saved]", summary_csv)
    print("[saved]", best_csv)

    return summary, best_df


def make_shot_beta_summary_heatmap(summary):
    mat_mean = (
        summary.pivot(index="shots", columns="beta", values="mean_delta")
        .reindex(SHOTS)
        .reindex(columns=BETAS)
    )
    mat_pos = (
        summary.pivot(index="shots", columns="beta", values="positive_datasets")
        .reindex(SHOTS)
        .reindex(columns=BETAS)
    )

    vmax = max(0.6, float(np.nanmax(np.abs(mat_mean.values))) * 1.25)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10.6, 5.1), dpi=240)

    im = ax.imshow(mat_mean.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_title(
        "Mean few-shot EMRC gain over NLC for each shot and β",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(np.arange(len(BETAS)))
    ax.set_xticklabels([f"β={b}" for b in BETAS], fontsize=11)
    ax.set_yticks(np.arange(len(SHOTS)))
    ax.set_yticklabels([f"{s}-shot" for s in SHOTS], fontsize=11)

    ax.set_xticks(np.arange(-0.5, len(BETAS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SHOTS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, shots in enumerate(SHOTS):
        for j, beta in enumerate(BETAS):
            mean_delta = mat_mean.loc[shots, beta]
            pos = int(mat_pos.loc[shots, beta])
            ax.text(
                j,
                i,
                f"{mean_delta:+.3f}\n{pos}/10 ds +",
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.025)
    cbar.set_label("Mean Δ accuracy vs NLC β=0", fontsize=11)

    fig.text(
        0.5,
        -0.03,
        "Each cell reports the mean dataset-level gain and the number of target datasets with positive gain.",
        ha="center",
        fontsize=10,
    )

    out_png = FIG_DIR / "few_shot_shot_beta_summary_heatmap.png"
    out_svg = FIG_DIR / "few_shot_shot_beta_summary_heatmap.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", out_png)
    print("[saved]", out_svg)


def make_best_beta_per_dataset_heatmap(best_df):
    mat_delta = (
        best_df.pivot(index="dataset", columns="shots", values="best_delta")
        .reindex(DATASET_ORDER)
        .reindex(columns=SHOTS)
    )
    mat_beta = (
        best_df.pivot(index="dataset", columns="shots", values="best_beta")
        .reindex(DATASET_ORDER)
        .reindex(columns=SHOTS)
    )

    vmax = max(0.8, float(np.nanmax(np.abs(mat_delta.values))) * 1.15)
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(10.2, 8.8), dpi=240)

    im = ax.imshow(mat_delta.values, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)

    ax.set_title(
        "Best available EMRC gain per dataset and shot",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    pos_counts = {shots: int((mat_delta[shots].values > 0).sum()) for shots in SHOTS}

    ax.set_xticks(np.arange(len(SHOTS)))
    ax.set_xticklabels([f"{s}-shot\n{pos_counts[s]}/10 positive" for s in SHOTS], fontsize=10)
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([PRETTY[d] for d in DATASET_ORDER], fontsize=11)

    ax.set_xticks(np.arange(-0.5, len(SHOTS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, dataset in enumerate(DATASET_ORDER):
        for j, shots in enumerate(SHOTS):
            val = mat_delta.loc[dataset, shots]
            beta = mat_beta.loc[dataset, shots]
            ax.text(
                j,
                i,
                f"{val:+.2f}\nβ={beta}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.025)
    cbar.set_label("Best Δ accuracy vs NLC β=0", fontsize=11)

    fig.text(
        0.5,
        -0.035,
        "Diagnostic view: each cell selects the best beta among tested values for that dataset and shot.",
        ha="center",
        fontsize=10,
    )

    out_png = FIG_DIR / "few_shot_best_beta_per_dataset.png"
    out_svg = FIG_DIR / "few_shot_best_beta_per_dataset.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", out_png)
    print("[saved]", out_svg)


def make_beta_sensitivity(summary):
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=240)

    x = np.arange(len(BETAS))
    for shots in SHOTS:
        sub = summary[summary["shots"] == shots].set_index("beta").reindex(BETAS)
        ax.plot(
            x,
            sub["mean_delta"].values,
            marker="o",
            linewidth=2.2,
            label=f"{shots}-shot",
        )

    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"β={b}" for b in BETAS])
    ax.set_ylabel("Mean Δ accuracy vs NLC β=0")
    ax.set_title("Beta sensitivity of train-aligned EMRC routing", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.30)
    ax.legend(frameon=False, ncol=4, loc="upper left")

    out_png = FIG_DIR / "few_shot_beta_sensitivity.png"
    out_svg = FIG_DIR / "few_shot_beta_sensitivity.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", out_png)
    print("[saved]", out_svg)


def main():
    raw = load_raw()
    delta_df = build_dataset_delta(raw)
    summary, best_df = save_tables(delta_df)

    make_shot_beta_summary_heatmap(summary)
    make_best_beta_per_dataset_heatmap(best_df)
    make_beta_sensitivity(summary)


if __name__ == "__main__":
    main()
