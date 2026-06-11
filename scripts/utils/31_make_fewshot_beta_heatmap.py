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
    "caltech101",
    "dtd",
    "eurosat",
    "fgvc_aircraft",
    "food101",
    "oxford_flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "ucf101",
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


def load_fewshot_raw():
    raw_dir = ROOT / "outputs" / "few_shot_raw"
    paths = sorted(glob.glob(str(raw_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw few-shot csv files found under {raw_dir}")

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    needed_cols = {"dataset", "seed", "shots", "method", "acc"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in raw few-shot CSVs: {missing}")
    return df


def build_delta_table(df):
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


def make_heatmap(delta_df):
    max_abs = float(np.nanmax(np.abs(delta_df["delta"].values)))
    vmax = max(0.6, min(3.0, np.ceil(max_abs * 2) / 2))
    vmin = -vmax

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.2, 10.8),
        dpi=220,
        constrained_layout=True,
    )
    axes = axes.ravel()

    for ax, shots in zip(axes, SHOTS):
        sub = delta_df[delta_df["shots"] == shots].copy()
        mat = (
            sub.pivot(index="dataset", columns="beta", values="delta")
            .reindex(DATASET_ORDER)
            .reindex(columns=BETAS)
        )

        im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax, cmap="RdYlGn")

        ax.set_title(f"{shots}-shot", fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(len(BETAS)))
        ax.set_xticklabels([f"β={b}" for b in BETAS], rotation=0, fontsize=10)
        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels([PRETTY[d] for d in DATASET_ORDER], fontsize=10)

        ax.set_xticks(np.arange(-0.5, len(BETAS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DATASET_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                text = "NA" if np.isnan(val) else f"{val:+.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="black")

        vals = mat.values.flatten()
        vals = vals[~np.isnan(vals)]
        wins = int((vals > 0).sum())
        losses = int((vals < 0).sum())
        mean_delta = float(np.mean(vals))

        ax.text(
            1.02,
            0.5,
            f"positive cells: {wins}/{len(vals)}\nnegative cells: {losses}/{len(vals)}\nmean Δ: {mean_delta:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.80", boxstyle="round,pad=0.35"),
        )

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.86, pad=0.03)
    cbar.set_label("Δ accuracy vs NLC β=0", fontsize=11)

    fig.suptitle(
        "Few-shot train-aligned EMRC gains over NLC across shots and β values",
        fontsize=16,
        fontweight="bold",
    )

    out_png = FIG_DIR / "few_shot_beta_delta_heatmap.png"
    out_svg = FIG_DIR / "few_shot_beta_delta_heatmap.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", out_png)
    print("[saved]", out_svg)


def make_beta_sensitivity(delta_df):
    g = (
        delta_df.groupby(["shots", "beta"])["delta"]
        .agg(mean_delta="mean", std_delta="std")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=220)
    x = np.arange(len(BETAS))

    for shots in SHOTS:
        sub = g[g["shots"] == shots].set_index("beta").reindex(BETAS)
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


def save_delta_csv(delta_df):
    out_csv = ROOT / "summary_tables" / "few_shot" / "few_shot_beta_delta_by_dataset.csv"
    out_md = ROOT / "summary_tables" / "few_shot" / "few_shot_beta_delta_by_dataset.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    delta_df.to_csv(out_csv, index=False)

    with open(out_md, "w", encoding="utf-8") as f:
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

    print("[saved]", out_csv)
    print("[saved]", out_md)


def main():
    raw = load_fewshot_raw()
    delta_df = build_delta_table(raw)
    save_delta_csv(delta_df)
    make_heatmap(delta_df)
    make_beta_sensitivity(delta_df)


if __name__ == "__main__":
    main()
