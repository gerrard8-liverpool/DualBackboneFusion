#!/usr/bin/env python3
from pathlib import Path
import glob
import math
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


def find_existing(candidates):
    for p in candidates:
        p = ROOT / p
        if p.exists():
            return p
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(str(ROOT / x) for x in candidates))


def nice_delta_range(values, pad_ratio=0.18):
    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if abs(vmax - vmin) < 1e-9:
        vmin -= 0.1
        vmax += 0.1

    pad = (vmax - vmin) * pad_ratio
    vmin -= pad
    vmax += pad

    # Always include zero so gains/losses are visually anchored.
    vmin = min(vmin, 0.0)
    vmax = max(vmax, 0.0)

    # Round to readable ticks.
    step_raw = (vmax - vmin) / 4.0
    if step_raw <= 0:
        step = 0.1
    else:
        pow10 = 10 ** math.floor(math.log10(step_raw))
        step = min([1, 2, 5, 10], key=lambda x: abs(x * pow10 - step_raw)) * pow10

    lo = math.floor(vmin / step) * step
    hi = math.ceil(vmax / step) * step

    # Avoid too narrow range.
    if hi - lo < 0.5:
        lo = min(lo, -0.25)
        hi = max(hi, 0.25)

    ticks = np.arange(lo, hi + step * 0.5, step)
    return float(lo), float(hi), ticks


def closed(arr):
    arr = np.asarray(arr, dtype=float)
    return np.r_[arr, arr[0]]


def make_delta_radar(categories, series, title, out_stem, ylabel="Δ accuracy"):
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
    dmin, dmax, delta_ticks = nice_delta_range(all_values)

    # Shift to positive radius because polar negative radius is visually confusing.
    offset = -dmin
    rmax = dmax + offset
    rticks = delta_ticks + offset

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig = plt.figure(figsize=(10.8, 10.8), dpi=240)
    ax = plt.subplot(111, polar=True)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, rmax)

    # Grid and tick labels show original delta values.
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{x:+.2f}" for x in delta_ticks], fontsize=8)
    ax.yaxis.grid(True, linewidth=0.65, alpha=0.38)
    ax.xaxis.grid(True, linewidth=0.65, alpha=0.30)

    # Zero reference ring.
    zero_r = offset
    theta_dense = np.linspace(0, 2 * np.pi, 512)
    ax.plot(theta_dense, np.full_like(theta_dense, zero_r), linewidth=1.5, linestyle="--", alpha=0.75, label="0 reference")

    # Dataset labels outside the plot, with white background to avoid overlap.
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    label_r = rmax + 0.08 * max(rmax, 1.0)
    for angle, label in zip(angles, categories):
        deg = np.degrees(angle)
        ha = "center"
        if 0 < deg < 180:
            ha = "left"
        elif 180 < deg < 360:
            ha = "right"

        ax.text(
            angle,
            label_r,
            label,
            ha=ha,
            va="center",
            fontsize=10,
            fontweight="semibold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=1.6),
        )

    styles = [
        dict(linewidth=2.5, marker="o", markersize=4.5),
        dict(linewidth=2.3, marker="s", markersize=4.3),
        dict(linewidth=2.3, marker="^", markersize=4.3),
        dict(linewidth=2.0, marker="D", markersize=4.0),
    ]

    for i, (name, vals) in enumerate(series.items()):
        vals = np.asarray(vals, dtype=float)
        shifted = vals + offset
        style = styles[i % len(styles)]
        ax.plot(angles_closed, closed(shifted), label=name, **style)
        ax.fill(angles_closed, closed(shifted), alpha=0.055)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=36)

    # Small annotation for scale.
    ax.text(
        0.5,
        -0.10,
        f"{ylabel}; radial ticks are shifted for visualization, labels show real deltas.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=min(3, len(series) + 1),
        frameon=False,
        fontsize=10,
    )

    png = FIG_DIR / f"{out_stem}.png"
    svg = FIG_DIR / f"{out_stem}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print("[saved]", png)
    print("[saved]", svg)


def make_zero_shot_delta_radar():
    table_path = find_existing([
        "summary_tables/zero_shot/table1_main_results_prob_emrc_clean_no_oracle.csv",
        "summary_tables/zero_shot/table1_main_results_prob_emrc.csv",
        "summary_tables/reliability_prior_cache_hier/paper_ready_clean/table1_main_results_prob_emrc_clean_no_oracle.csv",
        "summary_tables/reliability_prior_cache_hier/paper_ready/table1_main_results_prob_emrc.csv",
    ])

    df = pd.read_csv(table_path)
    df = df[df["dataset"].astype(str) != "Average"].copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df = df.sort_values("dataset")

    required = ["EMRC-TopK", "BSS-ZSEn", "ImageNet Dataset Cache", "Fixed Raw w0.50"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {table_path}: {missing}")

    categories = [PRETTY[x] for x in DATASET_ORDER]

    series = {
        "EMRC − BSS-ZSEn": df["EMRC-TopK"].values - df["BSS-ZSEn"].values,
        "EMRC − ImageNet Cache": df["EMRC-TopK"].values - df["ImageNet Dataset Cache"].values,
        "EMRC − Fixed Raw": df["EMRC-TopK"].values - df["Fixed Raw w0.50"].values,
    }

    make_delta_radar(
        categories=categories,
        series=series,
        title="Zero-shot EMRC gains over strong fusion baselines",
        out_stem="zero_shot_delta_radar",
        ylabel="Δ accuracy",
    )


def make_few_shot_delta_radar():
    raw_dir = ROOT / "outputs/few_shot_raw"
    paths = sorted(glob.glob(str(raw_dir / "*.csv")))

    if not paths:
        raise FileNotFoundError(f"No few-shot raw CSV files found under {raw_dir}")

    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    raw = raw[raw["shots"].isin([1, 2, 4, 8])].copy()

    methods = [
        "nlc_original_train_aligned_beta0.00",
        "nlc_emrc_train_aligned_beta0.20",
        "nlc_emrc_train_aligned_beta0.30",
    ]

    missing_methods = [m for m in methods if m not in set(raw["method"])]
    if missing_methods:
        raise RuntimeError(f"Missing methods in few-shot raw CSVs: {missing_methods}")

    raw = raw[raw["method"].isin(methods)].copy()

    g = (
        raw.groupby(["dataset", "method"])["acc"]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="method", values="acc")
        .reindex(DATASET_ORDER)
    )

    categories = [PRETTY[x] for x in DATASET_ORDER]
    base = g["nlc_original_train_aligned_beta0.00"].values

    series = {
        "β=0.20 − NLC": g["nlc_emrc_train_aligned_beta0.20"].values - base,
        "β=0.30 − NLC": g["nlc_emrc_train_aligned_beta0.30"].values - base,
    }

    make_delta_radar(
        categories=categories,
        series=series,
        title="Few-shot train-aligned EMRC gains over NLC",
        out_stem="few_shot_delta_radar",
        ylabel="Δ accuracy averaged over 1/2/4/8-shot",
    )


if __name__ == "__main__":
    make_zero_shot_delta_radar()
    make_few_shot_delta_radar()
