#!/usr/bin/env python
import argparse
import re
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

ACC_RE = re.compile(r"\*\s*accuracy:\s*([0-9.]+)%")


def parse_acc(log_path: Path):
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    vals = ACC_RE.findall(text)
    if not vals:
        return None
    return float(vals[-1])


def fmt(vals):
    if not vals:
        return "-"
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f}±{s:.2f} ({len(vals)})"


def infer_dataset_split_seed(path: Path):
    parts = path.parts
    dataset = None
    split = None
    seed = None
    for i, x in enumerate(parts):
        if x == "test" and i + 1 < len(parts):
            dataset = parts[i + 1]
        if x.startswith("split_"):
            split = x.replace("split_", "")
        if x.startswith("seed"):
            try:
                seed = int(x.replace("seed", ""))
            except Exception:
                pass
    return dataset, split, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    records = []
    for log in sorted(root.rglob("log.txt")):
        acc = parse_acc(log)
        if acc is None:
            continue
        dataset, split, seed = infer_dataset_split_seed(log)
        if dataset is None or split is None or seed is None:
            continue
        records.append({"dataset": dataset, "split": split, "seed": seed, "accuracy": acc, "path": str(log)})

    grouped = defaultdict(dict)
    for r in records:
        grouped[(r["dataset"], r["seed"])][r["split"]] = r["accuracy"]

    datasets = sorted({r["dataset"] for r in records})
    lines = []
    lines.append("# CoOpDualEnc Feature Fusion B2N Summary")
    lines.append("")
    lines.append(f"Found `{len(records)}` evaluated split logs.")
    lines.append("")
    lines.append("| Dataset | Base | New | HM | All | Seeds |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for dataset in datasets:
        base_vals, new_vals, hm_vals, all_vals, seeds = [], [], [], [], []
        for seed in sorted({r["seed"] for r in records if r["dataset"] == dataset}):
            d = grouped.get((dataset, seed), {})
            b = d.get("base")
            n = d.get("new")
            a = d.get("all")
            if b is not None:
                base_vals.append(b)
            if n is not None:
                new_vals.append(n)
            if a is not None:
                all_vals.append(a)
            if b is not None and n is not None:
                hm_vals.append(2 * b * n / (b + n) if b + n > 0 else 0.0)
                seeds.append(seed)
        lines.append(f"| {dataset} | {fmt(base_vals)} | {fmt(new_vals)} | {fmt(hm_vals)} | {fmt(all_vals)} | {len(seeds)} |")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WROTE] {out}")


if __name__ == "__main__":
    main()
