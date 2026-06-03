#!/usr/bin/env python3
"""Summarize outputs produced by eval_late_fusion_logits.py."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple


def load_records(root: Path) -> List[dict]:
    records = []
    for path in sorted(root.rglob("results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] skip unreadable {path}: {exc}")
            continue
        meta = payload.get("meta", {})
        for result in payload.get("results", []):
            rec = dict(meta)
            rec.update(result)
            rec["path"] = str(path)
            records.append(rec)
    return records


def fmt_mean_std(values: List[float]) -> str:
    if not values:
        return "-"
    m = mean(values)
    s = pstdev(values) if len(values) > 1 else 0.0
    return f"{m:.2f}±{s:.2f} ({len(values)})"


def group_values(records: Iterable[dict], keys: Tuple[str, ...]) -> Dict[Tuple, List[float]]:
    out: Dict[Tuple, List[float]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(k, "") for k in keys)
        out[key].append(float(r["accuracy"]))
    return out


def get_best_by_file(records: List[dict], mode: str) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if r.get("mode") == mode:
            grouped[r["path"]].append(r)
    best = []
    for subset in grouped.values():
        best.append(max(subset, key=lambda x: float(x["accuracy"])))
    return best


def write_standard_summary(records: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Late Fusion Summary")
    lines.append("")
    lines.append(f"Found `{len(records)}` result rows.")
    lines.append("")

    if not records:
        output.write_text("\n".join(lines), encoding="utf-8")
        return

    # Fixed-weight summaries are the fair diagnostic table.
    fixed_pairs = [
        ("raw_logits", 0.0, "RN101 only"),
        ("raw_logits", 0.5, "Raw 0.5 fusion"),
        ("raw_logits", 1.0, "ViT-B/16 only"),
        ("std_logits", 0.5, "Std-logits 0.5 fusion"),
        ("prob_avg", 0.5, "Prob 0.5 fusion"),
    ]

    lines.append("## Target-wise fixed-weight results")
    lines.append("")
    lines.append("| Target | RN101 only | ViT-B/16 only | Raw 0.5 | Std 0.5 | Prob 0.5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    targets = sorted({r.get("target", "") for r in records})
    for target in targets:
        row = [target]
        for mode, weight, _ in fixed_pairs:
            vals = [
                float(r["accuracy"])
                for r in records
                if r.get("target", "") == target
                and r.get("mode") == mode
                and abs(float(r.get("weight_vit", -999)) - weight) < 1e-9
            ]
            row.append(fmt_mean_std(vals))
        # order in fixed_pairs is RN, raw0.5, ViT, std0.5, prob0.5; reorder visually
        lines.append(f"| {row[0]} | {row[1]} | {row[3]} | {row[2]} | {row[4]} | {row[5]} |")

    lines.append("")
    lines.append("## Overall fixed-weight results")
    lines.append("")
    lines.append("| Setting | Accuracy |")
    lines.append("|---|---:|")
    label_map = {
        ("raw_logits", 0.0): "RN101 only",
        ("raw_logits", 1.0): "ViT-B/16 only",
        ("raw_logits", 0.5): "Raw logits fusion, w=0.5",
        ("std_logits", 0.5): "Standardized logits fusion, w=0.5",
        ("prob_avg", 0.5): "Probability average, w=0.5",
    }
    for mode, weight in [("raw_logits", 0.0), ("raw_logits", 1.0), ("raw_logits", 0.5), ("std_logits", 0.5), ("prob_avg", 0.5)]:
        vals = [
            float(r["accuracy"])
            for r in records
            if r.get("mode") == mode and abs(float(r.get("weight_vit", -999)) - weight) < 1e-9
        ]
        lines.append(f"| {label_map[(mode, weight)]} | {fmt_mean_std(vals)} |")

    lines.append("")
    lines.append("## Best-over-weight diagnostic")
    lines.append("")
    lines.append("This table is diagnostic only. Do not report best-over-target weights as a fair main result unless the weight-selection rule is fixed without target labels.")
    lines.append("")
    lines.append("| Mode | Best-over-weight Accuracy | Mean selected w |")
    lines.append("|---|---:|---:|")
    for mode in ["raw_logits", "std_logits", "prob_avg"]:
        best = get_best_by_file(records, mode)
        vals = [float(r["accuracy"]) for r in best]
        weights = [float(r["weight_vit"]) for r in best]
        mean_w = mean(weights) if weights else float("nan")
        lines.append(f"| {mode} | {fmt_mean_std(vals)} | {mean_w:.2f} |")

    output.write_text("\n".join(lines), encoding="utf-8")


def harmonic_mean(base: float, new: float) -> float:
    if base + new <= 0:
        return 0.0
    return 2 * base * new / (base + new)


def write_b2n_summary(records: List[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Late Fusion B2N Sanity Summary")
    lines.append("")
    lines.append(f"Found `{len(records)}` result rows.")
    lines.append("")

    if not records:
        output.write_text("\n".join(lines), encoding="utf-8")
        return

    # Fixed settings only; HM requires paired base/new per dataset/seed/mode/weight.
    settings = [
        ("raw_logits", 0.0, "RN101 only"),
        ("raw_logits", 1.0, "ViT-B/16 only"),
        ("raw_logits", 0.5, "Raw 0.5 fusion"),
        ("std_logits", 0.5, "Std 0.5 fusion"),
        ("prob_avg", 0.5, "Prob 0.5 fusion"),
    ]

    datasets = sorted({r.get("source", "") for r in records})
    lines.append("| Dataset | Setting | Base | New | HM |")
    lines.append("|---|---|---:|---:|---:|")

    for dataset in datasets:
        for mode, weight, label in settings:
            by_seed = defaultdict(dict)
            for r in records:
                if r.get("source") != dataset:
                    continue
                if r.get("mode") != mode:
                    continue
                if abs(float(r.get("weight_vit", -999)) - weight) > 1e-9:
                    continue
                split = r.get("subsample_classes")
                seed = str(r.get("seed"))
                if split in {"base", "new"}:
                    by_seed[seed][split] = float(r["accuracy"])

            base_vals, new_vals, hm_vals = [], [], []
            for pair in by_seed.values():
                if "base" in pair and "new" in pair:
                    base_vals.append(pair["base"])
                    new_vals.append(pair["new"])
                    hm_vals.append(harmonic_mean(pair["base"], pair["new"]))

            lines.append(
                f"| {dataset} | {label} | {fmt_mean_std(base_vals)} | {fmt_mean_std(new_vals)} | {fmt_mean_std(hm_vals)} |"
            )

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing result.json files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--b2n", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.root))
    if args.b2n:
        write_b2n_summary(records, Path(args.output))
    else:
        write_standard_summary(records, Path(args.output))
    print(f"[DONE] wrote {args.output}")


if __name__ == "__main__":
    main()
