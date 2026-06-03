#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

def fmt(vals):
    if not vals:
        return "-"
    m = mean(vals)
    s = pstdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f}±{s:.2f} ({len(vals)})"

def collect(root: Path):
    records = []
    for p in sorted(root.rglob("results.json")):
        data = json.loads(p.read_text())
        meta = data.get("meta", {})
        for r in data.get("results", []):
            rec = {}
            rec.update(meta)
            rec.update(r)
            rec["_path"] = str(p)
            records.append(rec)
    return records

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    records = collect(root)
    if not records:
        raise SystemExit(f"No results.json found under {root}")

    targets = sorted({r.get("target", "unknown") for r in records})
    modes = ["raw_logits", "std_logits", "prob_avg"]
    weights = sorted({float(r["weight_vit"]) for r in records})

    lines = []
    lines.append("# Late Fusion ImageNet-source Cross-Dataset Summary")
    lines.append("")
    lines.append(f"Found `{len(records)}` result rows.")
    lines.append("")
    lines.append("Fusion definition:")
    lines.append("")
    lines.append("`fused = w * logits_vit + (1 - w) * logits_rn`")
    lines.append("")

    # fixed target-wise summary for key weights
    key_settings = [
        ("RN101 only", "prob_avg", 0.0),
        ("ViT-B/16 only", "prob_avg", 1.0),
        ("Raw w=0.50", "raw_logits", 0.5),
        ("Raw w=0.75", "raw_logits", 0.75),
        ("Std w=0.50", "std_logits", 0.5),
        ("Std w=0.75", "std_logits", 0.75),
        ("Prob w=0.50", "prob_avg", 0.5),
        ("Prob w=0.75", "prob_avg", 0.75),
    ]

    lines.append("## Target-wise fixed-weight results")
    lines.append("")
    lines.append("| Target | " + " | ".join(k[0] for k in key_settings) + " |")
    lines.append("|---" + "|---:" * len(key_settings) + "|")

    for target in targets:
        row = [target]
        for _, mode, w in key_settings:
            vals = [
                float(r["accuracy"])
                for r in records
                if r.get("target") == target
                and r.get("mode") == mode
                and abs(float(r["weight_vit"]) - w) < 1e-9
            ]
            row.append(fmt(vals))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Overall fixed-weight results")
    lines.append("")
    lines.append("| Setting | Accuracy |")
    lines.append("|---|---:|")
    for name, mode, w in key_settings:
        vals = [
            float(r["accuracy"])
            for r in records
            if r.get("mode") == mode and abs(float(r["weight_vit"]) - w) < 1e-9
        ]
        lines.append(f"| {name} | {fmt(vals)} |")

    lines.append("")
    lines.append("## Full overall by mode and weight")
    lines.append("")
    for mode in modes:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| w | Accuracy |")
        lines.append("|---:|---:|")
        for w in weights:
            vals = [
                float(r["accuracy"])
                for r in records
                if r.get("mode") == mode and abs(float(r["weight_vit"]) - w) < 1e-9
            ]
            lines.append(f"| {w:.2f} | {fmt(vals)} |")
        lines.append("")

    lines.append("## Best-over-weight diagnostic")
    lines.append("")
    lines.append("This table is diagnostic only. Do not report best-over-target weights as a fair main result unless the weight-selection rule is fixed without target labels.")
    lines.append("")
    lines.append("| Mode | Best-over-weight Accuracy | Mean selected w |")
    lines.append("|---|---:|---:|")
    for mode in modes:
        best_vals = []
        best_ws = []
        grouped = defaultdict(list)
        for r in records:
            if r.get("mode") != mode:
                continue
            key = (r.get("target"), r.get("seed"))
            grouped[key].append(r)
        for _, rows in grouped.items():
            best = max(rows, key=lambda x: float(x["accuracy"]))
            best_vals.append(float(best["accuracy"]))
            best_ws.append(float(best["weight_vit"]))
        lines.append(f"| {mode} | {fmt(best_vals)} | {mean(best_ws):.2f} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WROTE] {out}")

if __name__ == "__main__":
    main()
