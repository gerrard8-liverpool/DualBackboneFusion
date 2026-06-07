#!/usr/bin/env python
import argparse, json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

def fmt(vals):
    if not vals: return "-"
    return f"{mean(vals):.2f}±{(pstdev(vals) if len(vals)>1 else 0.0):.2f} ({len(vals)})"

def split_from(p, meta):
    s = meta.get("subsample_classes") or meta.get("split") or ""
    if s: return s
    for part in str(p).split("/"):
        if part.startswith("split_"): return part.replace("split_", "")
    return "all"

def protocol_from(p):
    s = str(p)
    if "/b2n/" in s: return "b2n"
    if "/strict_dg/" in s: return "strict_dg"
    if "/xd/" in s: return "xd"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = []
    for p in sorted(Path(args.root).rglob("results.json")):
        data = json.loads(p.read_text())
        meta = data.get("meta", {})
        for r in data.get("results", []):
            rec = dict(meta); rec.update(r)
            rec["protocol"] = protocol_from(p)
            rec["dataset"] = meta.get("target") or meta.get("dataset") or meta.get("source") or "unknown"
            rec["split"] = split_from(p, meta)
            rows.append(rec)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["protocol"], r["dataset"], r["split"], r["mode"], float(r["weight_vit"]))].append(float(r["accuracy"]))
    lines = ["# Dataset-level Cached Fusion from Existing Results", "", "GPU-free summary from existing `outputs/dualenc/late_fusion/**/results.json`.", "", "| Protocol | Dataset | Split | Mode | Best w | Best Acc | ViT-only | Delta |", "|---|---|---|---|---:|---:|---:|---:|"]
    keys = sorted({k[:4] for k in groups})
    for protocol, dataset, split, mode in keys:
        w_acc = {w: mean(vals) for (p,d,s,m,w), vals in groups.items() if (p,d,s,m)==(protocol,dataset,split,mode)}
        if not w_acc: continue
        best_w = max(w_acc, key=w_acc.get); best = w_acc[best_w]; vit = w_acc.get(1.0, float('nan'))
        lines.append(f"| {protocol} | {dataset} | {split} | {mode} | {best_w:.2f} | {best:.2f} | {vit:.2f} | {best-vit:+.2f} |")
    lines += ["", "## Overall by protocol / mode / weight", "", "| Protocol | Mode | w | Accuracy |", "|---|---|---:|---:|"]
    og = defaultdict(list)
    for r in rows: og[(r["protocol"], r["mode"], float(r["weight_vit"]))].append(float(r["accuracy"]))
    for (protocol, mode, w), vals in sorted(og.items()): lines.append(f"| {protocol} | {mode} | {w:.2f} | {fmt(vals)} |")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[WROTE] {out}")
if __name__ == "__main__": main()
