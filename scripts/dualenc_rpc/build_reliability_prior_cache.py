#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from rpc_core import (
    accuracy,
    best_weight,
    entropy_from_scores,
    load_combined_record,
    read_csv,
    transform_logits,
    true_margin,
    weight_grid,
    write_json,
)


def resolve(project_root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else project_root / q


def build_from_record(row: Dict[str, str], rec, fusion_mode: str, grid: List[float], shrink_lambda: float, tie_break_w: float):
    zv, zr = transform_logits(rec.logits_vit, rec.logits_rn, fusion_mode)
    y = rec.labels.astype(int)
    n, c = zv.shape
    wD, accD = best_weight(zv, zr, y, grid, tie_break_w)
    mv, mr = true_margin(zv, y), true_margin(zr, y)
    ev, er = entropy_from_scores(zv, already_prob=(fusion_mode == "prob_avg")), entropy_from_scores(zr, already_prob=(fusion_mode == "prob_avg"))
    class_names = rec.class_names or [f"class_{j}" for j in range(c)]
    entries = []
    for j in range(c):
        idx = np.where(y == j)[0]
        if idx.size == 0:
            continue
        wj, accj = best_weight(zv[idx], zr[idx], y[idx], grid, tie_break_w=wD)
        shrink = idx.size / (idx.size + max(shrink_lambda, 0.0))
        wjs = float(shrink * wj + (1.0 - shrink) * wD)
        pv = np.argmax(zv[idx], axis=1)
        pr = np.argmax(zr[idx], axis=1)
        entry = {
            "dataset": row.get("dataset", "unknown"), "seed": row.get("seed", "unknown"),
            "split": row.get("split", "unknown"), "protocol": row.get("protocol", "unknown"),
            "class_index": int(j), "class_name": str(class_names[j]), "num_samples": int(idx.size),
            "best_w": float(wj), "best_w_shrunk": wjs, "fusion_acc": float(accj),
            "vit_acc": float(np.mean(pv == y[idx]) * 100.0), "rn_acc": float(np.mean(pr == y[idx]) * 100.0),
            "vit_margin": float(np.mean(mv[idx])), "rn_margin": float(np.mean(mr[idx])),
            "vit_entropy": float(np.mean(ev[idx])), "rn_entropy": float(np.mean(er[idx])),
            "source_file": row.get("path", ""),
        }
        # Do not inline large text embeddings by default; class-name retrieval is the robust fallback.
        entries.append(entry)
    summary = {
        "dataset": row.get("dataset", "unknown"), "seed": row.get("seed", "unknown"),
        "split": row.get("split", "unknown"), "protocol": row.get("protocol", "unknown"),
        "source_file": row.get("path", ""), "num_samples": int(n), "num_classes": int(c),
        "global_best_w": float(wD), "global_fusion_acc": float(accD),
        "vit_acc": float(accuracy(zv, y)), "rn_acc": float(accuracy(zr, y)),
    }
    return summary, entries


def main():
    ap = argparse.ArgumentParser(description="Build dataset/class Reliability Prior Cache from standardized dual logits.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-datasets", nargs="+", required=True)
    ap.add_argument("--source-splits", nargs="+", default=["val", "base", "train", "test"])
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--fusion-mode", choices=["raw_logits", "std_logits", "prob_avg"], default="std_logits")
    ap.add_argument("--grid-step", type=float, default=0.05)
    ap.add_argument("--shrink-lambda", type=float, default=20.0)
    ap.add_argument("--tie-break-w", type=float, default=0.75)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    source_datasets = {x.lower() for x in args.source_datasets}
    source_splits = {x.lower() for x in args.source_splits}
    protocols = {x.lower() for x in args.protocols} if args.protocols else None
    rows = []
    for row in read_csv(args.manifest):
        if row.get("dataset", "").lower() not in source_datasets:
            continue
        if row.get("split", "").lower() not in source_splits:
            continue
        if protocols is not None and row.get("protocol", "").lower() not in protocols:
            continue
        rows.append(row)
    if args.max_records:
        rows = rows[:args.max_records]
    if not rows:
        raise SystemExit("No source rows matched filters. Inspect manifest CSV first.")

    grid = weight_grid(args.grid_step)
    summaries, entries, failures = [], [], []
    for i, row in enumerate(rows, 1):
        path = resolve(root, row["path"])
        print(f"[{i}/{len(rows)}] cache-source dataset={row.get('dataset')} split={row.get('split')} seed={row.get('seed')} file={path}")
        try:
            rec = load_combined_record(path)
            s, e = build_from_record(row, rec, args.fusion_mode, grid, args.shrink_lambda, args.tie_break_w)
            summaries.append(s); entries.extend(e)
            print(f"  ok: wD={s['global_best_w']:.2f} acc={s['global_fusion_acc']:.2f} C={s['num_classes']} N={s['num_samples']}")
        except Exception as e:
            failures.append({"row": row, "error": str(e)})
            print(f"  [WARN] failed: {e}")
    cache = {
        "cache_type": "ReliabilityPriorCache",
        "created_at_unix": int(time.time()),
        "fusion_mode": args.fusion_mode,
        "grid_step": args.grid_step,
        "shrink_lambda": args.shrink_lambda,
        "source_datasets": sorted(source_datasets),
        "source_splits": sorted(source_splits),
        "protocols": sorted(protocols) if protocols else None,
        "dataset_summaries": summaries,
        "class_entries": entries,
        "failures": failures,
    }
    write_json(cache, args.out)
    print(f"[DONE] cache={args.out} datasets={len(summaries)} class_entries={len(entries)} failures={len(failures)}")

if __name__ == "__main__":
    main()
