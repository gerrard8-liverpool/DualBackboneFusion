#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from rpc_core import (
    accuracy,
    best_weight,
    fuse_outputs,
    harmonic_mean,
    load_combined_record,
    md_table,
    name_similarity,
    read_csv,
    read_json,
    soft_topk,
    transform_logits,
    weight_grid,
    write_csv,
)


def resolve(project_root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else project_root / q


def dataset_fallback(cache: Dict[str, Any], fallback_w: float, exclude: Sequence[str] = ()) -> float:
    excluded = {x.lower() for x in exclude}
    vals = [float(x["global_best_w"]) for x in cache.get("dataset_summaries", []) if x.get("dataset", "").lower() not in excluded]
    return float(np.mean(vals)) if vals else float(fallback_w)


def class_cache_weights(cache: Dict[str, Any], class_names: List[str], fallback_w: float, top_k: int, temp: float, value_key: str, exclude: Sequence[str], min_conf: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    excluded = {x.lower() for x in exclude}
    entries = [e for e in cache.get("class_entries", []) if e.get("dataset", "").lower() not in excluded]
    if not entries:
        return np.full(len(class_names), fallback_w, dtype=np.float64), []
    keys = [str(e.get("class_name", f"class_{i}")) for i, e in enumerate(entries)]
    values = np.asarray([float(e.get(value_key, e.get("best_w_shrunk", fallback_w))) for e in entries], dtype=np.float64)
    weights, traces = [], []
    for j, name in enumerate(class_names):
        sim = name_similarity(name, keys)
        idx, alpha = soft_topk(sim, top_k, temp)
        if idx.size == 0:
            wc, conf = fallback_w, 0.0
        else:
            wc, conf = float(np.sum(alpha * values[idx])), float(np.max(sim[idx]))
        if min_conf > 0:
            rho = min(1.0, max(0.0, conf / min_conf))
            wf = rho * wc + (1 - rho) * fallback_w
        else:
            rho, wf = 1.0, wc
        weights.append(wf)
        traces.append({
            "target_class_index": j, "target_class_name": name, "w_cache": wc, "w_final": wf,
            "fallback_w": fallback_w, "confidence": conf, "rho": rho,
            "retrieved": [{"dataset": entries[int(k)].get("dataset", ""), "class_name": entries[int(k)].get("class_name", ""), "value": float(values[int(k)]), "similarity": float(sim[int(k)]), "alpha": float(a)} for k, a in zip(idx.tolist(), alpha.tolist())],
        })
    return np.asarray(weights, dtype=np.float64), traces


def eval_one(row: Dict[str, str], rec, cache: Dict[str, Any], mode: str, fallback_w: float, top_k: int, temp: float, value_key: str, exclude_target_dataset: bool, min_conf: float):
    zv, zr = transform_logits(rec.logits_vit, rec.logits_rn, cache["fusion_mode"])
    y = rec.labels.astype(int)
    class_names = rec.class_names or [f"class_{j}" for j in range(zv.shape[1])]
    ds = row.get("dataset", "unknown")
    exclude = [ds] if exclude_target_dataset else []
    if mode == "vit_only":
        scores, used_w, traces = zv, 1.0, []
    elif mode == "rn_only":
        scores, used_w, traces = zr, 0.0, []
    elif mode == "fixed":
        scores, used_w, traces = fuse_outputs(zv, zr, fallback_w), fallback_w, []
    elif mode == "dataset_cache":
        used_w = dataset_fallback(cache, fallback_w, exclude)
        scores, traces = fuse_outputs(zv, zr, used_w), []
    elif mode == "class_cache":
        fw = dataset_fallback(cache, fallback_w, exclude)
        w, traces = class_cache_weights(cache, class_names, fw, top_k, temp, value_key, exclude, min_conf)
        scores, used_w = fuse_outputs(zv, zr, w), float(np.mean(w))
    elif mode == "oracle_dataset":
        used_w, _ = best_weight(zv, zr, y, weight_grid(0.05), tie_break_w=fallback_w)
        scores, traces = fuse_outputs(zv, zr, used_w), []
    else:
        raise ValueError(mode)
    return {
        "dataset": ds, "seed": row.get("seed", "unknown"), "split": row.get("split", "unknown"),
        "protocol": row.get("protocol", "unknown"), "mode": mode, "fusion_mode": cache["fusion_mode"],
        "acc": accuracy(scores, y), "mean_w": float(used_w), "n": int(y.size), "num_classes": int(zv.shape[1]),
        "path": row.get("path", ""),
    }, traces


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["mode"], r["split"])].append(r)
    out = []
    for (ds, mode, split), vals in sorted(groups.items()):
        accs = np.asarray([float(v["acc"]) for v in vals], dtype=float)
        out.append({"dataset": ds, "mode": mode, "split": split, "seeds": len(vals), "acc_mean": float(np.nanmean(accs)), "acc_std": float(np.nanstd(accs)), "mean_w": float(np.nanmean([float(v["mean_w"]) for v in vals]))})
    by = defaultdict(dict)
    for r in out:
        by[(r["dataset"], r["mode"])][r["split"]] = r
    for (ds, mode), d in sorted(by.items()):
        if "base" in d and "new" in d:
            out.append({"dataset": ds, "mode": mode, "split": "HM(base,new)", "seeds": min(d["base"]["seeds"], d["new"]["seeds"]), "acc_mean": harmonic_mean(d["base"]["acc_mean"], d["new"]["acc_mean"]), "acc_std": float("nan"), "mean_w": float("nan")})
    # global average by mode/split across datasets
    mode_split = defaultdict(list)
    for r in out:
        if not r["dataset"].startswith("__"):
            mode_split[(r["mode"], r["split"])].append(r)
    for (mode, split), vals in sorted(mode_split.items()):
        out.append({"dataset": "__AVG__", "mode": mode, "split": split, "seeds": sum(int(v["seeds"]) for v in vals), "acc_mean": float(np.nanmean([float(v["acc_mean"]) for v in vals])), "acc_std": float("nan"), "mean_w": float(np.nanmean([float(v["mean_w"]) for v in vals]))})
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate Reliability Prior Cache on standardized dual logits.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cache-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-datasets", nargs="*", default=None)
    ap.add_argument("--target-splits", nargs="+", default=["new", "all", "test"])
    ap.add_argument("--protocols", nargs="*", default=None)
    ap.add_argument("--modes", nargs="+", default=["vit_only", "rn_only", "fixed", "dataset_cache", "class_cache", "oracle_dataset"])
    ap.add_argument("--fallback-w", type=float, default=0.75)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--sim-temp", type=float, default=0.07)
    ap.add_argument("--value-key", choices=["best_w", "best_w_shrunk"], default="best_w_shrunk")
    ap.add_argument("--exclude-target-dataset", action="store_true")
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    cache = read_json(args.cache_json)
    targets = {x.lower() for x in args.target_datasets} if args.target_datasets else None
    splits = {x.lower() for x in args.target_splits}
    protocols = {x.lower() for x in args.protocols} if args.protocols else None
    selected = []
    for row in read_csv(args.manifest):
        if targets is not None and row.get("dataset", "").lower() not in targets:
            continue
        if row.get("split", "").lower() not in splits:
            continue
        if protocols is not None and row.get("protocol", "").lower() not in protocols:
            continue
        selected.append(row)
    if not selected:
        raise SystemExit("No target rows matched filters.")

    results, traces, failures = [], [], []
    for i, row in enumerate(selected, 1):
        path = resolve(root, row["path"])
        print(f"[{i}/{len(selected)}] eval dataset={row.get('dataset')} split={row.get('split')} seed={row.get('seed')} file={path}")
        try:
            rec = load_combined_record(path)
        except Exception as e:
            failures.append({"row": row, "error": str(e)})
            print(f"  [WARN] failed load: {e}")
            continue
        for mode in args.modes:
            try:
                r, tr = eval_one(row, rec, cache, mode, args.fallback_w, args.top_k, args.sim_temp, args.value_key, args.exclude_target_dataset, args.min_confidence)
                results.append(r)
                for t in tr:
                    t.update({"dataset": r["dataset"], "seed": r["seed"], "split": r["split"], "mode": mode})
                    traces.append(t)
                print(f"  {mode:14s} acc={r['acc']:.2f} mean_w={r['mean_w']:.3f}")
            except Exception as e:
                failures.append({"row": row, "mode": mode, "error": str(e)})
                print(f"  [WARN] mode failed {mode}: {e}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_csv(results, out / "rpc_eval_detail.csv")
    srows = summarize(results)
    write_csv(srows, out / "rpc_eval_summary.csv")
    with open(out / "rpc_eval_summary.md", "w", encoding="utf-8") as f:
        f.write("# Reliability Prior Cache Evaluation\n\n")
        f.write(f"cache: `{args.cache_json}`\n\n")
        f.write(md_table(srows, ["dataset", "mode", "split", "seeds", "acc_mean", "acc_std", "mean_w"]))
    with open(out / "rpc_class_retrieval_trace.json", "w", encoding="utf-8") as f:
        json.dump(traces[:50000], f, indent=2, ensure_ascii=False)
    with open(out / "rpc_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)
    print(f"[DONE] summary={out / 'rpc_eval_summary.md'} failures={len(failures)}")

if __name__ == "__main__":
    main()
