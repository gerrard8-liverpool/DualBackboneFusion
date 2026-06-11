#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ["rn50", "rn101", "vit_b32", "vit_b16"]


def parse_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def parse_grid(spec):
    if ":" in spec:
        a, b, step = map(float, spec.split(":"))
        vals = []
        x = a
        while x <= b + 1e-9:
            vals.append(round(x, 6))
            x += step
        return vals
    return parse_float_list(spec)


def softmax_np(x, axis=1):
    x = x.astype("float64")
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)).astype("float32")


def standardize_logits(x):
    return ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)).astype("float32")


def normalize(x):
    x = x.astype("float32")
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def clip01(x):
    return np.clip(x, 0.0, 1.0).astype("float32")


def accuracy(scores, labels):
    return float((scores.argmax(axis=1) == labels).mean() * 100.0)


def mix_vit_anchor(outs, alpha):
    weak = (outs["rn50"] + outs["rn101"] + outs["vit_b32"]) / 3.0
    return alpha * outs["vit_b16"] + (1.0 - alpha) * weak


def mix_classwise_vit_anchor(outs, alpha_vec):
    weak = (outs["rn50"] + outs["rn101"] + outs["vit_b32"]) / 3.0
    a = np.asarray(alpha_vec, dtype="float32")[None, :]
    return a * outs["vit_b16"] + (1.0 - a) * weak


def best_alpha_dataset(outs, labels, grid):
    best_a, best_acc = None, -1.0
    for a in grid:
        acc = accuracy(mix_vit_anchor(outs, a), labels)
        if acc > best_acc:
            best_a, best_acc = a, acc
    return float(best_a), float(best_acc)


def load_row_outputs(row, fusion_mode):
    outs = {}
    labels = None
    class_names = None
    text_features = None

    for bk in BACKBONES:
        z = np.load(row[f"{bk}_path"], allow_pickle=True)
        logits = z["logits"].astype("float32")

        if fusion_mode == "prob_avg":
            out = softmax_np(logits)
        elif fusion_mode == "raw_logits":
            out = logits
        elif fusion_mode == "std_logits":
            out = standardize_logits(logits)
        else:
            raise ValueError(f"Unknown fusion_mode={fusion_mode}")

        outs[bk] = out

        cur_labels = z["labels"].astype("int64")
        if labels is None:
            labels = cur_labels
        elif not np.array_equal(labels, cur_labels):
            raise RuntimeError("Label mismatch across backbones.")

        if bk == "vit_b16":
            class_names = [str(x) for x in z["class_names"].tolist()]
            if "text_features" in z.files:
                text_features = z["text_features"].astype("float32")
            elif "text_embeddings" in z.files:
                text_features = z["text_embeddings"].astype("float32")
            else:
                raise RuntimeError("Missing text_features/text_embeddings.")

    return outs, labels, class_names, text_features


def load_one_cache(path):
    cache = json.loads(Path(path).read_text())
    keys = np.asarray([c["key"] for c in cache["classes"]], dtype="float32")
    keys = normalize(keys)
    values = np.asarray([c["best_alpha"] for c in cache["classes"]], dtype="float32")
    dataset_alpha = float(cache["dataset_cache"]["best_alpha"])
    return {
        "path": str(path),
        "cache": cache,
        "keys": keys,
        "values": values,
        "dataset_alpha": dataset_alpha,
    }


def retrieve_one_cache(target_keys, cache_obj, topk, temperature):
    q = normalize(target_keys)
    keys = cache_obj["keys"]
    values = cache_obj["values"]

    sim = q @ keys.T
    topk = min(topk, keys.shape[0])

    idx = np.argpartition(-sim, kth=topk - 1, axis=1)[:, :topk]
    top_sim = np.take_along_axis(sim, idx, axis=1)

    order = np.argsort(-top_sim, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    top_sim = np.take_along_axis(top_sim, order, axis=1)

    logits = top_sim / max(temperature, 1e-8)
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)

    top_alpha = values[idx]
    alpha_meta = (w * top_alpha).sum(axis=1)

    alpha_var = (w * (top_alpha - alpha_meta[:, None]) ** 2).sum(axis=1)
    alpha_std_within = np.sqrt(np.maximum(alpha_var, 0.0))

    max_sim = top_sim[:, 0]
    gap = top_sim[:, 0] - top_sim[:, 1] if topk >= 2 else np.ones_like(max_sim)

    entropy = -(w * np.log(w + 1e-12)).sum(axis=1)
    entropy_norm = entropy / np.log(topk) if topk > 1 else np.zeros_like(entropy)

    return {
        "alpha_meta": alpha_meta.astype("float32"),
        "alpha_std_within": alpha_std_within.astype("float32"),
        "max_sim": max_sim.astype("float32"),
        "gap": gap.astype("float32"),
        "entropy_norm": entropy_norm.astype("float32"),
    }


def normalized_gate_by_quantile(x, low_q=0.25, high_q=0.75):
    lo = float(np.quantile(x, low_q))
    hi = float(np.quantile(x, high_q))
    return clip01((x - lo) / (hi - lo + 1e-12))


def retrieve_ensemble(target_keys, caches, topk, temperature):
    per = [retrieve_one_cache(target_keys, c, topk, temperature) for c in caches]

    alpha_stack = np.stack([r["alpha_meta"] for r in per], axis=0)
    max_sim_stack = np.stack([r["max_sim"] for r in per], axis=0)
    gap_stack = np.stack([r["gap"] for r in per], axis=0)
    entropy_stack = np.stack([r["entropy_norm"] for r in per], axis=0)
    within_std_stack = np.stack([r["alpha_std_within"] for r in per], axis=0)

    alpha_mean = alpha_stack.mean(axis=0)
    alpha_seed_std = alpha_stack.std(axis=0)

    return {
        "alpha_mean": alpha_mean.astype("float32"),
        "alpha_seed_std": alpha_seed_std.astype("float32"),
        "max_sim_mean": max_sim_stack.mean(axis=0).astype("float32"),
        "gap_mean": gap_stack.mean(axis=0).astype("float32"),
        "entropy_norm_mean": entropy_stack.mean(axis=0).astype("float32"),
        "alpha_within_std_mean": within_std_stack.mean(axis=0).astype("float32"),
        "alpha_stack": alpha_stack.astype("float32"),
    }


def compute_confidence(retr, dataset_alpha, args):
    sim_gate = normalized_gate_by_quantile(retr["max_sim_mean"], args.sim_low_q, args.sim_high_q)
    gap_gate = normalized_gate_by_quantile(retr["gap_mean"], args.gap_low_q, args.gap_high_q)
    entropy_gate = clip01(1.0 - retr["entropy_norm_mean"])

    if args.within_sigma <= 0:
        within_cons = np.ones_like(retr["alpha_within_std_mean"], dtype="float32")
    else:
        within_cons = np.exp(-((retr["alpha_within_std_mean"] / args.within_sigma) ** 2)).astype("float32")

    if args.seed_sigma <= 0:
        seed_cons = np.ones_like(retr["alpha_seed_std"], dtype="float32")
    else:
        seed_cons = np.exp(-((retr["alpha_seed_std"] / args.seed_sigma) ** 2)).astype("float32")

    semantic_soft = (
        args.sim_weight * sim_gate
        + args.gap_weight * gap_gate
        + args.entropy_weight * entropy_gate
    ) / max(args.sim_weight + args.gap_weight + args.entropy_weight, 1e-12)

    if args.conf_formula == "semantic_seed":
        rho = semantic_soft * seed_cons
    elif args.conf_formula == "semantic_within_seed":
        rho = semantic_soft * within_cons * seed_cons
    elif args.conf_formula == "sim_seed":
        rho = sim_gate * seed_cons
    elif args.conf_formula == "sim_gap_seed":
        rho = np.sqrt(sim_gate * gap_gate) * seed_cons
    elif args.conf_formula == "seed_only":
        rho = seed_cons
    elif args.conf_formula == "softavg_only":
        rho = semantic_soft
    else:
        raise ValueError(f"Unknown conf_formula={args.conf_formula}")

    if args.use_alpha_impact:
        impact = clip01(np.abs(retr["alpha_mean"] - dataset_alpha) / max(args.impact_scale, 1e-8))
        rho = rho * impact

    if args.hard_filter:
        rho = rho * (
            (retr["alpha_seed_std"] <= args.seed_std_thr)
            & (retr["alpha_within_std_mean"] <= args.within_std_thr)
        ).astype("float32")

    return clip01(rho), {
        "sim_gate": sim_gate,
        "gap_gate": gap_gate,
        "entropy_gate": entropy_gate,
        "within_consistency": within_cons,
        "seed_consistency": seed_cons,
        "semantic_soft": semantic_soft,
    }


def compute_target_dataset_alpha(alpha_meta, rho, dataset_alpha, args):
    if not args.use_target_dataset_prior:
        return float(dataset_alpha), 0.0

    weights = rho.astype("float64") + 1e-6
    target_alpha = float(np.sum(weights * alpha_meta) / np.sum(weights))

    if args.dataset_eta_mode == "fixed":
        eta = float(args.dataset_eta)
    elif args.dataset_eta_mode == "mean_rho":
        eta = float(np.mean(rho))
    else:
        raise ValueError(f"Unknown dataset_eta_mode={args.dataset_eta_mode}")

    eta = float(np.clip(eta, 0.0, 1.0))
    alpha_base = (1.0 - eta) * float(dataset_alpha) + eta * target_alpha
    return float(np.clip(alpha_base, 0.0, 1.0)), eta


def evaluate_row(row, caches, dataset_alpha, args):
    outs, labels, class_names, text_features = load_row_outputs(row, args.fusion_mode)

    retr = retrieve_ensemble(text_features, caches, args.topk, args.temperature)
    rho, parts = compute_confidence(retr, dataset_alpha, args)

    alpha_meta = retr["alpha_mean"]

    alpha_base, dataset_eta = compute_target_dataset_alpha(alpha_meta, rho, dataset_alpha, args)

    alpha_conf_global = rho * alpha_meta + (1.0 - rho) * float(dataset_alpha)
    alpha_hier = rho * alpha_meta + (1.0 - rho) * float(alpha_base)

    methods = {}

    for bk in BACKBONES:
        methods[f"{bk}_only"] = accuracy(outs[bk], labels)

    methods["fixed_equal4"] = accuracy(
        (outs["rn50"] + outs["rn101"] + outs["vit_b32"] + outs["vit_b16"]) / 4.0,
        labels,
    )

    for a in args.fixed_alphas:
        methods[f"fixed_vit_anchor_{a:.2f}"] = accuracy(mix_vit_anchor(outs, a), labels)

    methods["dataset_cache"] = accuracy(mix_vit_anchor(outs, dataset_alpha), labels)
    methods["target_semantic_dataset_cache"] = accuracy(mix_vit_anchor(outs, alpha_base), labels)
    methods["meta_ensemble_topk"] = accuracy(mix_classwise_vit_anchor(outs, alpha_meta), labels)
    methods["meta_ensemble_conf_gate"] = accuracy(mix_classwise_vit_anchor(outs, alpha_conf_global), labels)
    methods["meta_ensemble_hier_conf"] = accuracy(mix_classwise_vit_anchor(outs, alpha_hier), labels)

    if args.include_oracle:
        oracle_alpha, oracle_acc = best_alpha_dataset(outs, labels, args.oracle_grid)
        methods["oracle_dataset"] = oracle_acc
    else:
        oracle_alpha = ""

    rows = []
    for method, acc in methods.items():
        rows.append({
            "dataset": row["dataset"],
            "split": row["split"],
            "seed": int(row["seed"]),
            "fusion_mode": args.fusion_mode,
            "method": method,
            "accuracy": float(acc),
            "dataset_alpha": float(dataset_alpha),
            "target_dataset_alpha": float(alpha_base),
            "dataset_eta": float(dataset_eta),
            "oracle_alpha": oracle_alpha,
            "rho_mean": float(rho.mean()),
            "alpha_seed_std_mean": float(retr["alpha_seed_std"].mean()),
            "alpha_within_std_mean": float(retr["alpha_within_std_mean"].mean()),
            "max_sim_mean": float(retr["max_sim_mean"].mean()),
            "gap_mean": float(retr["gap_mean"].mean()),
            "entropy_norm_mean": float(retr["entropy_norm_mean"].mean()),
        })

    routing = pd.DataFrame({
        "dataset": row["dataset"],
        "split": row["split"],
        "seed": int(row["seed"]),
        "class_name": class_names,
        "alpha_meta_ensemble": alpha_meta,
        "alpha_conf_global": alpha_conf_global,
        "alpha_hier": alpha_hier,
        "alpha_base": alpha_base,
        "rho": rho,
        "alpha_seed_std": retr["alpha_seed_std"],
        "alpha_within_std_mean": retr["alpha_within_std_mean"],
        "max_sim_mean": retr["max_sim_mean"],
        "gap_mean": retr["gap_mean"],
        "entropy_norm_mean": retr["entropy_norm_mean"],
        "sim_gate": parts["sim_gate"],
        "gap_gate": parts["gap_gate"],
        "entropy_gate": parts["entropy_gate"],
        "within_consistency": parts["within_consistency"],
        "seed_consistency": parts["seed_consistency"],
        "semantic_soft": parts["semantic_soft"],
    })

    return rows, routing


def write_summary(df, out_md):
    summary = (
        df.groupby(["fusion_mode", "method"], as_index=False)
        .agg(mean_acc=("accuracy", "mean"), std_acc=("accuracy", "std"), n=("accuracy", "count"))
        .sort_values(["fusion_mode", "mean_acc"], ascending=[True, False])
    )

    lines = ["# Ensemble Confidence-Gated MetaCache Summary\n"]
    for fm, g in summary.groupby("fusion_mode"):
        lines.append(f"\n## Fusion mode: `{fm}`\n")
        lines.append("| Method | Mean Acc | Std | n |")
        lines.append("|---|---:|---:|---:|")
        for _, r in g.iterrows():
            lines.append(f"| {r['method']} | {r['mean_acc']:.4f} | {r['std_acc']:.4f} | {int(r['n'])} |")

    Path(out_md).write_text("\n".join(lines))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--caches", required=True, help="Comma-separated meta cache json files.")
    ap.add_argument("--protocol", default="zeroshot")
    ap.add_argument("--split", default="all")
    ap.add_argument("--exclude_datasets", default="imagenet")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--fusion_mode", default="prob_avg", choices=["prob_avg", "raw_logits", "std_logits"])

    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.05)

    ap.add_argument("--conf_formula", default="semantic_seed",
                    choices=["semantic_seed", "semantic_within_seed", "sim_seed", "sim_gap_seed", "seed_only", "softavg_only"])
    ap.add_argument("--seed_sigma", type=float, default=0.10)
    ap.add_argument("--within_sigma", type=float, default=0.15)

    ap.add_argument("--sim_low_q", type=float, default=0.25)
    ap.add_argument("--sim_high_q", type=float, default=0.75)
    ap.add_argument("--gap_low_q", type=float, default=0.25)
    ap.add_argument("--gap_high_q", type=float, default=0.75)

    ap.add_argument("--sim_weight", type=float, default=0.5)
    ap.add_argument("--gap_weight", type=float, default=0.3)
    ap.add_argument("--entropy_weight", type=float, default=0.2)

    ap.add_argument("--hard_filter", action="store_true")
    ap.add_argument("--seed_std_thr", type=float, default=0.12)
    ap.add_argument("--within_std_thr", type=float, default=0.20)

    ap.add_argument("--use_alpha_impact", action="store_true")
    ap.add_argument("--impact_scale", type=float, default=0.10)

    ap.add_argument("--use_target_dataset_prior", action="store_true")
    ap.add_argument("--dataset_eta_mode", default="mean_rho", choices=["fixed", "mean_rho"])
    ap.add_argument("--dataset_eta", type=float, default=0.5)

    ap.add_argument("--fixed_alphas", default="0.50,0.75")
    ap.add_argument("--include_oracle", action="store_true")
    ap.add_argument("--oracle_grid", default="0:1:0.05")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    args.fixed_alphas = parse_float_list(args.fixed_alphas)
    args.oracle_grid = parse_grid(args.oracle_grid)

    cache_paths = parse_list(args.caches)
    caches = [load_one_cache(p) for p in cache_paths]

    dataset_alphas = [c["dataset_alpha"] for c in caches]
    dataset_alpha = float(np.mean(dataset_alphas))

    if max(dataset_alphas) - min(dataset_alphas) > 1e-6:
        print("[WARN] dataset alpha differs across caches:", dataset_alphas)
    print("[dataset_alpha]", dataset_alpha)
    print("[num_caches]", len(caches))

    df = pd.read_csv(args.manifest)
    df = df[
        (df["protocol"].astype(str) == args.protocol)
        & (df["split"].astype(str) == args.split)
    ].copy()

    excludes = parse_list(args.exclude_datasets)
    if excludes:
        df = df[~df["dataset"].astype(str).isin(excludes)].copy()

    seeds = [int(x) for x in parse_list(args.seeds)]
    df = df[df["seed"].astype(int).isin(seeds)].copy()

    if len(df) == 0:
        raise SystemExit("No target rows found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    routings = []

    print(f"[eval rows] {len(df)}")
    print(f"[conf] formula={args.conf_formula} topk={args.topk} temp={args.temperature} seed_sigma={args.seed_sigma} within_sigma={args.within_sigma}")

    for _, row in df.iterrows():
        print(f"[eval] dataset={row['dataset']} seed={row['seed']} fusion={args.fusion_mode}", flush=True)
        rows, routing = evaluate_row(row, caches, dataset_alpha, args)
        all_rows.extend(rows)
        routings.append(routing)

    res = pd.DataFrame(all_rows)
    routing_df = pd.concat(routings, ignore_index=True)

    result_csv = out_dir / f"results_{args.fusion_mode}.csv"
    routing_csv = out_dir / f"routing_{args.fusion_mode}.csv"
    summary_csv = out_dir / f"summary_{args.fusion_mode}.csv"
    summary_md = out_dir / f"summary_{args.fusion_mode}.md"

    res.to_csv(result_csv, index=False)
    routing_df.to_csv(routing_csv, index=False)

    summary = write_summary(res, summary_md)
    summary.to_csv(summary_csv, index=False)

    print("[write]", result_csv)
    print("[write]", routing_csv)
    print("[write]", summary_csv)
    print("[write]", summary_md)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
