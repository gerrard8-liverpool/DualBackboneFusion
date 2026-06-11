#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ["rn50", "rn101", "vit_b32", "vit_b16"]


def parse_grid(spec):
    if ":" in spec:
        a, b, s = map(float, spec.split(":"))
        vals = []
        x = a
        while x <= b + 1e-9:
            vals.append(round(x, 6))
            x += s
        return vals
    return [float(x) for x in spec.split(",") if x.strip()]


def softmax_np(x, axis=1):
    x = x.astype("float64")
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return (e / np.sum(e, axis=axis, keepdims=True)).astype("float32")


def standardize_logits(x):
    return ((x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)).astype("float32")


def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def spherical_kmeans(x, k, iters=50, seed=1):
    rng = np.random.default_rng(seed)
    x = normalize(x.astype("float32"))

    n = x.shape[0]
    init = rng.choice(n, size=k, replace=False)
    centers = x[init].copy()

    assign = None
    for _ in range(iters):
        sim = x @ centers.T
        new_assign = sim.argmax(axis=1)

        if assign is not None and np.array_equal(assign, new_assign):
            break

        assign = new_assign

        for c in range(k):
            idx = np.where(assign == c)[0]
            if len(idx) == 0:
                centers[c] = x[rng.integers(0, n)]
            else:
                centers[c] = normalize(x[idx].mean(axis=0, keepdims=True))[0]

    return assign.astype("int64"), centers.astype("float32")


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
        elif fusion_mode == "std_logits":
            out = standardize_logits(logits)
        elif fusion_mode == "raw_logits":
            out = logits
        else:
            raise ValueError(f"Unknown fusion_mode={fusion_mode}")

        outs[bk] = out

        if labels is None:
            labels = z["labels"].astype("int64")
        else:
            if not np.array_equal(labels, z["labels"].astype("int64")):
                raise RuntimeError("Label mismatch across backbones.")

        if bk == "vit_b16":
            class_names = [str(x) for x in z["class_names"].tolist()]
            if "text_features" in z.files:
                text_features = z["text_features"].astype("float32")
            else:
                text_features = z["text_embeddings"].astype("float32")

    return outs, labels, class_names, text_features


def mix_vit_anchor(outs, alpha):
    weak = (outs["rn50"] + outs["rn101"] + outs["vit_b32"]) / 3.0
    return alpha * outs["vit_b16"] + (1.0 - alpha) * weak


def accuracy(scores, labels):
    return float((scores.argmax(axis=1) == labels).mean() * 100.0)


def best_alpha_for_mask(outs, labels, mask, grid):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 1.0, 0.0

    outs_sub = {bk: v[idx] for bk, v in outs.items()}
    labels_sub = labels[idx]

    best_a, best_acc = None, -1.0
    for a in grid:
        acc = accuracy(mix_vit_anchor(outs_sub, a), labels_sub)
        if acc > best_acc:
            best_a, best_acc = a, acc

    return float(best_a), float(best_acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", default="zeroshot")
    ap.add_argument("--source_dataset", default="imagenet")
    ap.add_argument("--split", default="all")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--fusion_mode", default="prob_avg", choices=["prob_avg", "raw_logits", "std_logits"])
    ap.add_argument("--num_clusters", type=int, default=50)
    ap.add_argument("--kmeans_iters", type=int, default=50)
    ap.add_argument("--kmeans_seed", type=int, default=1)
    ap.add_argument("--weight_grid", default="0:1:0.05")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    grid = parse_grid(args.weight_grid)

    src = df[
        (df["protocol"].astype(str) == args.protocol) &
        (df["dataset"].astype(str) == args.source_dataset) &
        (df["split"].astype(str) == args.split) &
        (df["seed"].astype(int).isin(seeds))
    ].copy()

    if len(src) == 0:
        raise SystemExit("No ImageNet source rows found.")

    all_outs = {bk: [] for bk in BACKBONES}
    all_labels = []
    class_names_ref = None
    text_features_ref = None

    for _, row in src.iterrows():
        print(f"[load source] {row['dataset']} seed={row['seed']}", flush=True)
        outs, labels, class_names, text_features = load_row_outputs(row, args.fusion_mode)

        if class_names_ref is None:
            class_names_ref = class_names
            text_features_ref = text_features
        else:
            if class_names_ref != class_names:
                raise RuntimeError("ImageNet class names mismatch across seeds.")

        for bk in BACKBONES:
            all_outs[bk].append(outs[bk])
        all_labels.append(labels)

    outs_cat = {bk: np.concatenate(all_outs[bk], axis=0) for bk in BACKBONES}
    labels_cat = np.concatenate(all_labels, axis=0)

    n_classes = len(class_names_ref)
    print(f"[source] samples={len(labels_cat)} classes={n_classes}", flush=True)

    # Dataset-level fallback.
    dataset_alpha, dataset_acc = best_alpha_for_mask(
        outs_cat, labels_cat, np.ones_like(labels_cat, dtype=bool), grid
    )

    single_acc = {bk: accuracy(outs_cat[bk], labels_cat) for bk in BACKBONES}
    fixed_acc = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fixed_acc[f"vit_anchor_{a:.2f}"] = accuracy(mix_vit_anchor(outs_cat, a), labels_cat)

    print(f"[dataset cache] alpha={dataset_alpha} acc={dataset_acc:.4f}", flush=True)

    # Semantic meta-class clustering.
    assign, centers = spherical_kmeans(
        text_features_ref,
        k=args.num_clusters,
        iters=args.kmeans_iters,
        seed=args.kmeans_seed,
    )

    clusters = []
    for cid in range(args.num_clusters):
        if cid % 10 == 0:
            print(f"[cluster] {cid}/{args.num_clusters}", flush=True)

        class_ids = np.where(assign == cid)[0]
        if len(class_ids) == 0:
            continue

        mask = np.isin(labels_cat, class_ids)
        best_a, best_acc = best_alpha_for_mask(outs_cat, labels_cat, mask, grid)

        cname = "meta_" + str(cid)
        member_names = [class_names_ref[i] for i in class_ids.tolist()]

        clusters.append({
            "class_id": int(cid),
            "class_name": cname,
            "key": centers[cid].astype(float).tolist(),
            "best_alpha": float(best_a),
            "best_acc": float(best_acc),
            "num_classes": int(len(class_ids)),
            "num_samples": int(mask.sum()),
            "member_class_ids": [int(i) for i in class_ids.tolist()],
            "member_class_names": member_names,
        })

    cache = {
        "cache_type": "imagenet_multibackbone_meta_cache",
        "prompt_learner": "zeroshot_clip",
        "protocol": args.protocol,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "seeds": seeds,
        "fusion_mode": args.fusion_mode,
        "weight_family": "vit_b16_anchor_equal_weak",
        "weight_grid": grid,
        "backbones": BACKBONES,
        "num_clusters": args.num_clusters,
        "kmeans_seed": args.kmeans_seed,
        "dataset_cache": {
            "best_alpha": float(dataset_alpha),
            "best_acc": float(dataset_acc),
            "single_acc": single_acc,
            "fixed_acc": fixed_acc,
        },
        # Keep key name "classes" so existing eval script can directly reuse it.
        "num_classes": int(len(clusters)),
        "classes": clusters,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    alphas = np.asarray([c["best_alpha"] for c in clusters], dtype=float)
    sizes = np.asarray([c["num_classes"] for c in clusters], dtype=float)

    print(f"[write] {out}", flush=True)
    print(f"[meta alpha] mean={alphas.mean():.4f} std={alphas.std():.4f} min={alphas.min():.2f} max={alphas.max():.2f}", flush=True)
    print(f"[cluster size] mean={sizes.mean():.2f} min={sizes.min():.0f} max={sizes.max():.0f}", flush=True)


if __name__ == "__main__":
    main()
