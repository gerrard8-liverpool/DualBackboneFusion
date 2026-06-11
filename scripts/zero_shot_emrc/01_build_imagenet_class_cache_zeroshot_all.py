#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = ["rn50", "rn101", "vit_b32", "vit_b16"]


def parse_grid(spec: str):
    # "0:1:0.05" or "0,0.25,0.5,0.75,1"
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
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype("float32")


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z


def load_row_outputs(row, fusion_mode):
    outs = {}
    labels = None
    class_names = None
    text_features = None

    for bk in BACKBONES:
        z = load_npz(row[f"{bk}_path"])
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
                raise RuntimeError(f"Label mismatch in row {row.to_dict()}")

        if bk == "vit_b16":
            if "class_names" in z.files:
                class_names = [str(x) for x in z["class_names"].tolist()]
            if "text_features" in z.files:
                text_features = z["text_features"].astype("float32")
            elif "text_embeddings" in z.files:
                text_features = z["text_embeddings"].astype("float32")

    if class_names is None:
        class_names = [str(i) for i in range(outs["vit_b16"].shape[1])]

    if text_features is None:
        raise RuntimeError("Cannot find text_features/text_embeddings from vit_b16 npz.")

    return outs, labels, class_names, text_features


def mix_vit_anchor(outs, alpha):
    weak = (outs["rn50"] + outs["rn101"] + outs["vit_b32"]) / 3.0
    return alpha * outs["vit_b16"] + (1.0 - alpha) * weak


def accuracy(scores, labels):
    return float((scores.argmax(axis=1) == labels).mean() * 100.0)


def best_alpha_for_mask(outs, labels, mask, grid):
    """Find best alpha on a subset.

    Important: subset first, then fuse. The previous implementation fused the
    full ImageNet matrix for every class and every alpha, which is extremely slow.
    """
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 1.0, 0.0

    outs_sub = {bk: v[idx] for bk, v in outs.items()}
    labels_sub = labels[idx]

    best_a = None
    best_acc = -1.0
    for a in grid:
        scores = mix_vit_anchor(outs_sub, a)
        acc = accuracy(scores, labels_sub)
        if acc > best_acc:
            best_acc = acc
            best_a = a

    return float(best_a), float(best_acc)


def normalize_keys(x):
    x = x.astype("float32")
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", default="zeroshot")
    ap.add_argument("--source_dataset", default="imagenet")
    ap.add_argument("--split", default="all")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--fusion_mode", default="prob_avg", choices=["prob_avg", "std_logits", "raw_logits"])
    ap.add_argument("--weight_grid", default="0:1:0.05")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    grid = parse_grid(args.weight_grid)

    sub = df[
        (df["protocol"].astype(str) == args.protocol) &
        (df["dataset"].astype(str) == args.source_dataset) &
        (df["split"].astype(str) == args.split) &
        (df["seed"].astype(int).isin(seeds))
    ].copy()

    if len(sub) == 0:
        raise SystemExit(f"No source rows found for dataset={args.source_dataset}, protocol={args.protocol}, split={args.split}, seeds={seeds}")

    all_outs = {bk: [] for bk in BACKBONES}
    all_labels = []
    class_names_ref = None
    text_features_ref = None

    for _, row in sub.iterrows():
        print(f"[load source] dataset={row['dataset']} split={row['split']} seed={row['seed']}")
        outs, labels, class_names, text_features = load_row_outputs(row, args.fusion_mode)

        if class_names_ref is None:
            class_names_ref = class_names
            text_features_ref = text_features
        else:
            if class_names_ref != class_names:
                raise RuntimeError("Class names mismatch across ImageNet seed rows.")

        for bk in BACKBONES:
            all_outs[bk].append(outs[bk])
        all_labels.append(labels)

    outs_cat = {bk: np.concatenate(all_outs[bk], axis=0) for bk in BACKBONES}
    labels_cat = np.concatenate(all_labels, axis=0)

    n_classes = len(class_names_ref)
    print(f"[source] rows={len(sub)} samples={len(labels_cat)} classes={n_classes}")

    # Dataset-level cache.
    best_dataset_alpha, best_dataset_acc = best_alpha_for_mask(
        outs_cat, labels_cat, np.ones_like(labels_cat, dtype=bool), grid
    )

    single_acc = {}
    for bk in BACKBONES:
        single_acc[bk] = accuracy(outs_cat[bk], labels_cat)

    fixed_acc = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        fixed_acc[f"vit_anchor_{a:.2f}"] = accuracy(mix_vit_anchor(outs_cat, a), labels_cat)

    classes = []
    for c in range(n_classes):
        if c % 100 == 0:
            print(f"[class cache] {c}/{n_classes}", flush=True)
        mask = labels_cat == c
        best_a, best_acc = best_alpha_for_mask(outs_cat, labels_cat, mask, grid)

        per_bk_acc = {}
        for bk in BACKBONES:
            per_bk_acc[bk] = accuracy(outs_cat[bk][mask], labels_cat[mask]) if mask.sum() else 0.0

        classes.append({
            "class_id": int(c),
            "class_name": str(class_names_ref[c]),
            "key": text_features_ref[c].astype(float).tolist(),
            "best_alpha": float(best_a),
            "best_acc": float(best_acc),
            "num_samples": int(mask.sum()),
            "acc_rn50": float(per_bk_acc["rn50"]),
            "acc_rn101": float(per_bk_acc["rn101"]),
            "acc_vit_b32": float(per_bk_acc["vit_b32"]),
            "acc_vit_b16": float(per_bk_acc["vit_b16"]),
        })

    cache = {
        "cache_type": "imagenet_multibackbone_class_cache",
        "prompt_learner": "zeroshot_clip",
        "protocol": args.protocol,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "seeds": seeds,
        "fusion_mode": args.fusion_mode,
        "weight_family": "vit_b16_anchor_equal_weak",
        "weight_grid": grid,
        "backbones": BACKBONES,
        "dataset_cache": {
            "best_alpha": float(best_dataset_alpha),
            "best_acc": float(best_dataset_acc),
            "single_acc": single_acc,
            "fixed_acc": fixed_acc,
        },
        "num_classes": int(n_classes),
        "classes": classes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"[write] {out}")
    print(f"[dataset_cache] alpha={best_dataset_alpha:.3f} acc={best_dataset_acc:.4f}")
    print("[single_acc]", single_acc)
    print("[fixed_acc]", fixed_acc)


if __name__ == "__main__":
    main()
