#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


BACKBONES = ["rn50", "rn101", "vit_b32", "vit_b16"]
PATH_COLS = {
    "rn50": "rn50_path",
    "rn101": "rn101_path",
    "vit_b32": "vit_b32_path",
    "vit_b16": "vit_b16_path",
}

LOGIT_KEYS = ["logits", "image_logits", "clip_logits", "scores"]
FEATURE_KEYS = [
    "image_features",
    "image_feature",
    "features",
    "feature",
    "img_features",
    "img_feature",
    "embeddings",
    "embedding",
]
LABEL_KEYS = ["labels", "label", "targets", "target", "y", "gt"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve(root: str, p: str) -> str:
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def find_2d_by_keys(npz, keys, name):
    all_keys = list(npz.keys())
    for k in keys:
        if k in all_keys:
            arr = np.asarray(npz[k])
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return k, arr.astype("float32")
    raise RuntimeError(f"Cannot find required {name}. expected={keys}, available={all_keys}")


def find_labels(npz, n):
    all_keys = list(npz.keys())
    for k in LABEL_KEYS:
        if k in all_keys:
            arr = np.asarray(npz[k]).reshape(-1)
            if len(arr) == n and np.issubdtype(arr.dtype, np.integer):
                return k, arr.astype("int64")

    for k in all_keys:
        arr = np.asarray(npz[k]).reshape(-1)
        if len(arr) == n and np.issubdtype(arr.dtype, np.integer):
            return k, arr.astype("int64")

    raise RuntimeError(f"Cannot find labels. available={all_keys}")


def load_one_npz(path):
    z = np.load(path, allow_pickle=True)
    logit_key, logits = find_2d_by_keys(z, LOGIT_KEYS, "logits")
    feature_key, features = find_2d_by_keys(z, FEATURE_KEYS, "image_features")
    label_key, labels = find_labels(z, logits.shape[0])

    if features.shape[0] != logits.shape[0]:
        raise RuntimeError(f"N mismatch: features={features.shape}, logits={logits.shape}, path={path}")

    return logits, features, labels, {
        "path": path,
        "logit_key": logit_key,
        "feature_key": feature_key,
        "label_key": label_key,
        "feature_dim": int(features.shape[1]),
    }


def load_manifest_row(root, row):
    logits_list, feature_list, labels, meta = [], [], None, {}

    for b in BACKBONES:
        path = resolve(root, row[PATH_COLS[b]])
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        logits, features, y, m = load_one_npz(path)

        logits_list.append(logits)
        feature_list.append(features)
        meta[b] = m

        if labels is None:
            labels = y
        else:
            if len(labels) != len(y):
                raise RuntimeError(f"Label length mismatch at {path}")
            if not np.array_equal(labels, y):
                raise RuntimeError(f"Label order mismatch at {path}")

    logits = np.stack(logits_list, axis=1)          # [N, B, C]
    features = np.concatenate(feature_list, axis=1) # [N, sumD]
    return logits, features, labels, meta


def select_manifest_row(df, dataset, seed):
    q = df[df["dataset"].astype(str) == str(dataset)].copy()
    if len(q) == 0:
        raise RuntimeError(f"No manifest row for dataset={dataset}")

    if "seed" in q.columns:
        q_seed = q[q["seed"].astype(int) == int(seed)]
        if len(q_seed) > 0:
            q = q_seed

    if "split" in q.columns:
        q_all = q[q["split"].astype(str) == "all"]
        if len(q_all) > 0:
            q = q_all

    return q.iloc[0]


def sample_fewshot(labels, shots, seed):
    rng = np.random.default_rng(seed)
    train_idx = []

    for c in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == c)[0]
        take = min(shots, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        train_idx.extend(chosen.tolist())

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    mask = np.ones(len(labels), dtype=bool)
    mask[train_idx] = False
    test_idx = np.where(mask)[0].astype(np.int64)
    return train_idx, test_idx


class NeuralLogitController(nn.Module):
    """
    Strict NLC-style controller:
    concatenated image representations -> one hidden layer MLP -> per-backbone temperatures.
    """
    def __init__(self, input_dim, hidden_dim=128, num_backbones=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_backbones)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        raw_t = self.fc2(h)
        t = F.softplus(raw_t) + 1e-6
        return t


def nlc_scaled_logits(logits, features, model):
    t = model(features)                  # [N, B]
    scaled = logits / t[:, :, None]      # [N, B, C]
    final_logits = scaled.mean(dim=1)    # [N, C]
    return final_logits, scaled, t


def accuracy(scores, labels):
    pred = scores.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100.0


def build_weights_from_alpha(alpha):
    """
    alpha[c] is ViT-B/16 weight.
    Remaining weight is uniformly assigned to RN50, RN101, ViT-B/32.
    return [C, 4].
    """
    alpha = np.asarray(alpha, dtype="float32")
    alpha = np.clip(alpha, 1e-4, 1 - 1e-4)

    C = len(alpha)
    w = np.zeros((C, 4), dtype="float32")
    w[:, 3] = alpha
    w[:, 0:3] = (1.0 - alpha[:, None]) / 3.0
    return w


def load_converted_prior(path, num_classes, dataset, split, seed, fallback_alpha=0.4):
    alpha = np.full(num_classes, fallback_alpha, dtype="float32")

    if not path:
        print(f"[prior] no prior path; use fallback alpha={fallback_alpha}")
        return torch.tensor(build_weights_from_alpha(alpha), dtype=torch.float32)

    if not os.path.exists(path):
        raise FileNotFoundError(f"EMRC prior not found: {path}")

    df = pd.read_csv(path)
    required = {"dataset", "seed", "class_id", "alpha"}
    if not required.issubset(set(df.columns)):
        raise RuntimeError(f"Converted prior must contain {required}; got columns={list(df.columns)}")

    q = df[
        (df["dataset"].astype(str) == str(dataset))
        & (df["seed"].astype(int) == int(seed))
    ].copy()

    if "split" in q.columns:
        q2 = q[q["split"].astype(str) == str(split)]
        if len(q2) > 0:
            q = q2

    if len(q) == 0:
        raise RuntimeError(f"No matched prior rows for dataset={dataset}, split={split}, seed={seed}")

    seen = set()
    for _, r in q.iterrows():
        cid = int(r["class_id"])
        if 0 <= cid < num_classes:
            alpha[cid] = float(r["alpha"])
            seen.add(cid)

    missing = sorted(set(range(num_classes)) - seen)
    if missing:
        raise RuntimeError(
            f"Prior missing class ids for dataset={dataset}, split={split}, seed={seed}: "
            f"{missing[:30]} total_missing={len(missing)}"
        )

    print(
        f"[prior] loaded true EMRC prior: dataset={dataset} split={split} seed={seed} "
        f"rows={len(q)} alpha_mean={alpha.mean():.6f} alpha_min={alpha.min():.6f} alpha_max={alpha.max():.6f}"
    )

    return torch.tensor(build_weights_from_alpha(alpha), dtype=torch.float32)


def route_scaled_logits(scaled_logits, weights_c_b):
    """
    scaled_logits: [N, B, C]
    weights_c_b: [C, B]
    return: [N, C]
    """
    return (scaled_logits.permute(0, 2, 1) * weights_c_b[None, :, :]).sum(dim=-1)


def interpolate_weights(prior_w, beta):
    """
    prior_w: [C, B]
    beta=0 -> uniform NLC mean
    beta=1 -> full EMRC class-wise routing
    """
    B = prior_w.shape[1]
    uniform = torch.full_like(prior_w, 1.0 / B)
    return (1.0 - beta) * uniform + beta * prior_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/meta_prompt_1")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--emrc_prior", required=True)
    ap.add_argument("--fallback_alpha", type=float, default=0.4)
    ap.add_argument("--betas", default="0.25,0.50,0.75,1.00")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(resolve(args.root, args.manifest))
    row = select_manifest_row(df, args.dataset, args.seed)

    dataset = str(row.get("dataset", args.dataset))
    split = str(row.get("split", "all"))
    seed = int(args.seed)

    logits_np, features_np, labels_np, meta = load_manifest_row(args.root, row)
    train_idx, test_idx = sample_fewshot(labels_np, args.shots, args.seed)

    logits = torch.tensor(logits_np, dtype=torch.float32)
    features = torch.tensor(features_np, dtype=torch.float32)
    labels = torch.tensor(labels_np, dtype=torch.long)

    train_logits = logits[train_idx].to(device)
    train_features = features[train_idx].to(device)
    train_y = labels[train_idx].to(device)

    test_logits = logits[test_idx].to(device)
    test_features = features[test_idx].to(device)
    test_y = labels[test_idx].to(device)

    C = logits.shape[-1]
    prior_w = load_converted_prior(
        path=resolve(args.root, args.emrc_prior),
        num_classes=C,
        dataset=dataset,
        split=split,
        seed=seed,
        fallback_alpha=args.fallback_alpha,
    ).to(device)

    model = NeuralLogitController(
        input_dim=train_features.shape[1],
        hidden_dim=args.hidden,
        num_backbones=len(BACKBONES),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Strict NLC training objective: equal mean of temperature-scaled logits.
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        final_logits, _, _ = nlc_scaled_logits(train_logits, train_features, model)
        loss = F.cross_entropy(final_logits, train_y)
        loss.backward()
        optimizer.step()

    rows = []

    model.eval()
    with torch.no_grad():
        for i, b in enumerate(BACKBONES):
            rows.append({
                "dataset": args.dataset,
                "seed": args.seed,
                "shots": args.shots,
                "method": f"{b}_only",
                "acc": accuracy(test_logits[:, i, :], test_y),
            })

        log_avg = test_logits.mean(dim=1)
        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": "log_avg",
            "acc": accuracy(log_avg, test_y),
        })

        nlc_logits, scaled_logits, temps = nlc_scaled_logits(test_logits, test_features, model)
        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": "nlc_original",
            "acc": accuracy(nlc_logits, test_y),
        })

        # Raw EMRC logit routing without NLC temperature calibration.
        raw_route = route_scaled_logits(test_logits, prior_w)
        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": "emrc_raw_logit_route_beta1.00",
            "acc": accuracy(raw_route, test_y),
        })

        # NLC calibrated logits + EMRC class-wise routing.
        for beta in [float(x) for x in args.betas.split(",") if x.strip()]:
            w_beta = interpolate_weights(prior_w, beta)
            routed_logits = route_scaled_logits(scaled_logits, w_beta)
            rows.append({
                "dataset": args.dataset,
                "seed": args.seed,
                "shots": args.shots,
                "method": f"nlc_emrc_logit_route_beta{beta:.2f}",
                "acc": accuracy(routed_logits, test_y),
            })

    out_path = resolve(args.root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    print("Used feature keys:")
    for b, m in meta.items():
        print(b, m)
    print(out_df.to_string(index=False))
    print("[saved]", out_path)


if __name__ == "__main__":
    main()
