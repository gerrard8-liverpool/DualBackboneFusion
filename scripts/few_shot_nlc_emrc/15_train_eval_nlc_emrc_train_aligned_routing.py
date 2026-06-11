#!/usr/bin/env python3
import argparse
import importlib.util
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


def load_base_module(root):
    path = os.path.join(root, "scripts/nlc_emrc/13_train_eval_nlc_emrc_logit_routing.py")
    spec = importlib.util.spec_from_file_location("nlc_logit_base", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(scores, labels):
    pred = scores.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100.0


def train_one_beta(base, beta, train_logits, train_features, train_y, input_dim, hidden, lr, weight_decay, epochs, prior_w, seed, device):
    set_seed(seed)

    model = base.NeuralLogitController(
        input_dim=input_dim,
        hidden_dim=hidden,
        num_backbones=len(base.BACKBONES),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    w_beta = base.interpolate_weights(prior_w, beta).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()

        _, scaled_logits, _ = base.nlc_scaled_logits(train_logits, train_features, model)
        routed_logits = base.route_scaled_logits(scaled_logits, w_beta)

        loss = F.cross_entropy(routed_logits, train_y)
        loss.backward()
        optimizer.step()

    return model


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
    ap.add_argument("--betas", default="0.00,0.02,0.05,0.10,0.20,0.30")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = args.root
    base = load_base_module(root)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(base.resolve(root, args.manifest))
    row = base.select_manifest_row(df, args.dataset, args.seed)

    dataset = str(row.get("dataset", args.dataset))
    split = str(row.get("split", "all"))
    seed = int(args.seed)

    logits_np, features_np, labels_np, meta = base.load_manifest_row(root, row)
    train_idx, test_idx = base.sample_fewshot(labels_np, args.shots, args.seed)

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

    prior_w = base.load_converted_prior(
        path=base.resolve(root, args.emrc_prior),
        num_classes=C,
        dataset=dataset,
        split=split,
        seed=seed,
        fallback_alpha=args.fallback_alpha,
    ).to(device)

    rows = []

    # Raw frozen baselines
    with torch.no_grad():
        for i, b in enumerate(base.BACKBONES):
            rows.append({
                "dataset": args.dataset,
                "seed": args.seed,
                "shots": args.shots,
                "method": f"{b}_only",
                "acc": accuracy(test_logits[:, i, :], test_y),
            })

        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": "log_avg",
            "acc": accuracy(test_logits.mean(dim=1), test_y),
        })

        raw_route = base.route_scaled_logits(test_logits, prior_w)
        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": "emrc_raw_logit_route_beta1.00",
            "acc": accuracy(raw_route, test_y),
        })

    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    for beta in betas:
        model = train_one_beta(
            base=base,
            beta=beta,
            train_logits=train_logits,
            train_features=train_features,
            train_y=train_y,
            input_dim=train_features.shape[1],
            hidden=args.hidden,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            prior_w=prior_w,
            seed=args.seed,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            _, scaled_logits, _ = base.nlc_scaled_logits(test_logits, test_features, model)
            w_beta = base.interpolate_weights(prior_w, beta).to(device)
            routed_logits = base.route_scaled_logits(scaled_logits, w_beta)
            acc = accuracy(routed_logits, test_y)

        if abs(beta) < 1e-12:
            method = "nlc_original_train_aligned_beta0.00"
        else:
            method = f"nlc_emrc_train_aligned_beta{beta:.2f}"

        rows.append({
            "dataset": args.dataset,
            "seed": args.seed,
            "shots": args.shots,
            "method": method,
            "acc": acc,
        })

    out_path = base.resolve(root, args.out)
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
