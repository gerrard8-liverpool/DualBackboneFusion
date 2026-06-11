#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd


PATH_COLS = ["rn50_path", "rn101_path", "vit_b32_path", "vit_b16_path"]


def resolve(root, p):
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def norm_name(s):
    return str(s).strip().lower().replace("_", " ").replace("-", " ")


def load_class_names(root, row):
    # 任意 backbone npz 都有 class_names，取 rn50 即可
    p = resolve(root, row["rn50_path"])
    z = np.load(p, allow_pickle=True)
    if "class_names" not in z.keys():
        raise RuntimeError(f"No class_names in {p}")
    return [str(x) for x in np.asarray(z["class_names"]).reshape(-1).tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/meta_prompt_1")
    ap.add_argument("--feature_manifest", required=True)
    ap.add_argument("--routing_csv", required=True)
    ap.add_argument("--alpha_col", default="alpha_meta_ensemble")
    ap.add_argument("--fallback_alpha_col", default="alpha_hier")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = args.root
    man = pd.read_csv(resolve(root, args.feature_manifest))
    routing = pd.read_csv(resolve(root, args.routing_csv))

    required = ["dataset", "split", "seed", "class_name"]
    for c in required:
        if c not in routing.columns:
            raise RuntimeError(f"routing csv missing column {c}: {args.routing_csv}")

    if args.alpha_col not in routing.columns:
        if args.fallback_alpha_col in routing.columns:
            print(f"[warn] alpha_col={args.alpha_col} missing; use fallback {args.fallback_alpha_col}")
            alpha_col = args.fallback_alpha_col
        else:
            raise RuntimeError(
                f"Neither alpha_col={args.alpha_col} nor fallback={args.fallback_alpha_col} exists. "
                f"columns={list(routing.columns)}"
            )
    else:
        alpha_col = args.alpha_col

    rows = []
    misses = []

    for _, mrow in man.iterrows():
        dataset = str(mrow["dataset"])
        split = str(mrow.get("split", "all"))
        seed = int(mrow["seed"])

        class_names = load_class_names(root, mrow)

        rsub = routing[
            (routing["dataset"].astype(str) == dataset)
            & (routing["split"].astype(str) == split)
            & (routing["seed"].astype(int) == seed)
        ].copy()

        if len(rsub) == 0:
            # 有些 routing 可能 split 固定 all，兜底用 all
            rsub = routing[
                (routing["dataset"].astype(str) == dataset)
                & (routing["split"].astype(str) == "all")
                & (routing["seed"].astype(int) == seed)
            ].copy()

        mapping = {
            norm_name(r["class_name"]): float(r[alpha_col])
            for _, r in rsub.iterrows()
        }

        for cid, cname in enumerate(class_names):
            key = norm_name(cname)
            if key not in mapping:
                # 再尝试原始字符串匹配
                candidates = [k for k in mapping if k.replace(" ", "") == key.replace(" ", "")]
                if candidates:
                    alpha = mapping[candidates[0]]
                else:
                    alpha = 0.4
                    misses.append((dataset, split, seed, cid, cname))
            else:
                alpha = mapping[key]

            rows.append({
                "dataset": dataset,
                "split": split,
                "seed": seed,
                "class_id": cid,
                "class_name": cname,
                "alpha": alpha,
                "source_alpha_col": alpha_col,
            })

    out = resolve(root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print("[saved]", out)
    print("[rows]", len(rows))
    print("[misses]", len(misses))
    if misses:
        print("[first 30 misses]")
        for x in misses[:30]:
            print(x)

    df = pd.DataFrame(rows)
    print(df.groupby(["dataset", "seed"]).size().head(20).to_string())
    print(df["alpha"].describe().to_string())


if __name__ == "__main__":
    main()
