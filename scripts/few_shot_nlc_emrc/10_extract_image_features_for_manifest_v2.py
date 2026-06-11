#!/usr/bin/env python3
import argparse
import os
import sys
import json
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from tqdm import tqdm


BACKBONES = ["rn50", "rn101", "vit_b32", "vit_b16"]
PATH_COLS = {
    "rn50": "rn50_path",
    "rn101": "rn101_path",
    "vit_b32": "vit_b32_path",
    "vit_b16": "vit_b16_path",
}
CLIP_MODEL_NAMES = {
    "rn50": "RN50",
    "rn101": "RN101",
    "vit_b32": "ViT-B/32",
    "vit_b16": "ViT-B/16",
}


def parse_csv_list(x):
    if x is None or str(x).strip() == "":
        return None
    return [t.strip() for t in str(x).split(",") if t.strip()]


def is_valid_path_value(x):
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s not in ["", "nan", "None", "NONE", "null", "NULL"]


def resolve_path(root, p):
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(root, p)


def resolve_image_path(root, dataset_root, p):
    p = str(p)
    candidates = []
    if os.path.isabs(p):
        candidates.append(p)
    else:
        candidates.extend([
            os.path.join(root, p),
            os.path.join(dataset_root, p),
            p,
        ])
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Cannot resolve image path: {p}; tried={candidates}")


def load_clip(root, backbone, device):
    sys.path.insert(0, os.path.join(root, "third_party/CoOp_clean"))
    import clip

    model_name = CLIP_MODEL_NAMES[backbone]
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def encode_images(model, preprocess, image_paths, root, dataset_root, batch_size, device, feature_mode):
    raw_feats = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="encode", leave=False):
            batch_paths = image_paths[start:start + batch_size]
            imgs = []

            for p in batch_paths:
                rp = resolve_image_path(root, dataset_root, p)
                img = Image.open(rp).convert("RGB")
                imgs.append(preprocess(img))

            images = torch.stack(imgs, dim=0).to(device)

            feats = model.encode_image(images)
            feats = feats.float()
            raw_feats.append(feats.detach().cpu())

    raw_feats = torch.cat(raw_feats, dim=0)
    norm_feats = raw_feats / raw_feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    raw_np = raw_feats.numpy().astype("float32")
    norm_np = norm_feats.numpy().astype("float32")

    if feature_mode == "normalized":
        main_np = norm_np
    elif feature_mode == "raw":
        main_np = raw_np
    else:
        raise ValueError(feature_mode)

    return main_np, raw_np


def save_augmented_npz(src_npz_path, out_npz_path, image_features, image_features_raw, extra_meta):
    z = np.load(src_npz_path, allow_pickle=True)

    payload = {}
    for k in z.keys():
        payload[k] = z[k]

    payload["image_features"] = image_features
    payload["image_features_raw"] = image_features_raw
    payload["nlc_feature_meta_json"] = np.array(json.dumps(extra_meta, ensure_ascii=False), dtype=object)

    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(out_npz_path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/meta_prompt_1")
    ap.add_argument("--dataset-root", default="/workspace/datasets")
    ap.add_argument("--manifest", default="outputs/zeroshot_clip_logits/four_backbone_manifest_zeroshot_all.csv")
    ap.add_argument("--out-manifest", default="outputs/nlc_emrc/manifests/four_backbone_manifest_with_image_features.csv")
    ap.add_argument("--out-root", default="outputs/nlc_emrc/image_feature_cache")
    ap.add_argument("--protocols", default="")
    ap.add_argument("--datasets", default="")
    ap.add_argument("--splits", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--backbones", default="rn50,rn101,vit_b32,vit_b16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-mode", choices=["normalized", "raw"], default="normalized")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.root
    manifest_path = resolve_path(root, args.manifest)
    df = pd.read_csv(manifest_path)

    protocols = parse_csv_list(args.protocols)
    datasets = parse_csv_list(args.datasets)
    splits = parse_csv_list(args.splits)
    seeds = parse_csv_list(args.seeds)
    backbones = parse_csv_list(args.backbones)

    if protocols and "protocol" in df.columns:
        df = df[df["protocol"].astype(str).isin(protocols)].copy()
    if datasets and "dataset" in df.columns:
        df = df[df["dataset"].astype(str).isin(datasets)].copy()
    if splits and "split" in df.columns:
        df = df[df["split"].astype(str).isin(splits)].copy()
    if seeds and "seed" in df.columns:
        seed_ints = [int(s) for s in seeds]
        df = df[df["seed"].astype(int).isin(seed_ints)].copy()

    needed_cols = [PATH_COLS[b] for b in backbones]
    before = len(df)

    valid_mask = np.ones(len(df), dtype=bool)
    for col in needed_cols:
        valid_mask &= df[col].map(is_valid_path_value).values

    df = df[valid_mask].copy()
    dropped = before - len(df)

    # 进一步过滤本地源 npz 不存在的 row
    exist_mask = np.ones(len(df), dtype=bool)
    for i, (_, row) in enumerate(df.iterrows()):
        for col in needed_cols:
            src = resolve_path(root, row[col])
            if not os.path.exists(src):
                exist_mask[i] = False
                break

    df = df[exist_mask].copy()

    print("[selected rows after filters]", len(df))
    print("[dropped invalid path rows]", dropped)
    if len(df) == 0:
        raise RuntimeError("No valid rows selected after filtering.")

    print(df[["prompt_learner", "protocol", "dataset", "split", "seed"]].to_string(index=False))

    out_df = df.copy()

    for backbone in backbones:
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}")

        col = PATH_COLS[backbone]

        print("\n" + "=" * 80)
        print("[backbone]", backbone, "model=", CLIP_MODEL_NAMES[backbone])
        print("=" * 80)

        model, preprocess = load_clip(root, backbone, args.device)

        for idx, row in df.iterrows():
            src_rel = str(row[col])
            src_path = resolve_path(root, src_rel)

            out_rel = os.path.join(args.out_root, src_rel)
            out_path = resolve_path(root, out_rel)
            out_df.loc[idx, col] = out_rel

            if os.path.exists(out_path) and not args.overwrite:
                print("[skip exists]", out_rel)
                continue

            z = np.load(src_path, allow_pickle=True)
            if "image_paths" not in z.keys():
                raise RuntimeError(f"No image_paths in {src_path}")

            image_paths = [str(x) for x in np.asarray(z["image_paths"]).reshape(-1).tolist()]
            print(
                f"[encode] protocol={row['protocol']} dataset={row['dataset']} "
                f"split={row['split']} seed={row['seed']} n={len(image_paths)} -> {out_rel}"
            )

            image_features, image_features_raw = encode_images(
                model=model,
                preprocess=preprocess,
                image_paths=image_paths,
                root=root,
                dataset_root=args.dataset_root,
                batch_size=args.batch_size,
                device=args.device,
                feature_mode=args.feature_mode,
            )

            meta = {
                "feature_extractor": "OpenAI CLIP via third_party/CoOp_clean/clip",
                "backbone": backbone,
                "clip_model_name": CLIP_MODEL_NAMES[backbone],
                "feature_mode_for_image_features": args.feature_mode,
                "image_features_key": "image_features",
                "image_features_raw_key": "image_features_raw",
                "n": int(image_features.shape[0]),
                "dim": int(image_features.shape[1]),
            }

            save_augmented_npz(src_path, out_path, image_features, image_features_raw, meta)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_manifest = resolve_path(root, args.out_manifest)
    os.makedirs(os.path.dirname(out_manifest), exist_ok=True)
    out_df.to_csv(out_manifest, index=False)

    print("\n[saved manifest]", out_manifest)
    print(out_df[["prompt_learner", "protocol", "dataset", "split", "seed"] + needed_cols].to_string(index=False))


if __name__ == "__main__":
    main()
