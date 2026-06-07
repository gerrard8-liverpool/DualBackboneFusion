# Reliability Prior Cache (RPC) scripts

This directory implements the next-stage ImageNet Cache and Meta Cache experiments for dual-backbone late fusion.

## What this patch does

1. Standardize existing logits cache files into paired ViT/RN `.npz` files.
2. Build a class-level reliability cache from source logits.
3. Evaluate ImageNet Cache on B2N sanity and strict DG targets.
4. Evaluate Leave-one-dataset-out Meta Cache on B2N targets.

The scripts support both existing combined dual-logits files and separate per-branch files.

## Main entry points

```bash
bash scripts/dualenc_rpc/run_rpc_00_standardize_logits.sh
bash scripts/dualenc_rpc/run_rpc_01_imagenet_cache_b2n_sanity.sh
bash scripts/dualenc_rpc/run_rpc_02_meta_lodo_b2n.sh
bash scripts/dualenc_rpc/run_rpc_03_imagenet_cache_strict_dg.sh
```

## Important limitation

ImageNet Cache needs ImageNet source logits. If the standardized manifest does not contain rows with `dataset=imagenet`, the ImageNet Cache scripts will stop with `No source rows matched filters`. In that case, first collect RN101 and ViT-B/16 logits on ImageNet validation or an ImageNet train subset.


## v4 note
This patch fixes branch inference for the existing cache layout `logits_rn.npz` / `logits_vit.npz`.
