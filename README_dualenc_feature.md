# CoOpDualEnc Feature Fusion Experiment

This package adds a new trainer:

- `CoOpDualEnc`: ViT-B/16 anchored feature-level residual fusion with RN101 auxiliary image features.

Core formulation:

```text
v_fused = normalize(v_vit + alpha * MLP([v_vit, v_rn]))
```

The final MLP layer is zero-initialized, so the feature fusion module is identity at initialization and starts exactly from ViT-B/16 CoOp image-text alignment.

## Install on A100

```bash
cd /home/ubuntu/code/meta_prompt_1
cp -r scripts/dualenc_feature /home/ubuntu/code/meta_prompt_1/scripts/
bash scripts/dualenc_feature/00_install_dualenc_feature.sh
```

## Run B2N sanity

```bash
cd /home/ubuntu/code/meta_prompt_1
GPU=0 DATA_ROOT=/home/ubuntu/datasets \
DATASETS="dtd eurosat oxford_pets" \
SEEDS="1 2 3" \
bash scripts/dualenc_feature/01_run_dualenc_feature_b2n_sanity.sh \
  2>&1 | tee logs_dualenc_feature_b2n_sanity.txt
```

## Check summary

```bash
cat summary_tables/dualenc_feature/dualenc_feature_b2n_sanity.md
```

## Environment knobs

```bash
DUALENC_AUX_BACKBONE=RN101
DUALENC_HIDDEN_DIM=512
DUALENC_DROPOUT=0.0
DUALENC_ALPHA_INIT=1.0
DUALENC_USE_TEXT_ADAPTER=0
```

First version defaults to image-side fusion only. Keep text adapter disabled until image fusion is validated.
