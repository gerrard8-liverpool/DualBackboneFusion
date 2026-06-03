# CoCoOp Dual-Backbone Late Fusion Scripts

This package extends the existing CoOp dual-backbone late-fusion diagnostic to CoCoOp.

Fusion definition:

```text
fused = w * logits_vit + (1 - w) * logits_rn
```

Default protocol:
- trainer: CoCoOp
- backbones: RN101 and ViT-B/16
- context length: `NCTX=4`
- context initialization: `CTX_INIT=a_photo_of_a`
- load epoch: `LOAD_EPOCH=10`
- shots: 16
- weights: 0, 0.25, 0.5, 0.75, 1.0

## Install

```bash
cd /home/ubuntu/code/meta_prompt_1
unzip -o dualenc_cocoop_code.zip
chmod +x scripts/dualenc_cocoop/*.sh scripts/dualenc_cocoop/*.py
bash scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh
```

## Strict DG

```bash
GPU=0 DATA_ROOT=/home/ubuntu/datasets \
bash scripts/dualenc_cocoop/01_run_cocoop_late_fusion_strict_dg.sh \
  2>&1 | tee logs_cocoop_late_fusion_strict_dg.txt
```

## ImageNet-source cross-dataset

```bash
GPU=0 DATA_ROOT=/home/ubuntu/datasets \
bash scripts/dualenc_cocoop/02_run_cocoop_late_fusion_xd_imagenet_source.sh \
  2>&1 | tee logs_cocoop_late_fusion_xd_imagenet_source.txt
```

## B2N sanity

```bash
GPU=0 DATA_ROOT=/home/ubuntu/datasets \
DATASETS="dtd eurosat oxford_pets" \
bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh \
  2>&1 | tee logs_cocoop_late_fusion_b2n_sanity.txt
```

## Full B2N

```bash
GPU=0 DATA_ROOT=/home/ubuntu/datasets \
DATASETS="imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101" \
bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh \
  2>&1 | tee logs_cocoop_late_fusion_b2n_full.txt
```

## Outputs

```text
outputs/dualenc_cocoop_late_fusion/
summary_tables/dualenc_cocoop/
third_party/CoOp_clean/output_dualenc_cocoop/
```
