# Dual-Encoder Late Fusion Code Pack

This pack adds a first-stage diagnostic for RN101 + ViT-B/16 CoOp complementarity.
It does not modify the existing CoOp trainer.

## Files

```text
scripts/dualenc/eval_late_fusion_logits.py
scripts/dualenc/summarize_late_fusion.py
scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
scripts/dualenc/02_eval_late_fusion_strict_dg.sh
scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
```

## Install into project

From your local machine or server after extracting this pack:

```bash
cp -r scripts/dualenc /workspace/meta_prompt_1/scripts/
cd /workspace/meta_prompt_1
chmod +x scripts/dualenc/*.sh scripts/dualenc/*.py
```

## Strict DG run on A100

```bash
cd /workspace/meta_prompt_1
GPU=0 DATA_ROOT=/workspace/datasets bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
GPU=0 DATA_ROOT=/workspace/datasets bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh
```

Default strict-DG targets:

```text
imagenetv2 imagenet_sketch imagenet_a imagenet_r
```

Summary output:

```text
summary_tables/dualenc/late_fusion_strict_dg_imagenet.md
```

## B2N sanity run

```bash
cd /workspace/meta_prompt_1
GPU=0 DATA_ROOT=/workspace/datasets bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
```

Default datasets:

```text
dtd eurosat oxford_pets
```

Summary output:

```text
summary_tables/dualenc/late_fusion_b2n_sanity.md
```

## Fusion convention

The fusion score is:

```text
fused = w * logits_vit + (1 - w) * logits_rn
```

So:

```text
w=0.0 -> pure RN101
w=1.0 -> pure ViT-B/16
w=0.5 -> equal fusion
```

The script reports three fusion modes:

```text
raw_logits: direct logit addition
std_logits: per-sample standardized logits before addition
prob_avg: softmax probability average
```

For fair reporting, prefer fixed weights such as `w=0.5`. The best-over-weight table is diagnostic only.
