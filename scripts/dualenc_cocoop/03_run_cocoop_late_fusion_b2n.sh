#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
source "$ROOT/scripts/dualenc_cocoop/_cocoop_late_fusion_common.sh"
DATASETS_STR=${DATASETS:-"dtd eurosat oxford_pets"}
SEEDS_STR=${SEEDS:-"1 2 3"}
SPLITS_STR=${SPLITS:-"base new all"}
TASK_DIR="$ROOT/outputs/dualenc_cocoop_late_fusion/b2n"
SUMMARY_OUT="$ROOT/summary_tables/dualenc_cocoop/cocoop_late_fusion_b2n.md"
mkdir -p "$TASK_DIR" "$ROOT/summary_tables/dualenc_cocoop"
ensure_configs
for DATASET in $DATASETS_STR; do
  for SEED in $SEEDS_STR; do
    RN_TRAIN_DIR="$COOP_ROOT/output_dualenc_cocoop/b2n/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${RN_CFG_TAG}/nctx${NCTX}_ctx${CTX_INIT}_base/seed${SEED}"
    VIT_TRAIN_DIR="$COOP_ROOT/output_dualenc_cocoop/b2n/train/${DATASET}/shots_${SHOTS}/${TRAINER}/${VIT_CFG_TAG}/nctx${NCTX}_ctx${CTX_INIT}_base/seed${SEED}"
    train_one_cocoop "rn101" "$RN_CFG" "$RN_CFG_TAG" "$DATASET" "$SEED" "base" "$RN_TRAIN_DIR"
    train_one_cocoop "vit_b16" "$VIT_CFG" "$VIT_CFG_TAG" "$DATASET" "$SEED" "base" "$VIT_TRAIN_DIR"
    for SPLIT in $SPLITS_STR; do
      eval_fusion_one "$DATASET" "$DATASET" "$SEED" "$SPLIT" "$RN_TRAIN_DIR" "$VIT_TRAIN_DIR" "$TASK_DIR/${DATASET}/split_${SPLIT}/shots_${SHOTS}/nctx${NCTX}_ctx${CTX_INIT}/seed${SEED}"
    done
  done
done
python "$ROOT/scripts/dualenc_cocoop/summarize_late_fusion_b2n.py" --root "$TASK_DIR" --output "$SUMMARY_OUT" --title "CoCoOp Dual-Backbone Late Fusion B2N Summary"
echo "[DONE] $SUMMARY_OUT"
