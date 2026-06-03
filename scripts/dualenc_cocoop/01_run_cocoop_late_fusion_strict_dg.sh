#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
source "$ROOT/scripts/dualenc_cocoop/_cocoop_late_fusion_common.sh"
SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"imagenetv2 imagenet_sketch imagenet_a imagenet_r"}
TASK_DIR="$ROOT/outputs/dualenc_cocoop_late_fusion/strict_dg/source_${SOURCE}"
SUMMARY_OUT="$ROOT/summary_tables/dualenc_cocoop/cocoop_late_fusion_strict_dg_source_${SOURCE}.md"
mkdir -p "$TASK_DIR" "$ROOT/summary_tables/dualenc_cocoop"
ensure_configs
for SEED in $SEEDS_STR; do
  RN_TRAIN_DIR="$COOP_ROOT/output_dualenc_cocoop/train/source_${SOURCE}/shots_${SHOTS}/${TRAINER}/${RN_CFG_TAG}/nctx${NCTX}_ctx${CTX_INIT}_all/seed${SEED}"
  VIT_TRAIN_DIR="$COOP_ROOT/output_dualenc_cocoop/train/source_${SOURCE}/shots_${SHOTS}/${TRAINER}/${VIT_CFG_TAG}/nctx${NCTX}_ctx${CTX_INIT}_all/seed${SEED}"
  train_one_cocoop "rn101" "$RN_CFG" "$RN_CFG_TAG" "$SOURCE" "$SEED" "all" "$RN_TRAIN_DIR"
  train_one_cocoop "vit_b16" "$VIT_CFG" "$VIT_CFG_TAG" "$SOURCE" "$SEED" "all" "$VIT_TRAIN_DIR"
  for TARGET in $TARGETS_STR; do
    eval_fusion_one "$SOURCE" "$TARGET" "$SEED" "all" "$RN_TRAIN_DIR" "$VIT_TRAIN_DIR" "$TASK_DIR/${TARGET}/seed${SEED}"
  done
done
python "$ROOT/scripts/dualenc_cocoop/summarize_late_fusion_accuracy.py" --root "$TASK_DIR" --output "$SUMMARY_OUT" --title "CoCoOp Dual-Backbone Late Fusion Strict DG Summary"
echo "[DONE] $SUMMARY_OUT"
