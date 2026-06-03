#!/usr/bin/env bash
set -euo pipefail

# Evaluate post-hoc late fusion on strict DG targets.
# Must run after 01_prepare_late_fusion_strict_dg.sh has prepared both checkpoints.
#
# Usage:
#   GPU=0 bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh
#
# Fusion convention inside eval_late_fusion_logits.py:
#   fused = w * logits_vit + (1-w) * logits_rn

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
SOURCE=${SOURCE:-imagenet}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"imagenetv2 imagenet_sketch imagenet_a imagenet_r"}
WEIGHTS_STR=${WEIGHTS:-"0.0 0.25 0.5 0.75 1.0"}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-128}

RN_CFG_TAG="rn101_ep50"
VIT_CFG_TAG="vit_b16_ep50"
RN_CFG="$COOP_ROOT/configs/trainers/CoOp/${RN_CFG_TAG}.yaml"
VIT_CFG="$COOP_ROOT/configs/trainers/CoOp/${VIT_CFG_TAG}.yaml"
COMMON_TAIL="nctx${NCTX}_csc${CSC}_ctp${CTX_POS}"

cd "$ROOT"

read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a TARGET_ARR <<< "$TARGETS_STR"
read -r -a WEIGHT_ARR <<< "$WEIGHTS_STR"

export TEXT_BATCH_SIZE

for TARGET in "${TARGET_ARR[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
  if [ ! -f "$TARGET_CFG" ]; then
    echo "[ERROR] Missing target dataset config: $TARGET_CFG"
    exit 1
  fi

  for SEED in "${SEED_ARR[@]}"; do
    RN_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${RN_CFG_TAG}/${COMMON_TAIL}/seed${SEED}"
    VIT_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${VIT_CFG_TAG}/${COMMON_TAIL}/seed${SEED}"
    OUT_DIR="$ROOT/outputs/dualenc/late_fusion/strict_dg/source_${SOURCE}/target_${TARGET}/shots_${SHOTS}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"

    RN_CKPT="$RN_MODEL_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
    VIT_CKPT="$VIT_MODEL_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
    if [ ! -f "$RN_CKPT" ]; then
      echo "[ERROR] Missing RN101 checkpoint: $RN_CKPT"
      echo "Run: GPU=$GPU bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh"
      exit 1
    fi
    if [ ! -f "$VIT_CKPT" ]; then
      echo "[ERROR] Missing ViT-B/16 checkpoint: $VIT_CKPT"
      echo "Run: GPU=$GPU bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh"
      exit 1
    fi

    if [ -f "$OUT_DIR/results.json" ]; then
      echo "[SKIP] existing $OUT_DIR/results.json"
      continue
    fi

    echo "============================================================"
    echo "[LATE FUSION STRICT DG] ${SOURCE}->${TARGET} seed=${SEED}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU" python scripts/dualenc/eval_late_fusion_logits.py \
      --project-root "$ROOT" \
      --coop-root "$COOP_ROOT" \
      --data-root "$DATA_ROOT" \
      --dataset-config-file "$TARGET_CFG" \
      --rn-config "$RN_CFG" \
      --vit-config "$VIT_CFG" \
      --rn-model-dir "$RN_MODEL_DIR" \
      --vit-model-dir "$VIT_MODEL_DIR" \
      --output-dir "$OUT_DIR" \
      --source "$SOURCE" \
      --target "$TARGET" \
      --seed "$SEED" \
      --load-epoch "$LOAD_EPOCH" \
      --shots "$SHOTS" \
      --nctx "$NCTX" \
      --csc "$CSC" \
      --ctx-pos "$CTX_POS" \
      --subsample-classes all \
      --weights "${WEIGHT_ARR[@]}"
  done
done

python scripts/dualenc/summarize_late_fusion.py \
  --root "$ROOT/outputs/dualenc/late_fusion/strict_dg/source_${SOURCE}" \
  --output "$ROOT/summary_tables/dualenc/late_fusion_strict_dg_${SOURCE}.md"

echo "[DONE] summary: $ROOT/summary_tables/dualenc/late_fusion_strict_dg_${SOURCE}.md"
