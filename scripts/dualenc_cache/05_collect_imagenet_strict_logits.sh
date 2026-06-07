#!/usr/bin/env bash
set -euo pipefail

# Collect per-branch logits for ImageNet-source strict DG targets.
# It loads one backbone at a time (RN101 then ViT-B/16) to keep GPU memory low.
# Default targets include ImageNet itself and four ImageNet generalization datasets:
#   imagenet, imagenetv2, imagenet_sketch, imagenet_a, imagenet_r
#
# Usage:
#   cd /home/ubuntu/code/meta_prompt_1
#   GPU=0 DATA_ROOT=/home/ubuntu/datasets BATCH_SIZE=4 \
#     bash scripts/dualenc_cache/05_collect_imagenet_strict_logits.sh
#
# Optional env:
#   ROOT=/home/ubuntu/code/meta_prompt_1
#   COOP_ROOT=$ROOT/third_party/CoOp_clean
#   DATA_ROOT=/home/ubuntu/datasets
#   GPU=0
#   SEEDS="1 2 3"
#   TARGETS="imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r"
#   BATCH_SIZE=4
#   NUM_WORKERS=2
#   LOAD_EPOCH=50

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/datasets}
GPU=${GPU:-0}
SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r"}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}

RN_CFG_TAG="rn101_ep50"
VIT_CFG_TAG="vit_b16_ep50"
RN_CFG="configs/trainers/CoOp/${RN_CFG_TAG}.yaml"
VIT_CFG="configs/trainers/CoOp/${VIT_CFG_TAG}.yaml"
COMMON_TAIL="nctx${NCTX}_csc${CSC}_ctp${CTX_POS}"

cd "$ROOT"

if [ ! -f "scripts/dualenc_cache/collect_branch_logits_coop.py" ]; then
  echo "[ERROR] Missing scripts/dualenc_cache/collect_branch_logits_coop.py"
  echo "Install dualenc_cache_code first."
  exit 1
fi

read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a TARGET_ARR <<< "$TARGETS_STR"

mkdir -p "$ROOT/outputs/dualenc_cache/logits/strict_dg/source_${SOURCE}"

for TARGET in "${TARGET_ARR[@]}"; do
  TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
  if [ ! -f "$TARGET_CFG" ]; then
    echo "[ERROR] Missing target dataset config: $TARGET_CFG"
    exit 1
  fi

  for SEED in "${SEED_ARR[@]}"; do
    RN_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${RN_CFG_TAG}/${COMMON_TAIL}/seed${SEED}"
    VIT_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${VIT_CFG_TAG}/${COMMON_TAIL}/seed${SEED}"

    RN_CKPT="$RN_MODEL_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
    VIT_CKPT="$VIT_MODEL_DIR/prompt_learner/model.pth.tar-${LOAD_EPOCH}"

    if [ ! -f "$RN_CKPT" ]; then
      echo "[ERROR] Missing RN101 checkpoint: $RN_CKPT"
      exit 1
    fi
    if [ ! -f "$VIT_CKPT" ]; then
      echo "[ERROR] Missing ViT-B/16 checkpoint: $VIT_CKPT"
      exit 1
    fi

    OUT_PREFIX="$ROOT/outputs/dualenc_cache/logits/strict_dg/source_${SOURCE}/target_${TARGET}/shots_${SHOTS}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}/logits"

    echo "============================================================"
    echo "[COLLECT STRICT LOGITS] source=${SOURCE} target=${TARGET} seed=${SEED}"
    echo "[GPU] CUDA_VISIBLE_DEVICES=${GPU}; batch_size=${BATCH_SIZE}; num_workers=${NUM_WORKERS}"
    echo "[OUT_PREFIX] ${OUT_PREFIX}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      python scripts/dualenc_cache/collect_branch_logits_coop.py \
        --project-root "$ROOT" \
        --data-root "$DATA_ROOT" \
        --dataset-config-file "$TARGET_CFG" \
        --config-file "$RN_CFG" \
        --model-dir "$RN_MODEL_DIR" \
        --output-prefix "$OUT_PREFIX" \
        --branch rn \
        --source "$SOURCE" \
        --target "$TARGET" \
        --seed "$SEED" \
        --subsample-classes all \
        --load-epoch "$LOAD_EPOCH" \
        --shots "$SHOTS" \
        --nctx "$NCTX" \
        --csc "$CSC" \
        --ctx-pos "$CTX_POS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"

    CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
      python scripts/dualenc_cache/collect_branch_logits_coop.py \
        --project-root "$ROOT" \
        --data-root "$DATA_ROOT" \
        --dataset-config-file "$TARGET_CFG" \
        --config-file "$VIT_CFG" \
        --model-dir "$VIT_MODEL_DIR" \
        --output-prefix "$OUT_PREFIX" \
        --branch vit \
        --source "$SOURCE" \
        --target "$TARGET" \
        --seed "$SEED" \
        --subsample-classes all \
        --load-epoch "$LOAD_EPOCH" \
        --shots "$SHOTS" \
        --nctx "$NCTX" \
        --csc "$CSC" \
        --ctx-pos "$CTX_POS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS"
  done
done

echo "[DONE] strict DG logits collected under:"
echo "  $ROOT/outputs/dualenc_cache/logits/strict_dg/source_${SOURCE}"
