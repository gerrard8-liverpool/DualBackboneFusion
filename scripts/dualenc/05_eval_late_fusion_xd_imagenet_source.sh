#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/datasets}
GPU=${GPU:-0}

SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"caltech101 oxford_pets dtd eurosat food101 oxford_flowers stanford_cars fgvc_aircraft ucf101 sun397"}

SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}

RN_CFG_TAG="rn101_ep50"
VIT_CFG_TAG="vit_b16_ep50"

RN_CFG="$COOP_ROOT/configs/trainers/CoOp/${RN_CFG_TAG}.yaml"
VIT_CFG="$COOP_ROOT/configs/trainers/CoOp/${VIT_CFG_TAG}.yaml"

OUT_ROOT="$ROOT/outputs/dualenc/late_fusion/xd/source_${SOURCE}"
SUMMARY_OUT="$ROOT/summary_tables/dualenc/late_fusion_xd_source_${SOURCE}.md"

mkdir -p "$OUT_ROOT" "$ROOT/summary_tables/dualenc"

echo "============================================================"
echo "[LATE FUSION XD] source=${SOURCE}"
echo "TARGETS=${TARGETS_STR}"
echo "SEEDS=${SEEDS_STR}"
echo "GPU=${GPU}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "============================================================"

for SEED in $SEEDS_STR; do
  RN_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${RN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"
  VIT_MODEL_DIR="$COOP_ROOT/output/xd/train/source_${SOURCE}/shots_${SHOTS}/CoOp/${VIT_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"

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

  for TARGET in $TARGETS_STR; do
    TARGET_CFG="$COOP_ROOT/configs/datasets/${TARGET}.yaml"
    OUT_DIR="$OUT_ROOT/${TARGET}/seed${SEED}"

    if [ ! -f "$TARGET_CFG" ]; then
      echo "[ERROR] Missing target config: $TARGET_CFG"
      exit 1
    fi

    if [ -f "$OUT_DIR/results.json" ]; then
      echo "[SKIP] ${SOURCE}->${TARGET} seed=${SEED}: $OUT_DIR/results.json"
      continue
    fi

    echo "============================================================"
    echo "[LATE FUSION XD] ${SOURCE}->${TARGET} seed=${SEED}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/dualenc/eval_late_fusion_logits.py \
      --project-root "$ROOT" \
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
      --subsample-classes all
  done
done

python scripts/dualenc/summarize_late_fusion_xd.py \
  --root "$OUT_ROOT" \
  --output "$SUMMARY_OUT"

echo "[DONE] $SUMMARY_OUT"
