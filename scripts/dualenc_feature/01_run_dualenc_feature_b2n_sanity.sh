#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/datasets}
GPU=${GPU:-0}

DATASETS_STR=${DATASETS:-"dtd eurosat oxford_pets"}
SEEDS_STR=${SEEDS:-"1 2 3"}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}

TRAINER=${TRAINER:-CoOpDualEnc}
CFG_TAG=${CFG_TAG:-vit_b16_ep50}
TRAIN_CFG="$COOP_ROOT/configs/trainers/CoOp/${CFG_TAG}.yaml"

# DualEnc hyperparameters are passed through environment variables.
export DUALENC_AUX_BACKBONE=${DUALENC_AUX_BACKBONE:-RN101}
export DUALENC_HIDDEN_DIM=${DUALENC_HIDDEN_DIM:-512}
export DUALENC_DROPOUT=${DUALENC_DROPOUT:-0.0}
export DUALENC_ALPHA_INIT=${DUALENC_ALPHA_INIT:-1.0}
export DUALENC_USE_TEXT_ADAPTER=${DUALENC_USE_TEXT_ADAPTER:-0}
export TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-0}

OUT_BASE="$COOP_ROOT/output_dualenc_feature/b2n"
SUMMARY_OUT="$ROOT/summary_tables/dualenc_feature/dualenc_feature_b2n_sanity.md"
mkdir -p "$OUT_BASE" "$ROOT/summary_tables/dualenc_feature"

cd "$COOP_ROOT"

if [ ! -f "$TRAIN_CFG" ]; then
  echo "[ERROR] Missing train config: $TRAIN_CFG"
  echo "Run backbone config creation first if needed."
  exit 1
fi

echo "============================================================"
echo "[DUALENC FEATURE B2N]"
echo "DATASETS=${DATASETS_STR}"
echo "SEEDS=${SEEDS_STR}"
echo "TRAIN_CFG=${TRAIN_CFG}"
echo "AUX=${DUALENC_AUX_BACKBONE}"
echo "TEXT_ADAPTER=${DUALENC_USE_TEXT_ADAPTER}"
echo "============================================================"

for DATASET in $DATASETS_STR; do
  DATASET_CFG="$COOP_ROOT/configs/datasets/${DATASET}.yaml"
  if [ ! -f "$DATASET_CFG" ]; then
    echo "[ERROR] Missing dataset config: $DATASET_CFG"
    exit 1
  fi

  for SEED in $SEEDS_STR; do
    COMMON="${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_vit_anchor_rn101_aux/seed${SEED}"
    TRAIN_DIR="$OUT_BASE/train/${COMMON}"
    TRAIN_OK="$TRAIN_DIR/dualenc/model.pth.tar-${LOAD_EPOCH}"

    if [ -f "$TRAIN_OK" ]; then
      echo "[SKIP TRAIN] dataset=${DATASET} seed=${SEED}"
    else
      echo "============================================================"
      echo "[TRAIN DUALENC FEATURE] dataset=${DATASET} seed=${SEED} split=base"
      echo "out=${TRAIN_DIR}"
      echo "============================================================"
      CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --root "$DATA_ROOT" \
        --trainer "$TRAINER" \
        --dataset-config-file "$DATASET_CFG" \
        --config-file "$TRAIN_CFG" \
        --output-dir "$TRAIN_DIR" \
        --seed "$SEED" \
        TRAINER.COOP.N_CTX "$NCTX" \
        TRAINER.COOP.CSC "$CSC" \
        TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
        DATASET.NUM_SHOTS "$SHOTS" \
        DATASET.SUBSAMPLE_CLASSES base
    fi

    for SPLIT in base new all; do
      TEST_DIR="$OUT_BASE/test/${DATASET}/split_${SPLIT}/shots_${SHOTS}/${TRAINER}/${CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_vit_anchor_rn101_aux/seed${SEED}"
      if [ -f "$TEST_DIR/log.txt" ] && grep -q "accuracy:" "$TEST_DIR/log.txt"; then
        echo "[SKIP TEST] dataset=${DATASET} split=${SPLIT} seed=${SEED}"
        continue
      fi

      echo "============================================================"
      echo "[TEST DUALENC FEATURE] dataset=${DATASET} split=${SPLIT} seed=${SEED}"
      echo "model=${TRAIN_DIR}"
      echo "out=${TEST_DIR}"
      echo "============================================================"
      CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --root "$DATA_ROOT" \
        --trainer "$TRAINER" \
        --dataset-config-file "$DATASET_CFG" \
        --config-file "$TRAIN_CFG" \
        --output-dir "$TEST_DIR" \
        --model-dir "$TRAIN_DIR" \
        --load-epoch "$LOAD_EPOCH" \
        --eval-only \
        --seed "$SEED" \
        TRAINER.COOP.N_CTX "$NCTX" \
        TRAINER.COOP.CSC "$CSC" \
        TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
        DATASET.NUM_SHOTS "$SHOTS" \
        DATASET.SUBSAMPLE_CLASSES "$SPLIT"
    done
  done
done

cd "$ROOT"
python scripts/dualenc_feature/summarize_dualenc_feature_b2n.py \
  --root "$OUT_BASE/test" \
  --output "$SUMMARY_OUT"

echo "[DONE] $SUMMARY_OUT"
