#!/usr/bin/env bash
set -euo pipefail

# Train RN101/ViT-B/16 CoOp on base classes and evaluate late fusion on base/new/all.
# This is a small B2N sanity runner; use it after strict DG looks promising.
#
# Usage:
#   GPU=0 bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
#
# Optional env:
#   DATASETS="dtd eurosat oxford_pets"
#   SEEDS="1 2 3"

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}
SEEDS_STR=${SEEDS:-"1 2 3"}
DATASETS_STR=${DATASETS:-"dtd eurosat oxford_pets"}
WEIGHTS_STR=${WEIGHTS:-"0.0 0.25 0.5 0.75 1.0"}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-128}

RN_CFG_TAG="rn101_ep50"
VIT_CFG_TAG="vit_b16_ep50"
RN_CFG="$COOP_ROOT/configs/trainers/CoOp/${RN_CFG_TAG}.yaml"
VIT_CFG="$COOP_ROOT/configs/trainers/CoOp/${VIT_CFG_TAG}.yaml"
COMMON_TAIL="nctx${NCTX}_csc${CSC}_ctp${CTX_POS}"

cd "$ROOT"
bash scripts/backbone_dg/01_create_backbone_configs.sh

read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a DATASET_ARR <<< "$DATASETS_STR"
read -r -a WEIGHT_ARR <<< "$WEIGHTS_STR"

export TEXT_BATCH_SIZE

train_one() {
  local backbone_tag=$1
  local cfg_tag=$2
  local dataset=$3
  local seed=$4
  local train_cfg="configs/trainers/CoOp/${cfg_tag}.yaml"
  local dataset_cfg="$COOP_ROOT/configs/datasets/${dataset}.yaml"
  local out_dir="$COOP_ROOT/output_dualenc/b2n/train/${dataset}/shots_${SHOTS}/CoOp/${cfg_tag}/${COMMON_TAIL}_base/seed${seed}"
  local ckpt="$out_dir/prompt_learner/model.pth.tar-${LOAD_EPOCH}"

  if [ -f "$ckpt" ]; then
    echo "[SKIP TRAIN] B2N CoOp ${backbone_tag} dataset=${dataset} seed=${seed}"
    return
  fi

  echo "============================================================"
  echo "[TRAIN B2N BASE] backbone=${backbone_tag} dataset=${dataset} seed=${seed}"
  echo "============================================================"

  cd "$COOP_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --root "$DATA_ROOT" \
    --trainer CoOp \
    --dataset-config-file "$dataset_cfg" \
    --config-file "$train_cfg" \
    --output-dir "$out_dir" \
    --seed "$seed" \
    TRAINER.COOP.N_CTX "$NCTX" \
    TRAINER.COOP.CSC "$CSC" \
    TRAINER.COOP.CLASS_TOKEN_POSITION "$CTX_POS" \
    DATASET.NUM_SHOTS "$SHOTS" \
    DATASET.SUBSAMPLE_CLASSES base
  cd "$ROOT"
}

for DATASET in "${DATASET_ARR[@]}"; do
  DATASET_CFG="$COOP_ROOT/configs/datasets/${DATASET}.yaml"
  if [ ! -f "$DATASET_CFG" ]; then
    echo "[ERROR] Missing dataset config: $DATASET_CFG"
    exit 1
  fi

  for SEED in "${SEED_ARR[@]}"; do
    train_one rn101 "$RN_CFG_TAG" "$DATASET" "$SEED"
    train_one vit_b16 "$VIT_CFG_TAG" "$DATASET" "$SEED"

    RN_MODEL_DIR="$COOP_ROOT/output_dualenc/b2n/train/${DATASET}/shots_${SHOTS}/CoOp/${RN_CFG_TAG}/${COMMON_TAIL}_base/seed${SEED}"
    VIT_MODEL_DIR="$COOP_ROOT/output_dualenc/b2n/train/${DATASET}/shots_${SHOTS}/CoOp/${VIT_CFG_TAG}/${COMMON_TAIL}_base/seed${SEED}"

    for SPLIT in base new all; do
      OUT_DIR="$ROOT/outputs/dualenc/late_fusion/b2n/${DATASET}/split_${SPLIT}/shots_${SHOTS}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}"
      if [ -f "$OUT_DIR/results.json" ]; then
        echo "[SKIP] existing $OUT_DIR/results.json"
        continue
      fi

      echo "============================================================"
      echo "[LATE FUSION B2N] dataset=${DATASET} split=${SPLIT} seed=${SEED}"
      echo "============================================================"

      CUDA_VISIBLE_DEVICES="$GPU" python scripts/dualenc/eval_late_fusion_logits.py \
        --project-root "$ROOT" \
        --coop-root "$COOP_ROOT" \
        --data-root "$DATA_ROOT" \
        --dataset-config-file "$DATASET_CFG" \
        --rn-config "$RN_CFG" \
        --vit-config "$VIT_CFG" \
        --rn-model-dir "$RN_MODEL_DIR" \
        --vit-model-dir "$VIT_MODEL_DIR" \
        --output-dir "$OUT_DIR" \
        --source "$DATASET" \
        --target "$DATASET" \
        --seed "$SEED" \
        --load-epoch "$LOAD_EPOCH" \
        --shots "$SHOTS" \
        --nctx "$NCTX" \
        --csc "$CSC" \
        --ctx-pos "$CTX_POS" \
        --subsample-classes "$SPLIT" \
        --weights "${WEIGHT_ARR[@]}"
    done
  done
done

python scripts/dualenc/summarize_late_fusion.py \
  --root "$ROOT/outputs/dualenc/late_fusion/b2n" \
  --output "$ROOT/summary_tables/dualenc/late_fusion_b2n_sanity.md" \
  --b2n

echo "[DONE] summary: $ROOT/summary_tables/dualenc/late_fusion_b2n_sanity.md"
