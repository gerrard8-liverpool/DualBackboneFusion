#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/datasets}
GPU=${GPU:-0}
DATASETS=${DATASETS:-"dtd eurosat oxford_pets"}
SEEDS=${SEEDS:-"1"}
SPLITS=${SPLITS:-"base new all"}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}
LOAD_EPOCH=${LOAD_EPOCH:-50}
RN_CFG_TAG=${RN_CFG_TAG:-rn101_ep50}
VIT_CFG_TAG=${VIT_CFG_TAG:-vit_b16_ep50}
cd "$ROOT"
mkdir -p outputs/dualenc_cache/logits/b2n summary_tables/dualenc_cache

echo "[INFO] branch-by-branch logits collection; lower BATCH_SIZE to 4 if CUDA OOM."
for DATASET in $DATASETS; do
  DATASET_CFG="$ROOT/third_party/CoOp_clean/configs/datasets/${DATASET}.yaml"
  for SEED in $SEEDS; do
    for SPLIT in $SPLITS; do
      OUT_DIR="$ROOT/outputs/dualenc_cache/logits/b2n/${DATASET}/split_${SPLIT}/seed${SEED}"
      mkdir -p "$OUT_DIR"
      RN_MODEL_DIR="$ROOT/third_party/CoOp_clean/output_dualenc/b2n/train/${DATASET}/shots_${SHOTS}/CoOp/${RN_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_base/seed${SEED}"
      VIT_MODEL_DIR="$ROOT/third_party/CoOp_clean/output_dualenc/b2n/train/${DATASET}/shots_${SHOTS}/CoOp/${VIT_CFG_TAG}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}_base/seed${SEED}"
      for BRANCH in rn vit; do
        if [ "$BRANCH" = "rn" ]; then CFG_TAG="$RN_CFG_TAG"; MODEL_DIR="$RN_MODEL_DIR"; else CFG_TAG="$VIT_CFG_TAG"; MODEL_DIR="$VIT_MODEL_DIR"; fi
        echo "============================================================"
        echo "[COLLECT] dataset=${DATASET} split=${SPLIT} seed=${SEED} branch=${BRANCH}"
        echo "============================================================"
        CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 \
        python scripts/dualenc_cache/collect_branch_logits_coop.py \
          --project-root "$ROOT" --data-root "$DATA_ROOT" \
          --dataset-config-file "$DATASET_CFG" \
          --config-file "$ROOT/third_party/CoOp_clean/configs/trainers/CoOp/${CFG_TAG}.yaml" \
          --model-dir "$MODEL_DIR" \
          --output-prefix "$OUT_DIR/logits" \
          --branch "$BRANCH" --source "$DATASET" --target "$DATASET" \
          --seed "$SEED" --subsample-classes "$SPLIT" \
          --load-epoch "$LOAD_EPOCH" --shots "$SHOTS" --nctx "$NCTX" --csc "$CSC" --ctx-pos "$CTX_POS" \
          --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
      done
    done
  done
done
