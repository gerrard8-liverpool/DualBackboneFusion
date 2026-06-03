#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/home/ubuntu/datasets}
GPU=${GPU:-0}
TRAINER=${TRAINER:-CoCoOp}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-4}
CTX_INIT=${CTX_INIT:-a_photo_of_a}
LOAD_EPOCH=${LOAD_EPOCH:-10}
RN_CFG_TAG=${RN_CFG_TAG:-rn101_c4_ep10_batch4_a100}
VIT_CFG_TAG=${VIT_CFG_TAG:-vit_b16_c4_ep10_batch4_a100}
RN_CFG="$COOP_ROOT/configs/trainers/CoCoOp/${RN_CFG_TAG}.yaml"
VIT_CFG="$COOP_ROOT/configs/trainers/CoCoOp/${VIT_CFG_TAG}.yaml"
ensure_configs() {
  bash "$ROOT/scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh"
  [ -f "$RN_CFG" ] || { echo "[ERROR] Missing RN config: $RN_CFG"; exit 1; }
  [ -f "$VIT_CFG" ] || { echo "[ERROR] Missing ViT config: $VIT_CFG"; exit 1; }
}
train_one_cocoop() {
  local backbone_name="$1" cfg_file="$2" cfg_tag="$3" dataset="$4" seed="$5" subsample="$6" train_dir="$7"
  local dataset_cfg="$COOP_ROOT/configs/datasets/${dataset}.yaml"
  local ckpt="$train_dir/prompt_learner/model.pth.tar-${LOAD_EPOCH}"
  [ -f "$dataset_cfg" ] || { echo "[ERROR] Missing dataset config: $dataset_cfg"; exit 1; }
  if [ -f "$ckpt" ]; then
    echo "[SKIP TRAIN] ${TRAINER} ${backbone_name} dataset=${dataset} split=${subsample} seed=${seed}"
    return 0
  fi
  echo "============================================================"
  echo "[TRAIN ${TRAINER}] backbone=${backbone_name} dataset=${dataset} split=${subsample} seed=${seed}"
  echo "out=${train_dir}"
  echo "============================================================"
  ( cd "$COOP_ROOT" && CUDA_VISIBLE_DEVICES=$GPU python train.py \
      --root "$DATA_ROOT" \
      --trainer "$TRAINER" \
      --dataset-config-file "$dataset_cfg" \
      --config-file "$cfg_file" \
      --output-dir "$train_dir" \
      --seed "$seed" \
      DATASET.NUM_SHOTS "$SHOTS" \
      DATASET.SUBSAMPLE_CLASSES "$subsample" \
      TRAINER.COCOOP.N_CTX "$NCTX" \
      TRAINER.COCOOP.CTX_INIT "$CTX_INIT" )
}
eval_fusion_one() {
  local source="$1" target="$2" seed="$3" subsample="$4" rn_train_dir="$5" vit_train_dir="$6" out_dir="$7"
  local target_cfg="$COOP_ROOT/configs/datasets/${target}.yaml"
  [ -f "$target_cfg" ] || { echo "[ERROR] Missing target dataset config: $target_cfg"; exit 1; }
  if [ -f "$out_dir/results.json" ]; then
    echo "[SKIP EVAL] ${source}->${target} split=${subsample} seed=${seed}: $out_dir/results.json"
    return 0
  fi
  echo "============================================================"
  echo "[LATE FUSION ${TRAINER}] ${source}->${target} split=${subsample} seed=${seed}"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES=$GPU python "$ROOT/scripts/dualenc_cocoop/eval_late_fusion_logits_any.py" \
    --project-root "$ROOT" \
    --data-root "$DATA_ROOT" \
    --trainer "$TRAINER" \
    --dataset-config-file "$target_cfg" \
    --rn-config "$RN_CFG" \
    --vit-config "$VIT_CFG" \
    --rn-model-dir "$rn_train_dir" \
    --vit-model-dir "$vit_train_dir" \
    --output-dir "$out_dir" \
    --source "$source" \
    --target "$target" \
    --seed "$seed" \
    --load-epoch "$LOAD_EPOCH" \
    --shots "$SHOTS" \
    --nctx "$NCTX" \
    --ctx-init "$CTX_INIT" \
    --subsample-classes "$subsample"
}
