#!/usr/bin/env bash
set -euo pipefail

# Prepare single-backbone CoOp checkpoints and target eval logs for strict DG.
# This script does NOT run fusion. It only ensures RN101 and ViT-B/16 CoOp models exist.
#
# Usage:
#   GPU=0 bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
#
# Optional env:
#   ROOT=/workspace/meta_prompt_1
#   DATA_ROOT=/workspace/datasets
#   SEEDS="1 2 3"
#   SOURCE=imagenet
#   TARGETS="imagenetv2 imagenet_sketch imagenet_a imagenet_r"

ROOT=${ROOT:-/workspace/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
DATA_ROOT=${DATA_ROOT:-/workspace/datasets}
GPU=${GPU:-0}
SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"imagenetv2 imagenet_sketch imagenet_a imagenet_r"}

cd "$ROOT"

bash scripts/backbone_dg/01_create_backbone_configs.sh

read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a TARGET_ARR <<< "$TARGETS_STR"

for BACKBONE in rn101 vit_b16; do
  for SEED in "${SEED_ARR[@]}"; do
    echo "============================================================"
    echo "[PREPARE] CoOp ${BACKBONE} source=${SOURCE} seed=${SEED} targets=${TARGET_ARR[*]}"
    echo "============================================================"
    GPU="$GPU" DATA_ROOT="$DATA_ROOT" \
      bash scripts/backbone_dg/03_run_backbone_xd_one.sh \
      coop "$BACKBONE" "$SOURCE" "$SEED" "${TARGET_ARR[@]}"
  done
done

echo "[DONE] Prepared RN101 and ViT-B/16 CoOp checkpoints/logs for strict DG."
