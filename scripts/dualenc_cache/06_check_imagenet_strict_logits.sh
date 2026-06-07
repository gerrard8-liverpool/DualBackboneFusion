#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
SOURCE=${SOURCE:-imagenet}
SEEDS_STR=${SEEDS:-"1 2 3"}
TARGETS_STR=${TARGETS:-"imagenet imagenetv2 imagenet_sketch imagenet_a imagenet_r"}
SHOTS=${SHOTS:-16}
NCTX=${NCTX:-16}
CSC=${CSC:-False}
CTX_POS=${CTX_POS:-end}

cd "$ROOT"
read -r -a SEED_ARR <<< "$SEEDS_STR"
read -r -a TARGET_ARR <<< "$TARGETS_STR"

OK=0
MISS=0
for TARGET in "${TARGET_ARR[@]}"; do
  for SEED in "${SEED_ARR[@]}"; do
    PREFIX="$ROOT/outputs/dualenc_cache/logits/strict_dg/source_${SOURCE}/target_${TARGET}/shots_${SHOTS}/nctx${NCTX}_csc${CSC}_ctp${CTX_POS}/seed${SEED}/logits"
    for BR in rn vit; do
      NPZ="${PREFIX}_${BR}.npz"
      META="${PREFIX}_${BR}.meta.json"
      if [ -f "$NPZ" ] && [ -f "$META" ]; then
        echo "[OK] target=${TARGET} seed=${SEED} branch=${BR}"
        OK=$((OK+1))
      else
        echo "[MISS] target=${TARGET} seed=${SEED} branch=${BR} npz=$NPZ"
        MISS=$((MISS+1))
      fi
    done
  done
done

echo "============================================================"
echo "OK=$OK"
echo "MISS=$MISS"
echo "Expected OK=$(( ${#TARGET_ARR[@]} * ${#SEED_ARR[@]} * 2 ))"
