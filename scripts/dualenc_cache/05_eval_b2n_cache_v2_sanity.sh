#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT=${CACHE_ROOT:-outputs/dualenc_cache/logits/b2n}
DATASETS=${DATASETS:-"dtd eurosat oxford_pets"}
SEEDS=${SEEDS:-"1"}
MODES=${MODES:-"std_logits prob_avg"}
OUT=${OUT:-summary_tables/dualenc_cache/b2n_cache_v2_sanity.md}

python scripts/dualenc_cache/eval_b2n_cache_v2.py \
  --cache-root "$CACHE_ROOT" \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --modes $MODES \
  --text-space concat \
  --weight-step 0.05 \
  --candidate-shrink 4 8 16 32 \
  --candidate-topk 1 3 5 \
  --candidate-sem-temp 0.07 0.10 0.20 0.35 \
  --candidate-rho-power 1 2 4 8 \
  --rel-temp 1.0 \
  --safe-delta 0.2 \
  --output "$OUT"

echo "[DONE] $OUT"
