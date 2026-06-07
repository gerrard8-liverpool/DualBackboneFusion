#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
DATASETS=${DATASETS:-"dtd eurosat oxford_pets"}
SEEDS=${SEEDS:-"1"}
OUT=${OUT:-summary_tables/dualenc_cache/b2n_cache_sanity.md}
cd "$ROOT"
mkdir -p summary_tables/dualenc_cache
python scripts/dualenc_cache/eval_b2n_cache.py \
  --cache-root outputs/dualenc_cache/logits/b2n \
  --datasets $DATASETS \
  --seeds $SEEDS \
  --output "$OUT"
cat "$OUT"
