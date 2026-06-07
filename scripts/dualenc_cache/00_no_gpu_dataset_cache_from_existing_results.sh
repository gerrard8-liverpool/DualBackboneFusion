#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
cd "$ROOT"
mkdir -p summary_tables/dualenc_cache
python scripts/dualenc_cache/dataset_cache_from_existing_results.py \
  --root outputs/dualenc/late_fusion \
  --output summary_tables/dualenc_cache/dataset_cached_fusion_existing_results.md
cat summary_tables/dualenc_cache/dataset_cached_fusion_existing_results.md
