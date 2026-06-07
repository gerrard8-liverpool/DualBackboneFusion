#!/usr/bin/env bash
set -euo pipefail
export DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101"}
export SEEDS=${SEEDS:-"1 2 3"}
export OUT=${OUT:-summary_tables/dualenc_cache/b2n_cache_full.md}
bash scripts/dualenc_cache/02_eval_b2n_cache.sh
