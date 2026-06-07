#!/usr/bin/env bash
set -euo pipefail
export DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101"}
export SEEDS=${SEEDS:-"1 2 3"}
export SPLITS=${SPLITS:-"base new all"}
export BATCH_SIZE=${BATCH_SIZE:-8}
bash scripts/dualenc_cache/01_collect_b2n_logits.sh
