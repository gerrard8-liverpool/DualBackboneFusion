#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
OUT_ROOT=${OUT_ROOT:-outputs/reliability_prior_cache}
MANIFEST=${MANIFEST:-$OUT_ROOT/paired_logits_manifest.csv}
FUSION_MODE=${FUSION_MODE:-std_logits}

# Clean B2N ImageNet source cache:
# use ONLY b2n/imagenet/split_all, not strict_dg/target_imagenet,
# and not b2n/imagenet/base/new to avoid duplicated ImageNet class entries.
SOURCE_PROTOCOLS=${SOURCE_PROTOCOLS:-"b2n"}
SOURCE_SPLITS=${SOURCE_SPLITS:-"all"}

# For ImageNet Cache -> B2N, exclude ImageNet target by default.
TARGET_DATASETS=${TARGET_DATASETS:-"caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101"}
TARGET_SPLITS=${TARGET_SPLITS:-"base new all"}
FALLBACK_W=${FALLBACK_W:-0.75}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/scripts/dualenc_rpc:$PROJECT_ROOT:${PYTHONPATH:-}"

RUN_DIR="$OUT_ROOT/imagenet_cache_b2n_clean/${FUSION_MODE}"
mkdir -p "$RUN_DIR"

python scripts/dualenc_rpc/build_reliability_prior_cache.py \
  --manifest "$MANIFEST" \
  --project-root "$PROJECT_ROOT" \
  --out "$RUN_DIR/imagenet_cache_clean.json" \
  --source-datasets imagenet \
  --source-splits $SOURCE_SPLITS \
  --protocols $SOURCE_PROTOCOLS \
  --fusion-mode "$FUSION_MODE" \
  --grid-step 0.05 \
  --shrink-lambda 20 \
  --tie-break-w "$FALLBACK_W"

python scripts/dualenc_rpc/eval_reliability_prior_cache.py \
  --manifest "$MANIFEST" \
  --project-root "$PROJECT_ROOT" \
  --cache-json "$RUN_DIR/imagenet_cache_clean.json" \
  --out-dir "$RUN_DIR/eval" \
  --target-datasets $TARGET_DATASETS \
  --target-splits $TARGET_SPLITS \
  --protocols b2n \
  --fallback-w "$FALLBACK_W" \
  --top-k 10 \
  --sim-temp 0.07 \
  --modes vit_only rn_only fixed dataset_cache class_cache oracle_dataset

echo "[DONE] $RUN_DIR/eval/rpc_eval_summary.md"
