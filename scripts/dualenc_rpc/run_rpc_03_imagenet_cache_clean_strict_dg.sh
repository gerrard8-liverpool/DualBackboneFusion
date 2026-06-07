#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
OUT_ROOT=${OUT_ROOT:-outputs/reliability_prior_cache}
MANIFEST=${MANIFEST:-$OUT_ROOT/paired_logits_manifest.csv}
FUSION_MODE=${FUSION_MODE:-std_logits}

# Clean source: ImageNet val logits in strict_dg/target_imagenet.
SOURCE_PROTOCOLS=${SOURCE_PROTOCOLS:-"strict_dg"}
SOURCE_SPLITS=${SOURCE_SPLITS:-"unknown"}

TARGET_DATASETS=${TARGET_DATASETS:-"imagenetv2 imagenet_sketch imagenet_a imagenet_r"}
TARGET_SPLITS=${TARGET_SPLITS:-"unknown"}
FALLBACK_W=${FALLBACK_W:-0.75}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/scripts/dualenc_rpc:$PROJECT_ROOT:${PYTHONPATH:-}"

RUN_DIR="$OUT_ROOT/imagenet_cache_strict_dg_clean/${FUSION_MODE}"
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
  --protocols strict_dg \
  --fallback-w "$FALLBACK_W" \
  --top-k 10 \
  --sim-temp 0.07 \
  --modes vit_only rn_only fixed dataset_cache class_cache oracle_dataset

echo "[DONE] $RUN_DIR/eval/rpc_eval_summary.md"
