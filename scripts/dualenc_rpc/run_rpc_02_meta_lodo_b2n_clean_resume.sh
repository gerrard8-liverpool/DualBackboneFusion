#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
OUT_ROOT=${OUT_ROOT:-outputs/reliability_prior_cache}
MANIFEST=${MANIFEST:-$OUT_ROOT/paired_logits_manifest.csv}
FUSION_MODE=${FUSION_MODE:-prob_avg}

SOURCE_DATASETS=${SOURCE_DATASETS:-"caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101"}
TARGET_DATASETS=${TARGET_DATASETS:-"food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101"}

SOURCE_SPLITS=${SOURCE_SPLITS:-"all base new"}
TARGET_SPLITS=${TARGET_SPLITS:-"base new all"}
FALLBACK_W=${FALLBACK_W:-0.75}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/scripts/dualenc_rpc:$PROJECT_ROOT:${PYTHONPATH:-}"

RUN_ROOT="$OUT_ROOT/meta_lodo_b2n_clean/${FUSION_MODE}"
mkdir -p "$RUN_ROOT"

for TARGET in $TARGET_DATASETS; do
  SOURCES=""
  for D in $SOURCE_DATASETS; do
    if [ "$D" != "$TARGET" ]; then
      SOURCES="$SOURCES $D"
    fi
  done

  TARGET_DIR="$RUN_ROOT/target_${TARGET}"
  CACHE_JSON="$TARGET_DIR/meta_cache_exclude_${TARGET}.json"
  SUMMARY="$TARGET_DIR/eval/rpc_eval_summary.md"

  mkdir -p "$TARGET_DIR"

  echo "============================================================"
  echo "[Meta LODO B2N CLEAN RESUME] target=$TARGET"
  echo "sources=$SOURCES"
  echo "fusion_mode=$FUSION_MODE"
  echo "cache_json=$CACHE_JSON"
  echo "summary=$SUMMARY"
  echo "============================================================"

  if [ -f "$SUMMARY" ]; then
    echo "[SKIP] summary already exists: $SUMMARY"
    continue
  fi

  if [ ! -f "$CACHE_JSON" ]; then
    echo "[BUILD] $CACHE_JSON"
    python scripts/dualenc_rpc/build_reliability_prior_cache.py \
      --manifest "$MANIFEST" \
      --project-root "$PROJECT_ROOT" \
      --out "$CACHE_JSON" \
      --source-datasets $SOURCES \
      --source-splits $SOURCE_SPLITS \
      --protocols b2n \
      --fusion-mode "$FUSION_MODE" \
      --grid-step 0.05 \
      --shrink-lambda 20 \
      --tie-break-w "$FALLBACK_W"
  else
    echo "[SKIP BUILD] cache json exists: $CACHE_JSON"
  fi

  echo "[EVAL] target=$TARGET"
  python scripts/dualenc_rpc/eval_reliability_prior_cache.py \
    --manifest "$MANIFEST" \
    --project-root "$PROJECT_ROOT" \
    --cache-json "$CACHE_JSON" \
    --out-dir "$TARGET_DIR/eval" \
    --target-datasets "$TARGET" \
    --target-splits $TARGET_SPLITS \
    --protocols b2n \
    --fallback-w "$FALLBACK_W" \
    --top-k 10 \
    --sim-temp 0.07 \
    --modes vit_only rn_only fixed dataset_cache class_cache oracle_dataset

  echo "[DONE] $SUMMARY"
done

echo "[DONE] resume meta LODO B2N: $RUN_ROOT"
