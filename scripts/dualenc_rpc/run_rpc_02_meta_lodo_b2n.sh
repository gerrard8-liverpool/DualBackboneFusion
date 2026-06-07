#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
OUT_ROOT=${OUT_ROOT:-outputs/reliability_prior_cache}
MANIFEST=${MANIFEST:-$OUT_ROOT/paired_logits_manifest.csv}
FUSION_MODE=${FUSION_MODE:-std_logits}
DATASETS=${DATASETS:-"caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101"}
SOURCE_SPLITS=${SOURCE_SPLITS:-"all base new val train test unknown"}
TARGET_SPLITS=${TARGET_SPLITS:-"base new all"}
FALLBACK_W=${FALLBACK_W:-0.75}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/scripts/dualenc_rpc:$PROJECT_ROOT:${PYTHONPATH:-}"

for TARGET in $DATASETS; do
  SRC=""
  for D in $DATASETS; do
    if [ "$D" != "$TARGET" ]; then SRC="$SRC $D"; fi
  done
  RUN_DIR="$OUT_ROOT/meta_lodo_b2n/${FUSION_MODE}/target_${TARGET}"
  mkdir -p "$RUN_DIR"
  echo "============================================================"
  echo "[Meta LODO] target=$TARGET sources=$SRC"
  echo "============================================================"
  python scripts/dualenc_rpc/build_reliability_prior_cache.py \
    --manifest "$MANIFEST" \
    --project-root "$PROJECT_ROOT" \
    --out "$RUN_DIR/meta_cache_exclude_${TARGET}.json" \
    --source-datasets $SRC \
    --source-splits $SOURCE_SPLITS \
    --fusion-mode "$FUSION_MODE" \
    --grid-step 0.05 \
    --shrink-lambda 20 \
    --tie-break-w "$FALLBACK_W"

  python scripts/dualenc_rpc/eval_reliability_prior_cache.py \
    --manifest "$MANIFEST" \
    --project-root "$PROJECT_ROOT" \
    --cache-json "$RUN_DIR/meta_cache_exclude_${TARGET}.json" \
    --out-dir "$RUN_DIR/eval" \
    --target-datasets "$TARGET" \
    --target-splits $TARGET_SPLITS \
    --protocols b2n \
    --fallback-w "$FALLBACK_W" \
    --top-k 10 \
    --sim-temp 0.07 \
    --exclude-target-dataset \
    --modes vit_only rn_only fixed dataset_cache class_cache oracle_dataset
done

echo "[DONE] meta LODO summaries under $OUT_ROOT/meta_lodo_b2n/${FUSION_MODE}"
