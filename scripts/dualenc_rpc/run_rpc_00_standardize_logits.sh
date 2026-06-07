#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/meta_prompt_1}
CACHE_ROOT=${CACHE_ROOT:-outputs/dualenc_cache/logits}
OUT_ROOT=${OUT_ROOT:-outputs/reliability_prior_cache}
PAYLOAD_SPLIT=${PAYLOAD_SPLIT:-}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/scripts/dualenc_rpc:$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

ARGS=()
if [ -n "$PAYLOAD_SPLIT" ]; then
  ARGS+=(--payload-split "$PAYLOAD_SPLIT")
fi

python scripts/dualenc_rpc/inspect_logits_cache.py \
  --cache-root "$CACHE_ROOT" \
  --out "$OUT_ROOT/inspect_logits_cache.json" \
  --max-files 200 \
  "${ARGS[@]}"

python scripts/dualenc_rpc/standardize_dual_logits_cache.py \
  --cache-root "$CACHE_ROOT" \
  --out-dir "$OUT_ROOT/paired_logits" \
  --manifest "$OUT_ROOT/paired_logits_manifest.csv" \
  --relative-to "$PROJECT_ROOT" \
  --allow-combined \
  "${ARGS[@]}"

echo "[DONE] manifest: $OUT_ROOT/paired_logits_manifest.csv"
echo "[CHECK]"
python - <<'PY'
import csv
from collections import Counter
p='outputs/reliability_prior_cache/paired_logits_manifest.csv'
rows=list(csv.DictReader(open(p,encoding='utf-8-sig')))
print('rows=',len(rows))
print('by_protocol=',Counter(r['protocol'] for r in rows))
print('by_dataset=',Counter(r['dataset'] for r in rows))
print('by_split=',Counter(r['split'] for r in rows))
PY
