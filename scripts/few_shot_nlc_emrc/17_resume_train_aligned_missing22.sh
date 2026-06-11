#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/workspace/meta_prompt_1}
RUN_TAG=${RUN_TAG:-trainaligned_full_20260611_093157}
MANIFEST=${MANIFEST:-outputs/nlc_emrc/manifests/four_backbone_manifest_with_image_features_zeroshot_10target_s123.csv}
PRIOR=${PRIOR:-outputs/nlc_emrc/priors/nlc_true_emrc_prior_from_routing.csv}
BETAS=${BETAS:-"0.00,0.02,0.05,0.10,0.20,0.30"}
EPOCHS=${EPOCHS:-300}

OUT_DIR="outputs/nlc_emrc/train_aligned_routing_runs/${RUN_TAG}"
SUMMARY_DIR="summary_tables/nlc_emrc/train_aligned_routing_${RUN_TAG}"
LOG_FILE="logs/nlc_emrc/${RUN_TAG}_resume_missing22.log"

mkdir -p "$OUT_DIR" "$SUMMARY_DIR" logs/nlc_emrc
cd "$ROOT"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[resume missing22] RUN_TAG=${RUN_TAG}"
echo "[out] ${OUT_DIR}"
echo "[summary] ${SUMMARY_DIR}"
echo "[prior] ${PRIOR}"
echo "[betas] ${BETAS}"

python -m py_compile scripts/nlc_emrc/15_train_eval_nlc_emrc_train_aligned_routing.py

run_one () {
  local dataset=$1
  local shots=$2
  local seed=$3
  local out="${OUT_DIR}/${dataset}_shot${shots}_seed${seed}.csv"

  if [ -s "$out" ] && grep -q "nlc_emrc_train_aligned_beta" "$out"; then
    echo "[skip existing] dataset=${dataset} shots=${shots} seed=${seed}"
    return
  fi

  echo "============================================================"
  echo "[run] dataset=${dataset} shots=${shots} seed=${seed}"
  echo "============================================================"

  python scripts/nlc_emrc/15_train_eval_nlc_emrc_train_aligned_routing.py \
    --root "$ROOT" \
    --manifest "$MANIFEST" \
    --dataset "$dataset" \
    --seed "$seed" \
    --shots "$shots" \
    --epochs "$EPOCHS" \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --hidden 128 \
    --emrc_prior "$PRIOR" \
    --fallback_alpha 0.4 \
    --betas "$BETAS" \
    --out "$out"
}

# missing from shot16 seed1
run_one sun397 16 1
run_one ucf101 16 1

# missing from shot16 seed2 / seed3
for seed in 2 3; do
  for dataset in caltech101 dtd eurosat fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101; do
    run_one "$dataset" 16 "$seed"
  done
done

echo "============================================================"
echo "[summarize]"
echo "============================================================"

python scripts/nlc_emrc/03_summarize_nlc_original.py \
  --input_glob "${OUT_DIR}/*.csv" \
  --out_dir "${SUMMARY_DIR}"

cat "${SUMMARY_DIR}/nlc_original_summary.md"

echo "[done] missing22 resume completed"
echo "[log] ${LOG_FILE}"
