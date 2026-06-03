#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
COOP_ROOT=${COOP_ROOT:-$ROOT/third_party/CoOp_clean}
CFG_DIR="$COOP_ROOT/configs/trainers/CoCoOp"

mkdir -p "$CFG_DIR"

cat > "$CFG_DIR/rn101_c4_ep10_batch4_a100.yaml" <<'YAML'
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 100
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.002
  MAX_EPOCH: 10
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5

TRAIN:
  PRINT_FREQ: 5

MODEL:
  BACKBONE:
    NAME: "RN101"

TRAINER:
  COCOOP:
    N_CTX: 4
    CTX_INIT: "a_photo_of_a"
    PREC: "fp16"
YAML

cat > "$CFG_DIR/vit_b16_c4_ep10_batch4_a100.yaml" <<'YAML'
DATALOADER:
  TRAIN_X:
    BATCH_SIZE: 4
  TEST:
    BATCH_SIZE: 100
  NUM_WORKERS: 8

INPUT:
  SIZE: (224, 224)
  INTERPOLATION: "bicubic"
  PIXEL_MEAN: [0.48145466, 0.4578275, 0.40821073]
  PIXEL_STD: [0.26862954, 0.26130258, 0.27577711]
  TRANSFORMS: ["random_resized_crop", "random_flip", "normalize"]

OPTIM:
  NAME: "sgd"
  LR: 0.002
  MAX_EPOCH: 10
  LR_SCHEDULER: "cosine"
  WARMUP_EPOCH: 1
  WARMUP_TYPE: "constant"
  WARMUP_CONS_LR: 1e-5

TRAIN:
  PRINT_FREQ: 5

MODEL:
  BACKBONE:
    NAME: "ViT-B/16"

TRAINER:
  COCOOP:
    N_CTX: 4
    CTX_INIT: "a_photo_of_a"
    PREC: "fp16"
YAML

echo "[WROTE] $CFG_DIR/rn101_c4_ep10_batch4_a100.yaml"
echo "[WROTE] $CFG_DIR/vit_b16_c4_ep10_batch4_a100.yaml"
ls -lh "$CFG_DIR"/*c4_ep10_batch4_a100.yaml
