# Dual-Backbone Logit Fusion Cache Scripts

Implements the first cache-based adaptive fusion route:

1. Dataset-level Cached Fusion
2. Training-free Class-wise Reliability Cache
3. Class-wise oracle upper-bound diagnostic

Recommended current sanity while A100 is occupied:

```bash
bash scripts/dualenc_cache/00_no_gpu_dataset_cache_from_existing_results.sh
SEEDS="1" DATASETS="dtd eurosat oxford_pets" BATCH_SIZE=8 bash scripts/dualenc_cache/01_collect_b2n_logits.sh
SEEDS="1" DATASETS="dtd eurosat oxford_pets" bash scripts/dualenc_cache/02_eval_b2n_cache.sh
```

Full B2N after sanity:

```bash
SEEDS="1 2 3" DATASETS="caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101" BATCH_SIZE=8 bash scripts/dualenc_cache/01_collect_b2n_logits.sh
SEEDS="1 2 3" DATASETS="caltech101 dtd eurosat fgvc_aircraft food101 imagenet oxford_flowers oxford_pets stanford_cars sun397 ucf101" bash scripts/dualenc_cache/02_eval_b2n_cache.sh
```
