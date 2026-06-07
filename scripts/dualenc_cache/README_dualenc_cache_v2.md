# DualEnc Cache V2

V2 adds two sanity-level improvements over the first training-free class-wise reliability cache:

1. `episodic_selected_cache`: selects retrieval hyperparameters on base-class pseudo-new episodes only.
2. `safe_fallback_cache`: falls back to dataset-level cached weight when class-wise cache does not improve base accuracy enough.

It uses existing logits under `outputs/dualenc_cache/logits/b2n` and does not load CLIP or occupy GPU.
