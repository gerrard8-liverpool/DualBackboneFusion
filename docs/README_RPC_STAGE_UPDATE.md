## Reliability Prior Cache Stage

This repository now includes a clean Reliability Prior Cache evaluation stage for RN101 / ViT-B/16 dual-backbone prediction fusion.

The goal is to test whether cached backbone reliability priors can improve over fixed late fusion without retraining CLIP, CoOp, or the visual backbones.

### Clean protocols

| Task | Source cache | Target | Metric |
|---|---|---|---|
| B2N ImageNet Cache | `b2n / imagenet / split_all / seed1-3` | 10 non-ImageNet B2N datasets | HM(base,new) |
| B2N Meta LODO Cache | all B2N datasets except the target | 10 non-ImageNet B2N datasets | HM(base,new) |
| strict DG ImageNet Cache | `strict_dg / imagenet / unknown / seed1-3` | ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R | Accuracy |

### Main clean results

| Task | Line | Fusion | Fixed | Dataset Cache | Class Cache | Oracle |
|---|---|---|---:|---:|---:|---:|
| B2N | ImageNet Cache | prob_avg | 75.99 | 76.86 | 76.92 | 77.41 |
| B2N | Meta LODO Cache | prob_avg | 75.99 | 76.87 | 76.66 | 77.41 |
| strict DG | ImageNet Cache | prob_avg | 59.29 | 59.18 | 59.06 | 59.61 |

### Key observations

- Reliability Prior Cache is effective on B2N.
- `dataset_cache` is consistently positive across ImageNet Cache and Meta LODO Cache.
- `prob_avg` is currently the strongest and safest fusion interface.
- `raw_logits + class_cache` is unstable, suggesting that class-wise reliability transfer is calibration-sensitive.
- strict ImageNet-derived DG is not improved by the current ImageNet Cache, so DG requires a stronger domain-aware or shift-aware routing design.

Detailed handoff notes are available in `docs/NEXT_CHAT_HANDOFF_RPC_20260606.md`.
