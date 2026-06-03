# Late Fusion Summary

Found `180` result rows.

## Target-wise fixed-weight results

| Target | RN101 only | ViT-B/16 only | Raw 0.5 | Std 0.5 | Prob 0.5 |
|---|---:|---:|---:|---:|---:|
| imagenet_a | 28.90±0.41 (3) | 48.68±0.35 (3) | 44.08±0.02 (3) | 43.46±0.14 (3) | 45.61±0.13 (3) |
| imagenet_r | 62.70±0.61 (3) | 73.98±0.29 (3) | 74.08±0.41 (3) | 73.82±0.42 (3) | 74.29±0.48 (3) |
| imagenet_sketch | 39.62±0.06 (3) | 46.97±0.36 (3) | 48.82±0.13 (3) | 48.62±0.08 (3) | 48.43±0.16 (3) |
| imagenetv2 | 58.36±0.08 (3) | 64.34±0.04 (3) | 65.89±0.20 (3) | 65.82±0.21 (3) | 65.74±0.08 (3) |

## Overall fixed-weight results

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.39±13.76 (12) |
| ViT-B/16 only | 58.50±11.22 (12) |
| Raw logits fusion, w=0.5 | 58.22±12.23 (12) |
| Standardized logits fusion, w=0.5 | 57.93±12.36 (12) |
| Probability average, w=0.5 | 58.52±11.93 (12) |

## Best-over-weight diagnostic

This table is diagnostic only. Do not report best-over-target weights as a fair main result unless the weight-selection rule is fixed without target labels.

| Mode | Best-over-weight Accuracy | Mean selected w |
|---|---:|---:|
| raw_logits | 59.75±11.38 (12) | 0.79 |
| std_logits | 59.75±11.38 (12) | 0.81 |
| prob_avg | 59.45±11.36 (12) | 0.71 |