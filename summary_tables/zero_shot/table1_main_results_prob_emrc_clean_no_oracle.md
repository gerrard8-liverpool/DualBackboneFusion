# Main Table: Zero-shot EMRC, excluding ImageNet

Protocol: 10 non-ImageNet target datasets, 3 target seeds, 5 k-means meta-cache seeds. ImageNet is used only as the source/cache dataset and is excluded from the target average. EMRC setting: k=100, topk=2, temperature=0.07.

Oracle-based diagnostic results are excluded from this main table.

## Per-dataset Results

| dataset        |   ViT-B/16 |   BSS-ZSEn |   Fixed Raw w0.50 |   ImageNet Dataset Cache |   EMRC Conf-Gate |   EMRC-TopK |   Δ EMRC vs BSS |   Δ EMRC vs Dataset Cache |   Δ EMRC vs Fixed Raw |
|:---------------|-----------:|-----------:|------------------:|-------------------------:|-----------------:|------------:|----------------:|--------------------------:|----------------------:|
| caltech101     |    92.6978 |    93.7931 |           93.8742 |                  93.9148 |          94.0365 |     93.9554 |          0.1623 |                    0.0406 |                0.0811 |
| dtd            |    38.2979 |    40.7801 |           40.8392 |                  40.8392 |          40.8392 |     41.0165 |          0.2364 |                    0.1773 |                0.1773 |
| eurosat        |    21.1605 |    22.3086 |           22.3704 |                  22.3827 |          22.5309 |     22.5679 |          0.2593 |                    0.1852 |                0.1975 |
| fgvc_aircraft  |    22.0522 |    23.1023 |           22.5623 |                  22.8323 |          22.8923 |     23.0723 |         -0.0300 |                    0.2400 |                0.5101 |
| food101        |    78.6799 |    79.8119 |           79.4917 |                  79.4587 |          79.6766 |     79.9010 |          0.0891 |                    0.4422 |                0.4092 |
| oxford_flowers |    64.3524 |    67.7629 |           67.6817 |                  67.5599 |          67.4787 |     68.0065 |          0.2436 |                    0.4466 |                0.3248 |
| oxford_pets    |    86.0725 |    88.4982 |           88.6072 |                  88.5255 |          88.4982 |     88.4982 |          0.0000 |                   -0.0273 |               -0.1090 |
| stanford_cars  |    59.8806 |    66.0739 |           67.0190 |                  66.9568 |          66.9071 |     66.7579 |          0.6840 |                   -0.1990 |               -0.2612 |
| sun397         |    61.2997 |    64.0403 |           63.8237 |                  63.8690 |          63.8942 |     64.0353 |         -0.0050 |                    0.1662 |                0.2116 |
| ucf101         |    64.0233 |    65.6622 |           65.6886 |                  65.5829 |          65.7943 |     65.9529 |          0.2908 |                    0.3701 |                0.2643 |
| Average        |    58.8517 |    61.1834 |           61.1958 |                  61.1922 |          61.2548 |     61.3764 |          0.1930 |                    0.1842 |                0.1806 |

## Paired Delta Summary

| baseline               |   mean_delta |   median_delta | wins/ties/losses   |
|:-----------------------|-------------:|---------------:|:-------------------|
| BSS-ZSEn               |       0.1930 |         0.1994 | 7/1/2              |
| ImageNet Dataset Cache |       0.1842 |         0.1813 | 8/0/2              |
| Fixed Raw w0.50        |       0.1806 |         0.2046 | 8/0/2              |
