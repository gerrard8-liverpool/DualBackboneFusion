\# DualBackboneFusion



\[English](#english) | \[中文](#中文)



\---



<a id="english"></a>



\## DualBackboneFusion: Prediction-Level Dual-Backbone Fusion for CLIP Prompt Transfer



This repository studies \*\*visual-backbone complementarity in CLIP prompt learning\*\*.

The central observation is that a weaker CLIP visual backbone, such as \*\*RN101\*\*, can still provide complementary prediction information to a stronger \*\*ViT-B/16\*\* backbone. Instead of directly merging feature embeddings, this project first evaluates a simple but effective \*\*prediction-level late fusion\*\* strategy:



```text

fused\_logits = w \* logits\_vit + (1 - w) \* logits\_rn

```



where `w = 1.0` corresponds to pure ViT-B/16 and `w = 0.0` corresponds to pure RN101.



The current results suggest that RN101 is much weaker than ViT-B/16 as a standalone prompt learner, but its prediction distribution can still improve ViT-B/16 under several CLIP prompt-learning transfer settings, including:



\* Base-to-novel generalization

\* ImageNet strict domain generalization

\* ImageNet-source cross-dataset transfer

\* CoCoOp base-to-novel sanity evaluation



The project is currently positioned as:



```text

Backbone Complementarity in CLIP Prompt Learning

```



or more specifically:



```text

Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer

```



\---



\## Motivation



CLIP prompt learning methods such as CoOp and CoCoOp usually evaluate one visual backbone at a time, for example RN50, RN101, or ViT-B/16. However, different visual backbones can encode different inductive biases.



A CNN-style backbone tends to emphasize local visual patterns, texture, edges, and object-level cues. A ViT-style backbone can exploit global patch interactions and long-range contextual information. These behaviors are not strictly better or worse in all cases. Instead, they may fail on different examples and different target domains.



This motivates the following research question:



> Can a weaker CLIP visual backbone provide complementary prediction signals to a stronger backbone under prompt learning?



The first step in this repository is deliberately simple: do not directly mix feature embeddings, because this may disturb each backbone's own CLIP image-text alignment space. Instead, each backbone keeps its own CLIP-aligned representation, and fusion is performed at the prediction/logit level.



\---



\## Main Idea



Given two CLIP prompt learners trained under the same protocol:



\* RN101-based prompt learner

\* ViT-B/16-based prompt learner



we evaluate both models on the same dataloader and combine their output logits:



```text

fused = w \* logits\_vit + (1 - w) \* logits\_rn

```



The current implementation supports three fusion variants:



```text

raw\_logits

std\_logits

prob\_avg

```



and commonly evaluates the following fixed weights:



```text

w = 0.00, 0.25, 0.50, 0.75, 1.00

```



The goal is not to claim that fixed late fusion is the final method. Instead, fixed late fusion is used as the first-stage diagnostic tool to verify whether RN101 and ViT-B/16 have meaningful prediction-level complementarity.



\---



\## Repository Structure



```text

DualBackboneFusion/

├── README.md

├── scripts/

│   ├── dualenc/

│   │   ├── eval\_late\_fusion\_logits.py

│   │   ├── summarize\_late\_fusion.py

│   │   ├── 01\_prepare\_late\_fusion\_strict\_dg.sh

│   │   ├── 02\_eval\_late\_fusion\_strict\_dg.sh

│   │   ├── 03\_run\_late\_fusion\_b2n\_sanity.sh

│   │   ├── 04\_run\_late\_fusion\_b2n\_full.sh

│   │   └── 05\_eval\_late\_fusion\_xd\_imagenet\_source.sh

│   ├── dualenc\_cocoop/

│   │   ├── 00\_create\_cocoop\_dualenc\_configs.sh

│   │   ├── 01\_run\_cocoop\_late\_fusion\_strict\_dg.sh

│   │   ├── 02\_run\_cocoop\_late\_fusion\_xd\_imagenet\_source.sh

│   │   ├── 03\_run\_cocoop\_late\_fusion\_b2n.sh

│   │   ├── eval\_late\_fusion\_logits\_any.py

│   │   ├── summarize\_late\_fusion\_accuracy.py

│   │   └── summarize\_late\_fusion\_b2n.py

│   └── dualenc\_feature/

│       ├── 00\_install\_dualenc\_feature.sh

│       ├── 01\_run\_dualenc\_feature\_b2n\_sanity.sh

│       ├── coop\_dualenc.py

│       └── summarize\_dualenc\_feature\_b2n.py

├── summary\_tables/

│   ├── dualenc/

│   ├── dualenc\_cocoop/

│   └── dualenc\_feature/

└── third\_party/

&#x20;   └── CoOp\_clean/

&#x20;       ├── trainers/

&#x20;       │   └── coop\_dualenc.py

&#x20;       └── configs/

&#x20;           ├── trainers/

&#x20;           └── datasets/

```



\### Main branches



| Branch               | Directory                  | Purpose                                          |

| -------------------- | -------------------------- | ------------------------------------------------ |

| CoOp late fusion     | `scripts/dualenc/`         | Main prediction-level fusion experiments         |

| CoCoOp late fusion   | `scripts/dualenc\_cocoop/`  | Extension to dynamic prompt learning             |

| Feature-level fusion | `scripts/dualenc\_feature/` | Negative baseline for embedding-level MLP fusion |



\---



\## Current Results



\### 1. CoOp strict domain generalization



Setting:



```text

Source = ImageNet

Targets = ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R

Backbones = RN101 + ViT-B/16

Seeds = 1, 2, 3

```



| Setting           | Accuracy |

| ----------------- | -------: |

| RN101 only        |    47.39 |

| ViT-B/16 only     |    58.50 |

| Raw logits w=0.75 |    59.50 |

| Std logits w=0.75 |    59.45 |

| Prob avg w=0.75   |    59.30 |



Observation: RN101 is much weaker than ViT-B/16 as a standalone model, but adding it as a 25% auxiliary branch improves ViT-B/16. The best fixed weight is ViT-dominant in this setting.



\---



\### 2. CoOp ImageNet-source cross-dataset transfer



Setting:



```text

Source = ImageNet

Targets = caltech101, oxford\_pets, dtd, eurosat, food101,

&#x20;         oxford\_flowers, stanford\_cars, fgvc\_aircraft, ucf101, sun397

Backbones = RN101 + ViT-B/16

```



| Setting           | Accuracy |

| ----------------- | -------: |

| RN101 only        |    55.96 |

| ViT-B/16 only     |    61.82 |

| Raw logits w=0.50 |    63.03 |

| Raw logits w=0.75 |    63.26 |

| Std logits w=0.75 |    63.25 |

| Prob avg w=0.50   |    63.63 |



Observation: Prediction-level fusion gives a stronger gain in cross-dataset transfer. The best setting, `prob\_avg w=0.50`, improves ViT-B/16 by about +1.81 points.



\---



\### 3. CoOp base-to-novel full evaluation



Setting:



```text

Datasets = caltech101, dtd, eurosat, fgvc\_aircraft, food101, imagenet,

&#x20;          oxford\_flowers, oxford\_pets, stanford\_cars, sun397, ucf101

Seeds = 1, 2, 3

```



| Setting           |  Base |   New |    HM |   All |

| ----------------- | ----: | ----: | ----: | ----: |

| RN101 only        | 79.83 | 62.52 | 69.20 | 63.07 |

| ViT-B/16 only     | 82.72 | 68.11 | 74.20 | 67.75 |

| Raw logits w=0.50 | 84.50 | 70.93 | 76.67 | 70.24 |

| Std logits w=0.50 | 84.38 | 70.62 | 76.41 | 70.05 |

| Prob avg w=0.50   | 84.23 | 70.59 | 76.33 | 69.94 |



Observation: Base and new class performance are both improved. The harmonic mean improves from 74.20 to 76.67, suggesting that the fusion does not simply improve base-class fitting but also helps unseen-class generalization.



\---



\### 4. CoCoOp base-to-novel sanity evaluation



Setting:



```text

Datasets = DTD, EuroSAT, OxfordPets

Seeds = 1, 2, 3

Backbones = RN101 + ViT-B/16

```



| Setting           |  Base |   New |    HM |   All |

| ----------------- | ----: | ----: | ----: | ----: |

| RN101 only        | 82.65 | 62.40 | 68.91 | 60.29 |

| ViT-B/16 only     | 83.14 | 72.61 | 77.08 | 66.34 |

| Raw logits w=0.50 | 87.05 | 73.24 | 78.70 | 69.18 |

| Raw logits w=0.75 | 86.54 | 74.59 | 79.59 | 69.49 |

| Std logits w=0.75 | 86.16 | 74.56 | 79.42 | 69.43 |



Observation: CoCoOp also benefits from dual-backbone prediction-level fusion. This suggests that the complementarity is not only a CoOp-specific phenomenon.



\---



\### 5. Feature-level MLP fusion sanity result



The first feature-level fusion baseline uses:



```text

v\_fused = normalize(v\_vit + alpha \* MLP(\[v\_vit, v\_rn]))

```



Current sanity results:



| Dataset    | Feature MLP HM | ViT-B/16 HM | Late Fusion HM |

| ---------- | -------------: | ----------: | -------------: |

| DTD        |          56.34 |       61.27 |          64.59 |

| EuroSAT    |          73.67 |       77.47 |          78.34 |

| OxfordPets |          93.44 |       94.01 |          95.06 |



Observation: Naive feature-level MLP fusion underperforms ViT-B/16 and is clearly worse than late fusion. This supports the current interpretation that RN101 and ViT-B/16 are complementary at the prediction level, while directly mixing image embeddings may damage CLIP image-text alignment.



\---



\## Running Instructions



\### Environment



This project is built on top of the CoOp/CoCoOp codebase and Dassl.



Typical requirements:



```text

Python 3.x

PyTorch

torchvision

Dassl.pytorch

CLIP

CoOp\_clean

```



The expected project root on the server is usually:



```text

/workspace/meta\_prompt\_1

```



or:



```text

/home/ubuntu/code/meta\_prompt\_1

```



Dataset root is usually:



```text

/workspace/datasets

```



or:



```text

/home/ubuntu/datasets

```



Please adjust paths inside the scripts according to your local or server environment.



\---



\### 1. CoOp strict domain generalization



Prepare ImageNet-source checkpoints and evaluate ImageNet variants:



```bash

bash scripts/dualenc/01\_prepare\_late\_fusion\_strict\_dg.sh

bash scripts/dualenc/02\_eval\_late\_fusion\_strict\_dg.sh

```



This evaluates:



```text

ImageNet -> ImageNetV2

ImageNet -> ImageNet-Sketch

ImageNet -> ImageNet-A

ImageNet -> ImageNet-R

```



\---



\### 2. CoOp base-to-novel evaluation



Sanity version:



```bash

bash scripts/dualenc/03\_run\_late\_fusion\_b2n\_sanity.sh

```



Full version:



```bash

bash scripts/dualenc/04\_run\_late\_fusion\_b2n\_full.sh

```



The full version evaluates 11 datasets:



```text

caltech101

dtd

eurosat

fgvc\_aircraft

food101

imagenet

oxford\_flowers

oxford\_pets

stanford\_cars

sun397

ucf101

```



\---



\### 3. CoOp ImageNet-source cross-dataset transfer



```bash

bash scripts/dualenc/05\_eval\_late\_fusion\_xd\_imagenet\_source.sh

```



This evaluates ImageNet-source checkpoints on 10 heterogeneous target datasets:



```text

caltech101

oxford\_pets

dtd

eurosat

food101

oxford\_flowers

stanford\_cars

fgvc\_aircraft

ucf101

sun397

```



\---



\### 4. CoCoOp late fusion



Generate dual-backbone CoCoOp configs:



```bash

bash scripts/dualenc\_cocoop/00\_create\_cocoop\_dualenc\_configs.sh

```



Strict domain generalization:



```bash

bash scripts/dualenc\_cocoop/01\_run\_cocoop\_late\_fusion\_strict\_dg.sh

```



ImageNet-source cross-dataset transfer:



```bash

bash scripts/dualenc\_cocoop/02\_run\_cocoop\_late\_fusion\_xd\_imagenet\_source.sh

```



Base-to-novel evaluation:



```bash

bash scripts/dualenc\_cocoop/03\_run\_cocoop\_late\_fusion\_b2n.sh

```



\---



\### 5. Feature-level MLP fusion baseline



Install or copy the dual-encoder trainer:



```bash

bash scripts/dualenc\_feature/00\_install\_dualenc\_feature.sh

```



Run B2N sanity evaluation:



```bash

bash scripts/dualenc\_feature/01\_run\_dualenc\_feature\_b2n\_sanity.sh

```



Summarize results:



```bash

python scripts/dualenc\_feature/summarize\_dualenc\_feature\_b2n.py

```



\---



\## Result Summaries



GitHub-ready summary tables are stored under:



```text

summary\_tables/dualenc/

summary\_tables/dualenc\_cocoop/

summary\_tables/dualenc\_feature/

```



Raw outputs, checkpoints, tensorboard files, and raw logs should not be committed to GitHub.



\---



\## What Should Be Committed



Recommended files:



```text

README.md

scripts/

summary\_tables/

third\_party/CoOp\_clean/trainers/coop\_dualenc.py

third\_party/CoOp\_clean/configs/trainers/

third\_party/CoOp\_clean/configs/datasets/

```



Do not commit:



```text

datasets/

data/

third\_party/CoOp\_clean/output/

third\_party/CoOp\_clean/output\_\*/

logs/

tensorboard/

\*.pth

\*.pth.tar

\*.pt

\*.ckpt

\*.pkl

\*.npy

\*.npz

```



\---



\## Current Interpretation



The current evidence supports the following interpretation:



1\. RN101 is weaker than ViT-B/16 as a standalone CLIP prompt learner.

2\. However, RN101 can still provide complementary prediction information to ViT-B/16.

3\. Prediction-level late fusion improves ViT-B/16 under CoOp B2N, strict DG, and ImageNet-source cross-dataset transfer.

4\. CoCoOp B2N sanity results suggest that this complementarity is not limited to static prompt learning.

5\. Naive feature-level fusion is weaker than prediction-level fusion, suggesting that preserving each backbone's own CLIP alignment space is important.



The current method should not be overclaimed as a universal replacement for stronger prompt-learning methods such as MaPLe, PromptSRC, DPC, or Promise. A more appropriate claim is that dual-backbone prediction fusion is a potentially orthogonal module that can be attached to different prompt learners.



\---



\## Future Work



Planned next steps:



1\. Extend late fusion to stronger prompt learners such as MaPLe, PromptSRC, and newer prompt-robust methods.

2\. Complete CoCoOp strict domain generalization and ImageNet-source cross-dataset transfer.

3\. Add control experiments:



&#x20;  \* RN logits image-shuffle

&#x20;  \* RN logits class-shuffle

&#x20;  \* matched Gaussian logits noise

&#x20;  \* temperature or calibration controls

4\. Add mechanism analysis:



&#x20;  \* disagreement rescue analysis

&#x20;  \* confidence/entropy bin analysis

&#x20;  \* optimal fusion weight heatmap

&#x20;  \* source/task-dependent fusion ratio analysis

5\. Replace fixed fusion weights with validation-trained adaptive gating:



```text

w\_i = f(conf\_vit, conf\_rn, entropy\_vit, entropy\_rn, margin\_vit, margin\_rn, top1\_agreement)

```



Target test labels must not be used for selecting or training fusion weights.



\---



\## Citation



This repository is an active research codebase. A formal citation will be added after the paper or preprint is available.



\---



<a id="中文"></a>



\# DualBackboneFusion：面向 CLIP Prompt Transfer 的双主干预测层融合



本仓库研究 \*\*CLIP prompt learning 中的视觉主干互补性\*\*。

核心观察是：虽然 \*\*RN101\*\* 作为单模型通常弱于 \*\*ViT-B/16\*\*，但它的预测分布可能包含 ViT-B/16 没有利用到的互补信息。因此，本项目第一阶段不直接混合 embedding，而是在 prediction/logit 层做 late fusion：



```text

fused\_logits = w \* logits\_vit + (1 - w) \* logits\_rn

```



其中 `w = 1.0` 表示纯 ViT-B/16，`w = 0.0` 表示纯 RN101。



当前结果显示：RN101 单模型明显弱于 ViT-B/16，但其预测分布可以在多个 CLIP prompt-learning transfer 设置下增强 ViT-B/16，包括：



\* Base-to-novel generalization

\* ImageNet strict domain generalization

\* ImageNet-source cross-dataset transfer

\* CoCoOp base-to-novel sanity evaluation



当前项目定位为：



```text

Backbone Complementarity in CLIP Prompt Learning

```



或：



```text

Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer

```



\---



\## 研究动机



CoOp、CoCoOp 等 CLIP prompt learning 方法通常一次只评估一个 visual backbone，例如 RN50、RN101 或 ViT-B/16。然而，不同视觉主干具有不同的归纳偏置。



CNN 风格主干更容易关注局部视觉模式、纹理、边缘和目标本体特征；ViT 风格主干更容易利用 patch 之间的全局交互和长程上下文信息。这两类行为不是绝对优劣关系，而可能在不同样本、不同目标域上表现出不同错误模式。



因此，本项目关注以下问题：



> 一个较弱的 CLIP visual backbone 是否能为较强的 visual backbone 提供互补预测信号？



本仓库的第一步故意采用简单方案：不直接混合 feature embedding，因为这可能破坏每个 backbone 自身的 CLIP image-text alignment space。相反，每个 backbone 保留自己的 CLIP 对齐空间，只在最终 prediction/logit 层融合。



\---



\## 核心方法



给定两个在相同协议下训练的 CLIP prompt learners：



\* RN101-based prompt learner

\* ViT-B/16-based prompt learner



在同一个 dataloader 上分别计算 logits，然后做融合：



```text

fused = w \* logits\_vit + (1 - w) \* logits\_rn

```



当前实现支持三种融合方式：



```text

raw\_logits

std\_logits

prob\_avg

```



常用固定权重为：



```text

w = 0.00, 0.25, 0.50, 0.75, 1.00

```



固定权重 late fusion 不是最终目标，而是第一阶段用来验证 RN101 与 ViT-B/16 是否具有 prediction-level complementarity 的诊断工具。



\---



\## 仓库结构



```text

DualBackboneFusion/

├── README.md

├── scripts/

│   ├── dualenc/

│   │   ├── eval\_late\_fusion\_logits.py

│   │   ├── summarize\_late\_fusion.py

│   │   ├── 01\_prepare\_late\_fusion\_strict\_dg.sh

│   │   ├── 02\_eval\_late\_fusion\_strict\_dg.sh

│   │   ├── 03\_run\_late\_fusion\_b2n\_sanity.sh

│   │   ├── 04\_run\_late\_fusion\_b2n\_full.sh

│   │   └── 05\_eval\_late\_fusion\_xd\_imagenet\_source.sh

│   ├── dualenc\_cocoop/

│   │   ├── 00\_create\_cocoop\_dualenc\_configs.sh

│   │   ├── 01\_run\_cocoop\_late\_fusion\_strict\_dg.sh

│   │   ├── 02\_run\_cocoop\_late\_fusion\_xd\_imagenet\_source.sh

│   │   ├── 03\_run\_cocoop\_late\_fusion\_b2n.sh

│   │   ├── eval\_late\_fusion\_logits\_any.py

│   │   ├── summarize\_late\_fusion\_accuracy.py

│   │   └── summarize\_late\_fusion\_b2n.py

│   └── dualenc\_feature/

│       ├── 00\_install\_dualenc\_feature.sh

│       ├── 01\_run\_dualenc\_feature\_b2n\_sanity.sh

│       ├── coop\_dualenc.py

│       └── summarize\_dualenc\_feature\_b2n.py

├── summary\_tables/

│   ├── dualenc/

│   ├── dualenc\_cocoop/

│   └── dualenc\_feature/

└── third\_party/

&#x20;   └── CoOp\_clean/

&#x20;       ├── trainers/

&#x20;       │   └── coop\_dualenc.py

&#x20;       └── configs/

&#x20;           ├── trainers/

&#x20;           └── datasets/

```



\### 三条实验线



| 分支                   | 目录                         | 作用                          |

| -------------------- | -------------------------- | --------------------------- |

| CoOp late fusion     | `scripts/dualenc/`         | 主线：预测层 late fusion          |

| CoCoOp late fusion   | `scripts/dualenc\_cocoop/`  | 扩展到 dynamic prompt learning |

| Feature-level fusion | `scripts/dualenc\_feature/` | embedding 层 MLP 融合负例        |



\---



\## 当前结果



\### 1. CoOp strict domain generalization



设置：



```text

Source = ImageNet

Targets = ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R

Backbones = RN101 + ViT-B/16

Seeds = 1, 2, 3

```



| Setting           | Accuracy |

| ----------------- | -------: |

| RN101 only        |    47.39 |

| ViT-B/16 only     |    58.50 |

| Raw logits w=0.75 |    59.50 |

| Std logits w=0.75 |    59.45 |

| Prob avg w=0.75   |    59.30 |



结论：RN101 单模型明显弱于 ViT-B/16，但作为 25% 辅助分支可以提升 ViT-B/16。该设置中最优固定权重更偏向 ViT-dominant。



\---



\### 2. CoOp ImageNet-source cross-dataset transfer



设置：



```text

Source = ImageNet

Targets = caltech101, oxford\_pets, dtd, eurosat, food101,

&#x20;         oxford\_flowers, stanford\_cars, fgvc\_aircraft, ucf101, sun397

Backbones = RN101 + ViT-B/16

```



| Setting           | Accuracy |

| ----------------- | -------: |

| RN101 only        |    55.96 |

| ViT-B/16 only     |    61.82 |

| Raw logits w=0.50 |    63.03 |

| Raw logits w=0.75 |    63.26 |

| Std logits w=0.75 |    63.25 |

| Prob avg w=0.50   |    63.63 |



结论：cross-dataset transfer 中 late fusion 提升更明显。最佳设置 `prob\_avg w=0.50` 比 ViT-B/16 高约 +1.81。



\---



\### 3. CoOp base-to-novel full evaluation



设置：



```text

Datasets = caltech101, dtd, eurosat, fgvc\_aircraft, food101, imagenet,

&#x20;          oxford\_flowers, oxford\_pets, stanford\_cars, sun397, ucf101

Seeds = 1, 2, 3

```



| Setting           |  Base |   New |    HM |   All |

| ----------------- | ----: | ----: | ----: | ----: |

| RN101 only        | 79.83 | 62.52 | 69.20 | 63.07 |

| ViT-B/16 only     | 82.72 | 68.11 | 74.20 | 67.75 |

| Raw logits w=0.50 | 84.50 | 70.93 | 76.67 | 70.24 |

| Std logits w=0.50 | 84.38 | 70.62 | 76.41 | 70.05 |

| Prob avg w=0.50   | 84.23 | 70.59 | 76.33 | 69.94 |



结论：Base 与 New class 同时提升，HM 从 74.20 提升到 76.67。这说明收益不只是增强 base fitting，也改善了 unseen-class generalization。



\---



\### 4. CoCoOp base-to-novel sanity evaluation



设置：



```text

Datasets = DTD, EuroSAT, OxfordPets

Seeds = 1, 2, 3

Backbones = RN101 + ViT-B/16

```



| Setting           |  Base |   New |    HM |   All |

| ----------------- | ----: | ----: | ----: | ----: |

| RN101 only        | 82.65 | 62.40 | 68.91 | 60.29 |

| ViT-B/16 only     | 83.14 | 72.61 | 77.08 | 66.34 |

| Raw logits w=0.50 | 87.05 | 73.24 | 78.70 | 69.18 |

| Raw logits w=0.75 | 86.54 | 74.59 | 79.59 | 69.49 |

| Std logits w=0.75 | 86.16 | 74.56 | 79.42 | 69.43 |



结论：CoCoOp 上也观察到明显正收益，说明 backbone complementarity 不是 CoOp 静态 prompt 的偶然现象。



\---



\### 5. Feature-level MLP fusion sanity result



第一版 feature-level fusion 使用：



```text

v\_fused = normalize(v\_vit + alpha \* MLP(\[v\_vit, v\_rn]))

```



当前 sanity 结果：



| Dataset    | Feature MLP HM | ViT-B/16 HM | Late Fusion HM |

| ---------- | -------------: | ----------: | -------------: |

| DTD        |          56.34 |       61.27 |          64.59 |

| EuroSAT    |          73.67 |       77.47 |          78.34 |

| OxfordPets |          93.44 |       94.01 |          95.06 |



结论：naive feature-level MLP fusion 不如 ViT-B/16，更不如 late fusion。该结果支持当前解释：RN101 与 ViT-B/16 在 prediction level 有互补性，但直接在 embedding 层融合可能破坏 CLIP image-text alignment。



\---



\## 运行指令



\### 环境



本项目基于 CoOp / CoCoOp 代码库和 Dassl。



典型依赖：



```text

Python 3.x

PyTorch

torchvision

Dassl.pytorch

CLIP

CoOp\_clean

```



服务器上的常见项目路径：



```text

/workspace/meta\_prompt\_1

```



或：



```text

/home/ubuntu/code/meta\_prompt\_1

```



常见数据集路径：



```text

/workspace/datasets

```



或：



```text

/home/ubuntu/datasets

```



请根据实际服务器路径修改脚本中的路径变量。



\---



\### 1. CoOp strict domain generalization



准备 ImageNet-source checkpoint 并评估 ImageNet variants：



```bash

bash scripts/dualenc/01\_prepare\_late\_fusion\_strict\_dg.sh

bash scripts/dualenc/02\_eval\_late\_fusion\_strict\_dg.sh

```



评估目标：



```text

ImageNet -> ImageNetV2

ImageNet -> ImageNet-Sketch

ImageNet -> ImageNet-A

ImageNet -> ImageNet-R

```



\---



\### 2. CoOp base-to-novel evaluation



sanity 版本：



```bash

bash scripts/dualenc/03\_run\_late\_fusion\_b2n\_sanity.sh

```



full 版本：



```bash

bash scripts/dualenc/04\_run\_late\_fusion\_b2n\_full.sh

```



full 版本评估 11 个数据集：



```text

caltech101

dtd

eurosat

fgvc\_aircraft

food101

imagenet

oxford\_flowers

oxford\_pets

stanford\_cars

sun397

ucf101

```



\---



\### 3. CoOp ImageNet-source cross-dataset transfer



```bash

bash scripts/dualenc/05\_eval\_late\_fusion\_xd\_imagenet\_source.sh

```



该脚本将 ImageNet-source checkpoint 评估到 10 个异构目标数据集：



```text

caltech101

oxford\_pets

dtd

eurosat

food101

oxford\_flowers

stanford\_cars

fgvc\_aircraft

ucf101

sun397

```



\---



\### 4. CoCoOp late fusion



生成 CoCoOp 双 backbone 配置：



```bash

bash scripts/dualenc\_cocoop/00\_create\_cocoop\_dualenc\_configs.sh

```



strict domain generalization：



```bash

bash scripts/dualenc\_cocoop/01\_run\_cocoop\_late\_fusion\_strict\_dg.sh

```



ImageNet-source cross-dataset transfer：



```bash

bash scripts/dualenc\_cocoop/02\_run\_cocoop\_late\_fusion\_xd\_imagenet\_source.sh

```



base-to-novel evaluation：



```bash

bash scripts/dualenc\_cocoop/03\_run\_cocoop\_late\_fusion\_b2n.sh

```



\---



\### 5. Feature-level MLP fusion baseline



安装或复制 dual-encoder trainer：



```bash

bash scripts/dualenc\_feature/00\_install\_dualenc\_feature.sh

```



运行 B2N sanity：



```bash

bash scripts/dualenc\_feature/01\_run\_dualenc\_feature\_b2n\_sanity.sh

```



汇总结果：



```bash

python scripts/dualenc\_feature/summarize\_dualenc\_feature\_b2n.py

```



\---



\## 结果文件



GitHub-ready summary tables 位于：



```text

summary\_tables/dualenc/

summary\_tables/dualenc\_cocoop/

summary\_tables/dualenc\_feature/

```



raw outputs、checkpoint、tensorboard 文件和原始 log 不应提交到 GitHub。



\---



\## GitHub 提交范围



建议提交：



```text

README.md

scripts/

summary\_tables/

third\_party/CoOp\_clean/trainers/coop\_dualenc.py

third\_party/CoOp\_clean/configs/trainers/

third\_party/CoOp\_clean/configs/datasets/

```



不要提交：



```text

datasets/

data/

third\_party/CoOp\_clean/output/

third\_party/CoOp\_clean/output\_\*/

logs/

tensorboard/

\*.pth

\*.pth.tar

\*.pt

\*.ckpt

\*.pkl

\*.npy

\*.npz

```



\---



\## 当前解释



当前证据支持以下判断：



1\. RN101 作为单模型弱于 ViT-B/16。

2\. 但是 RN101 仍然能为 ViT-B/16 提供互补预测信息。

3\. CoOp B2N、strict DG、ImageNet-source cross-dataset transfer 中，prediction-level late fusion 均提升 ViT-B/16。

4\. CoCoOp B2N sanity 结果说明该现象不局限于静态 prompt learning。

5\. Naive feature-level fusion 表现较弱，说明保留每个 backbone 自身的 CLIP alignment space 可能很重要。



当前工作不应被过度表述为对 MaPLe、PromptSRC、DPC 或 Promise 等强 prompt-learning 方法的直接替代。更合理的定位是：dual-backbone prediction fusion 是一个可能可以挂接到不同 prompt learner 上的 backbone-side complementary module。



\---



\## 后续工作



计划中的下一步：



1\. 将 late fusion 扩展到 MaPLe、PromptSRC 和更新的 prompt-robust 方法。

2\. 完成 CoCoOp strict domain generalization 和 ImageNet-source cross-dataset transfer。

3\. 补充 control experiments：



&#x20;  \* RN logits image-shuffle

&#x20;  \* RN logits class-shuffle

&#x20;  \* matched Gaussian logits noise

&#x20;  \* temperature / calibration controls

4\. 补充机制分析：



&#x20;  \* disagreement rescue analysis

&#x20;  \* confidence / entropy bin analysis

&#x20;  \* optimal fusion weight heatmap

&#x20;  \* source/task-dependent fusion ratio analysis

5\. 将固定融合权重替换为 validation-trained adaptive gating：



```text

w\_i = f(conf\_vit, conf\_rn, entropy\_vit, entropy\_rn, margin\_vit, margin\_rn, top1\_agreement)

```



不能使用 target test labels 来选择或训练融合权重。



\---



\## 引用



本仓库目前是活跃研究代码库。正式论文或预印本发布后会补充 citation 信息。



