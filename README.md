# DualBackboneFusion：面向 CLIP Prompt Transfer 的双主干预测层融合

[中文](#中文) | [English](#english)

---

<a id="中文"></a>

## 1. 项目定位

本仓库研究 **CLIP prompt learning 中的视觉主干互补性**。核心观察是：虽然 **RN101** 作为单模型通常弱于 **ViT-B/16**，但它的预测分布可能包含 ViT-B/16 没有利用到的互补信息。因此，本项目首先不直接混合 image embedding，而是在 prediction / logit 层进行 dual-backbone late fusion：

```text
fused_logits = w * logits_vit + (1 - w) * logits_rn
```

其中：

```text
w = 1.0  -> ViT-B/16 only
w = 0.0  -> RN101 only
```

当前项目定位为：

```text
Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer
```

当前仓库包含三条主要实验线：

1. **Fixed Late Fusion**：验证 RN101 与 ViT-B/16 是否存在稳定的 prediction-level complementarity。
2. **Feature-level Fusion Baseline**：验证直接融合 embedding 是否会破坏 CLIP image-text alignment。
3. **Reliability Prior Cache**：在已有 logits 基础上构建 backbone reliability prior，尝试自动替代固定融合权重。

---

## 2. 研究动机

CoOp、CoCoOp 等 CLIP prompt learning 方法通常一次只评估一个 visual backbone，例如 RN50、RN101 或 ViT-B/16。然而，不同视觉主干具有不同的视觉归纳偏置。

CNN 风格主干更容易关注局部纹理、边缘和目标本体特征；ViT 风格主干更容易利用 patch 之间的全局交互和长程上下文。这两类行为并不是绝对优劣关系，而可能在不同样本、不同类别和不同目标域上表现出不同错误模式。

因此，本项目关注以下问题：

> 一个较弱的 CLIP visual backbone 是否能为较强的 visual backbone 提供互补预测信号？

第一阶段采用 prediction-level late fusion，而不是 feature-level fusion，是因为直接混合 embedding 可能破坏每个 backbone 自身的 CLIP image-text alignment space。当前实验也显示，naive feature-level MLP fusion 明显不如 prediction-level late fusion。

---

## 3. 方法概述

给定两个在相同协议下训练的 CLIP prompt learners：

```text
RN101-based prompt learner
ViT-B/16-based prompt learner
```

在同一个 dataloader 上分别计算 logits，然后做融合：

```text
fused = w * logits_vit + (1 - w) * logits_rn
```

当前支持三种融合接口：

```text
raw_logits   : 直接融合原始 logits
std_logits   : 对每个样本的 logits 标准化后融合
prob_avg     : softmax probability 后平均
```

常用固定权重为：

```text
w = 0.00, 0.25, 0.50, 0.75, 1.00
```

Fixed late fusion 不是最终目标，而是第一阶段用来验证 RN101 与 ViT-B/16 是否具有 prediction-level complementarity 的诊断工具。

---

## 4. Reliability Prior Cache：两条新路线

在 fixed late fusion 证明双主干存在互补性之后，本仓库进一步加入 **Reliability Prior Cache** 阶段。目标是：在不重新训练 CLIP、CoOp 或 visual backbone 的情况下，利用已经缓存的 RN101 / ViT-B/16 logits 构建 backbone reliability prior，并评估它是否能优于固定 late fusion。

### 4.1 ImageNet Cache route

ImageNet Cache 是一条 **wide-source prior** 路线。它使用 ImageNet 作为宽域 source cache，估计 RN101 与 ViT-B/16 在不同粒度上的可靠性，并迁移到 B2N target。

Clean B2N 设置：

```text
source = b2n / imagenet / split_all / seed1-3
target = 10 non-ImageNet B2N datasets
metric = HM(base,new)
```

Clean strict DG 设置：

```text
source = strict_dg / imagenet / unknown / seed1-3
target = ImageNetV2 / ImageNet-Sketch / ImageNet-A / ImageNet-R
metric = accuracy
```

### 4.2 Meta LODO Cache route

Meta LODO Cache 是一条 **multi-source meta-prior** 路线。它不依赖单一 ImageNet source，而是从多个 B2N 数据集构建 leave-one-dataset-out 的 reliability prior。对每个 target dataset，使用其它 B2N 数据集作为 source cache。

Clean B2N 设置：

```text
for each target:
    source = other B2N datasets
    source splits = all / base / new
    source seeds = 1 / 2 / 3
target = current B2N dataset
metric = HM(base,new)
```

### 4.3 Reliability Prior Cache 主结果

| Task | Line | Fusion | Fixed | Dataset Cache | Class Cache | Oracle |
|---|---|---|---:|---:|---:|---:|
| B2N | ImageNet Cache | prob_avg | 75.99 | 76.86 | 76.92 | 77.41 |
| B2N | Meta LODO Cache | prob_avg | 75.99 | 76.87 | 76.66 | 77.41 |
| strict DG | ImageNet Cache | prob_avg | 59.29 | 59.18 | 59.06 | 59.61 |

当前结论：

- Reliability Prior Cache 在 B2N 上有稳定正信号。
- `dataset_cache` 在 ImageNet Cache 和 Meta LODO Cache 两条线上均稳定为正。
- `prob_avg` 是当前最强、最安全的 fusion interface。
- ImageNet Cache 的 `class_cache` 在 `prob_avg` 下达到最佳 B2N 结果。
- Meta LODO Cache 的 `dataset_cache` 在 `prob_avg` 下最稳。
- `raw_logits + class_cache` 明显不稳定，说明 class-wise reliability transfer 对 calibration 非常敏感。
- strict ImageNet-derived DG 当前没有被 ImageNet Cache 改进。

因此当前结论应表述为：

```text
Reliability Prior Cache is effective for B2N, but does not yet provide robust improvement on strict ImageNet-derived DG targets.
```

DG 后续需要重新设计 domain-aware 或 shift-aware routing，而不是继续机械堆 cache。

---

## 5. 仓库结构

```text
DualBackboneFusion/
├── README.md
├── docs/
│   └── README_RPC_STAGE_UPDATE.md
├── scripts/
│   ├── dualenc/
│   ├── dualenc_cocoop/
│   ├── dualenc_feature/
│   ├── dualenc_cache/
│   └── dualenc_rpc/
├── summary_tables/
├── outputs/
│   └── reliability_prior_cache/
│       ├── paired_logits_manifest.csv
│       └── reports/
└── third_party/
    └── CoOp_clean/
```

---

## 6. 当前主要结果

### 6.1 CoOp fixed late fusion：strict domain generalization

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.39 |
| ViT-B/16 only | 58.50 |
| Raw logits w=0.75 | 59.50 |
| Std logits w=0.75 | 59.45 |
| Prob avg w=0.75 | 59.30 |

观察：RN101 单模型明显弱于 ViT-B/16，但作为 25% 辅助分支可以提升 ViT-B/16。strict DG 中最优固定权重更偏 ViT-dominant。

### 6.2 CoOp fixed late fusion：ImageNet-source cross-dataset transfer

| Setting | Accuracy |
|---|---:|
| RN101 only | 55.96 |
| ViT-B/16 only | 61.82 |
| Raw logits w=0.50 | 63.03 |
| Raw logits w=0.75 | 63.26 |
| Std logits w=0.75 | 63.25 |
| Prob avg w=0.50 | 63.63 |

观察：prediction-level fusion 在 cross-dataset transfer 中提升更明显，`prob_avg w=0.50` 比 ViT-B/16 提升约 +1.81。

### 6.3 CoOp fixed late fusion：base-to-novel full evaluation

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 79.83 | 62.52 | 69.20 | 63.07 |
| ViT-B/16 only | 82.72 | 68.11 | 74.20 | 67.75 |
| Raw logits w=0.50 | 84.50 | 70.93 | 76.67 | 70.24 |
| Std logits w=0.50 | 84.38 | 70.62 | 76.41 | 70.05 |
| Prob avg w=0.50 | 84.23 | 70.59 | 76.33 | 69.94 |

观察：Base 与 New class 均提升，HM 从 74.20 提升到 76.67，说明提升不只是 base fitting，而是也有利于 unseen-class generalization。

### 6.4 CoCoOp fixed late fusion：strict domain generalization

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.98 |
| ViT-B/16 only | 58.86 |
| Raw logits w=0.75 | 59.81 |
| Std logits w=0.75 | 59.73 |
| Prob avg w=0.75 | 59.65 |

观察：CoCoOp strict DG 中也存在双主干互补性。最强固定结果是 `raw_logits w=0.75`，比 ViT-B/16 高约 +0.95。

### 6.5 CoCoOp fixed late fusion：base-to-novel full evaluation

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 82.65 | 62.40 | 68.91 | 60.29 |
| ViT-B/16 only | 83.14 | 72.61 | 77.08 | 66.34 |
| Raw logits w=0.50 | 87.05 | 73.24 | 78.70 | 69.18 |
| Raw logits w=0.75 | 86.54 | 74.59 | 79.59 | 69.49 |
| Std logits w=0.50 | 86.67 | 72.41 | 78.03 | 69.05 |
| Std logits w=0.75 | 86.16 | 74.56 | 79.42 | 69.43 |
| Prob avg w=0.50 | 86.93 | 73.22 | 78.71 | 68.73 |
| Prob avg w=0.75 | 85.88 | 74.02 | 78.94 | 68.94 |

观察：CoCoOp B2N full 结果进一步说明，RN101 与 ViT-B/16 的互补性不局限于 CoOp。最强 HM 是 `raw_logits w=0.75`，从 ViT-B/16 的 77.08 提升到 79.59，约 +2.51 HM。

### 6.6 Feature-level MLP fusion sanity

第一版 feature-level fusion 使用：

```text
v_fused = normalize(v_vit + alpha * MLP([v_vit, v_rn]))
```

| Dataset | Feature MLP HM | ViT-B/16 HM | Late Fusion HM |
|---|---:|---:|---:|
| DTD | 56.34 | 61.27 | 64.59 |
| EuroSAT | 73.67 | 77.47 | 78.34 |
| OxfordPets | 93.44 | 94.01 | 95.06 |

观察：naive feature-level MLP fusion 不如 ViT-B/16，更不如 late fusion。这支持当前解释：RN101 与 ViT-B/16 的互补性主要体现在 prediction distribution，而直接混合 embedding 可能破坏 CLIP image-text alignment。

---

## 7. 运行说明

### 7.1 环境

本项目基于 CoOp / CoCoOp / Dassl 代码体系。

典型依赖：

```text
Python 3.x
PyTorch
torchvision
Dassl.pytorch
CLIP
CoOp_clean
```

服务器项目根目录通常为：

```text
/workspace/meta_prompt_1
```

或：

```text
/home/ubuntu/code/meta_prompt_1
```

数据集目录通常为：

```text
/workspace/datasets
```

或：

```text
/home/ubuntu/datasets
```

不同服务器上需要根据实际路径修改脚本中的 `PROJECT_ROOT`、`DATASET_ROOT` 等变量。

### 7.2 CoOp fixed late fusion

```bash
bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh
bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
bash scripts/dualenc/04_run_late_fusion_b2n_full.sh
bash scripts/dualenc/05_eval_late_fusion_xd_imagenet_source.sh
```

### 7.3 CoCoOp late fusion

```bash
bash scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh
bash scripts/dualenc_cocoop/01_run_cocoop_late_fusion_strict_dg.sh
bash scripts/dualenc_cocoop/02_run_cocoop_late_fusion_xd_imagenet_source.sh
bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh
```

### 7.4 Feature-level MLP fusion baseline

```bash
bash scripts/dualenc_feature/00_install_dualenc_feature.sh
bash scripts/dualenc_feature/01_run_dualenc_feature_b2n_sanity.sh
python scripts/dualenc_feature/summarize_dualenc_feature_b2n.py
```

### 7.5 Reliability Prior Cache

标准化 logits cache：

```bash
PROJECT_ROOT=/workspace/meta_prompt_1 \
CACHE_ROOT=outputs/dualenc_cache/logits \
bash scripts/dualenc_rpc/run_rpc_00_standardize_logits.sh
```

B2N ImageNet Cache clean：

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_01_imagenet_cache_clean_b2n.sh
done
```

B2N Meta LODO Cache clean：

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_02_meta_lodo_b2n_clean.sh
done
```

strict DG ImageNet Cache clean：

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_03_imagenet_cache_clean_strict_dg.sh
done
```

---

## 8. 结果文件

GitHub-ready summary tables are stored under:

```text
summary_tables/dualenc/
summary_tables/dualenc_cocoop/
summary_tables/dualenc_feature/
outputs/reliability_prior_cache/reports/
```

Reliability Prior Cache manifest:

```text
outputs/reliability_prior_cache/paired_logits_manifest.csv
```

注意：manifest 中可能包含服务器绝对路径，仅用于 provenance，不代表 GitHub 仓库内包含对应 `.npz` logits 文件。

---

## 9. 建议提交与不提交的文件

建议提交：

```text
README.md
docs/
scripts/
summary_tables/
outputs/reliability_prior_cache/reports/
outputs/reliability_prior_cache/paired_logits_manifest.csv
third_party/CoOp_clean/trainers/coop_dualenc.py
third_party/CoOp_clean/configs/trainers/
third_party/CoOp_clean/configs/datasets/
```

不要提交：

```text
datasets/
data/
logs/
tensorboard/
wandb/
outputs/dualenc_cache/logits/
outputs/reliability_prior_cache/paired_logits/
third_party/CoOp_clean/output/
third_party/CoOp_clean/output_*/
*.pth
*.pth.tar
*.pt
*.ckpt
*.pkl
*.npy
*.npz
*.tar
*.tar.gz
```

---

## 10. 当前解释与下一步

当前证据支持：

1. RN101 作为单模型弱于 ViT-B/16。
2. RN101 的 prediction distribution 仍然可以为 ViT-B/16 提供互补信息。
3. Prediction-level late fusion 在 CoOp 和 CoCoOp 的 B2N、strict DG、ImageNet-source transfer 等设置中均有正收益。
4. Feature-level MLP fusion 的失败说明保留各 backbone 自身 CLIP alignment space 很重要。
5. ImageNet Cache 和 Meta LODO Cache 两条路线都在 B2N 上获得正信号，说明 reliability prior 不是单一设置的偶然现象。
6. 当前 strict DG 上的 ImageNet Cache 尚未超过 fixed fusion，因此 DG 需要更强的 domain-aware / shift-aware routing。

当前方法不应被过度表述为 MaPLe、PromptSRC、DPC 或 Promise 等强 prompt-learning 方法的替代品。更合适的定位是：dual-backbone prediction fusion 是一个可以附加到不同 prompt learner 上的正交模块。

后续计划：

1. 扩展到更强 prompt learner，例如 MaPLe、PromptSRC 或其他 robust prompt learning 方法。
2. 补充 control experiments：
   - RN logits image-shuffle
   - RN logits class-shuffle
   - matched Gaussian logits noise
   - temperature / calibration controls
3. 补充机制分析：
   - disagreement rescue analysis
   - confidence / entropy bin analysis
   - optimal fusion weight heatmap
   - source / task-dependent fusion ratio analysis
4. 重新设计 DG routing：
   - domain-aware source selection
   - shift-aware reliability prior
   - ImageNet-A negative transfer analysis

---

## 11. Citation

This repository is an active research codebase. A formal citation will be added after the paper or preprint is available.

---

<a id="english"></a>

# DualBackboneFusion: Prediction-Level Dual-Backbone Fusion for CLIP Prompt Transfer

[中文](#中文) | [English](#english)

---

## 1. Project Positioning

This repository studies **visual-backbone complementarity in CLIP prompt learning**.

The central observation is that a weaker CLIP visual backbone, such as **RN101**, can still provide complementary prediction information to a stronger **ViT-B/16** backbone. Instead of directly merging image embeddings, this project first evaluates prediction-level dual-backbone fusion:

```text
fused_logits = w * logits_vit + (1 - w) * logits_rn
```

where:

```text
w = 1.0  -> ViT-B/16 only
w = 0.0  -> RN101 only
```

The project is currently positioned as:

```text
Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer
```

This repository currently contains three main experimental lines:

1. **Fixed Late Fusion**: verifying whether RN101 and ViT-B/16 exhibit stable prediction-level complementarity.
2. **Feature-level Fusion Baseline**: testing whether direct embedding fusion damages CLIP image-text alignment.
3. **Reliability Prior Cache**: building backbone reliability priors from cached logits to improve over fixed late fusion.

---

## 2. Motivation

CLIP prompt learning methods such as CoOp and CoCoOp usually evaluate one visual backbone at a time, such as RN50, RN101, or ViT-B/16. However, different visual backbones may encode different visual inductive biases.

CNN-style backbones tend to emphasize local textures, edges, and object-level patterns, while ViT-style backbones can exploit global patch interactions and long-range context. These behaviors are not strictly better or worse in all cases. Instead, they may fail on different examples, categories, and target domains.

This motivates the question:

> Can a weaker CLIP visual backbone provide complementary prediction signals to a stronger backbone under prompt learning?

The first stage uses prediction-level late fusion instead of feature-level fusion, because directly mixing embeddings may disturb each backbone's own CLIP image-text alignment space. Current experiments also show that naive feature-level MLP fusion is clearly weaker than prediction-level late fusion.

---

## 3. Main Method

Given two CLIP prompt learners trained under the same protocol, we evaluate both models on the same dataloader and combine their logits:

```text
fused = w * logits_vit + (1 - w) * logits_rn
```

The current implementation supports three fusion interfaces:

```text
raw_logits   : direct raw-logit fusion
std_logits   : per-sample standardized-logit fusion
prob_avg     : probability-level averaging
```

Common fixed weights are:

```text
w = 0.00, 0.25, 0.50, 0.75, 1.00
```

Fixed late fusion is not the final goal. It is used as a first-stage diagnostic tool to verify whether RN101 and ViT-B/16 have meaningful prediction-level complementarity.

---

## 4. Reliability Prior Cache: Two New Routes

After fixed fusion confirms dual-backbone complementarity, this repository further introduces a **Reliability Prior Cache** stage. The goal is to build backbone reliability priors from cached RN101 / ViT-B/16 logits and evaluate whether they can improve over fixed late fusion without retraining CLIP, CoOp, or the visual backbones.

### 4.1 ImageNet Cache route

ImageNet Cache is a **wide-source prior** route. It uses ImageNet as a broad source cache to estimate RN101 / ViT-B/16 reliability and transfer it to B2N targets.

Clean B2N setting:

```text
source = b2n / imagenet / split_all / seed1-3
target = 10 non-ImageNet B2N datasets
metric = HM(base,new)
```

Clean strict DG setting:

```text
source = strict_dg / imagenet / unknown / seed1-3
target = ImageNetV2 / ImageNet-Sketch / ImageNet-A / ImageNet-R
metric = accuracy
```

### 4.2 Meta LODO Cache route

Meta LODO Cache is a **multi-source meta-prior** route. It does not rely on a single ImageNet source. Instead, it builds a leave-one-dataset-out reliability prior from multiple B2N datasets.

Clean B2N setting:

```text
for each target:
    source = other B2N datasets
    source splits = all / base / new
    source seeds = 1 / 2 / 3
target = current B2N dataset
metric = HM(base,new)
```

### 4.3 Main Reliability Prior Cache results

| Task | Line | Fusion | Fixed | Dataset Cache | Class Cache | Oracle |
|---|---|---|---:|---:|---:|---:|
| B2N | ImageNet Cache | prob_avg | 75.99 | 76.86 | 76.92 | 77.41 |
| B2N | Meta LODO Cache | prob_avg | 75.99 | 76.87 | 76.66 | 77.41 |
| strict DG | ImageNet Cache | prob_avg | 59.29 | 59.18 | 59.06 | 59.61 |

Stage interpretation:

- Reliability Prior Cache is effective on B2N.
- `dataset_cache` is consistently positive under both ImageNet Cache and Meta LODO Cache.
- `prob_avg` is currently the strongest and safest fusion interface.
- `raw_logits + class_cache` is unstable, suggesting that class-wise reliability transfer is highly calibration-sensitive.
- strict ImageNet-derived DG is not improved by the current ImageNet Cache and requires stronger domain-aware or shift-aware routing.

---

## 5. Repository Structure

```text
DualBackboneFusion/
├── README.md
├── docs/
│   └── README_RPC_STAGE_UPDATE.md
├── scripts/
│   ├── dualenc/
│   ├── dualenc_cocoop/
│   ├── dualenc_feature/
│   ├── dualenc_cache/
│   └── dualenc_rpc/
├── summary_tables/
├── outputs/
│   └── reliability_prior_cache/
└── third_party/
    └── CoOp_clean/
```

---

## 6. Main Results

### 6.1 CoOp fixed late fusion: strict domain generalization

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.39 |
| ViT-B/16 only | 58.50 |
| Raw logits w=0.75 | 59.50 |
| Std logits w=0.75 | 59.45 |
| Prob avg w=0.75 | 59.30 |

### 6.2 CoOp fixed late fusion: ImageNet-source cross-dataset transfer

| Setting | Accuracy |
|---|---:|
| RN101 only | 55.96 |
| ViT-B/16 only | 61.82 |
| Raw logits w=0.50 | 63.03 |
| Raw logits w=0.75 | 63.26 |
| Std logits w=0.75 | 63.25 |
| Prob avg w=0.50 | 63.63 |

### 6.3 CoOp fixed late fusion: base-to-novel full evaluation

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 79.83 | 62.52 | 69.20 | 63.07 |
| ViT-B/16 only | 82.72 | 68.11 | 74.20 | 67.75 |
| Raw logits w=0.50 | 84.50 | 70.93 | 76.67 | 70.24 |
| Std logits w=0.50 | 84.38 | 70.62 | 76.41 | 70.05 |
| Prob avg w=0.50 | 84.23 | 70.59 | 76.33 | 69.94 |

### 6.4 CoCoOp fixed late fusion: strict domain generalization

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.98 |
| ViT-B/16 only | 58.86 |
| Raw logits w=0.75 | 59.81 |
| Std logits w=0.75 | 59.73 |
| Prob avg w=0.75 | 59.65 |

### 6.5 CoCoOp fixed late fusion: base-to-novel full evaluation

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 82.65 | 62.40 | 68.91 | 60.29 |
| ViT-B/16 only | 83.14 | 72.61 | 77.08 | 66.34 |
| Raw logits w=0.50 | 87.05 | 73.24 | 78.70 | 69.18 |
| Raw logits w=0.75 | 86.54 | 74.59 | 79.59 | 69.49 |
| Std logits w=0.50 | 86.67 | 72.41 | 78.03 | 69.05 |
| Std logits w=0.75 | 86.16 | 74.56 | 79.42 | 69.43 |
| Prob avg w=0.50 | 86.93 | 73.22 | 78.71 | 68.73 |
| Prob avg w=0.75 | 85.88 | 74.02 | 78.94 | 68.94 |

### 6.6 Feature-level MLP fusion sanity

| Dataset | Feature MLP HM | ViT-B/16 HM | Late Fusion HM |
|---|---:|---:|---:|
| DTD | 56.34 | 61.27 | 64.59 |
| EuroSAT | 73.67 | 77.47 | 78.34 |
| OxfordPets | 93.44 | 94.01 | 95.06 |

---

## 7. Running Instructions

### 7.1 Fixed late fusion

```bash
bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh
bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
bash scripts/dualenc/04_run_late_fusion_b2n_full.sh
bash scripts/dualenc/05_eval_late_fusion_xd_imagenet_source.sh
```

### 7.2 CoCoOp late fusion

```bash
bash scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh
bash scripts/dualenc_cocoop/01_run_cocoop_late_fusion_strict_dg.sh
bash scripts/dualenc_cocoop/02_run_cocoop_late_fusion_xd_imagenet_source.sh
bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh
```

### 7.3 Reliability Prior Cache

```bash
PROJECT_ROOT=/workspace/meta_prompt_1 \
CACHE_ROOT=outputs/dualenc_cache/logits \
bash scripts/dualenc_rpc/run_rpc_00_standardize_logits.sh
```

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_01_imagenet_cache_clean_b2n.sh
done
```

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_02_meta_lodo_b2n_clean.sh
done
```

```bash
for MODE in std_logits raw_logits prob_avg; do
  PROJECT_ROOT=/workspace/meta_prompt_1 \
  FUSION_MODE=$MODE \
  bash scripts/dualenc_rpc/run_rpc_03_imagenet_cache_clean_strict_dg.sh
done
```

---

## 8. Result Files

GitHub-ready summary tables are stored under:

```text
summary_tables/
outputs/reliability_prior_cache/reports/
```

The Reliability Prior Cache manifest is stored at:

```text
outputs/reliability_prior_cache/paired_logits_manifest.csv
```

The manifest may contain server-specific absolute paths and is mainly used for provenance. The corresponding `.npz` logits are not included in this repository.

---

## 9. Current Interpretation and Next Steps

The current evidence supports the following interpretation:

1. RN101 is weaker than ViT-B/16 as a standalone CLIP prompt learner.
2. RN101 can still provide complementary prediction information to ViT-B/16.
3. Prediction-level late fusion improves ViT-B/16 under CoOp and CoCoOp B2N / strict DG settings.
4. Naive feature-level fusion is weaker than prediction-level fusion, suggesting that preserving each backbone's own CLIP alignment space is important.
5. ImageNet Cache and Meta LODO Cache both provide positive B2N signals, showing that reliability prior is not a single-source artifact.
6. strict DG still requires stronger domain-aware or shift-aware routing.

The current method should not be overclaimed as a universal replacement for stronger prompt-learning methods such as MaPLe, PromptSRC, DPC, or Promise. A more appropriate claim is that dual-backbone prediction fusion is a potentially orthogonal module that can be attached to different prompt learners.

---

## 10. Citation

This repository is an active research codebase. A formal citation will be added after the paper or preprint is available.
