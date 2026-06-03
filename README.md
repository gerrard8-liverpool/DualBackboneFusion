# DualBackboneFusion

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## DualBackboneFusion: Prediction-Level Dual-Backbone Fusion for CLIP Prompt Transfer

This repository studies **visual-backbone complementarity in CLIP prompt learning**.

The central observation is that a weaker CLIP visual backbone, such as **RN101**, can still provide complementary prediction information to a stronger **ViT-B/16** backbone. Instead of directly merging feature embeddings, this project first evaluates a simple but effective **prediction-level late fusion** strategy:

    fused_logits = w * logits_vit + (1 - w) * logits_rn

where `w = 1.0` corresponds to pure ViT-B/16 and `w = 0.0` corresponds to pure RN101.

The current project is positioned as:

    Backbone Complementarity in CLIP Prompt Learning

or more specifically:

    Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer

---

## Motivation

CLIP prompt learning methods such as CoOp and CoCoOp usually evaluate one visual backbone at a time, such as RN50, RN101, or ViT-B/16. However, different visual backbones have different visual inductive biases.

CNN-style backbones tend to emphasize local visual patterns, texture, edges, and object-level cues. ViT-style backbones can exploit global patch interactions and long-range contextual information. These behaviors are not strictly better or worse in all cases. Instead, they may fail on different examples and different target domains.

This motivates the following research question:

> Can a weaker CLIP visual backbone provide complementary prediction signals to a stronger backbone under prompt learning?

The first step in this repository is deliberately simple. We do not directly mix feature embeddings, because this may disturb each backbone's own CLIP image-text alignment space. Instead, each backbone keeps its own CLIP-aligned representation, and fusion is performed at the prediction/logit level.

---

## Main Idea

Given two CLIP prompt learners trained under the same protocol:

- RN101-based prompt learner
- ViT-B/16-based prompt learner

we evaluate both models on the same dataloader and combine their output logits:

    fused = w * logits_vit + (1 - w) * logits_rn

The current implementation supports three fusion variants:

    raw_logits
    std_logits
    prob_avg

and commonly evaluates the following fixed weights:

    w = 0.00, 0.25, 0.50, 0.75, 1.00

Fixed-weight late fusion is not the final goal. It is used as a first-stage diagnostic tool to verify whether RN101 and ViT-B/16 have meaningful prediction-level complementarity.

---

## Repository Structure

    DualBackboneFusion/
    ├── README.md
    ├── scripts/
    │   ├── dualenc/
    │   ├── dualenc_cocoop/
    │   └── dualenc_feature/
    ├── summary_tables/
    │   ├── dualenc/
    │   ├── dualenc_cocoop/
    │   └── dualenc_feature/
    └── third_party/
        └── CoOp_clean/
            ├── trainers/
            └── configs/

### Main branches

| Branch | Directory | Purpose |
|---|---|---|
| CoOp late fusion | `scripts/dualenc/` | Main prediction-level fusion experiments |
| CoCoOp late fusion | `scripts/dualenc_cocoop/` | Extension to dynamic prompt learning |
| Feature-level fusion | `scripts/dualenc_feature/` | Negative baseline for embedding-level MLP fusion |

---

## Current Results

### 1. CoOp strict domain generalization

Setting:

    Source = ImageNet
    Targets = ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R
    Backbones = RN101 + ViT-B/16
    Seeds = 1, 2, 3

| Setting | Accuracy |
|---|---:|
| RN101 only | 47.39 |
| ViT-B/16 only | 58.50 |
| Raw logits w=0.75 | 59.50 |
| Std logits w=0.75 | 59.45 |
| Prob avg w=0.75 | 59.30 |

Observation: RN101 is much weaker than ViT-B/16 as a standalone model, but adding it as a 25% auxiliary branch improves ViT-B/16.

---

### 2. CoOp ImageNet-source cross-dataset transfer

Setting:

    Source = ImageNet
    Targets = caltech101, oxford_pets, dtd, eurosat, food101,
              oxford_flowers, stanford_cars, fgvc_aircraft, ucf101, sun397
    Backbones = RN101 + ViT-B/16

| Setting | Accuracy |
|---|---:|
| RN101 only | 55.96 |
| ViT-B/16 only | 61.82 |
| Raw logits w=0.50 | 63.03 |
| Raw logits w=0.75 | 63.26 |
| Std logits w=0.75 | 63.25 |
| Prob avg w=0.50 | 63.63 |

Observation: Prediction-level fusion gives a stronger gain in cross-dataset transfer. The best setting, `prob_avg w=0.50`, improves ViT-B/16 by about +1.81 points.

---

### 3. CoOp base-to-novel full evaluation

Setting:

    Datasets = caltech101, dtd, eurosat, fgvc_aircraft, food101, imagenet,
               oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101
    Seeds = 1, 2, 3

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 79.83 | 62.52 | 69.20 | 63.07 |
| ViT-B/16 only | 82.72 | 68.11 | 74.20 | 67.75 |
| Raw logits w=0.50 | 84.50 | 70.93 | 76.67 | 70.24 |
| Std logits w=0.50 | 84.38 | 70.62 | 76.41 | 70.05 |
| Prob avg w=0.50 | 84.23 | 70.59 | 76.33 | 69.94 |

Observation: Base and new class performance are both improved. The harmonic mean improves from 74.20 to 76.67.

---

### 4. CoCoOp base-to-novel sanity evaluation

Setting:

    Datasets = DTD, EuroSAT, OxfordPets
    Seeds = 1, 2, 3
    Backbones = RN101 + ViT-B/16

| Setting | Base | New | HM | All |
|---|---:|---:|---:|---:|
| RN101 only | 82.65 | 62.40 | 68.91 | 60.29 |
| ViT-B/16 only | 83.14 | 72.61 | 77.08 | 66.34 |
| Raw logits w=0.50 | 87.05 | 73.24 | 78.70 | 69.18 |
| Raw logits w=0.75 | 86.54 | 74.59 | 79.59 | 69.49 |
| Std logits w=0.75 | 86.16 | 74.56 | 79.42 | 69.43 |

Observation: CoCoOp also benefits from dual-backbone prediction-level fusion.

---

### 5. Feature-level MLP fusion sanity result

The first feature-level fusion baseline uses:

    v_fused = normalize(v_vit + alpha * MLP([v_vit, v_rn]))

Current sanity results:

| Dataset | Feature MLP HM | ViT-B/16 HM | Late Fusion HM |
|---|---:|---:|---:|
| DTD | 56.34 | 61.27 | 64.59 |
| EuroSAT | 73.67 | 77.47 | 78.34 |
| OxfordPets | 93.44 | 94.01 | 95.06 |

Observation: Naive feature-level MLP fusion underperforms ViT-B/16 and is clearly worse than late fusion. This supports the interpretation that directly mixing image embeddings may damage CLIP image-text alignment.

---

## Running Instructions

### Environment

This project is built on top of the CoOp/CoCoOp codebase and Dassl.

Typical requirements:

    Python 3.x
    PyTorch
    torchvision
    Dassl.pytorch
    CLIP
    CoOp_clean

Typical server project root:

    /workspace/meta_prompt_1

or:

    /home/ubuntu/code/meta_prompt_1

Typical dataset root:

    /workspace/datasets

or:

    /home/ubuntu/datasets

Please adjust paths inside the scripts according to your local or server environment.

---

### 1. CoOp strict domain generalization

    bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
    bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh

### 2. CoOp base-to-novel evaluation

    bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
    bash scripts/dualenc/04_run_late_fusion_b2n_full.sh

### 3. CoOp ImageNet-source cross-dataset transfer

    bash scripts/dualenc/05_eval_late_fusion_xd_imagenet_source.sh

### 4. CoCoOp late fusion

    bash scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh
    bash scripts/dualenc_cocoop/01_run_cocoop_late_fusion_strict_dg.sh
    bash scripts/dualenc_cocoop/02_run_cocoop_late_fusion_xd_imagenet_source.sh
    bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh

### 5. Feature-level MLP fusion baseline

    bash scripts/dualenc_feature/00_install_dualenc_feature.sh
    bash scripts/dualenc_feature/01_run_dualenc_feature_b2n_sanity.sh
    python scripts/dualenc_feature/summarize_dualenc_feature_b2n.py

---

## Result Summaries

GitHub-ready summary tables are stored under:

    summary_tables/dualenc/
    summary_tables/dualenc_cocoop/
    summary_tables/dualenc_feature/

Raw outputs, checkpoints, tensorboard files, and raw logs should not be committed to GitHub.

---

## What Should Be Committed

Recommended files:

    README.md
    scripts/
    summary_tables/
    third_party/CoOp_clean/trainers/coop_dualenc.py
    third_party/CoOp_clean/configs/trainers/
    third_party/CoOp_clean/configs/datasets/

Do not commit:

    datasets/
    data/
    third_party/CoOp_clean/output/
    third_party/CoOp_clean/output_*/
    logs/
    tensorboard/
    *.pth
    *.pth.tar
    *.pt
    *.ckpt
    *.pkl
    *.npy
    *.npz

---

## Current Interpretation

The current evidence supports the following interpretation:

1. RN101 is weaker than ViT-B/16 as a standalone CLIP prompt learner.
2. RN101 can still provide complementary prediction information to ViT-B/16.
3. Prediction-level late fusion improves ViT-B/16 under CoOp B2N, strict DG, and ImageNet-source cross-dataset transfer.
4. CoCoOp B2N sanity results suggest that this complementarity is not limited to static prompt learning.
5. Naive feature-level fusion is weaker than prediction-level fusion, suggesting that preserving each backbone's own CLIP alignment space is important.

This project should not be overclaimed as a universal replacement for stronger prompt-learning methods such as MaPLe, PromptSRC, DPC, or Promise. A more appropriate claim is that dual-backbone prediction fusion is a potentially orthogonal module that can be attached to different prompt learners.

---

## Future Work

Planned next steps:

1. Extend late fusion to stronger prompt learners such as MaPLe, PromptSRC, and newer prompt-robust methods.
2. Complete CoCoOp strict domain generalization and ImageNet-source cross-dataset transfer.
3. Add control experiments:
   - RN logits image-shuffle
   - RN logits class-shuffle
   - matched Gaussian logits noise
   - temperature or calibration controls
4. Add mechanism analysis:
   - disagreement rescue analysis
   - confidence/entropy bin analysis
   - optimal fusion weight heatmap
   - source/task-dependent fusion ratio analysis
5. Replace fixed fusion weights with validation-trained adaptive gating:

    w_i = f(conf_vit, conf_rn, entropy_vit, entropy_rn, margin_vit, margin_rn, top1_agreement)

Target test labels must not be used for selecting or training fusion weights.

---

## Citation

This repository is an active research codebase. A formal citation will be added after the paper or preprint is available.

---

<a id="中文"></a>

# DualBackboneFusion：面向 CLIP Prompt Transfer 的双主干预测层融合

本仓库研究 **CLIP prompt learning 中的视觉主干互补性**。

核心观察是：虽然 **RN101** 作为单模型通常弱于 **ViT-B/16**，但它的预测分布可能包含 ViT-B/16 没有利用到的互补信息。因此，本项目第一阶段不直接混合 embedding，而是在 prediction/logit 层做 late fusion：

    fused_logits = w * logits_vit + (1 - w) * logits_rn

其中 `w = 1.0` 表示纯 ViT-B/16，`w = 0.0` 表示纯 RN101。

当前项目定位为：

    Backbone Complementarity in CLIP Prompt Learning

或：

    Prediction-Level Dual-Backbone Fusion for Robust CLIP Prompt Transfer

---

## 研究动机

CoOp、CoCoOp 等 CLIP prompt learning 方法通常一次只评估一个 visual backbone，例如 RN50、RN101 或 ViT-B/16。然而，不同视觉主干具有不同的视觉归纳偏置。

CNN 风格主干更容易关注局部视觉模式、纹理、边缘和目标本体特征；ViT 风格主干更容易利用 patch 之间的全局交互和长程上下文信息。这两类行为不是绝对优劣关系，而可能在不同样本、不同目标域上表现出不同错误模式。

因此，本项目关注以下问题：

> 一个较弱的 CLIP visual backbone 是否能为较强的 visual backbone 提供互补预测信号？

本仓库的第一步故意采用简单方案：不直接混合 feature embedding，因为这可能破坏每个 backbone 自身的 CLIP image-text alignment space。相反，每个 backbone 保留自己的 CLIP 对齐空间，只在最终 prediction/logit 层融合。

---

## 核心方法

给定两个在相同协议下训练的 CLIP prompt learners：

- RN101-based prompt learner
- ViT-B/16-based prompt learner

在同一个 dataloader 上分别计算 logits，然后做融合：

    fused = w * logits_vit + (1 - w) * logits_rn

当前实现支持三种融合方式：

    raw_logits
    std_logits
    prob_avg

常用固定权重为：

    w = 0.00, 0.25, 0.50, 0.75, 1.00

固定权重 late fusion 不是最终目标，而是第一阶段用来验证 RN101 与 ViT-B/16 是否具有 prediction-level complementarity 的诊断工具。

---

## 当前结果概览

| Setting | Main result |
|---|---:|
| CoOp strict DG | ViT-B/16 58.50 -> best fusion 59.50 |
| CoOp ImageNet-source XD | ViT-B/16 61.82 -> best fusion 63.63 |
| CoOp B2N full | ViT-B/16 HM 74.20 -> best fusion HM 76.67 |
| CoCoOp B2N sanity | ViT-B/16 HM 77.08 -> best fusion HM 79.59 |

---

## 运行指令

### CoOp strict domain generalization

    bash scripts/dualenc/01_prepare_late_fusion_strict_dg.sh
    bash scripts/dualenc/02_eval_late_fusion_strict_dg.sh

### CoOp base-to-novel evaluation

    bash scripts/dualenc/03_run_late_fusion_b2n_sanity.sh
    bash scripts/dualenc/04_run_late_fusion_b2n_full.sh

### CoOp ImageNet-source cross-dataset transfer

    bash scripts/dualenc/05_eval_late_fusion_xd_imagenet_source.sh

### CoCoOp late fusion

    bash scripts/dualenc_cocoop/00_create_cocoop_dualenc_configs.sh
    bash scripts/dualenc_cocoop/01_run_cocoop_late_fusion_strict_dg.sh
    bash scripts/dualenc_cocoop/02_run_cocoop_late_fusion_xd_imagenet_source.sh
    bash scripts/dualenc_cocoop/03_run_cocoop_late_fusion_b2n.sh

### Feature-level MLP fusion baseline

    bash scripts/dualenc_feature/00_install_dualenc_feature.sh
    bash scripts/dualenc_feature/01_run_dualenc_feature_b2n_sanity.sh
    python scripts/dualenc_feature/summarize_dualenc_feature_b2n.py

---

## 当前解释

当前证据支持以下判断：

1. RN101 作为单模型弱于 ViT-B/16。
2. 但是 RN101 仍然能为 ViT-B/16 提供互补预测信息。
3. CoOp B2N、strict DG、ImageNet-source cross-dataset transfer 中，prediction-level late fusion 均提升 ViT-B/16。
4. CoCoOp B2N sanity 结果说明该现象不局限于静态 prompt learning。
5. Naive feature-level fusion 表现较弱，说明保留每个 backbone 自身的 CLIP alignment space 可能很重要。

当前工作不应被过度表述为对 MaPLe、PromptSRC、DPC 或 Promise 等强 prompt-learning 方法的直接替代。更合理的定位是：dual-backbone prediction fusion 是一个可能可以挂接到不同 prompt learner 上的 backbone-side complementary module。

---

## 后续工作

计划中的下一步：

1. 将 late fusion 扩展到 MaPLe、PromptSRC 和更新的 prompt-robust 方法。
2. 完成 CoCoOp strict domain generalization 和 ImageNet-source cross-dataset transfer。
3. 补充 RN logits image-shuffle、class-shuffle、matched Gaussian logits noise 等 control experiments。
4. 补充 disagreement rescue、confidence/entropy bin、optimal fusion weight heatmap 等机制分析。
5. 将固定融合权重替换为 validation-trained adaptive gating。

不能使用 target test labels 来选择或训练融合权重。

---

## 引用

本仓库目前是活跃研究代码库。正式论文或预印本发布后会补充 citation 信息。
