# Third-Party Notices

This file summarizes third-party components included in or referenced by
DualBackboneFusion. These components are provided under their own licenses.
Nothing in the repository-level LICENSE file is intended to relicense
third-party code, datasets, models, checkpoints, or external dependencies.

## 1. CoOp / CoCoOp Codebase

- Component: CoOp / CoCoOp prompt learning codebase
- Repository: https://github.com/KaiyangZhou/CoOp
- Original author: Kaiyang Zhou and contributors
- License: MIT License
- Local path in this repository: `third_party/CoOp_clean/`

The code under `third_party/CoOp_clean/` is derived from or based on the
CoOp / CoCoOp project. It remains subject to the original MIT License and
copyright notices from the upstream project.

Users should preserve the original copyright and license notices when copying,
modifying, or redistributing this component.

## 2. CLIP

- Component: CLIP model and related APIs
- Original project: OpenAI CLIP
- Repository: https://github.com/openai/CLIP
- License: Please refer to the original CLIP repository

This project may use CLIP models, APIs, or pretrained backbones such as
RN101 and ViT-B/16. These components are governed by their original license
and terms of use.

## 3. Datasets

This repository may contain scripts or configuration files for experiments on
datasets such as ImageNet, Caltech101, OxfordPets, StanfordCars, Food101,
FGVCAircraft, SUN397, DTD, EuroSAT, UCF101, ImageNetV2, ImageNet-Sketch,
ImageNet-A, and ImageNet-R.

Unless explicitly stated otherwise, datasets are not included in this
repository. Dataset files, annotations, and metadata are governed by their
respective original licenses, terms of use, and citation requirements.

Users are responsible for obtaining datasets from their official sources and
complying with the corresponding usage terms.

## 4. Python Dependencies

This project may depend on third-party Python packages such as PyTorch,
torchvision, NumPy, pandas, scikit-learn, tqdm, yacs, dassl, and other
libraries.

These dependencies are not relicensed by this repository. They remain governed
by their own licenses as distributed by their respective maintainers.

## 5. Model Weights and Checkpoints

Unless explicitly stated otherwise, pretrained model weights, checkpoints,
and cached logits are not covered by the repository-level MIT License.

If model weights or experiment artifacts are released with this repository,
their license or usage terms should be specified separately.

## 6. Attribution

If you use this project, please cite or acknowledge the relevant upstream
projects when applicable, including but not limited to:

- CoOp / CoCoOp: Kaiyang Zhou et al.
- CLIP: OpenAI
- The original datasets used in the experiments

## 7. Disclaimer

The authors of DualBackboneFusion provide this repository on an "as is" basis.
Users are responsible for ensuring that their use of third-party code,
datasets, models, and dependencies complies with all applicable licenses and
terms.
