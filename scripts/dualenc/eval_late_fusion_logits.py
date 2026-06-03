#!/usr/bin/env python3
"""
Post-hoc late-fusion evaluation for two CoOp checkpoints.

Primary use case in task_level_CoOp:
  - model A: CoOp RN101 trained on the same source dataset
  - model B: CoOp ViT-B/16 trained on the same source dataset
  - target loader: a strict-DG / cross-dataset / B2N target split

The script does NOT train anything. It builds two CoOp models with the target
classnames, loads their source-trained prompt_learner checkpoints, computes
logits on the same test loader, and evaluates several fusion rules.

Fusion convention:
  fused = w * logits_vit + (1 - w) * logits_rn
So w=0 is pure RN101 and w=1 is pure ViT-B/16.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class FusionResult:
    mode: str
    weight_vit: float
    accuracy: float
    correct: int
    total: int


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = str(x).lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


def make_train_args(
    *,
    data_root: str,
    output_dir: str,
    seed: int,
    dataset_config_file: str,
    train_config_file: str,
    trainer: str,
    shots: int,
    nctx: int,
    csc: str,
    ctx_pos: str,
    subsample_classes: str,
) -> SimpleNamespace:
    # This mirrors third_party/CoOp_clean/train.py arguments closely enough for setup_cfg().
    return SimpleNamespace(
        root=data_root,
        output_dir=output_dir,
        resume="",
        seed=seed,
        source_domains=None,
        target_domains=None,
        transforms=None,
        trainer=trainer,
        backbone="",
        head="",
        dataset_config_file=dataset_config_file,
        config_file=train_config_file,
        eval_only=True,
        model_dir="",
        load_epoch=None,
        no_train=True,
        opts=[
            "TRAINER.COOP.N_CTX", str(nctx),
            "TRAINER.COOP.CSC", str(csc),
            "TRAINER.COOP.CLASS_TOKEN_POSITION", str(ctx_pos),
            "DATASET.NUM_SHOTS", str(shots),
            "DATASET.SUBSAMPLE_CLASSES", str(subsample_classes),
        ],
    )


def add_repo_paths(project_root: Path, coop_root: Path) -> None:
    sys.path.insert(0, str(coop_root))
    sys.path.insert(0, str(project_root / "src"))


def build_and_load_trainer(
    *,
    project_root: Path,
    coop_root: Path,
    data_root: str,
    scratch_output_dir: Path,
    seed: int,
    dataset_config_file: str,
    train_config_file: str,
    model_dir: str,
    load_epoch: int,
    shots: int,
    nctx: int,
    csc: str,
    ctx_pos: str,
    subsample_classes: str,
):
    # Import after sys.path is patched.
    import train as coop_train  # noqa: WPS433
    from dassl.engine import build_trainer  # noqa: WPS433

    args = make_train_args(
        data_root=data_root,
        output_dir=str(scratch_output_dir),
        seed=seed,
        dataset_config_file=dataset_config_file,
        train_config_file=train_config_file,
        trainer="CoOp",
        shots=shots,
        nctx=nctx,
        csc=csc,
        ctx_pos=ctx_pos,
        subsample_classes=subsample_classes,
    )
    cfg = coop_train.setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=load_epoch)
    trainer.model.eval()
    return trainer


def parse_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        image = batch.get("img", batch.get("image", None))
        label = batch.get("label", batch.get("labels", None))
        if image is None or label is None:
            raise KeyError(f"Unsupported batch dict keys: {list(batch.keys())}")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        image, label = batch[0], batch[1]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    return image.to(device), label.to(device)


def row_standardize(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Per-sample standardization removes scale domination by one backbone.
    mean = logits.mean(dim=1, keepdim=True)
    std = logits.std(dim=1, keepdim=True).clamp_min(eps)
    return (logits - mean) / std


def safe_log_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.float(), dim=1)


def update_counts(
    counts: Dict[Tuple[str, float], List[int]],
    *,
    mode: str,
    weight: float,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    pred = logits.argmax(dim=1)
    correct = int((pred == labels).sum().item())
    total = int(labels.numel())
    key = (mode, float(weight))
    if key not in counts:
        counts[key] = [0, 0]
    counts[key][0] += correct
    counts[key][1] += total


def evaluate_fusion(
    *,
    trainer_rn,
    trainer_vit,
    weights: Iterable[float],
) -> List[FusionResult]:
    device = trainer_vit.device
    model_rn = trainer_rn.model.to(device)
    model_vit = trainer_vit.model.to(device)
    model_rn.eval()
    model_vit.eval()

    loader = trainer_vit.test_loader
    counts: Dict[Tuple[str, float], List[int]] = {}

    with torch.no_grad():
        for batch in loader:
            images, labels = parse_batch(batch, device)

            logits_rn = model_rn(images).float()
            logits_vit = model_vit(images).float()

            logits_rn_std = row_standardize(logits_rn)
            logits_vit_std = row_standardize(logits_vit)

            prob_rn = F.softmax(logits_rn, dim=1)
            prob_vit = F.softmax(logits_vit, dim=1)

            for w in weights:
                w = float(w)
                raw = w * logits_vit + (1.0 - w) * logits_rn
                std = w * logits_vit_std + (1.0 - w) * logits_rn_std
                prob = w * prob_vit + (1.0 - w) * prob_rn

                update_counts(counts, mode="raw_logits", weight=w, logits=raw, labels=labels)
                update_counts(counts, mode="std_logits", weight=w, logits=std, labels=labels)
                update_counts(counts, mode="prob_avg", weight=w, logits=prob, labels=labels)

    results: List[FusionResult] = []
    for (mode, weight), (correct, total) in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        acc = 100.0 * correct / max(total, 1)
        results.append(FusionResult(mode=mode, weight_vit=weight, accuracy=acc, correct=correct, total=total))
    return results


def write_markdown(path: Path, payload: dict) -> None:
    rows = payload["results"]
    lines = []
    lines.append("# Late Fusion Logits Result")
    lines.append("")
    meta = payload["meta"]
    for key in [
        "source", "target", "seed", "subsample_classes", "rn_model_dir", "vit_model_dir",
        "rn_config", "vit_config", "load_epoch",
    ]:
        lines.append(f"- **{key}**: `{meta.get(key)}`")
    lines.append("")
    lines.append("| Mode | Weight ViT | Accuracy | Correct | Total |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['weight_vit']:.2f} | {r['accuracy']:.2f} | {r['correct']} | {r['total']} |"
        )
    lines.append("")

    # Also show best rows by mode for quick reading.
    lines.append("## Best by fusion mode")
    lines.append("")
    lines.append("| Mode | Best Weight ViT | Best Accuracy |")
    lines.append("|---|---:|---:|")
    for mode in sorted({r["mode"] for r in rows}):
        subset = [r for r in rows if r["mode"] == mode]
        best = max(subset, key=lambda x: x["accuracy"])
        lines.append(f"| {mode} | {best['weight_vit']:.2f} | {best['accuracy']:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.environ.get("ROOT", "/workspace/meta_prompt_1"))
    parser.add_argument("--coop-root", default=None)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/workspace/datasets"))
    parser.add_argument("--dataset-config-file", required=True)
    parser.add_argument("--rn-config", default="configs/trainers/CoOp/rn101_ep50.yaml")
    parser.add_argument("--vit-config", default="configs/trainers/CoOp/vit_b16_ep50.yaml")
    parser.add_argument("--rn-model-dir", required=True)
    parser.add_argument("--vit-model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--target", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--load-epoch", type=int, default=50)
    parser.add_argument("--shots", type=int, default=16)
    parser.add_argument("--nctx", type=int, default=16)
    parser.add_argument("--csc", default="False")
    parser.add_argument("--ctx-pos", default="end")
    parser.add_argument("--subsample-classes", default="all", choices=["all", "base", "new"])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    coop_root = Path(args.coop_root or project_root / "third_party" / "CoOp_clean").resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    add_repo_paths(project_root, coop_root)

    # Build ViT second but use its test_loader for evaluation. The target dataset config is the same.
    scratch = out_dir / "_scratch_build"
    scratch.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building RN101 CoOp target model")
    trainer_rn = build_and_load_trainer(
        project_root=project_root,
        coop_root=coop_root,
        data_root=args.data_root,
        scratch_output_dir=scratch / "rn101",
        seed=args.seed,
        dataset_config_file=args.dataset_config_file,
        train_config_file=args.rn_config,
        model_dir=args.rn_model_dir,
        load_epoch=args.load_epoch,
        shots=args.shots,
        nctx=args.nctx,
        csc=args.csc,
        ctx_pos=args.ctx_pos,
        subsample_classes=args.subsample_classes,
    )

    print("[INFO] Building ViT-B/16 CoOp target model")
    trainer_vit = build_and_load_trainer(
        project_root=project_root,
        coop_root=coop_root,
        data_root=args.data_root,
        scratch_output_dir=scratch / "vit_b16",
        seed=args.seed,
        dataset_config_file=args.dataset_config_file,
        train_config_file=args.vit_config,
        model_dir=args.vit_model_dir,
        load_epoch=args.load_epoch,
        shots=args.shots,
        nctx=args.nctx,
        csc=args.csc,
        ctx_pos=args.ctx_pos,
        subsample_classes=args.subsample_classes,
    )

    print("[INFO] Evaluating fusion weights:", args.weights)
    results = evaluate_fusion(trainer_rn=trainer_rn, trainer_vit=trainer_vit, weights=args.weights)

    payload = {
        "meta": {
            "source": args.source,
            "target": args.target,
            "seed": args.seed,
            "subsample_classes": args.subsample_classes,
            "rn_config": args.rn_config,
            "vit_config": args.vit_config,
            "rn_model_dir": args.rn_model_dir,
            "vit_model_dir": args.vit_model_dir,
            "dataset_config_file": args.dataset_config_file,
            "load_epoch": args.load_epoch,
            "shots": args.shots,
            "nctx": args.nctx,
            "csc": args.csc,
            "ctx_pos": args.ctx_pos,
            "weights": args.weights,
        },
        "results": [asdict(r) for r in results],
    }

    json_path = out_dir / "results.json"
    md_path = out_dir / "results.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)

    print(f"[DONE] wrote {json_path}")
    print(f"[DONE] wrote {md_path}")


if __name__ == "__main__":
    main()
