#!/usr/bin/env python
import argparse, json, os, sys
from dataclasses import dataclass, asdict
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

def setup_imports(project_root: Path, coop_root: Path):
    sys.path.insert(0, str(coop_root))
    sys.path.insert(0, str(project_root / 'src'))
    sys.path.insert(0, str(project_root / 'third_party' / 'Dassl.pytorch'))

def build_train_args(data_root, output_dir, trainer, dataset_config_file, config_file, seed, shots, subsample_classes, nctx, ctx_init, load_epoch):
    opts = ['DATASET.NUM_SHOTS', str(shots), 'DATASET.SUBSAMPLE_CLASSES', str(subsample_classes)]
    if trainer.lower() == 'cocoop':
        opts += ['TRAINER.COCOOP.N_CTX', str(nctx), 'TRAINER.COCOOP.CTX_INIT', str(ctx_init)]
    elif trainer.lower() == 'coop':
        opts += ['TRAINER.COOP.N_CTX', str(nctx), 'TRAINER.COOP.CLASS_TOKEN_POSITION', 'end', 'TRAINER.COOP.CSC', 'False']
    return SimpleNamespace(
        root=data_root, output_dir=output_dir, resume='', seed=seed,
        source_domains=None, target_domains=None, transforms=None,
        trainer=trainer, backbone='', head='', eval_only=True,
        model_dir='', load_epoch=load_epoch, no_train=True,
        config_file=config_file, dataset_config_file=dataset_config_file,
        opts=opts,
    )

def build_and_load_trainer(project_root, coop_root, data_root, trainer_name, dataset_config_file, config_file, model_dir, output_dir, seed, load_epoch, shots, subsample_classes, nctx, ctx_init):
    setup_imports(project_root, coop_root)
    os.chdir(str(coop_root))
    import train as coop_train
    from dassl.engine import build_trainer
    args = build_train_args(data_root, output_dir, trainer_name, dataset_config_file, config_file, seed, shots, subsample_classes, nctx, ctx_init, load_epoch)
    cfg = coop_train.setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=load_epoch)
    trainer.set_model_mode('eval')
    return trainer

def row_standardize(logits, eps=1e-6):
    return (logits - logits.mean(dim=1, keepdim=True)) / (logits.std(dim=1, keepdim=True) + eps)

def update_counts(counts, mode, weight, logits, labels):
    pred = logits.argmax(dim=1)
    correct = int((pred == labels).sum().item())
    total = int(labels.numel())
    key = (mode, float(weight))
    if key not in counts:
        counts[key] = [0, 0]
    counts[key][0] += correct
    counts[key][1] += total

@torch.no_grad()
def evaluate_fusion(trainer_rn, trainer_vit, weights):
    device = trainer_vit.device
    loader = trainer_vit.test_loader
    model_rn, model_vit = trainer_rn.model, trainer_vit.model
    model_rn.eval(); model_vit.eval()
    counts: Dict[Tuple[str, float], List[int]] = {}
    for batch in loader:
        images = batch['img'].to(device)
        labels = batch['label'].to(device)
        logits_rn = model_rn(images).float()
        logits_vit = model_vit(images).float()
        rn_std, vit_std = row_standardize(logits_rn), row_standardize(logits_vit)
        prob_rn, prob_vit = F.softmax(logits_rn, dim=1), F.softmax(logits_vit, dim=1)
        for w in weights:
            w = float(w)
            update_counts(counts, 'raw_logits', w, w*logits_vit + (1-w)*logits_rn, labels)
            update_counts(counts, 'std_logits', w, w*vit_std + (1-w)*rn_std, labels)
            update_counts(counts, 'prob_avg', w, w*prob_vit + (1-w)*prob_rn, labels)
    results = []
    for (mode, weight), (correct, total) in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        results.append(FusionResult(mode, weight, 100.0*correct/max(total,1), correct, total))
    return results

def write_markdown(path, payload):
    rows = payload['results']; meta = payload['meta']
    lines = ['# Late Fusion Logits Result', '']
    for key in ['trainer','source','target','seed','subsample_classes','rn_model_dir','vit_model_dir','rn_config','vit_config','load_epoch']:
        lines.append(f'- **{key}**: `{meta.get(key)}`')
    lines += ['', '| Mode | Weight ViT | Accuracy | Correct | Total |', '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['weight_vit']:.2f} | {r['accuracy']:.2f} | {r['correct']} | {r['total']} |")
    lines += ['', '## Best by fusion mode', '', '| Mode | Best Weight ViT | Best Accuracy |', '|---|---:|---:|']
    for mode in sorted({r['mode'] for r in rows}):
        best = max([r for r in rows if r['mode'] == mode], key=lambda x: x['accuracy'])
        lines.append(f"| {mode} | {best['weight_vit']:.2f} | {best['accuracy']:.2f} |")
    path.write_text('\n'.join(lines), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default=os.environ.get('ROOT', '/home/ubuntu/code/meta_prompt_1'))
    p.add_argument('--coop-root', default=None)
    p.add_argument('--data-root', default=os.environ.get('DATA_ROOT', '/home/ubuntu/datasets'))
    p.add_argument('--trainer', default='CoCoOp')
    p.add_argument('--dataset-config-file', required=True)
    p.add_argument('--rn-config', required=True)
    p.add_argument('--vit-config', required=True)
    p.add_argument('--rn-model-dir', required=True)
    p.add_argument('--vit-model-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source', default='')
    p.add_argument('--target', default='')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--load-epoch', type=int, default=10)
    p.add_argument('--shots', type=int, default=16)
    p.add_argument('--nctx', type=int, default=4)
    p.add_argument('--ctx-init', default='a_photo_of_a')
    p.add_argument('--subsample-classes', default='all')
    p.add_argument('--weights', default='0,0.25,0.5,0.75,1.0')
    args = p.parse_args()
    project_root = Path(args.project_root).resolve()
    coop_root = Path(args.coop_root).resolve() if args.coop_root else project_root / 'third_party/CoOp_clean'
    weights = [float(x) for x in args.weights.replace(' ', '').split(',') if x]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    print('[INFO] Building RN model')
    trn = build_and_load_trainer(project_root, coop_root, args.data_root, args.trainer, args.dataset_config_file, args.rn_config, args.rn_model_dir, str(out/'_tmp_rn_eval'), args.seed, args.load_epoch, args.shots, args.subsample_classes, args.nctx, args.ctx_init)
    print('[INFO] Building ViT model')
    tvit = build_and_load_trainer(project_root, coop_root, args.data_root, args.trainer, args.dataset_config_file, args.vit_config, args.vit_model_dir, str(out/'_tmp_vit_eval'), args.seed, args.load_epoch, args.shots, args.subsample_classes, args.nctx, args.ctx_init)
    results = evaluate_fusion(trn, tvit, weights)
    payload = {'meta': vars(args), 'results': [asdict(r) for r in results]}
    payload['meta'].update({'weights': weights})
    (out/'results.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_markdown(out/'results.md', payload)
    print(f'[WROTE] {out / "results.json"}')
    print(f'[WROTE] {out / "results.md"}')
if __name__ == '__main__':
    main()
