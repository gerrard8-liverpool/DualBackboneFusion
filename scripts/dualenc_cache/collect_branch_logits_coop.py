#!/usr/bin/env python
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

def add_paths(root):
    root = Path(root).resolve(); coop = root/"third_party"/"CoOp_clean"; dassl = root/"third_party"/"Dassl.pytorch"
    for p in [str(coop), str(dassl), str(root)]:
        if p not in sys.path: sys.path.insert(0, p)
    return coop

def make_args(args, config_file, output_dir, model_dir):
    class A: pass
    a=A(); a.root=args.data_root; a.output_dir=str(output_dir); a.resume=""; a.seed=args.seed
    a.source_domains=None; a.target_domains=None; a.transforms=None
    a.config_file=str(config_file); a.dataset_config_file=str(args.dataset_config_file)
    a.trainer="CoOp"; a.backbone=""; a.head=""; a.eval_only=True; a.model_dir=str(model_dir); a.load_epoch=args.load_epoch; a.no_train=True
    a.opts=["DATASET.NUM_SHOTS",str(args.shots),"DATASET.SUBSAMPLE_CLASSES",args.subsample_classes,
            "DATALOADER.TEST.BATCH_SIZE",str(args.batch_size),"DATALOADER.NUM_WORKERS",str(args.num_workers),
            "TRAINER.COOP.N_CTX",str(args.nctx),"TRAINER.COOP.CSC",str(args.csc),"TRAINER.COOP.CLASS_TOKEN_POSITION",str(args.ctx_pos)]
    return a

def parse_batch(batch, device):
    if isinstance(batch, dict):
        img=batch.get("img", batch.get("image")); lab=batch.get("label", batch.get("labels")); path=batch.get("impath", batch.get("img_path"))
    else:
        img, lab = batch[0], batch[1]; path=None
    return img.to(device), lab.to(device), path

def get_model(trainer):
    model=trainer.model
    return model.module if hasattr(model,"module") else model

def classnames(trainer):
    dm=getattr(trainer,"dm",None); ds=getattr(dm,"dataset",None)
    if ds is not None and hasattr(ds,"classnames"): return list(ds.classnames)
    if dm is not None and hasattr(dm,"classnames"): return list(dm.classnames)
    return []

def text_features(model):
    try:
        with torch.no_grad():
            prompts=model.prompt_learner(); toks=model.tokenized_prompts
            txt=model.text_encoder(prompts, toks); txt=txt/txt.norm(dim=-1, keepdim=True)
            return txt.detach().cpu().float().numpy()
    except Exception as e:
        print(f"[WARN] cannot save text features: {e}")
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", default=os.environ.get("ROOT","/home/ubuntu/code/meta_prompt_1"))
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT","/home/ubuntu/datasets"))
    ap.add_argument("--dataset-config-file", required=True); ap.add_argument("--config-file", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-prefix", required=True); ap.add_argument("--branch", choices=["rn","vit"], required=True)
    ap.add_argument("--source", default=""); ap.add_argument("--target", default=""); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--subsample-classes", choices=["base","new","all"], required=True)
    ap.add_argument("--load-epoch", type=int, default=50); ap.add_argument("--shots", type=int, default=16); ap.add_argument("--nctx", type=int, default=16)
    ap.add_argument("--csc", default="False"); ap.add_argument("--ctx-pos", default="end"); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--num-workers", type=int, default=2)
    args=ap.parse_args()
    coop=add_paths(args.project_root); os.chdir(str(coop))
    import train as coop_train
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    set_random_seed(args.seed)
    out=Path(args.output_prefix); out.parent.mkdir(parents=True, exist_ok=True)
    npz=Path(str(out)+f"_{args.branch}.npz"); meta_path=Path(str(out)+f"_{args.branch}.meta.json")
    if npz.exists() and meta_path.exists(): print(f"[SKIP] {npz}"); return
    build_args=make_args(args,args.config_file,out.parent/"_build_tmp"/args.branch,args.model_dir)
    cfg=coop_train.setup_cfg(build_args); trainer=build_trainer(cfg); trainer.load_model(str(args.model_dir), epoch=args.load_epoch)
    model=get_model(trainer); model.eval(); device=next(model.parameters()).device; loader=trainer.test_loader
    L=[]; Y=[]; paths=[]
    with torch.no_grad():
        for batch in loader:
            img, lab, impath=parse_batch(batch, device)
            logits=model(img).float().detach().cpu(); L.append(logits); Y.append(lab.detach().cpu())
            if impath is None: paths.extend([""]*lab.numel())
            elif isinstance(impath,(list,tuple)): paths.extend([str(x) for x in impath])
            else: paths.extend([str(impath)]*lab.numel())
    logits=torch.cat(L,0).numpy(); labels=torch.cat(Y,0).numpy(); txt=text_features(model)
    save={"logits":logits.astype("float32"),"labels":labels.astype("int64")}
    if txt is not None: save["text_features"]=txt.astype("float32")
    np.savez_compressed(npz, **save)
    meta={"branch":args.branch,"source":args.source,"target":args.target,"seed":args.seed,"subsample_classes":args.subsample_classes,
          "dataset_config_file":str(args.dataset_config_file),"config_file":str(args.config_file),"model_dir":str(args.model_dir),"load_epoch":args.load_epoch,
          "num_samples":int(labels.shape[0]),"num_classes":int(logits.shape[1]),"classnames":classnames(trainer),"paths_head":paths[:10]}
    meta_path.write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"[WROTE] {npz}"); print(f"[WROTE] {meta_path}")
if __name__=="__main__": main()
