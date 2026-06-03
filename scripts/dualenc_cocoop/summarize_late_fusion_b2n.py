#!/usr/bin/env python
import argparse, json
from pathlib import Path
from statistics import mean, pstdev

def fmt(vals):
    if not vals: return '-'
    return f'{mean(vals):.2f}±{pstdev(vals) if len(vals)>1 else 0.0:.2f} ({len(vals)})'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--title',default='CoCoOp Dual-Backbone Late Fusion B2N Summary')
    args=ap.parse_args(); root=Path(args.root); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for p in sorted(root.rglob('results.json')):
        data=json.loads(p.read_text()); meta=data.get('meta',{})
        for r in data.get('results',[]):
            rec={}; rec.update(meta); rec.update(r); records.append(rec)
    if not records: raise SystemExit(f'No results.json found under {root}')
    acc={}
    for r in records:
        acc[(r['target'], int(r['seed']), r['mode'], float(r['weight_vit']), r['subsample_classes'])]=float(r['accuracy'])
    datasets=sorted({r['target'] for r in records}); modes=['raw_logits','std_logits','prob_avg']; weights=sorted({float(r['weight_vit']) for r in records})
    key=[('RN101 only','prob_avg',0.0),('ViT-B/16 only','prob_avg',1.0),('Raw 0.5 fusion','raw_logits',0.5),('Raw 0.75 fusion','raw_logits',0.75),('Std 0.5 fusion','std_logits',0.5),('Std 0.75 fusion','std_logits',0.75),('Prob 0.5 fusion','prob_avg',0.5),('Prob 0.75 fusion','prob_avg',0.75)]
    lines=[f'# {args.title}','',f'Found `{len(records)}` result rows.','','Fusion definition: `fused = w * logits_vit + (1 - w) * logits_rn`','']
    lines += ['## Compact B2N table','', '| Dataset | Setting | Base | New | HM | All |', '|---|---|---:|---:|---:|---:|']
    for d in datasets:
        seeds=sorted({int(r['seed']) for r in records if r['target']==d})
        for name,m,w in key:
            bv=[]; nv=[]; av=[]; hv=[]
            for s in seeds:
                b=acc.get((d,s,m,w,'base')); n=acc.get((d,s,m,w,'new')); a=acc.get((d,s,m,w,'all'))
                if b is not None: bv.append(b)
                if n is not None: nv.append(n)
                if a is not None: av.append(a)
                if b is not None and n is not None: hv.append(2*b*n/(b+n) if b+n>0 else 0.0)
            lines.append(f'| {d} | {name} | {fmt(bv)} | {fmt(nv)} | {fmt(hv)} | {fmt(av)} |')
    lines += ['', '## Overall by mode and weight', '', '| Mode | w | Base | New | HM | All |', '|---|---:|---:|---:|---:|---:|']
    for m in modes:
        for w in weights:
            bv=[]; nv=[]; av=[]; hv=[]
            for d in datasets:
                seeds=sorted({int(r['seed']) for r in records if r['target']==d})
                for s in seeds:
                    b=acc.get((d,s,m,w,'base')); n=acc.get((d,s,m,w,'new')); a=acc.get((d,s,m,w,'all'))
                    if b is not None: bv.append(b)
                    if n is not None: nv.append(n)
                    if a is not None: av.append(a)
                    if b is not None and n is not None: hv.append(2*b*n/(b+n) if b+n>0 else 0.0)
            lines.append(f'| {m} | {w:.2f} | {fmt(bv)} | {fmt(nv)} | {fmt(hv)} | {fmt(av)} |')
    out.write_text('\n'.join(lines),encoding='utf-8'); print(f'[WROTE] {out}')
if __name__=='__main__': main()
