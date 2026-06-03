#!/usr/bin/env python
import argparse, json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev

def fmt(vals):
    if not vals: return '-'
    return f'{mean(vals):.2f}±{pstdev(vals) if len(vals)>1 else 0.0:.2f} ({len(vals)})'

def collect(root):
    records=[]
    for p in sorted(root.rglob('results.json')):
        data=json.loads(p.read_text()); meta=data.get('meta',{})
        for r in data.get('results',[]):
            rec={}; rec.update(meta); rec.update(r); rec['_path']=str(p); records.append(rec)
    return records

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--title',default='CoCoOp Dual-Backbone Late Fusion Accuracy Summary')
    args=ap.parse_args(); root=Path(args.root); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    records=collect(root)
    if not records: raise SystemExit(f'No results.json found under {root}')
    targets=sorted({r.get('target','unknown') for r in records}); weights=sorted({float(r['weight_vit']) for r in records}); modes=['raw_logits','std_logits','prob_avg']
    key=[('RN101 only','prob_avg',0.0),('ViT-B/16 only','prob_avg',1.0),('Raw w=0.50','raw_logits',0.5),('Raw w=0.75','raw_logits',0.75),('Std w=0.50','std_logits',0.5),('Std w=0.75','std_logits',0.75),('Prob w=0.50','prob_avg',0.5),('Prob w=0.75','prob_avg',0.75)]
    lines=[f'# {args.title}','',f'Found `{len(records)}` result rows.','','Fusion definition: `fused = w * logits_vit + (1 - w) * logits_rn`','']
    lines += ['## Target-wise fixed-weight results','', '| Target | '+' | '.join(k[0] for k in key)+' |', '|---'+'|---:'*len(key)+'|']
    for t in targets:
        row=[t]
        for _,m,w in key:
            vals=[float(r['accuracy']) for r in records if r.get('target')==t and r.get('mode')==m and abs(float(r['weight_vit'])-w)<1e-9]
            row.append(fmt(vals))
        lines.append('| '+' | '.join(row)+' |')
    lines += ['', '## Overall fixed-weight results', '', '| Setting | Accuracy |', '|---|---:|']
    for name,m,w in key:
        vals=[float(r['accuracy']) for r in records if r.get('mode')==m and abs(float(r['weight_vit'])-w)<1e-9]
        lines.append(f'| {name} | {fmt(vals)} |')
    lines += ['', '## Full overall by mode and weight', '']
    for m in modes:
        lines += [f'### {m}', '', '| w | Accuracy |', '|---:|---:|']
        for w in weights:
            vals=[float(r['accuracy']) for r in records if r.get('mode')==m and abs(float(r['weight_vit'])-w)<1e-9]
            lines.append(f'| {w:.2f} | {fmt(vals)} |')
        lines.append('')
    lines += ['## Best-over-weight diagnostic', '', 'Diagnostic only. Do not report best-over-target weights as a fair main result unless the weight-selection rule is fixed without target labels.', '', '| Mode | Best-over-weight Accuracy | Mean selected w |', '|---|---:|---:|']
    for m in modes:
        grouped=defaultdict(list); best_vals=[]; best_ws=[]
        for r in records:
            if r.get('mode')==m: grouped[(r.get('target'),r.get('seed'))].append(r)
        for rows in grouped.values():
            best=max(rows,key=lambda x: float(x['accuracy'])); best_vals.append(float(best['accuracy'])); best_ws.append(float(best['weight_vit']))
        lines.append(f'| {m} | {fmt(best_vals)} | {mean(best_ws):.2f} |')
    out.write_text('\n'.join(lines),encoding='utf-8'); print(f'[WROTE] {out}')
if __name__=='__main__': main()
