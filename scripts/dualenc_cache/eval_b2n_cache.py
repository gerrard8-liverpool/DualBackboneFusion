#!/usr/bin/env python
import argparse, json
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
import numpy as np
EPS=1e-12

def softmax(x):
    x=x-np.max(x,axis=1,keepdims=True); e=np.exp(x); return e/np.maximum(e.sum(axis=1,keepdims=True),EPS)
def ent(logits):
    p=softmax(logits); return -(p*np.log(np.maximum(p,EPS))).sum(axis=1)
def std(logits):
    return (logits-logits.mean(axis=1,keepdims=True))/np.maximum(logits.std(axis=1,keepdims=True),1e-6)
def scores(rn,vit,mode,w):
    if mode=="raw_logits": return w*vit+(1-w)*rn
    if mode=="std_logits": return w*std(vit)+(1-w)*std(rn)
    if mode=="prob_avg": return w*softmax(vit)+(1-w)*softmax(rn)
    raise ValueError(mode)
def acc(s,y):
    p=s.argmax(axis=1); return 100.0*float((p==y).mean())
def hmean(b,n): return 2*b*n/(b+n) if b+n>0 else 0.0
def fmt(v): return "-" if not v else f"{mean(v):.2f}±{(pstdev(v) if len(v)>1 else 0.0):.2f} ({len(v)})"
def load(prefix,branch):
    p=Path(str(prefix)+f"_{branch}.npz"); mp=Path(str(prefix)+f"_{branch}.meta.json")
    if not p.exists(): raise FileNotFoundError(p)
    return np.load(p,allow_pickle=True), json.loads(mp.read_text()) if mp.exists() else {}
def best_w(rn,vit,y,mode,weights):
    vals=[(w,acc(scores(rn,vit,mode,w),y)) for w in weights]
    return max(vals,key=lambda x:x[1])
def class_weights(rn,vit,y,wD,alpha,beta,gamma,temp,shrink):
    C=rn.shape[1]; pr=rn.argmax(1); pv=vit.argmax(1); er=ent(rn); ev=ent(vit); W=np.zeros(C,np.float32); stats=[]
    for c in range(C):
        idx=np.where(y==c)[0]; n=len(idx)
        if n==0: W[c]=wD; continue
        mask=np.ones(C,dtype=bool); mask[c]=False
        mr=(rn[idx,c]-rn[idx][:,mask].max(1)).mean(); mv=(vit[idx,c]-vit[idx][:,mask].max(1)).mean()
        ar=(pr[idx]==c).mean(); av=(pv[idx]==c).mean(); hr=er[idx].mean(); hv=ev[idx].mean()
        Rr=alpha*ar+beta*mr-gamma*hr; Rv=alpha*av+beta*mv-gamma*hv
        wraw=np.exp(Rv/temp)/max(np.exp(Rv/temp)+np.exp(Rr/temp),EPS)
        W[c]=(n*wraw+shrink*wD)/(n+shrink)
        stats.append({"class":int(c),"n":int(n),"w":float(W[c]),"w_raw":float(wraw),"acc_rn":float(ar),"acc_vit":float(av),"margin_rn":float(mr),"margin_vit":float(mv)})
    return W,stats
def transfer(base_w,base_txt,target_txt,wD,topk,temp):
    if base_txt is None or target_txt is None: return np.ones(target_txt.shape[0] if target_txt is not None else len(base_w),np.float32)*wD
    B=base_txt/np.maximum(np.linalg.norm(base_txt,axis=1,keepdims=True),EPS); T=target_txt/np.maximum(np.linalg.norm(target_txt,axis=1,keepdims=True),EPS)
    sim=T@B.T; k=min(topk,B.shape[0]); out=[]
    for i in range(sim.shape[0]):
        idx=np.argsort(sim[i])[-k:][::-1]; s=sim[i,idx]; pi=np.exp(s/temp); pi=pi/np.maximum(pi.sum(),EPS)
        wc=float((pi*base_w[idx]).sum()); rho=float(np.clip(s.max(),0,1)); out.append(rho*wc+(1-rho)*wD)
    return np.array(out,np.float32)
def class_scores(rn,vit,wvec,mode):
    a,b=(rn,vit) if mode=="raw_logits" else (std(rn),std(vit)) if mode=="std_logits" else (softmax(rn),softmax(vit))
    w=wvec[None,:]; return w*b+(1-w)*a
def oracle_trueclass(rn,vit,y,mode,weights):
    C=rn.shape[1]; cw=np.ones(C,np.float32)
    for c in range(C):
        idx=np.where(y==c)[0]
        if len(idx)==0: continue
        cw[c]=max([(w,acc(scores(rn[idx],vit[idx],mode,w),y[idx])) for w in weights],key=lambda x:x[1])[0]
    correct=0
    for i in range(len(y)):
        correct += int(scores(rn[i:i+1],vit[i:i+1],mode,float(cw[int(y[i])])).argmax(1)[0]==y[i])
    return 100.0*correct/len(y)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache-root",required=True); ap.add_argument("--datasets",nargs="+",required=True); ap.add_argument("--seeds",nargs="+",type=int,default=[1]); ap.add_argument("--output",required=True)
    ap.add_argument("--weights",nargs="+",type=float,default=[0,0.25,0.5,0.75,1]); ap.add_argument("--modes",nargs="+",default=["raw_logits","std_logits","prob_avg"])
    ap.add_argument("--alpha",type=float,default=1.0); ap.add_argument("--beta",type=float,default=0.5); ap.add_argument("--gamma",type=float,default=0.1); ap.add_argument("--rel-temp",type=float,default=0.5); ap.add_argument("--shrink-lambda",type=float,default=8.0)
    ap.add_argument("--topk",type=int,default=5); ap.add_argument("--sem-temp",type=float,default=0.07); args=ap.parse_args()
    rows=[]; stats={}; root=Path(args.cache_root)
    for d in args.datasets:
      for seed in args.seeds:
        try:
          br,bmr=load(root/d/"split_base"/f"seed{seed}"/"logits","rn"); bv,bmv=load(root/d/"split_base"/f"seed{seed}"/"logits","vit")
          nr,nmr=load(root/d/"split_new"/f"seed{seed}"/"logits","rn"); nv,nmv=load(root/d/"split_new"/f"seed{seed}"/"logits","vit")
        except FileNotFoundError as e:
          print(f"[WARN] missing {d} seed{seed}: {e}"); continue
        have_all=(root/d/"split_all"/f"seed{seed}"/"logits_rn.npz").exists() and (root/d/"split_all"/f"seed{seed}"/"logits_vit.npz").exists()
        if have_all: ar,amr=load(root/d/"split_all"/f"seed{seed}"/"logits","rn"); av,amv=load(root/d/"split_all"/f"seed{seed}"/"logits","vit")
        for mode in args.modes:
          rn_b,vit_b,yb=br["logits"],bv["logits"],br["labels"]; rn_n,vit_n,yn=nr["logits"],nv["logits"],nr["labels"]
          wD,_=best_w(rn_b,vit_b,yb,mode,args.weights); bw,st=class_weights(rn_b,vit_b,yb,wD,args.alpha,args.beta,args.gamma,args.rel_temp,args.shrink_lambda); stats[f"{d}/seed{seed}/{mode}"]=st
          base_txt=bv["text_features"] if "text_features" in bv.files else None; new_txt=nv["text_features"] if "text_features" in nv.files else None; nw=transfer(bw,base_txt,new_txt,wD,args.topk,args.sem_temp)
          methods={"fixed_w0.50":0.5,"fixed_w0.75":0.75,"dataset_cached_wD":wD}
          for name,w in methods.items():
            b=acc(scores(rn_b,vit_b,mode,w),yb); n=acc(scores(rn_n,vit_n,mode,w),yn); allv=None
            if have_all: allv=acc(scores(ar["logits"],av["logits"],mode,w),ar["labels"])
            rows.append({"dataset":d,"seed":seed,"mode":mode,"method":name,"base":b,"new":n,"hm":hmean(b,n),"all":allv,"wD":wD})
          b=acc(class_scores(rn_b,vit_b,bw,mode),yb); n=acc(class_scores(rn_n,vit_n,nw,mode),yn); allv=None
          if have_all:
            all_txt=av["text_features"] if "text_features" in av.files else None; aw=transfer(bw,base_txt,all_txt,wD,args.topk,args.sem_temp)
            base_names=bmv.get("classnames",[]); all_names=amv.get("classnames",[]); bm={n:i for i,n in enumerate(base_names)}
            for i,nm in enumerate(all_names):
                if nm in bm: aw[i]=bw[bm[nm]]
            allv=acc(class_scores(ar["logits"],av["logits"],aw,mode),ar["labels"])
          rows.append({"dataset":d,"seed":seed,"mode":mode,"method":"classwise_reliability_cache","base":b,"new":n,"hm":hmean(b,n),"all":allv,"wD":wD})
          b=oracle_trueclass(rn_b,vit_b,yb,mode,args.weights); n=oracle_trueclass(rn_n,vit_n,yn,mode,args.weights); allv=None
          if have_all: allv=oracle_trueclass(ar["logits"],av["logits"],ar["labels"],mode,args.weights)
          rows.append({"dataset":d,"seed":seed,"mode":mode,"method":"classwise_oracle_true_label_diagnostic","base":b,"new":n,"hm":hmean(b,n),"all":allv,"wD":wD})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    lines=["# B2N Cache Fusion Summary","","`classwise_oracle_true_label_diagnostic` uses target labels and is diagnostic only.","","## Overall","","| Mode | Method | Base | New | HM | All |","|---|---|---:|---:|---:|---:|"]
    G=defaultdict(lambda:{"base":[],"new":[],"hm":[],"all":[]})
    for r in rows:
      g=G[(r["mode"],r["method"])]
      for k in ["base","new","hm"]: g[k].append(float(r[k]))
      if r["all"] is not None: g["all"].append(float(r["all"]))
    for (mode,meth),v in sorted(G.items()): lines.append(f"| {mode} | {meth} | {fmt(v['base'])} | {fmt(v['new'])} | {fmt(v['hm'])} | {fmt(v['all'])} |")
    lines += ["","## Dataset-wise","","| Dataset | Mode | Method | Base | New | HM | All | Mean wD |","|---|---|---|---:|---:|---:|---:|---:|"]
    D=defaultdict(lambda:{"base":[],"new":[],"hm":[],"all":[],"wD":[]})
    for r in rows:
      g=D[(r["dataset"],r["mode"],r["method"])]
      for k in ["base","new","hm"]: g[k].append(float(r[k]))
      if r["all"] is not None: g["all"].append(float(r["all"]))
      g["wD"].append(float(r["wD"]))
    for (d,mode,meth),v in sorted(D.items()): lines.append(f"| {d} | {mode} | {meth} | {fmt(v['base'])} | {fmt(v['new'])} | {fmt(v['hm'])} | {fmt(v['all'])} | {mean(v['wD']):.2f} |")
    out.write_text("\n".join(lines),encoding="utf-8"); out.with_suffix(".class_stats.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(f"[WROTE] {out}")
if __name__=="__main__": main()
