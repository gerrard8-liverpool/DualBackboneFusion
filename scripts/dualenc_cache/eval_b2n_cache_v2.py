#!/usr/bin/env python
import argparse, json, math
from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
import numpy as np

EPS = 1e-12

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=1, keepdims=True), EPS)

def entropy(logits):
    p = softmax(logits)
    return -(p * np.log(np.maximum(p, EPS))).sum(axis=1)

def std(logits):
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(logits.std(axis=1, keepdims=True), 1e-6)

def fuse_scores(rn, vit, mode, w):
    if mode == "raw_logits":
        return w * vit + (1 - w) * rn
    if mode == "std_logits":
        return w * std(vit) + (1 - w) * std(rn)
    if mode == "prob_avg":
        return w * softmax(vit) + (1 - w) * softmax(rn)
    raise ValueError(mode)

def class_fuse_scores(rn, vit, mode, wvec):
    if mode == "raw_logits":
        a, b = rn, vit
    elif mode == "std_logits":
        a, b = std(rn), std(vit)
    elif mode == "prob_avg":
        a, b = softmax(rn), softmax(vit)
    else:
        raise ValueError(mode)
    w = wvec[None, :]
    return w * b + (1 - w) * a

def acc_from_scores(s, y):
    return 100.0 * float((s.argmax(axis=1) == y).mean()) if len(y) else 0.0

def hmean(b, n):
    return 2 * b * n / (b + n) if (b + n) > 0 else 0.0

def fmt(vals):
    if not vals:
        return "-"
    return f"{mean(vals):.2f}±{(pstdev(vals) if len(vals) > 1 else 0.0):.2f} ({len(vals)})"

def load(prefix, branch):
    p = Path(str(prefix) + f"_{branch}.npz")
    mp = Path(str(prefix) + f"_{branch}.meta.json")
    if not p.exists():
        raise FileNotFoundError(p)
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    return np.load(p, allow_pickle=True), meta

def best_w(rn, vit, y, mode, weights):
    vals = [(float(w), acc_from_scores(fuse_scores(rn, vit, mode, float(w)), y)) for w in weights]
    return max(vals, key=lambda x: x[1])

def normalize_txt(x):
    if x is None:
        return None
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)

def get_text_features(rn_npz, vit_npz, text_space="concat"):
    rt = rn_npz["text_features"] if "text_features" in rn_npz.files else None
    vt = vit_npz["text_features"] if "text_features" in vit_npz.files else None
    rt, vt = normalize_txt(rt), normalize_txt(vt)
    if text_space == "rn":
        return rt
    if text_space == "vit":
        return vt
    if text_space == "concat":
        if rt is not None and vt is not None:
            return normalize_txt(np.concatenate([vt, rt], axis=1))
        return vt if vt is not None else rt
    raise ValueError(text_space)

def class_weights(rn, vit, y, wD, alpha=1.0, beta=0.5, gamma=0.1, rel_temp=1.0, shrink=8.0):
    C = rn.shape[1]
    pr, pv = rn.argmax(1), vit.argmax(1)
    er, ev = entropy(rn), entropy(vit)
    W = np.ones(C, np.float32) * float(wD)
    stats = []
    for c in range(C):
        idx = np.where(y == c)[0]
        n = len(idx)
        if n == 0:
            continue
        mask = np.ones(C, dtype=bool)
        mask[c] = False
        mr = (rn[idx, c] - rn[idx][:, mask].max(1)).mean()
        mv = (vit[idx, c] - vit[idx][:, mask].max(1)).mean()
        ar = (pr[idx] == c).mean()
        av = (pv[idx] == c).mean()
        hr = er[idx].mean()
        hv = ev[idx].mean()
        Rr = alpha * ar + beta * mr - gamma * hr
        Rv = alpha * av + beta * mv - gamma * hv
        # Stable sigmoid over reliability difference.
        wraw = 1.0 / (1.0 + math.exp(-float((Rv - Rr) / max(rel_temp, 1e-6))))
        W[c] = (n * wraw + shrink * wD) / (n + shrink)
        stats.append({"class": int(c), "n": int(n), "w": float(W[c]), "w_raw": float(wraw),
                      "acc_rn": float(ar), "acc_vit": float(av), "margin_rn": float(mr), "margin_vit": float(mv)})
    return W, stats

def transfer_weights(base_w, base_txt, target_txt, wD, topk=3, sem_temp=0.2, rho_power=1.0):
    if base_txt is None or target_txt is None:
        n = target_txt.shape[0] if target_txt is not None else len(base_w)
        return np.ones(n, np.float32) * float(wD)
    B = normalize_txt(base_txt)
    T = normalize_txt(target_txt)
    sim = T @ B.T
    k = min(int(topk), B.shape[0])
    out = []
    for i in range(sim.shape[0]):
        idx = np.argsort(sim[i])[-k:][::-1]
        s = sim[i, idx]
        z = np.exp((s - s.max()) / max(sem_temp, 1e-6))
        pi = z / np.maximum(z.sum(), EPS)
        wc = float((pi * base_w[idx]).sum())
        rho = float(np.clip(s.max(), 0.0, 1.0) ** rho_power)
        out.append(rho * wc + (1 - rho) * float(wD))
    return np.array(out, dtype=np.float32)

def subset_columns(x, cols):
    return x[:, cols]

def remap_labels(y, cols):
    mp = {int(c): i for i, c in enumerate(cols)}
    return np.array([mp[int(v)] for v in y], dtype=np.int64)

def eval_candidate_pseudo(rn, vit, y, txt, mode, weights, cfg, n_folds=4):
    C = rn.shape[1]
    classes = np.arange(C)
    folds = np.array_split(classes, n_folds)
    vals = []
    for query in folds:
        query = np.asarray(query, dtype=np.int64)
        seen = np.asarray([c for c in classes if c not in set(query.tolist())], dtype=np.int64)
        if len(query) == 0 or len(seen) == 0:
            continue
        seen_mask = np.isin(y, seen)
        query_mask = np.isin(y, query)
        if seen_mask.sum() == 0 or query_mask.sum() == 0:
            continue
        rn_seen = subset_columns(rn[seen_mask], seen)
        vit_seen = subset_columns(vit[seen_mask], seen)
        y_seen = remap_labels(y[seen_mask], seen)
        rn_query = subset_columns(rn[query_mask], query)
        vit_query = subset_columns(vit[query_mask], query)
        y_query = remap_labels(y[query_mask], query)
        wD, _ = best_w(rn_seen, vit_seen, y_seen, mode, weights)
        bw, _ = class_weights(rn_seen, vit_seen, y_seen, wD, cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["rel_temp"], cfg["shrink"])
        if txt is not None:
            seen_txt = txt[seen]
            query_txt = txt[query]
        else:
            seen_txt = query_txt = None
        qw = transfer_weights(bw, seen_txt, query_txt, wD, cfg["topk"], cfg["sem_temp"], cfg["rho_power"])
        vals.append(acc_from_scores(class_fuse_scores(rn_query, vit_query, mode, qw), y_query))
    return float(mean(vals)) if vals else -1.0

def oracle_trueclass(rn, vit, y, mode, weights):
    C = rn.shape[1]
    cw = np.ones(C, np.float32)
    for c in range(C):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        cw[c] = max([(w, acc_from_scores(fuse_scores(rn[idx], vit[idx], mode, float(w)), y[idx])) for w in weights], key=lambda x: x[1])[0]
    return acc_from_scores(class_fuse_scores(rn, vit, mode, cw), y)

def make_weight_grid(step):
    n = int(round(1.0 / step))
    return [round(i * step, 6) for i in range(n + 1)]

def add_row(rows, dataset, seed, mode, method, base, new, allv, wD, extra=None):
    rows.append({"dataset": dataset, "seed": seed, "mode": mode, "method": method,
                 "base": base, "new": new, "hm": hmean(base, new), "all": allv, "wD": wD, "extra": extra or {}})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--modes", nargs="+", default=["std_logits", "prob_avg"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--weight-step", type=float, default=0.05)
    ap.add_argument("--text-space", choices=["vit", "rn", "concat"], default="concat")
    ap.add_argument("--candidate-shrink", nargs="+", type=float, default=[4, 8, 16, 32])
    ap.add_argument("--candidate-topk", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--candidate-sem-temp", nargs="+", type=float, default=[0.07, 0.10, 0.20, 0.35])
    ap.add_argument("--candidate-rho-power", nargs="+", type=float, default=[1, 2, 4, 8])
    ap.add_argument("--rel-temp", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--safe-delta", type=float, default=0.2, help="Base-acc margin required before using classwise over dataset cache")
    args = ap.parse_args()

    weights = args.weights if args.weights is not None else make_weight_grid(args.weight_step)
    root = Path(args.cache_root)
    rows, stats, selected = [], {}, {}

    candidates = []
    for sh in args.candidate_shrink:
        for topk in args.candidate_topk:
            for st in args.candidate_sem_temp:
                for rp in args.candidate_rho_power:
                    candidates.append({"shrink": sh, "topk": topk, "sem_temp": st, "rho_power": rp,
                                       "rel_temp": args.rel_temp, "alpha": args.alpha, "beta": args.beta, "gamma": args.gamma})

    for d in args.datasets:
        for seed in args.seeds:
            try:
                br, bmr = load(root / d / "split_base" / f"seed{seed}" / "logits", "rn")
                bv, bmv = load(root / d / "split_base" / f"seed{seed}" / "logits", "vit")
                nr, nmr = load(root / d / "split_new" / f"seed{seed}" / "logits", "rn")
                nv, nmv = load(root / d / "split_new" / f"seed{seed}" / "logits", "vit")
            except FileNotFoundError as e:
                print(f"[WARN] missing {d} seed{seed}: {e}")
                continue
            have_all = (root / d / "split_all" / f"seed{seed}" / "logits_rn.npz").exists() and (root / d / "split_all" / f"seed{seed}" / "logits_vit.npz").exists()
            if have_all:
                ar, amr = load(root / d / "split_all" / f"seed{seed}" / "logits", "rn")
                av, amv = load(root / d / "split_all" / f"seed{seed}" / "logits", "vit")
            rn_b, vit_b, yb = br["logits"], bv["logits"], br["labels"].astype(np.int64)
            rn_n, vit_n, yn = nr["logits"], nv["logits"], nr["labels"].astype(np.int64)
            base_txt = get_text_features(br, bv, args.text_space)
            new_txt = get_text_features(nr, nv, args.text_space)
            all_txt = get_text_features(ar, av, args.text_space) if have_all else None

            for mode in args.modes:
                wD, base_wD_acc = best_w(rn_b, vit_b, yb, mode, weights)
                fixed05_b = acc_from_scores(fuse_scores(rn_b, vit_b, mode, 0.5), yb)
                fixed05_n = acc_from_scores(fuse_scores(rn_n, vit_n, mode, 0.5), yn)
                fixed75_b = acc_from_scores(fuse_scores(rn_b, vit_b, mode, 0.75), yb)
                fixed75_n = acc_from_scores(fuse_scores(rn_n, vit_n, mode, 0.75), yn)
                all05 = all75 = allD = None
                if have_all:
                    all05 = acc_from_scores(fuse_scores(ar["logits"], av["logits"], mode, 0.5), ar["labels"])
                    all75 = acc_from_scores(fuse_scores(ar["logits"], av["logits"], mode, 0.75), ar["labels"])
                    allD = acc_from_scores(fuse_scores(ar["logits"], av["logits"], mode, wD), ar["labels"])
                add_row(rows, d, seed, mode, "fixed_w0.50", fixed05_b, fixed05_n, all05, wD)
                add_row(rows, d, seed, mode, "fixed_w0.75", fixed75_b, fixed75_n, all75, wD)
                add_row(rows, d, seed, mode, "dataset_cached_wD_fine", base_wD_acc, acc_from_scores(fuse_scores(rn_n, vit_n, mode, wD), yn), allD, wD)

                # Select retrieval hyperparameters by pseudo-new episodes on base classes only.
                scored = []
                for cfg in candidates:
                    score = eval_candidate_pseudo(rn_b, vit_b, yb, base_txt, mode, weights, cfg, n_folds=4)
                    scored.append((score, cfg))
                score, best_cfg = max(scored, key=lambda x: x[0])
                selected[f"{d}/seed{seed}/{mode}"] = {"pseudo_score": score, "cfg": best_cfg, "wD": wD}

                bw, st = class_weights(rn_b, vit_b, yb, wD, best_cfg["alpha"], best_cfg["beta"], best_cfg["gamma"], best_cfg["rel_temp"], best_cfg["shrink"])
                stats[f"{d}/seed{seed}/{mode}"] = st
                nw = transfer_weights(bw, base_txt, new_txt, wD, best_cfg["topk"], best_cfg["sem_temp"], best_cfg["rho_power"])
                base_class_acc = acc_from_scores(class_fuse_scores(rn_b, vit_b, mode, bw), yb)
                new_class_acc = acc_from_scores(class_fuse_scores(rn_n, vit_n, mode, nw), yn)
                all_class_acc = None
                if have_all:
                    aw = transfer_weights(bw, base_txt, all_txt, wD, best_cfg["topk"], best_cfg["sem_temp"], best_cfg["rho_power"])
                    base_names = bmv.get("classnames", [])
                    all_names = amv.get("classnames", [])
                    bm = {n: i for i, n in enumerate(base_names)}
                    for i, nm in enumerate(all_names):
                        if nm in bm:
                            aw[i] = bw[bm[nm]]
                    all_class_acc = acc_from_scores(class_fuse_scores(ar["logits"], av["logits"], mode, aw), ar["labels"])
                add_row(rows, d, seed, mode, "episodic_selected_cache", base_class_acc, new_class_acc, all_class_acc, wD, best_cfg)

                # Safe fallback: only use class-wise cache if it improves base over dataset cache by margin.
                use_class = base_class_acc >= (base_wD_acc + args.safe_delta)
                safe_new = new_class_acc if use_class else acc_from_scores(fuse_scores(rn_n, vit_n, mode, wD), yn)
                safe_base = base_class_acc if use_class else base_wD_acc
                safe_all = all_class_acc if (use_class and have_all) else allD
                add_row(rows, d, seed, mode, "safe_fallback_cache", safe_base, safe_new, safe_all, wD,
                        {"use_classwise": bool(use_class), "safe_delta": args.safe_delta, **best_cfg})

                ob = oracle_trueclass(rn_b, vit_b, yb, mode, weights)
                on = oracle_trueclass(rn_n, vit_n, yn, mode, weights)
                oa = oracle_trueclass(ar["logits"], av["logits"], ar["labels"], mode, weights) if have_all else None
                add_row(rows, d, seed, mode, "classwise_oracle_true_label_diagnostic", ob, on, oa, wD)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# B2N Cache Fusion V2 Summary", "", "`classwise_oracle_true_label_diagnostic` uses target labels and is diagnostic only.",
             "`episodic_selected_cache` selects retrieval hyperparameters on base-class pseudo-new episodes only.",
             "`safe_fallback_cache` falls back to dataset-level wD when classwise cache does not improve base accuracy enough.", "",
             "## Overall", "", "| Mode | Method | Base | New | HM | All |", "|---|---|---:|---:|---:|---:|"]
    G = defaultdict(lambda: {"base": [], "new": [], "hm": [], "all": []})
    for r in rows:
        g = G[(r["mode"], r["method"])]
        for k in ["base", "new", "hm"]:
            g[k].append(float(r[k]))
        if r["all"] is not None:
            g["all"].append(float(r["all"]))
    for (mode, method), v in sorted(G.items()):
        lines.append(f"| {mode} | {method} | {fmt(v['base'])} | {fmt(v['new'])} | {fmt(v['hm'])} | {fmt(v['all'])} |")
    lines += ["", "## Dataset-wise", "", "| Dataset | Mode | Method | Base | New | HM | All | Mean wD |", "|---|---|---|---:|---:|---:|---:|---:|"]
    D = defaultdict(lambda: {"base": [], "new": [], "hm": [], "all": [], "wD": []})
    for r in rows:
        g = D[(r["dataset"], r["mode"], r["method"])]
        for k in ["base", "new", "hm"]:
            g[k].append(float(r[k]))
        if r["all"] is not None:
            g["all"].append(float(r["all"]))
        g["wD"].append(float(r["wD"]))
    for (d, mode, method), v in sorted(D.items()):
        lines.append(f"| {d} | {mode} | {method} | {fmt(v['base'])} | {fmt(v['new'])} | {fmt(v['hm'])} | {fmt(v['all'])} | {mean(v['wD']):.2f} |")
    lines += ["", "## Selected configs", "", "```json", json.dumps(selected, indent=2), "```"]
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".class_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    out.with_suffix(".selected_configs.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"[WROTE] {out}")

if __name__ == "__main__":
    main()
