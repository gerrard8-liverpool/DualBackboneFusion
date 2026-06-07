#!/usr/bin/env python3
"""Utilities for Reliability Prior Cache (RPC).

This module is intentionally self-contained and tolerant to multiple logits-cache
formats used by previous dual-backbone experiments.

Supported cache layouts:
1) combined files containing both ViT and RN logits;
2) separate branch files, one file for ViT and one file for RN.
"""
from __future__ import annotations

import csv
import json
import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

KNOWN_DATASETS = [
    "imagenet_sketch", "imagenetv2", "imagenet_a", "imagenet-r", "imagenet_r", "imagenet-a",
    "fgvc_aircraft", "oxford_flowers", "oxford_pets", "stanford_cars", "caltech101",
    "food101", "eurosat", "sun397", "ucf101", "imagenet", "dtd",
]

VIT_ALIASES = [
    "logits_vit", "vit_logits", "logits_vit_b16", "vit_b16_logits", "z_vit", "vit", "logits1", "logits_a",
    "vitb16_logits", "logits_vit-b16",
]
RN_ALIASES = [
    "logits_rn", "rn_logits", "logits_rn101", "rn101_logits", "z_rn", "rn", "logits2", "logits_b",
    "resnet_logits", "resnet101_logits",
]
SINGLE_LOGITS_ALIASES = [
    "logits", "scores", "outputs", "output", "pred", "preds", "y_score", "all_logits", "cls_logits",
]
LABEL_ALIASES = ["labels", "targets", "y", "label", "gt", "ground_truth", "true_labels", "all_labels"]
CLASS_ALIASES = ["class_names", "classnames", "classes", "class_name", "text_labels", "labels_text", "class_names_all"]
TEXT_EMB_ALIASES = ["class_text_features", "text_features", "classnames_features", "class_embeddings", "text_embeddings"]

@dataclass
class CombinedLogitsRecord:
    logits_vit: np.ndarray
    logits_rn: np.ndarray
    labels: np.ndarray
    class_names: Optional[List[str]] = None
    text_embeddings: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        self.logits_vit = np.asarray(self.logits_vit, dtype=np.float64)
        self.logits_rn = np.asarray(self.logits_rn, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if self.logits_vit.ndim != 2 or self.logits_rn.ndim != 2:
            raise ValueError(f"logits must be 2D, got vit={self.logits_vit.shape}, rn={self.logits_rn.shape}")
        if self.logits_vit.shape != self.logits_rn.shape:
            raise ValueError(f"vit/rn shape mismatch: {self.logits_vit.shape} vs {self.logits_rn.shape}")
        if self.labels.shape[0] != self.logits_vit.shape[0]:
            raise ValueError(f"labels N mismatch: labels={self.labels.shape}, logits={self.logits_vit.shape}")
        if self.class_names is not None and len(self.class_names) != self.logits_vit.shape[1]:
            raise ValueError(f"class_names length mismatch: {len(self.class_names)} vs {self.logits_vit.shape[1]}")
        if self.text_embeddings is not None:
            self.text_embeddings = np.asarray(self.text_embeddings, dtype=np.float64)

@dataclass
class BranchLogitsRecord:
    branch: str
    logits: np.ndarray
    labels: np.ndarray
    class_names: Optional[List[str]] = None
    text_embeddings: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

    def validate(self) -> None:
        self.logits = np.asarray(self.logits, dtype=np.float64)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if self.branch not in {"vit", "rn"}:
            raise ValueError(f"unknown branch={self.branch}")
        if self.logits.ndim != 2:
            raise ValueError(f"logits must be 2D, got {self.logits.shape}")
        if self.labels.shape[0] != self.logits.shape[0]:
            raise ValueError(f"labels N mismatch: labels={self.labels.shape}, logits={self.logits.shape}")
        if self.class_names is not None and len(self.class_names) != self.logits.shape[1]:
            raise ValueError(f"class_names length mismatch: {len(self.class_names)} vs {self.logits.shape[1]}")
        if self.text_embeddings is not None:
            self.text_embeddings = np.asarray(self.text_embeddings, dtype=np.float64)


def to_numpy(x: Any) -> Any:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, Mapping):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) > 0 and all(isinstance(v, str) for v in x):
            return list(x)
        try:
            arr = np.asarray(x)
            if arr.dtype.kind in "biufc":
                return arr
        except Exception:
            pass
        return [to_numpy(v) for v in x]
    return x


def load_any(path: str | Path) -> Any:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".pt", ".pth", ".tar"] or path.name.endswith(".pth.tar"):
        if torch is None:
            raise RuntimeError("torch is required to load torch checkpoint/logit files")
        return to_numpy(torch.load(path, map_location="cpu"))
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return to_numpy(pickle.load(f))
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return to_numpy(json.load(f))
    raise ValueError(f"unsupported file suffix: {path}")


def summarize_object(obj: Any, max_depth: int = 3, depth: int = 0) -> Any:
    if isinstance(obj, np.ndarray):
        return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, Mapping):
        if depth >= max_depth:
            return {"type": "dict", "keys": list(obj.keys())[:50], "truncated": True}
        return {str(k): summarize_object(v, max_depth, depth + 1) for k, v in list(obj.items())[:80]}
    if isinstance(obj, (list, tuple)):
        if not obj:
            return {"type": type(obj).__name__, "len": 0}
        if all(isinstance(v, str) for v in obj[:20]):
            return {"type": type(obj).__name__, "len": len(obj), "sample": list(obj[:5])}
        if depth >= max_depth:
            return {"type": type(obj).__name__, "len": len(obj), "truncated": True}
        return {"type": type(obj).__name__, "len": len(obj), "first": summarize_object(obj[0], max_depth, depth + 1)}
    return {"type": type(obj).__name__, "value": repr(obj)[:160]}


def _get_first(d: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    lower = {str(k).lower(): k for k in d.keys()}
    for a in aliases:
        if a.lower() in lower:
            return d[lower[a.lower()]]
    return None


def _as_list_str(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        # np scalar strings may be bytes
        return [str(v.decode("utf-8") if isinstance(v, bytes) else v) for v in x]
    return None


def _find_combined_dict(obj: Any, payload_split: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        if payload_split and payload_split in obj and isinstance(obj[payload_split], Mapping):
            found = _find_combined_dict(obj[payload_split], None)
            if found is not None:
                return found
        if _get_first(obj, VIT_ALIASES) is not None and _get_first(obj, RN_ALIASES) is not None and _get_first(obj, LABEL_ALIASES) is not None:
            return obj
        for key in ["data", "payload", "result", "results", "logits", "eval", "cache"]:
            if key in obj:
                found = _find_combined_dict(obj[key], payload_split)
                if found is not None:
                    return found
        for v in obj.values():
            if isinstance(v, Mapping):
                found = _find_combined_dict(v, payload_split)
                if found is not None:
                    return found
    return None


def _find_branch_dict(obj: Any, payload_split: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        if payload_split and payload_split in obj and isinstance(obj[payload_split], Mapping):
            found = _find_branch_dict(obj[payload_split], None)
            if found is not None:
                return found
        if _get_first(obj, SINGLE_LOGITS_ALIASES + VIT_ALIASES + RN_ALIASES) is not None and _get_first(obj, LABEL_ALIASES) is not None:
            return obj
        for key in ["data", "payload", "result", "results", "logits", "eval", "cache"]:
            if key in obj:
                found = _find_branch_dict(obj[key], payload_split)
                if found is not None:
                    return found
        for v in obj.values():
            if isinstance(v, Mapping):
                found = _find_branch_dict(v, payload_split)
                if found is not None:
                    return found
    return None


def infer_branch_from_path(path: str | Path) -> Optional[str]:
    text = str(path).lower().replace("-", "_").replace("\\", "/")
    name = Path(path).name.lower().replace("-", "_")

    # Current dualenc_cache layout uses filenames such as:
    #   logits_vit.npz / logits_rn.npz
    # The earlier implementation recognized plain "vit" but deliberately did
    # not recognize plain "rn", which caused all RN files to fail parsing.
    # Handle explicit filename conventions first to avoid accidental matches.
    if name in {"logits_vit.npz", "vit_logits.npz", "logits_vit.npy", "vit_logits.npy"} or name.startswith("logits_vit") or name.startswith("vit_logits"):
        return "vit"
    if name in {"logits_rn.npz", "rn_logits.npz", "logits_rn.npy", "rn_logits.npy"} or name.startswith("logits_rn") or name.startswith("rn_logits"):
        return "rn"

    vit_patterns = ["vit_b16", "vitb16", "vit_b_16", "visiontransformer"]
    rn_patterns = ["rn101", "resnet101", "resnet_101", "rn_101"]
    if any(p in text for p in vit_patterns):
        return "vit"
    if any(p in text for p in rn_patterns):
        return "rn"

    # Safe path-token fallback. Do not match arbitrary substrings such as
    # 'return' or 'kernel'; only exact path components / filename tokens.
    parts = [x for x in text.split("/") if x]
    if "vit" in parts or name.startswith("vit_"):
        return "vit"
    if "rn" in parts or name.startswith("rn_"):
        return "rn"
    return None




def _load_sidecar_meta(path: str | Path) -> Mapping[str, Any] | None:
    """Load sibling metadata such as logits_vit.meta.json / logits_rn.meta.json.

    Earlier dualenc logits files store only logits/labels/text_features in .npz,
    while class names and branch metadata are stored in a sibling .meta.json file.
    RPC needs these class names for semantic class-cache retrieval.
    """
    path = Path(path)
    candidates = [
        path.with_name(path.stem + ".meta.json"),          # logits_vit.npz -> logits_vit.meta.json
        path.with_suffix(path.suffix + ".meta.json"),      # logits_vit.npz -> logits_vit.npz.meta.json
        path.with_name(path.name + ".meta.json"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, Mapping):
                    obj = to_numpy(obj)
                    obj["_sidecar_path"] = str(c)
                    return obj
            except Exception:
                continue
    return None


def load_combined_record(path: str | Path, payload_split: Optional[str] = None) -> CombinedLogitsRecord:
    obj = load_any(path)
    d = _find_combined_dict(obj, payload_split)
    if d is None:
        raise ValueError("not a combined dual-branch logits file")
    rec = CombinedLogitsRecord(
        logits_vit=np.asarray(_get_first(d, VIT_ALIASES), dtype=np.float64),
        logits_rn=np.asarray(_get_first(d, RN_ALIASES), dtype=np.float64),
        labels=np.asarray(_get_first(d, LABEL_ALIASES), dtype=np.int64),
        class_names=_as_list_str(_get_first(d, CLASS_ALIASES)),
        text_embeddings=_get_first(d, TEXT_EMB_ALIASES),
        meta={"available_keys": list(d.keys()), "source_path": str(path)},
    )
    rec.validate()
    return rec


def load_branch_record(path: str | Path, payload_split: Optional[str] = None, branch: Optional[str] = None) -> BranchLogitsRecord:
    obj = load_any(path)
    d = _find_branch_dict(obj, payload_split)
    if d is None:
        raise ValueError("not a single-branch logits file")
    sidecar = _load_sidecar_meta(path)
    branch = branch or infer_branch_from_path(path)
    # Key-level branch fallback if path is ambiguous.
    if branch is None:
        if _get_first(d, VIT_ALIASES) is not None:
            branch = "vit"
        elif _get_first(d, RN_ALIASES) is not None:
            branch = "rn"
        elif sidecar is not None and str(sidecar.get("branch", "")).lower() in {"vit", "vit_b16", "vit-b16"}:
            branch = "vit"
        elif sidecar is not None and str(sidecar.get("branch", "")).lower() in {"rn", "rn101", "rn_101", "resnet101"}:
            branch = "rn"
    if branch is None:
        raise ValueError("could not infer branch from path or keys; expected rn101/rn or vit_b16/vit")
    if branch == "vit":
        logits = _get_first(d, VIT_ALIASES)
        if logits is None:
            logits = _get_first(d, SINGLE_LOGITS_ALIASES)
    else:
        logits = _get_first(d, RN_ALIASES)
        if logits is None:
            logits = _get_first(d, SINGLE_LOGITS_ALIASES)
    if logits is None:
        logits = _get_first(d, SINGLE_LOGITS_ALIASES)
    class_names = _as_list_str(_get_first(d, CLASS_ALIASES))
    if class_names is None and sidecar is not None:
        class_names = _as_list_str(_get_first(sidecar, CLASS_ALIASES))
    text_embeddings = _get_first(d, TEXT_EMB_ALIASES)
    meta = {"available_keys": list(d.keys()), "source_path": str(path)}
    if sidecar is not None:
        meta["sidecar_path"] = sidecar.get("_sidecar_path", "")
        meta["sidecar_keys"] = [k for k in sidecar.keys() if k != "_sidecar_path"]
    rec = BranchLogitsRecord(
        branch=branch,
        logits=np.asarray(logits, dtype=np.float64),
        labels=np.asarray(_get_first(d, LABEL_ALIASES), dtype=np.int64),
        class_names=class_names,
        text_embeddings=text_embeddings,
        meta=meta,
    )
    rec.validate()
    return rec


def save_combined_npz(rec: CombinedLogitsRecord, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "logits_vit": rec.logits_vit.astype(np.float32),
        "logits_rn": rec.logits_rn.astype(np.float32),
        "labels": rec.labels.astype(np.int64),
    }
    if rec.class_names is not None:
        kwargs["class_names"] = np.asarray(rec.class_names, dtype=object)
    if rec.text_embeddings is not None:
        kwargs["text_embeddings"] = np.asarray(rec.text_embeddings, dtype=np.float32)
    np.savez_compressed(path, **kwargs)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmax(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(np.nansum(e, axis=axis, keepdims=True), 1e-12)


def standardize_per_sample(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return (logits - logits.mean(axis=1, keepdims=True)) / np.maximum(logits.std(axis=1, keepdims=True), 1e-6)


def transform_logits(logits_vit: np.ndarray, logits_rn: np.ndarray, fusion_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    if fusion_mode == "raw_logits":
        return logits_vit, logits_rn
    if fusion_mode == "std_logits":
        return standardize_per_sample(logits_vit), standardize_per_sample(logits_rn)
    if fusion_mode == "prob_avg":
        return softmax_np(logits_vit, axis=1), softmax_np(logits_rn, axis=1)
    raise ValueError(f"unknown fusion_mode={fusion_mode}")


def fuse_outputs(z_vit: np.ndarray, z_rn: np.ndarray, w: float | np.ndarray) -> np.ndarray:
    if np.isscalar(w):
        w = float(w)
        return w * z_vit + (1.0 - w) * z_rn
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1 and w.shape[0] == z_vit.shape[1]:
        return z_vit * w.reshape(1, -1) + z_rn * (1.0 - w.reshape(1, -1))
    if w.ndim == 1 and w.shape[0] == z_vit.shape[0]:
        return z_vit * w.reshape(-1, 1) + z_rn * (1.0 - w.reshape(-1, 1))
    raise ValueError(f"unsupported weight shape {w.shape} for logits {z_vit.shape}")


def accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return float(np.mean(np.argmax(scores, axis=1) == labels) * 100.0) if labels.size else float("nan")


def entropy_from_scores(scores: np.ndarray, already_prob: bool = False) -> np.ndarray:
    p = scores if already_prob else softmax_np(scores, axis=1)
    return -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=1)


def true_margin(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    s_true = scores[np.arange(labels.size), labels]
    masked = scores.copy()
    masked[np.arange(labels.size), labels] = -np.inf
    return s_true - np.max(masked, axis=1)


def best_weight(zv: np.ndarray, zr: np.ndarray, labels: np.ndarray, grid: Sequence[float], tie_break_w: float) -> Tuple[float, float]:
    best_acc, best_w = -1.0, float(tie_break_w)
    for w in grid:
        acc = accuracy(fuse_outputs(zv, zr, float(w)), labels)
        if acc > best_acc + 1e-10:
            best_acc, best_w = acc, float(w)
        elif abs(acc - best_acc) <= 1e-10 and abs(float(w) - tie_break_w) < abs(best_w - tie_break_w):
            best_w = float(w)
    return best_w, best_acc


def weight_grid(step: float) -> List[float]:
    n = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(n + 1)]


def harmonic_mean(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a + b <= 0:
        return float("nan")
    return 2.0 * a * b / (a + b)


def safe_dataset_name_from_path(path: str | Path) -> str:
    text = str(path).lower().replace("-", "_")
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        ds_norm = ds.replace("-", "_").lower()
        if re.search(rf"(^|[/_.-]){re.escape(ds_norm)}($|[/_.-])", text):
            return ds_norm
    # fallback substring
    for ds in sorted(KNOWN_DATASETS, key=len, reverse=True):
        ds_norm = ds.replace("-", "_").lower()
        if ds_norm in text:
            return ds_norm
    return "unknown"


def safe_seed_from_path(path: str | Path) -> str:
    text = str(path).lower()
    for pat in [r"seed[_-]?(\d+)", r"[/_]s(\d+)([/_.-]|$)", r"_seed(\d+)"]:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return "unknown"


def safe_split_from_path(path: str | Path) -> str:
    parts = [p.lower() for p in Path(path).parts]
    for s in ["base", "new", "all", "val", "valid", "validation", "test", "train"]:
        if s in parts:
            return "val" if s in ["valid", "validation"] else s
    text = str(path).lower()
    for s in ["base", "new", "all", "val", "test", "train"]:
        if re.search(rf"(^|[_/.-]){s}($|[_/.-])", text):
            return s
    return "unknown"


def infer_protocol(path: str | Path) -> str:
    text = str(path).lower()
    if "b2n" in text or "base2new" in text or "base_to_new" in text:
        return "b2n"
    if "strict" in text:
        return "strict_dg"
    if "xd" in text or "cross" in text or "source_imagenet" in text:
        return "xd"
    return "unknown"


def clean_name(x: Any) -> str:
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)


def char_counter(text: str) -> Counter:
    text = " " + clean_name(text) + " "
    c = Counter()
    for tok in clean_name(text).split():
        c["tok:" + tok] += 3
    for n in [2, 3, 4]:
        for i in range(max(0, len(text) - n + 1)):
            c[text[i:i+n]] += 1
    return c


def counter_cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in set(a) & set(b))
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    return float(dot / max(na * nb, 1e-12))


def name_similarity(query: str, keys: Sequence[str]) -> np.ndarray:
    q = char_counter(query)
    return np.asarray([counter_cos(q, char_counter(k)) for k in keys], dtype=np.float64)


def soft_topk(sim: np.ndarray, top_k: int, temp: float) -> Tuple[np.ndarray, np.ndarray]:
    sim = np.asarray(sim, dtype=np.float64)
    if sim.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    k = max(1, min(int(top_k), sim.size))
    idx = np.argpartition(-sim, np.arange(k))[:k]
    idx = idx[np.argsort(-sim[idx])]
    z = sim[idx] / max(temp, 1e-6)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.maximum(np.sum(p), 1e-12)
    return idx, p


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not keys:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "No rows.\n"
    out = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"
