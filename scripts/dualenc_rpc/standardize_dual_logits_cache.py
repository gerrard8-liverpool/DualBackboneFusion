#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from rpc_core import (
    CombinedLogitsRecord,
    infer_protocol,
    load_branch_record,
    load_combined_record,
    safe_dataset_name_from_path,
    safe_seed_from_path,
    safe_split_from_path,
    save_combined_npz,
    write_csv,
    write_json,
)

SUFFIXES = {".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth", ".json"}
BRANCH_TOKENS = [
    "vit_b16", "vit-b16", "vitb16", "vit_b_16", "vit", "rn101", "rn_101", "resnet101", "resnet_101", "rn"
]


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and (p.suffix.lower() in SUFFIXES or p.name.endswith(".pth.tar")):
            yield p


def rel_or_abs(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return str(path.relative_to(base))
        except Exception:
            pass
    return str(path)


def label_hash(labels: np.ndarray) -> str:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()[:12]


def normalized_rel_key(path: Path, root: Path) -> str:
    """Path-based fallback key with branch tokens removed.

    This helps pair files such as .../rn101/.../dtd_seed1_new.npz and
    .../vit_b16/.../dtd_seed1_new.npz when explicit dataset/seed/split parsing is weak.
    """
    try:
        s = str(path.relative_to(root)).lower()
    except Exception:
        s = str(path).lower()
    s = s.replace("\\", "/")
    for tok in BRANCH_TOKENS:
        s = s.replace(tok.lower(), "BRANCH")
    # Keep only stable separators.
    while "BRANCH/BRANCH" in s:
        s = s.replace("BRANCH/BRANCH", "BRANCH")
    return s


def primary_key(path: Path, rec, root: Path) -> Tuple[Any, ...]:
    return (
        safe_dataset_name_from_path(path),
        safe_seed_from_path(path),
        safe_split_from_path(path),
        infer_protocol(path),
        int(rec.logits.shape[0]),
        int(rec.logits.shape[1]),
        label_hash(rec.labels),
    )


def loose_key(path: Path, rec, root: Path) -> Tuple[Any, ...]:
    return (
        safe_dataset_name_from_path(path),
        safe_seed_from_path(path),
        infer_protocol(path),
        int(rec.logits.shape[0]),
        int(rec.logits.shape[1]),
        label_hash(rec.labels),
    )


def path_key(path: Path, rec, root: Path) -> Tuple[Any, ...]:
    return (
        normalized_rel_key(path, root),
        int(rec.logits.shape[0]),
        int(rec.logits.shape[1]),
        label_hash(rec.labels),
    )


def meta_from_pair(vit_p: Path, rn_p: Path, rec) -> Tuple[str, str, str, str]:
    # Prefer values inferred from the ViT path, then RN path, then unknown.
    ds = safe_dataset_name_from_path(vit_p)
    if ds == "unknown":
        ds = safe_dataset_name_from_path(rn_p)
    seed = safe_seed_from_path(vit_p)
    if seed == "unknown":
        seed = safe_seed_from_path(rn_p)
    split = safe_split_from_path(vit_p)
    if split == "unknown":
        split = safe_split_from_path(rn_p)
    proto = infer_protocol(vit_p)
    if proto == "unknown":
        proto = infer_protocol(rn_p)
    return ds, seed, split, proto


def same_labels(a, b) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def merge_branch(vit_rec, rn_rec) -> CombinedLogitsRecord:
    if vit_rec.logits.shape != rn_rec.logits.shape:
        raise ValueError(f"branch logits shape mismatch: vit={vit_rec.logits.shape}, rn={rn_rec.logits.shape}")
    if not same_labels(vit_rec.labels, rn_rec.labels):
        raise ValueError("branch labels differ")
    class_names = vit_rec.class_names or rn_rec.class_names
    text_embeddings = vit_rec.text_embeddings if vit_rec.text_embeddings is not None else rn_rec.text_embeddings
    rec = CombinedLogitsRecord(vit_rec.logits, rn_rec.logits, vit_rec.labels, class_names, text_embeddings)
    rec.validate()
    return rec


def try_make_pairs(branch_items: List[Tuple[Path, object]], root: Path, key_fn):
    buckets: Dict[Tuple[Any, ...], Dict[str, List[Tuple[Path, object]]]] = defaultdict(lambda: {"vit": [], "rn": []})
    for p, rec in branch_items:
        buckets[key_fn(p, rec, root)][rec.branch].append((p, rec))
    pairs = []
    leftovers = []
    used = set()
    for key, d in buckets.items():
        if d["vit"] and d["rn"]:
            for vit_p, vit in d["vit"]:
                matched = False
                for rn_p, rn in d["rn"]:
                    if id((rn_p, rn)) in used:
                        continue
                    if vit.logits.shape == rn.logits.shape and same_labels(vit.labels, rn.labels):
                        pairs.append((key, vit_p, vit, rn_p, rn))
                        matched = True
                        break
                if not matched:
                    leftovers.append({"key": str(key), "error": f"no label-compatible rn for vit={vit_p}"})
        else:
            leftovers.append({"key": str(key), "error": f"missing branch: vit={len(d['vit'])}, rn={len(d['rn'])}"})
    return pairs, leftovers


def main():
    ap = argparse.ArgumentParser(description="Standardize existing logits cache into paired ViT/RN .npz files and manifest CSV.")
    ap.add_argument("--cache-root", required=True, help="Root containing b2n/strict_dg/xd logits files.")
    ap.add_argument("--out-dir", required=True, help="Output dir for standardized paired npz files.")
    ap.add_argument("--manifest", required=True, help="Output manifest CSV.")
    ap.add_argument("--relative-to", default=None, help="Store manifest paths relative to this directory, usually project root.")
    ap.add_argument("--payload-split", default=None)
    ap.add_argument("--allow-combined", action="store_true", help="Also accept already combined dual-logits files.")
    args = ap.parse_args()

    root = Path(args.cache_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rel_base = Path(args.relative_to).resolve() if args.relative_to else None
    rows = []
    failures = []
    branch_items: List[Tuple[Path, object]] = []

    for p in iter_files(root):
        if args.allow_combined:
            try:
                rec = load_combined_record(p, payload_split=args.payload_split)
                ds = safe_dataset_name_from_path(p)
                seed = safe_seed_from_path(p)
                split = safe_split_from_path(p)
                proto = infer_protocol(p)
                out_path = out_dir / proto / ds / split / f"seed{seed}" / "dual_logits.npz"
                save_combined_npz(rec, out_path)
                rows.append({
                    "dataset": ds, "seed": seed, "split": split, "protocol": proto,
                    "path": rel_or_abs(out_path, rel_base), "source_kind": "combined", "source_file_vit": str(p), "source_file_rn": str(p),
                    "n": rec.logits_vit.shape[0], "num_classes": rec.logits_vit.shape[1],
                })
                continue
            except Exception as e:
                combined_error = str(e)
        else:
            combined_error = "combined disabled"
        try:
            rec = load_branch_record(p, payload_split=args.payload_split)
            branch_items.append((p, rec))
        except Exception as e:
            failures.append({"path": str(p), "combined_error": combined_error, "branch_error": str(e)})

    # Pair in increasingly relaxed ways. Primary key includes split. Loose key drops split.
    # Path key removes branch tokens from relative path. Later duplicate output rows are skipped.
    all_pairs = []
    all_pair_failures = []
    for name, fn in [("primary", primary_key), ("loose", loose_key), ("path", path_key)]:
        pairs, pair_failures = try_make_pairs(branch_items, root, fn)
        if pairs:
            print(f"[PAIRING] mode={name} pairs={len(pairs)}")
            all_pairs = [(name, *p) for p in pairs]
            all_pair_failures = pair_failures
            break
        print(f"[PAIRING] mode={name} pairs=0")
        all_pair_failures = pair_failures

    seen_outputs = set()
    for mode, key, vit_p, vit, rn_p, rn in all_pairs:
        try:
            rec = merge_branch(vit, rn)
            ds, seed, split, proto = meta_from_pair(vit_p, rn_p, rec)
            # Avoid overwriting if metadata inference is weak.
            sig = label_hash(rec.labels)
            out_path = out_dir / proto / ds / split / f"seed{seed}" / sig / "dual_logits.npz"
            if str(out_path) in seen_outputs:
                continue
            seen_outputs.add(str(out_path))
            save_combined_npz(rec, out_path)
            rows.append({
                "dataset": ds, "seed": seed, "split": split, "protocol": proto,
                "path": rel_or_abs(out_path, rel_base), "source_kind": f"paired_branches:{mode}",
                "source_file_vit": str(vit_p), "source_file_rn": str(rn_p),
                "n": rec.logits_vit.shape[0], "num_classes": rec.logits_vit.shape[1],
                "label_hash": sig,
            })
        except Exception as e:
            failures.append({"key": str(key), "vit": str(vit_p), "rn": str(rn_p), "error": str(e)})

    if not rows:
        failures.extend(all_pair_failures[:200])
    write_csv(rows, args.manifest)
    write_json(failures, str(args.manifest) + ".failures.json")
    print(f"[DONE] branch_records={len(branch_items)}")
    print(f"[DONE] standardized paired files={len(rows)} manifest={args.manifest}")
    print(f"[DONE] failures={len(failures)} failure_log={args.manifest}.failures.json")
    if rows:
        print("[SAMPLE]", rows[0])
    else:
        print("[HINT] No pairs were created. Inspect the first failures with:")
        print(f"python - <<'PY'\nimport json\nf='{args.manifest}.failures.json'\nrows=json.load(open(f))\nprint(len(rows))\nfor r in rows[:20]: print(r)\nPY")


if __name__ == "__main__":
    main()
