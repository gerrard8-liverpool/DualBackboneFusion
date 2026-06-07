#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rpc_core import load_any, load_branch_record, load_combined_record, summarize_object, write_json

SUFFIXES = {".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth", ".json"}


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and (p.suffix.lower() in SUFFIXES or p.name.endswith(".pth.tar")):
            yield p


def main():
    ap = argparse.ArgumentParser(description="Inspect dual-branch logits cache format.")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=80)
    ap.add_argument("--payload-split", default=None)
    args = ap.parse_args()

    rows = []
    for i, p in enumerate(iter_files(Path(args.cache_root)), start=1):
        if i > args.max_files:
            break
        row = {"path": str(p)}
        try:
            rec = load_combined_record(p, payload_split=args.payload_split)
            row.update({"kind": "combined", "parse_ok": True, "shape": list(rec.logits_vit.shape), "classes": len(rec.class_names or [])})
        except Exception as e1:
            try:
                rec = load_branch_record(p, payload_split=args.payload_split)
                row.update({"kind": f"branch:{rec.branch}", "parse_ok": True, "shape": list(rec.logits.shape), "classes": len(rec.class_names or [])})
            except Exception as e2:
                row.update({"kind": "unknown", "parse_ok": False, "combined_error": str(e1), "branch_error": str(e2)})
                try:
                    row["object_summary"] = summarize_object(load_any(p))
                except Exception as e3:
                    row["load_error"] = str(e3)
        rows.append(row)
    write_json(rows, args.out)
    ok = sum(1 for r in rows if r.get("parse_ok"))
    print(f"[DONE] inspected={len(rows)} parse_ok={ok} out={args.out}")


if __name__ == "__main__":
    main()
