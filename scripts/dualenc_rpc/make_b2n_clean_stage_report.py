from pathlib import Path
import csv
from datetime import datetime

PROJECT_ROOT = Path("/workspace/meta_prompt_1")
REPORT_DIR = PROJECT_ROOT / "outputs/reliability_prior_cache/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = REPORT_DIR / "b2n_clean_stage_main_table.md"
OUT_CSV = REPORT_DIR / "b2n_clean_stage_compact_hm.csv"

MODES = ["std_logits", "raw_logits", "prob_avg"]

NON_IMAGENET_TARGETS = [
    "caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101",
    "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"
]

def parse_summary_file(path, line_name, fusion_mode, target_name=None):
    rows = []
    if not path.exists():
        return rows

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "---" in line or "dataset | mode" in line:
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7:
            continue

        dataset, method, split, seeds, acc_mean, acc_std, mean_w = cells

        if split != "HM(base,new)" or dataset == "__AVG__":
            continue

        try:
            rows.append({
                "line": line_name,
                "fusion_mode": fusion_mode,
                "target": target_name if target_name is not None else dataset,
                "dataset": dataset,
                "method": method,
                "seeds": int(seeds),
                "hm": float(acc_mean),
            })
        except Exception:
            pass

    return rows

all_rows = []

# ImageNet Cache B2N clean
img_root = PROJECT_ROOT / "outputs/reliability_prior_cache/imagenet_cache_b2n_clean"
for mode in MODES:
    p = img_root / mode / "eval/rpc_eval_summary.md"
    all_rows.extend(parse_summary_file(p, "ImageNet Cache", mode))

# Meta LODO B2N clean
meta_root = PROJECT_ROOT / "outputs/reliability_prior_cache/meta_lodo_b2n_clean"
for mode in MODES:
    for f in sorted((meta_root / mode).glob("target_*/eval/rpc_eval_summary.md")):
        target = f.parent.parent.name.replace("target_", "")
        all_rows.extend(parse_summary_file(f, "Meta LODO Cache", mode, target_name=target))

def get_value(line, mode, target, method):
    vals = [
        r["hm"] for r in all_rows
        if r["line"] == line
        and r["fusion_mode"] == mode
        and r["target"] == target
        and r["method"] == method
    ]
    return vals[0] if vals else None

def get_seed(line, mode, target):
    vals = [
        r["seeds"] for r in all_rows
        if r["line"] == line
        and r["fusion_mode"] == mode
        and r["target"] == target
        and r["method"] == "fixed"
    ]
    return vals[0] if vals else None

def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None

def fmt(x):
    return "NA" if x is None else f"{x:.2f}"

def fmt_delta(x):
    return "NA" if x is None else f"{x:+.2f}"

compact_rows = []

for line in ["ImageNet Cache", "Meta LODO Cache"]:
    for mode in MODES:
        targets = NON_IMAGENET_TARGETS

        fixeds = [get_value(line, mode, t, "fixed") for t in targets]
        dcs = [get_value(line, mode, t, "dataset_cache") for t in targets]
        ccs = [get_value(line, mode, t, "class_cache") for t in targets]
        ocs = [get_value(line, mode, t, "oracle_dataset") for t in targets]

        af, adc, acc, aoc = avg(fixeds), avg(dcs), avg(ccs), avg(ocs)
        if af is None:
            continue

        compact_rows.append({
            "line": line,
            "fusion_mode": mode,
            "n_targets": len([v for v in fixeds if v is not None]),
            "fixed": af,
            "dataset_cache": adc,
            "class_cache": acc,
            "oracle_dataset": aoc,
            "dataset_cache_minus_fixed": None if adc is None else adc - af,
            "class_cache_minus_fixed": None if acc is None else acc - af,
            "oracle_minus_fixed": None if aoc is None else aoc - af,
        })

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "line", "fusion_mode", "target", "seeds",
        "fixed", "dataset_cache", "class_cache", "oracle_dataset",
        "dataset_cache_minus_fixed", "class_cache_minus_fixed", "oracle_minus_fixed",
    ])

    for line in ["ImageNet Cache", "Meta LODO Cache"]:
        for mode in MODES:
            for t in NON_IMAGENET_TARGETS:
                fixed = get_value(line, mode, t, "fixed")
                dc = get_value(line, mode, t, "dataset_cache")
                cc = get_value(line, mode, t, "class_cache")
                oc = get_value(line, mode, t, "oracle_dataset")
                if fixed is None:
                    continue
                w.writerow([
                    line, mode, t, get_seed(line, mode, t),
                    fmt(fixed), fmt(dc), fmt(cc), fmt(oc),
                    fmt_delta(None if dc is None else dc - fixed),
                    fmt_delta(None if cc is None else cc - fixed),
                    fmt_delta(None if oc is None else oc - fixed),
                ])

lines = []
lines.append("# Reliability Prior Cache: Clean B2N Stage Main Table")
lines.append("")
lines.append("Generated at: `{}`".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
lines.append("")
lines.append("## Protocol")
lines.append("")
lines.append("This report summarizes the current clean B2N stage before adding strict-DG results.")
lines.append("")
lines.append("- **ImageNet Cache B2N clean**: source cache uses `b2n / imagenet / split_all / seed1-3`; target evaluation uses the 10 non-ImageNet B2N datasets.")
lines.append("- **Meta LODO B2N clean**: for each target, source cache uses the other B2N datasets with `all/base/new` splits and seed1-3. For fair comparison with ImageNet Cache, the main average excludes ImageNet target.")
lines.append("- **Metric**: HM(base,new), averaged over 3 seeds.")
lines.append("- **Fair methods**: `fixed`, `dataset_cache`, `class_cache`. `oracle_dataset` is diagnostic upper bound only.")
lines.append("")
lines.append("## Main Table: Clean B2N, Non-ImageNet 10 Targets")
lines.append("")
lines.append("| Line | Fusion | n | Fixed | Dataset Cache | Class Cache | Oracle | Dataset-Fixed | Class-Fixed | Oracle-Fixed |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in compact_rows:
    lines.append(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["line"], r["fusion_mode"], r["n_targets"],
            fmt(r["fixed"]), fmt(r["dataset_cache"]), fmt(r["class_cache"]), fmt(r["oracle_dataset"]),
            fmt_delta(r["dataset_cache_minus_fixed"]),
            fmt_delta(r["class_cache_minus_fixed"]),
            fmt_delta(r["oracle_minus_fixed"]),
        )
    )

lines.append("")
lines.append("## Recommended Stage Interpretation")
lines.append("")
lines.append("- `dataset_cache` is consistently positive across both ImageNet Cache and Meta LODO Cache under all three fusion modes.")
lines.append("- `prob_avg` is currently the strongest and safest fusion interface.")
lines.append("- ImageNet Cache gives the strongest class-wise result under `prob_avg`; Meta LODO Cache gives a robust dataset-level prior.")
lines.append("- `raw_logits + class_cache` is consistently unstable and should be treated as negative evidence for uncalibrated class-wise reliability retrieval.")
lines.append("- Strict-DG results are not included yet and should be appended after the DG run finishes.")
lines.append("")
lines.append("## Per-target Detail: ImageNet Cache B2N Clean")
lines.append("")

for mode in MODES:
    lines.append("### {}".format(mode))
    lines.append("")
    lines.append("| Target | Seeds | Fixed | Dataset Cache | Class Cache | Oracle | Dataset-Fixed | Class-Fixed | Oracle-Fixed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in NON_IMAGENET_TARGETS:
        fixed = get_value("ImageNet Cache", mode, t, "fixed")
        dc = get_value("ImageNet Cache", mode, t, "dataset_cache")
        cc = get_value("ImageNet Cache", mode, t, "class_cache")
        oc = get_value("ImageNet Cache", mode, t, "oracle_dataset")
        if fixed is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                t, get_seed("ImageNet Cache", mode, t),
                fmt(fixed), fmt(dc), fmt(cc), fmt(oc),
                fmt_delta(dc - fixed if dc is not None else None),
                fmt_delta(cc - fixed if cc is not None else None),
                fmt_delta(oc - fixed if oc is not None else None),
            )
        )
    lines.append("")

lines.append("## Per-target Detail: Meta LODO B2N Clean")
lines.append("")

for mode in MODES:
    lines.append("### {}".format(mode))
    lines.append("")
    lines.append("| Target | Seeds | Fixed | Dataset Cache | Class Cache | Oracle | Dataset-Fixed | Class-Fixed | Oracle-Fixed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in NON_IMAGENET_TARGETS:
        fixed = get_value("Meta LODO Cache", mode, t, "fixed")
        dc = get_value("Meta LODO Cache", mode, t, "dataset_cache")
        cc = get_value("Meta LODO Cache", mode, t, "class_cache")
        oc = get_value("Meta LODO Cache", mode, t, "oracle_dataset")
        if fixed is None:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                t, get_seed("Meta LODO Cache", mode, t),
                fmt(fixed), fmt(dc), fmt(cc), fmt(oc),
                fmt_delta(dc - fixed if dc is not None else None),
                fmt_delta(cc - fixed if cc is not None else None),
                fmt_delta(oc - fixed if oc is not None else None),
            )
        )
    lines.append("")

lines.append("## Source Files")
lines.append("")
lines.append("- ImageNet Cache summaries: `outputs/reliability_prior_cache/imagenet_cache_b2n_clean/*/eval/rpc_eval_summary.md`")
lines.append("- Meta LODO summaries: `outputs/reliability_prior_cache/meta_lodo_b2n_clean/*/target_*/eval/rpc_eval_summary.md`")
lines.append("- Compact CSV: `outputs/reliability_prior_cache/reports/b2n_clean_stage_compact_hm.csv`")
lines.append("")

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("[DONE] Markdown report:", OUT_MD)
print("[DONE] Compact CSV:    ", OUT_CSV)
print()
print("Preview:")
print("-" * 100)
print("\n".join(lines[:45]))
