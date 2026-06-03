#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ubuntu/code/meta_prompt_1}
cd "$ROOT"

mkdir -p third_party/CoOp_clean/trainers
cp scripts/dualenc_feature/coop_dualenc.py third_party/CoOp_clean/trainers/coop_dualenc.py

python - <<'PY'
from pathlib import Path
p = Path("third_party/CoOp_clean/train.py")
s = p.read_text()
line = "import trainers.coop_dualenc\n"
if line not in s:
    anchor = "import trainers.coop\n"
    if anchor not in s:
        raise RuntimeError("Cannot find import trainers.coop anchor in train.py")
    s = s.replace(anchor, anchor + line)
    p.write_text(s)
    print("[PATCHED] train.py imported trainers.coop_dualenc")
else:
    print("[SKIP] train.py already imports trainers.coop_dualenc")
PY

python -m py_compile third_party/CoOp_clean/trainers/coop_dualenc.py

echo "[OK] CoOpDualEnc installed."
