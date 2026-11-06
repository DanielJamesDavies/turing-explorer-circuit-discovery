"""Exp 093's v2 circuits as the 083 WCM reference (REF_RUN) for the 8-target local run of exp 098.

cmp_common reads the references from REF_RUN in the 062 sweep layout:
- REF_RUN/sweep/eval.*.jsonl: rows with seed, arm and n;
- REF_RUN/sweep/circuits/<arm>/<key>.pt: the circuits (the per-site counts, the re-score).

Exp 093 fitted one circuit per target and version, at the production price (λ 1e-3, driver.LAM), into
093/out/circuits/v2/. This script writes them in that layout as arm W_0.001. The rows are 093's own score rows (arm
renamed); table2 uses the local re-score (rescore_ref.py, REF=local) instead.

It also writes REF_RUN/WCM_VERSION = v2. The driver's default OUT is REF_RUN, and a v2 process refuses an unmarked
OUT that holds circuits.

  python experiments/098-circuitlasso-baseline/make_ref093.py          (CPU, seconds; idempotent)
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
E093 = HERE.parent / "093-v2-validation" / "out"
REF = HERE / "out" / "ref093"


def main():
    rows = [json.loads(line) for line in open(E093 / "rows.jsonl")]
    v2 = {}
    for r in rows:
        if r.get("arm") == "v2" and "error" not in r and "skip" not in r:
            v2[r["seed"]] = r                                   # the last row per target (resumed runs append)
    (REF / "sweep" / "circuits" / "W_0.001").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(REF / "sweep" / "eval.093v2.jsonl", "w") as fh:
        for key, r in sorted(v2.items()):
            src = E093 / "circuits" / "v2" / ("%s.pt" % key)
            if not src.exists():
                continue
            shutil.copy2(src, REF / "sweep" / "circuits" / "W_0.001" / src.name)
            fh.write(json.dumps(dict(r, arm="W_0.001", arm093="v2", source093="093-v2-validation/out/rows.jsonl")) + "\n")
            n += 1
    (REF / "WCM_VERSION").write_text("v2\n")
    print("wrote %d v2 reference circuits + rows to %s" % (n, REF))


if __name__ == "__main__":
    main()
