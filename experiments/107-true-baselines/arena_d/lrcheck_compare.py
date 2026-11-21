"""Arena D lr check 2026-11-19, step 5 (CPU; lrcheck_1119.sh): v3 vs v4 on the production path, lrtest's pre-stated rule.

Checks:
  1. provenance: every v3 fit has lr_peak 0.05, every v4 fit 0.1575 (so the version reached the engine);
  2. reproduction: v3 res rc lam 1e-3 vs lrtest C0 (549 latents; size within 25%, metrics within 0.05);
  3. verdict on the new pairs (res rc 3e-3; plt within_rc 1e-4, 1e-3, 1e-2) with lrtest's compare(): v4 "helps" only if
     >= 2/3 of the decided pairs are "better" and no pair loses a pass. Otherwise the revert to v3 is confirmed.
Writes <out>/RESULT.md. Exit 1 on a provenance or reproduction failure (the verdict itself never fails the script).

  python lrcheck_compare.py <lrcheck out dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lrtest"))
from analyze import METRICS, SHARE, compare  # noqa: E402  (lrtest's pre-stated rule, unchanged)

LR = {"v3": 0.05, "v4": 0.1575}


def rows(out: Path):
    got = []
    for ev in sorted(out.glob("*_v[34]/eval.jsonl")):
        fam, ver = ev.parent.name.split("_")
        for line in open(ev):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            fit = {}
            cp = ev.parent / r.get("circuit", "")
            if r.get("circuit") and cp.exists():
                fit = json.loads(cp.read_text()).get("fit", {})
            d = dict(fam=fam, ver=ver, wcm=r.get("wcm"), task=r["sfc_task"], lam=float(r["lam"]), n=r.get("n_latents"),
                     passed=r.get("task_pass"), lr_peak=fit.get("lr_peak"))
            for k, f in METRICS:
                d[k] = r.get(f)
            got.append(d)
    return got


def c0_row():
    p = HERE / "lrtest" / "out" / "runs" / "C0" / "res" / "eval.jsonl"
    for line in open(p):
        r = json.loads(line)
        if r["sfc_task"] == "rc" and abs(float(r["lam"]) - 1e-3) < 1e-12:
            return r
    return None


def fmt(r):
    return "%d | %s | %s" % (r["n"], " / ".join("-" if r[k] is None else "%.3f" % r[k] for k, _ in METRICS), r["passed"])


def main() -> int:
    out = Path(sys.argv[1])
    rs = rows(out)
    lines, fails = ["# Arena D lr check 2026-11-19: v3 vs v4 (production path, bf16 local)", ""], []
    for r in rs:
        if r["wcm"] != r["ver"] or r["lr_peak"] is None or abs(r["lr_peak"] - LR[r["ver"]]) > 1e-9:
            fails.append("provenance %s %s %s lam %g: wcm %s lr_peak %s" % (r["fam"], r["ver"], r["task"], r["lam"],
                                                                         r["wcm"], r["lr_peak"]))
    lines.append("Provenance: %s" % ("OK (v3 lr_peak 0.05, v4 0.1575 on every fit)" if not fails else "; ".join(fails)))
    rep = [r for r in rs if r["fam"] == "res" and r["ver"] == "v3" and abs(r["lam"] - 1e-3) < 1e-12]
    c0 = c0_row()
    if rep and c0:
        r, n0 = rep[0], c0["n_latents"]
        dm = max(abs(r[k] - c0[f]) for k, f in METRICS if r[k] is not None and c0.get(f) is not None)
        ok = abs(math.log(max(r["n"], 1) / n0)) <= math.log(1.25) and dm <= 0.05
        lines.append("Reproduction of lrtest C0 (res rc 1e-3): %d vs %d latents, max |dmetric| %.3f -> %s"
                     % (r["n"], n0, dm, "OK" if ok else "FAIL"))
        if not ok:
            fails.append("C0 reproduction")
    else:
        fails.append("no v3 res rc 1e-3 row or no lrtest C0 row")
        lines.append("Reproduction: MISSING")
    lines += ["", "| fam | task | lam | v3: n / Z / A / C / nec / pass | v4: n / Z / A / C / nec / pass | class |",
              "|---|---|---|---|---|---|"]
    by = {(r["fam"], r["task"], r["lam"], r["ver"]): r for r in rs}
    pairs = []
    for (fam, task, lam, ver), r0 in sorted(by.items()):
        if ver != "v3" or (fam, task, lam, "v4") not in by:
            continue
        r1 = by[(fam, task, lam, "v4")]
        cls, why = compare(r0, r1)
        pairs.append((cls, r0, r1))
        lines.append("| %s | %s | %g | %s | %s | %s (%s) |" % (fam, task, lam, fmt(r0), fmt(r1), cls, why))
    dec = [p for p in pairs if p[0] != "undecided"]
    nb = sum(p[0] == "better" for p in dec)
    lost = [p for p in pairs if p[1]["passed"] and p[2]["passed"] is False]
    helps = bool(dec) and nb >= SHARE * len(dec) and not lost
    lines += ["", "Verdict (lrtest's rule): %d pairs, %d decided, %d better, %d lose a pass -> %s"
              % (len(pairs), len(dec), nb, len(lost),
                 "v4 HELPS: revisit the revert" if helps else "v4 does not help: REVERT TO v3 CONFIRMED")]
    if fails:
        lines.append("CHECK FAILURES: " + "; ".join(fails))
    (out / "RESULT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
