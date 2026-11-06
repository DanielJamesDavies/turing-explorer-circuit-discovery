"""The 48 dev targets never used to choose anything (targets_dev48.txt): production v3 (102's phase-1 dev fits,
arm r1.0_rk1e-1) against geometric x3.15 and the fixes (censored term, lr warm-up, necessity term, combinations).
Per arm: passes, gained / lost vs production and vs geometric x3.15 alone, and sizes on targets passing in both
(Daniel's criterion: passing targets must not grow); then the targets geometric x3.15 lost, under every arm.

  PYTHONIOENCODING=utf-8 python experiments/107-true-baselines/latent_pilot/compare_dev48.py
"""
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "adaptive_lambda"))
import compare_dev24 as C  # noqa: E402

E102 = HERE.parents[1] / "102-pass-improvement" / "out" / "phase1"
ARMS = ["dev48_geo_x3.15", "dev48_geo_x3.15_censor", "dev48_geo_x3.15_warmup", "dev48_geo_x3.15_nec",
        "dev48_geo_x3.15_warmup_censor"]


def production():
    ev, st = {}, {}
    for r in C.jl(E102 / "rows.jsonl"):
        if r.get("arm") == "r1.0_rk1e-1":
            ev[r["seed"]] = dict(r["prod"], vacuous_tk=r.get("vacuous_tk"))
            st[r["seed"]] = dict(n=r.get("n"))
    return ev, st, {}


def row(name, a, ref_name, ref):
    ev, st = a[:2]
    rev, rst = ref[:2]
    ks = [k for k in ev if k in rev and C.passes(ev[k]) is not None and C.passes(rev[k]) is not None]
    if not ks:
        return "| %s | %s | 0 | - | - | - |" % (name, ref_name)
    both = [k for k in ks if C.passes(ev[k]) and C.passes(rev[k]) and (st.get(k) or {}).get("n") and (rst.get(k) or {}).get("n")]
    rat = sorted(st[k]["n"] / rst[k]["n"] for k in both)
    return "| %s | %s | %d | %d (ref %d) | +%d / -%d | %s |" % (
        name, ref_name, len(ks), sum(C.passes(ev[k]) for k in ks), sum(C.passes(rev[k]) for k in ks),
        sum(1 for k in ks if C.passes(ev[k]) and not C.passes(rev[k])),
        sum(1 for k in ks if C.passes(rev[k]) and not C.passes(ev[k])),
        "%.2f (%.2f-%.2f), max %.2f" % (S.median(rat), rat[len(rat) // 10], rat[(9 * len(rat)) // 10], rat[-1]) if rat else "-")


def main():
    prod = production()
    arms = {a: C.arm([HERE / "out" / a]) for a in ARMS}
    geo = arms["dev48_geo_x3.15"]
    print("| arm | reference | targets | passes | gained / lost | size on passing-in-both: median (p10-p90), max |")
    print("|---|---|---|---|---|---|")
    for a, d in arms.items():
        print(row(a.replace("dev48_", ""), d, "production", prod))
        if a != "dev48_geo_x3.15":
            print(row(a.replace("dev48_", ""), d, "geo x3.15", geo))
    lost = [k for k in geo[0] if k in prod[0] and C.passes(prod[0][k]) and C.passes(geo[0][k]) is False]
    f = lambda x: "-" if not isinstance(x, (int, float)) else "%.2f" % x  # noqa: E731
    print("\ntargets geometric x3.15 lost (production passes):")
    for k in lost:
        print("  " + k)
        for a, d in [("production", prod)] + list(arms.items()):
            e = d[0].get(k)
            if e is None:
                continue
            print("    %-28s %s n %5s Z/A/C %s/%s/%s nec %s" % (
                a.replace("dev48_", ""), "PASS" if C.passes(e) else "fail", (d[1].get(k) or {}).get("n"),
                f(e.get("free0_tk")), f(e.get("freeM_topk_tk")), f(e.get("freeN_topk_tk")), f(e.get("phi_sup_blind_tk"))))


if __name__ == "__main__":
    main()
