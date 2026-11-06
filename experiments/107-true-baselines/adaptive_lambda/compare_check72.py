"""The 72-target check panel: production v3 (102's check fits) against geometric x3.15 at fixed λ, + the censored
term (latent_pilot), and adaptive λ B3 (this folder). Per arm: passes, gained / lost vs production, and sizes on
targets passing in both (Daniel's criterion: passing targets must not grow), plus the same against the fixed-λ
geometric arm (the like-for-like reference for B3: same lr schedule, only λ adapts).

  PYTHONIOENCODING=utf-8 python experiments/107-true-baselines/adaptive_lambda/compare_check72.py
"""
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_dev24 as C  # noqa: E402

E102 = HERE.parents[1] / "102-pass-improvement" / "out" / "check"
PILOT = HERE.parent / "latent_pilot" / "out"


def production():
    ev, st = {}, {}
    for r in C.jl(E102 / "rows.jsonl"):
        if r.get("arm") == "r1.0_rk1e-1":
            ev[r["seed"]] = dict(r["prod"], vacuous_tk=r.get("vacuous_tk"))
            st[r["seed"]] = dict(n=r.get("n"))
    return ev, st


def compare(name, arm, ref_name, ref):
    ev, st = arm[:2]
    rev, rst = ref[:2]
    ks = [k for k in ev if k in rev and C.passes(ev[k]) is not None and C.passes(rev[k]) is not None]
    if not ks:
        return "| %s | vs %s | 0 | - | - | - |" % (name, ref_name)
    both = [k for k in ks if C.passes(ev[k]) and C.passes(rev[k]) and (st.get(k) or {}).get("n") and (rst.get(k) or {}).get("n")]
    rat = sorted(st[k]["n"] / rst[k]["n"] for k in both)
    return "| %s | vs %s | %d | %d (ref %d) | +%d / -%d | %s |" % (
        name, ref_name, len(ks), sum(C.passes(ev[k]) for k in ks), sum(C.passes(rev[k]) for k in ks),
        sum(1 for k in ks if C.passes(ev[k]) and not C.passes(rev[k])),
        sum(1 for k in ks if C.passes(rev[k]) and not C.passes(ev[k])),
        "%.2f (%.2f-%.2f), max %.2f" % (S.median(rat), rat[len(rat) // 10], rat[(9 * len(rat)) // 10], rat[-1])
        if rat else "-")


def main():
    prod = production()
    geo = C.arm([PILOT / "check_geo_x3.15"])
    arms = {"geo x3.15 (fixed λ)": geo, "geo x3.15 + censored": C.arm([PILOT / "check_geo_x3.15_censor"]),
            "B3 (adaptive λ)": C.arm([HERE / "out" / "check_B3_eps0.10_up10"]),
            # (2026-11-04 overnight) the likely pick: lr warm-up 0.1, ± the censored term
            "geo x3.15 + warm-up": C.arm([PILOT / "check_geo_x3.15_warmup"]),
            "geo x3.15 + warm-up + censored": C.arm([PILOT / "check_geo_x3.15_warmup_censor"])}
    print("| arm | reference | targets | passes | gained / lost | size on passing-in-both: median (p10-p90), max |")
    print("|---|---|---|---|---|---|")
    for name, a in arms.items():
        print(compare(name, a, "production", prod))
    for name in ("geo x3.15 + censored", "geo x3.15 + warm-up", "geo x3.15 + warm-up + censored"):
        print(compare(name, arms[name], "geo x3.15 fixed λ", geo))
    print(compare("B3 (adaptive λ)", arms["B3 (adaptive λ)"], "geo x3.15 fixed λ", geo))
    b4 = C.arm([HERE / "out" / "check36_B4_eps0.10_up3"])
    print(compare("B4 (adaptive λ, rise x3; first 36)", b4, "geo x3.15 fixed λ", geo))
    b3_36 = C.arm([HERE / "out" / "check_B3_eps0.10_up10"])
    keys36 = set((HERE / "check_targets_first36.txt").read_text().split()) if (HERE / "check_targets_first36.txt").exists() else set()
    b3_36 = tuple({k: v for k, v in d.items() if k in keys36} for d in b3_36)
    print(compare("B3 on the same first 36", b3_36, "geo x3.15 fixed λ", geo))
    # C raise-only: C's fit where C raised λ (and it has finished), the fixed-λ geometric fit elsewhere
    up = C.arm(sorted((HERE / "out").glob("check72_C_l*")))
    cup = ({**geo[0], **up[0]}, {**geo[1], **up[1]}, {**geo[2], **up[2]})
    print(compare("C raise-only (%d raised fits done)" % len(up[0]), cup, "geo x3.15 fixed λ", geo))


if __name__ == "__main__":
    main()
