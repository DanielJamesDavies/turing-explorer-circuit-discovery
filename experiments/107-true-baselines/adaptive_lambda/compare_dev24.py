"""Adaptive λ on the 24 dev targets: every arm against the fixed-λ reference (dev24_base). Per arm: passes (102's rule,
held-out strongest), gained / lost vs the reference, and sizes: on targets passing in BOTH (Daniel's criterion: passing
targets must not grow), and the median passing size. C's per-λ OUTs (dev24_C_l*) are merged into one arm. --detail
prints every target.

  PYTHONIOENCODING=utf-8 python experiments/107-true-baselines/adaptive_lambda/compare_dev24.py [--detail]
"""
import glob
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BAND = (0.75, 1.5)


def jl(pattern):
    out = []
    for p in sorted(glob.glob(str(pattern))):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def passes(e):
    if e is None or e.get("vacuous_tk"):
        return None
    v = [e.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = e.get("phi_sup_blind_tk")
    return bool(all(isinstance(x, (int, float)) and BAND[0] <= x <= BAND[1] for x in v)
                and isinstance(nec, (int, float)) and nec >= 0.9)


def arm(dirs):
    ev, st, tr = {}, {}, {}
    for d in dirs:
        ev.update({r["seed"]: r for r in jl(d / "main" / "eval.shard*.jsonl") if r.get("held") == "strong"})
        st.update({r["seed"]: r for r in jl(d / "main" / "status.shard*.jsonl") if r.get("ok")})
        tr.update({r["seed"]: r for r in jl(d / "main" / "train.shard*.jsonl")})
    return ev, st, tr


def main():
    detail = "--detail" in sys.argv
    arms = {"fixed λ 1e-3": [OUT / "dev24_base"], "C (predicted λ)": sorted(OUT.glob("dev24_C_l*"))}
    for d in sorted(OUT.glob("dev24_B*")) + sorted(OUT.glob("dev24_X_*")):
        arms[d.name.replace("dev24_", "")] = [d]
    data = {k: arm(v) for k, v in arms.items()}
    ref_ev, ref_st, _ = data["fixed λ 1e-3"]
    print("| arm | done | passes | gained / lost vs fixed | size on passing-in-both: median ratio (p10-p90) | "
          "median passing size |")
    print("|---|---|---|---|---|---|")
    for name, (ev, st, tr) in data.items():
        ks = [k for k in ev if k in ref_ev and passes(ev[k]) is not None and passes(ref_ev[k]) is not None]
        if not ks:
            print("| %s | 0 | - | - | - | - |" % name)
            continue
        npass = sum(passes(ev[k]) for k in ks)
        ref_pass = sum(passes(ref_ev[k]) for k in ks)
        gained = sum(1 for k in ks if passes(ev[k]) and not passes(ref_ev[k]))
        lost = sum(1 for k in ks if passes(ref_ev[k]) and not passes(ev[k]))
        both = [k for k in ks if passes(ev[k]) and passes(ref_ev[k]) and k in st and k in ref_st]
        rat = sorted(st[k]["n"] / ref_st[k]["n"] for k in both)
        sizes = sorted(st[k]["n"] for k in ks if passes(ev[k]) and k in st)
        print("| %s | %d | %d (fixed: %d) | +%d / -%d | %s | %s |" % (
            name, len(ks), npass, ref_pass, gained, lost,
            "%.2f (%.2f-%.2f)" % (S.median(rat), rat[len(rat) // 10], rat[(9 * len(rat)) // 10]) if rat else "-",
            "%d" % S.median(sizes) if sizes else "-"))
        if detail:
            for k in sorted(ks):
                e, r = ev[k], ref_ev[k]
                lam = ((tr.get(k) or {}).get("curve") or {}).get("lambda")
                f = lambda x: "-" if not isinstance(x, (int, float)) else "%.2f" % x  # noqa: E731
                print("    %-15s %s n %5s (fixed %s %5s) Z/A/C %s/%s/%s nec %s%s" % (
                    k, "PASS" if passes(e) else "fail", (st.get(k) or {}).get("n"), "PASS" if passes(r) else "fail",
                    (ref_st.get(k) or {}).get("n"), f(e.get("free0_tk")), f(e.get("freeM_topk_tk")),
                    f(e.get("freeN_topk_tk")), f(e.get("phi_sup_blind_tk")), " | λ end %.2g" % lam[-1] if lam else ""))


if __name__ == "__main__":
    main()
