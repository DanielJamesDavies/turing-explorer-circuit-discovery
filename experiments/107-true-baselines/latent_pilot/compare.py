"""Latent pilot: per target and arm, production pass (102's rule: Z = free0_tk, A = freeM_topk_tk, C = freeN_topk_tk
in [0.75, 1.5], necessity phi_sup_blind_tk >= 0.9, held-out strongest, not vacuous_tk), size, and the training curve's
zero term (min, worst value after the min, end). Reads out/<arm>/main/{eval,status,train}.shard*.jsonl; the 102 check
row (r1.0_rk1e-1, = native v3, 104 part A) is printed as a reference column.

  python experiments/107-true-baselines/latent_pilot/compare.py
"""
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E102 = HERE.parents[1] / "102-pass-improvement" / "out" / "check"
ARMS = ("const_l1e-3", "geo_l1e-3", "geo_l3e-3", "geo_x3.15_l1e-3", "geo_x3.15_l1e-3_cut32", "geo_x3.15_l1e-3_cut64")
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


def passes(r):
    if r is None or r.get("vacuous_tk"):
        return None
    zac = [r.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = r.get("phi_sup_blind_tk")
    ok = all(isinstance(x, (int, float)) and BAND[0] <= x <= BAND[1] for x in zac)
    return bool(ok and isinstance(nec, (int, float)) and nec >= 0.9)


def zero_curve(tr):
    z = ((tr or {}).get("curve") or {}).get("terms", {}).get("zero") or []
    if not z:
        return None
    i = min(range(len(z)), key=lambda j: z[j])
    return dict(min=z[i], at=i, worst_after=max(z[i:]), end=z[-1])


def main():
    keys = [k for k in (HERE / "targets.txt").read_text().split() if k]
    ref = {r["seed"]: r for r in jl(E102 / "rows.jsonl") if r.get("arm") == "r1.0_rk1e-1"}
    ref_tr = {r["seed"]: r for r in jl(E102 / "train.jsonl") if r.get("arm102") == "r1.0_rk1e-1"}
    data = {}
    # the control: 102's check fits of r1.0_rk1e-1 (= native v3, constant lr, bit-identical per 104 part A); their
    # curve is on a stride-20 grid, so its min / worst are coarser than the pilot's every-step curve
    data["102 v3 (constant)"] = ({k: dict(r["prod"], seed=k) for k, r in ref.items()},
                                 {k: dict(n=r.get("n")) for k, r in ref.items()}, ref_tr)
    for a in ARMS:
        d = HERE / "out" / a / "main"
        ev = {r["seed"]: r for r in jl(d / "eval.shard*.jsonl") if r.get("held") == "strong"}
        st = {r["seed"]: r for r in jl(d / "status.shard*.jsonl") if r.get("ok")}
        tr = {r["seed"]: r for r in jl(d / "train.shard*.jsonl")}
        data[a] = (ev, st, tr)
    fmt = lambda x: "-" if not isinstance(x, (int, float)) else "%.2f" % x  # noqa: E731
    arms = ["102 v3 (constant)"] + list(ARMS)
    print("cells: pass, n, Z/A/C/nec, zero term min→worst after min→end")
    print("| target | " + " | ".join(arms) + " |")
    print("|---|" + "---|" * len(arms))
    tot = {a: [0, 0] for a in arms}
    for k in keys:
        cells = []
        for a in arms:
            ev, st, tr = data[a]
            r = ev.get(k)
            p = passes(r)
            if p is not None:
                tot[a][0] += p
                tot[a][1] += 1
            zc = zero_curve(tr.get(k))
            cells.append("%s, %s, %s/%s/%s/%s, %s" % (
                {True: "PASS", False: "fail", None: "-"}[p], (st.get(k) or {}).get("n", "-"),
                *(fmt((r or {}).get(m)) for m in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")),
                "-" if zc is None else "%.4f→%.3f→%.4f" % (zc["min"], zc["worst_after"], zc["end"])))
        print("| %s | %s |" % (k, " | ".join(cells)))
    print("\npasses: " + ", ".join("%s %d/%d" % (a, *tot[a]) for a in arms))


if __name__ == "__main__":
    main()
