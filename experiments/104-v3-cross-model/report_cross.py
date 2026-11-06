"""104 part B report: v2 vs v3 per public model, paired over targets. CPU only, read-only on the run's rows.

Rates are over the targets with a production row in both arms, minus targets vacuous (vacuous_tk) in either arm.
production pass = eval.wcm_eval.passes on the held-out strongest row (Z, A, C in [0.75, 1.5], necessity >= 0.9; a
missing / undefined score fails); resample pass = the donor test's resample_pass (production pass AND fit recovery A
and C in [0.75, 1.5]; protocol §12.3). Paired bootstrap: 10,000 resamples of targets, seed 20261101, percentile 95% CI
of the (v3 - v2) pass-rate difference; w/l = targets v3 passes and v2 fails / the reverse. Size = members (n); fit
time from the driver's status rows (t_fit, s); donor time = the donor row's t_target_total (one arm per launch).

  PYTHONPATH=src python experiments/104-v3-cross-model/report_cross.py   -> out/tables.md, out/report.json
"""
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.wcm_eval import passes  # noqa: E402

MODELS = ("pythia-70m", "gpt2-small")
ARMS = ("v2", "v3")
SEED, NBOOT = 20261101, 10000
BANDS = {"pythia-70m": (("L0-1", 0, 1), ("L2-3", 2, 3), ("L4-5", 4, 5)),
         "gpt2-small": (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))}


def jl(d, pattern):
    out = []
    for p in sorted(Path(d).glob(pattern)):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def last_by_seed(rows, cond=lambda r: True):
    out = {}
    for r in rows:
        if "error" not in r and cond(r):
            out[r["seed"]] = r
    return out


def boot(a, b):
    """95% percentile CI of mean(b) - mean(a) over paired 0/1 lists (pp)."""
    n = len(a)
    if n == 0:
        return None, None
    rng = random.Random(SEED)
    ds = []
    for _ in range(NBOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        ds.append(sum(b[i] - a[i] for i in idx) / n)
    ds.sort()
    return 100 * ds[int(0.025 * NBOOT)], 100 * ds[int(0.975 * NBOOT) - 1]


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def model_report(F):
    T = [k for k in (HERE / ("targets_%s.txt" % F)).read_text().split() if k]
    data = {}
    for V in ARMS:
        o = HERE / "out" / F / V
        data[V] = dict(status=last_by_seed(jl(o / "main", "status.shard*.jsonl"), lambda r: r.get("ok")),
                       status_all=jl(o / "main", "status.shard*.jsonl"),
                       ev=last_by_seed(jl(o / "main", "eval.shard*.jsonl"), lambda r: r.get("held") == "strong"),
                       donor=last_by_seed(jl(o / "donor", "rows*.jsonl"), lambda r: not r.get("skip")),
                       donor_all=jl(o / "donor", "rows*.jsonl"))
    both = [k for k in T if all(k in data[V]["ev"] for V in ARMS)]
    vac = [k for k in both if any(data[V]["ev"][k].get("vacuous_tk") for V in ARMS)]
    keys = [k for k in both if k not in vac]
    res = dict(n_targets=len(T), n_both=len(both), vacuous=vac, n=len(keys),
               errors={V: sum(1 for r in data[V]["status_all"] if "error" in r) for V in ARMS},
               donor_errors={V: sum(1 for r in data[V]["donor_all"] if "error" in r) for V in ARMS},
               missing_donor={V: [k for k in keys if k not in data[V]["donor"]] for V in ARMS})
    pp = {V: [int(bool(passes(data[V]["ev"][k]))) for k in keys] for V in ARMS}
    rp = {V: [int(bool((data[V]["donor"].get(k) or {}).get("resample_pass"))) for k in keys] for V in ARMS}
    # cross-check: the donor row's pass_prod equals passes(eval row)
    res["pass_prod_mismatch"] = {V: [k for k in keys if k in data[V]["donor"]
                                     and bool(data[V]["donor"][k].get("pass_prod")) != bool(pp[V][keys.index(k)])]
                                 for V in ARMS}
    for name, P in (("production", pp), ("resample", rp)):
        lo, hi = boot(P["v2"], P["v3"])
        res[name] = dict(v2=sum(P["v2"]), v3=sum(P["v3"]),
                         d_pp=100 * (sum(P["v3"]) - sum(P["v2"])) / max(len(keys), 1), ci=(lo, hi),
                         wins=sum(1 for a, b in zip(P["v2"], P["v3"]) if b and not a),
                         losses=sum(1 for a, b in zip(P["v2"], P["v3"]) if a and not b))
    st = {V: data[V]["status"] for V in ARMS}
    res["size"] = {V: med([st[V].get(k, {}).get("n") for k in keys]) for V in ARMS}
    res["size_ratio_median"] = med([st["v3"][k]["n"] / st["v2"][k]["n"] for k in keys
                                    if st["v2"].get(k, {}).get("n") and st["v3"].get(k, {}).get("n")])
    res["fit_s_median"] = {V: med([st[V].get(k, {}).get("t_fit") for k in keys]) for V in ARMS}
    res["fit_s_total"] = {V: sum(st[V].get(k, {}).get("t_fit") or 0 for k in keys) for V in ARMS}
    res["fit_ratio_median"] = med([st["v3"][k]["t_fit"] / st["v2"][k]["t_fit"] for k in keys
                                   if st["v2"].get(k, {}).get("t_fit") and st["v3"].get(k, {}).get("t_fit")])
    res["eval_s_median"] = {V: med([st[V].get(k, {}).get("t_eval") for k in keys]) for V in ARMS}
    dn = {V: data[V]["donor"] for V in ARMS}
    res["donor_s_median"] = {V: med([(dn[V].get(k) or {}).get("t_target_total") for k in keys]) for V in ARMS}
    res["donor_s_total"] = {V: sum((dn[V].get(k) or {}).get("t_target_total") or 0 for k in keys) for V in ARMS}
    res["recovery_median"] = {V: {p: med([((dn[V].get(k) or {}).get("recovery") or {}).get("fit", {}).get(p)
                                          for k in keys]) for p in ("A", "C", "R")} for V in ARMS}
    res["by_band"] = {}
    for b, lo, hi in BANDS[F]:
        ks = [i for i, k in enumerate(keys) if lo <= int(k.split(".")[0]) <= hi]
        res["by_band"][b] = dict(n=len(ks), **{"%s_%s" % (m, V): sum(P[V][i] for i in ks)
                                               for m, P in (("prod", pp), ("rs", rp)) for V in ARMS})
    res["per_target"] = {k: {V: dict(n=st[V].get(k, {}).get("n"), t_fit=st[V].get(k, {}).get("t_fit"),
                                     prod=pp[V][i], rs=rp[V][i],
                                     Z=data[V]["ev"][k].get("free0_tk"), A=data[V]["ev"][k].get("freeM_topk_tk"),
                                     C=data[V]["ev"][k].get("freeN_topk_tk"), nec=data[V]["ev"][k].get("phi_sup_blind_tk"),
                                     rA=((dn[V].get(k) or {}).get("recovery") or {}).get("fit", {}).get("A"),
                                     rC=((dn[V].get(k) or {}).get("recovery") or {}).get("fit", {}).get("C"))
                             for V in ARMS} for i, k in enumerate(keys)}
    return res


def f(v, p=0):
    return "-" if v is None else ("%.*f" % (p, v))


def main():
    J = {F: model_report(F) for F in MODELS if (HERE / ("targets_%s.txt" % F)).exists()}
    L = ["# 104 part B: v2 vs v3 on the public models (floor 0.75, paired, n = targets minus vacuous)", "",
         "| model | n | arm | production | resample | d production [95% CI] (w/l) | d resample [95% CI] (w/l) | "
         "median size | size x v2 | median fit s | fit x v2 | median donor s |",
         "|---|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for F, r in J.items():
        for V in ARMS:
            d = ""
            if V == "v3":
                d = " | ".join("%+.0f pp [%s, %s] (%d/%d)" % (r[m]["d_pp"], f(r[m]["ci"][0]), f(r[m]["ci"][1]),
                                                              r[m]["wins"], r[m]["losses"])
                               for m in ("production", "resample"))
            else:
                d = "- | -"
            L.append("| %s | %d | %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                F, r["n"], V, r["production"][V], r["resample"][V], d, f(r["size"][V]),
                f(r["size_ratio_median"], 2) if V == "v3" else "1.00", f(r["fit_s_median"][V], 1),
                f(r["fit_ratio_median"], 2) if V == "v3" else "1.00", f(r["donor_s_median"][V], 1)))
    L += ["", "## By band (passes of n)", "", "| model | band | n | prod v2 | prod v3 | rs v2 | rs v3 |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for F, r in J.items():
        for b, x in r["by_band"].items():
            L.append("| %s | %s | %d | %d | %d | %d | %d |" % (F, b, x["n"], x["prod_v2"], x["prod_v3"], x["rs_v2"],
                                                            x["rs_v3"]))
    L += ["", "## Checks", ""]
    for F, r in J.items():
        L.append("- %s: %d targets, %d with rows in both arms, vacuous %s; fit error rows %s, donor error rows %s; "
                 "missing donor rows %s; donor pass_prod vs eval mismatches %s; median fit recovery %s; fit s total %s, "
                 "donor s total %s" % (F, r["n_targets"], r["n_both"], r["vacuous"], r["errors"], r["donor_errors"],
                                       {V: len(v) for V, v in r["missing_donor"].items()},
                                       {V: len(v) for V, v in r["pass_prod_mismatch"].items()},
                                       {V: {p: f(x, 3) for p, x in rv.items()} for V, rv in r["recovery_median"].items()},
                                       {V: round(x) for V, x in r["fit_s_total"].items()},
                                       {V: round(x) for V, x in r["donor_s_total"].items()}))
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "tables.md").write_text("\n".join(L) + "\n")
    (HERE / "out" / "report.json").write_text(json.dumps(J, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
