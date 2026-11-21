"""131 report (CPU, reads only): out/rows.jsonl + out/panel.json -> out/report.md.

Per arm: production pass (vacuous counted as a fail, count shown), selective pass (production pass AND silent firing
pool C <= 0.01; also <= 0.05), silent firing (SF) pool C median / IQR (variant fit, mean over draws), the magnitude
read (pool C, fit: mean and p95 of the post-Top-K target at the silent positions / natural mean anchor activation),
circuit size n, Z / A / C / nec medians, resample pass. Paired deltas against the in-run v4 baseline over the targets
both arms finished, with sign counts. Per-target table; sanity (in-run v4 vs the 2026-11-07 production values); the
run's checks (off rows never held out, no self-draws, full model silent, capture == row SF).

  python -I experiments/131-off-term/report.py [--out experiments/131-off-term/out]
"""
import argparse
import json
import statistics as S
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    out = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" in r:
                out.setdefault((r["seed"], r["arm"]), r)        # an error row stands only until a good row exists
                continue
            out[(r["seed"], r["arm"])] = r
    return out


def med(xs):
    xs = [x for x in xs if x is not None]
    return S.median(xs) if xs else None


def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = p * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def f(v, d=2):
    return "-" if v is None else ("%.*f" % (d, v))


def sf(r, pool="C", var="fit"):
    m = r.get("mag") or {}
    v = (m.get("%s_%s" % (var, pool)) or {}).get("sf")
    if v is None:
        d = ((r.get("donor") or {}).get("silent_firing") or {}).get(var, {}).get(pool)
        v = sum(d) / len(d) if d else None
    return v


def magv(r, k, pool="C", var="fit"):
    return ((r.get("mag") or {}).get("%s_%s" % (var, pool)) or {}).get(k)


def ok(r):
    return r is not None and "error" not in r and not r.get("skip")


def passed(r):
    return bool(ok(r) and r.get("pass_prod") and not r.get("vacuous"))


def sel(r, thr):
    s = sf(r)
    return bool(passed(r) and s is not None and s <= thr)


def prod(r, k):
    return (r.get("prod") or {}).get(k)


FIELDS = [("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", "phi_sup_blind_tk")]


def arm_summary(rs):
    rs = [r for r in rs if ok(r)]
    n = len(rs)
    sfs = [sf(r) for r in rs]
    line = dict(
        N=n, vac=sum(bool(r.get("vacuous")) for r in rs), pass_=sum(passed(r) for r in rs),
        sel01=sum(sel(r, 0.01) for r in rs), sel05=sum(sel(r, 0.05) for r in rs),
        sf_med=med(sfs), sf_q1=q(sfs, 0.25), sf_q3=q(sfs, 0.75),
        mag_mean=med([magv(r, "mean") for r in rs]), mag_p95=med([magv(r, "p95") for r in rs]),
        sf_A=med([sf(r, "A") for r in rs]), sf_R=med([sf(r, "R") for r in rs]), sf_a1_C=med([sf(r, "C", "a1") for r in rs]),
        n_med=med([r.get("n") for r in rs]), rs_pass=sum(bool(r.get("resample_pass")) for r in rs),
        t_fit=med([r.get("t_fit") for r in rs]))
    for nm, k in FIELDS:
        line[nm] = med([prod(r, k) for r in rs])
    return line


def summary_table(rows, arms, keys, title):
    out = ["### %s" % title, "",
           "| arm | N | pass | selective pass (SF_C <= 0.01 / <= 0.05) | SF_C median (IQR) | mag_C mean / p95 (median) "
           "| SF_A / SF_R / SF_C a1 | n median | Z / A / C / nec median | resample pass | fit s median |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = arm_summary([rows.get((k, a)) for k in keys if rows.get((k, a)) is not None])
        if not s["N"]:
            continue
        out.append("| %s | %d | %d%s | %d / %d | %s (%s-%s) | %s / %s | %s / %s / %s | %s | %s / %s / %s / %s | %d | %s |" % (
            a, s["N"], s["pass_"], " (%d vacuous)" % s["vac"] if s["vac"] else "", s["sel01"], s["sel05"],
            f(s["sf_med"], 3), f(s["sf_q1"], 3), f(s["sf_q3"], 3), f(s["mag_mean"], 3), f(s["mag_p95"], 3),
            f(s["sf_A"], 3), f(s["sf_R"], 3), f(s["sf_a1_C"], 3), f(s["n_med"], 0), f(s["Z"]), f(s["A"]), f(s["C"]),
            f(s["nec"]), s["rs_pass"], f(s["t_fit"], 0)))
    return out + [""]


def signs(ds, eps=1e-9):
    return "%d- / %d= / %d+" % (sum(d < -eps for d in ds), sum(abs(d) <= eps for d in ds), sum(d > eps for d in ds))


def paired(rows, arm, keys, base="v4"):
    ks = [k for k in keys if ok(rows.get((k, base))) and ok(rows.get((k, arm)))]
    if not ks:
        return None
    b, o = [rows[(k, base)] for k in ks], [rows[(k, arm)] for k in ks]

    def flips(fn):
        lost = sum(fn(x) and not fn(y) for x, y in zip(b, o))
        gain = sum(fn(y) and not fn(x) for x, y in zip(b, o))
        return "%d -> %d (lost %d, gained %d)" % (sum(map(fn, b)), sum(map(fn, o)), lost, gain)

    def dl(get):
        ds = [get(y) - get(x) for x, y in zip(b, o) if get(x) is not None and get(y) is not None]
        return "%s [%s]" % (f(med(ds), 3), signs(ds)) if ds else "-"

    nr = [y["n"] / x["n"] for x, y in zip(b, o) if x.get("n") and y.get("n")]
    tr = [y["t_fit"] / x["t_fit"] for x, y in zip(b, o) if x.get("t_fit") and y.get("t_fit")]
    return dict(N=len(ks), pass_=flips(passed), sel01=flips(lambda r: sel(r, 0.01)), sel05=flips(lambda r: sel(r, 0.05)),
                rs=flips(lambda r: bool(r.get("resample_pass"))), dsf=dl(sf), dmag=dl(lambda r: magv(r, "mean")),
                dp95=dl(lambda r: magv(r, "p95")), dsfA=dl(lambda r: sf(r, "A")),
                dZ=dl(lambda r: prod(r, "free0_tk")), dA=dl(lambda r: prod(r, "freeM_topk_tk")),
                dC=dl(lambda r: prod(r, "freeN_topk_tk")), dnec=dl(lambda r: prod(r, "phi_sup_blind_tk")),
                n_ratio=med(nr), t_ratio=med(tr))


def paired_table(rows, arms, keys, title):
    out = ["### %s" % title, "",
           "Deltas are arm minus v4, median over targets [sign counts: lower / equal / higher].", "",
           "| arm | N paired | pass | selective <= 0.01 | selective <= 0.05 | resample pass | dSF_C | dmag_C mean | "
           "dmag_C p95 | dSF_A | dZ | dA | dC | dnec | n ratio | fit-time ratio |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a == "v4":
            continue
        p = paired(rows, a, keys)
        if p is None:
            continue
        out.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, p["N"], p["pass_"], p["sel01"], p["sel05"], p["rs"], p["dsf"], p["dmag"], p["dp95"], p["dsfA"], p["dZ"],
            p["dA"], p["dC"], p["dnec"], f(p["n_ratio"]), f(p["t_ratio"])))
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--panel", default=str(HERE / "out" / "panel.json"))   # the deep slice: out_deep/panel_<...>.json
    a = ap.parse_args()
    O = Path(a.out)
    rows = load(O / "rows.jsonl")
    pj = json.loads(Path(a.panel).read_text())
    panel = pj["targets"]
    gdesc = pj.get("groups") or {"hi": "passing, production SF_C >= 0.2", "sel": "passing, production SF_C <= 0.01 "
                                 "(controls)", "fail": "failing, SF_C >= 0.2"}
    pk = {t["seed"]: t for t in panel}
    keys = [t["seed"] for t in panel]
    arms = [x for x in ("v4", "o0.25", "o1", "o4") if any(k[1] == x for k in rows)]
    arms += sorted({k[1] for k in rows} - set(arms))
    tripped = (O / "tripped.txt").read_text().strip() if (O / "tripped.txt").exists() else ""
    L = ["# 131 off term: report", "", "Panel: %s (%d targets)." % (a.panel, len(keys)), "",
         "Rows: %s (%d (target, arm) rows, %d error rows). Arms: %s. Tripped targets: %s." % (
             O / "rows.jsonl", len(rows), sum("error" in r for r in rows.values()), ", ".join(arms), tripped or "none"),
         "Pass = production rule (Z, A, C in [0.75, 1.5], nec >= 0.9; vacuous = fail). SF_C = silent firing, pool C, "
         "variant fit (members alpha x live on the 16 held-out silent contrast contexts, rest from another of them), mean "
         "over draws. mag_C = post-Top-K target at those positions / natural mean anchor activation (median over targets "
         "of the per-target mean / p95).", ""]
    L += summary_table(rows, arms, keys, "All targets")
    for g in gdesc:
        L += summary_table(rows, arms, [k for k in keys if pk[k]["group"] == g], "Group %s (%s)" % (g, gdesc[g]))
    L += paired_table(rows, arms, keys, "Paired vs v4, all targets")
    L += paired_table(rows, arms, [k for k in keys if pk[k]["group"] == "hi"], "Paired vs v4, group hi")
    L += paired_table(rows, arms, [k for k in keys if pk[k]["group"] == "sel"], "Paired vs v4, group sel")
    # per target
    L += ["### Per target", "", "Production (11-07 pod) then each arm: pass / SF_C / mag_C p95 / n / Z A C nec.", "",
          "| target | cell | prod | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    for k in keys:
        t = pk[k]
        cells = []
        for x in arms:
            r = rows.get((k, x))
            if r is None:
                cells.append("")
            elif not ok(r):
                cells.append("error" if "error" in r else r.get("skip", "?"))
            else:
                cells.append("%s / %s / %s / %s / %s %s %s %s" % (
                    "P" if passed(r) else "F", f(sf(r), 3), f(magv(r, "p95"), 2), r.get("n"),
                    f(prod(r, "free0_tk")), f(prod(r, "freeM_topk_tk")), f(prod(r, "freeN_topk_tk")),
                    f(prod(r, "phi_sup_blind_tk"))))
        L.append("| %s | %s | %s / %s / - / %d / %s %s %s %s | %s |" % (
            k, t["cell"], "P" if t["pass_"] else "F", f(t["sf_fit_C"], 3), int(t["n"]), f(t["free0_tk"]),
            f(t["freeM_topk_tk"]), f(t["freeN_topk_tk"]), f(t["phi_sup_blind_tk"]), " | ".join(cells)))
    L.append("")
    # sanity: in-run v4 vs production
    v = [(pk[k], rows[(k, "v4")]) for k in keys if ok(rows.get((k, "v4")))]
    if v:
        agree = sum((t["pass_"]) == passed(r) for t, r in v)
        dd = {nm: med([abs(prod(r, kk) - t[kk]) for t, r in v if prod(r, kk) is not None and t[kk] is not None])
              for nm, kk in FIELDS}
        L += ["### Sanity: in-run v4 vs the 2026-11-07 production circuits", "",
              "Pass agreement %d / %d; median |delta| Z %s A %s C %s nec %s; median n ratio (in-run / production) %s; "
              "SF_C production median %s vs in-run %s (per target in the table above). The in-run fit "
              "differs from the pod's by GPU, DEEP_BS (1 here, 4 on the pod) and kernel noise." % (
                  agree, len(v), f(dd["Z"], 3), f(dd["A"], 3), f(dd["C"], 3), f(dd["nec"], 3),
                  f(med([r["n"] / t["n"] for t, r in v if r.get("n") and t["n"]])), f(med([t["sf_fit_C"] for t, _ in v]), 3),
                  f(med([sf(r) for _, r in v]), 3)), ""]
    # checks
    good = [r for r in rows.values() if ok(r)]
    ch = [r.get("checks") or {} for r in good if r.get("w_O")]
    L += ["### Checks", "",
          "- off rows in the held-out contrast set: %s (O arms, max over fits); off rows == R donors == rec['neg'][:48]: %s"
          % (max([c.get("off_rows_in_heldout", 0) for c in ch], default="-"),
             all(c.get("off_eq_resample_donors") and c.get("off_eq_neg48") for c in ch) if ch else "-"),
          "- off-term self-draws: %s; rows used per fit (min): %s; off term built in every O fit: %s" % (
              sum((r.get("off") or {}).get("self_draws", 0) for r in good),
              min([(r.get("off") or {}).get("rows_seen", 0) for r in good if r.get("w_O")], default="-"),
              all("off" in (r.get("term_names") or []) for r in good if r.get("w_O")) if any(r.get("w_O") for r in good)
              else "-"),
          "- v4 arm term names: %s" % sorted({tuple(r.get("term_names") or []) for r in good if not r.get("w_O")}),
          "- full model at the silent positions (max over rows): %s; capture vs donor-row SF (max diff): %s" % (
              max([(r.get("mag") or {}).get("natural_silent_max") or 0 for r in good], default="-"),
              max([(r.get("mag") or {}).get("sf_check_max_diff") or 0 for r in good], default="-")), ""]
    (O / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
