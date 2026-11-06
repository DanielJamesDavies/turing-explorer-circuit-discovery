"""095 report (CPU): out/rows.jsonl (+ out/fits.jsonl, out/check.jsonl) -> out/tables.md. Three arms, paired:
v2 (093's circuits), v2R (Z + A + R_C, R in place of C), v2CR (Z + A + C + R_C, R added); plus 079's v1 confirmation
(W0 vs RS0.25_S0.25, recomputed from 079 out_confirm/rows.jsonl with the same definitions) for reference.

Definitions (as run.py / 079 report.py):
  resample pass  Z (production, post-Top-K) in [0.8, 1.5], rs A and rs C (073, C donors = the 16 held-out contrast
                 contexts) in [0.8, 1.5], necessity (production, post-Top-K) >= 0.9
  rule pass      Z, A_sp, C_sp in [0.8, 1.5], necessity >= 0.9, post-Top-K read (the headline rule)
  pre pass       the same rule on the pre-activation read, Z and necessity on the 0 baseline (protocol 12)
Rows with vacuous_tk are excluded (079); a target counts only when every arm in the report has a complete row.
Pass-rate differences: McNemar exact (two-sided binomial on the discordant pairs) and a paired bootstrap 95% CI
(10,000 resamples of targets, seed 20261026, percentile).

  .venv/bin/python experiments/095-v2-resample-appendix/report.py      (env OUT95 as run.py; ARMS default
  v2,v2R,v2CR: the arms that must be complete for a target to count)
"""
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT95 = Path(os.environ.get("OUT95", str(HERE / "out")))
E79 = EXP / "079-resample-training" / "out_confirm"
ARMS = tuple(a for a in os.environ.get("ARMS", "v2,v2R,v2CR").split(",") if a)
PAIRS = [(a, b) for a, b in (("v2", "v2R"), ("v2", "v2CR"), ("v2R", "v2CR")) if a in ARMS and b in ARMS]
BANDS = ("L0-4", "L5-7", "L8-11")
KINDS = ("attn", "mlp", "resid")
N_BOOT, BOOT_SEED = 10000, 20261026


def ib(v):
    return v is not None and v == v and 0.8 <= v <= 1.5


def load_jsonl(path):
    out = []
    if path.exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


# ------------------------------------------------------------------------------------------------ statistics
def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans a (base), b (new): (n base-only, n new-only, p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * p)


def boot_ci(a, b, seed=BOOT_SEED):
    """Paired bootstrap 95% CI of mean(b) - mean(a) (targets resampled with replacement)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = (b[idx] - a[idx]).mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def med(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else None


def f2(x, nd=2):
    return ("%%.%df" % nd) % x if isinstance(x, (int, float)) and x == x else "-"


def pct(x):
    return "%.0f%%" % (100 * x) if x is not None else "-"


def pp(x):
    return "%+.0f" % (100 * x) if x is not None else "-"


def pval(p):
    return "%.2g" % p if p >= 1e-3 else "%.1e" % p


# ------------------------------------------------------------------------------------------------ data
def unify95(r):
    p = r["prod"]
    return dict(seed=r["seed"], band=r["band"], kind=r["kind"], n=r.get("n"), t_fit=r.get("t_fit"),
                rs_pass=bool(r["rs_pass"]), rule=bool(r["pass_rule"]), pre=bool(r["pass_pre"]),
                pre_e0=bool(r.get("pass_pre_e0")),
                Z=p.get("free0_tk"), A=p.get("freeM_topk_tk"), C=p.get("freeN_topk_tk"), nec=p.get("phi_sup_blind_tk"),
                Zp=r.get("z_pre0"), Ap=p.get("freeM_topk_pre"), Cp=p.get("freeN_topk_pre"), necp=r.get("nec_pre0"),
                rsA=r.get("rs_A"), rsC=r.get("rs_C"), rsR=r.get("rs_R"), alpha=r.get("alpha_med"),
                r_C=(r.get("resample") or {}).get("ap_C_wcm_r"))


def load95():
    """[{arm: unified row}] for targets complete and non-vacuous in every arm of ARMS."""
    rows = {}
    for r in load_jsonl(OUT95 / "rows.jsonl"):
        if "error" not in r:
            rows[(r["seed"], r["arm"])] = r                  # the last complete row per (target, arm)
    seeds = sorted({s for s, _ in rows})
    data, excluded = [], dict(missing=[], skip=[], vacuous=[])
    for s in seeds:
        rr = {a: rows.get((s, a)) for a in ARMS}
        if any(v is None for v in rr.values()):
            excluded["missing"].append(s)
        elif any("skip" in v for v in rr.values()):
            excluded["skip"].append(s)
        elif any(v.get("vacuous_tk") for v in rr.values()):
            excluded["vacuous"].append(s)
        else:
            data.append({a: unify95(v) for a, v in rr.items()})
    return data, excluded, rows


def load79():
    """079's 99-target confirmation, with 079 report.py's definitions (vacuous_tk excluded; complete for prod, W0, RS)."""
    rows = [r for r in load_jsonl(E79 / "rows.jsonl") if "error" not in r and "skip" not in r]
    by = {}
    for r in rows:
        p, s = r["prod_scorer"], r["resample"]
        if p.get("vacuous_tk"):
            continue
        nec = (p.get("phi_sup_blind_tk") or 0) >= 0.9
        by[(r["seed"], r["arm"])] = dict(
            seed=r["seed"], band=r["band"], kind=r["kind"], n=r.get("n"), t_fit=r.get("t_fit"),
            rule=bool(ib(p.get("free0_tk")) and ib(p.get("freeM_topk_tk")) and ib(p.get("freeN_topk_tk")) and nec),
            rs_pass=bool(ib(p.get("free0_tk")) and ib(s.get("rs_A_wcm")) and ib(s.get("rs_C_wcm")) and nec),
            Z=p.get("free0_tk"), A=p.get("freeM_topk_tk"), C=p.get("freeN_topk_tk"), nec=p.get("phi_sup_blind_tk"),
            rsA=s.get("rs_A_wcm"), rsC=s.get("rs_C_wcm"), rsR=s.get("rs_R_wcm"), r_C=s.get("ap_C_wcm_r"))
    seeds = sorted({s for s, _ in by})
    keep = [s for s in seeds if all((s, a) in by for a in ("prod", "W0", "RS0.25_S0.25"))]
    return [{"W0": by[(s, "W0")], "RS": by[(s, "RS0.25_S0.25")]} for s in keep]


# ------------------------------------------------------------------------------------------------ tables
def groups(data):
    yield "all", data
    first = lambda d: next(iter(d.values()))  # noqa: E731
    for b in BANDS:
        yield b, [d for d in data if first(d)["band"] == b]
    for k in KINDS:
        yield k, [d for d in data if first(d)["kind"] == k]


def rates_table(w, data, field, arms):
    w("| group | n | " + " | ".join(arms) + " |")
    w("|---|---|" + "---|" * len(arms))
    for g, ds in groups(data):
        if ds:
            w("| %s | %d | %s |" % (g, len(ds), " | ".join(
                "%s (%d)" % (pct(np.mean([d[a][field] for d in ds])), sum(d[a][field] for d in ds)) for a in arms)))
    w("")


def pass_table(w, data, field, base, new):
    w("| group | n | %s | %s | diff (pp) | 95%% CI (pp) | %s-only / %s-only | McNemar p |" % (base, new, base, new))
    w("|---|---|---|---|---|---|---|---|")
    for g, ds in groups(data):
        if not ds:
            continue
        a = [d[base][field] for d in ds]
        b = [d[new][field] for d in ds]
        lo, hi = boot_ci(a, b)
        n10, n01, p = mcnemar(a, b)
        w("| %s | %d | %s (%d) | %s (%d) | %s | [%s, %s] | %d / %d | %s |" % (
            g, len(ds), pct(np.mean(a)), sum(a), pct(np.mean(b)), sum(b), pp(np.mean(b) - np.mean(a)), pp(lo), pp(hi),
            n10, n01, pval(p)))
    w("")


def median_table(w, data, arms, keys=(("Z", "Z"), ("A", "A"), ("C", "C"), ("nec", "nec"), ("rs A", "rsA"),
                                       ("rs C", "rsC"), ("rs R", "rsR"), ("r all-pos (C)", "r_C"))):
    sl = " / ".join(arms)
    w("| group | n | " + " | ".join("%s %s" % (lab, sl) for lab, _ in keys) + " |")
    w("|---|---|" + "---|" * len(keys))
    for g, ds in groups(data):
        if ds:
            w("| %s | %d | %s |" % (g, len(ds), " | ".join(
                " / ".join(f2(med([d[a][k] for d in ds])) for a in arms) for _, k in keys)))
    w("")


def cost_table(w, data, arms):
    base = arms[0]
    head = ["size %s" % a for a in arms] + ["size ratio %s/%s (median per target)" % (a, base) for a in arms[1:]] + \
           ["fit s %s" % a for a in arms] + ["fit ratio %s/%s (median per target; of sums)" % (a, base) for a in arms[1:]]
    w("| group | n | " + " | ".join(head) + " |")
    w("|---|---|" + "---|" * len(head))
    for g, ds in groups(data):
        if not ds:
            continue
        cells = [f2(med([d[a]["n"] for d in ds]), 0) for a in arms]
        for a in arms[1:]:
            cells.append(f2(med([d[a]["n"] / d[base]["n"] for d in ds if d[a]["n"] and d[base]["n"]])) + "x")
        cells += [f2(med([d[a]["t_fit"] for d in ds]), 0) for a in arms]
        for a in arms[1:]:
            ok = [(d[base]["t_fit"], d[a]["t_fit"]) for d in ds if d[base]["t_fit"] and d[a]["t_fit"]]
            cells.append("%sx; %sx" % (f2(med([y / x for x, y in ok])),
                                       f2(sum(y for _, y in ok) / sum(x for x, _ in ok) if ok else None)))
        w("| %s | %d | %s |" % (g, len(ds), " | ".join(cells)))
    w("")


def split_table(w, data, arm):
    """Protocol 12's 2 x 2: passes both reads / pre-activation only / post-Top-K only / neither."""
    w("| group | n | both | pre-activation only | post-Top-K only | neither | pre pass, e0 baseline (attn + mlp) |")
    w("|---|---|---|---|---|---|---|")
    for g, ds in groups(data):
        if not ds:
            continue
        r = [d[arm] for d in ds]
        am = [x for x in r if x["kind"] in ("attn", "mlp")]
        w("| %s | %d | %d | %d | %d | %d | %s |" % (
            g, len(r), sum(x["rule"] and x["pre"] for x in r), sum(x["pre"] and not x["rule"] for x in r),
            sum(x["rule"] and not x["pre"] for x in r), sum(not x["rule"] and not x["pre"] for x in r),
            ("%s (%d/%d)" % (pct(np.mean([x["pre_e0"] for x in am])), sum(x["pre_e0"] for x in am), len(am)))
            if am else "-"))
    w("")


PASSES = (("rs_pass", "Resample pass (079's definition: production Z, 073 rs A, rs C with held-out contrast donors, "
                      "necessity)"),
          ("rule", "Rule pass, post-Top-K (the headline rule)"),
          ("pre", "Pre-activation pass (protocol 12: Z and necessity on the 0 baseline)"))


def main():
    data, excluded, rows = load95()
    p79 = load79()
    L = ["# 095: resample appendix under protocol v2 (%s, paired; 079 v1 for reference)\n" % " vs ".join(ARMS)]
    w = L.append
    w("Arms: **v2** = 093's production v2 circuits (reused); **v2R** = Z + A + R_C (R in place of C, 092 t3); "
      "**v2CR** = Z + A + C + R_C (R added, C kept: 079's shape without its silent-position term). R = donor resample, "
      "48 training contrast donors redrawn per forward, sparsity-preserving donor fill, weight 0.25.\n")
    w("Targets complete and non-vacuous in every arm (%s): **%d** of %d in rows (missing an arm %d, rejected %d, "
      "vacuous %d). Contexts: the 15k run's as-run records (DAN-157 duplicate leakage in the held-out sets: absolute "
      "rates are on leaky held-out sets; the comparison is paired, all arms on the same records).\n" % (
          ", ".join(ARMS), len(data), len({s for s, _ in rows}), len(excluded["missing"]), len(excluded["skip"]),
          len(excluded["vacuous"])))

    w("## 1. Pass rates, all arms\n")
    for field, label in PASSES:
        w("**%s**\n" % label)
        rates_table(w, data, field, ARMS)
    w("## 1b. Paired differences (McNemar exact, paired bootstrap 95% CI)\n")
    for base, new in PAIRS:
        w("### %s -> %s\n" % (base, new))
        for field, label in PASSES:
            w("**%s**\n" % label.split(" (")[0])
            pass_table(w, data, field, base, new)

    w("## 2. Medians (%s)\n" % " / ".join(ARMS))
    w("Z, A, C, nec: production scorer, post-Top-K read; rs A / C / R: 073 resample (held-out contrast donors for C); "
      "r all-pos (C): 074's all-position correlation under the sparsity-preserving contrast-mean fill.\n")
    median_table(w, data, ARMS)
    w("Pre-activation read medians (Z, necessity on the 0 baseline; A, C as scored)\n")
    median_table(w, data, ARMS, keys=(("Z_pre", "Zp"), ("A_pre", "Ap"), ("C_pre", "Cp"), ("nec_pre", "necp")))

    w("## 3. Cost: circuit size and fit time\n")
    w("v2 fit times are 093's (same GPU, 2026-10-25/26); v2R and v2CR fit times are this experiment's.\n")
    cost_table(w, data, ARMS)

    w("## 4. Pre-activation vs post-Top-K split (protocol 12)\n")
    for arm in ARMS:
        w("**%s**\n" % arm)
        split_table(w, data, arm)

    w("## 5. Reference: 079 v1 confirmation (W0 -> RS0.25_S0.25), recomputed from 079 out_confirm/rows.jsonl\n")
    w("v1 production config (W0, refitted) vs v1 + R (mixed contrast + random training donors, no donor fill) + "
      "silent-position term, both at weight 0.25, C kept; 99 targets, 11 per band x kind; the same resample scorer and "
      "pass definitions.\n")
    if p79:
        for field, label in PASSES[:2]:
            w("**%s**\n" % label.split(" (")[0])
            pass_table(w, p79, field, "W0", "RS")
        median_table(w, p79, ("W0", "RS"))
        cost_table(w, p79, ("W0", "RS"))
    else:
        w("(079 out_confirm/rows.jsonl not found)\n")

    w("## 6. Checks\n")
    v2rows = [r for (s, a), r in rows.items() if a == "v2" and "skip" not in r]
    d = [r["max_diff_vs_093"] for r in v2rows if r.get("max_diff_vs_093") is not None]
    agree = sum(r.get("pass_rule") == r.get("pass_rule_093") for r in v2rows)
    w("- v2 re-scored here vs 093's rows (same circuits): max |diff| over Z / A / C / necessity (both reads) median "
      "%s, max %s; rule pass agrees on %d / %d." % (f2(med(d), 4), f2(max(d), 4) if d else "-", agree, len(v2rows)))
    fits = {}
    for r in load_jsonl(OUT95 / "fits.jsonl"):
        if "error" not in r:
            fits[(r["seed"], r["arm"])] = r
    for arm in [a for a in ARMS if a != "v2"]:
        fl = [r for (s, a), r in fits.items() if a == arm]
        tn = {}
        for r in fl:
            k = ", ".join(r.get("term_names") or ())
            tn[k] = tn.get(k, 0) + 1
        w("- %s fits: %d; training term names %s; donor sparsity-preserving fill calls median %s (min %s); "
          "A / C sparse fill calls median %s; train_means_source %s; donors per target %s; peak memory median %s MB."
          % (arm, len(fl), tn, f2(med([r.get("donor_sp_calls") for r in fl]), 0),
             min((r.get("donor_sp_calls") or 0) for r in fl) if fl else "-",
             f2(med([r.get("sparse_fill_calls") for r in fl]), 0),
             sorted({str(r.get("train_means_source")) for r in fl}), sorted({r.get("n_donors") for r in fl}),
             f2(med([r.get("fit_peak_mb") for r in fl]), 0)))
    for c in load_jsonl(OUT95 / "check.jsonl"):
        w("- CHECK %s (rs2_engine refit of 093's v2 circuit %s, %s steps; off = every switch off, keepC_w0 = v2CR's "
          "switches with R weight 0): identical %s, Jaccard %s, n %s vs %s." % (
              c.get("mode", "off"), c["key"], c["steps"], c["identical"], f2(c["jaccard"]), c["n"], c["n_ref"]))
    w("")
    txt = "\n".join(L)
    (OUT95 / "tables.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
