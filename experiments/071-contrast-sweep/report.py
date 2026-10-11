"""DAN-64 report: the 4 x 3 train-source x eval-source table (trained on close / random / distant / fused, scored on
close / random / distant, all post-Top-K silent) at matched size, by depth band and overall; circuit sizes;
contamination per source (fused per sub-mode too); random baselines; set integrity; timings; and one paired line,
fused - random (the variety hypothesis). Descriptive only: no decision rule, no winner (README section 4). No GPU; safe
to run at any time on partial results (it uses the targets complete for every arm, lambda and eval set).

  OUT=experiments/071-contrast-sweep/out PYTHONPATH=src .venv/bin/python experiments/071-contrast-sweep/report.py
  env: OUT  B (bootstrap resamples, 2000)  SMOKE=1 (the smoke checks of README section 7 instead of the report)
       SRC_OUT (062 out_full/out: production eval rows, for the smoke's reproduction check)
Writes OUT/report.md (or OUT/smoke.md) and prints it.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import logic as L  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out")))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out")))
B = int(os.environ.get("B", 2000))
SMOKE = os.environ.get("SMOKE") == "1"
LINES = []
SHORT = {"close_silent": "close", "random_silent": "random", "distant_silent": "distant", "fused_silent": "fused"}


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def table(df, fmt="%.3f"):
    say("```")
    say(df.to_string(float_format=lambda x: fmt % x))
    say("```")


def load(kind, keys):
    rows = []
    for f in sorted(glob.glob(str(OUT / ("%s.shard*.jsonl" % kind)))):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                pass
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    if keys:
        d = d.drop_duplicates(keys, keep="last")
    return d


def ok(d):
    for c in ("error", "skip"):
        if c in d:
            d = d[d[c].isna()]
    return d


def with_pass(d):
    d = d.copy()
    d["pass"] = [L.passes(r) for r in d.to_dict("records")]
    d["layer"] = d.seed.str.split(".").str[0].astype(int)
    d["band"] = d.layer.map(L.band)
    return d


def usable(sets):
    s = sets.copy()
    s["unusable"] = s.get("unusable", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    return s, s[~s.unusable]


def ovl_cols(d):
    return [c for c in d.columns if c.startswith("ovl_")]


def ho_cols(d):
    return [c for c in d.columns if c.startswith("ho_in_train_")]


# ============================================================================================= sections
def integrity(sets, ev, rn, status):
    say("## Set integrity")
    s, use = usable(sets)
    oc = ovl_cols(use)
    tot = int(use[oc].fillna(0).sum().sum()) if oc else None
    say("- sets: %s; unusable: %s" % (dict(s.groupby("set").size()), dict(s.groupby("set").unusable.sum())))
    say("- cross-set overlap (content and corpus id, train + held-out + reserve), summed over targets and pairs: %s"
        % tot)
    hc = ho_cols(use)
    if hc:
        m = use.groupby("set")[hc].sum().astype(int)
        m.index.name = "eval set"
        say("- held-out contexts of the eval set (row) that are training contexts of the arm (column), summed over "
            "targets. Only close/close may be nonzero: the production set's own exact duplicates (used as is)")
        table(m.rename(columns=lambda c: c.replace("ho_in_train_", "train:")), "%d")
    cs = use[use.set == L.REFERENCE]
    if len(cs) and "ho_in_train_%s" % L.REFERENCE in cs:
        k = cs[cs["ho_in_train_%s" % L.REFERENCE].fillna(0) > 0]
        say("- production close sets with a held-out context duplicated in their training contexts: %d of %d targets%s"
            % (len(k), len(cs), (" (%s)" % ", ".join(k.seed)) if len(k) else ""))
    if len(cs) and "v1_match" in cs:
        say("- close_silent re-selection vs the production set (Jaccard): mean %.4f, min %.4f" % (
            cs.v1_match.mean(), cs.v1_match.min()))
    e = ev[ev.get("error").notna()] if "error" in ev else ev.iloc[:0]
    r = rn[rn.get("error").notna()] if len(rn) and "error" in rn else rn.iloc[:0]
    st = status[status.get("error").notna()] if len(status) and "error" in status else status.iloc[:0]
    say("- errors: %d eval rows, %d random rows, %d targets (status)" % (len(e), len(r), len(st)))
    if len(ev) and "n_ho_trained" in ev:
        x = ok(ev)
        bad = x[(x.n_ho_trained.fillna(0) > 0) & ~((x.arm == L.REFERENCE) & (x["eval"] == L.REFERENCE))]
        say("- eval rows with a held-out contrast context the arm trained on (must be 0): %d" % len(bad))
    say()


def contamination(sets):
    say("## Contamination per contrast source")
    say("frac_fire: share of contexts where the target fires post-Top-K at some position (0 by construction for the "
        "silent filter). pre_rel: max relu(w.x + b) over positions / the activating reference (median over "
        "strongest-train of the same). sim: max cosine to the top-context store (the close ranking's score). "
        "act_overlap: contexts (incl. reserve) that are also activating contexts.")
    _, use = usable(sets)
    agg = dict(targets=("seed", "nunique"), n=("n", "median"), fire=("frac_fire", "mean"),
               fire_train=("frac_fire_train", "mean"), fire_held=("frac_fire_held", "mean"),
               pre_rel=("pre_rel_med", "median"), pre_rel_p90=("pre_rel_p90", "median"),
               pre_rel_max=("pre_rel_max", "median"), sim=("sim_med", "median"), sim_min=("sim_min", "median"),
               act_overlap=("n_activating_overlap", "mean"), reserve=("n_reserve", "median"),
               t_select=("t_select", "median"))
    g = use.groupby("set").agg(**{k: v for k, v in agg.items() if v[0] in use})
    table(g.reindex([x for x in L.BUILD_ORDER if x in g.index]))
    say("By depth band:")
    pv = use.pivot_table(index="set", columns="band", values=["frac_fire", "pre_rel_med", "sim_med"],
                         aggfunc={"frac_fire": "mean", "pre_rel_med": "median", "sim_med": "median"})
    table(pv.reindex([x for x in L.BUILD_ORDER if x in pv.index]))
    f = use[use.set == L.FUSED]
    if len(f) and "part_n_close" in f:
        say("%s per sub-mode (contexts in the set / train / held-out / reserve: median; fire: mean; pre_rel, sim: "
            "median over targets of the per-target median):" % L.FUSED)
        g = {}
        for p in L.FUSED_PARTS:
            col = lambda w: f.get("%s_%s" % (w, p), pd.Series(dtype=float))  # noqa: E731
            g[p] = dict(n=col("part_n").median(), train=col("part_train").median(), held=col("part_held").median(),
                        reserve=col("part_reserve").median(), fire=col("part_fire").mean(),
                        pre_rel=col("part_pre_rel_med").median(), sim=col("part_sim_med").median())
        table(pd.DataFrame(g).T)
    say()


def arrays(ev):
    """Pass indicators, sizes and the reported scores as [target, arm, lambda, eval] arrays over the targets complete
    for every arm x lambda x eval set (vacuous targets excluded)."""
    ev = with_pass(ok(ev))
    vac = set(ev.loc[ev["vacuous_tk"].fillna(False).astype(bool), "seed"]) if "vacuous_tk" in ev else set()
    ev = ev[~ev.seed.isin(vac)]
    arms = [a for a in L.ARMS if a in set(ev.arm)]
    lams = sorted(set(ev.lam_value))
    evals = [e for e in L.EVAL_SETS if e in set(ev["eval"])]
    full = ev.groupby("seed").size()
    need = len(arms) * len(lams) * len(evals)
    common = sorted(full[full == need].index)
    ev = ev[ev.seed.isin(common)]
    ti = {t: i for i, t in enumerate(common)}
    ai, li, ei = ({a: i for i, a in enumerate(arms)}, {l: i for i, l in enumerate(lams)},
                  {e: i for i, e in enumerate(evals)})
    shape = (len(common), len(arms), len(lams), len(evals))
    P = np.zeros(shape)
    N = np.zeros(shape[:3])
    X = {m: np.full(shape, np.nan) for m in L.METRICS}
    num = lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)  # noqa: E731
    for r in ev.to_dict("records"):
        t, a, l, e = ti[r["seed"]], ai[r["arm"]], li[r["lam_value"]], ei[r["eval"]]
        P[t, a, l, e] = float(r["pass"])
        N[t, a, l] = r["n_nodes"]
        for m, f in L.METRICS.items():
            X[m][t, a, l, e] = num(r.get(f))
    return dict(ev=ev, vac=vac, n_rows=len(full) + len(vac), common=common, arms=arms, lams=lams, evals=evals,
                P=P, N=N, X=X, ai=ai, li=li)


def matrix(D):
    say("## Train source x eval source")
    arms, lams, evals, common, P, N, X = (D[k] for k in ("arms", "lams", "evals", "common", "P", "N", "X"))
    say("grid: %d arms x %d lambdas x %d eval sets; %d targets complete (of %d with rows; %d vacuous excluded)"
        % (len(arms), len(lams), len(evals), len(common), D["n_rows"], len(D["vac"])))
    if L.REFERENCE not in arms or not any(abs(l - L.REF_LAM) < 1e-12 for l in lams) or not common:
        say("the size reference (%s at lambda %g) or complete targets are missing: no matched-size table"
            % (L.REFERENCE, L.REF_LAM))
        return
    ra, rl = D["ai"][L.REFERENCE], [i for l, i in D["li"].items() if abs(l - L.REF_LAM) < 1e-12][0]
    allidx = np.arange(len(common))
    layer = np.array([int(t.split(".")[0]) for t in common])
    scopes = [("overall", allidx)] + [(b, allidx[np.array([L.band(x) == b for x in layer])]) for b in L.BANDS]
    lcols = ["%g" % l for l in lams]

    say("### Circuit size (median members) per arm and lambda")
    sz = {}
    for name, idx in scopes:
        if len(idx):
            for a in arms:
                sz[(name, a)] = np.median(N[idx, D["ai"][a], :], axis=0)
    table(pd.DataFrame(sz, index=lcols).T, "%.0f")

    say("### Pass rate per lambda (not size-matched), train x eval, overall")
    raw = {(a, SHORT[e]): P[:, D["ai"][a], :, j].mean(axis=0) for a in arms for j, e in enumerate(evals)}
    table(pd.DataFrame(raw, index=lcols).T)

    say("### At matched size")
    say("Each arm's value at its three lambdas is interpolated in log10(median size) to n* = the median size of %s at "
        "lambda %g within the scope. pass = protocol v1 (Z, A, C in [0.8, 1.5], necessity >= 0.9) with the eval set's "
        "own C; Z / A / C / nec / ind = medians over targets (ind = sufficiency to induce, phi_cf_alpha_blind_tk, on "
        "the eval set's 16 held-out contexts). Z, A and nec do not depend on the eval set (up to GPU numerics). "
        "* = n* outside the arm's size range (endpoint used)." % (L.REFERENCE, L.REF_LAM))
    for name, idx in scopes:
        if not len(idx):
            continue
        n_star = float(np.median(N[idx, ra, rl]))
        pr, ex = L.matched_cells(P, N, idx, n_star, "mean")
        med = {m: L.matched_cells(X[m], N, idx, n_star, "median")[0] for m in L.METRICS}
        say("#### %s (%d targets, n* = %.0f)" % (name, len(idx), n_star))
        pv = pd.DataFrame(pr, index=["%s%s" % (a, "*" if ex[i] else "") for i, a in enumerate(arms)],
                          columns=["eval:" + SHORT[e] for e in evals])
        pv["mean"] = pv.mean(axis=1)
        say("pass rate, train (rows) x eval (columns):")
        table(pv)
        rows = {}
        for i, a in enumerate(arms):
            for j, e in enumerate(evals):
                rows[(SHORT[a], SHORT[e])] = dict(pass_=pr[i, j], **{m: med[m][i, j] for m in L.METRICS})
        t = pd.DataFrame(rows).T.rename(columns={"pass_": "pass"})
        t.index.names = ["train", "eval"]
        table(t)
    bootstrap(D, scopes, ra, rl)


def bootstrap(D, scopes, ra, rl):
    """Paired bootstrap over targets of the arm differences in matched pass rate (descriptive intervals)."""
    arms, evals, P, N = D["arms"], D["evals"], D["P"], D["N"]
    pairs = [(a, L.REFERENCE) for a in arms if a != L.REFERENCE]
    if {"random_silent", "distant_silent"} <= set(arms):
        pairs.append(("random_silent", "distant_silent"))
    hyp = (L.FUSED, "random_silent")                      # the variety hypothesis: fused vs random
    if set(hyp) <= set(arms):
        pairs.append(hyp)
    if not pairs:
        return
    hyp_line = None
    say("### Arm differences in matched pass rate: estimate [95%% paired-bootstrap interval] (%d resamples over "
        "targets, n* and sizes recomputed per resample; descriptive)" % B)
    rng = np.random.default_rng(0)
    ai = D["ai"]
    for name, idx in scopes:
        if len(idx) < 2:
            continue
        pt = L.matched_cells(P, N, idx, float(np.median(N[idx, ra, rl])), "mean")[0]
        R = np.zeros((B,) + pt.shape)
        for bi in range(B):
            s = idx[rng.integers(0, len(idx), len(idx))]
            R[bi] = L.matched_cells(P, N, s, float(np.median(N[s, ra, rl])), "mean")[0]
        out = {}
        for a, b in pairs:
            cell = {}
            for j, e in list(enumerate(evals)) + [(None, "mean")]:
                est = (pt[ai[a]] - pt[ai[b]]).mean() if j is None else pt[ai[a], j] - pt[ai[b], j]
                d = (R[:, ai[a]] - R[:, ai[b]]).mean(axis=1) if j is None else R[:, ai[a], j] - R[:, ai[b], j]
                d = d[~np.isnan(d)]
                lo, hi = np.percentile(d, [2.5, 97.5]) if len(d) else (np.nan, np.nan)
                cell["eval:" + SHORT.get(e, e)] = "%+.3f [%+.3f, %+.3f]" % (est, lo, hi)
            out["%s - %s" % (SHORT[a], SHORT[b])] = cell
            if (a, b) == hyp and name == "overall":
                hyp_line = (len(idx), cell)
        say("#### %s (%d targets)" % (name, len(idx)))
        table(pd.DataFrame(out).T)
    if hyp_line:
        n_t, cell = hyp_line
        say("### Fused vs random (variety hypothesis: fused spans close, mid-range and distant contexts, so 'variety "
            "helps' predicts fused >= random)")
        say("- fused - random, matched pass rate averaged over the three eval sets, paired over %d targets: %s "
            "(per eval set: %s; per band in the table above; descriptive interval)" % (
                n_t, cell["eval:mean"], ", ".join("%s %s" % (k[5:], v) for k, v in cell.items() if k != "eval:mean")))
    say()


def randoms(rn, D):
    rn = with_pass(ok(rn))
    if rn.empty:
        return
    ev = D["ev"]
    say("## Random circuits with fitted coefficients (site-matched to each arm's lambda %g circuit)" % L.REF_LAM)
    ref = ev[(abs(ev.lam_value - L.REF_LAM) < 1e-12) & ev.seed.isin(set(rn.seed))]
    order = [x for x in L.ARMS]
    for lab, d in (("random (%d targets)" % rn.seed.nunique(), rn), ("fitted WCM at lambda %g, same targets" % L.REF_LAM, ref)):
        if d.empty:
            continue
        say(lab + ": pass rate and median C / necessity / induce, train (rows) x eval (columns)")
        vals = {"pass": d.pivot_table(index="arm", columns="eval", values="pass", aggfunc="mean")}
        for m in ("C", "nec", "ind"):
            f = L.METRICS[m]
            if f in d:
                vals[m] = d.pivot_table(index="arm", columns="eval", values=f, aggfunc="median")
        t = pd.concat(vals, axis=1)
        table(t.reindex([x for x in order if x in t.index]))
    say("members drawn / reference: median %.2f; short sites on %d rows" % (
        (rn.n_drawn / rn.n_ref).median(), int((rn.short_sites > 0).sum())))
    say()


def timings(ev, sets):
    say("## Timings")
    e = ok(ev)
    if len(e):
        f = e.drop_duplicates(["seed", "arm", "lam"])
        fresh = f[f.get("fresh_fit", True) == True]  # noqa: E712
        say("fresh fits %d: t_fit median %.1f s (by band: %s); scorer run median %.2f s" % (
            len(fresh), fresh.t_fit.median(),
            ", ".join("%s %.1f" % (b, fresh[fresh.seed.str.split('.').str[0].astype(int).map(L.band) == b].t_fit.median())
                      for b in L.BANDS), e.t_score.median()))
    if len(sets):
        say("set construction median %.1f s per source (incl. the reserve)" % sets.t_select.median())
    say()


# ============================================================================================= smoke
def smoke(ev, sets, rn, status):
    say("# DAN-64 smoke checks (%s)" % OUT)
    bad = []

    def check(name, cond, detail=""):
        say("- [%s] %s %s" % ("ok" if cond else "FAIL", name, detail))
        if not cond:
            bad.append(name)

    s, use = usable(sets)
    say("sets built: %s; unusable: %s" % (dict(s.groupby("set").size()), dict(s.groupby("set").unusable.sum())))
    check("every set usable on every target", not s.unusable.any())
    for name in [x for x in L.BUILD_ORDER if x in set(use.set)]:
        u = use[use.set == name]
        say("  %-15s n %s  fire %.3f (train %.3f, held %.3f)  pre_rel med %.3f p90 %.3f  sim med %s  act_overlap %d"
            "  reserve %s  dropped %s" % (
                name, sorted(set(u.n)), u.frac_fire.mean(), u.frac_fire_train.mean(), u.frac_fire_held.mean(),
                u.pre_rel_med.median(), u.pre_rel_p90.median(),
                "%.3f" % u.sim_med.median() if "sim_med" in u and u.sim_med.notna().any() else "-",
                int(u.n_activating_overlap.sum()), sorted(set(u.n_reserve)),
                [(m or {}).get("dropped") for m in u.meta][:3]))
        check("%s is silent post-Top-K" % name, u.frac_fire.max() <= 0.02,
              "(max per-target fire rate %.3f; a context or two can flip with the forward batch size)"
              % u.frac_fire.max())
        check("%s excludes the activating contexts" % name, int(u.n_activating_overlap.sum()) == 0)
        check("%s has a reserve" % name, (u.n_reserve > 0).all(), "(min %d)" % u.n_reserve.min())
    oc = ovl_cols(use)
    if oc:
        tot = use[oc].fillna(0).sum()
        check("the %d sets are pairwise disjoint (content and id, incl. reserves)" % use.set.nunique(),
              int(tot.sum()) == 0,
              "(%s)" % ", ".join("%s %d" % (c[4:], v) for c, v in tot.items() if v) if tot.sum() else "")
    else:
        check("cross-set overlap counts logged", False)
    hc = ho_cols(use)
    if hc:
        h = use[hc].fillna(0).sum()
        check("no held-out context of any set is a training context of any arm (sets)", int(h.sum()) == 0,
              "(%s)" % dict(h[h > 0]) if h.sum() else "")
    sims = {n: use[use.set == n].set_index("seed").sim_med for n in L.EVAL_SETS if n in set(use.set)}
    if len(sims) == 3:
        c, r, d = (sims[n].median() for n in L.EVAL_SETS)
        j = pd.concat(sims, axis=1).dropna()
        share = float(((j.close_silent > j.random_silent) & (j.random_silent > j.distant_silent)).mean())
        check("similarity to the top contexts: close > random > distant", c > r > d,
              "(medians %.3f > %.3f > %.3f; ordered on %.0f%% of targets)" % (c, r, d, 100 * share))
    cs = use[use.set == L.REFERENCE]
    if len(cs) and "v1_match" in cs:
        # the arm trains on the production set itself; this checks the selector path. A context at the Top-K border
        # can flip on other hardware (an old CPU dry run: 63 of 64 shared), so a near-match passes.
        check("close_silent re-selection reproduces the production set",
              cs.v1_match.mean() >= 0.97 and cs.v1_match.min() >= 0.9,
              "(Jaccard mean %.4f, min %.4f)" % (cs.v1_match.mean(), cs.v1_match.min()))
    fs = use[use.set == L.FUSED]
    if not len(fs):
        check("%s built" % L.FUSED, False)
    else:
        cols = ["ovl_%s_%s" % (w, e) for w in ("content", "id") for e in L.EVAL_SETS]
        check("%s is disjoint from the three eval sets (content and id, incl. reserves)" % L.FUSED,
              all(c in fs for c in cols) and int(fs[cols].fillna(0).sum().sum()) == 0)
        hc = "ho_in_train_%s" % L.FUSED
        ho_e = use[use.set.isin(L.EVAL_SETS)]
        check("%s trains on no held-out context of an eval set" % L.FUSED,
              hc in ho_e and ho_e[hc].notna().all() and int(ho_e[hc].sum()) == 0,
              "(%s logged on %d of %d eval-set rows)" % (hc, int(ho_e[hc].notna().sum()) if hc in ho_e else 0,
                                                         len(ho_e)))
        qm, qr = L.fused_quotas(L.CONTRAST_COUNT), L.fused_quotas(L.RESERVE)
        pc = lambda w, p: fs.get("part_%s_%s" % (w, p), pd.Series(np.nan, index=fs.index))  # noqa: E731
        want = L.part_counts([p for p, _ in L.fused_split_merge({p: [0] * v for p, v in qm.items()})],
                             L.n_train_of(L.CONTRAST_COUNT))           # the exact per-part split (train 16/16/16)
        thirds = all((pc("n", p) == qm[p]).all() and (pc("reserve", p) == qr[p]).all()
                     and (pc("train", p) == want[p]["train"]).all() and (pc("held", p) == want[p]["held"]).all()
                     for p in L.FUSED_PARTS)
        say("  %s parts (set / train / held / reserve, per target): %s" % (L.FUSED, "; ".join(
            "%s %s / %s / %s / %s" % (p, sorted(set(pc("n", p))), sorted(set(pc("train", p))),
                                      sorted(set(pc("held", p))), sorted(set(pc("reserve", p))))
            for p in L.FUSED_PARTS)))
        check("%s has its sub-mode thirds (set %s, train %s, held-out %s, reserve %s)" % (
            L.FUSED, *("/".join(str(d[p]) for p in L.FUSED_PARTS) for d in (
                qm, {p: want[p]["train"] for p in want}, {p: want[p]["held"] for p in want}, qr))), thirds)
        ps = {p: pc("sim_med", p).median() for p in L.FUSED_PARTS}
        check("%s parts' similarity: close > random > distant" % L.FUSED,
              ps["close"] > ps["random"] > ps["distant"],
              "(medians %s)" % ", ".join("%s %.3f" % (p, ps[p]) for p in ("close", "random", "distant")))

    st_err = status[status.error.notna()] if len(status) and "error" in status else status.iloc[:0]
    check("no target errors (status)", len(st_err) == 0, "" if not len(st_err) else str(st_err.error.iloc[0])[:200])
    errs = ev[ev.error.notna()] if "error" in ev else ev.iloc[:0]
    check("no errors in eval rows", len(errs) == 0, "" if not len(errs) else str(errs.error.iloc[0])[:200])
    e = ok(ev)
    if len(e):
        g = e.groupby(["arm", "eval"]).size()
        say("eval rows per (arm, eval): %s" % dict(g))
        n_t = status.seed.nunique() if len(status) else e.seed.nunique()
        check("every arm scored on every eval set",
              len(g) == len(L.ARMS) * len(L.EVAL_SETS) and (g == n_t * e.lam.nunique()).all(),
              "(%d targets x %d lambdas)" % (n_t, e.lam.nunique()))
        for ev_name in L.EVAL_SETS:
            x = e[e["eval"] == ev_name]
            if not len(x):
                continue
            check("eval %s: the scorer read 48 + 16 activating contexts" % ev_name,
                  ((x.n_train == 48) & (x.n_ho == 16)).all(),
                  "(n_train %s, n_ho %s)" % (sorted(set(x.n_train)), sorted(set(x.n_ho))))
            check("eval %s: the scorer read 48 + 16 contrast contexts" % ev_name,
                  ((x.n_neg_train == 48) & (x.n_neg_ho == 16)).all(),
                  "(n_neg_train %s, n_neg_ho %s)" % (sorted(set(x.n_neg_train)), sorted(set(x.n_neg_ho))))
        rows_ho = pd.concat([e[["n_ho_trained"]], ok(rn)[["n_ho_trained"]]]) if len(rn) and "n_ho_trained" in rn \
            else e[["n_ho_trained"]]
        check("n_ho_trained = 0 on every eval and random row", (rows_ho.n_ho_trained.fillna(0) == 0).all(),
              "(max %s)" % rows_ho.n_ho_trained.max())
        check("every upstream site ablated", e.all_sites_ablated.all())
        cross = e[e.arm != e["eval"]]
        check("cross-source rows share no training contexts", (cross.n_ctr_train_overlap == 0).all())
        fe = e[e.arm == L.FUSED]
        check("%s scored on all three eval sets" % L.FUSED,
              len(fe) and set(fe["eval"]) == set(L.EVAL_SETS) and (fe.groupby("eval").size() == n_t * e.lam.nunique()).all(),
              "(rows per eval set %s)" % dict(fe.groupby("eval").size()))
        fr = ok(rn)[ok(rn).arm == L.FUSED] if len(rn) and "arm" in rn else rn.iloc[:0]
        fh = pd.concat([fe[["n_ho_trained"]]] + ([fr[["n_ho_trained"]]] if "n_ho_trained" in fr else []))
        check("%s: n_ho_trained = 0 on its eval and random rows" % L.FUSED,
              len(fe) and (fh.n_ho_trained.fillna(0) == 0).all() and fe.n_ho_trained.notna().all(),
              "(%d rows, max %s)" % (len(fh), fh.n_ho_trained.max()))

        ref = e[(e.arm == L.REFERENCE) & (e["eval"] == L.REFERENCE) & (abs(e.lam_value - L.REF_LAM) < 1e-12)]
        prod = []
        for f in glob.glob(str(SRC_OUT / "main" / "eval.shard*.jsonl")):
            for line in open(f):
                if any(k in line for k in ref.seed):
                    r = json.loads(line)
                    if r.get("held") == "strong" and r.get("seed") in set(ref.seed) and "error" not in r:
                        prod.append(r)
        if len(ref) and prod:
            p = pd.DataFrame(prod).drop_duplicates("seed", keep="last").set_index("seed")
            r_ = ref.set_index("seed")
            cols = list(L.HEAD) + [L.NEC, L.IND]
            dif = (r_[cols].astype(float) - p.loc[r_.index, cols].astype(float)).abs()
            say("close_silent at lambda %g (production circuit, re-scored on close_silent) vs the production eval rows:"
                % L.REF_LAM)
            table(pd.DataFrame({"median_abs_diff": dif.median(), "max_abs_diff": dif.max()}), "%.4f")
            good = float(np.nanmax(dif.median().values)) <= 0.005 and float(np.nanmax(dif.values)) <= 0.05
            devs = set(status.device.dropna()) if len(status) and "device" in status else set()
            name = "re-scoring the production circuit reproduces production (median |diff| <= 0.005, max <= 0.05)"
            if devs and all("H100" in d for d in devs):
                check("[pod-smoke, H100] " + name, good)
            else:
                say("- [%s, not binding here] [pod-smoke, H100] %s: device %s. RTX vs H100 float noise cannot meet "
                    "it locally; it is checked in the pod smoke" % ("ok" if good else "miss", name, sorted(devs) or "?"))
            check("same circuit size as production", (r_.n_nodes == p.loc[r_.index, "n"]).all())
        f = e.drop_duplicates(["seed", "arm", "lam"])
        say("timings: fit median %.1f s (fresh fits: %d), scorer run median %.2f s, set build median %.1f s" % (
            f[f.fresh_fit == True].t_fit.median() if (f.fresh_fit == True).any() else float("nan"),  # noqa: E712
            int((f.fresh_fit == True).sum()), e.t_score.median(), sets.t_select.median()))  # noqa: E712
        say("pass (protocol v1) by arm x eval: %s" % dict(with_pass(e).groupby(["arm", "eval"])["pass"].mean().round(2)))
    if len(rn):
        errs = rn[rn.error.notna()] if "error" in rn else rn.iloc[:0]
        check("random baseline rows without errors", len(errs) == 0,
              "" if not len(errs) else str(errs.error.iloc[0])[:200])
        r = ok(rn)
        if len(r):
            check("random baseline scored for every arm on every eval set",
                  len(r.groupby(["arm", "eval"]).size()) == len(L.ARMS) * len(L.EVAL_SETS))
            say("random baseline: %d rows over %s; members drawn / reference %.2f; pass by arm x eval %s" % (
                len(r), sorted(set(r.arm)), float((r.n_drawn / r.n_ref).median()),
                dict(with_pass(r).groupby(["arm", "eval"])["pass"].mean().round(2))))
    else:
        check("random baseline rows present", False)
    say("\n%s" % ("ALL CHECKS OK" if not bad else "FAILED: %s" % ", ".join(bad)))


def main():
    ev = load("eval", ["seed", "arm", "lam", "eval"])
    sets = load("sets", ["seed", "set"])
    rn = load("random", ["seed", "arm", "eval"])
    status = load("status", ["seed"])
    if SMOKE:
        smoke(ev, sets, rn, status)
        (OUT / "smoke.md").write_text("\n".join(LINES) + "\n")
        return
    say("# DAN-64 contrast-context sweep (%s)" % OUT)
    say("Descriptive: four disjoint contrast sources as training arms (close, random, distant, fused), each scored on "
        "the three held-out eval sets (close, random, distant). No decision rule.")
    say()
    if len(sets):
        integrity(sets, ev, rn, status)
        contamination(sets)
    if len(ev):
        D = arrays(ev)
        matrix(D)
        if len(rn) and len(D["common"]):
            randoms(rn, D)
        timings(ev, sets)
    (OUT / "report.md").write_text("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
