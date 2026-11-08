"""112: the harder-null appendix detail (DAN-181, remaining part) from the 2026-11-07/08 production run.

Reads the 084 harder-null rows of the production run (TuringLLM T6: out_turingllm/harder_nulls; public:
out_<model>/harder_nulls in the S10 results archives), the stored null circuits next to them and the production
circuits (main/circuits), all under experiments/109-pod-run-analysis/out/archives (unpacked by 109's unpack.sh).

Rules (as 109 t4_robustness.nulls): 084 report.frame (good rows, last per seed / arm / draw / split), then the CURRENT
pass rule and vacuous set from 062 merge.py (merge.passes, floor 0.75; merge.vacuous), site / band from 109 lib109
(family bands). Scored targets = those whose production (wcm) held-out row is non-vacuous (TuringLLM 112, Pythia 126,
GPT-2 128); every arm is restricted to them. Training-split rows are judged with their own vacuous flag (reported).
Bootstrap: 4,000 resamples over targets, seed 20261108 (lib109.boot_ci).

Items (CSV in out/, one LaTeX table paper/tables/run2026_nulls_detail.tex, figure paper/figures/run2026_nulls_detail):
  1  pass (held-out strongest, draw 0) with 95% bootstrap interval and the exact one-sided 95% upper bound for 0 of n,
     per model x arm x depth band (and all); pass on the training split           -> out/nulls_detail_bands.csv
  2  median Z / A / C / necessity on held-out and on training contexts, same grid  -> out/nulls_detail_bands.csv
  3  share of coefficients moved: |alpha - alpha_inert| > 0.01 (084 MOVE_TOL), alpha from the stored circuits
     (084 harder_nulls.circuit_alphas); alpha_inert = the gradient-free AdamW trajectory under the run's v4 schedule
     (geometric, peak 0.05 x 3.15, floor 0.05, warm-up 0.1, wd 0.05, 400 steps) = 0.7901, not protocol v1's constant-lr
     0.7975 (both computed here)                                                    -> out/nulls_detail_alpha.csv
  4  production circuit's quantile among each target's null draws (targets with >= 2 draws; 084 report.quantile)
                                                                                    -> out/nulls_detail_quantiles.csv
  5  overlap of the draws with the production members: count, share of the draw, Jaccard = ov / (n_drawn + n_ref - ov)
                                                                                    -> out/nulls_detail_overlap.csv
  6  matching shortfall: per-draw exact-bin share and bin distance (084 table 8), the pooled by-bin shortfall (084
     table 9), the drawn set's contribution relative to the production members' (set.contrib_* / profile set_wcm), and
     the top-decile fill                                     -> out/nulls_detail_match.csv, out/nulls_detail_shortfall.csv
  7  necessity diagnostics (why contrib keeps necessity yet never passes)          -> out/nulls_detail_necessity.csv

  python -X utf8 experiments/112-null-detail/null_detail.py
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E109 = ROOT / "experiments" / "109-pod-run-analysis"
E084 = ROOT / "experiments" / "084-harder-nulls"
os.environ.setdefault("OUT109", str(HERE / "out"))          # lib109 never writes here, but keep 109/out untouched
sys.path.insert(0, str(E109))
sys.path.insert(0, str(E084))
import lib109 as L  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R84 = _load("report84", E084 / "report.py")
NC = R84.NC

OUT = HERE / "out"
TABLE = ROOT / "paper" / "tables" / "run2026_nulls_detail.tex"
FIGURE = ROOT / "paper" / "figures" / "run2026_nulls_detail.png"
MOVE_TOL = 0.01                                             # 084 harder_nulls MOVE_TOL default
NULL_ARMS = ["dan17", "anchor", "contrib"]
ARMS = ["wcm", "wcm_a1", "dan17", "dan17_a1", "anchor", "anchor_a1", "contrib", "contrib_a1"]
ARM_TEX = {"wcm": "production (WCM)", "dan17": "weak (protocol v1)", "anchor": "anchor-active",
           "contrib": "contribution-matched"}
MET = [("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", L.NEC)]


# ------------------------------------------------------------------------------------------------- inert alpha
def lr_schedule(lr, steps, schedule, lr_min_frac=0.05, warmup_frac=0.0):
    """learned_mask.lr_at, copied (src/circuit/instrument/learned_mask.py ~L1630)."""
    lr_floor = lr * lr_min_frac
    n_warm = int(round(warmup_frac * steps))
    out = []
    for step in range(steps):
        if schedule == "constant":
            out.append(lr)
            continue
        if step < n_warm:
            out.append(lr_floor + (lr - lr_floor) * (step + 1) / n_warm)
            continue
        frac = (step - n_warm) / max(steps - n_warm - 1, 1)
        assert schedule == "geometric"
        out.append(lr * lr_min_frac ** frac)
    return out


def alpha_inert(lrs, wd):
    """softplus(softplus^-1(1) * prod(1 - lr_t wd)): AdamW's decoupled decay alone (nulls_core.alpha_inert,
    generalised to a schedule)."""
    psi = math.log(math.expm1(1.0))
    for lr in lrs:
        psi *= 1.0 - lr * wd
    return NC.softplus(psi)


INERT_V4 = alpha_inert(lr_schedule(0.05 * 3.15, 400, "geometric", 0.05, 0.1), 0.05)   # driver WCM_VERSIONS["v4"]
INERT_V1 = alpha_inert(lr_schedule(0.05, 400, "constant"), 0.05)                     # = 0.7975 (084 test)


# ------------------------------------------------------------------------------------------------- loading
def nulls_dirs(model):
    return L.find(model, "work", "harder_nulls") if model == "turingllm" else L.find(model, "out", "harder_nulls")


def load(model):
    rows = []
    for d in nulls_dirs(model):
        rows += R84.read_rows(d, "nulls")
    prof = []
    for d in nulls_dirs(model):
        prof += [r for r in R84.read_rows(d, "profile") if "error" not in r]
    df = R84.frame(rows)
    df["vacuous"] = L.vacuous(df).values
    df["pass"] = False
    ok = ~df.vacuous
    df.loc[ok, "pass"] = L.passes(df[ok]).values
    df["pass"] = df["pass"].astype(bool)
    df = L.add_site(df, model)
    df["fmin"] = df[L.HEAD].min(axis=1, skipna=False)
    w = df[(df.arm == "wcm") & (df.split == "held")]
    scored = set(w[~w.vacuous].seed)
    meta = dict(rows=len(rows), errors=sum("error" in r for r in rows), skips=sum("skip" in r for r in rows),
                targets_with_rows=int(df.seed.nunique()), wcm_targets=int(w.seed.nunique()),
                vacuous=int(w.vacuous.sum()), scored=len(scored))
    df = df[df.seed.isin(scored)].copy()
    prof = {r["seed"]: r for r in prof if r["seed"] in scored}
    return df, prof, meta


def exact_upper(n, conf=0.95):
    """One-sided exact (Clopper-Pearson) upper bound for 0 successes of n: 1 - (1 - conf)^(1/n)."""
    return 1 - (1 - conf) ** (1.0 / n) if n else float("nan")


# ------------------------------------------------------------------------------------------------- 1 + 2
def bands_table(model, df):
    out = []
    held = df[df.split == "held"]
    train = df[df.split == "train"]
    bands = ["all"] + [b for b, _, _ in L.bands_of(model)]
    for arm in ARMS:
        for b in bands:
            h = held[(held.arm == arm) & (held.draw == 0)]
            if b != "all":
                h = h[h.band == b]
            if h.empty:
                continue
            c = L.pass_cell(h["pass"])
            row = dict(model=L.LABEL[model], arm=arm, band=b, targets=len(h), pass_k=c["k"], pass_pct=c["pct"],
                       pass_lo=c["lo"], pass_hi=c["hi"], exact_upper95_pct=100 * exact_upper(len(h)) if c["k"] == 0
                       else np.nan, pass_all_draws_pct=100 * held[(held.arm == arm) & (
                           (held.band == b) if b != "all" else True)]["pass"].mean())
            for k, col in MET:
                row["%s_held" % k] = h[col].median()
            row["fmin_held"] = h["fmin"].median()
            row["nec_ge_0.9_held_pct"] = 100 * (h[L.NEC] >= L.NEC_MIN).mean()
            row["ZAC_in_band_held_pct"] = 100 * ((h[L.HEAD] >= L.BAND[0]) & (h[L.HEAD] <= L.BAND[1])).all(axis=1).mean()
            t = train[(train.arm == arm) & (train.draw == 0) & train.seed.isin(h.seed)]
            if len(t):
                tv = t[~t.vacuous]
                ct = L.pass_cell(tv["pass"])
                row.update(train_targets=len(t), train_vacuous=int(t.vacuous.sum()), pass_train_k=ct["k"],
                           pass_train_pct=ct["pct"], pass_train_lo=ct["lo"], pass_train_hi=ct["hi"])
                for k, col in MET:
                    row["%s_train" % k] = tv[col].median()
                row["fmin_train"] = tv["fmin"].median()
                row["Z_train_ge_0.75_pct"] = 100 * (tv[L.HEAD[0]] >= L.BAND[0]).mean()
                both = tv.set_index("seed")["pass"].reindex(h.seed).fillna(False).astype(bool).values
                row["fits_train_fails_held_pct"] = 100 * float(np.mean(both & ~h["pass"].values))
            out.append(row)
    return out


# ------------------------------------------------------------------------------------------------- 3 coefficients
def circuit_alphas(path):
    import torch
    c = torch.load(path, weights_only=False)
    if c is None:
        return None
    return np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values()
                     if n.metadata.get("role") != "seed"])


def circuit_path(model, arm, seed, draw):
    if arm == "wcm":
        for d in L.find(model, "out", "main/circuits"):
            p = d / ("%s.pt" % seed)
            if p.exists():
                return p
        return None
    for d in nulls_dirs(model):
        p = d / "circuits" / arm / ("%s.d%d.pt" % (seed, draw))
        if p.exists():
            return p
    return None


def alpha_table(model, df):
    out, per = [], []
    h = df[(df.split == "held") & (df.alpha == "fit")]
    for arm in ["wcm"] + NULL_ARMS:
        d = h[h.arm == arm]
        if d.empty:
            continue
        for r in d.itertuples():
            p = circuit_path(model, arm, r.seed, int(r.draw))
            a = circuit_alphas(p) if p is not None else None
            if a is None or not len(a):
                per.append(dict(model=L.LABEL[model], arm=arm, seed=r.seed, draw=int(r.draw), missing=True))
                continue
            per.append(dict(model=L.LABEL[model], arm=arm, seed=r.seed, draw=int(r.draw), n=len(a),
                            alpha_median=float(np.median(a)), moved=float(np.mean(np.abs(a - INERT_V4) > MOVE_TOL)),
                            up=float(np.mean(a > INERT_V4 + MOVE_TOL)), down=float(np.mean(a < INERT_V4 - MOVE_TOL)),
                            at_inert=float(np.mean(np.abs(a - INERT_V4) <= 1e-4)),
                            moved_v1ref=float(np.mean(np.abs(a - INERT_V1) > MOVE_TOL)),
                            stored_moved=getattr(r, "alpha_moved", np.nan), row_alpha_median=r.alpha_median))
    p = pd.DataFrame(per)
    for arm in ["wcm"] + NULL_ARMS:
        x = p[(p.arm == arm)] if len(p) else p
        if x.empty:
            continue
        miss = int(x.get("missing", pd.Series(dtype=bool)).fillna(False).sum())
        x = x[x.get("missing", pd.Series(False, index=x.index)).fillna(False) != True]  # noqa: E712
        if x.empty:
            continue
        row = dict(model=L.LABEL[model], arm=arm, circuits=len(x), missing=miss, alpha_inert=INERT_V4,
                   moved_median=x.moved.median(), moved_p25=x.moved.quantile(.25), moved_p75=x.moved.quantile(.75),
                   up_median=x.up.median(), down_median=x.down.median(), at_inert_median=x.at_inert.median(),
                   alpha_median=x.alpha_median.median(), all_moved_lt_50pct=100 * (x.moved < 0.5).mean())
        sm = pd.to_numeric(x.stored_moved, errors="coerce")
        if sm.notna().any():                                  # dan17 rows: 084 alpha_stats with the modal fallback
            row["stored_vs_recomputed_maxabs"] = float((sm - x.moved).abs().max())
        row["row_vs_circuit_alpha_median_maxabs"] = float((pd.to_numeric(x.row_alpha_median) - x.alpha_median).abs().max())
        out.append(row)
    return out, p


# ------------------------------------------------------------------------------------------------- 4 quantiles
def quantile_table(model, df):
    out = []
    h = df[df.split == "held"]
    w = h[h.arm == "wcm"].drop_duplicates("seed").set_index("seed")
    for arm in [a for a in ARMS if a not in ("wcm", "wcm_a1")]:
        d = h[h.arm == arm]
        g = {k: v for k, v in d.groupby("seed") if len(v) >= 2 and k in w.index}
        if not g:
            continue
        row = dict(model=L.LABEL[model], arm=arm, targets=len(g), draws_median=float(np.median([len(v) for v in g.values()])))
        for m, c in MET + [("fmin", "fmin")]:
            q = [R84.quantile(w.loc[k, c], v[c].tolist()) for k, v in g.items()]
            q = [x for x in q if not math.isnan(x)]
            row["q_%s_median" % m] = float(np.median(q)) if q else np.nan
            row["q_%s_mean" % m] = float(np.mean(q)) if q else np.nan
            row["top_%s_pct" % m] = 100 * float(np.mean([x == 1.0 for x in q])) if q else np.nan
        row["wcm_pass_all_fail_pct"] = 100 * float(np.mean([bool(w.loc[k, "pass"]) and not v["pass"].any()
                                                            for k, v in g.items()]))
        row["wcm_pass_pct"] = 100 * float(np.mean([bool(w.loc[k, "pass"]) for k in g]))
        row["any_draw_pass_pct"] = 100 * float(np.mean([bool(v["pass"].any()) for v in g.values()]))
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------- 5 overlap
def overlap_table(model, df):
    out = []
    h = df[(df.split == "held") & (df.alpha == "fit") & df.arm.isin(["anchor", "contrib"])].copy()
    for c in ("overlap_wcm", "n_drawn", "n_ref", "overlap_frac"):
        h[c] = pd.to_numeric(h[c], errors="coerce")
    h["jaccard"] = h.overlap_wcm / (h.n_drawn + h.n_ref - h.overlap_wcm)
    for arm in ("anchor", "contrib"):
        x = h[h.arm == arm]
        if x.empty:
            continue
        for scope, y in (("all draws", x), ("draw 0", x[x.draw == 0])):
            out.append(dict(model=L.LABEL[model], arm=arm, scope=scope, draws=len(y),
                            overlap_median=y.overlap_wcm.median(), overlap_max=y.overlap_wcm.max(),
                            overlap_frac_median=y.overlap_frac.median(), overlap_frac_p75=y.overlap_frac.quantile(.75),
                            overlap_frac_max=y.overlap_frac.max(), jaccard_median=y.jaccard.median(),
                            jaccard_p25=y.jaccard.quantile(.25), jaccard_p75=y.jaccard.quantile(.75),
                            jaccard_max=y.jaccard.max(), zero_overlap_pct=100 * (y.overlap_wcm == 0).mean(),
                            n_ref_median=y.n_ref.median(), drawn_over_ref_median=(y.n_drawn / y.n_ref).median()))
    return out


# ------------------------------------------------------------------------------------------------- 6 match
def _set(s):
    return s if isinstance(s, dict) and s.get("n") else None


def match_tables(model, df, prof):
    out, sf = [], []
    h = df[(df.split == "held") & (df.alpha == "fit") & df.arm.isin(["anchor", "contrib"])].copy()
    pool = np.median([r["pool_total"] for r in prof.values()]) if prof else np.nan
    for arm in ("anchor", "contrib"):
        x = h[h.arm == arm]
        if x.empty:
            continue
        nd = pd.to_numeric(x.n_drawn, errors="coerce")
        exact = pd.to_numeric(x.n_exact, errors="coerce") / nd
        dist = pd.to_numeric(x.dist0, errors="coerce") / nd
        rc_med, rc_mean, top_fill, above_max, below_min, rlog = [], [], [], [], [], []
        for r in x.itertuples():
            s, p = _set(r.set), prof.get(r.seed)
            pw = _set(p.get("set_wcm")) if p else None
            if not s or not pw:
                continue
            if pw["contrib_median"] > 0:
                rc_med.append(s["contrib_median"] / pw["contrib_median"])
            if pw["contrib_mean"] > 0:
                rc_mean.append(s["contrib_mean"] / pw["contrib_mean"])
            rlog.append(s["log10_count_median"] - pw["log10_count_median"])
            cb, cw = s.get("cbin_hist"), pw.get("cbin_hist")      # bins -1..10 -> index 0..11
            if cb and cw and sum(cw[10:12]):
                top_fill.append(sum(cb[10:12]) / sum(cw[10:12]))  # drawn in the members' top decile (or above) / members there
                above_max.append(cb[11] / max(s["n"], 1))
                below_min.append(cb[0] / max(s["n"], 1))
        out.append(dict(model=L.LABEL[model], arm=arm, draws=len(x), pool_median=pool,
                        exact_median=exact.median(), exact_p25=exact.quantile(.25), exact_p75=exact.quantile(.75),
                        dist_per_member_median=dist.median(), dist_per_member_mean=dist.mean(),
                        short_sites_pct=100 * (pd.to_numeric(x.short_sites, errors="coerce") > 0).mean(),
                        contrib_median_ratio_median=np.median(rc_med) if rc_med else np.nan,
                        contrib_median_ratio_p25=np.percentile(rc_med, 25) if rc_med else np.nan,
                        contrib_median_ratio_p75=np.percentile(rc_med, 75) if rc_med else np.nan,
                        contrib_mean_ratio_median=np.median(rc_mean) if rc_mean else np.nan,
                        top_decile_fill_median=np.median(top_fill) if top_fill else np.nan,
                        above_member_max_pct=100 * np.mean(above_max) if above_max else np.nan,
                        below_member_min_pct=100 * np.mean(below_min) if below_min else np.nan,
                        log10_count_diff_median=np.median(rlog) if rlog else np.nan))
        acc = {}
        for bb in x["by_bin"]:
            if not isinstance(bb, dict):
                continue
            for b, v in bb.items():
                s_ = acc.setdefault(b, [0, 0, 0, 0])
                for j in range(4):
                    s_[j] += int(v[j])
        for b in sorted(acc, key=lambda z: (z == "all", int(z) if z != "all" else 0)):
            m, dr, ex, di = acc[b]
            sf.append(dict(model=L.LABEL[model], arm=arm, member_bin=b, members=m,
                           drawn_pct=100 * dr / m if m else np.nan, exact_pct=100 * ex / m if m else np.nan,
                           mean_bin_dist=di / dr if dr else np.nan))
    return out, sf


# ------------------------------------------------------------------------------------------------- 7 necessity
def necessity_table(model, df, prof):
    out = []
    h = df[df.split == "held"]
    for arm in ("wcm", "dan17", "anchor", "contrib"):
        x = h[(h.arm == arm) & (h.alpha == "fit")]
        a1 = h[(h.arm == arm + "_a1")]
        if x.empty:
            continue
        j = x.merge(a1[["seed", "draw", L.NEC]], on=["seed", "draw"], suffixes=("", "_a1"), how="left")
        nec = x[L.NEC]
        row = dict(model=L.LABEL[model], arm=arm, draws=len(x), nec_median=nec.median(),
                   nec_p25=nec.quantile(.25), nec_p75=nec.quantile(.75), nec_ge_0p9_pct=100 * (nec >= L.NEC_MIN).mean(),
                   nec_eq_1_pct=100 * (nec >= 0.999).mean(),
                   # the target's Top-K activation is exactly 0 with the set removed from the live model (censored)
                   target_off_pct=100 * (pd.to_numeric(x["phi_sup_blind_raw_tk"], errors="coerce") <= 1e-9).mean(),
                   **{"target_off_pct_%s" % b: 100 * (pd.to_numeric(x[x.band == b]["phi_sup_blind_raw_tk"],
                                                                    errors="coerce") <= 1e-9).mean()
                      for b, _, _ in L.bands_of(model)},
                   **{"nec_median_%s" % b: x[x.band == b][L.NEC].median() for b, _, _ in L.bands_of(model)},
                   nec_fit_eq_a1_pct=100 * float(np.mean(np.isclose(j[L.NEC], j[L.NEC + "_a1"], atol=1e-9))),
                   Z_ge_0p75_pct=100 * (x[L.HEAD[0]] >= L.BAND[0]).mean(),
                   A_in_band_pct=100 * ((x[L.HEAD[1]] >= L.BAND[0]) & (x[L.HEAD[1]] <= L.BAND[1])).mean(),
                   C_in_band_pct=100 * ((x[L.HEAD[2]] >= L.BAND[0]) & (x[L.HEAD[2]] <= L.BAND[1])).mean(),
                   nec_ok_but_ZAC_fail_pct=100 * ((nec >= L.NEC_MIN) & ~((x[L.HEAD] >= L.BAND[0]) &
                                                                       (x[L.HEAD] <= L.BAND[1])).all(axis=1)).mean())
        if arm == "contrib":                                  # does necessity track how strong the drawn set is?
            rc = []
            for r in x.itertuples():
                s, p = _set(r.set), prof.get(r.seed)
                pw = _set(p.get("set_wcm")) if p else None
                rc.append(s["contrib_mean"] / pw["contrib_mean"] if s and pw and pw["contrib_mean"] > 0 else np.nan)
            rc = pd.Series(rc, index=x.index)
            ok = rc.notna() & nec.notna()
            row["spearman_nec_vs_contrib_ratio"] = float(pd.Series(nec[ok]).rank().corr(rc[ok].rank()))
            ex = pd.to_numeric(x.n_exact, errors="coerce") / pd.to_numeric(x.n_drawn, errors="coerce")
            ok2 = ex.notna() & nec.notna()
            row["spearman_nec_vs_exact"] = float(nec[ok2].rank().corr(ex[ok2].rank()))
            for k in L.KIND_ORDER:
                y = x[x.kind == k]
                if len(y):
                    row["nec_median_%s" % k] = y[L.NEC].median()
            # production necessity on the same targets, and contrib's nec relative to it
            w = h[(h.arm == "wcm")].set_index("seed")[L.NEC]
            row["wcm_nec_median_same_targets"] = w.reindex(x.seed).median()
        out.append(row)
    return out


# ------------------------------------------------------------------------------------------------- outputs
def f2(x):
    return L.DASH if x is None or (isinstance(x, float) and math.isnan(x)) else "%.2f" % x


def write_tex(bands, alpha, quant, overlap, match, metas):
    B = pd.DataFrame(bands)
    A = pd.DataFrame(alpha)
    Q = pd.DataFrame(quant)
    O = pd.DataFrame(overlap)
    Mt = pd.DataFrame(match)
    cols = ["Model", "Circuit", "Targets", r"Pass held / train (\%)", "Held Z / A / C", "Held nec.",
            "Train Z / A / C", "Train nec.", r"$\alpha$ moved (\%)", r"WCM $>$ both draws (\%)",
            "Jaccard w/ prod.", r"Exact bin (\%)"]
    lines = ["% generated by experiments/112-null-detail/null_detail.py; do not edit by hand",
             r"\begin{tabular}{llrlllllllll}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    first = True
    for model in L.MODELS:
        lab = L.LABEL[model]
        b = B[(B.model == lab) & (B.band == "all")]
        if b.empty:
            continue
        if not first:
            lines.append(r"\midrule")
        first = False
        shown = False
        for arm in ["wcm", "dan17", "anchor", "contrib"]:
            r = b[b.arm == arm]
            if r.empty:
                continue
            r = r.iloc[0]
            a = A[(A.model == lab) & (A.arm == arm)]
            q = Q[(Q.model == lab) & (Q.arm == arm)]
            o = O[(O.model == lab) & (O.arm == arm) & (O.scope == "all draws")]
            m = Mt[(Mt.model == lab) & (Mt.arm == arm)]
            cells = [lab if not shown else "", ARM_TEX[arm], L.tex_int(r.targets),
                     "%.1f / %.1f" % (r.pass_pct, r.pass_train_pct),
                     "%s / %s / %s" % (f2(r.Z_held), f2(r.A_held), f2(r.C_held)), f2(r.nec_held),
                     "%s / %s / %s" % (f2(r.Z_train), f2(r.A_train), f2(r.C_train)), f2(r.nec_train),
                     "%.0f (%.0f--%.0f)" % (100 * a.moved_median.iloc[0], 100 * a.moved_p25.iloc[0],
                                            100 * a.moved_p75.iloc[0]) if len(a) else L.DASH,
                     ("%.0f" % q.top_fmin_pct.iloc[0]) if len(q) else L.DASH,
                     ("%.3f" % o.jaccard_median.iloc[0]) if len(o) else L.DASH,
                     ("%.0f" % (100 * m.exact_median.iloc[0])) if len(m) else L.DASH]
            lines.append(" & ".join(cells) + r" \\")
            shown = True
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{%d}{l}{\footnotesize Medians, draw 0, fitted coefficients; $\alpha$ moved: "
                 r"$|\alpha - %.4f| > 0.01$, median (IQR) over every fitted draw.} \\" % (len(cols), INERT_V4))
    lines.append(r"\end{tabular}")
    ub = "; ".join("%s %.1f%% (n = %d)" % (L.LABEL[mo], 100 * exact_upper(metas[mo]["scored"]), metas[mo]["scored"])
                   for mo in L.MODELS if mo in metas)
    lines.append("% every null passes for 0 targets; exact one-sided 95% upper bounds: " + ub)
    lines.append("% WCM > both draws: two-draw targets where the production circuit's min(Z, A, C) exceeds both draws'; "
                 "Jaccard and exact bin (anchor: frequency decile; contrib: the members' contribution bin) over every draw")
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(frames):
    plt, st = L.figure_style()
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.titlepad": 4.0, "axes.labelsize": 7.5,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.7,
                         "grid.linewidth": 0.5, "axes.labelpad": 2.0, "axes.grid.axis": "both"})
    models = [m for m in L.MODELS if m in frames]
    fig, axes = plt.subplots(1, len(models), figsize=(5.5, 2.0), squeeze=False, sharey=True)
    col = {"wcm": st.CATEGORICAL[0], "anchor": st.CATEGORICAL[1], "contrib": st.CATEGORICAL[2]}
    lab = {"wcm": "production (WCM)", "anchor": "anchor-active", "contrib": "contribution-matched"}
    for ax, model in zip(axes[0], models):
        df = frames[model]
        h = df[(df.split == "held") & (df.alpha == "fit")]
        for arm in ("anchor", "contrib", "wcm"):
            x = h[h.arm == arm]
            ax.scatter(x[L.NEC].clip(-0.1, 1.1), x["fmin"].clip(-0.1, 1.6), s=6, alpha=0.6, color=col[arm],
                       linewidths=0, label=lab[arm], zorder=3 if arm == "wcm" else 2)
        ax.axvline(L.NEC_MIN, color=st.BASELINE, lw=0.6, ls="--")
        ax.axhline(L.BAND[0], color=st.BASELINE, lw=0.6, ls="--")
        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(-0.12, 1.62)
        ax.set_title(L.LABEL[model])
        ax.set_xlabel("necessity")
    axes[0][0].set_ylabel(r"min(Z, A, C)")
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, markerscale=2, bbox_to_anchor=(0.5, 1.0))
    L.save_with_legend(fig, FIGURE, 0.86)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bands, alpha, alpha_per, quant, overlap, match, shortfall, nec, metas, frames = [], [], [], [], [], [], [], [], {}, {}
    print("alpha_inert: v4 schedule %.4f; protocol-v1 constant lr %.4f" % (INERT_V4, INERT_V1))
    for model in L.MODELS:
        df, prof, meta = load(model)
        if df.empty:
            L.not_available("%s harder nulls" % model)
            continue
        metas[model], frames[model] = meta, df
        print("%-11s %s" % (L.LABEL[model], meta))
        bands += bands_table(model, df)
        a, p = alpha_table(model, df)
        alpha += a
        alpha_per.append(p)
        quant += quantile_table(model, df)
        overlap += overlap_table(model, df)
        m, s = match_tables(model, df, prof)
        match += m
        shortfall += s
        nec += necessity_table(model, df, prof)
    for name, rows in (("bands", bands), ("alpha", alpha), ("quantiles", quant), ("overlap", overlap),
                       ("match", match), ("shortfall", shortfall), ("necessity", nec)):
        pd.DataFrame(rows).to_csv(OUT / ("nulls_detail_%s.csv" % name), index=False)
    pd.concat(alpha_per).to_csv(OUT / "nulls_detail_alpha_per_circuit.csv", index=False)
    (OUT / "nulls_detail_meta.json").write_text(json.dumps(dict(
        metas=metas, alpha_inert_v4=INERT_V4, alpha_inert_v1=INERT_V1, move_tol=MOVE_TOL, boot_b=L.BOOT_B,
        boot_seed=L.BOOT_SEED, band=L.BAND, nec_min=L.NEC_MIN), indent=1), encoding="utf-8")
    write_tex(bands, alpha, quant, overlap, match, metas)
    write_figure(frames)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_rows", 500):
        for name, rows in (("bands", bands), ("alpha", alpha), ("quantiles", quant), ("overlap", overlap),
                           ("match", match), ("shortfall", shortfall), ("necessity", nec)):
            print("\n== %s" % name)
            print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("written %s, %s, %s/*.csv" % (TABLE, FIGURE, OUT))


if __name__ == "__main__":
    main()
