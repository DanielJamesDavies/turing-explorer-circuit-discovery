"""DAN-181 report: reads harder_nulls.py's rows (no model) and writes OUT84/report.md + summary.json.

Tables (held-out strongest unless stated; one row per target = draw 0 for pass rates, all draws pooled where said):
  1  per arm: pass rate with a 95% Wilson interval, Z / A / C / necessity median [IQR], share with any score in band
  2  pass rate by depth band x site kind, per arm
  3  training split vs held-out (fitted arms): medians and pass rates on both, so a null reads as "can't fit" or
     "fits but doesn't generalise"
  4  coefficients (fitted arms): alpha median, share moved from the gradient-free trajectory, modal share
  5  member sets vs the production circuit: frequency-decile and contribution-decile histograms, median log10 corpus
     count, anchor-active and prefix-active shares, median contribution
  6  per-target quantile of the WCM score within its null distribution (targets with >= 2 draws of that null)
  7  pool: anchor-active pool size and the production members' coverage by lag window; DAN-17 stored / refit counts
  8  match quality per null: exact-bin share, bin distance, drawn / reference size, overlap with the production members
  9  match shortfall by the member's own bin (contrib: the production members' contribution deciles)

  OUT84=experiments/084-harder-nulls/out PYTHONPATH=src python experiments/084-harder-nulls/report.py
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nulls_core as NC  # noqa: E402

OUT84 = Path(os.environ.get("OUT84", str(HERE / "out")))
METRICS = [("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", NC.NEC)]
ARM_ORDER = ["wcm", "wcm_a1", "dan17", "dan17_a1", "anchor", "anchor_a1", "contrib", "contrib_a1", "uniform",
             "uniform_a1"]


def read_rows(d, stem):
    rows = []
    for p in [Path(d) / ("%s.jsonl" % stem)] + sorted(Path(d).glob("%s.shard*.jsonl" % stem)):
        if p.exists():
            for line in open(p):
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
    return rows


def frame(rows):
    """Good rows (no error / skip), last row per (seed, arm, draw, split), with pass / vacuous / fmin columns."""
    rows = [r for r in rows if "error" not in r and "skip" not in r]
    last = {}
    for r in rows:
        last[(r.get("seed"), r.get("arm"), r.get("draw"), r.get("split"))] = r
    df = pd.DataFrame(list(last.values()))
    if df.empty:
        return df
    for _, c in METRICS + [("ind", NC.IND)]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["vacuous"] = df.get("vacuous_tk", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["pass"] = [NC.passes(r) for r in df.to_dict("records")]
    df["fmin"] = df[[c for _, c in METRICS[:3]]].min(axis=1, skipna=False)
    inb = lambda c: (df[c] >= NC.BAND[0]) & (df[c] <= NC.BAND[1])
    df["any_in_band"] = inb(METRICS[0][1]) | inb(METRICS[1][1]) | inb(METRICS[2][1])
    return df


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


def arms_in(df):
    present = list(df["arm"].unique())
    return [a for a in ARM_ORDER if a in present] + sorted(a for a in present if a not in ARM_ORDER)


def med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return "-"
    return "%.2f [%.2f, %.2f]" % (s.median(), s.quantile(0.25), s.quantile(0.75))


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.1f" % (100 * x)


def table_overview(df):
    """Table 1: per arm on held-out strongest; pass rate on draw 0 (one per target) and over all draws."""
    out = []
    h = df[(df.split == "held") & ~df.vacuous]
    for a in arms_in(h):
        d = h[h.arm == a]
        d0 = d[d.draw == 0]
        k, n = int(d0["pass"].sum()), len(d0)
        lo, hi = wilson(k, n)
        out.append(dict(arm=a, targets=n, draws=len(d), pass_pct=pct(k / n if n else float("nan")),
                        ci95="[%s, %s]" % (pct(lo), pct(hi)), pass_all_draws=pct(d["pass"].mean() if len(d) else float("nan")),
                        any_in_band=pct(d0["any_in_band"].mean() if n else float("nan")),
                        **{m: med_iqr(d0[c]) for m, c in METRICS}, induce=med_iqr(d0[NC.IND]),
                        n_median=float(d0["n"].median()) if "n" in d0 and n else None))
    return pd.DataFrame(out)


def table_cells(df):
    """Table 2: pass rate (%) by band x kind, per arm (held, draw 0)."""
    h = df[(df.split == "held") & (df.draw == 0) & ~df.vacuous]
    if h.empty:
        return pd.DataFrame()
    g = h.groupby(["band", "kind", "arm"])["pass"].agg(["mean", "size"]).reset_index()
    g["v"] = [("%s (%d)" % (pct(m), s)) for m, s in zip(g["mean"], g["size"])]
    t = g.pivot_table(index=["band", "kind"], columns="arm", values="v", aggfunc="first")
    tot = h.groupby("arm")["pass"].agg(["mean", "size"])
    t.loc[("all", ""), :] = ["%s (%d)" % (pct(tot.loc[a, "mean"]), tot.loc[a, "size"]) for a in t.columns]
    return t[[a for a in arms_in(h) if a in t.columns]]


def table_train_held(df):
    """Table 3: fitted arms, draw 0, paired on targets with both splits."""
    out = []
    f = df[(df.alpha == "fit") & (df.draw == 0) & ~df.vacuous]
    for a in arms_in(f):
        d = f[f.arm == a]
        tr, ho = d[d.split == "train"].set_index("seed"), d[d.split == "held"].set_index("seed")
        both = tr.index.intersection(ho.index)
        if not len(both):
            continue
        tr, ho = tr.loc[both], ho.loc[both]
        row = dict(arm=a, targets=len(both), pass_train=pct(tr["pass"].mean()), pass_held=pct(ho["pass"].mean()),
                   fits_not_generalise=pct((tr["pass"] & ~ho["pass"]).mean()), fails_train=pct((~tr["pass"]).mean()))
        for m, c in METRICS:
            row["%s_train" % m] = "%.2f" % tr[c].median() if tr[c].notna().any() else "-"
            row["%s_held" % m] = "%.2f" % ho[c].median() if ho[c].notna().any() else "-"
        row["Z_train_ge_0.8"] = pct((tr[METRICS[0][1]] >= 0.8).mean())
        out.append(row)
    return pd.DataFrame(out)


def table_alpha(df):
    """Table 4: coefficients of the fitted circuits (held rows, all draws)."""
    out = []
    f = df[(df.alpha == "fit") & (df.split == "held")]
    for a in arms_in(f):
        d = f[f.arm == a]
        g = lambda c: pd.to_numeric(d.get(c, pd.Series(dtype=float)), errors="coerce")
        out.append(dict(arm=a, circuits=len(d), alpha_inert=("%.4f" % g("alpha_inert").median()) if g("alpha_inert").notna().any() else "-",
                        alpha_median=med_iqr(g("alpha_median")), moved=med_iqr(g("alpha_moved")),
                        moved_up=med_iqr(g("alpha_up")), moved_down=med_iqr(g("alpha_down")),
                        mode_share=med_iqr(g("alpha_mode_share")), all_inert=pct((g("alpha_moved") == 0).mean())))
    return pd.DataFrame(out)


def _pool_sets(df, prof):
    """{set name: [set_profile dicts]}: production members and DAN-17 sets from the profile rows, the null sets from
    their held rows (draw 0)."""
    sets = {"wcm": [r["set_wcm"] for r in prof if r.get("set_wcm")],
            "dan17": [r["set_dan17"] for r in prof if r.get("set_dan17")]}
    if not df.empty and "set" in df:
        h = df[(df.split == "held") & (df.alpha == "fit") & (df.draw == 0)]
        for a in arms_in(h):
            if a in ("wcm", "dan17"):
                continue
            sets[a] = [s for s in h[h.arm == a]["set"] if isinstance(s, dict) and s.get("n")]
    return {k: v for k, v in sets.items() if v}


def table_sets(df, prof):
    """Table 5: pooled decile histograms (% of members) and medians per member set."""
    out = []
    for name, ss in _pool_sets(df, prof).items():
        fh = np.sum([s["fdec_hist"] for s in ss], axis=0).astype(float)
        ch = np.sum([s["cdec_hist"] for s in ss], axis=0).astype(float)
        n = sum(s["n"] for s in ss)
        bh = np.sum([s.get("cbin_hist") or [0] * 12 for s in ss], axis=0).astype(float)
        out.append(dict(set=name, circuits=len(ss), members=n,
                        cbin_pct=" ".join("%.0f" % x for x in 100 * bh / max(bh.sum(), 1)),
                        fdec_pct=" ".join("%.0f" % x for x in 100 * fh / max(fh.sum(), 1)),
                        never_active_pct=pct(sum(s.get("fdec_dead", 0) for s in ss) / max(n, 1)),
                        cdec_pct=" ".join("%.0f" % x for x in 100 * ch / max(ch.sum(), 1)),
                        log10_count=float(np.median([s["log10_count_median"] for s in ss])),
                        contrib_median=float(np.median([s["contrib_median"] for s in ss])),
                        anchor_active=pct(float(np.median([s["anchor_active"] for s in ss]))),
                        prefix_active=pct(float(np.median([s["prefix_active"] for s in ss])))))
    return pd.DataFrame(out)


def quantile(wcm, nulls):
    """Where the WCM value falls in the null draws: (#null < wcm + 0.5 #null == wcm) / #null; NaN if undefined."""
    v = [x for x in nulls if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if wcm is None or (isinstance(wcm, float) and math.isnan(wcm)) or not v:
        return float("nan")
    return (sum(x < wcm for x in v) + 0.5 * sum(x == wcm for x in v)) / len(v)


def table_quantiles(df, min_draws=2):
    """Table 6: per null arm, over targets with >= min_draws draws: median quantile of the WCM score per metric, the
    share where WCM beats every draw, and the share where WCM passes while every draw fails."""
    out = []
    h = df[(df.split == "held") & ~df.vacuous]
    w = h[h.arm == "wcm"].drop_duplicates("seed").set_index("seed")
    for a in arms_in(h):
        if a in ("wcm", "wcm_a1"):
            continue
        d = h[h.arm == a]
        g = {k: v for k, v in d.groupby("seed") if len(v) >= min_draws and k in w.index}
        if not g:
            continue
        row = dict(arm=a, targets=len(g), draws_median=float(np.median([len(v) for v in g.values()])))
        for m, c in METRICS + [("fmin", "fmin")]:
            q = [quantile(w.loc[k, c], v[c].tolist()) for k, v in g.items()]
            q = [x for x in q if not math.isnan(x)]
            row["q_%s" % m] = "%.2f" % np.median(q) if q else "-"
            row["top_%s" % m] = pct(np.mean([x == 1.0 for x in q])) if q else "-"
        row["wcm_pass_all_fail"] = pct(np.mean([bool(w.loc[k, "pass"]) and not v["pass"].any() for k, v in g.items()]))
        row["any_draw_pass"] = pct(np.mean([bool(v["pass"].any()) for v in g.values()]))
        out.append(row)
    return pd.DataFrame(out)


def table_match(df):
    """Table 8: how well each null's draws matched (held rows of the fitted arms, every draw): the share of members
    matched to their exact bin, the mean first-bin distance per drawn member, drawn / reference size, short sites."""
    out = []
    if "n_exact" not in df:
        return pd.DataFrame()
    h = df[(df.split == "held") & (df.alpha == "fit") & df["n_exact"].notna()]
    for a in arms_in(h):
        d = h[h.arm == a]
        nd = pd.to_numeric(d["n_drawn"], errors="coerce")
        out.append(dict(arm=a, draws=len(d), exact=med_iqr(pd.to_numeric(d["n_exact"]) / nd.where(nd > 0)),
                        dist_per_member=med_iqr(pd.to_numeric(d["dist0"]) / nd.where(nd > 0)),
                        drawn_over_ref=med_iqr(nd / pd.to_numeric(d["n_ref"])),
                        with_short_sites=pct((pd.to_numeric(d["short_sites"]) > 0).mean()),
                        overlap_wcm=med_iqr(d.get("overlap_wcm", pd.Series(dtype=float))),
                        overlap_frac=med_iqr(d.get("overlap_frac", pd.Series(dtype=float)))))
    return pd.DataFrame(out)


def table_shortfall(df):
    """Table 9: the match shortfall by the member's own first matching bin (anchor: frequency decile; contrib: the
    production members' contribution bin), pooled over every draw (held rows of the fitted null arms, one row per
    draw): members, the share that got a same-site draw, the share matched to their exact bin, and the mean |first-bin
    difference| per drawn member. A shortfall concentrated in the top bins means the pool has too few latents as
    strong as the production members."""
    if "by_bin" not in df:
        return pd.DataFrame()
    h = df[(df.split == "held") & (df.alpha == "fit")]
    out = []
    for a in arms_in(h):
        acc = {}
        for bb in h[h.arm == a]["by_bin"]:
            if not isinstance(bb, dict):
                continue
            for b, v in bb.items():
                s = acc.setdefault(b, [0, 0, 0, 0])
                for j in range(4):
                    s[j] += int(v[j])
        for b in sorted(acc, key=lambda x: (x == "all", int(x) if x != "all" else 0)):
            m, dr, ex, di = acc[b]
            out.append(dict(arm=a, member_bin=b, members=m, drawn_pct=pct(dr / m if m else float("nan")),
                            exact_pct=pct(ex / m if m else float("nan")),
                            mean_bin_dist=("%.2f" % (di / dr)) if dr else "-"))
    return pd.DataFrame(out)


def table_pool(prof):
    """Table 7: anchor-active pool and the production members' lag coverage (medians over targets)."""
    if not prof:
        return pd.DataFrame()
    wins = list(NC.COVER_WINDOWS) + ["prefix"]
    row = dict(targets=len(prof), win=prof[0].get("win"), pool_median=float(np.median([r["pool_total"] for r in prof])),
               n_ref_median=float(np.median([r["n_ref"] for r in prof])))
    for w in wins:
        k = w if w == "prefix" else "w%d" % w
        v = [r["cover_wcm"].get(k) for r in prof if r.get("cover_wcm")]
        row["wcm_%s" % k] = pct(float(np.median(v))) if v else "-"
    v17 = [r["cover_dan17"].get("w%d" % prof[0].get("win", NC.WIN_DEFAULT)) for r in prof if r.get("cover_dan17")]
    row["dan17_anchor_active"] = pct(float(np.median(v17))) if v17 else "-"
    src = [r.get("dan17_source") for r in prof]
    row["dan17_stored"] = sum(1 for s in src if s and s != "refit")
    row["dan17_refit"] = sum(1 for s in src if s == "refit")
    ex = set()
    for r in prof:                                  # a list of nulls (2026-10-24 final), or the earlier bool
        e = r.get("exclude_members")
        ex.update(e if isinstance(e, list) else (["all"] if e else []))
    row["members_in_pool"] = ("excluded from %s" % ",".join(sorted(ex))) if ex else "included"
    return pd.DataFrame([row])


def md(t):
    """A DataFrame as a markdown table (no tabulate dependency); a MultiIndex is written as leading columns."""
    if t is None or len(t) == 0:
        return "(no rows)\n"
    if isinstance(t.index, pd.MultiIndex):
        t = t.reset_index()
    cols = [str(c) for c in t.columns]
    fmt = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else (
        "%.3g" % v if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def build(out_dir=OUT84):
    rows, prof = read_rows(out_dir, "nulls"), [r for r in read_rows(out_dir, "profile") if "error" not in r]
    df = frame(rows)
    parts = ["# DAN-181 harder nulls: report", "",
             "Rows: %d good of %d (errors %d, skips %d); targets %d; profiles %d. Pass = DAN-8 (Z, A, C in [0.8, 1.5], "
             "necessity >= 0.9), held-out strongest unless stated; vacuous rows excluded (%d)." % (
                 len(df), len(rows), sum("error" in r for r in rows), sum("skip" in r for r in rows),
                 df.seed.nunique() if len(df) else 0, len(prof), int(df.vacuous.sum()) if len(df) else 0), ""]
    tabs = {}
    if len(df):
        tabs = dict(overview=table_overview(df), cells=table_cells(df), train_held=table_train_held(df),
                    alpha=table_alpha(df), sets=table_sets(df, prof), quantiles=table_quantiles(df), pool=table_pool(prof),
                    match=table_match(df), shortfall=table_shortfall(df))
    else:
        tabs = dict(sets=table_sets(df, prof), pool=table_pool(prof))
    titles = dict(overview="1. Per arm (held-out strongest; pass on draw 0 with 95% Wilson interval; medians [IQR])",
                  cells="2. Pass rate % (n) by depth band x site kind (draw 0)",
                  train_held="3. Training split vs held-out (fitted arms, draw 0, paired)",
                  alpha="4. Coefficients (fitted arms; moved = |alpha - gradient-free value| > MOVE_TOL)",
                  sets="5. Member sets: frequency / contribution decile histograms (% of members, deciles 0..9; "
                       "cbin = the production members' own contribution deciles, bins -1..10)",
                  quantiles="6. Per-target quantile of the WCM score within the null draws (targets with >= 2 draws)",
                  pool="7. Anchor-active pool and production-member lag coverage (medians over targets)",
                  match="8. Match quality of the drawn sets (exact-bin share, mean bin distance per member, size, "
                        "overlap with the production members)",
                  shortfall="9. Match shortfall by the member's own bin (every draw pooled)")
    for k in ("overview", "cells", "train_held", "alpha", "sets", "quantiles", "pool", "match", "shortfall"):
        if k in tabs:
            parts += ["## " + titles[k], "", md(tabs[k])]
    text = "\n".join(parts)
    summary = {k: (v.reset_index().to_dict("records") if isinstance(v.index, pd.MultiIndex) else v.to_dict("records"))
               for k, v in tabs.items()}
    return text, summary


def main():
    text, summary = build(OUT84)
    OUT84.mkdir(parents=True, exist_ok=True)
    (OUT84 / "report.md").write_text(text, encoding="utf-8")
    (OUT84 / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(text)
    print("written %s" % (OUT84 / "report.md"))


if __name__ == "__main__":
    main()
