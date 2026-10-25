"""Pass-rate bookkeeping for the 086 re-score report (rescore.py REPORT=1); no GPU, no driver import (unit-tested).

  classify(rows)  one row per target (or target x key) with rejected / scored / below (the floor) / vac / pass flags
  headline(d)     the headline over targets with >= FLOOR clean held-out strongest rows, plus the worst case (every
                  below-floor target counted as a fail, in the denominator). Below-floor targets are scored and
                  flagged, never dropped
  paired(a, b)    the leakage effect: as-run re-score a vs rebuilt re-score b on the same targets, overall and per
                  band x kind
"""
FLOOR = 8                # heldout_rule.FLOOR
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]


def band_of(key):
    l = int(key.split(".")[0])
    return "L0-4" if l <= 4 else ("L5-7" if l <= 7 else "L8-11")


def passes(d):
    """The DAN-8 rule (paper): Z, A, C faithfulness in [0.8, 1.5] and necessity >= 0.9 (a missing value fails)."""
    ok = d[HEAD].notna().all(axis=1)
    return ok & ((d[HEAD] >= 0.8) & (d[HEAD] <= 1.5)).all(axis=1) & (d["phi_sup_blind_tk"] >= 0.9)


def classify(rows, keys=("seed",)):
    """One row per key (last wins; error rows dropped) with: rejected (no circuit), below (fewer than FLOOR clean
    held-out strongest rows, or a no_heldout / no_contrast_heldout skip: scored and flagged, or unscoreable), scored
    (a score row), vac (vacuous denominator), pass (the DAN-8 rule on scored rows), band, kind."""
    import pandas as pd
    d = pd.DataFrame(list(rows))
    if not len(d):
        return d
    if "error" in d:
        d = d[d["error"].isna()]
    d = d.drop_duplicates(list(keys), keep="last").copy()
    skip = d["skip"] if "skip" in d else pd.Series(None, index=d.index, dtype=object)
    nh = pd.to_numeric(d["n_held"], errors="coerce") if "n_held" in d else pd.Series(16.0, index=d.index)
    d["rejected"] = (skip == "rejected").values
    d["scored"] = skip.isna().values
    d["below"] = ((nh.fillna(16) < FLOOR) | skip.isin(["no_heldout", "no_contrast_heldout"])).values
    vac = d["vacuous_tk"].astype(object).where(d["vacuous_tk"].notna(), False).astype(bool) if "vacuous_tk" in d \
        else pd.Series(False, index=d.index)
    d["vac"] = vac.values & d["scored"].values
    for h in HEAD + ["phi_sup_blind_tk"]:
        if h not in d:
            d[h] = float("nan")
        d[h] = pd.to_numeric(d[h], errors="coerce")
    d["pass"] = (passes(d) & d["scored"]).astype(bool)
    d["band"] = [band_of(k) for k in d["seed"]]
    d["kind"] = [k.split(".")[1] for k in d["seed"]]
    return d


def headline(d):
    """dict: n / passed / rate over targets with >= FLOOR clean held-out strongest rows (scored, non-vacuous: the
    headline); n_below and the below-floor targets' own pass rate; the worst case (every below-floor target counted as
    a fail, in the denominator). Rejected circuits are excluded throughout."""
    d = d[~d.rejected]
    head = d[d.scored & ~d.below & ~d.vac]
    below = d[d.below]
    nw = len(head) + len(below)
    nbs = int(below.scored.sum())
    return dict(n=len(head), passed=int(head["pass"].sum()), rate=float(head["pass"].mean()) if len(head) else None,
                n_below=len(below), below_scored_rate=float(below[below.scored]["pass"].mean()) if nbs else None,
                n_worst=nw, worst=float(head["pass"].sum() / nw) if nw else None)


def paired(a, b):
    """The paired leakage effect on the same targets: a = as-run re-score (ctx), b = rebuilt (ctx_v2), both from
    classify. Pairs: a circuit in both, scored and non-vacuous as run, not vacuous rebuilt. Headline pairs = b at or
    above the floor; worst case = every pair, b below the floor counted as a fail. Returns (overall dict, per band x
    kind DataFrame); effect = as-run pass rate - rebuilt pass rate."""
    import pandas as pd
    j = a.set_index("seed")[["pass", "vac", "rejected", "scored", "band", "kind"]].join(
        b.set_index("seed")[["pass", "vac", "rejected", "scored", "below"]], lsuffix="_a", rsuffix="_b", how="inner")
    j = j[~j.rejected_a & ~j.rejected_b & j.scored_a & ~j.vac_a & ~(j.scored_b & j.vac_b)]

    def stats(x):
        h = x[~x.below]
        f = lambda s: float(s.mean()) if len(s) else None
        worst_b = x.pass_b & ~x.below
        return dict(n_pairs=len(x), n_head=len(h), n_below=int(x.below.sum()),
                    pass_asrun=f(h.pass_a), pass_v2=f(h.pass_b),
                    effect=float(h.pass_a.mean() - h.pass_b.mean()) if len(h) else None,
                    pass_asrun_all=f(x.pass_a), pass_v2_worst=f(worst_b),
                    effect_worst=float(x.pass_a.mean() - worst_b.mean()) if len(x) else None)
    rows = [dict(band=bk[0], kind=bk[1], **stats(g)) for bk, g in j.groupby(["band", "kind"])]
    return stats(j), pd.DataFrame(rows)
