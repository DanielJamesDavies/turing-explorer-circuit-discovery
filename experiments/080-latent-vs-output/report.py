"""Experiment 080 report (CPU only): latent- vs output-targeted circuits, bootstrap 95% intervals over targets
(B = 2000, seed 20261018). Descriptive tables are grouped by the within-kind design's cells (GROUP=cell, default) or
by tier x band (GROUP=tier_band, for the SAMPLING=tierband list).

Per target, every OUT arm's values are matched in size to the target's LAT circuit (common.interp_log_n: linear in
log10 size over the arm's lambda grid, flagged when the LAT size is outside the arm's size range); pass indicators are
interpolated the same way (a fractional pass at the matched size, as 071 interpolates pass rates). Set overlaps use the
OUT_W lambda whose size is nearest the LAT size in log10 (common.nearest_in_log). Output-endpoint comparisons drop the
targets whose output denominator is vacuous (out_vacuous on the LAT row: |m(full) - m(empty)| < 0.1 nats; the
denominator is a property of the target's contexts, the same for every circuit of the target).

Tables
  1  OUT circuits on the LATENT metrics vs LAT (Z, A, C, necessity, latent pass)
  2  LAT circuits on the OUTPUT metrics vs OUT_W (output Z, A, C, necessity, output pass)
  3  pass rates: every arm on both pass rules
  4  overlap: Jaccard(LAT, OUT_W) and both containments, against the refit floor (062 stable-core Jaccard for the
     same targets where it exists, otherwise its overall median)
  5  OUT_U vs OUT_W: output pass and output Z at matched size; per-target size needed to reach output Z >= 0.8, ratio
  P1 tests: per pair of the design (causal: the 9 kind x band groups, high - low next-token effect; withinkind: its 6
     pairs), the difference of tier medians on J(LAT, OUT_W), OUT_W's latent Z and LAT's output Z; pooled as the simple
     mean of the pair differences (primary; inverse-variance as sensitivity); the slope on log(effect_abs) (legacy
     designs: log boost_peak) with kind x band fixed effects (secondary)
  measurement checks: effect_abs by group in nats, and Spearman(effect_abs, 064 boost_peak) over the measured pool
  + the registered predictions P1-P3 (README) and every number a paper paragraph would quote
Tiers are the design's own (META, default targets_meta.csv): "high / low next-token effect" (causal) or "direct-path
(064) high / mid / low" (legacy). Every target is an internal latent; the tier says how much it affects the next token.

  OUT=experiments/080-latent-vs-output/out PYTHONPATH=src python experiments/080-latent-vs-output/report.py
  env: B (2000)  SMOKE=1 (per-row listing + sanity checks; no prediction verdicts)  SRC_OUT (062 out_full/out)
       GROUP (cell | tier_band)  META (the design's targets_meta*.csv)  EFFECTS_FILE (effect/effects.parquet)
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = Path(os.environ.get("OUT", str(HERE / "out")))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))
STABLE = C.EXP / "062-h100-protocol-v1" / "out_stable_core" / "stable_core.jsonl"
B = int(os.environ.get("B", C.B_BOOT))
SMOKE = os.environ.get("SMOKE") == "1"
GROUP = os.environ.get("GROUP", "cell")      # row grouping of the descriptive tables: cell (default) | tier_band
OUT_ARMS = ("OUT_W", "OUT_U", "OUT_W_FULL", "OUT_U_FULL")
REACH = 0.8                    # output Z the size-to-reach comparison asks for (the pass band's lower edge)


# ------------------------------------------------------------------------------------------- loading
def load(out=OUT):
    """The last row without an error per (seed, arm, lam), plus the error rows (for the bookkeeping line)."""
    rows, errs = {}, []
    for f in sorted(glob.glob(str(Path(out) / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            k = (r["seed"], r["arm"], r["lam"])
            if r.get("error"):
                errs.append(r)
            else:
                rows[k] = r
    return rows, errs


def stable_core():
    """Per target, the 062 stable-core refit Jaccard (fit 1 vs fit 2), and the overall median."""
    js = {}
    if STABLE.exists():
        for line in open(STABLE):
            r = json.loads(line)
            if r.get("stage") == "fit2" and r.get("jaccard") is not None:
                js[r["seed"]] = float(r["jaccard"])
    import numpy as np
    return js, (float(np.median(list(js.values()))) if js else None)


def design_meta():
    """The design's own target metadata (META, default targets_meta.csv: the list run.py ran). Its tier wins over the
    tier a row carries, so a key that appears under several designs is labelled by the design being reported."""
    import csv
    p = Path(os.environ.get("META", str(HERE / "targets_meta.csv")))
    return {r["seed"]: r for r in csv.DictReader(open(p))} if p.exists() else {}


def per_target(rows, meta=None):
    """{key: dict(meta..., lat=row, arms={arm: [rows by size]})} for targets with a scored LAT row. Tier and
    effect_abs come from the design's metadata where it has the target; old tier names are mapped (common.norm_tier)."""
    meta = design_meta() if meta is None else meta
    T = {}
    for (seed, arm, lam), r in rows.items():
        t = T.setdefault(seed, dict(seed=seed, arms=defaultdict(list)))
        for k in ("tier", "band", "boost_peak"):
            if r.get(k) is not None:
                t[k] = r[k]
        m = meta.get(seed)
        if m:
            if m.get("tier"):
                t["tier"] = m["tier"]
            for k in ("effect_abs", "boost_peak"):
                if C.num(m.get(k)) is not None:
                    t[k] = float(m[k])
        t["tier"] = C.norm_tier(t.get("tier"))
        t["kind"] = C.parse(seed)[1]
        t["layer"] = C.parse(seed)[0]
        if arm == "LAT":
            t["lat"] = r
        elif "skip" not in r:
            t["arms"][arm].append(r)
    for t in T.values():
        for a in t["arms"]:
            t["arms"][a].sort(key=lambda r: C.num(r.get("n")) or 0)
        t.setdefault("band", C.band(t["layer"]))
    return {k: v for k, v in T.items() if "lat" in v and "skip" not in v["lat"]}


def matched(T):
    """One record per target: LAT's own values, each OUT arm's values at the LAT size, overlaps at the nearest-size
    lambda, sizes to reach output Z >= REACH, and the extrapolation flags."""
    out = []
    for key, t in T.items():
        lat = t["lat"]
        n_star = C.num(lat.get("n"))
        tier = t.get("tier") or (C.tier_of(float(t["boost_peak"])) if t.get("boost_peak") is not None else None)
        pair = C.pair_of(t["kind"], t.get("band"), tier) if tier else None
        rec = dict(seed=key, tier=tier, band=t.get("band"), kind=t["kind"], layer=t["layer"],
                   pair=C.pair_name(pair[0], pair[1]) if pair else None, kb="%s %s" % (t["kind"], t.get("band")),
                   boost_peak=t.get("boost_peak"), effect_abs=t.get("effect_abs"), lat_n=n_star,
                   out_vacuous=bool(lat.get("out_vacuous")),
                   lat_vacuous=bool(lat.get("vacuous_tk")), out_den=lat.get("out_den"))
        for m, f in C.METRICS.items():
            rec["LAT_" + m] = C.num(lat.get(f))
        rec["LAT_pass"] = float(C.passes_latent(lat))
        rec["LAT_opass"] = float(C.passes_output(lat))
        for arm, rs in t["arms"].items():
            ns = [r.get("n") for r in rs]
            for m, f in C.METRICS.items():
                rec["%s_%s" % (arm, m)], ex = C.interp_log_n(ns, [r.get(f) for r in rs], n_star)
            rec["%s_extrap" % arm] = ex
            rec["%s_pass" % arm] = C.interp_log_n(ns, [float(C.passes_latent(r)) for r in rs], n_star)[0]
            rec["%s_opass" % arm] = C.interp_log_n(ns, [float(C.passes_output(r)) for r in rs], n_star)[0]
            rec["%s_reach" % arm], rec["%s_reach_status" % arm] = C.size_to_reach(ns, [r.get("out_free0") for r in rs],
                                                                                  REACH)
            rec["%s_n_range" % arm] = (C.num(ns[0]), C.num(ns[-1])) if ns else (None, None)
            i = C.nearest_in_log(ns, n_star)
            if i is not None:
                r = rs[i]
                rec.update({"%s_J" % arm: C.num(r.get("jaccard_lat")), "%s_cont_lat" % arm: C.num(r.get("cont_lat_in_arm")),
                            "%s_cont_arm" % arm: C.num(r.get("cont_arm_in_lat")), "%s_J_n" % arm: C.num(r.get("n")),
                            "%s_J_lam" % arm: r.get("lam")})
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------- formatting
def f3(v):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else "%.3f" % v


def ci(d, k="a"):
    return "%s [%s, %s]" % (f3(d.get(k)), f3(d.get(k + "_lo")), f3(d.get(k + "_hi")))


def tiers_of(M):
    """The design's tiers present, highest first (causal high / low, or the direct-path dp_*)."""
    present = {m["tier"] for m in M}
    return [t for t in C.CAUSAL_TIERS + C.TIERS if t in present]


def groups(M, by):
    """[(label, records)]: overall, then per tier (or per tier x band / design cell / kind)."""
    g = [("all", M)]
    tiers = tiers_of(M)
    if by in ("tier", "tier_band", "cell"):
        for t in tiers:
            g.append((t, [m for m in M if m["tier"] == t]))
    if by == "tier_band":
        for t in tiers:
            for b in C.BANDS:
                g.append(("%s %s" % (t, b), [m for m in M if m["tier"] == t and m["band"] == b]))
    if by == "cell":                                     # the design's cells, pair by pair
        for k, b, hi, lo in C.design_pairs(tiers):
            for t in (hi, lo):
                g.append((C.cell_name(k, b, t), [m for m in M if m["kind"] == k and m["band"] == b and m["tier"] == t]))
    if by == "kind":
        for k in ("attn", "mlp", "resid"):
            g.append((k, [m for m in M if m["kind"] == k]))
    return [(l, ms) for l, ms in g if ms]


def col(ms, k):
    return [m.get(k) for m in ms]


def compare(ms, a, b, stat="median", drop_out_vac=False):
    ms = [m for m in ms if not (drop_out_vac and m["out_vacuous"])]
    return C.paired_bootstrap(col(ms, a), col(ms, b), stat=stat, B=B)


# ------------------------------------------------------------------------------------------- tables
def table_cross(M, arms, base, metrics, title, by, drop_out_vac):
    print("\n== %s (%s; median [95%% CI] unless noted; pass = mean)" % (title, by))
    for m in metrics:
        stat = "mean" if m in ("pass", "opass") else "median"
        print("\n  %s" % m)
        print("  %-20s %4s  %-28s" % ("group", "n", base) + "".join("%-28s %-28s" % (a, a + " - " + base) for a in arms))
        for label, ms in groups(M, by):
            line = None
            for a in arms:
                d = compare(ms, "%s_%s" % (a, m), "%s_%s" % (base, m), stat, drop_out_vac)
                if line is None:
                    line = "  %-20s %4d  %-28s" % (label, d["n_pair"], ci(d, "b"))
                line += "%-28s %-28s" % (ci(d, "a"), ci(d, "d"))
            print(line)


def table_pass(M):
    print("\n== 3. pass rates (DAN-8 rule on each endpoint; OUT arms at the LAT size; mean [95% CI])")
    arms = ["LAT"] + [a for a in OUT_ARMS if any(("%s_pass" % a) in m for m in M)]
    print("  %-20s %4s " % ("group", "n") + "".join("%-26s %-26s" % (a + " latent", a + " output") for a in arms))
    for label, ms in groups(M, GROUP):
        line = "  %-20s %4d " % (label, len(ms))
        for a in arms:
            lp = C.paired_bootstrap(col(ms, a + "_pass"), stat="mean", B=B)
            op = C.paired_bootstrap(col([m for m in ms if not m["out_vacuous"]], a + "_opass"), stat="mean", B=B)
            line += "%-26s %-26s" % (ci(lp), ci(op))
        print(line)


def table_overlap(M, sc, sc_all):
    print("\n== 4. overlap of LAT and OUT_W at the nearest-size lambda (median [95% CI]) vs the refit floor")
    print("  refit floor: 062 stable-core Jaccard (fit 1 vs a reordered refit, same protocol); overall median %s over "
          "%d targets" % (f3(sc_all), len(sc)))
    print("  %-20s %4s  %-26s %-26s %-26s %-30s" % ("group", "n", "Jaccard", "LAT in OUT_W", "OUT_W in LAT",
                                                   "refit floor (same targets)"))
    for label, ms in groups(M, GROUP):
        j = C.paired_bootstrap(col(ms, "OUT_W_J"), B=B)
        a = C.paired_bootstrap(col(ms, "OUT_W_cont_lat"), B=B)
        b = C.paired_bootstrap(col(ms, "OUT_W_cont_arm"), B=B)
        same = [sc[m["seed"]] for m in ms if m["seed"] in sc]
        floor = ("%s (n %d)" % (f3(sorted(same)[len(same) // 2]) if same else "-", len(same)))
        print("  %-20s %4d  %-26s %-26s %-26s %-30s" % (label, j["n"], ci(j), ci(a), ci(b), floor))
    paired = [m for m in M if m["seed"] in sc and m.get("OUT_W_J") is not None]
    if paired:
        d = C.paired_bootstrap([m["OUT_W_J"] for m in paired], [sc[m["seed"]] for m in paired], B=B)
        print("  targets with a stable-core refit (n %d): J(LAT, OUT_W) %s vs refit J %s, difference %s"
              % (d["n_pair"], ci(d, "a"), ci(d, "b"), ci(d, "d")))


def table_weights(M):
    print("\n== 5. OUT_U vs OUT_W (do coefficients keep output circuits small too?)")
    print("  at the LAT size: output pass (mean) and output Z (median), OUT_U - OUT_W paired; size to reach output Z "
          ">= %.1f: per-target ratio OUT_U / OUT_W (median)" % REACH)
    print("  %-20s %4s  %-26s %-26s %-26s %-26s %-12s" % ("group", "n", "opass U - W", "oZ U - W", "n_reach W",
                                                         "n_reach U / W", "reached W/U"))
    for label, ms in groups(M, GROUP):
        ms_ = [m for m in ms if not m["out_vacuous"]]
        p = C.paired_bootstrap(col(ms_, "OUT_U_opass"), col(ms_, "OUT_W_opass"), stat="mean", B=B)
        z = C.paired_bootstrap(col(ms_, "OUT_U_oZ"), col(ms_, "OUT_W_oZ"), B=B)
        ratio = [(m["OUT_U_reach"] / m["OUT_W_reach"]) if (m.get("OUT_U_reach") and m.get("OUT_W_reach")) else None
                 for m in ms_]
        r = C.paired_bootstrap(ratio, B=B)
        w = C.paired_bootstrap(col(ms_, "OUT_W_reach"), B=B)
        nw = sum(1 for m in ms_ if m.get("OUT_W_reach_status") in ("ok", "already"))
        nu = sum(1 for m in ms_ if m.get("OUT_U_reach_status") in ("ok", "already"))
        print("  %-20s %4d  %-26s %-26s %-26s %-26s %d/%d" % (label, len(ms_), ci(p, "d"), ci(z, "d"), ci(w), ci(r),
                                                              nw, nu))


# P1's three quantities: (label, matched field, drop output-vacuous targets)
P1_QUANT = (("J(LAT, OUT_W)", "OUT_W_J", False), ("OUT_W on latent Z (cross)", "OUT_W_Z", False),
            ("LAT on output Z (cross)", "LAT_oZ", True))


def slope_measure(M):
    """The secondary test's regressor: effect_abs (the causal design) when every target has it, else 064's
    boost_peak (the legacy designs)."""
    return "effect_abs" if M and all(C.num(m.get("effect_abs")) is not None for m in M) else "boost_peak"


def p1_tests(M):
    """P1 (re-registered a third time, 2026-10-18, on the causal measure): (a) PRIMARY, per pair of the design (causal:
    the 9 kind x band groups, high - low next-token effect; legacy: withinkind's 6), the difference of tier medians
    within one kind and band, layer-matched, pooled as the SIMPLE MEAN of the pair differences, bootstrap over targets
    within cells; (b) SECONDARY, the slope of the quantity on log(effect_abs) (legacy: log boost_peak) with kind x band
    fixed effects, bootstrap within kind x band. Returns {label: (a, b)}."""
    pairs = C.design_pairs(tiers_of(M))
    xk = slope_measure(M)
    print("\n== P1 tests: within-group tier contrasts over %d pairs (primary) and the fixed-effects slope on log(%s) "
          "(secondary)" % (len(pairs), xk))
    out = {}
    for label, k, drop in P1_QUANT:
        ms = [m for m in M if not (drop and m["out_vacuous"])]
        cells = {}
        for kind, b, hi, lo in pairs:
            p = C.pair_name(kind, b)
            cells[p] = ([m.get(k) for m in ms if m["pair"] == p and m["tier"] == hi],
                        [m.get(k) for m in ms if m["pair"] == p and m["tier"] == lo])
        a = C.pair_contrast(cells, B=B)
        s = C.fe_slope(col(ms, xk), col(ms, k), col(ms, "kb"), B=B)
        out[label] = (a, s)
        print("\n  %s" % label)
        for kind, b, hi, lo in pairs:
            d = a["per_pair"].get(C.pair_name(kind, b))
            if d:
                print("      %-12s %-7s - %-7s  %s [%s, %s]  (n %d / %d)" % (C.pair_name(kind, b), hi, lo, f3(d["d"]),
                                                                         f3(d["lo"]), f3(d["hi"]), d["n_hi"],
                                                                         d["n_lo"]))
        print("      pooled (simple mean of %d pair differences) %s [%s, %s]; inverse-variance %s [%s, %s]" % (
            a["n_pairs"], f3(a["pooled"]), f3(a["pooled_lo"]), f3(a["pooled_hi"]), f3(a["iv"]), f3(a["iv_lo"]),
            f3(a["iv_hi"])))
        print("      slope on log(%s), kind x band fixed effects: %s [%s, %s] (n %d, %d groups)" % (
            xk, f3(s["slope"]), f3(s["lo"]), f3(s["hi"]), s["n"], s["n_groups"]))
    return out


def measurement_checks(M):
    """The scale of the causal measure (effect_abs by group, in nats) and how far 064's direct-path proxy was from it
    (Spearman of effect_abs and boost_peak over the measured pool, overall and per group)."""
    path = Path(os.environ.get("EFFECTS_FILE", str(C.EFFECTS_FILE)))
    print("\n== the causal next-token effect (effect_abs, nats: mean |log p_clean - log p_ablated| of the actual next "
          "token)")
    if M and any(C.num(m.get("effect_abs")) for m in M):
        print("  sampled targets, median effect_abs by cell:")
        for label, ms in groups(M, "cell")[1:]:
            v = [m["effect_abs"] for m in ms if C.num(m.get("effect_abs")) is not None]
            if v:
                import numpy as np
                print("    %-22s n %3d  median %.4f  (p10 %.4f, p90 %.4f)" % (label, len(v), np.median(v),
                                                                           np.percentile(v, 10), np.percentile(v, 90)))
    if not path.exists():
        print("  (no %s: the measured pool's distribution and the proxy check need next_token_effect.py's output)"
              % path.name)
        return
    import pandas as pd
    d = pd.read_parquet(path)
    if "error" in d:
        d = d[d.error.isna()]
    print("  measured pool: %d targets" % len(d))
    print(d.groupby(["kind", "band"]).effect_abs.describe(percentiles=[0.1, 1 / 3, 0.5, 2 / 3, 0.9]).round(4)
          .to_string())
    if "boost_peak" in d:
        rho = lambda x: x.effect_abs.rank().corr(x.boost_peak.rank())  # noqa: E731
        print("  proxy check: Spearman(effect_abs, 064 boost_peak) = %.3f overall (n %d); by group: %s" % (
            rho(d), len(d), {"%s %s" % g: round(rho(x), 3) for g, x in d.groupby(["kind", "band"]) if len(x) > 2}))


def predictions(M, sc):
    """The registered predictions (README), each with its operational criterion."""
    print("\n== registered predictions (README; criteria fixed before the run; P1 and P2 re-registered 2026-10-18)")
    print("\n  P1  overlap and cross-faithfulness rise with the next-token effect, within kind and band")
    res = p1_tests(M)
    prim = {l: a["pooled_lo"] > 0 for l, (a, s) in res.items()}
    sec = {l: s["lo"] > 0 for l, (a, s) in res.items()}
    for l in res:
        print("      %-26s primary (pooled CI > 0): %s   secondary (slope CI > 0): %s" % (l, prim[l], sec[l]))
    verdict = ("SUPPORTED" if all(prim.values()) and all(sec.values()) else
               "MIXED (primary holds, secondary does not)" if all(prim.values()) else "NOT SUPPORTED")
    print("      P1 overall: %s" % verdict)
    xk = slope_measure(M)
    print("      descriptive only (pools kinds and bands): Spearman(%s, .) across all targets:" % xk)
    for label, k, drop in P1_QUANT:
        ms = [m for m in M if not (drop and m["out_vacuous"])]
        s = C.spearman_bootstrap(col(ms, xk), col(ms, k), B=B)
        print("        %-26s %s [%s, %s] (n %d)" % (label, f3(s["rho"]), f3(s["lo"]), f3(s["hi"]), s["n"]))
    # P2: low next-token effect targets: OUT circuits ~0 on latent metrics, LAT circuits ~0 on output faithfulness
    low = "low" if "low" in tiers_of(M) else "dp_low"
    print("\n  P2  %s targets: OUT circuits near 0 on latent Z, LAT circuits near 0 on output Z "
          "(median <= 0.2 and upper CI <= 0.35)" % C.TIER_LABEL[low])
    it = [m for m in M if m["tier"] == low]
    a = C.paired_bootstrap(col(it, "OUT_W_Z"), B=B)
    b = C.paired_bootstrap(col([m for m in it if not m["out_vacuous"]], "LAT_oZ"), B=B)
    near = lambda d: d["a"] <= 0.2 and d["a_hi"] <= 0.35  # noqa: E731
    print("      OUT_W latent Z %s (n %d) -> %s;  LAT output Z %s (n %d) -> %s  => P2 %s" % (
        ci(a), a["n"], near(a), ci(b), b["n"], near(b), "SUPPORTED" if near(a) and near(b) else "NOT SUPPORTED"))
    # P3: coefficients shrink OUT circuits too
    print("\n  P3  OUT_U needs more latents at matched pass rate and passes less at matched size")
    ms = [m for m in M if not m["out_vacuous"]]
    p = C.paired_bootstrap(col(ms, "OUT_U_opass"), col(ms, "OUT_W_opass"), stat="mean", B=B)
    ratio = [(m["OUT_U_reach"] / m["OUT_W_reach"]) if (m.get("OUT_U_reach") and m.get("OUT_W_reach")) else None
             for m in ms]
    r = C.paired_bootstrap([None if x is None else math.log10(x) for x in ratio], B=B)
    g1 = p["d_hi"] < 0
    g2 = r["a_lo"] > 0
    print("      output pass at the LAT size, OUT_U - OUT_W %s (n %d) -> %s" % (ci(p, "d"), p["n_pair"], g1))
    print("      size to reach output Z >= %.1f, OUT_U / OUT_W: median %s [%s, %s] (n %d with both) -> %s" % (
        REACH, f3(10 ** r["a"]) if r["a"] == r["a"] else "-", f3(10 ** r["a_lo"]) if r["a_lo"] == r["a_lo"] else "-",
        f3(10 ** r["a_hi"]) if r["a_hi"] == r["a_hi"] else "-", r["n"], g2))
    print("      P3 %s" % ("SUPPORTED" if g1 and g2 else "NOT SUPPORTED"))


def paper_numbers(M, sc_all):
    print("\n== numbers for the paper paragraph (medians [95% CI] over targets; OUT arms at the LAT size; a tier row "
          "pools the design's kind x band groups)")
    for t in tiers_of(M):
        ms = [m for m in M if m["tier"] == t]
        if not ms:
            continue
        mo = [m for m in ms if not m["out_vacuous"]]
        z = C.paired_bootstrap(col(ms, "OUT_W_Z"), col(ms, "LAT_Z"), B=B)
        oz = C.paired_bootstrap(col(mo, "LAT_oZ"), col(mo, "OUT_W_oZ"), B=B)
        j = C.paired_bootstrap(col(ms, "OUT_W_J"), B=B)
        lp = C.paired_bootstrap(col(ms, "OUT_W_pass"), col(ms, "LAT_pass"), stat="mean", B=B)
        op = C.paired_bootstrap(col(mo, "LAT_opass"), col(mo, "OUT_W_opass"), stat="mean", B=B)
        print("  %s (n %d): output-trained circuits reach latent Z %s (latent-trained %s); latent-trained circuits "
              "reach output Z %s (output-trained %s); Jaccard %s (refit floor %s); latent pass OUT_W %s vs LAT %s; "
              "output pass LAT %s vs OUT_W %s" % (C.TIER_LABEL.get(t, t), len(ms), ci(z, "a"), ci(z, "b"), ci(oz, "a"),
                                                   ci(oz, "b"),
                                                   ci(j), f3(sc_all), ci(lp, "a"), ci(lp, "b"), ci(op, "a"),
                                                   ci(op, "b")))
    lat_n = [m["lat_n"] for m in M if m["lat_n"]]
    import numpy as np
    for a in ("OUT_W", "OUT_U"):
        ex = [m.get(a + "_extrap") for m in M if (a + "_extrap") in m]
        if ex:
            print("  %s: LAT size outside its lambda grid's size range for %d of %d targets (values clamped to the "
                  "nearest end)" % (a, sum(bool(x) for x in ex), len(ex)))
    print("  LAT sizes: median %.0f (p10 %.0f, p90 %.0f); output-vacuous targets dropped from output comparisons: %d"
          % (np.median(lat_n), np.percentile(lat_n, 10), np.percentile(lat_n, 90), sum(m["out_vacuous"] for m in M)))


# ------------------------------------------------------------------------------------------- smoke
def smoke_listing(rows, errs, T):
    print("rows %d (errors %d) over %d targets" % (len(rows), len(errs), len({k[0] for k in rows})))
    for e in errs:
        print("  ERROR %s %s %s: %s" % (e["seed"], e["arm"], e["lam"], e["error"]))
    hdr = ("seed", "arm", "lam", "n", "Z", "A", "C", "nec", "pass", "oZ", "oA", "oC", "onec", "opass", "J", "t_fit",
           "t_score")
    print("  " + " ".join("%-8s" % h if h not in ("seed",) else "%-16s" % h for h in hdr))
    for (seed, arm, lam), r in sorted(rows.items()):
        vals = [seed, arm, lam, r.get("n"), r.get("free0_tk"), r.get("freeM_topk_tk"), r.get("freeN_topk_tk"),
                r.get("phi_sup_blind_tk"), C.passes_latent(r), r.get("out_free0"), r.get("out_freeM_topk"),
                r.get("out_freeN_topk"), r.get("out_phi_sup_blind"), C.passes_output(r), r.get("jaccard_lat"),
                r.get("t_fit"), r.get("t_score")]
        print("  " + " ".join(("%-16s" % v) if i == 0 else ("%-8s" % (f3(v) if isinstance(v, float) else v))
                              for i, v in enumerate(vals)))
    # sanity: the LAT rows reproduce 062's own eval rows (same scorer, same contexts)
    prod = {}
    for f in glob.glob(str(SRC_OUT / "main" / "eval.shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("held") == "strong" and "error" not in r and r["seed"] in T:
                prod[r["seed"]] = r
    for key, t in T.items():
        lat, p = t["lat"], prod.get(key)
        if p is None:
            continue
        d = max(abs((C.num(lat.get(f)) or 0) - (C.num(p.get(f)) or 0)) for f in C.LAT_FAITH + (C.LAT_NEC,))
        print("  check %-16s LAT re-score vs 062 eval row: n %s/%s, max |diff| over Z/A/C/nec %.4f %s" % (
            key, lat.get("n"), p.get("n"), d, "ok" if d < 0.02 and lat.get("n") == p.get("n") else "CHECK"))
        print("  check %-16s output denominators: m(full) %s  m(empty Z) %s  den %s  vacuous %s  scope sites ablated %s"
              % (key, f3(lat.get("out_full")), f3(lat.get("out_e0")), f3(lat.get("out_den")), lat.get("out_vacuous"),
                 lat.get("out_all_sites_ablated")))


def main():
    rows, errs = load()
    T = per_target(rows)
    if SMOKE:
        smoke_listing(rows, errs, T)
    M = matched(T)
    sc, sc_all = stable_core()
    print("\n%d targets with a scored LAT row | arms: %s | error rows %d | B = %d" % (
        len(M), sorted({k[1] for k in rows}), len(errs), B))
    if not M:
        return
    table_cross(M, [a for a in ("OUT_W", "OUT_U") if any(("%s_Z" % a) in m for m in M)], "LAT",
                ["Z", "A", "C", "nec", "pass"], "1. circuits on the LATENT metrics (OUT arms at the LAT size)",
                GROUP, False)
    if any("OUT_W_oZ" in m for m in M):
        table_cross(M, ["LAT"], "OUT_W", ["oZ", "oA", "oC", "onec", "opass"],
                    "2. circuits on the OUTPUT metrics (OUT_W at the LAT size; output-vacuous targets dropped)",
                    GROUP, True)
    table_pass(M)
    table_overlap(M, sc, sc_all)
    if any("OUT_U_oZ" in m for m in M):
        table_weights(M)
    print("\n== by site kind (descriptive; the per-tier rows above pool kinds, so read tier effects off the P1 "
          "within-pair contrasts)")
    table_cross(M, ["OUT_W"], "LAT", ["Z", "oZ"], "OUT_W vs LAT on latent Z and output Z", "kind", True)
    measurement_checks(M)
    if SMOKE:
        p1_tests(M)                                      # exercises the code; too few targets for a verdict
    else:
        predictions(M, sc)
    paper_numbers(M, sc_all)


if __name__ == "__main__":
    main()
