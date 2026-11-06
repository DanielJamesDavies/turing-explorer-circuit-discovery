"""100 report: E1 / E2 / E3 tables, cost projections and the E1 figure from out/main (no GPU).

  python experiments/100-scrub-frontier/report.py [--run experiments/100-scrub-frontier/out/main]
Writes <run>/tables.md, <run>/tables.json, <run>/e1_recovery_vs_size.png (+ .pdf).
"""
import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
E94 = HERE.parent / "094-tl-v2-records"
BGS = ("A", "C", "R")
SPEED = (2.0, 2.5)          # assumed H100 per-process speed-up over the local RTX 5070 Ti
PROCS, GPUS = 6, 8          # MPS co-tenants per GPU (062 pod), GPUs in the pod


def load_rows(run):
    rows = {}
    for line in open(run / "rows.jsonl"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        if "error" in r or "skip" in r:
            continue
        rows[(r["seed"], r["arm"])] = r
    return rows


def fin(v):
    return [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]


def med(v):
    v = fin(v)
    return statistics.median(v) if v else None


def q(v, p):
    v = sorted(fin(v))
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def f3(v):
    return "-" if v is None else ("%.3f" % v)


def miqr(v):
    m = med(v)
    if m is None:
        return "-"
    return "%.3f [%.3f, %.3f]" % (m, q(v, 0.25), q(v, 0.75))


def lam_of(arm):
    return float(arm.split("_")[1])


def rec(r, var, bg):
    return (r.get("recovery") or {}).get(var, {}).get(bg)


def cnt(rs, f):
    vals = [f(r) for r in rs]
    return "%d/%d" % (sum(bool(v) for v in vals), len(vals))


# ------------------------------------------------------------------------------------------- E3 helpers (raw)
def raw_for(run, key, arm):
    p = run / "targets" / ("%s.%s.json" % (key, arm))
    return json.loads(p.read_text()) if p.exists() else None


def pre_post(raw, var, bg):
    """diagnose_reads-style, pooled over draws: post recovery, raw pre ratio, empty pre ratio, normalized pre recovery."""
    vs = [d for d in raw["variants"].get(var, []) if d["background"] == bg]
    es = [d for d in raw["empty"] if d["background"] == bg]
    if not vs or not es:
        return None
    n = statistics.mean(raw["natural_raw"])
    npre = statistics.mean(raw["natural_pre_raw"])
    x, e = statistics.mean(d["mean"] for d in vs), statistics.mean(d["mean"] for d in es)
    xp, ep = statistics.mean(d["pre_mean"] for d in vs), statistics.mean(d["pre_mean"] for d in es)
    return dict(post=(x - e) / (n - e) if n - e > 1e-9 else None,
                raw_pre=xp / npre if npre > 1e-9 else None, empty_pre=ep / npre if npre > 1e-9 else None,
                norm_pre=(xp - ep) / (npre - ep) if npre - ep > 1e-9 else None)


def donor_split(raw, var, bg, flag="donor_fires_at_anchor"):
    """Pooled recovery over (draw, base) pairs whose donor context fires (or not) for the target at the base anchor."""
    vs = {d["draw"]: d for d in raw["variants"].get(var, []) if d["background"] == bg}
    out = {}
    for want in (True, False):
        xs, es, ns = [], [], []
        for e in raw["empty"]:
            if e["background"] != bg or e["draw"] not in vs:
                continue
            for i, fl in enumerate(e[flag]):
                if fl == want:
                    xs.append(vs[e["draw"]]["raw"][i]); es.append(e["raw"][i]); ns.append(raw["natural_raw"][i])
        if not xs:
            out[want] = (None, 0)
            continue
        x, ee, n = statistics.mean(xs), statistics.mean(es), statistics.mean(ns)
        out[want] = ((x - ee) / (n - ee) if n - ee > 1e-9 else None, len(xs))
    return out


# ------------------------------------------------------------------------------------------- cost model
def lstsq(X, y):
    """Ordinary least squares with numpy (small)."""
    import numpy as np
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def layer_counts(path):
    c = defaultdict(int)
    for k in path.read_text().split():
        if k:
            c[int(k.split(".")[0])] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "out" / "main")
    args = ap.parse_args()
    run = args.run
    rows = load_rows(run)
    L = []
    J = {}
    arms = sorted({a for _, a in rows})
    wcm_arms = sorted([a for a in arms if a.startswith("wcm_")], key=lambda a: -lam_of(a))
    v2cr_arms = sorted([a for a in arms if a.startswith("v2cr_")], key=lambda a: -lam_of(a))

    def sel(arm, split):
        return [r for (k, a), r in sorted(rows.items()) if a == arm and r.get("split") == split]

    # ------------------------------------------------------------------------- E1
    L += ["# 100 tables", "",
          "Development evidence on 099's small hand-picked, availability-based panel (few firing contexts per target). "
          "Recovery = 099's pooled-draw recovery; medians [IQR] across targets. Donor pass = 099's rule.", "",
          "## E1: donor-background robustness vs circuit size (development, 9 targets)", "",
          "| arm | n | median size [IQR] | prod pass | pre pass | donor A | donor C | donor R | donor pass | "
          "pinned A / C / R | random A / C / R | random prod pass | random donor pass |",
          "|---|---:|---|---:|---:|---|---|---|---:|---|---|---:|---:|"]
    J["E1"] = {}
    for arm in wcm_arms + v2cr_arms:
        rs = sel(arm, "dev")
        if not rs:
            continue
        line = "| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, len(rs), "%g [%g, %g]" % (med([r["n"] for r in rs]), q([r["n"] for r in rs], .25), q([r["n"] for r in rs], .75)),
            cnt(rs, lambda r: r["pass_rule"]), cnt(rs, lambda r: r["pass_pre"]),
            miqr([rec(r, "fit", "A") for r in rs]), miqr([rec(r, "fit", "C") for r in rs]),
            miqr([rec(r, "fit", "R") for r in rs]), cnt(rs, lambda r: r["donor_pass"].get("fit")),
            " / ".join(f3(med([rec(r, "pinned", b) for r in rs])) for b in BGS),
            " / ".join(f3(med([rec(r, "rand", b) for r in rs])) for b in BGS),
            cnt(rs, lambda r: r.get("pass_rule_rand")), cnt(rs, lambda r: r["donor_pass"].get("rand")))
        L.append(line)
        J["E1"][arm] = dict(n=len(rs), size_med=med([r["n"] for r in rs]),
                            rec={v: {b: [rec(r, v, b) for r in rs] for b in BGS} for v in ("fit", "a1", "pinned", "rand", "rand_pinned")},
                            pass_rule=sum(r["pass_rule"] for r in rs), pass_pre=sum(r["pass_pre"] for r in rs),
                            donor_pass=sum(bool(r["donor_pass"].get("fit")) for r in rs),
                            seeds=[r["seed"] for r in rs], sizes=[r["n"] for r in rs])
    L += ["", "Per target (development): size and pooled donor A / C recovery (live, fitted gains).", "",
          "| target | " + " | ".join(wcm_arms + v2cr_arms) + " |", "|---|" + "---|" * len(wcm_arms + v2cr_arms)]
    for key in sorted({k for (k, a), r in rows.items() if r.get("split") == "dev"}):
        cells = []
        for arm in wcm_arms + v2cr_arms:
            r = rows.get((key, arm))
            cells.append("-" if r is None else "%d: %s / %s%s" % (r["n"], f3(rec(r, "fit", "A")), f3(rec(r, "fit", "C")),
                                                               " P" if r["pass_rule"] else ""))
        L.append("| %s | %s |" % (key, " | ".join(cells)))
    L += ["", "P = production (post-Top-K) pass.", ""]

    # paired rule for the weaker-lambda v2CR arm
    base = "wcm_0.001" if any(a == "wcm_0.001" for a in arms) else "wcm_1e-3"
    L += ["### Pre-stated 'clearly rising' rule (development, paired vs lambda 1e-3)", "",
          "| arm | median dA | A rises | median dC | C rises | clearly rising |", "|---|---:|---:|---:|---:|---|"]
    J["rule"] = {}
    for arm in wcm_arms:
        if arm == base:
            continue
        dA, dC = [], []
        for r in sel(arm, "dev"):
            b = rows.get((r["seed"], base))
            if b is None:
                continue
            for d, bg in ((dA, "A"), (dC, "C")):
                x, y = rec(r, "fit", bg), rec(b, "fit", bg)
                d.append(None if x is None or y is None else x - y)
        ok = (med(dA) is not None and med(dC) is not None and med(dA) >= 0.05 and med(dC) >= 0.05
              and sum(1 for v in fin(dA) if v > 0) >= 6 and sum(1 for v in fin(dC) if v > 0) >= 6)
        J["rule"][arm] = dict(dA=dA, dC=dC, ok=ok)
        L.append("| %s | %s | %d/%d | %s | %d/%d | %s |" % (arm, f3(med(dA)), sum(1 for v in fin(dA) if v > 0), len(fin(dA)),
                                                         f3(med(dC)), sum(1 for v in fin(dC) if v > 0), len(fin(dC)), ok))
    L.append("")

    # ------------------------------------------------------------------------- E2
    L += ["## E2: v2 vs v2CR, paired", ""]
    J["E2"] = {}
    for v2cr in v2cr_arms:
        v2 = "wcm_" + v2cr.split("_", 1)[1]
        for split in ("dev", "check"):
            pairs = [(rows[(k, v2)], rows[(k, v2cr)]) for (k, a) in sorted(rows) if a == v2cr
                     and rows[(k, a)].get("split") == split and (k, v2) in rows]
            if not pairs:
                continue
            L += ["### %s vs %s, %s (n = %d)" % (v2, v2cr, split, len(pairs)), "",
                  "| measure | %s | %s | median paired diff |" % (v2, v2cr), "|---|---|---|---:|"]
            L.append("| production pass (post-Top-K) | %s | %s | |" % (cnt([a for a, _ in pairs], lambda r: r["pass_rule"]),
                                                                   cnt([b for _, b in pairs], lambda r: r["pass_rule"])))
            L.append("| pre-activation pass | %s | %s | |" % (cnt([a for a, _ in pairs], lambda r: r["pass_pre"]),
                                                          cnt([b for _, b in pairs], lambda r: r["pass_pre"])))
            L.append("| donor pass | %s | %s | |" % (cnt([a for a, _ in pairs], lambda r: r["donor_pass"].get("fit")),
                                                 cnt([b for _, b in pairs], lambda r: r["donor_pass"].get("fit"))))
            for var, lab in (("fit", "donor"), ("pinned", "pinned"), ("rand", "random control")):
                for bg in BGS:
                    xa = [rec(a, var, bg) for a, _ in pairs]
                    xb = [rec(b, var, bg) for _, b in pairs]
                    d = [None if x is None or y is None else y - x for x, y in zip(xa, xb)]
                    L.append("| %s %s recovery | %s | %s | %s |" % (lab, bg, miqr(xa), miqr(xb), f3(med(d))))
            for m, lab in (("free0_tk", "Z"), ("freeM_topk_tk", "A_sp"), ("freeN_topk_tk", "C_sp"), ("phi_sup_blind_tk", "necessity")):
                xa, xb = [a["prod"].get(m) for a, _ in pairs], [b["prod"].get(m) for _, b in pairs]
                d = [None if x is None or y is None else y - x for x, y in zip(xa, xb)]
                L.append("| production %s | %s | %s | %s |" % (lab, miqr(xa), miqr(xb), f3(med(d))))
            sz = [b["n"] / a["n"] for a, b in pairs if a["n"]]
            tf = [(b["fit"].get("t_fit") or 0) / a["fit"]["t_fit"] for a, b in pairs if a["fit"].get("t_fit")]
            L.append("| size (median) / ratio | %g | %g | x%.2f |" % (med([a["n"] for a, _ in pairs]), med([b["n"] for _, b in pairs]), med(sz)))
            sa = sum(a["fit"]["t_fit"] for a, _ in pairs if a["fit"].get("t_fit"))
            sb = sum(b["fit"].get("t_fit") or 0 for a, b in pairs if a["fit"].get("t_fit"))
            L.append("| fit time ratio (median per target; of sums) | | | x%.2f; x%.2f |" % (med(tf), sb / sa if sa else float("nan")))
            L += ["", "Per target: (v2 -> v2CR) pooled donor A / C, production pass.", "",
                  "| target | v2 A / C / pass | v2CR A / C / pass |", "|---|---|---|"]
            for a, b in pairs:
                L.append("| %s | %s / %s / %s | %s / %s / %s |" % (a["seed"], f3(rec(a, "fit", "A")), f3(rec(a, "fit", "C")),
                                                                 "P" if a["pass_rule"] else "-", f3(rec(b, "fit", "A")),
                                                                 f3(rec(b, "fit", "C")), "P" if b["pass_rule"] else "-"))
            L.append("")
            J["E2"]["%s|%s" % (v2cr, split)] = dict(
                seeds=[a["seed"] for a, _ in pairs], size_ratio=sz, fit_ratio=tf, fit_ratio_sums=sb / sa if sa else None,
                rec_v2={bg: [rec(a, "fit", bg) for a, _ in pairs] for bg in BGS},
                rec_v2cr={bg: [rec(b, "fit", bg) for _, b in pairs] for bg in BGS},
                pass_v2=[a["pass_rule"] for a, _ in pairs], pass_v2cr=[b["pass_rule"] for _, b in pairs])

    # ------------------------------------------------------------------------- E3
    L += ["## E3: what fails (scoring arms on the E1 / E2 circuits; all splits pooled per arm unless noted)", "",
          "### (a) fitted gains vs alpha = 1, and own-background overshoot", "",
          "| arm | split | fit A / C / R | a1 A / C / R | prod pass fit / a1 | self-donor fit/natural (median [IQR]) | "
          "self-donor a1 (identity) max abs, all positions | targets with fit recovery > 1.5 (A, C) |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in wcm_arms + v2cr_arms:
        for split in ("dev", "check"):
            rs = sel(arm, split)
            if not rs:
                continue
            selfr = [r["self_donor"]["fit"]["mean"] / r["natural_mean"] for r in rs
                     if r["self_donor"].get("fit") and r["natural_mean"] > 1e-9]
            L.append("| %s | %s | %s | %s | %s / %s | %s | %s | %d, %d |" % (
                arm, split, " / ".join(f3(med([rec(r, "fit", b) for r in rs])) for b in BGS),
                " / ".join(f3(med([rec(r, "a1", b) for r in rs])) for b in BGS),
                cnt(rs, lambda r: r["pass_rule"]), cnt(rs, lambda r: r["pass_rule_a1"]), miqr(selfr),
                "%s (anchor-mean ratio range %s-%s)" % (
                    f3(max(r["self_donor"]["a1"]["max_abs"] for r in rs if r["self_donor"].get("a1"))),
                    f3(min(r["self_donor"]["a1"]["mean"] / r["natural_mean"] for r in rs if r["self_donor"].get("a1"))),
                    f3(max(r["self_donor"]["a1"]["mean"] / r["natural_mean"] for r in rs if r["self_donor"].get("a1")))),
                sum(1 for r in rs if (rec(r, "fit", "A") or 0) > 1.5), sum(1 for r in rs if (rec(r, "fit", "C") or 0) > 1.5)))
    L += ["", "### (b) live vs pinned members (closure), and the random control pinned", "",
          "| arm | split | live A / C / R | pinned A / C / R | random live A / C / R | random pinned A / C / R |",
          "|---|---|---|---|---|---|"]
    for arm in wcm_arms + v2cr_arms:
        for split in ("dev", "check"):
            rs = sel(arm, split)
            if rs:
                L.append("| %s | %s | %s |" % (arm, split, " | ".join(
                    " / ".join(f3(med([rec(r, v, b) for r in rs])) for b in BGS)
                    for v in ("fit", "pinned", "rand", "rand_pinned"))))
    L += ["", "### (c) pre- vs post-Top-K reads per donor class (live, fitted gains; medians across targets of "
          "draw-pooled values)", "",
          "Post = pooled post-Top-K recovery; raw pre = circuit ReLU(pre) / clean; empty pre = empty circuit ReLU(pre) / "
          "clean; normalised pre = (circuit - empty) / (clean - empty) on the pre read. Lost encoder signal shows as low "
          "normalised pre; lost Top-K competition as normalised pre well above post.", "",
          "| arm | split | class | post | raw pre | empty pre | normalised pre | targets with norm pre - post > 0.2 |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
    J["E3c"] = {}
    for arm in wcm_arms + v2cr_arms:
        for split in ("dev", "check"):
            rs = sel(arm, split)
            if not rs:
                continue
            for bg in BGS:
                vals = [pre_post(raw_for(run, r["seed"], arm), "fit", bg) for r in rs]
                vals = [v for v in vals if v]
                gap = sum(1 for v in vals if v["norm_pre"] is not None and v["post"] is not None and v["norm_pre"] - v["post"] > 0.2)
                L.append("| %s | %s | %s | %s | %s | %s | %s | %d/%d |" % (
                    arm, split, bg, *(f3(med([v[k] for v in vals])) for k in ("post", "raw_pre", "empty_pre", "norm_pre")),
                    gap, len(vals)))
                J["E3c"]["%s|%s|%s" % (arm, split, bg)] = vals
    L += ["", "### (d) donor recovery split by whether the donor context fires for the target at the base anchor", "",
          "Pooled over (draw, base) pairs; share = fraction of pairs whose donor fires. C donors are verified-silent "
          "contexts.", "",
          "| arm | split | class | share firing | recovery, donor fires | recovery, donor silent |",
          "|---|---|---|---:|---:|---:|"]
    for arm in wcm_arms + v2cr_arms:
        for split in ("dev", "check"):
            rs = sel(arm, split)
            if not rs:
                continue
            for bg in BGS:
                sp = [donor_split(raw_for(run, r["seed"], arm), "fit", bg) for r in rs]
                share = [s[True][1] / max(s[True][1] + s[False][1], 1) for s in sp]
                L.append("| %s | %s | %s | %s | %s | %s |" % (arm, split, bg, f3(med(share)),
                                                             f3(med([s[True][0] for s in sp])), f3(med([s[False][0] for s in sp]))))
    L.append("")

    # ------------------------------------------------------------------------- cost
    allr = list(rows.values())
    import numpy as np
    lay = [r["layer"] for r in allr]
    shared = [r["t_donor_shared_target"] for r in allr]
    vfit = [r["t_donor"]["fit"] for r in allr]
    vdiag = [r["t_donor"]["fit"] + r["t_donor"]["a1"] + r["t_donor"]["pinned"] for r in allr]
    tprod = [r["t_prod"] for r in allr]
    ns = [r["n"] for r in allr]
    X = [[1, l, n / 1000] for l, n in zip(lay, ns)]
    b_sh = lstsq([[1, l] for l in lay], shared)
    b_v = lstsq(X, vfit)
    b_d = lstsq(X, vdiag)
    b_p = lstsq([[1, l] for l in lay], tprod)
    fit_rows = [r for r in allr if r["fit"].get("source") in ("099", "fit100") and r["fit"].get("t_fit")]
    fr_w = [r for r in fit_rows if r["arm"].startswith("wcm_")]
    b_f = lstsq([[1, r["layer"], r["n"] / 1000] for r in fr_w], [r["fit"]["t_fit"] for r in fr_w]) if len(fr_w) >= 3 else None
    v2cr_ratio = None
    for key_, val in J.get("E2", {}).items():
        if key_.startswith("v2cr_0.001") or key_.startswith("v2cr_1e-3"):
            v2cr_ratio = (v2cr_ratio or []) + val["fit_ratio"]
    v2cr_ratio = med(v2cr_ratio) if v2cr_ratio else None
    L += ["## Cost (local RTX 5070 Ti, measured) and pod projections", "",
          "Measured per circuit (seconds, medians by depth band; donor = 3 pools x 8 draws, 32 bases). "
          "'Shared' = per-target donor setup (capture A / C / R codes, donor natural reads, the empty circuit, 24 + 1 "
          "forwards); 'live' = the fitted circuit's 24 donor forwards; 'diag' = live + a1 + pinned.", "",
          "| band | n rows | median size | fit (s) | production score (s) | donor shared (s) | donor live (s) | donor diag (s) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for bname, lo, hi in (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11)):
        rb = [r for r in allr if lo <= r["layer"] <= hi]
        if not rb:
            continue
        L.append("| %s | %d | %g | %s | %s | %s | %s | %s |" % (
            bname, len(rb), med([r["n"] for r in rb]), f3(med([r["fit"].get("t_fit") for r in rb if r["fit"].get("source") in ("099", "fit100")])),
            f3(med([r["t_prod"] for r in rb])), f3(med([r["t_donor_shared_target"] for r in rb])),
            f3(med([r["t_donor"]["fit"] for r in rb])),
            f3(med([r["t_donor"]["fit"] + r["t_donor"]["a1"] + r["t_donor"]["pinned"] for r in rb]))))
    L += ["", "By size (all rows): live donor seconds by circuit-size bin.", "", "| size bin | n rows | median live (s) | median diag (s) |",
          "|---|---:|---:|---:|"]
    for lo, hi in ((0, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10 ** 9)):
        rb = [r for r in allr if lo <= r["n"] < hi]
        if rb:
            L.append("| %d-%s | %d | %s | %s |" % (lo, "inf" if hi > 10 ** 8 else hi, len(rb), f3(med([r["t_donor"]["fit"] for r in rb])),
                                               f3(med([r["t_donor"]["fit"] + r["t_donor"]["a1"] + r["t_donor"]["pinned"] for r in rb]))))
    L += ["", "Linear fits (seconds): shared = %.1f + %.2f x layer; live = %.1f + %.2f x layer + %.2f x size/1000; "
          "diag = %.1f + %.2f x layer + %.2f x size/1000; production score = %.1f + %.2f x layer%s." % (
              *b_sh, *b_v, *b_d, *b_p,
              ("; WCM fit = %.1f + %.2f x layer + %.2f x size/1000" % tuple(b_f)) if b_f is not None else ""), ""]

    def per_circuit(layer, n, diag=False, with_prod=False):
        t = b_sh[0] + b_sh[1] * layer + ((b_d if diag else b_v) @ np.array([1, layer, n / 1000]))
        if with_prod:
            t += b_p[0] + b_p[1] * layer
        return float(t)

    lc8k = layer_counts(E94 / "targets_tl8k.txt") if (E94 / "targets_tl8k.txt").exists() else {l: 8000 / 12 for l in range(12)}
    lc190 = layer_counts(E94 / "targets_tl8k_sweep190.txt") if (E94 / "targets_tl8k_sweep190.txt").exists() else {l: 190 / 12 for l in range(12)}
    # sizes: median size per layer band at lambda 1e-3 (all splits), and per lambda for the sweep
    size_by = {}
    for arm in wcm_arms:
        rs = [r for r in allr if r["arm"] == arm]
        size_by[arm] = med([r["n"] for r in rs])
    n_prod = size_by.get(base) or 500
    sweep_sizes = [size_by[a] for a in wcm_arms if size_by.get(a)]

    def total(lc, n_list, diag, with_prod=False, extra=None):
        tot = 0.0
        for l, c in lc.items():
            for n in n_list:
                tot += c / len(n_list) * per_circuit(l, n, diag, with_prod) * len(n_list) / len(n_list)
        return tot

    def wall(secs):
        return " / ".join("%.1f h" % (secs / s / (PROCS * GPUS) / 3600) for s in SPEED)

    def gpuh(secs):
        return " / ".join("%.1f" % (secs / s / PROCS / 3600) for s in SPEED)

    L += ["### Pod projections", "",
          "Assumption: an H100 process runs %.1f-%.1fx faster than this RTX 5070 Ti process, and %d processes share each GPU "
          "under MPS (062 pod) without slowing each other beyond that factor; %d GPUs. Wall = total local seconds / speed-up / "
          "(%d x %d). Layer mix from 094's target lists (8k: `targets_tl8k.txt`; sweep: `targets_tl8k_sweep190.txt`); size at "
          "lambda 1e-3 = this panel's median (%g). Sizes and per-layer costs come from 18 targets at 3 depths: a rough "
          "linear model, not a benchmark. Values: wall-clock hours at %sx / %sx; GPU-hours in brackets." % (
              SPEED[0], SPEED[1], PROCS, GPUS, PROCS, GPUS, n_prod, SPEED[0], SPEED[1]), "",
          "| stage | circuits | local hours (1 process) | pod wall h (%.1fx / %.1fx) | pod GPU-h |" % SPEED, "|---|---:|---:|---|---|"]
    J["cost"] = {}

    def add(name, nc, secs):
        L.append("| %s | %d | %.1f | %s | %s |" % (name, nc, secs / 3600, wall(secs), gpuh(secs)))
        J["cost"][name] = dict(circuits=nc, local_secs=secs)
    tot8k = sum(lc8k.values())
    for diag in (False, True):
        secs = sum(c * per_circuit(l, n_prod, diag) for l, c in lc8k.items())
        lab = "diag (live + a1 + pinned)" if diag else "live only"
        add("(a) donor scoring, 8k main, %s" % lab, int(tot8k), secs)
        for sub in (1000, 2000):
            add("(a) donor scoring, %dk stratified subset, %s" % (sub // 1000, lab), sub, secs * sub / tot8k)
    tot190 = sum(lc190.values())
    if sweep_sizes:
        # 8 WCM lambda values: use the measured E1 sizes (4 lambda) as the size distribution of the sweep's 8
        secs = sum(c * statistics.mean(per_circuit(l, n, False) for n in sweep_sizes) for l, c in lc190.items()) * 8
        add("(b) donor scoring, 190-target sweep x 8 WCM lambda, live only (sizes = E1's lambda grid)", int(tot190 * 8), secs)
        secs_d = sum(c * statistics.mean(per_circuit(l, n, True) for n in sweep_sizes) for l, c in lc190.items()) * 8
        add("(b) donor scoring, 190-target sweep x 8 WCM lambda, diag", int(tot190 * 8), secs_d)
    if b_f is not None:
        ratio = v2cr_ratio or 1.45
        fit_s = sum(c * float(b_f @ np.array([1, l, n_prod / 1000])) for l, c in lc190.items()) * ratio
        sc_s = sum(c * per_circuit(l, n_prod, False, with_prod=True) for l, c in lc190.items())
        add("(c) v2CR fits on the 190 sweep targets at lambda 1e-3 (fit x%.2f of v2)" % ratio, int(tot190), fit_s)
        add("(c) ... plus production + live donor scoring of those circuits", int(tot190), fit_s + sc_s)
        J["cost"]["v2cr_fit_ratio_used"] = ratio
    L.append("")
    (run / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (run / "tables.json").write_text(json.dumps(J, indent=1, default=str) + "\n")
    print("\n".join(L))

    # ------------------------------------------------------------------------- figure
    sys.path.insert(0, str(ROOT / "src"))
    from analysis import style as S
    plt = S.configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=S.FIGSIZE_WIDE, sharey=True)
    dev = sorted({k for (k, a), r in rows.items() if r.get("split") == "dev"})
    for ax, bg in zip(axes, ("A", "C")):
        ax.axhspan(0.8, 1.5, color=S.GRID, alpha=0.6, lw=0, zorder=0)
        for key in dev:
            pts = [(rows[(key, a)]["n"], rec(rows[(key, a)], "fit", bg)) for a in wcm_arms if (key, a) in rows]
            pts = [(x, y) for x, y in pts if y is not None]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color=S.BLUE_LIGHT, lw=0.8, alpha=0.8, zorder=1)
        series = [("WCM, live", "fit", S.CATEGORICAL[0], "o"), ("random control (fitted gains)", "rand", S.CATEGORICAL[1], "s"),
                  ("WCM, pinned members", "pinned", S.CATEGORICAL[2], "^")]
        for lab, var, colr, mk in series:
            xs, ys, lo, hi = [], [], [], []
            for a in wcm_arms:
                rs = sel(a, "dev")
                vals = [rec(r, var, bg) for r in rs]
                if not fin(vals):
                    continue
                xs.append(med([r["n"] if not var.startswith("rand") else r["n_rand"] for r in rs]))
                ys.append(med(vals)); lo.append(med(vals) - q(vals, .25)); hi.append(q(vals, .75) - med(vals))
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], color=colr, marker=mk, ms=6, lw=1.6, capsize=3, label=lab, zorder=3)
        for a in v2cr_arms:
            rs = sel(a, "dev")
            vals = [rec(r, "fit", bg) for r in rs]
            if fin(vals):
                ax.errorbar([med([r["n"] for r in rs])], [med(vals)], yerr=[[med(vals) - q(vals, .25)], [q(vals, .75) - med(vals)]],
                            color=S.CATEGORICAL[3], marker="D", ms=7, lw=1.6, capsize=3,
                            label="v2CR, live (lambda %g)" % lam_of(a), zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("circuit size (members)")
        ax.set_title("%s donors (%s)" % (bg, "training activating" if bg == "A" else "held-out silent"))
    axes[0].set_ylabel("pooled donor recovery")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=S.CATEGORICAL[0], marker="o", lw=1.6, label="WCM v2, live (lambda 1e-3 ... 1e-5)"),
               Line2D([], [], color=S.CATEGORICAL[1], marker="s", lw=1.6, label="random control, fitted gains"),
               Line2D([], [], color=S.CATEGORICAL[2], marker="^", lw=1.6, label="WCM v2, pinned members"),
               Line2D([], [], color=S.CATEGORICAL[3], marker="D", lw=0, label="v2CR, live (lambda 1e-3)"),
               Line2D([], [], color=S.BLUE_LIGHT, lw=0.8, label="single target, WCM v2 live")]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, frameon=False)
    for ax in axes:
        ax.text(0.02, 1.47, "donor-pass band [0.8, 1.5]", transform=ax.get_yaxis_transform(), fontsize=8,
                color=S.INK_MUTED, va="top")
    S.style_suptitle(fig, "Donor-background recovery vs circuit size (9 development targets; median, IQR)")
    S.save_figure(fig, run / "e1_recovery_vs_size.png")


if __name__ == "__main__":
    main()
