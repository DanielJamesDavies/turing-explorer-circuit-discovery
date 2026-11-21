"""127 analysis (CPU only): reads out/<stage>/rows.jsonl written by run127.py, merges arms per target, writes
out/report.md (and out/pass_<set>.csv).

  python -I -X utf8 experiments/127-norm-rescale/analyze.py [--out-dir experiments/127-norm-rescale/out]

Pass rule: 062 merge (Z, A, C in [0.75, 1.5], necessity >= 0.9); necessity and vacuity from 118's full production
score (alpha-independent; the home rows use the ho necessity). Vacuous targets (118's rule) are dropped. Every
comparison is paired on the same targets. Intervals: 95% bootstrap over targets (4,000 resamples).
Also paired: 118 job 3's refitted per-site / per-circuit scalars (118 out/fit_tk/rows.jsonl, ho only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
from merge import BAND, NEC_MIN  # noqa: E402

E118 = ROOT / "experiments" / "118-coefficient-diagnostics"
B_BOOT, SEED = 4000, 20261117
ORDER = ("fitted", "alpha1", "oracle", "oracle_lat", "rescale_site", "118_fit_site", "118_fit_global")
LAB = {"fitted": "fitted alpha (production)", "alpha1": "alpha = 1", "oracle": "oracle norm match (per row, position)",
       "oracle_lat": "oracle norm match, bias kept", "rescale_site": "site scalar, median train ratio (reference)",
       "118_fit_site": "118 job 3: refitted per-site scalar", "118_fit_global": "118 job 3: refitted per-circuit scalar"}
LINES = []
OUT_DIR = HERE / "out"


def say(s=""):
    print(s)
    LINES.append(s)


def band_of(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def ok3(d, nec):
    if d is None or nec is None:
        return False
    v = [d.get(f) for f in "ZAC"]
    return all(x is not None and BAND[0] <= x <= BAND[1] for x in v) and nec >= NEC_MIN


def boot(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def vac(r):
    ap = r.get("full_a_pos_tk")
    if ap is None or ap <= 0:
        return True
    return any((r.get("full_" + k) or 0) > 0.95 * ap for k in ("e0_tk", "eM_topk_tk", "eN_topk_tk"))


def load(out_dir, stages=None):
    T = {}
    for p in sorted(out_dir.glob("*/rows.jsonl")):
        if (stages is None and p.parent.name.startswith("cpu_test")) or (stages and p.parent.name not in stages):
            continue
        for line in open(p):
            r = json.loads(line)
            if "error" in r:
                continue
            t = T.setdefault(r["seed"], dict(seed=r["seed"], arms={}, diag={}, meta={}))
            t["arms"].update(r.get("arms", {}))
            t["diag"].update(r.get("diag", {}))
            for k, v in r.items():
                if k not in ("arms", "diag"):
                    t["meta"][k] = v
    p = E118 / "out" / "fit_tk" / "rows.jsonl"
    if p.exists():
        for line in open(p):
            r = json.loads(line)
            if "error" in r or r["seed"] not in T:
                continue
            for a, nm in (("fit_site", "118_fit_site"), ("fit_global", "118_fit_global")):
                if a in r.get("fit", {}):
                    T[r["seed"]]["arms"][nm] = {"ho": r["fit"][a]["red"]}
    return T


def pass_section(T, keys, which, title):
    arms = [a for a in ORDER if any(which in T[k]["arms"].get(a, {}) for k in keys)]
    if "fitted" not in arms:
        return
    say("### %s" % title)
    say()
    # pair on the targets that have every shown arm (118 refit arms: report their own paired n)
    P = {}
    for a in arms:
        P[a] = {k: ok3(T[k]["arms"][a][which], T[k]["meta"]["full_phi_sup_blind_tk"])
                for k in keys if which in T[k]["arms"].get(a, {})}
    say("| arm | n | pass % [95% CI] | Δ vs fitted, pp [95% CI] (paired) | share of the α = 1 → fitted gap | "
        "median Z / A / C |")
    say("|---|---|---|---|---|---|")
    pf = P["fitted"]
    pa1 = P.get("alpha1", {})
    for a in arms:
        ks = [k for k in P[a] if k in pf]
        x = np.array([P[a][k] for k in ks], dtype=float)
        f = np.array([pf[k] for k in ks], dtype=float)
        lo, hi = boot(x)
        dlo, dhi = boot(x - f)
        g = ""
        if a != "fitted" and pa1:
            k2 = [k for k in ks if k in pa1]
            a1 = np.array([pa1[k] for k in k2], dtype=float)
            fx = np.array([pf[k] for k in k2], dtype=float)
            xx = np.array([P[a][k] for k in k2], dtype=float)
            if fx.mean() > a1.mean():
                g = "%.0f%%" % (100 * (xx.mean() - a1.mean()) / (fx.mean() - a1.mean()))
        med = []
        for fl in "ZAC":
            v = [T[k]["arms"][a][which].get(fl) for k in ks]
            v = [np.nan if z is None else z for z in v]
            med.append(np.nanmedian(v) if len(v) else np.nan)
        say("| %s | %d | %.1f [%.1f, %.1f] | %s | %s | %.2f / %.2f / %.2f |" % (
            LAB[a], len(ks), 100 * x.mean(), 100 * lo, 100 * hi,
            "" if a == "fitted" else "%+.1f [%+.1f, %+.1f]" % (100 * (x - f).mean(), 100 * dlo, 100 * dhi), g, *med))
    say()
    # by band
    bands = sorted({band_of(int(k.split(".")[0])) for k in keys}, key=lambda b: int(b[1:].split("-")[0]))
    say("By depth band (pass %; n):")
    say()
    say("| arm | " + " | ".join(bands) + " |")
    say("|---|" + "---|" * len(bands))
    for a in arms:
        cells = []
        for b in bands:
            ks = [k for k in P[a] if band_of(int(k.split(".")[0])) == b]
            cells.append("%.0f (%d)" % (100 * np.mean([P[a][k] for k in ks]), len(ks)) if ks else "-")
        say("| %s | " % LAB[a] + " | ".join(cells) + " |")
    say()
    # failure side
    for a in arms:
        if a == "fitted":
            continue
        low = high = 0
        for k, p in P[a].items():
            if p:
                continue
            d = T[k]["arms"][a][which]
            v = [d.get(f) for f in "ZAC"]
            low += any(z is None or z < BAND[0] for z in v)
            high += any(z is not None and z > BAND[1] for z in v)
        say("- %s failures: %d with a score below 0.75 (or undefined), %d with a score above 1.5." % (LAB[a], low, high))
    say()
    with open(OUT_DIR / ("pass_%s.csv" % which), "w") as fh:
        fh.write("seed," + ",".join(arms) + "\n")
        for k in keys:
            fh.write(k + "," + ",".join("" if k not in P[a] else str(int(P[a][k])) for a in arms) + "\n")


def diag_section(T, keys):
    arms = [a for a in ("alpha1", "rescale_site", "oracle", "oracle_lat", "fitted")
            if any(a in T[k]["diag"] for k in keys)]
    if not arms:
        return
    say("## 2. Downstream reads (held-out rows, anchor): RMSNorm inputs the circuit feeds, vs the natural run")
    say()
    say("Per target: median over the 16 held-out rows; table: median over targets (p10 over targets in brackets for "
        "the cosine). 'own read' = the norm whose output the target's site reads (attn: norm_1 of its layer; mlp, "
        "resid: norm_2); 'all points' = the mean over every norm input after the earliest member site and before "
        "the target; 'target SAE input' = x at the seed site. Recon ratio = ||xhat|| / ||xhat_nat|| at member sites "
        "(118 §4: α = 1 0.53, fitted 0.91 in the Z run).")
    say()
    for fl in "ZAC":
        say("**Fill %s**" % fl)
        say()
        say("| arm | n | own read: cos post | own read: post / pre norm ratio | all points: cos post | "
            "target SAE input: cos / norm ratio | member-site recon ratio, median (p90) |")
        say("|---|---|---|---|---|---|---|")
        for a in arms:
            c_own, r_post, r_pre, c_all, sx_c, sx_r, rr, rr90 = ([] for _ in range(8))
            for k in keys:
                d = T[k]["diag"].get(a, {}).get(fl)
                if not d:
                    continue
                if d.get("own"):
                    c_own.append(d["own"]["cos"])
                    r_post.append(d["own"]["post_ratio"])
                    r_pre.append(d["own"]["pre_ratio"])
                if d.get("ln"):
                    c_all.append(np.mean([v["cos"] for v in d["ln"].values()]))
                if d.get("seed_x"):
                    sx_c.append(d["seed_x"]["cos"])
                    sx_r.append(d["seed_x"]["ratio"])
                if d.get("recon_ratio_med") is not None:
                    rr.append(d["recon_ratio_med"])
                    rr90.append(d["recon_ratio_p90"])

            def m(v, p=50):
                return "%.3f" % np.percentile(v, p) if v else "-"
            say("| %s | %d | %s (%s) | %s / %s | %s | %s / %s | %s (%s) |" % (
                LAB[a], len(sx_c), m(c_own), m(c_own, 10), m(r_post), m(r_pre), m(c_all), m(sx_c), m(sx_r),
                m(rr), m(rr90)))
        say()
    # paired cosine gaps
    for a in arms:
        if a == "fitted":
            continue
        d = []
        for k in keys:
            x = T[k]["diag"].get(a, {}).get("Z", {}).get("seed_x")
            y = T[k]["diag"].get("fitted", {}).get("Z", {}).get("seed_x")
            if x and y:
                d.append(x["cos"] - y["cos"])
        if d:
            lo, hi = boot(d)
            say("- Z run, target SAE input cosine, %s - fitted: mean %+.4f [%+.4f, %+.4f] (n = %d)." % (
                LAB[a], np.mean(d), lo, hi, len(d)))
    say()
    sc = [T[k]["diag"]["oracle"]["Z"]["scale"] for k in keys if "oracle" in T[k]["diag"] and
          T[k]["diag"]["oracle"].get("Z", {}).get("scale")]
    if sc:
        say("Oracle scale at the anchor (Z run; medians over targets): median %.2f, p10 %.2f, p90 %.2f, p99 %.2f, "
            "max (median over targets) %.1f." % tuple(np.median([s[x] for s in sc]) for x in ("med", "p10", "p90", "p99", "max")))
    st = [T[k]["meta"]["site_scale_stats"] for k in keys if "site_scale_stats" in T[k]["meta"]]
    if st:
        say("rescale_site scalars (medians over targets): median %.2f, p10 %.2f, p90 %.2f; calibration change per pass "
            "(median |log ratio| over sites): %s." % (
                np.median([s["med"] for s in st]), np.median([s["p10"] for s in st]), np.median([s["p90"] for s in st]),
                " -> ".join("%.3f" % np.median([s["iter_change"][i] for s in st]) for i in range(len(st[0]["iter_change"])))))
    bd = [T[k]["meta"].get("bdec_share_med") for k in keys if T[k]["meta"].get("bdec_share_med") is not None]
    if bd:
        say("Decoder bias ||b|| / natural ||xhat|| at member sites: median %.2f (p10 %.2f, p90 %.2f over targets)." % (
            np.median(bd), np.percentile(bd, 10), np.percentile(bd, 90)))
    say()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "out"))
    ap.add_argument("--stages", default=None, help="comma list of out/<stage> dirs to read (default: all but cpu_test*)")
    ap.add_argument("--report", default=None, help="report path (default OUT_DIR/report.md)")
    a = ap.parse_args()
    global OUT_DIR
    out_dir = OUT_DIR = Path(a.out_dir)
    T = load(out_dir, a.stages.split(",") if a.stages else None)
    if not T:
        sys.exit("no rows under %s" % out_dir)
    keys_all = sorted(T)
    keys = [k for k in keys_all if not vac(T[k]["meta"])]
    k54 = set((E118 / "targets_fit54.txt").read_text().split())
    say("# 127 results (auto-generated by analyze.py)")
    say()
    chk = [T[k]["meta"].get("check_vs_118", {}) for k in keys]
    say("Targets: %d scored, %d non-vacuous (%d of them in 118 job 3's 54). Reproduction of 118 (held-out Z / A / C, "
        "max |diff| over targets): alpha1 %s, fitted %s. Devices: %s." % (
            len(keys_all), len(keys), len([k for k in keys if k in k54]),
            "%.2g" % max([c["alpha1"] for c in chk if "alpha1" in c], default=float("nan")),
            "%.2g" % max([c["fitted"] for c in chk if "fitted" in c], default=float("nan")),
            ", ".join(sorted({T[k]["meta"].get("device", "?") for k in keys}))))
    say("xdoc: not available (TuringLLM records carry no per-window document split of the held-out rows).")
    say()
    say("## 1. Pass rates (062 merge rule)")
    say()
    sets = [("job-3 targets (118's 54; pairs with 118 job 3)", [k for k in keys if k in k54])]
    rest = [k for k in keys if k not in k54]
    if rest:
        sets.append(("all scored targets", keys))
    for nm, ks in sets:
        if not ks:
            continue
        pass_section(T, ks, "ho", "%s, held-out (ho, 16 rows)" % nm)
        pass_section(T, ks, "home", "%s, training rows (home, 48 rows; ho necessity)" % nm)
    diag_section(T, keys)
    Path(a.report or out_dir / "report.md").write_text("\n".join(LINES) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
