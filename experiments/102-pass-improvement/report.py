"""102 phase-1 tables -> out/phase1/tables.md (CPU only; recomputes every pass flag from the raw values).

Per arm: production / resample / pre-activation pass, min(production, resample), paired difference vs base (paired
bootstrap 95% CI, 10,000 target resamples, seed 20261031; wins / losses), median size (and median per-target ratio vs
base), fit time (median per-target ratio vs 095's base timing and vs this run's seed1), all and by depth band. Then the
seed-variance table, best3 (training-criterion pick, defined in README before any result) and the pre-stated selection
rule applied to the arms present.

  python experiments/102-pass-improvement/report.py [--out experiments/102-pass-improvement/out/phase1] [--floor 0.75]
"""
import argparse
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
E95 = HERE.parent / "095-v2-resample-appendix"
BANDS = ("L0-4", "L5-7", "L8-11")
ARM_ORDER = ("base", "seed1", "seed2", "best3", "rk3e-2", "rk1e-1", "r1.0", "r1.0_rk3e-2", "r1.0_rk1e-1", "lamdepth",
             "twostage", "r0.5")
# flagged by diag-102 (coordinator, 2026-10-31): an undefined term with vacuous_tk False (counted as failures in every
# arm, as the pass functions do for any undefined read), and a target with only 9 strongest contexts
FLAGS = {"11.attn.12206": "undefined term, vacuous_tk False", "2.attn.16773": "undefined term, vacuous_tk False",
         "5.resid.35667": "undefined term, vacuous_tk False", "7.attn.38584": "only 9 strongest contexts"}
NEC = 0.9
CEIL = 1.5
BOOT, BOOT_SEED = 10000, 20261031


def load_jsonl(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def ib(v, lo):
    return v is not None and v == v and lo <= v <= CEIL


def passes(r, lo):
    p = r["prod"]
    prod = bool(ib(p.get("free0_tk"), lo) and ib(p.get("freeM_topk_tk"), lo) and ib(p.get("freeN_topk_tk"), lo)
                and (p.get("phi_sup_blind_tk") is not None and p["phi_sup_blind_tk"] >= NEC))
    rs = bool(prod and ib(r.get("rs_A"), lo) and ib(r.get("rs_C"), lo))
    ap, x, xr = p.get("a_pos_pre"), p.get("free0_raw_pre"), p.get("phi_sup_blind_raw_pre")
    if ap is None or x is None or xr is None or ap <= 1e-9:
        pre = False
    else:
        pre = bool(ib(x / ap, lo) and ib(p.get("freeM_topk_pre"), lo) and ib(p.get("freeN_topk_pre"), lo)
                   and (ap - xr) / ap >= NEC)
    return dict(prod=prod, rs=rs, pre=pre)


def boot_ci(a, b):
    """Paired bootstrap 95% CI of mean(a) - mean(b) over targets (lists of 0/1, same order)."""
    n = len(a)
    if n == 0:
        return None, None
    rng = random.Random(BOOT_SEED)
    d = [x - y for x, y in zip(a, b)]
    vals = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return vals[int(0.025 * BOOT)], vals[int(0.975 * BOOT) - 1]


def train_criterion(train_row):
    """README best3 criterion: mean total training loss over the logged curve points at steps >= 300."""
    if not train_row:
        return None
    steps, loss = train_row.get("curve_steps"), (train_row.get("curve") or {}).get("loss")
    if not steps or not loss:
        return None
    v = [l for s, l in zip(steps, loss) if s >= 300]
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "out" / "phase1")
    ap.add_argument("--floor", type=float, default=0.75)
    args = ap.parse_args()
    lo = args.floor
    rows = {}
    for r in load_jsonl(args.out / "rows.jsonl"):
        if "error" in r or "prod" not in r or r.get("vacuous_tk"):
            continue
        rows[(r["seed"], r["arm"])] = r                        # last row wins
    base = {k: r for (k, a), r in rows.items() if a == "base"}
    keys = sorted(base, key=lambda k: base[k]["order"])
    arms = {}
    for a in ARM_ORDER:
        if a in ("best3",):
            continue
        d = {k: r for (k, a_), r in rows.items() if a_ == a}
        if a == "lamdepth" and d:
            for k in keys:
                if base[k]["band"] == "L0-4":
                    d[k] = dict(base[k], arm="lamdepth", alias_of_base=True)
        if d:
            arms[a] = d
    # ---- best3 (training criterion, README)
    tr95 = {}
    for t in load_jsonl(E95 / "out" / "train.jsonl"):
        if t.get("arm95") == "v2CR":
            tr95[t["seed"]] = t
    tr = {}
    for t in load_jsonl(args.out / "train.jsonl"):
        if t.get("arm102") in ("seed1", "seed2") and not t.get("check"):
            tr[(t["seed"], t["arm102"])] = t
    pick = {}
    if "seed1" in arms and "seed2" in arms:
        b3 = {}
        for k in keys:
            if k not in arms["seed1"] or k not in arms["seed2"]:
                continue
            crit = [train_criterion(tr95.get(k)), train_criterion(tr.get((k, "seed1"))), train_criterion(tr.get((k, "seed2")))]
            if any(c is None for c in crit):
                continue
            s = min(range(3), key=lambda i: (crit[i], i))
            pick[k] = (s, crit)
            src = (base, arms["seed1"], arms["seed2"])[s]
            b3[k] = dict(src[k], arm="best3", best3_seed=s)
        if b3:
            arms["best3"] = b3
    ordered = [a for a in ARM_ORDER if a in arms]

    t95 = {k: (base[k].get("fit") or {}).get("t_fit") for k in keys}
    s1 = arms.get("seed1", {})

    def tf(r):
        t = (r.get("fit") or {}).get("t_fit")
        if r.get("arm") == "twostage" and t is not None:
            t = t + (t95.get(r["seed"]) or 0)                 # stage 2 + the base fit it starts from
        return t

    L = []
    title = ("phase 3: check panel (the frozen 72, seed 20261031)" if args.out.name == "check"
             else "phase 1: arms on dev (095's 72 targets)")
    L.append("# 102 %s, floor %.2f\n" % (title, lo))
    L.append("Generated by `report.py` from `out/%s/rows.jsonl` (every flag recomputed from raw values). " % args.out.name
             + ("On check, base (v2CR) is FITTED in this run, so 'x095' below means x this run's base fit. "
                if args.out.name == "check" else "") + "Production"
             " pass = Z, A_sp, C_sp in [%.2f, 1.5] and necessity >= 0.9 (held-out strongest, post-Top-K); resample pass"
             " = production AND rs_A, rs_C in [%.2f, 1.5] (095's donor scorer); pre = 095's pre-activation pass."
             " Paired = targets present in both the arm and base. CI = paired bootstrap 95%% (10,000, seed %d)."
             " wins / losses = targets the arm passes and base fails / the reverse. Size ratio = median per-target n_arm / n_base."
             " Fit time x095 = median per-target t_fit / 095's base t_fit (26 Oct, GPU unshared);"
             " x seed1 = vs this run's seed1 refit of the same algorithm (same GPU conditions). twostage fit time includes the"
             " base fit it starts from. lamdepth L0-4 rows = base rows (not refitted).\n" % (lo, lo, BOOT_SEED))

    def summary(sub_keys, a):
        d = arms[a]
        ks = [k for k in sub_keys if k in d and k in base]
        if not ks:
            return None
        P = {k: passes(d[k], lo) for k in ks}
        B = {k: passes(base[k], lo) for k in ks}
        n = len(ks)
        out = dict(n=n)
        for m in ("prod", "rs", "pre"):
            av = [int(P[k][m]) for k in ks]
            bv = [int(B[k][m]) for k in ks]
            out[m] = sum(av)
            out[m + "_b"] = sum(bv)
            out[m + "_ci"] = boot_ci(av, bv)
            out[m + "_wl"] = (sum(1 for x, y in zip(av, bv) if x and not y), sum(1 for x, y in zip(av, bv) if y and not x))
        out["min"] = min(out["prod"], out["rs"])
        out["min_b"] = min(out["prod_b"], out["rs_b"])
        nn = [d[k]["n"] for k in ks]
        out["size_med"] = st.median(nn)
        rr = [d[k]["n"] / base[k]["n"] for k in ks if base[k].get("n")]
        out["size_ratio"] = st.median(rr) if rr else None
        r95 = [tf(d[k]) / t95[k] for k in ks if tf(d[k]) and t95.get(k)]
        out["t95"] = st.median(r95) if r95 else None
        rs1 = [tf(d[k]) / s1[k]["fit"]["t_fit"] for k in ks if tf(d[k]) and k in s1 and (s1[k].get("fit") or {}).get("t_fit")]
        out["ts1"] = st.median(rs1) if rs1 else None
        return out

    def pct(x, n):
        return "%d/%d (%.0f%%)" % (x, n, 100.0 * x / n) if n else "-"

    def ci(c):
        return "-" if c[0] is None else "[%+.0f, %+.0f]" % (100 * c[0], 100 * c[1])

    def f2(v, fmt="%.2f"):
        return "-" if v is None else fmt % v

    L.append("## 1. Per arm, all bands\n")
    L.append("| arm | n | production | resample | pre-act | min(prod, rs) | d prod vs base, pp [CI] (w/l) | d rs vs base [CI] (w/l) | d pre [CI] (w/l) | median size (x base) | fit time x095 / x seed1 |")
    L.append("|---|---:|---|---|---|---:|---|---|---|---|---|")
    S = {}
    for a in ordered:
        s = summary(keys, a)
        if s is None:
            continue
        S[a] = s
        n = s["n"]
        L.append("| %s | %d | %s | %s | %s | %d | %+.0f %s (%d/%d) | %+.0f %s (%d/%d) | %+.0f %s (%d/%d) | %s (x%s) | %s / %s |" % (
            a, n, pct(s["prod"], n), pct(s["rs"], n), pct(s["pre"], n), s["min"],
            100.0 * (s["prod"] - s["prod_b"]) / n, ci(s["prod_ci"]), *s["prod_wl"],
            100.0 * (s["rs"] - s["rs_b"]) / n, ci(s["rs_ci"]), *s["rs_wl"],
            100.0 * (s["pre"] - s["pre_b"]) / n, ci(s["pre_ci"]), *s["pre_wl"],
            f2(s["size_med"], "%.0f"), f2(s["size_ratio"]), f2(s["t95"]), f2(s["ts1"])))
    L.append("")
    L.append("## 2. By depth band (production / resample / pre-act pass; d prod, d rs vs base on the same targets)\n")
    L.append("| arm | " + " | ".join("%s prod / rs / pre (d prod, d rs)" % b for b in BANDS) + " |")
    L.append("|---|" + "---|" * len(BANDS))
    for a in ordered:
        cells = []
        for b in BANDS:
            s = summary([k for k in keys if base[k]["band"] == b], a)
            if s is None:
                cells.append("-")
                continue
            cells.append("%d / %d / %d of %d (%+d, %+d)" % (s["prod"], s["rs"], s["pre"], s["n"], s["prod"] - s["prod_b"],
                                                            s["rs"] - s["rs_b"]))
        L.append("| %s | %s |" % (a, " | ".join(cells)))
    L.append("")
    # ---- clean-rank stratum (diag-102: rank_clean >= 32 almost never passes); rank_clean from 062 out_full targets.csv
    import csv
    rk = {}
    with open(HERE.parent / "062-h100-protocol-v1" / "out_full" / "out" / "targets.csv") as fcsv:
        for r in csv.DictReader(fcsv):
            try:
                rk[r["seed"]] = float(r["rank_clean"])
            except (TypeError, ValueError):
                pass
    L.append("## 2b. By the target's clean rank at its site (062 targets.csv `rank_clean`; diag-102's stratum)\n")
    L.append("| arm | rank_clean < 32: prod / rs / pre (d prod, d rs) | rank_clean >= 32: prod / rs / pre (d prod, d rs) | rank unknown |")
    L.append("|---|---|---|---|")
    for a in ordered:
        cells = []
        for sel in (lambda k: k in rk and rk[k] < 32, lambda k: k in rk and rk[k] >= 32, lambda k: k not in rk):
            s = summary([k for k in keys if sel(k)], a)
            cells.append("-" if s is None else "%d / %d / %d of %d (%+d, %+d)" % (
                s["prod"], s["rs"], s["pre"], s["n"], s["prod"] - s["prod_b"], s["rs"] - s["rs_b"]))
        L.append("| %s | %s |" % (a, " | ".join(cells)))
    L.append("")
    L.append("## 3. Medians of the reads (all targets present)\n")
    L.append("| arm | Z | A_sp | C_sp | necessity | rs_A | rs_C | rs_R | alpha median | alpha p90 | share alpha > 2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ordered:
        d = arms[a]
        ks = [k for k in keys if k in d]

        def med(fn):
            v = [fn(d[k]) for k in ks]
            v = [x for x in v if x is not None]
            return f2(st.median(v)) if v else "-"
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a, med(lambda r: r["prod"].get("free0_tk")), med(lambda r: r["prod"].get("freeM_topk_tk")),
            med(lambda r: r["prod"].get("freeN_topk_tk")), med(lambda r: r["prod"].get("phi_sup_blind_tk")),
            med(lambda r: r.get("rs_A")), med(lambda r: r.get("rs_C")), med(lambda r: r.get("rs_R")),
            med(lambda r: (r.get("alpha") or {}).get("alpha_med")), med(lambda r: (r.get("alpha") or {}).get("alpha_p90")),
            med(lambda r: (r.get("alpha") or {}).get("share_gt2"))))
    L.append("")
    # ---- seed variance
    if "seed1" in arms:
        L.append("## 4. Seed variance (R donor generator seed 20261017 + s; the fit's only randomness)\n")
        seeds = [("base", base)] + [(a, arms[a]) for a in ("seed1", "seed2") if a in arms]
        ks = [k for k in keys if all(k in d for _, d in seeds)]
        L.append("Targets with all %d seeds: %d.\n" % (len(seeds), len(ks)))
        L.append("| measure | " + " | ".join(a for a, _ in seeds) + " | flips (targets whose flag differs across seeds) |")
        L.append("|---|" + "---|" * (len(seeds) + 1))
        for m, lab in (("prod", "production pass"), ("rs", "resample pass"), ("pre", "pre-act pass")):
            vals = {a: [passes(d[k], lo)[m] for k in ks] for a, d in seeds}
            flips = sum(1 for i in range(len(ks)) if len({vals[a][i] for a, _ in seeds}) > 1)
            L.append("| %s | %s | %d/%d |" % (lab, " | ".join("%d" % sum(vals[a]) for a, _ in seeds), flips, len(ks)))
        pairs = [(seeds[i][0], seeds[j][0]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
        L.append("")
        L.append("| pair | prod flips | rs flips | median abs dZ | median abs d rs_C | median Jaccard of members | median size ratio |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        D = dict(seeds)
        for a1, a2 in pairs:
            pf = sum(1 for k in ks if passes(D[a1][k], lo)["prod"] != passes(D[a2][k], lo)["prod"])
            rf = sum(1 for k in ks if passes(D[a1][k], lo)["rs"] != passes(D[a2][k], lo)["rs"])
            dz = [abs(D[a1][k]["prod"]["free0_tk"] - D[a2][k]["prod"]["free0_tk"]) for k in ks
                  if D[a1][k]["prod"].get("free0_tk") is not None and D[a2][k]["prod"].get("free0_tk") is not None]
            dc = [abs(D[a1][k]["rs_C"] - D[a2][k]["rs_C"]) for k in ks
                  if D[a1][k].get("rs_C") is not None and D[a2][k].get("rs_C") is not None]
            jac = [D[a1][k]["members_jaccard_vs_base"] for k in ks if a1 == "base" and D[a2][k].get("members_jaccard_vs_base") is not None]
            sr = [D[a2][k]["n"] / D[a1][k]["n"] for k in ks if D[a1][k].get("n")]
            L.append("| %s vs %s | %d | %d | %s | %s | %s | %s |" % (a1, a2, pf, rf, f2(st.median(dz)) if dz else "-",
                                                                   f2(st.median(dc)) if dc else "-",
                                                                   f2(st.median(jac)) if jac else "-",
                                                                   f2(st.median(sr)) if sr else "-"))
        if len(seeds) == 3:
            anyp = sum(1 for k in ks if any(passes(d[k], lo)["prod"] for _, d in seeds))
            anyr = sum(1 for k in ks if any(passes(d[k], lo)["rs"] for _, d in seeds))
            allp = sum(1 for k in ks if all(passes(d[k], lo)["prod"] for _, d in seeds))
            allr = sum(1 for k in ks if all(passes(d[k], lo)["rs"] for _, d in seeds))
            L.append("")
            L.append("Held-out ORACLE bounds (not selectable; for scale only): production passes under any seed %d/%d, all "
                     "seeds %d/%d; resample any %d/%d, all %d/%d." % (anyp, len(ks), allp, len(ks), anyr, len(ks), allr, len(ks)))
            if pick:
                cnt = [sum(1 for v in pick.values() if v[0] == s) for s in range(3)]
                L.append("best3 (training criterion) picked seed 0 / 1 / 2 on %d / %d / %d targets." % tuple(cnt))
        L.append("")
    # ---- flagged targets
    L.append("## 5. Flagged targets (diag-102; counted in every rate above, undefined reads = failure)\n")
    L.append("| target | flag | " + " | ".join(ordered) + " |")
    L.append("|---|---|" + "---|" * len(ordered))
    flags = dict(FLAGS)
    for k in keys:                       # any other target with an undefined production read under base (e.g. check)
        if k not in flags and any(base[k]["prod"].get(m) is None
                                  for m in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk")):
            flags[k] = "undefined term under base, vacuous_tk False (auto-detected)"
    for k, why in flags.items():
        cells = []
        for a in ordered:
            r = arms[a].get(k)
            if r is None:
                cells.append("-")
                continue
            p = passes(r, lo)
            undef = [m for m in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk") if r["prod"].get(m) is None]
            cells.append("%s/%s%s" % ("P" if p["prod"] else "f", "P" if p["rs"] else "f",
                                      (" (undef %s)" % ",".join(undef)) if undef else ""))
        L.append("| %s | %s | %s |" % (k, why, " | ".join(cells)))
    L.append("\n(prod / resample: P = pass, f = fail.)\n")
    # ---- selection rule
    L.append("## 6. Pre-stated selection rule applied (README)\n")
    L.append("Eligible = production-pass difference vs base has a CI upper end >= 0. Score = min(production, resample) count."
             " Ties within 2 targets of the best -> lower fit time (x seed1 where available, else x095).\n")
    L.append("| arm | eligible | min(prod, rs) | d prod CI | fit time |")
    L.append("|---|---|---:|---|---|")
    elig = []
    for a in ordered:
        s = S.get(a)
        if s is None:
            continue
        e = s["prod_ci"][1] is not None and s["prod_ci"][1] >= 0
        if a == "base":
            e = True
        if e:
            elig.append(a)
        L.append("| %s | %s | %d of %d | %s | %s |" % (a, "yes" if e else "no", s["min"], s["n"], ci(s["prod_ci"]),
                                                      f2(s["ts1"] if s["ts1"] is not None else s["t95"])))
    if elig:
        best = max(S[a]["min"] for a in elig)
        tied = [a for a in elig if S[a]["min"] >= best - 2]
        win = min(tied, key=lambda a: (S[a]["ts1"] if S[a]["ts1"] is not None else (S[a]["t95"] or 1.0)))
        L.append("")
        L.append("Best eligible score %d; tie band (>= %d): %s; rule winner on the arms present: **%s**. (Arms with"
                 " fewer than 72 targets are compared on the targets they have.)" % (best, best - 2, ", ".join(tied), win))
    (args.out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
