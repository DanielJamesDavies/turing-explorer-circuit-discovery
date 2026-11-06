"""Experiment 105: undefined pass-rule reads that the vacuous flag misses (CPU only, reads stored rows).

Every pass-rule term is a ratio against an EMPTY-circuit baseline (src/eval/wcm_eval.py):
  Z  free0_tk        = (x - e0) / (a_pos - e0)          None unless e0 < a_pos           (ratio)
  A  freeM_topk_tk   = (x - eM_topk) / (a_pos - eM_topk)  None unless eM_topk < a_pos
  C  freeN_topk_tk   = (x - eN_topk) / (a_pos - eN_topk)  None unless eN_topk < a_pos
  N  phi_sup_blind_tk = (a_pos - x) / (a_pos - e0)        None unless a_pos - e0 > 1e-9    (necessity)
  vacuous_tk         = |a_pos - e0| < 0.05 |a_pos|        (Z / N denominator only, unsigned)
a_pos, e0, eM_topk, eN_topk are read with the empty circuit, so whether a term is defined depends only on the target
and its contexts, never on the circuit. This script checks that on the stored rows, counts the undefined-but-not-
vacuous targets, and scores candidate vacuity rules.

  PYTHONPATH=src .venv/bin/python experiments/105-undefined-reads/undefined_reads.py   -> prints tables, writes
  experiments/105-undefined-reads/out/{flagged.csv, summary.json}
"""
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
OUT = HERE / "out"
BAND = (0.75, 1.5)
NEC_MIN = 0.9
REL = 0.05
TERMS = {"Z": ("free0_tk", "e0_tk"), "A": ("freeM_topk_tk", "eM_topk_tk"), "C": ("freeN_topk_tk", "eN_topk_tk"),
         "N": ("phi_sup_blind_tk", "e0_tk")}
KINDS = ("resid", "mlp", "attn")
BANDS = ("L0-4", "L5-7", "L8-11")


def num(v):
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else v


def band_of(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def scorer_defined(term, r):
    """The scorer's own definedness condition, from the baselines only (wcm_eval.ratio / necessity)."""
    ap, e = num(r["a_pos_tk"]), num(r[TERMS[term][1]])
    if term == "N":
        return ap - e > 1e-9
    return e < ap and abs(ap - e) > 1e-9


def passes(r):
    for t in ("Z", "A", "C"):
        v = num(r.get(TERMS[t][0]))
        if v is None or not BAND[0] <= v <= BAND[1]:
            return False
    n = num(r.get(TERMS["N"][0]))
    return n is not None and n >= NEC_MIN


def recompute(term, r):
    """The term from the row's raw reads (checks the stored value and the definedness condition)."""
    ap, e = num(r["a_pos_tk"]), num(r[TERMS[term][1]])
    raw = {"Z": "free0_raw_tk", "A": "freeM_topk_raw_tk", "C": "freeN_topk_raw_tk", "N": "phi_sup_blind_raw_tk"}[term]
    x = num(r[raw])
    if term == "N":
        return (ap - x) / (ap - e) if ap - e > 1e-9 else None
    return (x - e) / (ap - e) if (e < ap and abs(ap - e) > 1e-9) else None


# ---- candidate vacuity rules: all functions of (a_pos, e0, eM_topk, eN_topk) only, i.e. of target + contexts
def den(r, t):
    return num(r["a_pos_tk"]) - num(r[TERMS[t][1]])


def r0(r):        # current: unsigned, Z denominator only
    return bool(r.get("vacuous_tk"))


def r1(r):        # current OR any pass-rule term undefined by its baseline (denominator <= 0)
    return r0(r) or any(not scorer_defined(t, r) for t in TERMS)


def r2(r):        # signed 5% rule on every pass-rule denominator: a_pos - e < 0.05 a_pos for e in {e0, eM_topk, eN_topk}
    ap = num(r["a_pos_tk"])
    return ap <= 0 or any(den(r, t) < REL * ap for t in ("Z", "A", "C"))


def r1s(r):       # signed 5% rule on the Z / N denominator only (fixes the e0 >> a_pos blow-ups, not A / C)
    ap = num(r["a_pos_tk"])
    return ap <= 0 or den(r, "Z") < REL * ap


RULES = {"R0 current (fail)": r0, "R1s signed Z only": r1s, "R1 undefined -> vacuous": r1, "R2 signed 5% all dens": r2}


# ---- option O4 (a different metric, not a vacuity rule): Z and necessity on the post-Top-K read with baseline 0,
# "latent off", as protocol §12.1 already does for the pre-activation read; A and C keep their baselines and the
# signed 5% vacuity (a_pos - e < 0.05 a_pos) applies to them.
def o4_terms(r):
    ap = num(r["a_pos_tk"])
    z = num(r["free0_raw_tk"]) / ap if ap > 1e-9 else None
    n = (ap - num(r["phi_sup_blind_raw_tk"])) / ap if ap > 1e-9 else None
    return z, n


def o4_vacuous(r):
    ap = num(r["a_pos_tk"])
    return ap <= 0 or any(den(r, t) < REL * ap for t in ("A", "C"))


def o4_passes(r):
    z, n = o4_terms(r)
    for v in (z, num(r.get("freeM_topk_tk")), num(r.get("freeN_topk_tk"))):
        if v is None or not BAND[0] <= v <= BAND[1]:
            return False
    return n is not None and n >= NEC_MIN


# ---- sources
def load_062(path_glob, held="strong"):
    rows = []
    for f in sorted(glob.glob(str(path_glob))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("error") is not None or r.get("skip") is not None:
                continue
            if held is not None and r.get("held") not in (held, None):
                continue
            rows.append(r)
    keep = {}
    for r in rows:
        keep[(r["seed"], r.get("held"))] = r            # drop_duplicates(keep="last"), as merge.py
    out = []
    for r in keep.values():
        if held is not None and r.get("held") != held:
            continue
        r = dict(r)
        r["layer"] = int(r["seed"].split(".")[0]); r["kind"] = r["seed"].split(".")[1]
        r["_group"] = "v1-15k"
        out.append(r)
    return out


def load_nested(path, label, panel_key=None):
    out = []
    for line in open(path):
        o = json.loads(line)
        p = dict(o["prod"])
        p["layer"] = int(o["layer"]); p["kind"] = o["kind"]
        p["_arm"] = o["arm"]
        p["_group"] = "%s/%s" % (label, o.get(panel_key, "")) if panel_key else label
        out.append(p)
    return out


def cell(r):
    return "%s/%s" % (band_of(r["layer"]), r["kind"])


def causes(r):
    c = []
    for t in TERMS:
        if num(r.get(TERMS[t][0])) is None:
            c.append(t)
    return c


def main():
    OUT.mkdir(exist_ok=True)
    srcs = {
        "v1-15k strong": load_062(ROOT / "experiments/062-h100-protocol-v1/out_full/out/main/eval.*.jsonl"),
        "v1-15k mid": load_062(ROOT / "experiments/062-h100-protocol-v1/out_full/out/main/eval.*.jsonl", held="mid"),
        "v1 dedup rescore": load_062(ROOT / "experiments/062-h100-protocol-v1/out_rescore_dedup/eval_dedup.jsonl",
                                     held=None),
        "095": load_nested(ROOT / "experiments/095-v2-resample-appendix/out/rows.jsonl", "095"),
        "102 dev": load_nested(ROOT / "experiments/102-pass-improvement/out/phase1/rows.jsonl", "102 dev"),
        "102 check": load_nested(ROOT / "experiments/102-pass-improvement/out/check/rows.jsonl", "102 check"),
    }
    summary = {}
    flagged = []
    for name, rows in srcs.items():
        # 1. consistency: stored None <=> baseline condition; recomputed value == stored value
        mism, valmis = 0, 0
        for r in rows:
            for t in TERMS:
                st = num(r.get(TERMS[t][0]))
                if (st is None) != (not scorer_defined(t, r)):
                    mism += 1
                rc = recompute(t, r)
                if st is not None and rc is not None and abs(st - rc) > 1e-4 * max(1, abs(st)):
                    valmis += 1
        # 2. per target (arms share baselines: check)
        by_t = defaultdict(list)
        for r in rows:
            by_t[r["seed"]].append(r)
        base_var = 0
        for s, rs in by_t.items():
            for k in ("a_pos_tk", "e0_tk", "eM_topk_tk", "eN_topk_tk"):
                vals = {round(num(x[k]), 4) for x in rs}
                if len(vals) > 1:
                    base_var += 1
        tg = {s: rs[0] for s, rs in by_t.items()}
        und = {s: r for s, r in tg.items() if causes(r) and not r0(r)}
        cnt = Counter(cell(r) for r in tg.values())
        cnt_u = Counter(cell(r) for r in und.values())
        cause = Counter("+".join(causes(r)) for r in und.values())
        mech = Counter()
        for r in und.values():
            ap = num(r["a_pos_tk"])
            for t in causes(r):
                e = num(r[TERMS[t][1]])
                mech["%s: %s" % (t, "a_pos<=0" if ap <= 0 else ("e>=a_pos" if e >= ap else "other"))] += 1
        # extra exclusions under each rule, and pass rates (per arm where there are arms)
        arms = sorted({r.get("_arm") for r in rows}, key=str)
        res = {}
        for arm in arms:
            ar = [r for r in rows if r.get("_arm") == arm]
            d = {}
            for rn, fn in RULES.items():
                sc = [r for r in ar if not fn(r)]
                npass = sum(passes(r) for r in sc)
                d[rn] = dict(n=len(ar), excluded=len(ar) - len(sc), scored=len(sc), n_pass=npass,
                             rate=round(npass / len(sc), 4) if sc else None)
            # all targets in the denominator, vacuous counted as fails (lower bound)
            d["all-in (vacuous=fail)"] = dict(n=len(ar), excluded=0, scored=len(ar), n_pass=sum(passes(r) for r in ar),
                                             rate=round(sum(passes(r) for r in ar) / len(ar), 4))
            sc = [r for r in ar if not o4_vacuous(r)]
            np4 = sum(o4_passes(r) for r in sc)
            d["O4 Z/N baseline 0 + R2 on A/C"] = dict(n=len(ar), excluded=len(ar) - len(sc), scored=len(sc), n_pass=np4,
                                                     rate=round(np4 / len(sc), 4) if sc else None,
                                                     gained=sum(o4_passes(r) and not passes(r) for r in sc),
                                                     lost=sum(passes(r) and not o4_passes(r) for r in ar))
            # per band x kind pass rates under R0 and R2 (and the R2 stratum's pass on its defined terms)
            pc = {}
            for c in sorted({cell(r) for r in ar}):
                cr = [r for r in ar if cell(r) == c]
                s0 = [r for r in cr if not r0(r)]
                s2 = [r for r in cr if not r2(r)]
                pc[c] = dict(n=len(cr), r0="%d/%d" % (sum(passes(r) for r in s0), len(s0)),
                             r0_rate=round(sum(passes(r) for r in s0) / len(s0), 4) if s0 else None,
                             r2="%d/%d" % (sum(passes(r) for r in s2), len(s2)),
                             r2_rate=round(sum(passes(r) for r in s2) / len(s2), 4) if s2 else None,
                             r2_excl=len(cr) - len(s2))
            d["_by_cell"] = pc
            res[str(arm)] = d
        # by-cell exclusion counts under R1 / R2 (beyond R0)
        cell_extra = {}
        for rn in ("R1 undefined -> vacuous", "R2 signed 5% all dens", "R1s signed Z only"):
            fn = RULES[rn]
            cell_extra[rn] = dict(Counter(cell(r) for r in tg.values() if fn(r) and not r0(r)))
        # R2 extra beyond R1: defined but near-vacuous A / C / Z denominators; how many of those pass now
        near = [r for r in tg.values() if r2(r) and not r1(r)]
        near_terms = Counter("+".join(t for t in ("Z", "A", "C") if den(r, t) < REL * num(r["a_pos_tk"])) for r in near)
        # R0-vacuous that R2 would NOT exclude (R0 is unsigned: e0 slightly above a_pos is caught by both)
        r0_not_r2 = sum(1 for r in tg.values() if r0(r) and not r2(r))
        # size of the wrong-signed baselines, and how the excluded targets do on their defined terms
        def q(v):
            v = sorted(v)
            return [round(v[int(p * (len(v) - 1))], 3) for p in (0.1, 0.5, 0.9)] if v else None
        zr = [num(r["e0_tk"]) / num(r["a_pos_tk"]) for r in und.values() if "Z" in causes(r)]
        ar_ = [num(r["eM_topk_tk"]) / num(r["a_pos_tk"]) for r in und.values() if "A" in causes(r)]
        circ_z = [num(r["free0_raw_tk"]) / num(r["a_pos_tk"]) for r in und.values() if "Z" in causes(r)]
        circ_a = [num(r["freeM_topk_raw_tk"]) / num(r["a_pos_tk"]) for r in und.values() if "A" in causes(r)]

        def pass_defined(r):
            ok = True
            for t in TERMS:
                v = num(r.get(TERMS[t][0]))
                if v is None:
                    continue
                ok &= (BAND[0] <= v <= BAND[1]) if t != "N" else (v >= NEC_MIN)
            return ok
        r2x = [r for r in tg.values() if r2(r) and not r0(r)]
        n_e0pos = sum(1 for r in tg.values() if num(r["e0_tk"]) > 0)
        near = [r for r in tg.values() if r2(r) and not r1(r)]
        summary[name] = dict(n_rows=len(rows), n_targets=len(tg), mismatch_none_vs_baseline=mism,
                             mismatch_value=valmis, targets_with_baseline_varying_across_arms=base_var,
                             n_vacuous_R0=sum(r0(r) for r in tg.values()), n_undef_not_vacuous=len(und),
                             undef_by_cell=dict(cnt_u), targets_by_cell=dict(cnt), undef_terms=dict(cause),
                             undef_mechanism=dict(mech), rules=res, extra_excluded_by_cell=cell_extra,
                             r2_minus_r1=len(near), r2_minus_r1_terms=dict(near_terms),
                             r2_minus_r1_pass_any_arm=sum(1 for r in near if any(
                                 passes(x) for x in by_t[r["seed"]])),
                             r0_not_r2=r0_not_r2,
                             e0_over_apos_Zundef_p10_50_90=q(zr), eM_over_apos_Aundef_p10_50_90=q(ar_),
                             circuit_Z_raw_over_apos_Zundef=q(circ_z), circuit_A_raw_over_apos_Aundef=q(circ_a),
                             r2_excluded=len(r2x), r2_excluded_pass_defined_terms=sum(pass_defined(r) for r in r2x),
                             n_e0_tk_positive=n_e0pos, undef_seeds=sorted(und))
        for s, r in sorted(und.items()):
            ap = num(r["a_pos_tk"])
            flagged.append(dict(source=name, seed=s, cell=cell(r), undefined="+".join(causes(r)),
                                a_pos_tk=ap, e0_tk=num(r["e0_tk"]), eM_topk_tk=num(r["eM_topk_tk"]),
                                eN_topk_tk=num(r["eN_topk_tk"]), e0_over_a_pos=num(r["e0_tk"]) / ap if ap else None,
                                eM_over_a_pos=num(r["eM_topk_tk"]) / ap if ap else None,
                                free0_raw_over_a_pos=num(r["free0_raw_tk"]) / ap if ap else None,
                                pass_defined_terms=all(
                                    (num(r.get(TERMS[t][0])) is None) or
                                    (BAND[0] <= num(r[TERMS[t][0]]) <= BAND[1] if t != "N" else
                                     num(r[TERMS[t][0]]) >= NEC_MIN) for t in TERMS)))
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    keys = list(flagged[0]) if flagged else []
    with open(OUT / "flagged.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for d in flagged:
            f.write(",".join("" if d[k] is None else (("%.4g" % d[k]) if isinstance(d[k], float) else str(d[k]))
                             for k in keys) + "\n")
    # ---- print
    for name, s in summary.items():
        print("\n== %s: %d rows, %d targets; R0 vacuous %d; undefined & not vacuous %d" % (
            name, s["n_rows"], s["n_targets"], s["n_vacuous_R0"], s["n_undef_not_vacuous"]))
        print("   checks: stored-None vs baseline-condition mismatches %d; value mismatches %d; "
              "targets whose baselines vary across arms %d" % (
                  s["mismatch_none_vs_baseline"], s["mismatch_value"], s["targets_with_baseline_varying_across_arms"]))
        print("   undefined terms %s; mechanism %s" % (s["undef_terms"], s["undef_mechanism"]))
        print("   band x kind (undefined / targets):")
        for b in BANDS:
            print("     %-6s " % b + "  ".join("%s %d/%d" % (k, s["undef_by_cell"].get("%s/%s" % (b, k), 0),
                                                         s["targets_by_cell"].get("%s/%s" % (b, k), 0)) for k in KINDS))
        print("   extra excluded beyond R0 by cell: %s" % s["extra_excluded_by_cell"])
        print("   R2 - R1 (defined but < 5%% denominators): %d, terms %s, of which pass in some arm %d; "
              "R0-vacuous not R2 %d" % (s["r2_minus_r1"], s["r2_minus_r1_terms"], s["r2_minus_r1_pass_any_arm"],
                                         s["r0_not_r2"]))
        print("   e0/a_pos (Z undefined) p10/50/90 %s; circuit Z raw/a_pos %s" % (
            s["e0_over_apos_Zundef_p10_50_90"], s["circuit_Z_raw_over_apos_Zundef"]))
        print("   eM_topk/a_pos (A undefined) p10/50/90 %s; circuit A raw/a_pos %s" % (
            s["eM_over_apos_Aundef_p10_50_90"], s["circuit_A_raw_over_apos_Aundef"]))
        print("   R2 excludes %d beyond R0; %d of them pass every DEFINED term; targets with e0_tk > 0: %d" % (
            s["r2_excluded"], s["r2_excluded_pass_defined_terms"], s["n_e0_tk_positive"]))
        for arm, d in s["rules"].items():
            print("   arm %s: " % arm + " | ".join("%s %d/%d=%.1f%% (excl %d%s)" % (
                rn, v["n_pass"], v["scored"], 100 * (v["rate"] or 0), v["excluded"],
                ("; +%d -%d" % (v["gained"], v["lost"])) if "gained" in v else "")
                for rn, v in d.items() if not rn.startswith("_")))
            if len(d["_by_cell"]) > 1:
                print("     by cell R0 -> R2: " + "; ".join("%s %s=%.1f%% -> %s=%.1f%%" % (
                    c, v["r0"], 100 * (v["r0_rate"] or 0), v["r2"], 100 * (v["r2_rate"] or 0))
                    for c, v in d["_by_cell"].items()))
    # near-threshold overlap (062 merge.py targets.csv: rank_clean >= K/2 = 64), v1 15k strongest
    tc = ROOT / "experiments/062-h100-protocol-v1/out_full/out/targets.csv"
    if tc.exists():
        rank = {}
        for i, line in enumerate(open(tc)):
            f = line.rstrip("\n").split(",")
            if i == 0:
                hdr = f; continue
            d = dict(zip(hdr, f))
            rank[d["seed"]] = float(d["rank_clean"]) if d.get("rank_clean") else None
        rows15 = {r["seed"]: r for r in srcs["v1-15k strong"]}
        for lab, sel in (("Z/N undefined", lambda r: num(r.get("free0_tk")) is None and not r0(r)),
                         ("A undefined", lambda r: num(r.get("freeM_topk_tk")) is None and not r0(r)),
                         ("all targets", lambda r: True)):
            ss = [s for s, r in rows15.items() if sel(r) and rank.get(s) is not None]
            nt = sum(rank[s] >= 64 for s in ss)
            print("near-threshold share, v1 15k strongest, %s: %d / %d = %.1f%%" % (lab, nt, len(ss), 100 * nt / max(1, len(ss))))
    print("\nflagged targets:")
    for d in flagged:
        if not d["source"].startswith("v1"):
            print("  ", d)


if __name__ == "__main__":
    main()
