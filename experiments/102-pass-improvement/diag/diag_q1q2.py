"""102 diag Q1 (fit or generalisation?) and Q2 (binding term) on 095's 72 targets x {v2, v2CR, v2R}. CPU, no model.

Q1. The scorer logs a train-split read only for Z: free0_tr_tk = Z on the 48 strongest-TRAIN contexts (the eval
reference rows; arm B trains on 32 of them + 16 mid-band rows). A / C / necessity have no train-split read. For those
the fit's own training curve (train.jsonl, per-term weighted minibatch loss, stride 20) is used as a proxy: mean of
the last 5 logged points (steps 320-399, batch 4 each), per term. 093's train.jsonl covers v2 (and v1); 095's covers
v2CR / v2R.

Q2. Per circuit: every term's deficit (band terms vs [0.75, 1.5], necessity vs 0.9, donor A / C vs [0.75, 1.5]);
binding term = the largest deficit; solo failures = exactly one failing term; near misses = all deficits <= 0.10.

Writes out/q1q2.md, out/q1q2_rows.csv.
"""
import csv
from collections import Counter, defaultdict

import numpy as np

from diag_common import (BANDS, DIAG, E93, E95, FLOOR, KINDS, deficits, fnum, inband, load_095, load_jsonl,
                         md_table, rate)

OUT = DIAG / "out"
OUT.mkdir(exist_ok=True)
ARMS = ("v2", "v2CR", "v2R")
TERMS_ARM = {"v2": ("Z", "A", "C", "nec"), "v2CR": ("Z", "A", "C", "nec", "rsA", "rsC"),
             "v2R": ("Z", "A", "C", "nec", "rsA", "rsC")}


def curve_proxy():
    """{(arm, seed): {term: mean of last 5 logged per-term losses}}; arm names as in 095."""
    out = {}
    for path, lab in ((E93 / "out" / "train.jsonl", "arm93"), (E95 / "out" / "train.jsonl", "arm95")):
        for r in load_jsonl(path):
            arm = r.get(lab)
            if arm == "v1":
                continue
            t = (r.get("curve") or {}).get("terms") or {}
            out[(arm, r["seed"])] = {k: float(np.mean(v[-5:])) for k, v in t.items()}
            out[(arm, r["seed"])]["holdout_data_loss"] = r.get("holdout_data_loss")
            out[(arm, r["seed"])]["loss_final"] = r.get("loss_final")
    return out


def main():
    by = load_095()
    cp = curve_proxy()
    md = ["# Q1 / Q2 tables (diag_q1q2.py)\n",
          "Floor %.2f. n = 72 targets per arm (8 per band x kind). Train-split Z = free0_tr_tk (48 strongest-train)." % FLOOR]
    rows_out = []
    for arm in ARMS:
        d = by[arm]
        terms = TERMS_ARM[arm]
        q1 = Counter()
        q1_cell = defaultdict(Counter)
        bind = Counter()
        bind_cell = defaultdict(Counter)
        anyfail = Counter()
        solo = Counter()
        near = 0
        near_list = []
        nfail = 0
        gapZ = {True: [], False: []}
        for seed, r in sorted(d.items()):
            p = r["prod"]
            v = r["vals"]
            df = deficits(v)
            df = {t: df[t] for t in terms}
            fail_terms = [t for t in terms if df[t] > 0]
            ok = r["RS"] if arm != "v2" else r["P"]
            # for v2, also record the R-side failure for context but the arm's headline is production
            zt, zh = fnum(p.get("free0_tr_tk")), fnum(p.get("free0_tk"))
            if zt is not None and zh is not None:
                gapZ[bool(r["P"])].append(zh - zt)
            cell = (r["band"], r["kind"])
            row = dict(arm=arm, seed=seed, band=r["band"], kind=r["kind"], P=r["P"], RS=r["RS"], n=r.get("n"),
                       Z_tr=zt, Z_ho=zh, A=v["A"], C=v["C"], nec=v["nec"], rsA=v["rsA"], rsC=v["rsC"],
                       fail_terms="+".join(fail_terms))
            cpx = cp.get((arm, seed), {})
            for k in ("zero", "floor", "pos", "resample", "holdout_data_loss"):
                row["tr_" + k] = cpx.get(k)
            rows_out.append(row)
            if not fail_terms:
                continue
            nfail += 1
            worst = max(fail_terms, key=lambda t: df[t])
            bind[worst] += 1
            bind_cell[cell][worst] += 1
            for t in fail_terms:
                anyfail[t] += 1
            if len(fail_terms) == 1:
                solo[fail_terms[0]] += 1
            if max(df[t] for t in fail_terms) <= 0.10:
                near += 1
                near_list.append("%s (%s)" % (seed, ",".join("%s %.2f" % (t, v[t] if v[t] is not None else -1)
                                                             for t in fail_terms)))
            # Q1 class (production side): does the circuit fail Z on train too?
            prod_fail = [t for t in ("Z", "A", "C", "nec") if df[t] > 0]
            if "Z" in prod_fail:
                cls = "Z fails train too" if not inband(zt) else "Z passes train, fails held-out"
            elif prod_fail:
                cls = "prod fails on A/C/nec only (no train read)"
            else:
                cls = "prod passes; donor-only fail"
            q1[cls] += 1
            q1_cell[cell][cls] += 1
        hdr = "resample pass (prod + donor A, C)" if arm != "v2" else "production pass"
        md.append("\n## Arm %s: %d/%d fail (%s; production %s, resample %s)\n" % (
            arm, nfail, len(d), hdr, rate([r["P"] for r in d.values()]), rate([r["RS"] for r in d.values()])))
        md.append("**Q1 classes of the failing circuits** (failing = any term of %s out of band):\n" % "/".join(terms))
        md.append(md_table(["class", "n"], sorted(q1.items(), key=lambda x: -x[1])))
        md.append("\nZ held-out minus Z train (median [IQR]): production-pass %s (n %d); production-fail %s (n %d)\n" % (
            iqr(gapZ[True]), len(gapZ[True]), iqr(gapZ[False]), len(gapZ[False])))
        md.append("**Q2 binding term** (largest deficit) / any-fail count / solo-fail count:\n")
        md.append(md_table(["term", "binding", "fails (any)", "solo"],
                           [(t, bind[t], anyfail[t], solo[t]) for t in terms]))
        md.append("\nNear misses (all deficits <= 0.10): %d of %d failures: %s\n" % (near, nfail, "; ".join(near_list)))
        md.append("**Binding term by band x kind** (failures / 8 per cell):\n")
        hdr2 = ["cell", "fail"] + list(terms)
        trs = []
        for b, _, _ in BANDS:
            for k in KINDS:
                c = bind_cell[(b, k)]
                trs.append(["%s/%s" % (b, k), sum(c.values())] + [c[t] for t in terms])
        md.append(md_table(hdr2, trs))
        md.append("\n**Q1 class by band x kind:**\n")
        cls_names = sorted(q1)
        trs = []
        for b, _, _ in BANDS:
            for k in KINDS:
                c = q1_cell[(b, k)]
                trs.append(["%s/%s" % (b, k)] + [c[x] for x in cls_names])
        md.append(md_table(["cell"] + cls_names, trs))

    # Q1 proxy: training-curve losses, held-out pass vs fail per term
    md.append("\n## Q1 proxy: last-5 training-curve loss per term (weighted minibatch loss), by held-out outcome\n")
    md.append("If held-out failures of a term have the same TRAINING loss on that term as the passes, the failure is a"
              " generalisation gap, not a fit failure. Medians; AUC = P(fail-circuit train loss > pass-circuit train"
              " loss).\n")
    from diag_common import auc
    trs = []
    for arm in ARMS:
        d = [x for x in rows_out if x["arm"] == arm]
        for term, lk in (("Z", "tr_zero"), ("A", "tr_pos"), ("C", "tr_floor"), ("rsA", "tr_resample"),
                         ("rsC", "tr_resample")):
            if term.startswith("rs") and arm == "v2":
                continue
            if term == "C" and arm == "v2R":
                continue
            vals = [(x[lk], 0 < (deficits_row(x)[term])) for x in d if x.get(lk) is not None]
            if not vals:
                continue
            fl = [a for a, f in vals if f]
            ps = [a for a, f in vals if not f]
            a = auc([v for v, _ in vals], [f for _, f in vals])
            trs.append([arm, term, lk[3:], len(ps), med(ps), len(fl), med(fl), a])
    md.append(md_table(["arm", "held-out term", "train-loss term", "n pass", "median train loss (pass)", "n fail",
                        "median train loss (fail)", "AUC"], trs))
    md.extend(topk_cut(by))
    with open(OUT / "q1q2_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    (OUT / "q1q2.md").write_text("\n".join(md))
    print("\n".join(md))


PRE = dict(Z="z_pre0", A="freeM_topk_pre", C="freeN_topk_pre", nec="nec_pre0")


def topk_cut(by):
    """Is a failing production term lost at the target site's Top-K cut? Compare the post-Top-K read with protocol
    12's pre-activation read of the same term (Z and necessity on the 0 baseline, z_pre0 / nec_pre0; A / C =
    freeM_topk_pre / freeN_topk_pre). 'cut' = post fails, pre in band; 'both' = both fail."""
    import csv as _csv
    ctx = {}
    p72 = OUT / "ctxfeat_72.csv"
    if p72.exists():
        with open(p72) as fh:
            ctx = {r["seed"]: r for r in _csv.DictReader(fh)}
    md = ["\n## Top-K cut: failing production terms, post-Top-K vs pre-activation read\n",
          "cut = fails post-Top-K but its pre-activation read passes (target restored, then loses the Top-K"
          " competition at its own site); both = fails both reads.\n"]
    trs = []
    for arm in ARMS:
        cnt = Counter()
        prod_fail = 0
        prod_fail_pre_pass = 0
        for seed, r in by[arm].items():
            p = r["prod"]
            vals = dict(Z=r["vals"]["Z"], A=r["vals"]["A"], C=r["vals"]["C"], nec=r["vals"]["nec"])
            pre = dict(Z=r.get("z_pre0"), A=p.get("freeM_topk_pre"), C=p.get("freeN_topk_pre"), nec=r.get("nec_pre0"))
            failed = False
            for t in ("Z", "A", "C", "nec"):
                post_ok = (vals[t] is not None and vals[t] >= 0.9) if t == "nec" else inband(vals[t])
                if post_ok:
                    continue
                failed = True
                pre_ok = (pre[t] is not None and pre[t] >= 0.9) if t == "nec" else inband(pre[t])
                cnt[(t, "cut" if pre_ok else "both")] += 1
            if failed:
                prod_fail += 1
                pre_all = all(((pre[t] is not None and pre[t] >= 0.9) if t == "nec" else inband(pre[t]))
                              for t in ("Z", "A", "C", "nec"))
                prod_fail_pre_pass += pre_all
        for t in ("Z", "A", "C", "nec"):
            trs.append([arm, t, cnt[(t, "cut")], cnt[(t, "both")]])
        trs.append([arm, "ALL production failures: pass the pre-activation read on every term", prod_fail_pre_pass,
                    prod_fail])
    md.append(md_table(["arm", "term", "cut (pre passes)", "both fail / (last row: n failures)"], trs))
    # Q1 by leakage: Z train vs held-out only means something where held-out is not a copy of training
    if ctx:
        md.append("\n## Q1 by held-out leakage (ho_dup16 = share of held-out rows whose 16-token left prefix equals"
                  " a trained row's)\n")
        trs = []
        for arm in ARMS:
            for lo, hi, lab in ((0, 0.3125, "< 5/16 duplicated"), (0.3125, 2, ">= 5/16 duplicated")):
                sel = [r for s, r in by[arm].items() if s in ctx and lo <= float(ctx[s]["ho_dup16"]) < hi]
                zf = [r for r in sel if not inband(r["vals"]["Z"])]
                ztr = sum(1 for r in zf if not inband(r["prod"].get("free0_tr_tk")))
                gap = [r["vals"]["Z"] - r["prod"]["free0_tr_tk"] for r in sel
                       if r["vals"]["Z"] is not None and r["prod"].get("free0_tr_tk") is not None]
                trs.append([arm, lab, len(sel), rate([r["P"] for r in sel]), len(zf), ztr, iqr(gap)])
        md.append(md_table(["arm", "leakage", "n targets", "production pass", "Z fails held-out", "... and train too",
                            "Z held-out - train, median [IQR]"], trs))
    return md


def deficits_row(x):
    v = dict(Z=x["Z_ho"], A=x["A"], C=x["C"], nec=x["nec"], rsA=x["rsA"], rsC=x["rsC"])
    return deficits(v)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def iqr(xs):
    if not xs:
        return "-"
    q = np.percentile(xs, [25, 50, 75])
    return "%.3f [%.3f, %.3f]" % (q[1], q[0], q[2])


if __name__ == "__main__":
    main()
