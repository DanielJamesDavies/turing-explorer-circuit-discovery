"""134 step 3 report (CPU; ours): out/s3/rows.jsonl + out/s3/members/*.npz -> out/s3_report.md.

Groups: low = alpha < 0.8, ctl = alpha in [0.95, 1.05] (same circuits). Per member (sums over the rows of a set):
  read 1 over-drive   od_raw = sum raw mass / sum natural mass (positions <= anchor; raw = the re-encoded live value in
                      the circuit-only run, before alpha); od_val = the same with the written alpha x raw. Undefined when
                      the member's natural mass is 0 ("natural-silent"; share reported).
  read 2 brake        ctb = mean over rows of the first-order contribution to the target pre-activation, / the mean
                      natural target pre-activation at the anchors of the set (held-out rows). Brake = ctb < 0.
  read 3 redundancy   max decoder cosine to another member at the same site.
  read 4 selectivity  sel = mean natural value at the contrast rows' would-be anchors / mean natural value at the held-out
                      anchors; fire_ctr = share of ALL contrast positions where the member is > 0; fire_ho = share of
                      held-out positions <= anchor.
Group comparisons: pooled members (median / shares), and per target (median over the target's low members minus the
same over its ctl members) with sign counts over targets.
Causal arms: per arm pass count, medians, and the per-target paired deltas against (a) with sign counts.

  PYTHONPATH=src .venv/bin/python -B experiments/134-sub1-coefficients/s3_report.py [--out experiments/.../out/s3]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# MODEL134 [gpt2-small] | pythia-70m (step 3b): default out dir out/s3py, panel out/s3py_panel.json, report
# out/s3py_report.md; the gpt2-small output is unchanged.
MODEL = os.environ.get("MODEL134", "gpt2-small")
PREFIX = "s3" if MODEL == "gpt2-small" else "s3py"
MODEL_NAME = {"gpt2-small": "GPT-2 small", "pythia-70m": "Pythia-70m"}[MODEL]
LOW, CTL = 0.8, (0.95, 1.05)
ARM_LABEL = {"a_prod": "(a) production alpha", "b_low1": "(b) low -> alpha 1", "c_low0": "(c) low removed",
             "d_ctl0": "(d) matched control removed", "low_x0.5": "sweep: low alpha x 0.5",
             "low_x2": "sweep: low alpha x 2", "ctl_x2": "sweep: matched control alpha x 2"}
HEAD = (("Z", "free0_tk"), ("A", "freeM_topk_tk"), ("C", "freeN_topk_tk"), ("nec", "phi_sup_blind_tk"))


def load_jsonl(p):
    out = []
    if Path(p).exists():
        for line in open(p):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def med(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    return float(np.median(x)) if len(x) else float("nan")


def f(v, p=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if p == 3 and v != 0 and abs(v) < 0.01:
        return "%.2e" % v
    return ("%%.%df" % p) % v


def signs(d, tol=1e-9):
    d = [v for v in d if v is not None and np.isfinite(v)]
    return "%d+ / %d- / %d=" % (sum(v > tol for v in d), sum(v < -tol for v in d), sum(abs(v) <= tol for v in d))


def safe_ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 1e-9, a / np.where(b > 1e-9, b, 1.0), np.nan)
    return r


def member_metrics(z, fills):
    a = z["alpha"]
    out = {"alpha": a, "dec_cos": z["dec_cos_max"]}
    for st in ("ho", "ctr"):
        nat_ms = z["%s_nat_val_ms" % st].sum(0)
        out["%s_natsilent" % st] = (nat_ms <= 1e-9).astype(float)
        a_pre = float(np.mean(z["ho_nat_pre_at"]))                  # normaliser: the held-out natural pre
        den = a_pre if abs(a_pre) > 1e-9 else np.nan
        out["%s_ctb_nat" % st] = z["%s_nat_ctb" % st].mean(0) / den
        for F in fills:
            if "%s_%s_raw_ms" % (st, F) not in z:
                continue
            out["%s_od_raw_%s" % (st, F)] = safe_ratio(z["%s_%s_raw_ms" % (st, F)].sum(0), nat_ms)
            out["%s_od_val_%s" % (st, F)] = safe_ratio(z["%s_%s_val_ms" % (st, F)].sum(0), nat_ms)
            out["%s_ctb_%s" % (st, F)] = z["%s_%s_ctb" % (st, F)].mean(0) / den
    out["sel"] = safe_ratio(z["ctr_nat_val_at"].mean(0), z["ho_nat_val_at"].mean(0))
    out["fire_ctr"] = z["ctr_nat_fire_all"].mean(0)
    out["fire_ho"] = z["ho_nat_fire"].mean(0)
    out["fire_ctr_gt_ho"] = (out["fire_ctr"] > out["fire_ho"]).astype(float)
    return out


# (label, metric key, summary) summary: med | gt1 (share > 1) | neg (share < 0) | share (mean of a 0/1 metric) | log2med
def stat_rows(fills):
    rows = [("1 over-driven", None, None)]
    for st in ("ho", "ctr"):
        rows.append(("  %s natural-silent share (natural mass 0)" % st, "%s_natsilent" % st, "share"))
        for F in fills:
            rows += [("  %s %s: median od_raw (live / natural, before alpha)" % (st, F), "%s_od_raw_%s" % (st, F), "med"),
                     ("  %s %s: share od_raw > 1" % (st, F), "%s_od_raw_%s" % (st, F), "gt1"),
                     ("  %s %s: median od_val (alpha x live / natural)" % (st, F), "%s_od_val_%s" % (st, F), "med"),
                     ("  %s %s: share od_val > 1" % (st, F), "%s_od_val_%s" % (st, F), "gt1")]
    rows.append(("2 brake (contribution / natural target pre)", None, None))
    for st in ("ho", "ctr"):
        for run in ("nat",) + tuple(fills):
            rows += [("  %s %s: median ctb" % (st, run), "%s_ctb_%s" % (st, run), "med"),
                     ("  %s %s: median abs ctb" % (st, run), "%s_ctb_%s" % (st, run), "absmed"),
                     ("  %s %s: share ctb < 0" % (st, run), "%s_ctb_%s" % (st, run), "neg"),
                     ("  %s %s: group net ctb (sum over the target's group members; pooled col = median over "
                      "targets)" % (st, run), "%s_ctb_%s" % (st, run), "sum")]
    rows += [("3 redundancy", None, None),
             ("  median max decoder cosine (same site)", "dec_cos", "med"),
             ("  share max cosine > 0.5", "dec_cos", "gt0.5"),
             ("4 selectivity", None, None),
             ("  median sel (contrast would-be anchor / held-out anchor)", "sel", "med"),
             ("  share sel > 0.5", "sel", "gt0.5"),
             ("  median fire share, contrast (all positions)", "fire_ctr", "med"),
             ("  median fire share, held-out (<= anchor)", "fire_ho", "med"),
             ("  share fire_ctr > fire_ho", "fire_ctr_gt_ho", "share")]
    return rows


def summarise(x, how):
    x = np.asarray(x, dtype=float)
    fin = x[np.isfinite(x)]
    if how == "med":
        return float(np.median(fin)) if len(fin) else np.nan
    if how == "absmed":
        return float(np.median(np.abs(fin))) if len(fin) else np.nan
    if how == "sum":
        return float(np.sum(fin)) if len(fin) else np.nan
    if how == "share":
        return float(np.mean(fin)) if len(fin) else np.nan
    if how == "gt1":
        return float(np.mean(fin > 1)) if len(fin) else np.nan
    if how == "gt0.5":
        return float(np.mean(fin > 0.5)) if len(fin) else np.nan
    if how == "neg":
        return float(np.mean(fin < 0)) if len(fin) else np.nan
    raise ValueError(how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out" / PREFIX))
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    O = Path(a.out)
    rep = Path(a.report) if a.report else (HERE / "out" / ("%s_report.md" % PREFIX)
                                           if O.resolve() == (HERE / "out" / PREFIX).resolve() else O / "s3_report.md")
    panel = {}
    pj = HERE / "out" / ("%s_panel.json" % PREFIX)
    if pj.exists():
        panel = {q["seed"]: q for q in json.loads(pj.read_text())["queue"]}
    rows = [r for r in load_jsonl(O / "rows.jsonl") if "error" not in r]
    errs = [r for r in load_jsonl(O / "rows.jsonl") if "error" in r]
    by = {}
    for r in rows:
        by.setdefault(r["seed"], {})[r["phase"]] = r
    L = ["# 134 step 3: members with alpha < 0.8 vs near-1 controls (%s, 11-07 production circuits)" % MODEL_NAME, "",
         "Generated by `s3_report.py` from `%s`. low = alpha < 0.8, ctl = alpha in [0.95, 1.05], same circuits." % O,
         "Targets with rows: %d (reads %d, arms %d, sweep %d, donor %d); error rows %d; tripped: %s." % (
             len(by), sum("reads" in v for v in by.values()), sum("arms" in v for v in by.values()),
             sum("sweep" in v for v in by.values()), sum("donor" in v for v in by.values()), len(errs),
             (O / "tripped.txt").read_text().strip().replace("\n", "; ") if (O / "tripped.txt").exists() else "none"),
         ""]

    # ------------------------------------------------------------------ member reads
    fills = ("Z", "A", "C")
    pooled = {"low": {}, "ctl": {}}
    per_t = {}
    n_mem = {"low": 0, "ctl": 0}
    for key in sorted(by):
        p = O / "members" / ("%s.npz" % key)
        if "reads" not in by[key] or not p.exists():
            continue
        z = np.load(p)
        mm = member_metrics(z, fills)
        al = mm["alpha"]
        gm = {"low": al < LOW, "ctl": (al >= CTL[0]) & (al <= CTL[1])}
        per_t[key] = {}
        for g, sel in gm.items():
            n_mem[g] += int(sel.sum())
            for k, v in mm.items():
                pooled[g].setdefault(k, []).append(v[sel])
            per_t[key][g] = {k: v[sel] for k, v in mm.items()}
    if per_t:
        for g in pooled:
            pooled[g] = {k: np.concatenate(v) for k, v in pooled[g].items()}
        L += ["## Reads 1-4: low vs control", "",
              "%d targets; %d low members, %d control members. Pooled = over all members; per target = median over a "
              "target's low members minus the median over its control members (sign counts over targets: + means low "
              "is higher)." % (len(per_t), n_mem["low"], n_mem["ctl"]), "",
              "| read | pooled low | pooled ctl | per-target median diff (low - ctl) | sign counts |",
              "|---|---|---|---|---|"]
        for lab, k, how in stat_rows(fills):
            if k is None:
                L.append("| **%s** | | | | |" % lab)
                continue
            if k not in pooled["low"]:
                continue
            d = []
            for key, gg in per_t.items():
                lo, ct = summarise(gg["low"][k], how), summarise(gg["ctl"][k], how)
                d.append(lo - ct if np.isfinite(lo) and np.isfinite(ct) else np.nan)
            if how == "sum":                       # pooled column: median over targets of the per-target sums
                pl = med([summarise(gg["low"][k], how) for gg in per_t.values()])
                pc = med([summarise(gg["ctl"][k], how) for gg in per_t.values()])
            else:
                pl, pc = summarise(pooled["low"][k], how), summarise(pooled["ctl"][k], how)
            L.append("| %s | %s | %s | %s | %s |" % (lab, f(pl), f(pc), f(med(d)), signs(d)))
        # alpha bins inside the low group
        L += ["", "Low members split by alpha (pooled):", "",
              "| alpha | members | median od_raw ho A | median od_val ho A | share ctb_nat ho < 0 | median dec cos | "
              "median sel | natural-silent ho |", "|---|---|---|---|---|---|---|---|"]
        al = pooled["low"]["alpha"]
        for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.8)):
            s = (al >= lo) & (al < hi)
            if not s.any():
                continue
            P = {k: v[s] for k, v in pooled["low"].items()}
            L.append("| %.1f-%.1f | %d | %s | %s | %s | %s | %s | %s |" % (
                lo, hi, int(s.sum()), f(summarise(P.get("ho_od_raw_A", []), "med")),
                f(summarise(P.get("ho_od_val_A", []), "med")), f(summarise(P["ho_ctb_nat"], "neg")),
                f(summarise(P["dec_cos"], "med")), f(summarise(P["sel"], "med")), f(summarise(P["ho_natsilent"], "share"))))
        L.append("")

    # ------------------------------------------------------------------ causal arms
    def arm_vals(key, arm):
        r = by[key]
        v = {}
        for ph in ("arms", "sweep"):
            if ph in r and arm in r[ph]["arms"]:
                x = r[ph]["arms"][arm]
                for lab, fld in HEAD:
                    v[lab] = x["prod"].get(fld)
                v["pass"] = x["pass_prod"]
                v["vac"] = x["vacuous"]
                v["n"] = x["n"]
                vals = [v[l_] for l_, _ in HEAD[:3]]
                v["dev"] = max(abs(t - 1) for t in vals) if all(t is not None for t in vals) else None
        if "donor" in r and arm in r["donor"]["donor"]:
            d = r["donor"]["donor"][arm]
            v["SF_C"] = d["sf"].get("C")
            v["SF_A"] = d["sf"].get("A")
            v["SF_R"] = d["sf"].get("R")
            v["recA"], v["recC"], v["recR"] = (d["recovery"].get(x) for x in "ACR")
            v["rs_pass"] = d.get("resample_pass")
        return v

    arms_all = ("a_prod", "b_low1", "c_low0", "d_ctl0", "low_x0.5", "low_x2", "ctl_x2")
    keys_a = [k for k in sorted(by) if "arms" in by[k]]
    if keys_a:
        V = {k: {arm: arm_vals(k, arm) for arm in arms_all} for k in keys_a}
        L += ["## Causal arms (rescore, no refit)", "",
              "%d targets with arm rows (%d with donor rows). Pass = production rule (Z, A, C in [0.75, 1.5], nec >= 0.9; "
              "vacuous counts as fail). dev = max(|Z-1|, |A-1|, |C-1|). SF_C = donor silent firing, pool C, variant fit, "
              "mean over draws." % (len(keys_a), sum("donor" in by[k] for k in keys_a)), "",
              "| arm | n targets | pass | median n | median Z | median A | median C | median nec | median dev | "
              "median SF_C | median rec A | median rec C | median rec R | resample pass |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms_all:
            vs = [V[k][arm] for k in keys_a if "pass" in V[k][arm]]
            if not vs:
                continue
            ps = sum(bool(v["pass"]) and not v["vac"] for v in vs)
            rsv = [v.get("rs_pass") for v in vs if v.get("rs_pass") is not None]
            L.append("| %s | %d | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                ARM_LABEL[arm], len(vs), ps, len(vs), f(med([v["n"] for v in vs]), 0),
                *(f(med([v[l_] for v in vs])) for l_, _ in HEAD), f(med([v["dev"] for v in vs])),
                f(med([v.get("SF_C") for v in vs])), f(med([v.get("recA") for v in vs])),
                f(med([v.get("recC") for v in vs])), f(med([v.get("recR") for v in vs])),
                ("%d/%d" % (sum(rsv), len(rsv))) if rsv else "-"))
        L += ["", "### Paired against (a): per-target deltas (arm - a), median [sign counts +/-/=]", "",
              "| arm | pass lost / gained | d Z | d A | d C | d nec | d dev | d SF_C | d rec A | d rec C |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms_all[1:]:
            ks = [k for k in keys_a if "pass" in V[k][arm] and "pass" in V[k]["a_prod"]]
            if not ks:
                continue

            def pa(k, arm_):
                v = V[k][arm_]
                return bool(v["pass"]) and not v["vac"]
            lost = sum(pa(k, "a_prod") and not pa(k, arm) for k in ks)
            gained = sum(pa(k, arm) and not pa(k, "a_prod") for k in ks)
            cells = []
            for m in ("Z", "A", "C", "nec", "dev", "SF_C", "recA", "recC"):
                d = [(V[k][arm].get(m) - V[k]["a_prod"].get(m))
                     if V[k][arm].get(m) is not None and V[k]["a_prod"].get(m) is not None else None for k in ks]
                cells.append("%s [%s]" % (f(med(d)), signs(d)))
            L.append("| %s | %d / %d | %s |" % (ARM_LABEL[arm], lost, gained, " | ".join(cells)))
        # sensitivity: low vs matched control at the same dose
        L += ["", "### Dose sensitivity: low vs matched control at the same multiplier (per target, |metric - a|)", "",
              "Same number of members changed in both (control = the random near-1 set of arm d). Per target: "
              "|x_low - x_a| - |x_ctl - x_a|; + means the low members move the score more.", "",
              "| multiplier | metric | median low |dx| | median ctl |dx| | median diff | sign counts |",
              "|---|---|---|---|---|---|"]
        for c_, la, lc in (("0 (removed)", "c_low0", "d_ctl0"), ("2", "low_x2", "ctl_x2")):
            for m in ("Z", "A", "C", "nec", "dev", "SF_C"):
                dl, dc, dd = [], [], []
                for k in keys_a:
                    a0, xl, xc = V[k]["a_prod"].get(m), V[k][la].get(m), V[k][lc].get(m)
                    if None in (a0, xl, xc):
                        continue
                    dl.append(abs(xl - a0)); dc.append(abs(xc - a0)); dd.append(abs(xl - a0) - abs(xc - a0))
                if dd:
                    L.append("| %s | %s | %s | %s | %s | %s |" % (c_, m, f(med(dl)), f(med(dc)), f(med(dd)), signs(dd)))
        # dose-response curve of the low members
        L += ["", "### Dose-response, low members' alpha x c (median over targets)", "",
              "| c | arm | pass | Z | A | C | nec | dev | SF_C |", "|---|---|---|---|---|---|---|---|---|"]
        for c_, arm in (("0", "c_low0"), ("0.5", "low_x0.5"), ("1", "a_prod"), ("1/alpha", "b_low1"), ("2", "low_x2")):
            vs = [V[k][arm] for k in keys_a if "pass" in V[k][arm]]
            if vs:
                L.append("| %s | %s | %d/%d | %s | %s | %s | %s | %s | %s |" % (
                    c_, arm, sum(bool(v["pass"]) and not v["vac"] for v in vs), len(vs),
                    *(f(med([v[l_] for v in vs])) for l_, _ in HEAD), f(med([v["dev"] for v in vs])),
                    f(med([v.get("SF_C") for v in vs]))))
        L += ["", "### Per target", "",
              "| target | n | low | ctl removed | a: Z/A/C/nec | b | c | d | pass a b c d x0.5 x2 ctlx2 | SF_C a b c d |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for k in keys_a:
            r = by[k]["arms"]

            def zacn(arm):
                v = V[k][arm]
                return "/".join(f(v.get(l_), 2) for l_, _ in HEAD) if "pass" in v else "-"
            pp = " ".join(("P" if (V[k][arm].get("pass") and not V[k][arm].get("vac")) else "F") if "pass" in V[k][arm]
                          else "." for arm in arms_all)
            L.append("| %s | %s | %s | %s%s | %s | %s | %s | %s | %s | %s |" % (
                k, V[k]["a_prod"].get("n"), r.get("n_low"), r.get("n_ctl_removed"), " (short)" if r.get("ctl_short")
                else "", zacn("a_prod"), zacn("b_low1"), zacn("c_low0"), zacn("d_ctl0"), pp,
                " ".join(f(V[k][arm].get("SF_C"), 3) for arm in ("a_prod", "b_low1", "c_low0", "d_ctl0"))))
        # sanity: (a) vs the 11-07 production values
        L += ["", "## Checks", ""]
        diffs = {l_: [] for l_, _ in HEAD}
        dsf = []
        for k in keys_a:
            q = panel.get(k)
            if not q:
                continue
            for l_, _ in HEAD:
                if V[k]["a_prod"].get(l_) is not None:
                    diffs[l_].append(abs(V[k]["a_prod"][l_] - q[l_]))
            if V[k]["a_prod"].get("SF_C") is not None and q.get("sf_fit_C") is not None:
                dsf.append(abs(V[k]["a_prod"]["SF_C"] - q["sf_fit_C"]))
        L.append("- (a) rescored here vs the pod's 11-07 values (exp 130 per_target.csv): max |diff| " + ", ".join(
            "%s %s (median %s)" % (l_, f(max(v) if v else None, 4), f(med(v), 4)) for l_, v in diffs.items())
                 + "; SF_C max |diff| %s (median %s)" % (f(max(dsf) if dsf else None, 4), f(med(dsf), 4)))
    fd = [x for v in by.values() if "reads" in v for x in (v["reads"].get("fd_check") or [])]
    if fd:
        L.append("- finite-difference check of the natural contributions: " + "; ".join(
            "%s ctb %.4g fd %.4g" % (x["site"], x["ctb"], x["fd"]) for x in fd))
    t = {ph: [v[ph]["t"] for v in by.values() if ph in v] for ph in ("reads", "arms", "sweep", "donor")}
    pk = {ph: [v[ph].get("peak_mb") for v in by.values() if ph in v and v[ph].get("peak_mb")] for ph in t}
    L.append("- time per target (s): " + ", ".join("%s median %s max %s" % (ph, f(med(v), 0), f(max(v) if v else None, 0))
                                                   for ph, v in t.items()))
    L.append("- peak allocated GPU MB: " + ", ".join("%s max %s" % (ph, max(v) if v else "-") for ph, v in pk.items()))
    if errs:
        L.append("- error rows: " + "; ".join("%s %s: %s" % (r["seed"], r["phase"], r["error"][:120]) for r in errs))
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
