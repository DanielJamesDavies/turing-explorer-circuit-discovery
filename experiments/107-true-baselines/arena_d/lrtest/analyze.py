"""Arena D lr-budget test: the CPU analysis (README.md, "Pre-stated reading"). Reads every arm's eval.jsonl (wcm_gemma.run's
rows: the 081 task engine's own held-out scoring), the circuit JSONs' fit provenance, ledger.jsonl and gpu_mem.csv;
writes RESULTS.md + results.json.

  CUDA_VISIBLE_DEVICES= ./.venv/bin/python experiments/107-true-baselines/arena_d/lrtest/analyze.py [--out DIR] [--write DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOL = 0.02              # pre-stated: "equal" = within 0.02 on |Z - 1|, |A - 1|, |C - 1| and on necessity
SHARE = 2 / 3           # pre-stated: a family "helps" if >= 2/3 of its decided pairs are "better" ...
PAIRS = (("C0", "C1"), ("C0s2", "C1s2"))
FULL = {"gemma-2-2b-plt": 26 * 16384, "gemma-2-2b-gsres-gap6": 5 * 16384}
METRICS = (("Z", "task_free0"), ("A", "task_freeM_topk"), ("C", "task_freeN_topk"), ("nec", "task_nec"))


def load(out: Path):
    rows = []
    for ev in sorted((out / "runs").glob("*/*/eval.jsonl")):
        arm, fam = ev.parent.parent.name, ev.parent.name
        for line in open(ev):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            fit = {}
            cp = ev.parent / r.get("circuit", "")
            if r.get("circuit") and cp.exists():
                try:
                    fit = json.loads(cp.read_text()).get("fit", {})
                except Exception:  # noqa: BLE001
                    fit = {}
            d = dict(arm=arm, fam=fam, family=r["family"], task=r["sfc_task"], lam=float(r["lam"]), steps=r["steps"],
                     n=r.get("n_latents"), full=FULL.get(r["family"]), passed=r.get("task_pass"),
                     t_fit=r.get("t_fit"), s_step=(r["t_fit"] / r["steps"]) if r.get("t_fit") else None,
                     t_score=r.get("t_score"), peak_fit_mb=r.get("peak_fit_mb"), peak_score_mb=r.get("peak_score_mb"),
                     lr_peak=fit.get("lr_peak"), lr_sum=fit.get("lr_sum"), warmup_steps=fit.get("warmup_steps"),
                     weight_decay=fit.get("weight_decay"), loss_final=fit.get("loss_final"),
                     vacuous=r.get("task_vacuous"), task_full=r.get("task_full"))
            for k, f in METRICS:
                d[k] = r.get(f)
            rows.append(d)
    return rows


def dev(r, k):
    return None if r.get(k) is None else abs(r[k] - 1.0)


def compare(r0, r1):
    """One matched pair (same family, task, lambda): the pre-stated classification."""
    if any(r.get(k) is None for r in (r0, r1) for k, _ in METRICS) or r0["n"] is None or r1["n"] is None:
        return "undecided", "a metric is None (vacuous / missing)"
    worse = [k for k in ("Z", "A", "C") if dev(r1, k) > dev(r0, k) + TOL]
    if r1["nec"] < r0["nec"] - TOL:
        worse.append("nec")
    smaller = r1["n"] < r0["n"]
    if worse:
        return "worse_faith", "C1 worse on %s" % ",".join(worse)
    if smaller:
        return "better", "smaller (%d -> %d) at equal or better Z/A/C/nec" % (r0["n"], r1["n"])
    return "not_smaller", "equal or better faithfulness, not smaller (%d -> %d)" % (r0["n"], r1["n"])


def frontier(rows):
    """[descriptive, not in the verdict] each C1 point against C0's (log10 size -> worst |m - 1| over Z/A/C) curve of the
    same family and task, linearly interpolated in log size; None outside C0's size range."""
    out = []
    by = defaultdict(list)
    for r in rows:
        if r["arm"] in ("C0", "C1") and r["n"] and all(r.get(k) is not None for k in ("Z", "A", "C")):
            by[(r["fam"], r["task"], r["arm"])].append(r)
    for (fam, task, arm), rs in sorted(by.items()):
        if arm != "C1":
            continue
        c0 = sorted(by.get((fam, task, "C0"), []), key=lambda r: r["n"])
        pts = [(math.log10(r["n"]), max(dev(r, k) for k in ("Z", "A", "C"))) for r in c0]
        for r in sorted(rs, key=lambda r: r["lam"]):
            x, y = math.log10(r["n"]), max(dev(r, k) for k in ("Z", "A", "C"))
            y0 = None
            for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
                if xa <= x <= xb:
                    y0 = ya if xb == xa else ya + (yb - ya) * (x - xa) / (xb - xa)
                    break
            out.append(dict(fam=fam, task=task, lam=r["lam"], n=r["n"], dev_c1=round(y, 4),
                            dev_c0_at_size=None if y0 is None else round(y0, 4),
                            side=None if y0 is None else ("C1 below C0 (better)" if y < y0 - TOL else
                                                          "C1 above C0 (worse)" if y > y0 + TOL else "on C0's curve")))
    return out


def gpu_peaks(out: Path):
    p = out / "gpu_mem.csv"
    peaks = {}
    if p.exists():
        for row in csv.reader(open(p)):
            try:
                st, m = row[1], int(row[2])
            except Exception:  # noqa: BLE001
                continue
            peaks[st] = max(peaks.get(st, 0), m)
    return peaks


def ledger(out: Path):
    p = out / "ledger.jsonl"
    rows = []
    if p.exists():
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return rows


def f3(x):
    return "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--write", default=None, help="where RESULTS.md / results.json go (default --out)")
    a = ap.parse_args()
    out = Path(a.out)
    wr = Path(a.write) if a.write else out
    rows = load(out)
    idx = {(r["arm"], r["fam"], r["task"], r["lam"]): r for r in rows}
    pairs = []
    for x0, x1 in PAIRS:
        for (arm, fam, task, lam), r0 in sorted(idx.items()):
            if arm != x0 or (x1, fam, task, lam) not in idx:
                continue
            r1 = idx[(x1, fam, task, lam)]
            cls, why = compare(r0, r1)
            pairs.append(dict(pair="%s/%s" % (x0, x1), fam=fam, task=task, lam=lam, cls=cls, why=why,
                              n0=r0["n"], n1=r1["n"], pass0=r0["passed"], pass1=r1["passed"],
                              c0_keeps_all=r0["n"] == r0["full"], c1_keeps_all=r1["n"] == r1["full"],
                              **{k + "0": r0[k] for k, _ in METRICS}, **{k + "1": r1[k] for k, _ in METRICS}))
    fams = sorted({p["fam"] for p in pairs})
    verdict = {}
    for fam in fams:
        ps = [p for p in pairs if p["fam"] == fam]
        dec = [p for p in ps if p["cls"] != "undecided"]
        nb = sum(p["cls"] == "better" for p in dec)
        lost = [p for p in ps if p["pass0"] and p["pass1"] is False]
        helps = bool(dec) and nb >= SHARE * len(dec) and not lost
        verdict[fam] = dict(n_pairs=len(ps), n_decided=len(dec), n_better=nb,
                            n_worse_faith=sum(p["cls"] == "worse_faith" for p in dec),
                            n_not_smaller=sum(p["cls"] == "not_smaller" for p in dec),
                            n_pass_lost=len(lost), helps=helps if dec else None)
    both = [verdict.get(f, {}).get("helps") for f in ("plt", "res")]
    overall = (None if any(h is None for h in both) else bool(all(both)))
    fr = frontier(rows)
    led = ledger(out)
    res = dict(rows=rows, pairs=pairs, verdict=verdict, overall_helps=overall, frontier=fr, gpu_peak_mib=gpu_peaks(out),
               gpu_wall_s=round(sum(float(r.get("wall", 0)) for r in led)), tol=TOL, share=SHARE)
    wr.mkdir(parents=True, exist_ok=True)
    (wr / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")

    L = ["# Arena D lr-budget test: results", "",
         "Pre-stated reading (README): v4's lr budget helps if it lowers size at equal or better Z / A / C / nec on "
         "both families. Pair = same family, task, lambda; 'equal' = within %.2f on |Z-1|, |A-1|, |C-1| and on nec; a "
         "family helps if >= 2/3 of its decided pairs are 'better' and no pair loses 081's pass. bf16 local, labelled."
         % TOL, "",
         "**Overall: %s**" % {True: "HELPS (both families)", False: "DOES NOT HELP (on at least one family)",
                              None: "UNDECIDED (a family has no decided pair)"}[overall], ""]
    L.append("| family | pairs | decided | better | worse faith | not smaller | pass lost | helps |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fam, v in verdict.items():
        L.append("| %s | %d | %d | %d | %d | %d | %d | %s |" % (fam, v["n_pairs"], v["n_decided"], v["n_better"],
                                                             v["n_worse_faith"], v["n_not_smaller"], v["n_pass_lost"],
                                                             v["helps"]))
    L += ["", "## Pairs", "", "| pair | fam | task | lam | n C0 -> C1 | Z | A | C | nec | pass | class |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        L.append("| %s | %s | %s | %g | %s -> %s%s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s |" % (
            p["pair"], p["fam"], p["task"], p["lam"], p["n0"], p["n1"],
            " (C0 keeps all)" if p["c0_keeps_all"] else "", f3(p["Z0"]), f3(p["Z1"]), f3(p["A0"]), f3(p["A1"]),
            f3(p["C0"]), f3(p["C1"]), f3(p["nec0"]), f3(p["nec1"]), p["pass0"], p["pass1"], p["cls"]))
    L += ["", "## Every fit", "",
          "| arm | family | task | lam | n | Z | A | C | nec | t_fit s | s/step | peak fit MB | lr peak | sum lr | warm-up | wd |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["fam"], r["task"], r["lam"], r["arm"])):
        L.append("| %s | %s | %s | %g | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["arm"], r["fam"], r["task"], r["lam"], r["n"], f3(r["Z"]), f3(r["A"]), f3(r["C"]), f3(r["nec"]),
            f3(r["t_fit"]), f3(r["s_step"]), r["peak_fit_mb"], f3(r["lr_peak"]), f3(r["lr_sum"]), r["warmup_steps"],
            f3(r["weight_decay"])))
    L += ["", "## Frontier (descriptive; not in the verdict)", "",
          "Worst |m - 1| over Z / A / C of each C1 fit vs C0's curve of the same task, interpolated at C1's size "
          "(log10). A larger lr budget at fixed lambda also prices sparsity harder, so 'smaller' alone can be a move "
          "along the lambda curve; this table says whether C1 lies off C0's curve.", "",
          "| fam | task | lam | n C1 | dev C1 | dev C0 at that size | side |", "|---|---|---|---|---|---|---|"]
    for x in fr:
        L.append("| %s | %s | %g | %s | %s | %s | %s |" % (x["fam"], x["task"], x["lam"], x["n"], f3(x["dev_c1"]),
                                                          f3(x["dev_c0_at_size"]), x["side"]))
    L += ["", "GPU wall in the ledger (loads + fits + scoring): %s s; nvidia-smi peak MiB per stage: %s"
          % (res["gpu_wall_s"], res["gpu_peak_mib"]), ""]
    (wr / "RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
