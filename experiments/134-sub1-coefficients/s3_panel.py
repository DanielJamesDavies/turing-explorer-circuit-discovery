"""134 step 3: the target panel (CPU, reads only; ours). Writes out/s3_panel.json and s3_targets.txt (refuses to
overwrite).

Rule (fixed before any GPU result):
  source     GPT-2 small, the 2026-11-07 production run (WCM v4 main): exp 130 out/per_target.csv (held-out strongest,
             non-vacuous rows, one per target) with pass = True (062 merge.passes: Z, A, C in [0.75, 1.5], nec >= 0.9)
  members    113's member cache (members_gpt2-small_main.parquet: alpha = node amplitude, m = attribution_score):
             >= 10 members with alpha < 0.8 ("low") and >= 10 with alpha in [0.95, 1.05] ("control")
  depth      bands L0-4, L5-7, L8-9. L10-11 skipped (16 GB card; exp 121 / 131). L8-9 only circuits with n <= 1000
             (prefer smaller deep circuits: exp 131's L7-9 trips above ~15.3 GB were fits, but the reads here run a
             backward through every member site)
  files      the production circuit (.pt) and context record exist in the unpacked pod archive
  cells      3 bands x 4 site kinds (attn, resid_mid, mlp, resid) = 12 cells; up to 3 targets per cell drawn without
             replacement from the cell's candidates (sorted by key) with random.Random(SEED)
  order      round 1 = each cell's first draw (cells in a fixed order), round 2 = second draws, round 3 = third draws
             with the cell order shuffled by the same generator. The panel = the first 30 of this queue (rounds 1-2 =
             24 targets, 2 per cell, then 6 of round 3); the last 6 are extras, run only if time allows. Any prefix of the
             queue is stratified.

  python -I experiments/134-sub1-coefficients/s3_panel.py
"""
import csv
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# MODEL134 [gpt2-small] | pythia-70m (step 3b). The gpt2-small path is the step-3 panel, unchanged.
MODEL = os.environ.get("MODEL134", "gpt2-small")
PER_TARGET = ROOT / "experiments" / "130-selectivity-production" / "out" / "per_target.csv"
MEMBERS = ROOT / "experiments" / "113-coefficients-shape" / "out" / ("members_%s_main.parquet" % MODEL)
PROD = (ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives" / ("pub-%s-S10-results" % MODEL)
        / "experiments" / "082-public-runs" / ("out_%s" % MODEL))
SEED = 20261119
PANEL_N = 30
PER_CELL = 3
LOW, CTL = 0.8, (0.95, 1.05)
MIN_LOW, MIN_CTL = 10, 10
DEEP_MAX_N = 1000
BANDS = {"L0-4": (0, 4), "L5-7": (5, 7), "L8-9": (8, 9)}
KINDS = ("attn", "resid_mid", "mlp", "resid")
MODEL_NAME = "GPT-2 small"
PREFIX = "s3"
RULE_NOTE = ""
if MODEL == "pythia-70m":
    # step 3b (2026-11-20): Pythia-70m has 6 layers and sites attn / mlp / resid (no resid_mid), so 3 bands x 3 kinds
    # = 9 cells; 4 draws per cell (36 = the 30-target panel + 6 extras; round 4 is the shuffled one). No depth cap on n
    # (d_model 512, 6 layers: memory is not binding on the 16 GB card).
    BANDS = {"L0-1": (0, 1), "L2-3": (2, 3), "L4-5": (4, 5)}
    KINDS = ("attn", "mlp", "resid")
    PER_CELL = 4
    DEEP_MAX_N = None
    MODEL_NAME = "Pythia-70m"
    PREFIX = "s3py"
    RULE_NOTE = ("\nPYTHIA-70M (step 3b): source Pythia-70m rows of the same per_target.csv, members_pythia-70m_main.parquet, "
                 "archive pub-pythia-70m-S10-results; bands L0-1, L2-3, L4-5 x kinds attn, mlp, resid (9 cells); "
                 "4 draws per cell (round 4 shuffled); no n cap; panel = first 30 of 36.\n")
elif MODEL != "gpt2-small":
    sys.exit("MODEL134 %r: gpt2-small | pythia-70m" % MODEL)


def main():
    out = HERE / "out" / ("%s_panel.json" % PREFIX)
    tgt = HERE / ("%s_targets.txt" % PREFIX)
    if out.exists() or tgt.exists():
        sys.exit("%s / %s exist: not overwritten" % (out, tgt))
    rows = [r for r in csv.DictReader(open(PER_TARGET)) if r["model"] == MODEL_NAME and r["pass"] == "True"]
    m = pd.read_parquet(MEMBERS)
    g = m.groupby("seed")["alpha"].agg(n_low=lambda a: int((a < LOW).sum()),
                                       n_ctl=lambda a: int(((a >= CTL[0]) & (a <= CTL[1])).sum()), n_mem="size")
    cnt = g.to_dict("index")
    rng = random.Random(SEED)
    draws = {}
    for band, (lo, hi) in BANDS.items():
        for kind in KINDS:
            cand = []
            for r in rows:
                L, k = int(r["layer"]), r["kind"]
                c = cnt.get(r["seed"])
                if not (lo <= L <= hi and k == kind and c):
                    continue
                if c["n_low"] < MIN_LOW or c["n_ctl"] < MIN_CTL:
                    continue
                if band == "L8-9" and DEEP_MAX_N is not None and int(float(r["n"])) > DEEP_MAX_N:
                    continue
                if not ((PROD / "main" / "circuits" / ("%s.pt" % r["seed"])).exists()
                        and (PROD / "ctx" / ("%s.pt" % r["seed"])).exists()):
                    continue
                cand.append(r)
            cand.sort(key=lambda r: r["seed"])
            pick = rng.sample(cand, min(PER_CELL, len(cand)))
            draws[(band, kind)] = (pick, len(cand))
    cells = list(draws)
    queue = []
    for rnd in range(PER_CELL):
        order = list(cells)
        if rnd == PER_CELL - 1:
            rng.shuffle(order)
        for cell in order:
            pick, nc = draws[cell]
            if rnd < len(pick):
                r = pick[rnd]
                c = cnt[r["seed"]]
                queue.append(dict(seed=r["seed"], band=cell[0], kind=cell[1], round=rnd + 1, n_candidates=nc,
                                  layer=int(r["layer"]), n=int(float(r["n"])), n_low=c["n_low"], n_ctl=c["n_ctl"],
                                  Z=float(r["free0_tk"]), A=float(r["freeM_topk_tk"]), C=float(r["freeN_topk_tk"]),
                                  nec=float(r["phi_sup_blind_tk"]),
                                  sf_fit_C=float(r["sf_fit_C"]) if r["sf_fit_C"] else None,
                                  in_panel=len(queue) < PANEL_N))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seed=SEED, panel_n=PANEL_N, rule=__doc__ + RULE_NOTE, source=str(PER_TARGET.relative_to(ROOT)),
                                   queue=queue), indent=1) + "\n")
    tgt.write_text("\n".join(q["seed"] for q in queue) + "\n")
    for i, q in enumerate(queue):
        print("%2d %s r%d %-5s %-9s %-18s n %4d low %4d ctl %4d  Z %.2f A %.2f C %.2f nec %.2f SF_C %s (cands %d)" % (
            i + 1, "P" if q["in_panel"] else "x", q["round"], q["band"], q["kind"], q["seed"], q["n"], q["n_low"],
            q["n_ctl"], q["Z"], q["A"], q["C"], q["nec"], q["sf_fit_C"], q["n_candidates"]))


if __name__ == "__main__":
    main()
