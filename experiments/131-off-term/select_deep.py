"""131 deep slice (2026-11-20, DAN-222): the deep-target panel (CPU, reads only). GPT-2 small targets at layers 8-11
from the 2026-11-07 production run (exp 130 out/per_target.csv: held-out strongest, non-vacuous), PASSING only,
stratified by site kind, drawn with seed 20261120. Two panels are written, because whether a layer-10/11 fit fits the
16 GB card is decided by one timed GPU fit (README "Deep slice"):

  L8-11   cells alternate the bands L8-9 / L10-11 within each kind
  L8-9    the same cells with every band L8-9

Groups:
  hi    passing, production silent firing pool C (variant fit) >= 0.2      9 cells (attn x3, mlp, resid, resid_mid x2)
  sel   passing, production silent firing pool C <= 0.05 (controls)         3 cells (mlp, resid, resid_mid)
The CELLS order is the queue order (kinds and groups interleaved, so any prefix of the queue is mixed; the queue is
target-major, so a prefix is a set of complete targets).

  python -I experiments/131-off-term/select_deep.py   -> out_deep/panel_<L8-11|L8-9>.json, targets_<...>.txt
"""
import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PER_TARGET = ROOT / "experiments" / "130-selectivity-production" / "out" / "per_target.csv"
CTX = (ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives" / "pub-gpt2-small-S10-results" / "experiments"
       / "082-public-runs" / "out_gpt2-small" / "ctx")
OUT = HERE / "out_deep"
SEED = 20261120
# (group, band in the L8-11 panel, kind) in queue order; the L8-9 panel uses band L8-9 for every cell
CELLS = [("hi", "L10-11", "mlp"), ("hi", "L8-9", "attn"), ("sel", "L8-9", "resid_mid"), ("hi", "L10-11", "resid"),
         ("hi", "L8-9", "resid_mid"), ("hi", "L10-11", "attn"), ("sel", "L10-11", "mlp"), ("hi", "L8-9", "resid"),
         ("hi", "L10-11", "resid_mid"), ("hi", "L8-9", "attn"), ("sel", "L8-9", "resid"), ("hi", "L8-9", "mlp")]
GROUPS = {"hi": "passing, production SF_C >= 0.2", "sel": "passing, production SF_C <= 0.05 (controls)"}
KEEP = ("seed", "layer", "kind", "band", "n", "pass", "alpha_median", "free0_tk", "freeM_topk_tk", "freeN_topk_tk",
        "phi_sup_blind_tk", "phi_cf_alpha_blind_tk", "sf_fit_A", "sf_fit_C", "sf_fit_R", "sf_a1_C", "sf_empty_C",
        "rec_fit_A", "rec_fit_C", "rec_fit_R")


def band_ok(r, band):
    lo, hi = {"L8-9": (8, 9), "L10-11": (10, 11)}[band]
    return lo <= int(r["layer"]) <= hi


def group_ok(r, group):
    if r["pass"] != "True" or r["sf_fit_C"] in ("", None):
        return False
    sf = float(r["sf_fit_C"])
    return sf >= 0.2 if group == "hi" else sf <= 0.05


def pick(rows, name, deep):
    rng = random.Random(SEED)
    picked, used = [], set()
    for group, band, kind in CELLS:
        band = band if deep else "L8-9"
        cand = sorted((r for r in rows if r["kind"] == kind and band_ok(r, band) and group_ok(r, group)
                       and r["seed"] not in used and (CTX / ("%s.pt" % r["seed"])).exists()), key=lambda r: r["seed"])
        if not cand:
            sys.exit("%s: no candidate for %s %s %s" % (name, group, band, kind))
        r = rng.choice(cand)
        used.add(r["seed"])
        d = {k: r[k] for k in KEEP}
        for k in KEEP:
            if k not in ("seed", "kind", "band", "pass"):
                d[k] = float(d[k]) if d[k] not in ("", None) else None
        d.update(group=group, cell="%s/%s/%s" % (group, band, kind), n_candidates=len(cand), pass_=r["pass"] == "True")
        d.pop("pass")
        picked.append(d)
    return picked


def main():
    rows = [r for r in csv.DictReader(open(PER_TARGET)) if r["model"] == "GPT-2 small"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, deep in (("L8-11", True), ("L8-9", False)):
        pj, tx = OUT / ("panel_%s.json" % name), OUT / ("targets_%s.txt" % name)
        if pj.exists() or tx.exists():
            sys.exit("%s / %s exist: not overwritten" % (pj, tx))
        picked = pick(rows, name, deep)
        pj.write_text(json.dumps(dict(seed=SEED, source=str(PER_TARGET.relative_to(ROOT)), panel=name,
                                      cells=[(g, b if deep else "L8-9", k) for g, b, k in CELLS], groups=GROUPS,
                                      targets=picked), indent=1) + "\n")
        tx.write_text("\n".join(d["seed"] for d in picked) + "\n")
        print("panel %s" % name)
        for d in picked:
            print("  %-22s %-18s n %5d  Z %.2f A %.2f C %.2f nec %.2f  SF_C %.3f  (%d candidates)" % (
                d["cell"], d["seed"], d["n"], d["free0_tk"], d["freeM_topk_tk"], d["freeN_topk_tk"],
                d["phi_sup_blind_tk"], d["sf_fit_C"], d["n_candidates"]))


if __name__ == "__main__":
    main()
