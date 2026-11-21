"""131: the target panel (CPU, reads only). 16 GPT-2 small targets from the 2026-11-07 production run (exp 130
out/per_target.csv: held-out strongest, non-vacuous, merge.passes), one per cell below, drawn with a fixed seed.

  hi    passing, silent firing pool C (variant fit) >= 0.2      10 cells (3 L0-4, 3 L5-7, 4 L8-9; kinds mixed)
  sel   passing, silent firing pool C <= 0.01 (selective)       4 cells (controls: O should cost them nothing)
  fail  failing, silent firing pool C >= 0.2                    2 cells
Deep cells take layers 8-9 only: GPT-2 L10-11 fits did not fit the 16 GB card in exp 121 (with its train patch) and
are the slowest; L8-9 ran there (README 121, "Hardware").
The CELLS order is the queue order (groups interleaved, so any prefix of the queue is mixed).

  python -I experiments/131-off-term/select_panel.py      -> out/panel.json, targets.txt (refuses to overwrite)
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
SEED = 20261118
# (group, band, kind) in queue order
CELLS = [("hi", "L0-4", "attn"), ("hi", "L5-7", "attn"), ("hi", "L8-9", "attn"), ("sel", "L0-4", "resid"),
         ("hi", "L0-4", "mlp"), ("hi", "L5-7", "resid"), ("fail", "L5-7", "attn"), ("hi", "L8-9", "resid"),
         ("sel", "L0-4", "mlp"), ("hi", "L0-4", "resid_mid"), ("hi", "L5-7", "mlp"), ("sel", "L5-7", "resid_mid"),
         ("hi", "L8-9", "mlp"), ("fail", "L8-9", "resid"), ("hi", "L8-9", "resid_mid"), ("sel", "L8-9", "mlp")]
KEEP = ("seed", "layer", "kind", "band", "n", "pass", "alpha_median", "free0_tk", "freeM_topk_tk", "freeN_topk_tk",
        "phi_sup_blind_tk", "phi_cf_alpha_blind_tk", "sf_fit_A", "sf_fit_C", "sf_fit_R", "sf_a1_C", "sf_empty_C",
        "rec_fit_A", "rec_fit_C", "rec_fit_R")


def band_ok(r, band):
    layer = int(r["layer"])
    lo, hi = {"L0-4": (0, 4), "L5-7": (5, 7), "L8-9": (8, 9)}[band]
    return lo <= layer <= hi


def group_ok(r, group):
    sf = float(r["sf_fit_C"]) if r["sf_fit_C"] not in ("", None) else None
    if sf is None:
        return False
    if group == "hi":
        return r["pass"] == "True" and sf >= 0.2
    if group == "sel":
        return r["pass"] == "True" and sf <= 0.01
    return r["pass"] == "False" and sf >= 0.2


def main():
    out = HERE / "out" / "panel.json"
    if out.exists() or (HERE / "targets.txt").exists():
        sys.exit("%s / targets.txt exist: not overwritten" % out)
    rows = [r for r in csv.DictReader(open(PER_TARGET)) if r["model"] == "GPT-2 small"]
    rng = random.Random(SEED)
    picked, used = [], set()
    for group, band, kind in CELLS:
        cand = sorted((r for r in rows if r["kind"] == kind and band_ok(r, band) and group_ok(r, group)
                       and r["seed"] not in used and (CTX / ("%s.pt" % r["seed"])).exists()), key=lambda r: r["seed"])
        if not cand:
            sys.exit("no candidate for %s %s %s" % (group, band, kind))
        r = rng.choice(cand)
        used.add(r["seed"])
        d = {k: r[k] for k in KEEP}
        for k in KEEP:
            if k not in ("seed", "kind", "band", "pass"):
                d[k] = float(d[k]) if d[k] not in ("", None) else None
        d.update(group=group, cell="%s/%s/%s" % (group, band, kind), n_candidates=len(cand), pass_=r["pass"] == "True")
        d.pop("pass")
        picked.append(d)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seed=SEED, source=str(PER_TARGET.relative_to(ROOT)), cells=CELLS,
                                   targets=picked), indent=1) + "\n")
    (HERE / "targets.txt").write_text("\n".join(d["seed"] for d in picked) + "\n")
    for d in picked:
        print("%-22s %-16s n %5d  Z %.2f A %.2f C %.2f nec %.2f  SF_C %.3f  phi_ind %s" % (
            d["cell"], d["seed"], d["n"], d["free0_tk"], d["freeM_topk_tk"], d["freeN_topk_tk"],
            d["phi_sup_blind_tk"], d["sf_fit_C"],
            "-" if d["phi_cf_alpha_blind_tk"] is None else "%.2f" % d["phi_cf_alpha_blind_tk"]))


if __name__ == "__main__":
    main()
