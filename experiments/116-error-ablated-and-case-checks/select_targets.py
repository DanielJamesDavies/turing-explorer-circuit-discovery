"""116 target lists for the error-ablated re-score (DAN-214 §1). CPU only.

Scored, non-vacuous production targets (WCM v4, held-out strongest), drawn uniformly at random inside each depth band x
site-kind cell (NOT fail-enriched, unlike 115), fixed seed. Pass rule and vacuous rule: 062/merge.py through 109's
lib109 (imported). Each cell's production size is written next to the list so a population-weighted rate can be formed.

  python -X utf8 experiments/116-error-ablated-and-case-checks/select_targets.py
  -> targets_<model>.txt, out/targets_<model>.csv, out/cells_<model>.csv
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

SEED = 20261115
N_REFIT_CELL = 2                                                   # §1b refit pilot: 9 cells -> 18
PER_CELL = {"turingllm": 27, "pythia-70m": 6, "gpt2-small": 5}     # TL 9 cells -> 243; Pythia 9 -> 54; GPT-2 12 -> 60


def production_rows(model):
    df = L.main_eval(model)
    df = df[df.held == "strong"].copy()
    return df


def main():
    (HERE / "out").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    for model, n in PER_CELL.items():
        df = production_rows(model)
        if df.empty:
            print("%s: no production rows" % model)
            continue
        ok = df[~df.vac]
        pick, cells = [], []
        for band, _, _ in L.bands_of(model):
            for kind in sorted(ok.kind.unique()):
                cell = ok[(ok.band == band) & (ok.kind == kind)].sort_values("seed")
                if cell.empty:
                    continue
                idx = sorted(rng.choice(len(cell), size=min(n, len(cell)), replace=False))
                pick += cell.seed.iloc[idx].tolist()
                cells.append(dict(band=band, kind=kind, n_prod=len(cell), pass_prod=float(cell["pass"].mean()),
                                  n_pick=len(idx)))
        sel = ok.set_index("seed").loc[sorted(pick)]
        (HERE / ("targets_%s.txt" % model)).write_text("\n".join(sel.index) + "\n")
        cols = ["layer", "kind", "band", "n", "pass"] + L.HEAD + [L.NEC, "a_pos_tk"]
        sel[cols].to_csv(HERE / "out" / ("targets_%s.csv" % model))
        import pandas as pd
        pd.DataFrame(cells).to_csv(HERE / "out" / ("cells_%s.csv" % model), index=False)
        print("%s: production non-vacuous %d (pass %.1f%%); picked %d (pass %.1f%%)"
              % (model, len(ok), 100 * ok["pass"].mean(), len(sel), 100 * sel["pass"].mean()))
        if model == "turingllm":
            # §1b refit pilot: N_REFIT_CELL per cell, drawn from the §1 list (so every pilot target has its §1 rows)
            r2 = np.random.default_rng(SEED + 1)
            sub = []
            for (band, kind), cell in sel.reset_index().groupby(["band", "kind"]):
                cell = cell.sort_values("seed")
                sub += cell.seed.iloc[sorted(r2.choice(len(cell), size=min(N_REFIT_CELL, len(cell)),
                                                       replace=False))].tolist()
            (HERE / "targets_refit.txt").write_text("\n".join(sorted(sub)) + "\n")
            print("refit pilot: %d targets -> targets_refit.txt" % len(sub))


if __name__ == "__main__":
    main()
