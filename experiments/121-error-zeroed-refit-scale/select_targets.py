"""121 target lists: scale 116 §1b (error-zeroed refit) up. CPU only.

Drawn from 116's §1 lists (116 select_targets.py output, which are uniform inside each depth band x site-kind cell),
so every target already has its paired production-circuit rows (116 rs_<family>, errors live / zeroed, this GPU).

  TuringLLM   9 per cell (81): the 2 pilot targets of 116 §1b (targets_refit.txt, already refitted) + 7 new per cell
  Pythia-70m  4 per cell (36 of 116's 54)
  GPT-2 small 3 per cell (36 of 116's 60)

The run lists hold only the targets still to fit, in ROUND-ROBIN cell order (one target per cell, then the next round),
so a run cut short by the time limit still covers every cell evenly.

  python -X utf8 experiments/121-error-zeroed-refit-scale/select_targets.py
  -> targets_<family>.txt (run order, new fits only), out/targets_<family>.csv (all, with cell and source)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E116 = ROOT / "experiments" / "116-error-ablated-and-case-checks"
SEED = 20261121
PER_CELL = {"turingllm": 9, "pythia-70m": 4, "gpt2-small": 3}


def main():
    (HERE / "out").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    pilot = set((E116 / "targets_refit.txt").read_text().split())
    for fam, n in PER_CELL.items():
        df = pd.read_csv(E116 / "out" / ("targets_%s.csv" % fam))
        keys = set((E116 / ("targets_%s.txt" % fam)).read_text().split())
        assert set(df.seed) == keys, fam
        rows, order = [], []
        cells = sorted(df.groupby(["band", "kind"]).groups)
        per = {}
        for band, kind in cells:
            cell = df[(df.band == band) & (df.kind == kind)].sort_values("seed")
            have = [s for s in cell.seed if fam == "turingllm" and s in pilot]
            rest = [s for s in cell.seed if s not in have]
            new = [rest[i] for i in rng.permutation(len(rest))[:n - len(have)]]   # random run order inside the cell
            per[(band, kind)] = new
            rows += [dict(seed=s, band=band, kind=kind, source="116_pilot") for s in have]
            rows += [dict(seed=s, band=band, kind=kind, source="121", run_rank=i) for i, s in enumerate(new)]
        for i in range(max(len(v) for v in per.values())):
            order += [v[i] for v in per.values() if i < len(v)]
        (HERE / ("targets_%s.txt" % fam)).write_text("\n".join(order) + "\n")
        pd.DataFrame(rows).to_csv(HERE / "out" / ("targets_%s.csv" % fam), index=False)
        print("%s: %d cells, %d targets (%d pilot, %d to fit)" % (fam, len(cells), len(rows),
                                                                 sum(r["source"] == "116_pilot" for r in rows), len(order)))


if __name__ == "__main__":
    main()
