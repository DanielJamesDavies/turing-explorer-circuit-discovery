"""115 pilot target list: scored, non-vacuous TuringLLM production targets (WCM v4, T3 main), stratified by depth band x
site kind, with passes and fails in every cell. CPU only.

Reads the production eval rows (held-out strongest) through 109's lib109 (062/merge.py pass rule and vacuous rule,
imported, not re-implemented). Per band (L0-4 / L5-7 / L8-11) x kind (attn / mlp / resid) cell it draws N_PASS passes
and N_FAIL fails with a fixed seed (fail-enriched by design: the pilot needs both directions of flip; the sample pass
rate is therefore NOT the production pass rate).

  python -X utf8 experiments/115-docdisjoint-rescore-pilot/select_targets.py
Writes targets_pilot.txt (one key per line, sorted) and out/targets_pilot.csv (production scores per target).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "109-pod-run-analysis"))
import lib109 as L  # noqa: E402

SEED = 20261115
N_PASS, N_FAIL = 3, 1


def production_rows():
    dirs = [d / "main" for d in L.find("turingllm", "out") if (d / "main").exists()]
    df = L.frame(L.read_rows(dirs, "eval"))
    df = df[df.held == "strong"].copy()
    df["vacuous"] = L.vacuous(df).values
    df["passes"] = (L.passes(df) & ~df["vacuous"]).values
    return L.add_site(df, "turingllm")


def main():
    df = production_rows()
    ok = df[~df.vacuous]
    print("production held-out strongest rows: %d, non-vacuous %d, pass %.1f%%"
          % (len(df), len(ok), 100 * ok.passes.mean()))
    rng = np.random.default_rng(SEED)
    pick = []
    for band, _, _ in L.bands_of("turingllm"):
        for kind in ("attn", "mlp", "resid"):
            cell = ok[(ok.band == band) & (ok.kind == kind)].sort_values("seed")
            for want, n in ((True, N_PASS), (False, N_FAIL)):
                pool = cell[cell.passes == want].seed.tolist()
                pick += [pool[i] for i in sorted(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]
    sel = ok.set_index("seed").loc[sorted(pick)]
    (HERE / "targets_pilot.txt").write_text("\n".join(sel.index) + "\n")
    (HERE / "out").mkdir(exist_ok=True)
    cols = ["layer", "kind", "band", "n", "passes"] + L.HEAD + [L.NEC, L.IND, "a_pos_tk"]
    sel[cols].to_csv(HERE / "out" / "targets_pilot.csv")
    print("picked %d targets (%d pass, %d fail) -> targets_pilot.txt" % (len(sel), sel.passes.sum(), (~sel.passes).sum()))
    print(sel.groupby(["band", "kind"]).passes.agg(["size", "sum"]))


if __name__ == "__main__":
    main()
