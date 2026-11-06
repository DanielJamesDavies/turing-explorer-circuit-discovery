"""Arena C: why are behavioural WCM circuits larger than SFC's? Diagnostic refits (not production changes), each a
variant of the run's behavioural WCM (Z + A + C + R, noembed, 400 steps), on a finer λ grid around the size cliff:

  base       the run's setting (the reference, refitted on the finer grid)
  ge2        fit only layers >= 2 (12 sites): SFC's evaluated scope, so layer 0-1 members cannot pad the size
  noZ        no zero-ablation term in training (mask_floor_source "pn": C primary, A, R): is Z what forces the size?
  steps1200  3x the training steps: has the L1 pressure finished pruning at 400?

Each variant is fitted and scored by wcm_marks.run into out/size_explore/<variant>/ (circuit JSONs + eval.jsonl, so
donor_tasks.py and the cross-scoring can read them; resumable; STOP_EPOCH / STOP_AT respected).

  ./.venv/bin/python experiments/107-true-baselines/arena_c/size_explore.py --variants base,ge2,noZ,steps1200
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402

VARIANTS = {
    "base": dict(sites="noembed", steps=400, wcm={}),
    "ge2": dict(sites="ge2", steps=400, wcm={}),
    "noZ": dict(sites="noembed", steps=400, wcm=dict(mask_floor_source="pn")),
    "steps1200": dict(sites="noembed", steps=1200, wcm={}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tasks", default="simple,rc")
    ap.add_argument("--lams", default="3e-5,5e-5,1e-4,1.5e-4,2e-4,3e-4")
    ap.add_argument("--means", default="per_position")
    a = ap.parse_args()
    out = HERE / "out" / "size_explore"
    base = dict(WM.V3_LOGIT)
    lams = [float(x) for x in a.lams.split(",")]
    engines = {}
    try:
        for v in a.variants.split(","):
            cfg = VARIANTS[v]
            if cfg["sites"] not in engines:
                engines[cfg["sites"]] = WM.make_engine(cfg["sites"], means=a.means)
            WM.V3_LOGIT.clear()
            WM.V3_LOGIT.update(base, **cfg["wcm"])
            WM.run(a.tasks.split(","), lams, "sfc", cfg["steps"], cfg["sites"], out=out / v,
                   eng=engines[cfg["sites"]], means=a.means)
    finally:
        WM.V3_LOGIT.clear()
        WM.V3_LOGIT.update(base)


if __name__ == "__main__":
    main()
