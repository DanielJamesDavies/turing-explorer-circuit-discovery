"""Arena C: how much does a behavioural WCM fit depend on its randomness? The fit has no random initialisation
(theta_init is uniform); its only random source is R's donor draws (run_learned_mask resample_seed; the presentation
order is seeded by the task name and is kept fixed). This refits the run's grid with other donor seeds into
out/seed_variance/seed_<s>/ (wcm_marks.run: resumable per circuit, STOP_EPOCH / STOP_AT respected), so each (task, λ)
gets 1 + len(seeds) fits: the production run (seed RESAMPLE_SEED) and these.

  ./.venv/bin/python experiments/107-true-baselines/arena_c/seed_variance.py --seeds 1,2,3 --means pooled
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--tasks", default="simple,nounpp,within_rc,rc")
    ap.add_argument("--lams", default="3e-6,1e-5,3e-5,1e-4,3e-4,1e-3")
    ap.add_argument("--means", default="pooled")
    ap.add_argument("--sites", default="noembed")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr_schedule", default=None, choices=(None, "constant", "cosine", "geometric"),
                    help="override the run's lr schedule; the fits go to out/seed_variance_<schedule>/")
    a = ap.parse_args()
    out = HERE / "out" / ("seed_variance_%s" % a.lr_schedule if a.lr_schedule else "seed_variance")
    eng = WM.make_engine(a.sites, means=a.means)
    if a.lr_schedule:
        WM.V3_LOGIT["lr_schedule"] = a.lr_schedule
    base = dict(WM.V3_LOGIT)
    lams = [float(x) for x in a.lams.split(",")]
    try:
        for s in a.seeds.split(","):
            WM.V3_LOGIT.clear()
            WM.V3_LOGIT.update(base, resample_seed=int(s))
            # run() stops before a new fit once the deadline passes, and so does every later call
            WM.run(a.tasks.split(","), lams, "sfc", a.steps, a.sites, out=out / ("seed_%s" % s), eng=eng,
                   means=a.means)
    finally:
        WM.V3_LOGIT.clear()
        WM.V3_LOGIT.update(base)


if __name__ == "__main__":
    main()
