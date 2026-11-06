"""Arena C: behavioural WCM fitted along a λ PATH [ours: a fitting-procedure variant, not the run's protocol]. The
run fits every λ from the same cold start (all gates open, amplitudes 1); here λ goes up the grid and each fit starts
from the previous λ's final gates and amplitudes (learned_mask warm_state / return_state), as pathwise Lasso does.
Why: the cold 400-step fits are noisy (seed refits flip pass / fail and size up to 2-3x at λ >= 3e-5) and their size
curves are non-monotone; a path should give nested circuits that shrink with λ, and each fit inherits the previous
fits' steps.

Same objective, scope, means and scorer as wcm_marks.run (the circuit JSONs + eval.jsonl land in --out, so
donor_tasks.py / score_wcm_sfc.py / curves.py read them). The final state of each fit is saved next to its circuit
(<circuit>.state.pt), so a resumed path warm-starts from disk; the first λ of a path starts cold.

--hold (2026-11-02): a warm fit continues AT the final temperature (anneal_t0 0.05, held) with a constant lr of
warm_lr_frac (0.05) x the run's lr, i.e. the lr / T the cold anneal ends with. Without it each warm fit restarts the
anneal at T = 1, which re-softens the inherited gates (measured: the zero term 1e-4 at the end of one λ, 0.03-0.16 at
the start of the next; simple λ 1e-3 then collapsed, Z 0.14).

  ./.venv/bin/python experiments/107-true-baselines/arena_c/warm_path.py --tasks simple,rc --means per_position
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402

LAMS = "3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2"


def state_path(out: Path, task: str, mode: str, sites: str, means: str, lam: float, steps: int, arm: str = "W"):
    return out / "circuits" / task / mode / sites / means / ("%s_%g_s%d.state.pt" % (arm, lam, steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="simple,within_rc,rc,nounpp")
    ap.add_argument("--lams", default=LAMS)
    ap.add_argument("--means", default="per_position", choices=WM.MEANS_MODES)
    ap.add_argument("--steps", type=int, default=400, help="steps per λ (the path's total is len(lams) x steps)")
    ap.add_argument("--sites", default="noembed", choices=WM.SITE_SCOPES)
    ap.add_argument("--out", default=str(HERE / "out" / "warm_path"))
    ap.add_argument("--lr_schedule", default=None, choices=(None, "constant", "cosine", "geometric"),
                    help="override the run's lr schedule (constant); use a separate --out")
    ap.add_argument("--hold", action="store_true",
                    help="warm fits hold T = 0.05 at a constant lr of warm_lr_frac x the run's lr (see the docstring)")
    ap.add_argument("--warm_lr_frac", type=float, default=0.05)
    a = ap.parse_args()
    C, _ = WM._import_081()
    if a.lr_schedule:
        WM.V3_LOGIT["lr_schedule"] = a.lr_schedule
    out = Path(a.out)
    lams = sorted(float(x) for x in a.lams.split(","))
    eng = WM.make_engine(a.sites, means=a.means)
    fit0 = eng.fit
    box = {}

    def fit(B, arm, lam, lr, steps, *args, **kw):
        alphas, prov, t_fit, peak = fit0(B, arm, lam, lr, steps, *args, **kw)
        box["save"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(prov.pop("final_state"), box["save"])
        prov["warm_from"] = box["from"]
        return alphas, prov, t_fit, peak

    eng.fit = fit
    for task in a.tasks.split(","):
        prev = None
        for lam in lams:
            if WM.stop_passed(None, None):
                print("STOP_EPOCH passed: no new fit", flush=True)
                return
            sp = state_path(out, task, "sfc", a.sites, a.means, lam, a.steps)
            warm = torch.load(prev, map_location="cpu") if prev is not None and prev.exists() else None
            if prev is not None and warm is None:
                print("WARNING %s λ %g: no state at %s, this fit starts cold" % (task, lam, prev), flush=True)
            box.update(save=sp, **{"from": None if warm is None else str(prev.name)})
            eng.extra_kw = dict(return_state=True, warm_state=warm)
            lr = None
            if a.hold and warm is not None:
                eng.extra_kw.update(anneal_t0=0.05, lr_schedule="constant")
                lr = C.PROD_LR * a.warm_lr_frac
            try:
                WM.run([task], [lam], "sfc", a.steps, a.sites, out=out, eng=eng, means=a.means, lr=lr)
            finally:
                eng.extra_kw = None
            prev = sp


if __name__ == "__main__":
    main()
