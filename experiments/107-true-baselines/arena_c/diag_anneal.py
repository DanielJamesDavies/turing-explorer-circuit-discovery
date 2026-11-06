"""Arena C diagnostic: does the gate anneal break the behavioural WCM fits at the end of training? The 1,200-step
simple λ 1e-5 fit (per-position means) was near-perfect at mid-run (zero term 2e-4) and collapsed by the last step
(zero term 0.46, scored Z 0.008): the anneal (binarize="anneal", temperature 1 -> 0.05) reaches its floor only at the
last step under anneal_reach_frac 1.0, so the gates harden with no steps left to recover (learned_mask's own comment:
"the mask gets no time AT final sharpness to settle").

The full curve of that fit shows a sudden blow-up at step ~900 (zero term 0.0015 -> 12.2), never recovered, rather than
a slow hardening: with m = sigmoid(theta / T) and Adam moving theta by ~lr per step, a constant lr moves the gates
1/T = up to 20x further per step as T falls, so late steps flip gates wholesale. Variants (engine knobs, all existing):

  reach1     the run's schedule (anneal to T = 0.05 at the last step, constant lr)
  reach0.6   T reaches 0.05 at 60% of the run and holds
  cosine     lr cosine 1 -> 0.05 (lr_min_frac), tracking T's 1 -> 0.05, so lr / T stays ~constant
  ampl1      amp_l1 1e-4: a price on |alpha - 1| (the fit's final max alpha was 19.6)

Refits one (task, λ, steps) per variant and saves the FULL training curves (the run's circuit JSONs keep first / mid /
last only), plus the scored Z / A / C / necessity, to out/diag_anneal/.

  ./.venv/bin/python experiments/107-true-baselines/arena_c/diag_anneal.py --task simple --lam 1e-5 --steps 1200
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402


VARIANTS = {
    "reach1": dict(anneal_reach_frac=1.0),
    "reach0.6": dict(anneal_reach_frac=0.6),
    "cosine": dict(anneal_reach_frac=1.0, lr_schedule="cosine"),
    "ampl1": dict(anneal_reach_frac=1.0, amp_l1=1e-4),
    # lr * 0.05**progress: the anneal's own shape, so lr / T is constant (cosine's peaks ~2.5x mid-run)
    "geometric": dict(anneal_reach_frac=1.0, lr_schedule="geometric", lr_min_frac=0.05),
}


def _list(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _list(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_list(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="simple")
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--variants", default="reach1,reach0.6")
    ap.add_argument("--means", default="per_position", choices=WM.MEANS_MODES)
    a = ap.parse_args()
    C, _ = WM._import_081()
    out = HERE / "out" / "diag_anneal"
    out.mkdir(parents=True, exist_ok=True)
    eng = WM.make_engine("noembed", means=a.means)
    T = WM.build_task(a.task, "sfc")
    B = eng.batch(T)
    for v in a.variants.split(","):
        # the first two runs (before the variants existed) are named reach1 / reach0.6
        path = out / ("%s_%s_%g_s%d_%s.json" % (a.task, a.means, a.lam, a.steps, v))
        if path.exists():
            print("skip", path.name, flush=True)
            continue
        if WM.stop_passed(None, None):
            print("STOP_EPOCH passed", flush=True)
            return
        eng.extra_kw = dict(VARIANTS[v])
        try:
            alphas, prov, t_fit, _ = eng.fit(B, "W", a.lam, C.PROD_LR, a.steps)
        finally:
            eng.extra_kw = None
        sc = eng.score(B, alphas)
        keep = ("task_free0", "task_freeM_topk", "task_freeN_topk", "task_nec", "task_pass", "task_free0_a1")
        rec = dict(task=a.task, means=a.means, lam=a.lam, steps=a.steps, variant=v, overrides=VARIANTS[v],
                   n_latents=sum(len(d) for d in alphas.values()), t_fit=round(t_fit, 1),
                   score={k: sc.get(k) for k in keep}, curve=_list(prov.get("curve") or {}))
        path.write_text(json.dumps(rec) + "\n")
        print("%s %s reach %s: %d latents | Z %.3f A %.3f C %.3f nec %.3f | fit %.0fs" % (
            a.task, v, VARIANTS[v].get("anneal_reach_frac"), rec["n_latents"], *(sc.get(k) or float("nan") for k in keep[:4]), t_fit), flush=True)


if __name__ == "__main__":
    main()
