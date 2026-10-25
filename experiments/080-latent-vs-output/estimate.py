"""Experiment 080 runtime estimate from the smoke timings (CPU only).

smoke.sh fits the two smoke targets (a shallow one and the deepest kind of target) at two training budgets (8 steps in
OUT_SMOKE, 40 in OUT_TIMING). Per fitted arm the fit time is modelled as
    t_fit = a0 + a1 * sites + steps * (b0 + b1 * sites)
(sites = the target's upstream SAE sites, i.e. the masked sites; the logit objective runs the full model forward and
backward at every step, so depth enters only through the masked sites), and the scoring time per circuit as
t = d0 + d1 * sites. Both are projected over the target lists at 400 steps.

  PYTHONPATH=src python experiments/080-latent-vs-output/estimate.py
  env: OUT_SMOKE (out_smoke)  OUT_TIMING (out_smoke_timing)  CONFIG (config_main.json; absent: 4 lambdas per arm)
       CAL_FITS (15 per arm: 3 lr x 5 lambda, calibrate.py's default grid)  STARTUP_S (180)
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT_SMOKE = Path(os.environ.get("OUT_SMOKE", str(HERE / "out_smoke")))
OUT_TIMING = Path(os.environ.get("OUT_TIMING", str(HERE / "out_smoke_timing")))
CONFIG = Path(os.environ.get("CONFIG", str(HERE / "config_main.json")))
CAL_FITS = int(os.environ.get("CAL_FITS", 15))         # per arm: 3 lr x 5 lambda
STARTUP_S = float(os.environ.get("STARTUP_S", 180))
KIND_I = {"attn": 0, "mlp": 1, "resid": 2}


def sites(key):
    l, k, _ = C.parse(key)
    return 3 * l + KIND_I[k]                         # TuringLLM: every kind at lower layers + earlier kinds here


def rows(out):
    rs = []
    for f in glob.glob(str(out / "eval.shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not r.get("error") and not r.get("skip"):
                rs.append(r)
    return rs


def main():
    rs = rows(OUT_SMOKE) + rows(OUT_TIMING)
    fits = [r for r in rs if r["arm"] != "LAT" and r.get("fresh_fit") and r.get("t_fit")]
    if not fits:
        sys.exit("no fresh fit timings in %s / %s" % (OUT_SMOKE, OUT_TIMING))
    model = {}
    for arm in sorted({r["arm"] for r in fits}):
        fr = [r for r in fits if r["arm"] == arm]
        X = np.array([[1, sites(r["seed"]), r["steps"], r["steps"] * sites(r["seed"])] for r in fr], dtype=float)
        y = np.array([r["t_fit"] for r in fr], dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        model[arm] = coef
        print("%s fit time: %.1f + %.2f x sites + steps x (%.4f + %.5f x sites) s   (%d fits; residual rms %.1f s)" % (
            arm, *coef, len(fr), float(np.sqrt(np.mean((X @ coef - y) ** 2)))))
        for r in fr:
            print("    %-16s sites %2d steps %3d lam %-7s t_fit %6.1f  (model %6.1f)  peak %s MB" % (
                r["seed"], sites(r["seed"]), r["steps"], r["lam"], r["t_fit"],
                float(np.array([1, sites(r["seed"]), r["steps"], r["steps"] * sites(r["seed"])]) @ coef),
                r.get("peak_mb_fit")))
    sc = [r for r in rs if r.get("t_score_lat") is not None and r.get("t_score_out") is not None]
    Xs = np.array([[1, sites(r["seed"])] for r in sc], dtype=float)
    both = np.linalg.lstsq(Xs, np.array([r["t_score_lat"] + r["t_score_out"] for r in sc]), rcond=None)[0]
    outo = np.linalg.lstsq(Xs, np.array([r["t_score_out"] for r in sc]), rcond=None)[0]
    print("scoring per circuit: both endpoints %.1f + %.2f x sites s; output only %.1f + %.2f x sites s (%d rows)" % (
        *both, *outo, len(sc)))
    t_fit = lambda arm, s, steps=C.PROD_STEPS: float(np.array([1, s, steps, steps * s]) @ model[arm])  # noqa: E731
    t_sc = lambda s, coef: float(np.array([1, s]) @ coef)  # noqa: E731

    main_t = [k for k in (HERE / "targets.txt").read_text().split() if k]
    cal_t = [k for k in (HERE / "targets_calib.txt").read_text().split() if k]
    if CONFIG.exists():
        cfg = C.load_run_config(CONFIG)
        grid = {a: len(v["lams"]) for a, v in cfg["arms"].items() if a in model}
        src = CONFIG.name
    else:
        grid = {a: C.GRID_LEN for a in ("OUT_W", "OUT_U") if a in model}
        src = "no %s yet: %d lambdas per arm assumed" % (CONFIG.name, C.GRID_LEN)
    tot = STARTUP_S
    for k in main_t:
        s = sites(k)
        tot += t_sc(s, both)                                          # LAT re-score
        tot += sum(n * (t_fit(a, s) + t_sc(s, both)) for a, n in grid.items())
    n_fits = len(main_t) * sum(grid.values())
    print("\nmain run (%d targets, %s; %s): %d fits at %d steps + %d scorings -> %.1f h on this GPU" % (
        len(main_t), grid, src, n_fits, C.PROD_STEPS, n_fits + len(main_t), tot / 3600))
    cal = STARTUP_S
    for k in cal_t:
        s = sites(k)
        cal += sum(CAL_FITS * (t_fit(a, s) + t_sc(s, outo)) for a in ("OUT_W", "OUT_U") if a in model)
    print("calibration (%d targets x %d fits per arm x %d arms, output scorer only) -> %.1f h on this GPU" % (
        len(cal_t), CAL_FITS, len([a for a in ("OUT_W", "OUT_U") if a in model]), cal / 3600))
    mean_fit = np.mean([t_fit("OUT_W", sites(k)) for k in main_t]) if "OUT_W" in model else float("nan")
    print("mean OUT_W fit at 400 steps over the main targets: %.0f s (L-dependence via masked sites)" % mean_fit)


if __name__ == "__main__":
    main()
