"""Experiment 081, stage 2: a short, explicit calibration of the lambda grid (and, optionally, lr) for the W and U arms.

Why: the output endpoint's lambda scale is not the latent endpoint's (080's calibration exists for the same reason),
and a task circuit's size-faithfulness curve is only informative if the grid spans the pass threshold. The main run
never picks its own grid: run.py reads config_main.json, written only by an explicit ADOPT step.

No held-out prompt is ever seen here. Each calibration task is restricted to its TRAINING rows, which are re-split
75 / 25 by group (seed TASK_SEED + 1) into a fit set and a validation set (capped at HELDOUT_CAP); fits and scores use
only those (the fit set is the training pool: its 1,600 presentations, its A mean, its items' patch prompts as C).

Modes:
  (default)  fit + score every (task, arm, lr, lambda) of the grid exactly as run.py fits (engine.Engine.fit /
             score), rows -> OUT_CAL/eval.jsonl, curves -> OUT_CAL/train.jsonl. Resumable; error rows are retried.
  REPORT=1   CPU: per arm x lr the failure rate (fit failed, or validation objective Z missing / < 0.05), monotonicity
             in lambda, median size, median task Z and pass rate per lambda; the recommendation (task_common.recommend:
             the lr with the fewest failures, then the GRID_LEN consecutive lambdas whose median task Z best covers
             [0.5, 1.0], preferring larger lambdas, flagged unless the window brackets Z = 0.8)
             -> OUT_CAL/recommendation.json (adopted: false).
  ADOPT=1    write CONFIG (config_main_<MODEL>.json) from the recommendation; refuses one with warnings unless FORCE=1.
  DRY=1      print the plan.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/081-task-circuits/calibrate.py
  env: MODEL (gpt2-small)  OUT_CAL (calib/<MODEL>)  CAL_TASKS (ioi,greater_than,sva_across_rc: the tasks with
       published GPT-2 / SFC circuits and their own patch prompts; kept only if competent)  LAMS_W
       (1e-4,3e-4,1e-3,3e-3,1e-2)  LAMS_U (1e-5,3e-5,1e-4,3e-4,1e-3)  LRS_W / LRS_U (080's adopted lr for OUT_W /
       OUT_U when experiments/080-latent-vs-output/config_main.json exists, else 0.05)  STEPS (400)  LOGIT_NORM (gap)
       CONFIG (config_main_<MODEL>.json)  N_DRAWS (written into the config; 5)  FORCE=1
       FORCE_TASKS=1 (skip the competence filter; smoke only)  OUT_PROBE (where probe.py wrote competence.json)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task_common as C  # noqa: E402

OUT_CAL = Path(os.environ.get("OUT_CAL", str(HERE / "calib" / C.MODEL)))
CONFIG = Path(os.environ.get("CONFIG", str(HERE / ("config_main_%s.json" % C.MODEL))))
CAL_TASKS = [t for t in os.environ.get("CAL_TASKS", "ioi,greater_than,sva_across_rc").split(",") if t]
_LR80 = C.prod_lr_from_080() or {}
DEFAULT_LRS = {"W": str(_LR80.get("OUT_W", C.PROD_LR)), "U": str(_LR80.get("OUT_U", C.PROD_LR))}
LAMS = {"W": [float(x) for x in os.environ.get("LAMS_W", "1e-4,3e-4,1e-3,3e-3,1e-2").split(",") if x],
        "U": [float(x) for x in os.environ.get("LAMS_U", "1e-5,3e-5,1e-4,3e-4,1e-3").split(",") if x]}
LRS = {a: [float(x) for x in os.environ.get("LRS_" + a, DEFAULT_LRS[a]).split(",") if x] for a in C.FIT_ARMS}
STEPS = int(os.environ.get("STEPS", C.PROD_STEPS))
LOGIT_NORM = os.environ.get("LOGIT_NORM", C.LOGIT_NORM)
N_DRAWS = int(os.environ.get("N_DRAWS", C.N_DRAWS))


def calibration_view(task: dict) -> dict:
    """The task restricted to its training rows, re-split 75 / 25 by group (seed + 1): split["train"] = the fit set
    (its training pool), split["held"] = the validation set, capped like a held-out set (task_common.cap_held,
    HELDOUT_CAP; the validation items left out by the cap are not used). Held-out prompts are not in it."""
    tr = list(task["split"]["train"])
    sub_tr, sub_va = C.split_groups([task["rows"][i]["group"] for i in tr], seed=C.TASK_SEED + 1)
    va, n_va = C.cap_held([tr[i] for i in sub_va], seed="%s-calib" % task["name"])
    v = dict(task)
    v["split"] = dict(train=[tr[i] for i in sub_tr], held=va)
    v["calibration_view"] = True
    v["calibration_val_uncapped"] = n_va
    return v


def tasks():
    if os.environ.get("FORCE_TASKS") == "1":                 # smoke.sh: exercise the code path whatever the probe says
        return list(CAL_TASKS)
    from probe import competent_tasks
    ok = set(competent_tasks())
    keep = [t for t in CAL_TASKS if t in ok]
    dropped = [t for t in CAL_TASKS if t not in ok]
    if dropped:
        print("calibration: not competent, skipped: %s" % ", ".join(dropped))
    if not keep:
        sys.exit("no competent calibration task among %s" % CAL_TASKS)
    return keep


def grid():
    return [(a, lr, lam) for a in C.FIT_ARMS for lr in LRS[a] for lam in LAMS[a]]


def run():
    import engine as E
    import torch
    names = tasks()
    OUT_CAL.mkdir(parents=True, exist_ok=True)
    path = OUT_CAL / "eval.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r:
                done.add((r["task"], r["arm"], C.lam_tag(r["lr"]), r["lam"]))
    g = grid()
    print("calibration: %d tasks (%s) x %d fits | steps %d | logit_norm %s | lr %s | %d done -> %s" % (
        len(names), ",".join(names), len(g), STEPS, LOGIT_NORM, LRS, len(done), OUT_CAL), flush=True)
    eng = E.Engine(C.MODEL)
    fh, ftr = open(path, "a"), open(OUT_CAL / "train.jsonl", "a")
    t0 = time.time()
    for name in names:
        view = calibration_view(C.load_task(name))
        contrast = None
        if view["contrast"] == "generated":                  # the pool restricted to the fit set
            contrast = dict(C.load_contrast(name))
            fit_set = set(view["split"]["train"])
            contrast["pool"] = [i for i in contrast["pool"] if i in fit_set]
        B = eng.batch(view, contrast)
        for arm, lr, lam in g:
            lt = C.lam_tag(lam)
            if (name, arm, C.lam_tag(lr), lt) in done:
                continue
            row = dict(task=name, arm=arm, lr=lr, lam=lt, lam_value=lam, steps=STEPS, logit_norm=LOGIT_NORM,
                       n_fit_rows=B["n_train"], n_val_rows=B["n_held"], wcm_version=C.WCM_VERSION)
            try:
                alphas, prov, t_fit, peak = eng.fit(B, arm, lam, lr, STEPS, LOGIT_NORM)
                row.update(t_fit=round(t_fit, 1), peak_mb_fit=peak, sparse_fill_calls=prov.get("sparse_fill_calls"),
                           logit_term_scales=prov.get("logit_term_scales"))
                t = time.time()
                row.update(C80_scalars(eng.score(B, alphas)), t_score=round(time.time() - t, 1))
                ftr.write(json.dumps(dict(task=name, arm=arm, lr=lr, lam=lt, **train_stats(prov))) + "\n"); ftr.flush()
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("%-18s %-2s lr %-5g lam %-7s n=%-6s Z %s obj Z %s pass %s  %.0fs" % (
                name, arm, lr, lt, row.get("n"), _f(row.get("task_free0")), _f(row.get("obj_free0")),
                row.get("task_pass"), time.time() - t0), flush=True)
    print("DONE", flush=True)


def C80_scalars(row):
    return {k: v for k, v in row.items() if isinstance(v, (int, float, str, bool)) or v is None or k == "per_kind"}


TRAIN_STATS = ("loss_initial", "loss_final", "holdout_data_loss", "n_kept", "mean_m_kept", "amp_stats",
               "logit_scale", "dual_norm_zero", "dual_norm_floor", "n_train_pos", "n_holdout_pos", "batch_size_used",
               "steps", "logit_metric", "floors_valid_masked", "floors_precomputed", "present", "wcm_version",
               "train_fill", "sparse_fill_calls", "logit_norm", "logit_term_scales", "dual_norm_pos")


def train_stats(prov):
    out = {k: prov.get(k) for k in TRAIN_STATS}
    curve = prov.get("curve") or {}
    out["curve_loss"] = [float("%.4g" % v) for v in (curve.get("loss") or [])][::10]
    out["curve_n"] = (curve.get("n_members") or [])[::10]
    return out


def load_rows():
    rows = {}
    for line in open(OUT_CAL / "eval.jsonl"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        rows[(r["task"], r["arm"], C.lam_tag(r["lr"]), r["lam"])] = r
    return list(rows.values())


def report():
    rows = load_rows()
    S = C.calib_summary(rows)
    rec = C.recommend(S)
    print("calibration rows: %d over tasks %s (fit / validation = the tasks' TRAINING rows, re-split)" % (
        len(rows), sorted({r["task"] for r in rows})))
    for v in S.values():
        print("\n== %s lr %g: %d fits / %d tasks | failure %.0f%% | monotone in lambda: size %s, task Z %s" % (
            v["arm"], v["lr"], v["n_fits"], v["n_tasks"], 100 * v["fail_rate"], _pct(v["mono_size"]),
            _pct(v["mono_faith"])))
        print("   %-8s %5s %6s %8s %8s %6s" % ("lambda", "fits", "fail", "n med", "Z med", "pass"))
        for lam, s in sorted(v["lams"].items(), key=lambda x: float(x[0])):
            print("   %-8s %5d %5.0f%% %8s %8s %5.0f%%" % (lam, s["n_fits"], 100 * s["fail"],
                                                        "%.0f" % s["n_median"] if s["n_median"] else "-",
                                                        _f(s["Z_median"]), 100 * s["pass_rate"]))
    print("\nper task (validation task Z by lambda):")
    for name in sorted({r["task"] for r in rows}):
        for arm in C.FIT_ARMS:
            pts = sorted((r["lam_value"], r.get("n"), r.get("task_free0"), r.get("task_pass")) for r in rows
                         if r["task"] == name and r["arm"] == arm and not r.get("error"))
            print("  %-18s %-2s %s" % (name, arm, "  ".join("%g:n%s/Z%s%s" % (l, n, _f(z), "*" if p else "")
                                                         for l, n, z, p in pts)))
    print("\n== recommendation")
    for arm, a in rec["arms"].items():
        print("  %-2s lr %g lambdas %s (failure %.0f%%, task Z monotone %s, brackets Z = 0.8: %s)" % (
            arm, a["lr"], a["lams"], 100 * a["fail_rate"], _pct(a["mono_faith"]), a["brackets"]))
    for w in rec["warnings"]:
        print("  WARNING: " + w)
    rec.update(logit_norm=rows[0].get("logit_norm", LOGIT_NORM) if rows else LOGIT_NORM,
               steps=int(rows[0].get("steps", STEPS)) if rows else STEPS, n_draws=N_DRAWS, summary=S,
               adopted=False, generated=time.strftime("%Y-%m-%d %H:%M"), source=str(OUT_CAL / "eval.jsonl"),
               lr_source="080 config_main.json" if _LR80 else "production 0.05")
    (OUT_CAL / "recommendation.json").write_text(json.dumps(rec, indent=1))
    print("\nwrote %s (not adopted: ADOPT=1 writes %s)" % (OUT_CAL / "recommendation.json", CONFIG.name))


def adopt():
    rec = json.loads((OUT_CAL / "recommendation.json").read_text())
    if rec.get("warnings") and os.environ.get("FORCE") != "1":
        sys.exit("the recommendation has warnings (%s); extend the sweep, or FORCE=1 to adopt anyway"
                 % "; ".join(rec["warnings"]))
    cfg = dict(arms={a: dict(lams=v["lams"], lr=v["lr"]) for a, v in rec["arms"].items()},
               logit_norm=rec["logit_norm"], steps=rec["steps"], n_draws=rec.get("n_draws", C.N_DRAWS),
               adopted_from=str(OUT_CAL / "recommendation.json"), recommendation_generated=rec.get("generated"),
               adopted_at=time.strftime("%Y-%m-%d %H:%M"), warnings_at_adoption=rec.get("warnings"))
    CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")
    C.load_run_config(CONFIG)
    print("adopted -> %s:\n%s" % (CONFIG, json.dumps(cfg, indent=1)))


def _f(v):
    return "%.2f" % v if isinstance(v, (int, float)) and v == v else "-"


def _pct(v):
    return "-" if v is None else "%.0f%%" % (100 * v)


if __name__ == "__main__":
    if os.environ.get("DRY") == "1":
        g = grid()
        print("%s x %d fits = %d fits at %d steps: %s" % (CAL_TASKS, len(g), len(CAL_TASKS) * len(g), STEPS, g))
    elif os.environ.get("REPORT") == "1":
        report()
    elif os.environ.get("ADOPT") == "1":
        adopt()
    else:
        run()
