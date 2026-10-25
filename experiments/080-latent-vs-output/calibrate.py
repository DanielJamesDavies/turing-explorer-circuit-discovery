"""Experiment 080 calibration stage: lambda x lr for the output-endpoint arms (OUT_W, OUT_U) on the calibration targets
(targets_calib.txt, disjoint from the main targets: one per design cell, 18 under SAMPLING=causal, 12 under
withinkind).

Why: experiment 025's logit objective, run with the pre-activation objective's hyperparameters, failed on ~1/3 of
seeds, was non-monotone in lambda and went negative at intermediate lambda. The main run's lambda grid and lr must
come from this stage, through an explicit adoption step, never from a silent auto-pick.

Three modes, one script:
  (default)  fit + score: every (target, arm, lr, lambda) of the grid, fitted exactly as run.py fits (fitting.Fitter:
             the same engine config, contexts and scope), scored on the output endpoint on held-out strongest
             (SCORE_LATENT=1 adds the latent scorer). Rows -> OUT_CAL/eval.jsonl, curves -> OUT_CAL/train.jsonl.
             Resumable per (target, arm, lr, lambda); error rows are retried.
  REPORT=1   CPU: failure rate (fit failed, or held-out output Z missing / < 0.05), monotonicity of size and of output Z
             in lambda, size vs lambda against the calibration targets' LAT sizes, per arm x lr; the recommendation
             (common.recommend: the lr with the fewest failures, then the 4 consecutive lambdas whose median sizes best
             cover the LAT p10..p90) -> OUT_CAL/recommendation.json (adopted: false).
  ADOPT=1    copy the recommendation's grid and lr into CONFIG (default config_main.json, read by run.py), with its
             provenance. Refuses while the recommendation carries warnings unless FORCE=1. Edit the file by hand to
             override it; run.py reads whatever it says.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/080-latent-vs-output/calibrate.py
  env: OUT_CAL (experiments/080-latent-vs-output/calib)  TARGETS (targets_calib.txt)  CAL_ARMS (OUT_W,OUT_U)
       LAMS_OUT_W (1e-4,3e-4,1e-3,3e-3,1e-2)  LAMS_OUT_U (1e-5,3e-5,1e-4,3e-4,1e-3)  LRS (0.02,0.05,0.1)
       LAMS_OUT_W_FULL / LAMS_OUT_U_FULL (whole-model scope arms; CAL_ARMS=OUT_W_FULL,OUT_U_FULL, see queue_calib_full.sh)
       STEPS (400)  LOGIT_NORM (gap)  LIMIT (first N targets)  SCORE_LATENT (0)  SRC_OUT (062 out_full/out)
       DRY=1 (print the plan and exit; no model)
"""
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT_CAL = Path(os.environ.get("OUT_CAL", str(HERE / "calib")))
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets_calib.txt")))
CONFIG = Path(os.environ.get("CONFIG", str(HERE / "config_main.json")))
SRC_OUT = Path(os.environ.get("SRC_OUT", str(C.EXP / "062-h100-protocol-v1" / "out_full" / "out")))
CAL_ARMS = [a for a in os.environ.get("CAL_ARMS", "OUT_W,OUT_U").split(",") if a]
DEFAULT_LAMS = {"OUT_W": "1e-4,3e-4,1e-3,3e-3,1e-2", "OUT_U": "1e-5,3e-5,1e-4,3e-4,1e-3",
                "OUT_W_FULL": "1e-4,3e-4,1e-3,3e-3,1e-2", "OUT_U_FULL": "3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"}
LAMS = {a: [float(x) for x in os.environ.get("LAMS_" + a, DEFAULT_LAMS[a]).split(",") if x] for a in CAL_ARMS}
LRS = [float(x) for x in os.environ.get("LRS", "0.02,0.05,0.1").split(",") if x]
STEPS = int(os.environ.get("STEPS", C.PROD_STEPS))
LOGIT_NORM = os.environ.get("LOGIT_NORM", "gap")
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
SCORE_LATENT = os.environ.get("SCORE_LATENT") == "1"


def targets():
    t = [k for k in TARGETS.read_text().split() if k]
    return t[:LIMIT] if LIMIT else t


def grid():
    return [(a, lr, lam) for a in CAL_ARMS for lr in LRS for lam in LAMS[a]]


def lat_sizes(keys):
    meta = C.load_meta(HERE)
    return {k: float(meta[k]["lat_n"]) for k in keys if k in meta}


# ------------------------------------------------------------------------------------------- fit + score
def run():
    os.environ["OUT"] = str(OUT_CAL)                         # the engine's context cache (before fitting imports)
    import fitting as F
    import torch
    keys = targets()
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
                done.add((r["seed"], r["arm"], C.lam_tag(r["lr"]), r["lam"]))
    g = grid()
    print("calibration: %d targets x %d fits (%s) | steps %d | logit_norm %s | %d rows done -> %s" % (
        len(keys), len(g), ", ".join("%s lr %s lam %s" % (a, LRS, LAMS[a]) for a in CAL_ARMS), STEPS, LOGIT_NORM,
        len(done), OUT_CAL), flush=True)
    Fi = F.Fitter(dict(logit_norm=LOGIT_NORM))
    fh, ftr = open(path, "a"), open(OUT_CAL / "train.jsonl", "a")
    t0 = time.time()
    for n_, key in enumerate(keys):
        src = SRC_OUT / "ctx" / ("%s.pt" % key)
        dst = OUT_CAL / "ctx" / ("%s.pt" % key)
        if not dst.exists() and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        rec = Fi.R.contexts(key)
        if rec.get("strong") is None:
            continue
        for arm, lr, lam in g:
            lt = C.lam_tag(lam)
            if (key, arm, C.lam_tag(lr), lt) in done:
                continue
            a = C.ARMS[arm]
            row = dict(seed=key, arm=arm, lr=lr, lam=lt, lam_value=lam, steps=STEPS, logit_norm=LOGIT_NORM,
                       weighted=bool(a["weighted"]), scope=a["scope"])
            try:
                c, t_fit, peak = Fi.fit(arm, lam, lr, STEPS, rec, key,
                                        OUT_CAL / "circuits" / arm / ("lr%s" % C.lam_tag(lr)) / lt / ("%s.pt" % key),
                                        train_fh=ftr, label=dict(arm=arm, lam=lt, lr=lr, steps=STEPS))
                row.update(t_fit=round(t_fit, 1), peak_mb_fit=peak)
                if c is None:
                    row["skip"] = "rejected"
                else:
                    t = time.time()
                    row.update(F.scalars(Fi.score(c, rec, str(a["scope"]), latent=SCORE_LATENT)),
                               n=len(F.member_set(c)), t_score=round(time.time() - t, 1))
                    row.update(arm=arm, lam=lt)
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("%2d/%d %-16s %-5s lr %-5g lam %-7s n=%-6s oZ %s  %.0fs" % (
                n_ + 1, len(keys), key, arm, lr, lt, row.get("n"),
                "%.2f" % row["out_free0"] if isinstance(row.get("out_free0"), float) else "-",
                time.time() - t0), flush=True)
    print("DONE", flush=True)


# ------------------------------------------------------------------------------------------- report
def load_rows():
    rows = {}
    for line in open(OUT_CAL / "eval.jsonl"):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        rows[(r["seed"], r["arm"], C.lam_tag(r["lr"]), r["lam"])] = r      # the last row per fit
    return list(rows.values())


def report():
    rows = load_rows()
    keys = sorted({r["seed"] for r in rows})
    lat = lat_sizes(keys)
    S = C.calib_summary(rows, lat)
    rec = C.recommend(S)
    L = S["_lat"]
    print("calibration rows: %d over %d targets | LAT sizes of these targets: p10 %.0f  p25 %.0f  median %.0f  "
          "p75 %.0f  p90 %.0f" % (len(rows), len(keys), L["p10"] or 0, L["p25"] or 0, L["median"] or 0,
                                  L["p75"] or 0, L["p90"] or 0))
    for k, v in S.items():
        if k.startswith("_"):
            continue
        print("\n== %s  lr %g: %d fits / %d targets | failure rate %.0f%% | monotone in lambda: size %s, output Z %s"
              % (v["arm"], v["lr"], v["n_fits"], v["n_targets"], 100 * v["fail_rate"],
                 _pct(v["mono_size"]), _pct(v["mono_faith"])))
        print("   %-8s %6s %8s %8s %8s %6s" % ("lambda", "fits", "fail", "n med", "oZ med", "oZ<0"))
        for lam, s in sorted(v["lams"].items(), key=lambda x: float(x[0])):
            print("   %-8s %6d %7.0f%% %8s %8s %6d" % (lam, s["n_fits"], 100 * s["fail"],
                                                     "%.0f" % s["n_median"] if s["n_median"] else "-",
                                                     "%.2f" % s["oZ_median"] if s["oZ_median"] is not None else "-",
                                                     s["oZ_neg"]))
    print("\n== recommendation")
    for arm, a in rec["arms"].items():
        print("  %-6s lr %g  lambdas %s  (failure %.0f%%, output Z monotone %s, brackets the LAT sizes: %s)" % (
            arm, a["lr"], a["lams"], 100 * a["fail_rate"], _pct(a["mono_faith"]), a["brackets_lat"]))
    for w in rec["warnings"]:
        print("  WARNING: " + w)
    rec.update(logit_norm=LOGIT_NORM if not rows else rows[0].get("logit_norm", LOGIT_NORM),
               steps=int(rows[0].get("steps", STEPS)) if rows else STEPS, summary=S, adopted=False,
               generated=time.strftime("%Y-%m-%d %H:%M"), source=str(OUT_CAL / "eval.jsonl"))
    (OUT_CAL / "recommendation.json").write_text(json.dumps(rec, indent=1))
    print("\nwrote %s (not adopted: run ADOPT=1 to write %s)" % (OUT_CAL / "recommendation.json", CONFIG.name))


def adopt():
    rec = json.loads((OUT_CAL / "recommendation.json").read_text())
    if rec.get("warnings") and os.environ.get("FORCE") != "1":
        sys.exit("the recommendation has warnings (%s); fix the sweep, or FORCE=1 to adopt anyway"
                 % "; ".join(rec["warnings"]))
    cfg = dict(arms={a: dict(lams=v["lams"], lr=v["lr"]) for a, v in rec["arms"].items()},
               logit_norm=rec["logit_norm"], steps=rec["steps"],
               adopted_from=str(OUT_CAL / "recommendation.json"), recommendation_generated=rec.get("generated"),
               adopted_at=time.strftime("%Y-%m-%d %H:%M"), warnings_at_adoption=rec.get("warnings"))
    CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")
    C.load_run_config(CONFIG)                                # validates what was written
    print("adopted -> %s:\n%s" % (CONFIG, json.dumps(cfg, indent=1)))


def _pct(v):
    return "-" if v is None else "%.0f%%" % (100 * v)


if __name__ == "__main__":
    if os.environ.get("DRY") == "1":
        g = grid()
        print("%d targets x %d fits = %d fits: %s" % (len(targets()), len(g), len(targets()) * len(g), g))
    elif os.environ.get("REPORT") == "1":
        report()
    elif os.environ.get("ADOPT") == "1":
        adopt()
    else:
        run()
