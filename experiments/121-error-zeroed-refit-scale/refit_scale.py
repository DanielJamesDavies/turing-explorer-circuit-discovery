"""121: REFIT circuits with every upstream SAE error zeroed in training and scoring, at scale (116 §1b scaled up).

This is 116 refit_err.py's fit, run per model family. It reuses, imported and not re-implemented:
  - 116 refit_err.install_train_error_ablation: the in-process wrap of LearnedMaskPatcher.transform that subtracts each
    masked site's SAE error (the masked stream is decode(code) alone; not the seed tap, not the "invert" pass);
  - 116 errpatch.install / MODE: the scoring-time error switch of CircuitOnlyPatcher;
  - 062 driver.Runner (contexts / train_arm / method / fit / score): the production WCM v4 fit (lambda 1e-3, 400 steps,
    same records) and the production scorer, exactly as 116 used them.
The only change from refit_err.main is the family-aware paths (refit_err hard-codes TuringLLM) and a wall-clock stop.

Arms written here, per target: (a) fitted with errors zeroed, scored zeroed; (c) the same circuit scored live.
Arm (b) (the production circuit scored live on this GPU) is 116's rs_<family> live row, not re-run.

  wsl: FAMILY=<turingllm|pythia-70m|gpt2-small> MAX_MIN=75 STOP_AT=22:45 PYTHONPATH=src .venv/bin/python -X utf8 \
       experiments/121-error-zeroed-refit-scale/refit_scale.py
  env: FAMILY  MAX_MIN (stop starting targets after N minutes)  STOP_AT (HH:MM local: no new target after it)
       LIMIT  KEYS (comma list; overrides targets_<family>.txt)  DRY=1 (imports and config only, no GPU)
  -> out/refit_<family>/rows.jsonl (one row per (seed, err_mode); resumable), train.jsonl, main/circuits/<key>.pt
"""
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E116 = ROOT / "experiments" / "116-error-ablated-and-case-checks"
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
FAMILY = os.environ.get("FAMILY", "turingllm")
SRC = {   # (production OUT, records) as in 116 rescore_err.py
    "turingllm": (ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2",
                  ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"),
    "pythia-70m": (ARCH / "pub-pythia-70m-S10-results/experiments/082-public-runs/out_pythia-70m",
                   ARCH / "pub-pythia-70m-S10-results/experiments/082-public-runs/out_pythia-70m/ctx"),
    "gpt2-small": (ARCH / "pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small",
                   ARCH / "pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small/ctx"),
}
PROD_OUT, CTX = SRC[FAMILY]
RF = HERE / "out" / ("refit_%s" % FAMILY)
RF.mkdir(parents=True, exist_ok=True)
for m in ("WCM_VERSION", "WCM_R"):                 # the production markers: the Runner refuses another config
    (RF / m).write_bytes((PROD_OUT / m).read_bytes())
MY_ENV = dict(MODE="main", WCM_VERSION="v4", OUT=str(RF), CTX_DIR=str(CTX), FAMILY=FAMILY, SKIP_SPEC="1")
os.environ.update(MY_ENV)
os.environ.setdefault("TAG", "refit121_unused")
for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1"), str(E116)):
    if p not in sys.path:
        sys.path.insert(0, p)
import driver  # noqa: E402  (reads its config from MY_ENV at import)

_snap = dict(os.environ)
os.environ["ERR"] = "zero"
import refit_err as RE116  # noqa: E402  (its module body sets 116's OUT / CTX_DIR in os.environ: restored below)
os.environ.clear(); os.environ.update(_snap)
import errpatch as EP  # noqa: E402

MAX_MIN = float(os.environ.get("MAX_MIN", "0") or 0)
STOP_AT = os.environ.get("STOP_AT")


def past_stop():
    if not STOP_AT:
        return False
    h, m = (int(x) for x in STOP_AT.split(":"))
    now = dt.datetime.now()
    return now >= now.replace(hour=h, minute=m, second=0, microsecond=0)


def main():
    import torch
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    else:
        keys = [k for k in (HERE / ("targets_%s.txt" % FAMILY)).read_text().split() if k]
    if os.environ.get("LIMIT"):
        keys = keys[:int(os.environ["LIMIT"])]
    assert driver.FAMILY == FAMILY and Path(driver.OUT) == RF and driver.WCM_VERSION == "v4", "driver config"
    print("121 refit %s: %d targets; driver FAMILY=%s OUT=%s CTX_DIR=%s LAM=%g STEPS=%d WCM=%s" % (
        FAMILY, len(keys), driver.FAMILY, driver.OUT, driver.CTX_DIR, driver.LAM, driver.STEPS, driver.WCM_VERSION),
        flush=True)
    if os.environ.get("DRY") == "1":
        return
    path = RF / "rows.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["seed"], r["err_mode"]))
    RE116.install_train_error_ablation()
    R = driver.Runner()
    EP.install()
    M = R.method(driver.GAMMA, driver.LAM, True)
    ftr = open(RF / "train.jsonl", "a")
    t0 = time.time()
    with open(path, "a") as fh:
        for n_, key in enumerate(keys):
            if MAX_MIN and (time.time() - t0) / 60 > MAX_MIN:
                print("MAX_MIN reached, stopping (resumable)", flush=True)
                break
            if past_stop():
                print("STOP_AT %s reached, stopping (resumable)" % STOP_AT, flush=True)
                break
            if all((key, m) in done for m in ("live", "zero")) or (key, "none") in done:
                continue
            ts = time.time()
            try:
                rec = R.contexts(key)
                arm, thin = R.train_arm(rec)
                c0 = RE116.STATS["calls"]
                c, t_fit = R.fit(M, rec, key, arm, RF / "main" / "circuits" / ("%s.pt" % key), train_fh=ftr)
                if c is None:
                    fh.write(json.dumps(dict(seed=key, err_mode="none", skip="rejected", family=FAMILY)) + "\n")
                    fh.flush()
                    continue
                for m in ("zero", "live"):
                    if (key, m) in done:
                        continue
                    EP.MODE["m"] = m
                    try:
                        row = RE116.scalar(R.score(c, rec, "strong"))
                    finally:
                        EP.MODE["m"] = "live"
                    row.update(err_mode=m, fit_err="zero", held="strong", arm=arm, thin=thin, t_fit=round(t_fit, 1),
                               n_nodes=driver.n_nodes(c), train_err_calls=RE116.STATS["calls"] - c0, family=FAMILY)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
            except Exception as e:  # noqa: BLE001
                EP.MODE["m"] = "live"
                fh.write(json.dumps(dict(seed=key, err_mode="error", family=FAMILY,
                                         error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("%3d/%d %-20s %.0fs (total %.1f min)" % (n_ + 1, len(keys), key, time.time() - ts,
                                                          (time.time() - t0) / 60), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
