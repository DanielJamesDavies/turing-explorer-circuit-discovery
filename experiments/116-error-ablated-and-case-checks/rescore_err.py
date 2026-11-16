"""116 §1: re-score FIXED production circuits (WCM v4) with the SAE errors live / zeroed / mean-ablated. No refit.

Per target, in one process and on one GPU, the production scorer (062 driver.Runner.score: TuringLLM through 049's
score_circuit, public models through src wcm_eval) is run three times on the held-out strongest contexts, once per
errpatch.MODE (live = the production scoring, the paired baseline; zero; mean). The circuits are read from the
production archives (never written); the records from the production ctx dirs (CTX_DIR, explicit split, as 115).

  wsl: FAMILY=<turingllm|pythia-70m|gpt2-small> PYTHONPATH=src .venv/bin/python -X utf8 \
       experiments/116-error-ablated-and-case-checks/rescore_err.py
  env: FAMILY  LIMIT (first N targets)  KEYS (comma list, overrides the target file)  MODES (live,zero,mean)
       MAX_MIN (stop starting new targets after this many minutes; resumable)
  -> out/rs_<family>/rows.jsonl (one row per (seed, mode); resumable)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCH = ROOT / "experiments" / "109-pod-run-analysis" / "out" / "archives"
FAMILY = os.environ.get("FAMILY", "turingllm")
SRC = {
    "turingllm": (ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2",
                  ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"),
    "pythia-70m": (ARCH / "pub-pythia-70m-S10-results/experiments/082-public-runs/out_pythia-70m",
                   ARCH / "pub-pythia-70m-S10-results/experiments/082-public-runs/out_pythia-70m/ctx"),
    "gpt2-small": (ARCH / "pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small",
                   ARCH / "pub-gpt2-small-S10-results/experiments/082-public-runs/out_gpt2-small/ctx"),
}
PROD_OUT, CTX = SRC[FAMILY]
RS = HERE / "out" / ("rs_%s" % FAMILY)
RS.mkdir(parents=True, exist_ok=True)
os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(RS), CTX_DIR=str(CTX), FAMILY=FAMILY, SKIP_SPEC="1")
os.environ.setdefault("TAG", "rs116_unused")
for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import driver  # noqa: E402
import errpatch as EP  # noqa: E402

MODES = [m for m in os.environ.get("MODES", "live,zero,mean").split(",") if m]
MAX_MIN = float(os.environ.get("MAX_MIN", "0") or 0)


def scalar(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


def main():
    import torch
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    else:
        keys = [k for k in (HERE / ("targets_%s.txt" % FAMILY)).read_text().split() if k]
    if os.environ.get("LIMIT"):
        keys = keys[:int(os.environ["LIMIT"])]
    for m in ("WCM_VERSION", "WCM_R"):              # the production markers (the Runner refuses another version)
        (RS / m).write_bytes((PROD_OUT / m).read_bytes())
    path = RS / "rows.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "error" not in r:
                done.add((r["seed"], r["err_mode"]))
    R = driver.Runner()
    EP.install()
    print("116 rescore %s: %d targets x %s, %d rows done" % (FAMILY, len(keys), MODES, len(done)), flush=True)
    t0 = time.time()
    with open(path, "a") as fh:
        for n_, key in enumerate(keys):
            if MAX_MIN and (time.time() - t0) / 60 > MAX_MIN:
                print("MAX_MIN reached, stopping (resumable)", flush=True)
                break
            todo = [m for m in MODES if (key, m) not in done]
            if not todo:
                continue
            ts = time.time()
            try:
                c = torch.load(PROD_OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
                rec = R.contexts(key)
                arm, thin = R.train_arm(rec)
                for m in todo:
                    EP.MODE["m"] = m
                    try:
                        row = R.score(c, rec, "strong")
                    finally:
                        EP.MODE["m"] = "live"
                    row = scalar(row)
                    row.update(err_mode=m, held="strong", arm=arm, thin=thin, family=FAMILY)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
            except Exception as e:  # noqa: BLE001
                EP.MODE["m"] = "live"
                fh.write(json.dumps(dict(seed=key, err_mode="error", error="%s: %s" % (type(e).__name__, str(e)[:300])))
                         + "\n"); fh.flush()
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("%4d/%d %-18s %.0fs (total %.1f min)" % (n_ + 1, len(keys), key, time.time() - ts,
                                                         (time.time() - t0) / 60), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
