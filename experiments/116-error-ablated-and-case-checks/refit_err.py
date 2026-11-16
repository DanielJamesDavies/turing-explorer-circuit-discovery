"""116 §1b (pilot): REFIT circuits with the SAE errors ablated during training AND scoring.

§1 scores the fixed production circuits, whose coefficients were fitted with the errors live. This pilot asks whether
WCM can find circuits that pass when the errors are gone: the production fit (062 driver.Runner.fit, WCM v4, lambda
1e-3, 400 steps, same records) with one change in the learned-mask engine, then scored under the same error mode.

Training: circuit.instrument.learned_mask.LearnedMaskPatcher.transform is wrapped. At every site it masks (a site with
gates; not the seed tap, not the members-removed "invert" pass of the live-model terms), its output
x + decode(code - dense), which carries the error x - decode(dense) implicitly, has that error subtracted (ERR=zero),
so the masked stream is decode(code) alone, as errpatch's zero mode does at scoring time. Gradients flow through the
subtracted error. ERR=live runs the same fit unwrapped (the paired refit baseline on this GPU).

Scoring: errpatch MODE = ERR (zero or live), plus the other mode for reference.

  wsl: ERR=zero|live KEYS=... PYTHONPATH=src .venv/bin/python -X utf8 experiments/116-error-ablated-and-case-checks/refit_err.py
  env: ERR  KEYS (comma list) or TARGETS_FILE  MAX_MIN
  -> out/refit_<ERR>/rows.jsonl, out/refit_<ERR>/main/circuits/<key>.pt, out/refit_<ERR>/train.jsonl
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
PROD_OUT = ARCH / "pub-turingllm-T3-main/root/turing-tl/experiments/094-tl-v2-records/out_v2"
CTX = ARCH / "pub-turingllm-T2-records/root/turing-tl/experiments/082-public-runs/out_turingllm/ctx_fresh"
ERR = os.environ.get("ERR", "zero")
assert ERR in ("zero", "live"), ERR
RF = HERE / "out" / ("refit_%s" % ERR)
RF.mkdir(parents=True, exist_ok=True)
os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(RF), CTX_DIR=str(CTX), SKIP_SPEC="1")
os.environ.setdefault("TAG", "refit116_unused")
for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import driver  # noqa: E402
import errpatch as EP  # noqa: E402

MAX_MIN = float(os.environ.get("MAX_MIN", "0") or 0)
STATS = {"calls": 0}


def install_train_error_ablation():
    import circuit.instrument.learned_mask as LM
    from sae.dense import sparse_topk_to_dense
    P = LM.LearnedMaskPatcher
    orig = P.transform

    def transform(self, layer_idx, kind, x):
        out = orig(self, layer_idx, kind, x)
        if out is x or self.invert:
            return out
        if self.tap_seed and layer_idx == self.seed_layer and kind == self.seed_kind:
            return out
        if self.thetas.get((layer_idx, kind)) is None:
            return out
        ta, ti = self.bank.encode(x, kind, layer_idx)
        recon = self.bank.decode(sparse_topk_to_dense(ta, ti, self.bank.d_sae, dtype=x.dtype), kind, layer_idx)
        STATS["calls"] += 1
        return out - (x - recon.to(x.dtype))

    P.transform = transform


def scalar(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None}


def main():
    import torch
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    else:
        keys = [k for k in Path(os.environ["TARGETS_FILE"]).read_text().split() if k]
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
    if ERR == "zero":
        install_train_error_ablation()
    R = driver.Runner()
    EP.install()
    M = R.method(driver.GAMMA, driver.LAM, True)
    ftr = open(RF / "train.jsonl", "a")
    t0 = time.time()
    print("116 refit ERR=%s: %d targets" % (ERR, len(keys)), flush=True)
    with open(path, "a") as fh:
        for n_, key in enumerate(keys):
            if MAX_MIN and (time.time() - t0) / 60 > MAX_MIN:
                print("MAX_MIN reached, stopping (resumable)", flush=True)
                break
            if all((key, m) in done for m in ("live", "zero")):
                continue
            ts = time.time()
            try:
                rec = R.contexts(key)
                arm, thin = R.train_arm(rec)
                c0 = STATS["calls"]
                c, t_fit = R.fit(M, rec, key, arm, RF / "main" / "circuits" / ("%s.pt" % key), train_fh=ftr)
                if c is None:
                    fh.write(json.dumps(dict(seed=key, err_mode="none", skip="rejected")) + "\n"); fh.flush()
                    continue
                for m in ("live", "zero"):
                    if (key, m) in done:
                        continue
                    EP.MODE["m"] = m
                    try:
                        row = scalar(R.score(c, rec, "strong"))
                    finally:
                        EP.MODE["m"] = "live"
                    row.update(err_mode=m, fit_err=ERR, held="strong", arm=arm, thin=thin, t_fit=round(t_fit, 1),
                               n_nodes=driver.n_nodes(c), train_err_calls=STATS["calls"] - c0)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
            except Exception as e:  # noqa: BLE001
                EP.MODE["m"] = "live"
                fh.write(json.dumps(dict(seed=key, err_mode="error", error="%s: %s" % (type(e).__name__, str(e)[:300])))
                         + "\n"); fh.flush()
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print("%3d/%d %-18s %.0fs (total %.1f min)" % (n_ + 1, len(keys), key, time.time() - ts,
                                                        (time.time() - t0) / 60), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
