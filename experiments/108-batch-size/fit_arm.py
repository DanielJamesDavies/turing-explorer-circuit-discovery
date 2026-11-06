"""Experiment 108: one arm of the batch-size test through the UNCHANGED production driver (062 driver.py, MODE=main,
WCM_VERSION=v4), with the arm's batch / steps / peak lr / R seed applied as overrides after the driver configures each
fit. Nothing in src/ or 062/ is edited.

What is overridden (arms.py has the values):
  - steps: the driver's own STEPS env (protocol_harness.configure sets learned_mask.steps from it);
  - R donor seed (s2 arms): the driver's own R_SEED env (learned_mask.resample_seed; marked in OUT/LR_SCHEDULE);
  - effective batch: the discovery method's probe_batch_size (ablation_gradient passes it to run_learned_mask as
    batch_size; production 4 from amp_eval_pass_v2.setup), set on the method object after Runner.method builds it;
  - micro-batch: learned_mask.deep_site_threshold 0 and deep_batch_size 4, so every fit runs micro-batches of 4 with
    accum = batch / 4 (learned_mask.py:1389-1398). B4 is then accum 1 = production exactly (production's threshold 99
    never triggers; micro = batch = 4 either way);
  - peak lr (100-step arms only): learned_mask.lr after the driver set it (production 0.05 x 3.15; B4 / B16 keep the
    driver's value untouched).
Each fresh fit is checked against the engine's own provenance (batch_size_used, micro_batch, accum_chunks, steps) before
its circuit is moved into place, its circuit is saved atomically (temp file + os.replace), and a row goes to
OUT/b108_fits.jsonl (wall time, s / step, torch peak allocated / reserved MiB). OUT/b108_current.txt holds the key of
the fit in progress (empty between fits), so run.sh's watchdog can name a fit it aborts. STOP_EPOCH (env, epoch s): no
new fit starts after it (the process exits 75; finished fits are kept, the driver resumes per target).

  ARM=B48 OUT=experiments/108-batch-size/out/B48 TARGETS=... CTX_DIR=... PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 \
      .venv/bin/python -u experiments/108-batch-size/fit_arm.py
  CHECK=1 (CPU, CUDA_VISIBLE_DEVICES=""): import the driver, apply the arm's overrides to the real config through the
      real Runner.method with a stub method builder (no model, no OUT writes), print and assert the settings.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import arms as A  # noqa: E402

ARM = A.arm(os.environ.get("ARM", ""))
CHECK = os.environ.get("CHECK") == "1"
EXIT_STOP = 75

# ---- environment for the driver, set before it is imported (it reads env at import)
for k_ in ("DEEP_BS", "LR_SCHEDULE", "LR_PEAK_MULT", "CENSOR_T", "LR_WARMUP", "NEC_W", "RIVAL_W", "LAM_ADAPT_EPS",
           "RANK_CUT_K", "RANK_ZERO_MULT", "FAMILY", "SHARD"):
    if os.environ.get(k_):
        raise SystemExit("108: %s must be unset (production v4 settings only)" % k_)
if os.environ.get("WCM_VERSION", "v4") != "v4":
    raise SystemExit("108: WCM_VERSION must be v4")
if float(os.environ.get("LAM", "1e-3")) != 1e-3:
    raise SystemExit("108: LAM must be 1e-3")
if os.environ.get("STEPS") and int(os.environ["STEPS"]) != ARM["steps"]:
    raise SystemExit("108: STEPS=%s conflicts with arm %s (%d)" % (os.environ["STEPS"], ARM["name"], ARM["steps"]))
_rs = os.environ.get("R_SEED")
if _rs and _rs != str(ARM["r_seed"]):
    raise SystemExit("108: R_SEED=%s conflicts with arm %s (%s)" % (_rs, ARM["name"], ARM["r_seed"]))
os.environ.update(WCM_VERSION="v4", MODE="main", LAM="1e-3", STEPS=str(ARM["steps"]))
if ARM["r_seed"] is not None:
    os.environ["R_SEED"] = str(ARM["r_seed"])
os.environ.setdefault("OUT", str(HERE / "out" / ARM["name"]))
os.environ.setdefault("TARGETS", str(HERE / "targets.txt"))
os.environ.setdefault("CTX_DIR", str(REPO / "experiments" / "062-h100-protocol-v1" / "out_full" / "out" / "ctx"))
os.environ.setdefault("SKIP_SPEC", "1")
os.environ.setdefault("ORDER", "list")
os.environ.setdefault("CURVE_STRIDE", "1")
STOP_EPOCH = float(os.environ["STOP_EPOCH"]) if os.environ.get("STOP_EPOCH") else None

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments" / "062-h100-protocol-v1"))
import driver as D  # noqa: E402

OUT = D.OUT
_orig_method, _orig_fit = D.Runner.method, D.Runner.fit


def _lm():
    from config import config
    return config.discovery.learned_mask


def method(self, gamma, lam, free_amp, steps=D.STEPS, rank=None):
    M = _orig_method(self, gamma, lam, free_amp, steps, rank)
    lm = _lm()
    if abs(float(lm.lr) - A.PROD_PEAK) > 1e-9:
        raise RuntimeError("108: the driver set lr %r, expected production v4's %r" % (lm.lr, A.PROD_PEAK))
    if int(lm.steps) != ARM["steps"]:
        raise RuntimeError("108: learned_mask.steps %r != arm %d" % (lm.steps, ARM["steps"]))
    if ARM["peak_lr"] != A.PROD_PEAK:
        lm.lr = float(ARM["peak_lr"])
    lm.deep_site_threshold = 0
    lm.deep_batch_size = A.MICRO
    M.probe_batch_size = ARM["batch"]
    want = dict(lr_schedule="geometric", warmup_frac=0.1, censor_temp=0.05, resample_weight=1.0, rank_weight=0.1,
                l1_lambda=1e-3, train_fill="sparse", train_means_source="eval_ref")
    for k_, v_ in want.items():
        if getattr(lm, k_) != v_:
            raise RuntimeError("108: learned_mask.%s = %r, production v4 has %r" % (k_, getattr(lm, k_), v_))
    want_seed = ARM["r_seed"] if ARM["r_seed"] is not None else 20261017
    if int(lm.resample_seed) != want_seed:
        raise RuntimeError("108: resample_seed %r != %r" % (lm.resample_seed, want_seed))
    D.log("108 arm %s: batch %d = micro %d x accum %d, %d steps, peak lr %.6g (sum %.4f), R seed %s, lambda %g" % (
        ARM["name"], M.probe_batch_size, lm.deep_batch_size, ARM["accum"], lm.steps, lm.lr,
        A.lr_sum(lm.steps, lm.lr), lm.resample_seed, lm.l1_lambda))
    return M


def _write_current(key):
    (OUT / "b108_current.txt").write_text(key + ("\n" if key else ""))


def fit(self, M, rec, key, arm, path, train_fh=None, label=None):
    path = Path(path)
    if path.exists():
        return _orig_fit(self, M, rec, key, arm, path, train_fh=train_fh, label=label)
    if STOP_EPOCH is not None and time.time() >= STOP_EPOCH:
        D.log("108: STOP_EPOCH passed, no new fit (%s); exiting %d" % (key, EXIT_STOP))
        raise SystemExit(EXIT_STOP)
    import torch
    _write_current(key)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    tmp = path.parent / (".%s.%d.tmp.pt" % (key, os.getpid()))
    path.parent.mkdir(parents=True, exist_ok=True)
    c, t_fit = _orig_fit(self, M, rec, key, arm, tmp, train_fh=train_fh, label=label)
    prov = M.last_mask_provenance or {}
    got = dict(batch=prov.get("batch_size_used"), micro=prov.get("micro_batch"), accum=prov.get("accum_chunks"),
               steps=prov.get("steps"))
    want = dict(batch=ARM["batch"], micro=A.MICRO, accum=ARM["accum"], steps=ARM["steps"])
    if c is not None and got != want:
        raise RuntimeError("108: engine ran %s, arm %s wants %s (circuit left at %s)" % (got, ARM["name"], want, tmp))
    os.replace(tmp, path)
    n_mem = D.n_nodes(c) if c is not None else None
    row = dict(seed=key, arm=ARM["name"], train_arm=arm, n_sites=A.n_sites(key), t_fit=round(t_fit, 1),
               s_per_step=round(t_fit / ARM["steps"], 4), n=n_mem, accepted=c is not None,
               peak_alloc_mib=round(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None,
               peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2 ** 20) if cuda else None,
               resample_draws=prov.get("resample_draws"), engine=got, t_end=round(time.time()),
               **{k_: ARM[k_] for k_ in ("batch", "micro", "accum", "steps", "peak_lr", "r_seed")})
    with open(OUT / "b108_fits.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    _write_current("")
    return c, t_fit


def check_marker():
    """OUT/BATCH108.json names the arm OUT's circuits were fitted under; another arm's settings are refused."""
    OUT.mkdir(parents=True, exist_ok=True)
    m = OUT / "BATCH108.json"
    now = {k_: ARM[k_] for k_ in ("name", "batch", "micro", "steps", "peak_lr", "r_seed")}
    if m.exists():
        have = json.loads(m.read_text())
        if have != now:
            raise SystemExit("108: OUT %s holds arm %s; this process runs %s" % (OUT, have, now))
    else:
        if (OUT / "main" / "circuits").is_dir() and any((OUT / "main" / "circuits").glob("*.pt")):
            raise SystemExit("108: OUT %s holds unmarked circuits; use a fresh OUT" % OUT)
        m.write_text(json.dumps(now) + "\n")


def check():
    """CPU: the real Runner.method on a bare Runner with a stub method builder; nothing is loaded or written."""
    from types import SimpleNamespace
    import analysis.circuits.gradient_size_sweep_runner as GS
    import protocol_harness as H
    GS._build_mode_method = lambda *a, **k: SimpleNamespace(probe_batch_size=4)
    R = D.Runner.__new__(D.Runner)
    R.H, R.P, R.wcm_version = H, H.P, D.WCM_VERSION
    R.G = dict(inference=None, bank=None, avg_acts=None, M0=SimpleNamespace(probe_builder=None))
    M = D.Runner.method(R, D.GAMMA, D.LAM, True)
    lm = _lm()
    micro = A.MICRO if lm.deep_site_threshold is not None and 35 > int(lm.deep_site_threshold) else M.probe_batch_size
    accum = -(-M.probe_batch_size // micro)
    assert (M.probe_batch_size, micro, accum, int(lm.steps)) == (ARM["batch"], A.MICRO, ARM["accum"], ARM["steps"])
    assert abs(float(lm.lr) - ARM["peak_lr"]) < 1e-9
    print("CHECK ok %s: OUT %s TARGETS %s CTX_DIR %s STOP_EPOCH %s\n  engine: batch_size %d, micro %d, accum %d, steps %d, "
          "lr %.6g (%s, warm-up %g, sum %.4f), wd %g, R weight %g seed %d, rank-keep %g, censor %g, lambda %g, "
          "gamma %g/%g, fill %s, means %s; driver WCM %s, LR_MARK %r" % (
              ARM["name"], OUT, D.TARGETS, D.CTX_DIR, STOP_EPOCH, M.probe_batch_size, micro, accum, lm.steps, lm.lr,
              lm.lr_schedule, lm.warmup_frac, A.lr_sum(lm.steps, lm.lr), lm.weight_decay, lm.resample_weight,
              lm.resample_seed, lm.rank_weight, lm.censor_temp, lm.l1_lambda, lm.dual_floor_weight,
              lm.triple_floor_weight, lm.train_fill, lm.train_means_source, D.WCM_VERSION, D.LR_MARK))


def main():
    if CHECK:
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("108: CHECK=1 runs on CPU only (CUDA_VISIBLE_DEVICES=\"\")")
        D.Runner.method = method
        return check()
    if not D.TARGETS.exists():
        raise SystemExit("108: no TARGETS %s" % D.TARGETS)
    check_marker()
    D.Runner.method, D.Runner.fit = method, fit
    D.main()


if __name__ == "__main__":
    main()
