"""131: fit GPT-2 small circuits with and without the off term O and score them (production scorer + donor test +
the magnitude read). One process, one GPU; resumable per (target, arm).

Arms (ARMS, comma list, in this order per target):
  v4       the production WCM v4 fit, refitted here through the patched engine with OFF weight 0 (bit-identical to the
           unpatched engine: test_off_engine.py (a)); the paired in-run baseline
  o<w>     v4 + O at w_O = w (o0.25, o1, o4, ...)
Every arm: 062 driver.Runner (FAMILY gpt2-small, WCM_VERSION v4, lambda 1e-3, 400 steps, the 2026-11-07 production
records from the pod archive, the production train arm), DEEP_BS (default 1: every local fit since 126) set for all arms.
Per (target, arm), after the fit:
  prod     driver.Runner.score(c, rec, "strong") (eval.wcm_eval.score_circuit, held-out strongest): Z / A / C /
           necessity, pass = wcm_eval.passes (Z, A, C in [0.75, 1.5], nec_b >= 0.9), vacuous = wcm_eval.is_vacuous
  donor    src/eval/donor_test.donor_target (pools A / C / R as production: A = 48 strongest-train, C = the 16
           held-out contrast rows, R = the production run's random pool from the archive), DONOR_DRAWS [8] draws,
           variants fit and a1 (production also ran "pinned"; not needed here). silent_firing = share of the 16 x 64
           silent-base positions with the post-Top-K target > 0, per draw
  mag      the magnitude read, from the same forwards (donor_test.run_full is wrapped to keep its outputs): per pool and
           variant (fit / a1 / empty), the post-Top-K target at the silent-base positions over the natural mean anchor
           activation of the 16 held-out activating bases (donor row natural_mean): mean, p95, p99, max, mean over
           the firing positions; plus the FULL MODEL's own read at those positions (natural_silent_max / _mean: 0 on
           verified-silent contexts)
Rows: out/rows.jsonl (one per (seed, arm); error rows are retried once on a later launch, then skipped),
out/fits.jsonl (fit stats + off-term stats + the off-row checks), out/train.jsonl (training curves, stride 20),
out/circuits/<arm>/<key>.pt (saved to a temp path, renamed after its fits row).
Queue support: out/current.json names the (key, arm) being worked on (the queue's watchdog reads it on a trip);
targets listed in out/tripped.txt are skipped (every arm). STOP_EPOCH: no new (target, arm) is started when its
estimate (EST_FIT) would end after it (exit 7).

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ARMS=v4,o1 .venv/bin/python -X utf8 -u \
       experiments/131-off-term/run.py
  env: ARMS [v4,o1]  TARGETS [targets.txt]  LIMIT  KEYS (comma list, overrides TARGETS)  STEPS (driver; smoke only)
       DONOR_DRAWS [8]  STOP_EPOCH  OUT131 [experiments/131-off-term/out]  DRY_RUN=1 (plan only, no model)
       N131_DEEP_BS [1]  SKIP_DONOR=1 (smoke)
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
PROD_OUT = ARCH / "pub-gpt2-small-S10-results" / "experiments" / "082-public-runs" / "out_gpt2-small"
CTX = PROD_OUT / "ctx"
POOL = PROD_OUT / "donor" / "random_pool.pt"
OUT131 = Path(os.environ.get("OUT131", str(HERE / "out")))
DRV = OUT131 / "driver"                                    # the driver's OUT: only its version markers live here
ARMS = tuple(a for a in os.environ.get("ARMS", "v4,o1").split(",") if a)
DRY = os.environ.get("DRY_RUN") == "1"
STOP_EPOCH = float(os.environ["STOP_EPOCH"]) if os.environ.get("STOP_EPOCH") else None
DRAWS = int(os.environ.get("DONOR_DRAWS", "8"))
VARIANTS = ("fit", "a1")
POOLS = ("A", "C", "R")


# Estimated seconds per v4 fit on the local RTX 5070 Ti by target layer (exp 121's local GPT-2 fits, median per layer,
# L4 / L10 / L11 interpolated / extrapolated); an O arm adds one forward + backward per step on 4 of the 5 data terms'
# rows: x1.25. Scoring + donor test + magnitude ~90 s per (target, arm). Used for the dry-run estimate and the
# STOP_EPOCH rule (a fit is not started when its estimate would end after STOP_EPOCH).
EST_FIT = {0: 40, 1: 60, 2: 90, 3: 125, 4: 165, 5: 205, 6: 220, 7: 260, 8: 300, 9: 450, 10: 520, 11: 600}
EST_EVAL = 90
O_FACTOR = 1.25


def est_seconds(key, arm):
    return EST_FIT.get(int(key.split(".")[0]), 600) * (O_FACTOR if arm_weight(arm) > 0 else 1.0) + EST_EVAL


def arm_weight(arm):
    if arm == "v4":
        return 0.0
    if arm.startswith("o"):
        w = float(arm[1:])
        if w > 0:
            return w
    raise SystemExit("arm %r: v4 or o<w> with w > 0" % arm)


W = {a: arm_weight(a) for a in ARMS}


def load_jsonl(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:                                  # a truncated last line after a kill
                continue
    return out


def targets():
    if os.environ.get("KEYS"):
        keys = [k for k in os.environ["KEYS"].split(",") if k]
    else:
        keys = [k for k in Path(os.environ.get("TARGETS", str(HERE / "targets.txt"))).read_text().split() if k]
    if os.environ.get("LIMIT"):
        keys = keys[:int(os.environ["LIMIT"])]
    return keys


def tripped():
    p = OUT131 / "tripped.txt"
    return {line.split()[0] for line in p.read_text().splitlines() if line.strip()} if p.exists() else set()


def plan():
    rows = load_jsonl(OUT131 / "rows.jsonl")
    done = {(r["seed"], r["arm"]) for r in rows if "error" not in r}
    errs = {}
    for r in rows:
        if "error" in r:
            errs[(r["seed"], r["arm"])] = errs.get((r["seed"], r["arm"]), 0) + 1
    trip = tripped()
    todo = [(k, a) for k in targets() for a in ARMS
            if (k, a) not in done and k not in trip and errs.get((k, a), 0) < 2]
    return todo, done, errs, trip


# ------------------------------------------------------------------------------------------------ environment
def setup_env():
    DRV.mkdir(parents=True, exist_ok=True)
    for m in ("WCM_VERSION", "WCM_R"):                      # the production markers (the Runner refuses others)
        if not (DRV / m).exists():
            (DRV / m).write_bytes((PROD_OUT / m).read_bytes())
    os.environ.update(MODE="main", WCM_VERSION="v4", OUT=str(DRV), CTX_DIR=str(CTX), FAMILY="gpt2-small",
                      SKIP_SPEC="1", DEEP_BS=os.environ.get("N131_DEEP_BS", "1"))
    os.environ.setdefault("CURVE_STRIDE", "20")
    os.environ.setdefault("TAG", "o131_unused")
    for p in (str(ROOT / "src"), str(ROOT / "experiments" / "062-h100-protocol-v1"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------ magnitude read
class Capture:
    """Wraps eval.donor_test.run_full: while on, keeps every forward's post-Top-K read [n, T] (CPU) tagged by what ran
    (natural tap / empty / fit / a1 / pinned). donor_target's call order is fixed: natural(bases), then per pool
    natural(pool) and per draw empty, then each (arm, variant) in spec order; finally the self-donor empty."""

    def __init__(self, DT):
        self.DT, self.on, self.calls = DT, False, []
        self.orig = DT.run_full
        cap = self

        def run_full(inference, make, tokens, pre=False):
            made = []

            def mk(a, z):
                p = make(a, z)
                made.append(p)
                return p
            out = cap.orig(inference, mk, tokens, pre)
            if cap.on and made:
                p = made[0]
                if isinstance(p, DT.ResamplePatcher):
                    tag = "pinned" if p.pins is not None else ("empty" if not p.keep_t else
                                                               ("fit" if p.scales else "a1"))
                else:
                    tag = "natural"
                cap.calls.append((tag, (out[0] if pre else out).detach().float().cpu()))
            return out
        DT.run_full = run_full


def mag_stats(s, nat_mean):
    """s [draws, ns, T] post-Top-K target at the silent positions."""
    import torch
    f = s.flatten()
    pos = f[f > 0]
    den = nat_mean if nat_mean and nat_mean > 1e-9 else None

    def r(v):
        return None if den is None else float(v) / den
    return dict(sf=float((f > 0).float().mean()), mean=r(f.mean()), p95=r(torch.quantile(f, 0.95)),
                p99=r(torch.quantile(f, 0.99)), max=r(f.max()), mean_firing=r(pos.mean()) if pos.numel() else 0.0,
                sf_draws=[float((d > 0).float().mean()) for d in s])


def magnitude(calls, row):
    """The magnitude read from one donor_target call's captured forwards (one arm, variants fit / a1)."""
    import torch
    nb, ns, nat = int(row["n_base"]), int(row["n_silent"]), float(row["natural_mean"])
    by = {}
    for tag, t in calls:
        by.setdefault(tag, []).append(t)
    want = {"natural": 1 + len(POOLS), "empty": len(POOLS) * DRAWS + 1, "fit": len(POOLS) * DRAWS,
            "a1": len(POOLS) * DRAWS}
    got = {k: len(by.get(k, [])) for k in want}
    if got != want:
        return dict(error="capture count %s != %s" % (got, want))
    out = dict(natural_silent_max=float(by["natural"][0][nb:].max()) if ns else None,
               natural_silent_mean=float(by["natural"][0][nb:].mean()) if ns else None, natural_mean=nat)
    for tag in ("fit", "a1", "empty"):
        for i, ds in enumerate(POOLS):
            s = torch.stack([t[nb:] for t in by[tag][i * DRAWS:(i + 1) * DRAWS]])
            out["%s_%s" % (tag, ds)] = mag_stats(s, nat)
    # consistency: the capture reproduces the row's own silent_firing (same forwards)
    sf_row = row["silent_firing"]["fit"]["C"]
    out["sf_check_max_diff"] = max(abs(a - b) for a, b in zip(sf_row, out["fit_C"]["sf_draws"]))
    return out


# ------------------------------------------------------------------------------------------------ off-row checks
def off_rows_check(rec, dev, M):
    """(c) the off term's rows = the R donors = eval_reference's 48 contrast-train rows = rec["neg"][:48], and no row
    of them is one of the held-out contrast rows the donor test and the scorer read."""
    import torch
    from circuit.protocol_contexts import eval_probe, eval_reference
    from eval.wcm_eval import contrast_split
    ref_c = eval_reference(rec, dev, split="explicit")[1]
    probe = eval_probe(rec, "strong", dev)
    held = probe.neg_tokens[contrast_split(len(probe.neg_tokens), probe.metadata):]
    donors = M._resample_donors
    same = donors is not None and tuple(donors.shape) == tuple(ref_c.shape) and bool(torch.equal(donors, ref_c))
    neg48 = bool(torch.equal(ref_c.cpu(), torch.as_tensor(rec["neg"])[:ref_c.shape[0]].to(ref_c.dtype).cpu()))
    eq = (ref_c[:, None, :] == held[None, :, :]).all(-1)                     # [48, 16] exact row matches
    return dict(n_off=int(ref_c.shape[0]), n_held_contrast=int(held.shape[0]), off_eq_resample_donors=same,
                off_eq_neg48=neg48, off_rows_in_heldout=int(eq.any(-1).sum()), off_tokens_sum=int(ref_c.long().sum()))


def alpha_med(c):
    import numpy as np
    a = [float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"]
    return float(np.median(a)) if a else None


def scalars(d):
    return {k: v for k, v in d.items() if v is None or isinstance(v, (bool, int, float, str))}


# ------------------------------------------------------------------------------------------------ main
def main():
    todo, done, errs, trip = plan()
    keys = targets()
    print("131: %d targets x arms %s (w_O %s) | %d (key, arm) to do, %d done, %d error rows, %d tripped targets %s | "
          "draws %d | stop epoch %s | out %s" % (
              len(keys), ARMS, W, len(todo), len(done), sum(errs.values()), len(trip), sorted(trip), DRAWS,
              time.strftime("%a %H:%M", time.localtime(STOP_EPOCH)) if STOP_EPOCH else "none", OUT131), flush=True)
    for p in (CTX, POOL, PROD_OUT / "WCM_VERSION", PROD_OUT / "WCM_R"):
        if not p.exists():
            raise SystemExit("missing %s" % p)
    missing = [k for k in keys if not (CTX / ("%s.pt" % k)).exists()]
    if missing:
        raise SystemExit("no record for %s" % missing)
    if DRY:
        tot = 0.0
        for k, a in todo:
            tot += est_seconds(k, a)
            print("  PLAN %-18s %-6s w_O %-5g est %4.0f s (cumulative %5.1f min)" % (k, a, W[a], est_seconds(k, a),
                                                                                tot / 60))
        print("131: DRY RUN (no model loaded): %d fits, estimate %.0f min + ~2 min start-up" % (len(todo), tot / 60))
        return 0
    if not todo:
        print("131: DONE (nothing to do)")
        return 0
    setup_env()
    import torch
    import off_engine
    mod = off_engine.install()
    import driver  # noqa: E402  (reads its config from the env at import)
    import eval.donor_test as DT
    from eval.wcm_eval import is_vacuous, passes
    assert driver.FAMILY == "gpt2-small" and Path(driver.OUT) == DRV and driver.WCM_VERSION == "v4", "driver config"
    cap = Capture(DT)
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    from config import config
    lm = config.discovery.learned_mask
    print("131: driver FAMILY %s OUT %s CTX_DIR %s LAM %g STEPS %d WCM %s | lm: fill %s means %s R %g/%s/%d censor %s "
          "rank %g lr %g sched %s deep_bs %s binarize %s" % (
              driver.FAMILY, driver.OUT, driver.CTX_DIR, driver.LAM, driver.STEPS, driver.WCM_VERSION, lm.train_fill,
              lm.train_means_source, lm.resample_weight, lm.resample_pool, lm.resample_k, lm.censor_temp,
              lm.rank_weight, lm.lr, lm.lr_schedule, lm.deep_batch_size, lm.binarize), flush=True)
    G = R.G
    dev = G["device"]
    pool = torch.load(POOL, weights_only=False, map_location="cpu")["tokens"]
    OUT131.mkdir(parents=True, exist_ok=True)
    fits = {(f["seed"], f["arm"]): f for f in load_jsonl(OUT131 / "fits.jsonl") if "error" not in f}
    fh, ff, ftr = open(OUT131 / "rows.jsonl", "a"), open(OUT131 / "fits.jsonl", "a"), open(OUT131 / "train.jsonl", "a")
    tmp_root = OUT131 / "_tmp" / ("%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    cur = OUT131 / "current.json"
    t0 = time.time()
    rc = 0
    for i, (key, arm) in enumerate(todo):
        if key in tripped():
            continue
        if STOP_EPOCH and time.time() + est_seconds(key, arm) >= STOP_EPOCH:
            print("131: STOP_EPOCH: %s %s (est %.0f s) would end after %s: not started (resumable)" % (
                key, arm, est_seconds(key, arm), time.strftime("%H:%M", time.localtime(STOP_EPOCH))), flush=True)
            rc = 7
            break
        cur.write_text(json.dumps(dict(seed=key, arm=arm, pid=os.getpid(), t=time.strftime("%F %T"))) + "\n")
        ts = time.time()
        layer, kind = int(key.split(".")[0]), key.split(".")[1]
        row = dict(seed=key, arm=arm, w_O=W[arm], layer=layer, kind=kind)
        try:
            rec = R.contexts(key)
            tr_arm, thin = R.train_arm(rec)
            row.update(train_arm=tr_arm, thin=thin)
            final = OUT131 / "circuits" / arm / ("%s.pt" % key)
            if final.exists() and (key, arm) in fits:
                c = torch.load(final, weights_only=False)
                f = fits[(key, arm)]
            else:
                off_engine.configure(W[arm])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                tmp = tmp_root / arm / ("%s.pt" % key)
                c, t_fit = R.fit(M, rec, key, tr_arm, tmp, train_fh=ftr, label=dict(arm131=arm, w_O=W[arm]))
                prov = M.last_mask_provenance or {}
                f = dict(seed=key, arm=arm, w_O=W[arm], steps=driver.STEPS, t_fit=round(t_fit, 1),
                         fit_peak_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20)
                                      if torch.cuda.is_available() else None),
                         term_names=list(((prov.get("curve") or {}).get("terms") or {}).keys()),
                         resample_n_donors=prov.get("resample_n_donors"), train_fill=prov.get("train_fill"),
                         n=driver.n_nodes(c) if c is not None else None, rejected=c is None,
                         off=off_engine.stats_summary(), checks=off_rows_check(rec, dev, M))
                off_engine.configure(0.0)
                ff.write(json.dumps(f) + "\n"); ff.flush()
                fits[(key, arm)] = f
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp, final)
            row.update(t_fit=f.get("t_fit"), fit_peak_mb=f.get("fit_peak_mb"), term_names=f.get("term_names"),
                       off=f.get("off"), checks=f.get("checks"))
            if c is None:
                row["skip"] = "rejected"
            else:
                row["n"] = driver.n_nodes(c)
                row["alpha_med"] = alpha_med(c)
                t1 = time.time()
                p = R.score(c, rec, "strong")
                row["t_score"] = round(time.time() - t1, 1)
                row["prod"] = scalars(p)
                row.update(pass_prod=bool(passes(p)), vacuous=bool(is_vacuous(p)))
                if os.environ.get("SKIP_DONOR") != "1":
                    t1 = time.time()
                    cap.calls, cap.on = [], True
                    try:
                        drows, _raws = DT.donor_target(G, key, rec, {arm: c}, pool, variants=VARIANTS, draws=DRAWS)
                    finally:
                        cap.on = False
                    d = drows[arm]
                    row["t_donor"] = round(time.time() - t1, 1)
                    row["donor"] = {k: d[k] for k in ("recovery", "recovery_pre", "silent_firing", "empty_leak",
                                                      "natural_mean", "natural_pre_mean", "n_base", "n_silent",
                                                      "self_empty", "pool_sizes", "donor_fires_frac")}
                    row["resample_pass"] = bool(DT.resample_pass(p, d["recovery"]["fit"]))
                    row["mag"] = magnitude(cap.calls, d)
                    cap.calls = []
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            off_engine.configure(0.0)
            row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        row["t_total"] = round(time.time() - ts, 1)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        pr = row.get("prod", {})
        g = lambda v: "-" if not isinstance(v, (int, float)) else "%.2f" % v  # noqa: E731
        sfc = (row.get("mag") or {}).get("fit_C", {})
        print("[%5.0fs] %2d/%d %-18s %-5s n %5s | Z %s A %s C %s nec %s pass %s | SF_C %s mag_C %s p95 %s | rs %s | "
              "fit %ss total %ss %s" % (
                  time.time() - t0, i + 1, len(todo), key, arm, row.get("n"), g(pr.get("free0_tk")),
                  g(pr.get("freeM_topk_tk")), g(pr.get("freeN_topk_tk")), g(pr.get("phi_sup_blind_tk")),
                  row.get("pass_prod"), g(sfc.get("sf")), g(sfc.get("mean")), g(sfc.get("p95")),
                  row.get("resample_pass"), row.get("t_fit"), row.get("t_total"), row.get("error", "")), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cur.write_text(json.dumps(dict(idle=True, pid=os.getpid(), t=time.strftime("%F %T"))) + "\n")
    print("131: %s %.0fs" % ("STOPPED" if rc else "DONE", time.time() - t0), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
