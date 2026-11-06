"""093: WCM protocol v2 vs v1, paired, on 72 stratified TuringLLM targets (select_targets.py), this GPU.

Arms (both refitted here, through driver.Runner.fit; the only difference is the driver's WCM_VERSION table):
  v1  protocol v1 (dense A / C training fill, the engine's own means)
  v2  A / C training terms under the scorer's sparsity-preserving fill, straight-through gradients, against the
      scorer's own A / C means (train_fill sparse, sparse_grad ste, train_means_source eval_ref)
Contexts: the 15k run's as-run records (062 out_full/out/ctx, read through CTX_DIR, which forces the explicit eval
split). Both arms see the same records, so the held-out leakage of the as-run sets (DAN-157) affects them equally.
Scoring: driver.Runner.score (the production scorer) on held-out strongest, and driver.Runner.spec (056 specificity).

One process runs both arms: per target, the two fits back to back (order alternating by target), switching
Runner.wcm_version and the three engine fields (exactly what Runner.method sets for that version). Resumable per
(target, arm): a row is written once the fit, the score and the specificity are done; circuits are saved per arm
(a fitted circuit is reloaded, not refitted).

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/093-v2-validation/run.py
  env: ARMS (default v1,v2)  LIMIT (first N targets)  OUT93 (default experiments/093-v2-validation/out)
       STEPS (driver: smoke only)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E62 = EXP / "062-h100-protocol-v1"
OUT93 = Path(os.environ.get("OUT93", str(HERE / "out")))
os.environ.setdefault("OUT", str(OUT93 / "driver_out"))             # driver's OUT: nothing is written there by v1
os.environ.setdefault("CTX_DIR", str(E62 / "out_full" / "out" / "ctx"))
os.environ.setdefault("CURVE_STRIDE", "20")
os.environ.pop("WCM_VERSION", None)                                  # the process starts as v1; arms switch per fit
sys.path.insert(0, str(E62))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import driver  # noqa: E402

LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
ARMS = [a for a in os.environ.get("ARMS", "v1,v2").split(",") if a]
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
KEEP = ("n", "a_pos_tk", "a_pos_pre", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre",
        "freeM_topk_pre", "freeN_topk_pre", "phi_sup_blind_pre", "freeM_dense_tk", "freeN_dense_tk", "freeM_dense_pre",
        "freeN_dense_pre", "free0_tr_tk", "free0_a1_tk", "phi_cf_alpha_blind_tk", "vacuous_tk", "n_alpha_off", "e0_tk", "eM_topk_tk", "eN_topk_tk", "eM_dense_tk", "eN_dense_tk", "n_train",
        "n_ho", "n_neg_train", "n_neg_ho", "secs")
SPEC_KEEP = ("target_faith_pre", "sibling_faith_median", "specificity_gap", "rank_circuit", "in_topk_circuit",
             "switched_on_circuit", "n_siblings_scored")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def passes(r, dense=False):
    """The DAN-8 rule: Z, A, C in [0.8, 1.5] (sparsity-preserving A / C; dense=True: the dense fills) and
    necessity >= 0.9, all on the post-Top-K read."""
    names = ("free0_tk", "freeM_dense_tk", "freeN_dense_tk") if dense else ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")
    v = [r.get(k) for k in names]
    nec = r.get("phi_sup_blind_tk")
    if None in v or nec is None:
        return False
    return all(0.8 <= x <= 1.5 for x in v) and nec >= 0.9


def alpha_summary(c):
    a = np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"])
    if a.size == 0:
        return dict(alpha_med=None, alpha_p10=None, alpha_p90=None, alpha_max=None, alpha_min=None, frac_alpha_gt2=None)
    return dict(alpha_med=float(np.median(a)), alpha_p10=float(np.percentile(a, 10)),
                alpha_p90=float(np.percentile(a, 90)), alpha_max=float(a.max()), alpha_min=float(a.min()),
                frac_alpha_gt2=float((a > 2).mean()))


def amp_flag(spec_rows):
    """056 / merge.py's amplifier flag, per ablation and any: target_faith_pre >= 0.5 and sibling_faith_median >=
    0.8 x target_faith_pre."""
    out = {}
    for r in spec_rows:
        t, s = r.get("target_faith_pre"), r.get("sibling_faith_median")
        out[r["pi"]] = bool(t is not None and s is not None and t >= 0.5 and s >= 0.8 * t)
    return out


def set_arm(R, arm):
    """What Runner.method does for this version, without rebuilding the method: the version label and the three
    learned-mask fields (every other field is the same for both versions; H.configure set them)."""
    from config import config
    R.wcm_version = arm
    for k, v in driver.WCM_VERSIONS[arm].items():
        setattr(config.discovery.learned_mask, k, v)


def main():
    keys = [t for t in (HERE / "targets.txt").read_text().split() if t][:LIMIT]
    OUT93.mkdir(parents=True, exist_ok=True)
    rows_path, spec_path = OUT93 / "rows.jsonl", OUT93 / "spec.jsonl"
    done = set()
    if rows_path.exists():
        for line in open(rows_path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                continue
            if "error" not in r:
                done.add((r["seed"], r["arm"]))
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    fh, fs, fr = open(rows_path, "a"), open(spec_path, "a"), open(OUT93 / "train.jsonl", "a")
    t0 = time.time()
    todo = sum(1 for k in keys for a in ARMS if (k, a) not in done)
    print("093: %d targets x arms %s, %d (target, arm) to do, steps %d, out %s, ctx %s, eval split %s" % (
        len(keys), ARMS, todo, driver.STEPS, OUT93, driver.CTX_DIR, R.V.EVAL_SPLIT), flush=True)
    n_done = 0
    for ti, key in enumerate(keys):
        layer, kind = int(key.split(".")[0]), key.split(".")[1]
        for arm in (ARMS if ti % 2 == 0 else ARMS[::-1]):
            if (key, arm) in done:
                continue
            base = dict(seed=key, arm=arm, layer=layer, kind=kind, band=band_of(layer),
                        cell="%s/%s" % (band_of(layer), kind), steps=driver.STEPS, order=ti)
            try:
                rec = R.contexts(key)
                tr_arm, thin = R.train_arm(rec)
                set_arm(R, arm)
                path = OUT93 / "circuits" / arm / ("%s.pt" % key)
                reloaded = path.exists()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                c, t_fit = R.fit(M, rec, key, tr_arm, path, train_fh=fr, label=dict(arm93=arm))
                peak = (round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() and not reloaded
                        else None)
                prov = (M.last_mask_provenance or {}) if not reloaded else {}
                base.update(train_arm=tr_arm, thin=thin, t_fit=None if reloaded else round(t_fit, 1),
                            fit_peak_mb=peak, reloaded=reloaded, train_fill=prov.get("train_fill"),
                            train_means_source=prov.get("train_means_source"),
                            sparse_fill_calls=prov.get("sparse_fill_calls"))
                if c is None:
                    fh.write(json.dumps(dict(base, skip="rejected", wcm_version=arm)) + "\n"); fh.flush()
                    continue
                ts = time.time()
                ho = R.score(c, rec, "strong")
                t_score = time.time() - ts
                ts = time.time()
                spec = R.spec(c, rec, tr_arm)
                t_spec = time.time() - ts
                for r in spec:
                    fs.write(json.dumps(dict(r, arm93=arm)) + "\n")
                fs.flush()
                flags = amp_flag(spec)
                row = dict(base, **{k: ho.get(k) for k in KEEP if k in ho}, wcm_version=ho.get("wcm_version"),
                           n_nodes=driver.n_nodes(c), pass_rule=passes(ho), pass_dense=passes(ho, True),
                           amp_any=any(flags.values()) if flags else None, **{"amp_" + p: v for p, v in flags.items()},
                           **{"spec_%s_%s" % (r["pi"], k): r.get(k) for r in spec for k in SPEC_KEEP},
                           t_score=round(t_score, 1), t_spec=round(t_spec, 1), **alpha_summary(c))
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n_done += 1
                f = lambda x: "%.2f" % x if isinstance(x, (int, float)) else "-"   # noqa: E731
                print("[%5.0fs] %3d/%d %-3s %-16s n %4s | Z %s A %s C %s nec %s pass %s | dA %s dC %s | amp %s | fit %ss"
                      " %sMB score %.0fs spec %.0fs" % (
                          time.time() - t0, n_done, todo, arm, key, row["n_nodes"], f(row.get("free0_tk")),
                          f(row.get("freeM_topk_tk")), f(row.get("freeN_topk_tk")), f(row.get("phi_sup_blind_tk")),
                          row["pass_rule"], f(row.get("freeM_dense_tk")), f(row.get("freeN_dense_tk")),
                          row["amp_any"], base["t_fit"], peak, t_score, t_spec), flush=True)
            except Exception as e:  # noqa: BLE001
                set_arm(R, "v1")
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                fh.write(json.dumps(dict(base, error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n"); fh.flush()
    print("093: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
