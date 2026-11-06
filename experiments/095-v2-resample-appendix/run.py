"""095: the resample appendix (app:resample) under protocol v2. Paired v2 vs v2R on exp 093's 72 targets (DAN-184).

Arms
  v2    093's v2 circuits, REUSED (experiments/093-v2-validation/out/circuits/v2; production v2 = driver WCM_VERSION=v2:
        A / C trained under the sparsity-preserving fill, straight-through, against the scorer's means). Not refitted.
        Its fit time / peak memory are 093's (same GPU).
  v2R   v2 with R in place of C: Z + A + R_C, exp 092's test 3 (t3_ZAR). R_C = 079's donor-resample term with the
        target's 48 TRAINING contrast contexts as donors (091 setup_terms: one donor per training row per forward,
        redrawn every forward, DonorFloors seed 20261017), weight 0.25, under the sparsity-preserving donor fill; the C
        term dropped; no silent term. Engine: rs2_engine (092's t3 patch ported onto today's src, where v2 is native;
        see its docstring for why sx_engine itself cannot run on today's src). Fitted here through driver.Runner.fit
        with WCM_VERSION=v2 (so the A term keeps v2's fill and eval_ref A mean), 400 steps, gamma 0.25, lambda 1e-3.
  v2CR  (added 2026-10-26, Daniel) v2 with C KEPT and R ADDED: Z + A + C + R_C, 079's winning shape (RS0.25_S0.25)
        without its silent-position term (091 showed it adds nothing). R exactly as v2R (weight 0.25, 48 training
        contrast donors redrawn per forward, sparsity-preserving donor fill); C at v2's sparse fill and eval_ref C mean,
        gamma_C 0.25. Same engine: rs2_engine with drop_floor off.
Contexts: the 15k run's as-run records (CTX_DIR = 062 out_full/out/ctx, explicit eval split), as 093. The held-out
sets carry the DAN-157 duplicate leakage; both arms see the same records (paired).
Scoring, both arms, identical code in this process:
  production  driver.Runner.score, held-out strongest (the full row is kept as row["prod"]); rule pass (post-Top-K)
              and the pre-activation pass of protocol 12 (Z_pre = free0_raw_pre / a_pos_pre, necessity_pre =
              (a_pos_pre - phi_sup_blind_raw_pre) / a_pos_pre, A / C = freeM_topk_pre / freeN_topk_pre)
  resample    073 strict_eval.score_target(c_held=True) as in 079's confirmation and 091 / 092: A donors = the 48
              strongest-train contexts, C donors = the 16 HELD-OUT contrast contexts, R donors = 073's evaluation pool,
              N_DRAWS 2; resample pass (079 report.py) = Z (production) in [0.8, 1.5], rs A and rs C in [0.8, 1.5],
              necessity (production, post-Top-K) >= 0.9; vacuous_tk rows excluded in the report
Resumable per (target, arm): a row is written only once its fit (v2R), both scorers and the row are complete; a v2R
circuit is saved to a fresh temp path, its fits.jsonl row written, then renamed into circuits/v2R (a kill never leaves
a partial circuit under the final name; a stale temp file is never read).

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/095-v2-resample-appendix/run.py
  env: LIMIT (first N targets of the balanced order)  OUT95 (default experiments/095-v2-resample-appendix/out)
       STEPS (driver; smoke only)  CHECK=1 (refit the first target's v2 circuit through rs2_engine with every switch
       off and compare with 093's circuit, then exit)  CHECK=2 (the same with v2CR's switches but R weight 0: keep C,
       donor fill on, no R term)  DEADLINE=HH:MM (stop starting new targets when the projected
       end of the next one passes it)  ARMS (default v2,v2R,v2CR)
"""
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
E62, E91, E93 = EXP / "062-h100-protocol-v1", EXP / "091-v2-resample-small", EXP / "093-v2-validation"
OUT95 = Path(os.environ.get("OUT95", str(HERE / "out")))
os.environ.setdefault("OUT", str(OUT95 / "driver_out"))              # the driver's OUT (only its WCM_VERSION marker)
os.environ.setdefault("CTX_DIR", str(E62 / "out_full" / "out" / "ctx"))
os.environ.setdefault("CURVE_STRIDE", "20")
os.environ["WCM_VERSION"] = "v2"
sys.path.insert(0, str(HERE))

import rs2_engine  # noqa: E402

_spec = importlib.util.spec_from_file_location("run091", str(E91 / "run.py"))
R91 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R91)             # 091 helpers (setup_terms, circuit_sig); imports 073 strict_eval -> 062 driver

import torch  # noqa: E402

SE, driver = R91.SE, R91.driver
W_R = 0.25                                 # 092 t3 / 091 v2R
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
FIT_ARMS = {"v2R": dict(drop_floor=True, sp_donor=True),      # Z + A + R_C (092 t3)
            "v2CR": dict(drop_floor=False, sp_donor=True)}    # Z + A + C + R_C (079 shape, no silent term)
ARMS = tuple(a for a in os.environ.get("ARMS", "v2,v2R,v2CR").split(",") if a)
assert all(a == "v2" or a in FIT_ARMS for a in ARMS), ARMS
CMP = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre", "freeM_topk_pre",
       "freeN_topk_pre", "phi_sup_blind_pre")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def ib(v):
    return v is not None and v == v and 0.8 <= v <= 1.5


def rule_pass(p):
    """Post-Top-K rule (headline): Z, A_sp, C_sp in [0.8, 1.5] and necessity >= 0.9."""
    nec = p.get("phi_sup_blind_tk")
    return bool(ib(p.get("free0_tk")) and ib(p.get("freeM_topk_tk")) and ib(p.get("freeN_topk_tk"))
                and nec is not None and nec >= 0.9)


def pre_reads(p):
    """Protocol 12: the pre-activation read, Z and necessity on the 0 ("latent off") baseline."""
    ap, x, xr = p.get("a_pos_pre"), p.get("free0_raw_pre"), p.get("phi_sup_blind_raw_pre")
    if ap is None or x is None or xr is None or ap <= 1e-9:
        return None, None
    return x / ap, (ap - xr) / ap


def pre_pass(p):
    z, nec = pre_reads(p)
    return bool(ib(z) and ib(p.get("freeM_topk_pre")) and ib(p.get("freeN_topk_pre")) and nec is not None
                and nec >= 0.9)


def pre_pass_e0(p):
    """Sensitivity (protocol 12, attention / MLP): the same rule with the scorer's empty-circuit forms."""
    nec = p.get("phi_sup_blind_pre")
    return bool(ib(p.get("free0_pre")) and ib(p.get("freeM_topk_pre")) and ib(p.get("freeN_topk_pre"))
                and nec is not None and nec >= 0.9)


def scalars(d):
    return {k: v for k, v in d.items() if v is None or isinstance(v, (bool, int, float, str))}


def balanced_order():
    """093's targets, cells interleaved (round robin over its 9 band x kind cells), so a prefix stays balanced."""
    t = json.loads((E93 / "targets.json").read_text())
    cells = [c["targets"] for c in t["cells"].values()]
    keys = [cells[j][i] for i in range(max(len(c) for c in cells)) for j in range(len(cells)) if i < len(cells[j])]
    assert sorted(keys) == sorted(k for k in (E93 / "targets.txt").read_text().split() if k)
    return keys


def load_jsonl(path):
    out = []
    if path.exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                continue
    return out


def alpha_med(c):
    import numpy as np
    a = [float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"]
    return float(np.median(a)) if a else None


# ------------------------------------------------------------------------------------------------ CHECK
def check(R, M, key, mode="off"):
    """Refit 093's v2 circuit for `key` through rs2_engine and compare with 093's circuit: the same circuit = the
    patch is inert, on this GPU. mode "off": every switch off; "keepC_w0": v2CR's switches (drop_floor off, donor
    fill on) with R weight 0 (no R term is built)."""
    mod = sys.modules[rs2_engine.MODNAME]
    rs2_engine.reset(mod)
    if mode == "keepC_w0":
        mod.SPEED.update(FIT_ARMS["v2CR"])
        R91.setup_terms(mod, R, R.contexts(key), key, 0.0, 0.0)
    rec = R.contexts(key)
    tr_arm, _ = R.train_arm(rec)
    path = OUT95 / "_check" / time.strftime("%Y%m%d_%H%M%S") / ("%s.pt" % key)
    c, t_fit = R.fit(M, rec, key, tr_arm, path)
    ref = torch.load(E93 / "out" / "circuits" / "v2" / ("%s.pt" % key), weights_only=False)
    a, b = R91.circuit_sig(c), R91.circuit_sig(ref)
    ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
    rs2_engine.reset(mod)
    res = dict(key=key, mode=mode, steps=driver.STEPS, identical=a == b, n=len(a), n_ref=len(b),
               jaccard=len(ka & kb) / max(len(ka | kb), 1), t_fit=round(t_fit, 1),
               term_names=sorted(((M.last_mask_provenance or {}).get("curve") or {}).get("terms", {}).keys()))
    print("095 CHECK %s" % json.dumps(res), flush=True)
    OUT95.mkdir(parents=True, exist_ok=True)
    with open(OUT95 / "check.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


# ------------------------------------------------------------------------------------------------ main
def main():
    keys = balanced_order()[:LIMIT]
    mod = rs2_engine.install()
    R = driver.Runner()
    assert R.wcm_version == "v2", R.wcm_version
    M = R.method(driver.GAMMA, driver.LAM, True)
    from config import config
    lm = config.discovery.learned_mask
    assert (lm.train_fill, lm.sparse_grad, lm.train_means_source) == ("sparse", "ste", "eval_ref")
    if os.environ.get("CHECK") in ("1", "2"):
        check(R, M, keys[0], "off" if os.environ["CHECK"] == "1" else "keepC_w0")
        return
    G = dict(R.G)
    G["R"] = R
    dev = G["device"]
    eval_pool = torch.load(SE.OUT73 / "random_pool.pt")["tokens"].to(dev)
    SE.RAND_DIR = OUT95 / "_no_random_null"                   # random nulls are not part of this comparison
    rows93 = {r["seed"]: r for r in load_jsonl(E93 / "out" / "rows.jsonl") if r.get("arm") == "v2" and "error" not in r}
    OUT95.mkdir(parents=True, exist_ok=True)
    rows_path, fits_path = OUT95 / "rows.jsonl", OUT95 / "fits.jsonl"
    done = {(r["seed"], r["arm"]) for r in load_jsonl(rows_path) if "error" not in r}
    fits = {(r["seed"], r["arm"]): r for r in load_jsonl(fits_path) if "error" not in r}
    fh, ff, ftr = open(rows_path, "a"), open(fits_path, "a"), open(OUT95 / "train.jsonl", "a")
    tmp_root = OUT95 / "_tmp" / ("%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    deadline = os.environ.get("DEADLINE")
    todo = [k for k in keys if any((k, a) not in done for a in ARMS)]
    t0 = time.time()
    per_target = []
    print("095: %d targets (%d to do), arms %s, steps %d, out %s, ctx %s, eval split %s, WCM %s, deadline %s" % (
        len(keys), len(todo), ARMS, driver.STEPS, OUT95, driver.CTX_DIR, R.V.EVAL_SPLIT, R.wcm_version, deadline),
        flush=True)
    for ti, key in enumerate(todo):
        if deadline and per_target:
            hh, mm = (int(x) for x in deadline.split(":"))
            now = time.localtime()
            left = (hh * 60 + mm) * 60 - (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec)
            est = sum(per_target) / len(per_target)
            if est > left:
                print("095: STOP before %s: next target needs ~%.0fs, %.0fs left to the %s deadline" % (
                    key, est, left, deadline), flush=True)
                break
        tk0 = time.time()
        layer, kind = int(key.split(".")[0]), key.split(".")[1]
        rec = R.contexts(key)
        tr_arm, thin = R.train_arm(rec)
        for arm in ARMS:
            if (key, arm) in done:
                continue
            row = dict(seed=key, arm=arm, layer=layer, kind=kind, band=band_of(layer),
                       cell="%s/%s" % (band_of(layer), kind), train_arm=tr_arm, thin=thin, order=keys.index(key))
            try:
                if arm == "v2":
                    cdir = E93 / "out" / "circuits" / "v2"
                    c = torch.load(cdir / ("%s.pt" % key), weights_only=False)
                    r93 = rows93.get(key, {})
                    row.update(fit_source="093", t_fit=r93.get("t_fit"), fit_peak_mb=r93.get("fit_peak_mb"),
                               steps=r93.get("steps"))
                else:
                    cdir = OUT95 / "circuits" / arm
                    final = cdir / ("%s.pt" % key)
                    if final.exists() and (key, arm) in fits:
                        c = torch.load(final, weights_only=False)
                        f = fits[(key, arm)]
                    else:
                        rs2_engine.reset(mod)
                        mod.SPEED.update(FIT_ARMS[arm])
                        floors, _sfn, n_don = R91.setup_terms(mod, R, rec, key, W_R, 0.0)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                        tmp = tmp_root / ("%s.pt" % key)
                        c, t_fit = R.fit(M, rec, key, tr_arm, tmp, train_fh=ftr,
                                         label=dict(arm95=arm, w_R=W_R, donors="contrast_train48",
                                                    drop_C=FIT_ARMS[arm]["drop_floor"], sp_donor=True))
                        prov = M.last_mask_provenance or {}
                        f = dict(seed=key, arm=arm, steps=driver.STEPS, t_fit=round(t_fit, 1),
                                 fit_peak_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20)
                                              if torch.cuda.is_available() else None),
                                 n_donors=n_don, donor_draws=floors.n_draws, donor_sp_calls=mod.DONOR_STATS["calls"],
                                 sparse_fill_calls=prov.get("sparse_fill_calls"), train_fill=prov.get("train_fill"),
                                 train_means_source=prov.get("train_means_source"),
                                 term_names=list(((prov.get("curve") or {}).get("terms") or {}).keys()),
                                 n=driver.n_nodes(c) if c is not None else None, rejected=c is None)
                        rs2_engine.reset(mod)
                        ff.write(json.dumps(f) + "\n"); ff.flush()
                        fits[(key, arm)] = f
                        final.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(tmp, final)
                    row.update(fit_source="095", **{k: v for k, v in f.items() if k not in ("seed", "arm")})
                if c is None:
                    row["skip"] = "rejected"
                else:
                    row["n"] = driver.n_nodes(c)
                    row["alpha_med"] = alpha_med(c)
                    ts = time.time()
                    p = R.score(c, rec, "strong")
                    row["t_score"] = round(time.time() - ts, 1)
                    row["prod"] = scalars(p)
                    z_pre0, nec_pre0 = pre_reads(p)
                    row.update(pass_rule=rule_pass(p), pass_pre=pre_pass(p), pass_pre_e0=pre_pass_e0(p),
                               z_pre0=z_pre0, nec_pre0=nec_pre0, vacuous_tk=p.get("vacuous_tk"),
                               wcm_version=p.get("wcm_version"))
                    if arm == "v2":
                        r93 = rows93.get(key, {})
                        d = [abs(p[k] - r93[k]) for k in CMP if p.get(k) is not None and r93.get(k) is not None]
                        row["max_diff_vs_093"] = max(d) if d else None
                        row["pass_rule_093"] = r93.get("pass_rule")
                    ts = time.time()
                    SE.MAIN_DIR = cdir
                    s = SE.score_target(G, key, eval_pool, c_held=True)
                    row["t_resample"] = round(time.time() - ts, 1)
                    row["resample"] = {k: v for k, v in s.items() if not k.endswith("_draws")}
                    row.update(rs_A=s.get("rs_A_wcm"), rs_C=s.get("rs_C_wcm"), rs_R=s.get("rs_R_wcm"))
                    row["rs_pass"] = bool(ib(p.get("free0_tk")) and ib(row["rs_A"]) and ib(row["rs_C"]) and
                                          (p.get("phi_sup_blind_tk") or 0) >= 0.9)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rs2_engine.reset(mod)
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            g = lambda k, d=row: "%.2f" % d[k] if isinstance(d.get(k), (int, float)) else "-"  # noqa: E731
            pr = row.get("prod", {})
            print("[%5.0fs] %2d/%d %-3s %-16s n %4s | Z %s A %s C %s nec %s rule %s pre %s | rsA %s rsC %s rsR %s "
                  "rs_pass %s | fit %ss score %ss rs %ss %s%s" % (
                      time.time() - t0, ti + 1, len(todo), arm, key, row.get("n"), g("free0_tk", pr),
                      g("freeM_topk_tk", pr), g("freeN_topk_tk", pr), g("phi_sup_blind_tk", pr), row.get("pass_rule"),
                      row.get("pass_pre"), g("rs_A"), g("rs_C"), g("rs_R"), row.get("rs_pass"), row.get("t_fit"),
                      row.get("t_score"), row.get("t_resample"),
                      ("d093 %.2g " % row["max_diff_vs_093"]) if row.get("max_diff_vs_093") is not None else "",
                      row.get("error", "")), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        per_target.append(time.time() - tk0)
        left_n = len(todo) - ti - 1
        est = sum(per_target) / len(per_target) * left_n
        print("095: target %d/%d done in %.0fs; ~%.0f min left (projected end %s)" % (
            ti + 1, len(todo), per_target[-1], est / 60, time.strftime("%H:%M", time.localtime(time.time() + est))),
            flush=True)
    print("095: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
