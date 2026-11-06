"""102 phase 1 (DAN-187 track G): arms on top of the base algorithm v2CR (095: Z + A + C + R_C, lambda 1e-3, 400 steps,
R weight 0.25, donors = the target's 48 training contrast contexts redrawn per forward, sparsity-preserving donor fill,
no silent term), fitted and scored per (target, arm) with IDENTICAL scoring code for every arm.

Arms
  base      095's v2CR circuits, REUSED (095 out/circuits/v2CR; their fit rows from 095's fits.jsonl), re-scored here.
            Reuse is justified by --check-base (refits through this engine are bit-identical to 095's circuits).
            --base-source fit refits it here instead.
  seed1/2   the base refit with the R donor generator seeded 20261017 + 1 / + 2 (the fit's only randomness: the
            engine is full-batch with a fixed initialisation and no dropout; the micro-batch cycle is deterministic)
  r0.5/r1.0 the base with R weight 0.5 / 1.0, C kept (100 E4's v2cr-w0.5 / v2cr-w1)
  lamdepth  the base with lambda 1e-3 / 5e-4 / 2.5e-4 for L0-4 / L5-7 / L8-11; L0-4 targets are not refitted (no row:
            the report takes the base row, which is the same algorithm)
  twostage  the base circuit's membership fixed (100's support restriction: support = the base members, lambda 0,
            binarize none, theta_init 40), gains warm-started at the base circuit's alphas (p102 amp_init), then 200
            steps under every term (Z, A, C, R at 0.25, rank-keep); t_fit is this second stage only
Engine: p102_engine (100 e4_engine on a frozen copy of 095 / 100's learned_mask.py; see its docstring).
Contexts: as 093 / 095, the 15k run's as-run records (062 out_full/out/ctx; DAN-157 leakage applies, paired design).
Scoring (every arm, this process, the same functions as 095 run.py):
  production  driver.Runner.score(c, rec, "strong"); full scalar row kept as row["prod"]
  pre-act.    095's pass_pre (protocol 12: Z and necessity on the 0 baseline; A / C pre reads)
  resample    073 strict_eval.score_target(G, key, eval_pool, c_held=True) (095's call: A donors = 48 strongest-train,
              C donors = the 16 held-out contrast contexts, R donors = 073's evaluation pool, 2 draws)
Pass flags in rows are at floor FLOOR (0.75): production = Z, A_sp, C_sp in [0.75, 1.5] and necessity >= 0.9;
resample = production AND rs_A, rs_C in [0.75, 1.5]; rs_pass095 = 095's definition (Z, rs_A, rs_C, necessity) at
0.75. The report recomputes every flag from the raw values.
Resumable per (target, arm): circuits fit to a temp path, the fits.jsonl row is written, then renamed into
circuits/<arm>/<key>.pt; a rows.jsonl row is written only after both scorers finish. A kill leaves no partial row.

  wsl (repo root): PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \\
      experiments/102-pass-improvement/run.py --arms base,seed1,r1.0 [--start-by 2026-10-31T23:30] [--end-by ...]
  --check-base 2.attn.22342,9.resid.123   refit v2CR for these keys, compare with 095's circuits, then exit
  --panel check                           the frozen check panel (phase 3 only; needs --allow-check)
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
E62, E91, E93, E95, E100 = (EXP / "062-h100-protocol-v1", EXP / "091-v2-resample-small", EXP / "093-v2-validation",
                            EXP / "095-v2-resample-appendix", EXP / "100-scrub-frontier")
GAMMA = 0.25
LAM = 1e-3
FLOOR, CEIL, NEC = 0.75, 1.5, 0.9
BANDS = (("L0-4", 0, 4), ("L5-7", 5, 7), ("L8-11", 8, 11))
ARMS = {
    "base": dict(w=0.25, seed=0),
    "seed1": dict(w=0.25, seed=1),
    "seed2": dict(w=0.25, seed=2),
    "r0.5": dict(w=0.5, seed=0),
    "r1.0": dict(w=1.0, seed=0),
    "lamdepth": dict(w=0.25, seed=0, lam_band={"L0-4": None, "L5-7": 5e-4, "L8-11": 2.5e-4}),
    "twostage": dict(w=0.25, seed=0, two_stage=True, steps=200),
    # added 2026-10-31 ~11:55 (coordinator, from diag-102: most v2CR Z failures restore the pre-activation and lose at
    # the Top-K cut): stronger rank-keep (056's term, production mode "keep", delta 0; the engine also has a "cut" mode
    # with margin rank_delta, not used here), and R 1.0 combined with the better of the two
    "rk3e-2": dict(w=0.25, seed=0, rank=3e-2),
    "rk1e-1": dict(w=0.25, seed=0, rank=1e-1),
    "r1.0_rk3e-2": dict(w=1.0, seed=0, rank=3e-2),
    "r1.0_rk1e-1": dict(w=1.0, seed=0, rank=1e-1),
}
RANK = 3e-3                                     # 062 driver.RANK, the protocol's rank-keep weight
CMP = ("free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre", "freeM_topk_pre",
       "freeN_topk_pre", "phi_sup_blind_pre")


def band_of(layer):
    return next(b for b, lo, hi in BANDS if lo <= layer <= hi)


def ib(v, lo=FLOOR):
    return v is not None and v == v and lo <= v <= CEIL


# ---- 095 run.py definitions (copied; attribution experiments/095-v2-resample-appendix/run.py), floor as a parameter
def rule_pass(p, lo=FLOOR):
    nec = p.get("phi_sup_blind_tk")
    return bool(ib(p.get("free0_tk"), lo) and ib(p.get("freeM_topk_tk"), lo) and ib(p.get("freeN_topk_tk"), lo)
                and nec is not None and nec >= NEC)


def pre_reads(p):
    ap, x, xr = p.get("a_pos_pre"), p.get("free0_raw_pre"), p.get("phi_sup_blind_raw_pre")
    if ap is None or x is None or xr is None or ap <= 1e-9:
        return None, None
    return x / ap, (ap - xr) / ap


def pre_pass(p, lo=FLOOR):
    z, nec = pre_reads(p)
    return bool(ib(z, lo) and ib(p.get("freeM_topk_pre"), lo) and ib(p.get("freeN_topk_pre"), lo) and nec is not None
                and nec >= NEC)


def scalars(d):
    return {k: v for k, v in d.items() if v is None or isinstance(v, (bool, int, float, str))}


def load_jsonl(path):
    out = []
    if path.exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001  (a truncated last line after a kill)
                continue
    return out


def balanced(cells):
    """095 run.balanced_order: the panel's 9 band x kind cells interleaved round robin (a prefix stays balanced)."""
    cells = [c["targets"] for c in cells.values()]
    return [cells[j][i] for i in range(max(len(c) for c in cells)) for j in range(len(cells)) if i < len(cells[j])]


def panel_keys(panel):
    if panel == "dev":
        keys = balanced(json.loads((E93 / "targets.json").read_text())["cells"])
        assert sorted(keys) == sorted(k for k in (E93 / "targets.txt").read_text().split() if k)
        return keys
    return balanced(json.loads((HERE / "check_targets.json").read_text())["cells"])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_mem_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        return int(o.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="base")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "phase1")
    ap.add_argument("--panel", choices=("dev", "check"), default="dev")
    ap.add_argument("--allow-check", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keys", default=None, help="comma list: restrict to these keys (smoke)")
    ap.add_argument("--steps", type=int, default=400, help="the protocol's 400; smoke only")
    ap.add_argument("--base-source", choices=("095", "fit"), default="095")
    ap.add_argument("--check-base", default=None)
    ap.add_argument("--start-by", default=None, help="ISO datetime: no (target, arm) starts after it")
    ap.add_argument("--end-by", default=None, help="ISO datetime: no (target, arm) starts if its projected end passes it")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    assert all(a in ARMS for a in arms), arms
    if args.panel == "check" and not args.allow_check:
        raise SystemExit("the check panel is frozen for phase 3 (--allow-check)")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OUT", str(out / "driver_out"))
    os.environ.setdefault("CTX_DIR", str(E62 / "out_full" / "out" / "ctx"))
    os.environ["CURVE_STRIDE"] = "20"
    os.environ["WCM_VERSION"] = "v2"
    os.environ["STEPS"] = str(args.steps)
    sys.path.insert(0, str(HERE))
    import p102_engine as ENG                        # imports 100 e4_engine -> 095 rs2_engine -> 079 / 092 (095's order)
    spec = importlib.util.spec_from_file_location("run091", str(E91 / "run.py"))
    R91 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R91)                     # 091 setup_terms / donor_tokens; imports 073 strict_eval -> driver
    spec = importlib.util.spec_from_file_location("run100", str(E100 / "run.py"))
    R100 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R100)                    # 100 helpers: members_by_site, circuit_sig, alpha_stats, clean
    import torch
    SE, driver = R91.SE, R91.driver
    assert driver.STEPS == args.steps
    mod = ENG.install()
    SUPPORT = {}
    orig_run = mod.run_learned_mask

    def wrapped(inference, bank, **kw):              # 100 run.py's support wrapper (random control), reused for twostage
        if SUPPORT:
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in SUPPORT.items()},
                      l1_lambda=0.0, binarize="none", theta_init=40.0)
        return orig_run(inference, bank, **kw)
    mod.run_learned_mask = wrapped
    R = driver.Runner()
    assert R.wcm_version == "v2", R.wcm_version
    from config import config
    lm = config.discovery.learned_mask
    G = dict(R.G)
    G["R"] = R
    dev = G["device"]
    keys = panel_keys(args.panel)
    if args.keys:
        want = [k for k in args.keys.split(",") if k]
        keys = [k for k in want if k in keys] if args.check_base is None else want
    keys = keys[:args.limit]

    # ---- provenance: hashes of every source file this process can import, and the frozen engine text
    src_paths = sorted(set(ROOT.joinpath("src").rglob("*.py")) | set(HERE.glob("*.py")) | set(E95.glob("*.py"))
                       | set(E100.glob("*.py")) | {E91 / "run.py", E62 / "driver.py", EXP / "073-strict-eval/strict_eval.py",
                                                   EXP / "079-resample-training/rs_engine.py", ROOT / "config.yaml",
                                                   ENG.FROZEN})
    src_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in src_paths}
    m100 = json.loads((E100 / "out" / "e4" / "sources" / "20261030_222353.json").read_text())
    changed_vs_100 = sorted(k for k, v in m100.items() if src_hashes.get(k) != v)
    inv = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, arms=arms, panel=args.panel, n_keys=len(keys),
               torch=torch.__version__, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               run_py_sha256=file_hash(Path(__file__)), engine_sha256=hashlib.sha256(ENG.patched_source().encode()).hexdigest(),
               frozen_learned_mask_sha256=ENG.FROZEN_SHA256, changed_vs_100_manifest=changed_vs_100)
    with open(out / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(inv) + "\n")
    (out / "sources").mkdir(exist_ok=True)
    (out / "sources" / ("%s_%d.json" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid()))).write_text(
        json.dumps(src_hashes, indent=0))
    print("102: engine = frozen learned_mask %s + rs2 + e4 + p102; files changed vs 100's manifest: %s" % (
        ENG.FROZEN_SHA256[:12], changed_vs_100), flush=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def fit(arm, key, rec, tr_arm, tmp, ftr, label_extra=None):
        """One fit of `arm` through the engine. Returns (circuit, fit row fields)."""
        a = ARMS[arm]
        layer = int(key.split(".")[0])
        ENG.reset(mod)
        mod.SPEED.update(drop_floor=False, sp_donor=True)                       # 095 v2CR switches
        dseed = R91.TRAIN_POOL_SEED + int(a["seed"])
        floors, _sfn, n_don = R91.setup_terms(mod, R, rec, key, float(a["w"]), 0.0, seed=dseed)
        lam = LAM
        if a.get("lam_band"):
            lam = a["lam_band"][band_of(layer)]
            assert lam is not None, "lamdepth L0-4 is the base"
        steps = int(a.get("steps", args.steps)) if args.steps == 400 else args.steps
        SUPPORT.clear()
        n_base = None
        if a.get("two_stage"):
            c0 = torch.load(base_dir() / ("%s.pt" % key), weights_only=False)
            mem = R100.members_by_site(c0)
            SUPPORT.update({s: sorted(v) for s, v in mem.items()})
            amp = {}
            for nd in c0.nodes.values():
                md = nd.metadata
                if md.get("role") == "seed":
                    continue
                f = md["feature_id"]
                amp.setdefault((int(f.layer), str(f.kind)), {})[int(f.index)] = float(md.get("amplitude", 1.0))
            mod.P102["amp_init"] = amp
            n_base = driver.n_nodes(c0)
        rank = float(a.get("rank", RANK))
        M = R.method(GAMMA, lam, True, steps=steps, rank=rank)
        assert (lm.train_fill, lm.sparse_grad, lm.train_means_source) == ("sparse", "ste", "eval_ref")
        assert (lm.rank_weight, lm.rank_mode, lm.rank_delta) == (rank, "keep", 0.0), (lm.rank_weight, lm.rank_mode)
        mem0 = gpu_mem_used()
        if torch.cuda.is_available():
            sync()
            torch.cuda.reset_peak_memory_stats()
        try:
            c, t_fit = R.fit(M, rec, key, tr_arm, tmp, train_fh=ftr,
                             label=dict(exp="102", arm102=arm, w_R=a["w"], donor_seed=dseed, lam=lam, steps=steps,
                                        two_stage=bool(a.get("two_stage")), **(label_extra or {})))
        finally:
            SUPPORT.clear()
        prov = M.last_mask_provenance or {}
        curve = (prov.get("curve") or {}).get("loss") or []
        f = dict(steps=steps, lam=lam, w_R=a["w"], donor_seed=dseed, rank_weight=rank, t_fit=round(t_fit, 2),
                 fit_peak_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None),
                 gpu_mem_used_at_start=mem0, n_donors=n_don, donor_draws=floors.n_draws,
                 donor_sp_calls=mod.DONOR_STATS["calls"], train_fill=prov.get("train_fill"),
                 train_means_source=prov.get("train_means_source"), support_n=prov.get("support_n"),
                 amp_init_n=mod.P102["amp_init_n"], n_base=n_base,
                 term_names=list(((prov.get("curve") or {}).get("terms") or {}).keys()),
                 loss_final=prov.get("loss_final"),
                 loss_last20=(sum(curve[-20:]) / len(curve[-20:])) if curve else None,
                 n=driver.n_nodes(c) if c is not None else None, rejected=c is None)
        if a.get("two_stage"):
            if f["support_n"] is None or f["amp_init_n"] != n_base:
                raise RuntimeError("two-stage: support / amp_init did not reach the engine (%s)" % f)
        ENG.reset(mod)
        return c, f

    def base_dir():
        return (E95 / "out" / "circuits" / "v2CR") if args.base_source == "095" else (out / "circuits" / "base")

    # ---- check mode: refit the base (v2CR) for the listed keys, compare with 095's circuits
    if args.check_base:
        ck = [k for k in args.check_base.split(",") if k]
        with open(out / "train_check.jsonl", "a") as ftr, open(out / "check.jsonl", "a") as fc:
            for key in ck:
                rec = R.contexts(key)
                tr_arm, _ = R.train_arm(rec)
                tmp = out / "_check" / ("%s_%d" % (time.strftime("%H%M%S"), os.getpid())) / ("%s.pt" % key)
                c, f = fit("base", key, rec, tr_arm, tmp, ftr, dict(check=True))
                ref = torch.load(E95 / "out" / "circuits" / "v2CR" / ("%s.pt" % key), weights_only=False)
                a_, b_ = R100.circuit_sig(c), R100.circuit_sig(ref)
                ka, kb = {x[:3] for x in a_}, {x[:3] for x in b_}
                r = dict(seed=key, time=time.strftime("%Y-%m-%dT%H:%M:%S"), identical=a_ == b_, n=len(a_), n_ref=len(b_),
                         jaccard=len(ka & kb) / max(len(ka | kb), 1),
                         max_alpha_diff=(max(abs(x[3] - y[3]) for x, y in zip(a_, b_)) if ka == kb else None),
                         t_fit=f["t_fit"], term_names=f["term_names"], donor_sp_calls=f["donor_sp_calls"],
                         changed_vs_100_manifest=changed_vs_100)
                print("102 CHECK %s" % json.dumps(r), flush=True)
                fc.write(json.dumps(r) + "\n"); fc.flush()
        return

    eval_pool = torch.load(SE.OUT73 / "random_pool.pt")["tokens"].to(dev)
    SE.RAND_DIR = out / "_no_random_null"                    # random nulls are not part of this comparison (as 095)
    fits95 = {r["seed"]: r for r in load_jsonl(E95 / "out" / "fits.jsonl") if r.get("arm") == "v2CR" and "error" not in r}
    rows95 = {r["seed"]: r for r in load_jsonl(E95 / "out" / "rows.jsonl") if r.get("arm") == "v2CR" and "error" not in r}
    rows_path, fits_path = out / "rows.jsonl", out / "fits.jsonl"
    done = {(r["seed"], r["arm"]) for r in load_jsonl(rows_path) if "error" not in r}
    fits = {(r["seed"], r["arm"]): r for r in load_jsonl(fits_path) if "error" not in r}
    fh, ff, ftr = open(rows_path, "a"), open(fits_path, "a"), open(out / "train.jsonl", "a")
    tmp_root = out / "_tmp" / ("%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid()))
    start_by = dt.datetime.fromisoformat(args.start_by) if args.start_by else None
    end_by = dt.datetime.fromisoformat(args.end_by) if args.end_by else None
    durs = {}                                              # (arm, band) -> list of seconds per (target, arm)
    t0 = time.time()
    print("102: panel %s, %d targets, arms %s, steps %d, out %s, ctx %s, eval split %s" % (
        args.panel, len(keys), arms, args.steps, out, driver.CTX_DIR, R.V.EVAL_SPLIT), flush=True)
    for arm in arms:
        todo = [k for k in keys if (k, arm) not in done
                and not (ARMS[arm].get("lam_band") and ARMS[arm]["lam_band"][band_of(int(k.split(".")[0]))] is None)]
        print("102: arm %s: %d (target, arm) to do" % (arm, len(todo)), flush=True)
        stop = False
        for ti, key in enumerate(todo):
            layer, kind = int(key.split(".")[0]), key.split(".")[1]
            band = band_of(layer)
            now = dt.datetime.now()
            if start_by and now > start_by:
                print("102: STOP (start-by %s passed) before %s %s" % (start_by, arm, key), flush=True)
                stop = True
                break
            if end_by:
                hist = durs.get((arm, band)) or [s for (a_, b_), v in durs.items() if b_ == band for s in v]
                if hist and now + dt.timedelta(seconds=sum(hist) / len(hist)) > end_by:
                    print("102: STOP before %s %s: ~%.0fs needed, end-by %s" % (arm, key, sum(hist) / len(hist), end_by),
                          flush=True)
                    stop = True
                    break
            tk0 = time.time()
            rec = R.contexts(key)
            tr_arm, thin = R.train_arm(rec)
            row = dict(seed=key, arm=arm, panel=args.panel, layer=layer, kind=kind, band=band,
                       cell="%s/%s" % (band, kind), train_arm=tr_arm, thin=thin, order=keys.index(key))
            try:
                if arm == "base" and args.base_source == "095":
                    cdir = base_dir()
                    c = torch.load(cdir / ("%s.pt" % key), weights_only=False)
                    f95 = fits95.get(key, {})
                    f = dict(fit_source="095", t_fit=f95.get("t_fit"), fit_peak_mb=f95.get("fit_peak_mb"),
                             steps=f95.get("steps"), n_donors=f95.get("n_donors"), lam=LAM, w_R=0.25,
                             donor_seed=R91.TRAIN_POOL_SEED)
                else:
                    cdir = out / "circuits" / arm
                    final = cdir / ("%s.pt" % key)
                    if final.exists() and (key, arm) in fits:
                        c = torch.load(final, weights_only=False)
                        f = fits[(key, arm)]
                    else:
                        tmp = tmp_root / arm / ("%s.pt" % key)
                        c, f = fit(arm, key, rec, tr_arm, tmp, ftr)
                        f = dict(seed=key, arm=arm, fit_source="102", **f)
                        ff.write(json.dumps(R100.clean(f)) + "\n"); ff.flush()
                        fits[(key, arm)] = f
                        final.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(tmp, final)
                row["fit"] = {k: v for k, v in f.items() if k not in ("seed", "arm")}
                row["t_fit"] = f.get("t_fit")
                if c is None:
                    row["skip"] = "rejected"
                else:
                    row["n"] = driver.n_nodes(c)
                    row["alpha"] = R100.alpha_stats(c)
                    al = [float(nd.metadata.get("amplitude", 1.0)) for nd in c.nodes.values()
                          if nd.metadata.get("role") != "seed"]
                    row["alpha"]["share_gt2"] = (sum(1 for v in al if v > 2) / len(al)) if al else None
                    if arm != "base" and (base_dir() / ("%s.pt" % key)).exists():
                        cb = torch.load(base_dir() / ("%s.pt" % key), weights_only=False)
                        if cb is not None:
                            ka = {x[:3] for x in R100.circuit_sig(c)}
                            kb = {x[:3] for x in R100.circuit_sig(cb)}
                            row["members_jaccard_vs_base"] = len(ka & kb) / max(len(ka | kb), 1)
                    sync(); ts = time.time()
                    p = R.score(c, rec, "strong")
                    sync(); row["t_score"] = round(time.time() - ts, 2)
                    row["prod"] = scalars(p)
                    z_pre0, nec_pre0 = pre_reads(p)
                    row.update(pass_prod=rule_pass(p), pass_pre=pre_pass(p), z_pre0=z_pre0, nec_pre0=nec_pre0,
                               vacuous_tk=p.get("vacuous_tk"), wcm_version=p.get("wcm_version"))
                    sync(); ts = time.time()
                    SE.MAIN_DIR = cdir
                    s = SE.score_target(G, key, eval_pool, c_held=True)
                    sync(); row["t_resample"] = round(time.time() - ts, 2)
                    row["resample"] = {k: v for k, v in s.items() if not k.endswith("_draws")}
                    row.update(rs_A=s.get("rs_A_wcm"), rs_C=s.get("rs_C_wcm"), rs_R=s.get("rs_R_wcm"))
                    row["pass_rs"] = bool(row["pass_prod"] and ib(row["rs_A"]) and ib(row["rs_C"]))
                    row["rs_pass095"] = bool(ib(p.get("free0_tk")) and ib(row["rs_A"]) and ib(row["rs_C"])
                                             and (p.get("phi_sup_blind_tk") or 0) >= NEC)
                    if arm == "base":
                        r95 = rows95.get(key, {})
                        p95 = r95.get("prod", {})
                        d = [abs(p[k_] - p95[k_]) for k_ in CMP if p.get(k_) is not None and p95.get(k_) is not None]
                        d += [abs(row[k_] - r95[k_]) for k_ in ("rs_A", "rs_C", "rs_R")
                              if row.get(k_) is not None and r95.get(k_) is not None]
                        row["max_diff_vs_095"] = max(d) if d else None
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                ENG.reset(mod)
                SUPPORT.clear()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            row["t_total"] = round(time.time() - tk0, 1)
            row["run_py_sha256"] = inv["run_py_sha256"]
            fh.write(json.dumps(R100.clean(row), allow_nan=False) + "\n"); fh.flush()
            if "error" not in row:
                done.add((key, arm))
                durs.setdefault((arm, band), []).append(time.time() - tk0)
            g = lambda k_, d_=row: "%.2f" % d_[k_] if isinstance(d_.get(k_), (int, float)) else "-"  # noqa: E731
            pr = row.get("prod", {})
            print("[%s] %-8s %2d/%d %-16s n %5s | Z %s A %s C %s nec %s prod %s pre %s | rsA %s rsC %s rsR %s rs %s | "
                  "fit %ss score %ss rs %ss%s %s" % (
                      time.strftime("%H:%M:%S"), arm, ti + 1, len(todo), key, row.get("n"), g("free0_tk", pr),
                      g("freeM_topk_tk", pr), g("freeN_topk_tk", pr), g("phi_sup_blind_tk", pr), row.get("pass_prod"),
                      row.get("pass_pre"), g("rs_A"), g("rs_C"), g("rs_R"), row.get("pass_rs"), row.get("t_fit"),
                      row.get("t_score"), row.get("t_resample"),
                      (" d095 %.2g" % row["max_diff_vs_095"]) if row.get("max_diff_vs_095") is not None else "",
                      row.get("error", "")), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arm_rows = {r["seed"]: r for r in load_jsonl(rows_path) if r.get("arm") == arm and "error" not in r}
        n_ok = len(arm_rows)
        print("102: ARM %s rows %d: prod %d, rs %d, pre %d (floor %.2f) [%.0fs elapsed]" % (
            arm, n_ok, sum(bool(r.get("pass_prod")) for r in arm_rows.values()),
            sum(bool(r.get("pass_rs")) for r in arm_rows.values()), sum(bool(r.get("pass_pre")) for r in arm_rows.values()),
            FLOOR, time.time() - t0), flush=True)
        if stop:
            break
    print("102: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
