"""100: causal-scrubbing (donor-background) frontier on 099's frozen 18-target panel.

Development experiments that decide which donor-background stages go into the 8xH100 pod run. Small hand-picked
panel (099's availability-based panel, 9 development + 9 check), clean audited records (099 out/records/ctx_v2, no
rebuild). Nothing in 099 or 095 is written; everything goes to --out inside this experiment.

Per (target, arm), arm = "<method>_<lambda>":
  wcm_<lam>   standard v2 WCM (production settings, 400 steps) at sparsity penalty lam. wcm_1e-3 = production: 099's
              fitted circuit is reused (its manifest source hashes match today's tree; see README), all others fit here.
  v2cr_<lam>  095's v2CR arm: Z + A + C + R_C, R weight 0.25, donors = the target's 48 training contrast contexts,
              redrawn each forward, sparsity-preserving donor fill, no silent term (095 rs2_engine + 091 setup_terms,
              imported, not copied).
  Each arm also gets an equal-size ACTIVE RANDOM control: the arm's per-site member counts, drawn from latents active
  (post-Top-K) on the 48 training contexts, excluding the arm's members; its coefficients are fitted with the arm's
  own objective under 062 random_null's support restriction (support = the random members, lambda 0, gates open,
  theta_init 40, binarize none), so random vs WCM differs in membership, not gain capacity.

Scoring, identical code for every circuit:
  production   driver.Runner.score on held-out strongest: post-Top-K rule pass and protocol 12's pre-activation pass
               (095 definitions); also the circuit at alpha = 1.
  donor test   EXACTLY 099 run_baseline's whole-donor background test: bases = held-out activating + held-out silent;
               A = training activating, C = held-out silent, R = 099's filtered random pool; 8 paired draws per pool
               from the same torch generator sequence as 099 (so donors are identical to 099's); 073 ResamplePatcher
               (live SAE error, Top-K fill budget preserved). Variants: fit (live, fitted gains), a1 (live, gains 1),
               pinned (members at clean natural x fitted gains), rand / rand_pinned (the random control), empty.
  self donor   bases as their own donors: empty / a1 (identities) and fit / rand (own-background gain overshoot).
Pooled recovery (099 final_report): (mean_draws(out) - mean_draws(empty)) / (natural - mean_draws(empty)), None when
the denominator <= 1e-9. Donor pass (099): production free0_tk in [0.8, 1.5], necessity >= 0.9, pooled A and C
recovery each in [0.8, 1.5].

Resumable per (target, arm): circuits are fitted to a temp path and renamed after their fits.jsonl row; a (target,
arm) row (rows.jsonl + targets/<key>.<arm>.json) is written only when its fit, random control and every score are
complete. A kill leaves no partial row.

  wsl (repo root): PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
      experiments/100-scrub-frontier/run.py --targets <file> --arms wcm_1e-3,wcm_2.5e-4 --out <dir> \
      [--start-by 2026-10-31T00:30 --end-by 2026-10-31T01:00] [--steps 400] [--check-refit]
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
E99 = EXP / "099-recursive-causal-scrubbing"
E95 = EXP / "095-v2-resample-appendix"
E91 = EXP / "091-v2-resample-small"
RECORDS = E99 / "out" / "records" / "ctx_v2"
CIRC99 = E99 / "out" / "panel18" / "circuits"
TRAIN99 = E99 / "out" / "panel18" / "train.jsonl"
MANIFEST99 = E99 / "out" / "panel18" / "manifest.json"
POOL99 = E99 / "out" / "panel18" / "random_pool.pt"
SEED = 20261028                      # 099's random_seed (donor draws)
DRAWS = 8                            # 099's draws per background
GAMMA = 0.25
W_R = 0.25                           # 095 v2CR R weight
PROD_LAM = 1e-3
BACKGROUNDS = ("A", "C", "R")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def clean(v):
    """NaN / inf -> None, recursively (rows are strict JSON)."""
    if isinstance(v, float):
        return v if v == v and abs(v) != float("inf") else None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_jsonl(path):
    out = []
    if path.exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001  (truncated last line after a kill)
                continue
    return out


def parse_arm(arm):
    method, lam = arm.split("_", 1)
    method_spec(method)
    return method, float(lam)


def method_spec(method):
    """E4 arm grammar (added 30 Oct, Daniel's E4 follow-up): <base>[-<mod>...], base in wcm | v2cr | rsAC.
    mods: w<R weight> | mixCR | mixCAR | k<donors per row per step> | cap<alpha cap> | s<steps>.
    wcm and v2cr without mods are exactly the E1 / E2 arms."""
    parts = method.split("-")
    base, mods = parts[0], parts[1:]
    if base not in ("wcm", "v2cr", "rsAC"):
        raise ValueError(method)
    spec = dict(base=base, w_R=W_R, mix="C", k=1, cap=None, steps=None, e4=bool(mods) or base == "rsAC")
    for m in mods:
        if m.startswith("mix"):
            spec["mix"] = m[3:]
            if spec["mix"] not in ("CR", "CAR"):
                raise ValueError(method)
        elif m.startswith("w"):
            spec["w_R"] = float(m[1:])
        elif m.startswith("k"):
            spec["k"] = int(m[1:])
        elif m.startswith("cap"):
            spec["cap"] = float(m[3:])
        elif m.startswith("s"):
            spec["steps"] = int(m[1:])
        else:
            raise ValueError(method)
    if base == "wcm" and mods:
        raise ValueError("wcm takes no E4 modifiers: " + method)
    return spec


def ib(v):
    return v is not None and v == v and 0.8 <= v <= 1.5


# ---- 095 run.py definitions (copied, attribution: experiments/095-v2-resample-appendix/run.py)
def rule_pass(p):
    nec = p.get("phi_sup_blind_tk")
    return bool(ib(p.get("free0_tk")) and ib(p.get("freeM_topk_tk")) and ib(p.get("freeN_topk_tk"))
                and nec is not None and nec >= 0.9)


def pre_reads(p):
    ap, x, xr = p.get("a_pos_pre"), p.get("free0_raw_pre"), p.get("phi_sup_blind_raw_pre")
    if ap is None or x is None or xr is None or ap <= 1e-9:
        return None, None
    return x / ap, (ap - xr) / ap


def pre_pass(p):
    z, nec = pre_reads(p)
    return bool(ib(z) and ib(p.get("freeM_topk_pre")) and ib(p.get("freeN_topk_pre")) and nec is not None
                and nec >= 0.9)


def scalars(d):
    return {k: v for k, v in d.items() if v is None or isinstance(v, (bool, int, float, str))}


# ---- 099 final_report definitions (copied, attribution: experiments/099-recursive-causal-scrubbing/final_report.py)
def pooled(xs, es, natural):
    """(mean(out) - mean(empty)) / (natural - mean(empty)) over draws; None when undefined."""
    if not xs or not es:
        return None
    x, e = sum(xs) / len(xs), sum(es) / len(es)
    return (x - e) / (natural - e) if natural - e > 1e-9 else None


def donor_pass(p, rec_by_bg):
    return bool(ib(p.get("free0_tk")) and (p.get("phi_sup_blind_tk") or 0) >= 0.9
                and ib(rec_by_bg.get("A")) and ib(rec_by_bg.get("C")))


def members_by_site(c):
    out = defaultdict(list)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))].append(int(f.index))
    return dict(out)


def site_name(s):
    return "%d.%s" % s


def circuit_sig(c):
    out = []
    for n in c.nodes.values():
        if n.metadata.get("role") == "seed":
            continue
        f = n.metadata["feature_id"]
        out.append((int(f.layer), str(f.kind), int(f.index), float(n.metadata.get("amplitude", 1.0))))
    return sorted(out)


def alpha_stats(c):
    import numpy as np
    a = np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"])
    if a.size == 0:
        return dict(alpha_med=None, alpha_p90=None, alpha_max=None)
    return dict(alpha_med=float(np.median(a)), alpha_p90=float(np.percentile(a, 90)), alpha_max=float(a.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, required=True)
    ap.add_argument("--arms", required=True, help="comma list, e.g. wcm_1e-3,wcm_2.5e-4,v2cr_1e-3")
    ap.add_argument("--out", type=Path, default=HERE / "out" / "main")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--start-by", default=None, help="ISO datetime; no target starts after it")
    ap.add_argument("--end-by", default=None, help="ISO datetime; no target starts if its projected end passes it")
    ap.add_argument("--check-refit", action="store_true",
                    help="refit the production arm for the listed targets and compare with 099's circuits, then exit")
    ap.add_argument("--no-reuse-099", action="store_true")
    ap.add_argument("--engine", choices=("rs2", "e4"), default="rs2",
                    help="rs2 = 095's engine (E1-E3); e4 = rs2 + the E4 switches (e4_engine.py), needed by E4 arms")
    ap.add_argument("--no-rand", action="store_true", help="skip the random control (not used unless time is short)")
    ap.add_argument("--check-arm", default=None,
                    help="with --check-refit: refit this arm instead of wcm_1e-3 and compare with --check-ref/<arm>/<key>.pt")
    ap.add_argument("--check-ref", type=Path, default=None)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        parse_arm(a)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(OUT=str(out / "driver"), CTX_DIR=str(RECORDS), WCM_VERSION="v2", EVAL_SPLIT="explicit",
                      CURVE_STRIDE="20", STEPS=str(args.steps))
    sys.path[:0] = [str(ROOT / "src"), str(E95), str(EXP / "073-strict-eval")]
    import torch
    import rs2_engine  # 095: the v2CR engine patch (in memory; src untouched)
    spec = importlib.util.spec_from_file_location("run091", str(E91 / "run.py"))
    R91 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R91)       # 091 helpers: setup_terms (R term donors), donor_tokens
    SE, driver = R91.SE, R91.driver
    assert driver.STEPS == args.steps

    # ---- provenance / fixed-config manifest (a changed fixed config refuses to resume into this out dir)
    keys = [k for k in args.targets.read_text().split() if k]
    m99 = json.loads(MANIFEST99.read_text())
    fixed = dict(protocol="100-scrub-frontier/v1", steps=args.steps, draws=DRAWS, seed=SEED, gamma=GAMMA, w_R=W_R,
                 records=str(RECORDS.relative_to(ROOT)), random_pool_sha256=file_hash(POOL99),
                 donor_test="099 run_baseline whole-donor background; live SAE error; Top-K fill budget",
                 random_control="matched per-site counts, active on training contexts, excluding members, fitted gains")
    mpath = out / "manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != fixed:
        raise RuntimeError("fixed configuration changed; use a fresh --out")
    write_json(mpath, fixed)
    src_paths = sorted(set(ROOT.joinpath("src").rglob("*.py")) | set(HERE.glob("*.py")) | set(E95.glob("*.py"))
                       | {E91 / "run.py", EXP / "062-h100-protocol-v1/driver.py", EXP / "073-strict-eval/strict_eval.py",
                          ROOT / "config.yaml"})
    src_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): file_hash(p) for p in src_paths}
    changed_vs_099 = sorted(k for k, v in m99["source"].items()
                            if not (ROOT / k).exists() or file_hash(ROOT / k) != v)
    for k in keys:
        if k in m99["contexts"] and file_hash(RECORDS / (k + ".pt")) != m99["contexts"][k]:
            raise RuntimeError("record changed since 099: " + k)
    inv = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), argv=sys.argv, targets=keys, arms=arms,
               python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
               changed_vs_099_manifest=changed_vs_099, run_py_sha256=file_hash(Path(__file__)))
    with open(out / "invocations.jsonl", "a") as fh:
        fh.write(json.dumps(inv) + "\n")
    (out / "sources").mkdir(exist_ok=True)
    write_json(out / "sources" / ("%s.json" % time.strftime("%Y%m%d_%H%M%S")), src_hashes)
    reuse_ok = (not args.no_reuse_099) and args.steps == m99["steps"] and changed_vs_099 ==["experiments/099-recursive-causal-scrubbing/run_panel.py"]
    print("100: reuse 099 production circuits:", reuse_ok, "| files changed vs 099 manifest:", changed_vs_099, flush=True)

    # ---- engine: 095 rs2 patch (or 100's e4 = rs2 + default-off E4 switches), then 062 random_null's support wrapper
    if args.engine == "e4":
        sys.path.insert(0, str(HERE))
        import e4_engine as ENG
    else:
        ENG = rs2_engine
    mod = ENG.install(tag="100" + args.engine)
    SUPPORT = {}
    orig_run = mod.run_learned_mask

    def wrapped(inference, bank, **kw):
        if SUPPORT:
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in SUPPORT.items()},
                      l1_lambda=0.0, binarize="none", theta_init=40.0)
        return orig_run(inference, bank, **kw)
    mod.run_learned_mask = wrapped

    R = driver.Runner()
    assert R.wcm_version == "v2"
    from config import config
    lm = config.discovery.learned_mask
    G, bank, dev = R.G, R.G["bank"], R.G["device"]
    inf = G["inference"]
    from circuit.protocol_contexts import eval_probe, training_set
    from eval.ablation_faithfulness import upstream_sites
    from eval.wcm_eval import NaturalTap, circuit_alphas, contrast_split, site_encoder
    train99 = {r["seed"]: r for r in load_jsonl(TRAIN99)}

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    pool_eval = torch.load(POOL99, weights_only=False)["tokens"].cpu()
    E4POOL = HERE / "out" / "e4_train_random_pool.pt"

    def train_random_pool():
        """E4 mixed donors: 48 random-corpus TRAINING contexts, disjoint from the evaluation random pool and from every
        panel record row (099's exact / near rule), drawn with 073's random_pool procedure under a new seed. Built
        once, saved, then reused (atomic write)."""
        if E4POOL.exists():
            return torch.load(E4POOL, weights_only=False)["tokens"].cpu()
        import random as _r
        from circuit import heldout_rule as HR
        from circuit.protocol_contexts import _load
        loader = G["M0"].probe_builder.loader
        ranges = [(a, b_) for a, b_ in loader.shard_id_ranges if a >= 0]
        rng_ = _r.Random(20261030)
        ids = []
        while len(ids) < 160:
            a, b_ = rng_.choice(ranges)
            ids.append(rng_.randint(a, b_))
        toks, _, got = _load(loader, ids)
        toks = toks.cpu()
        panel_keys = json.loads((E99 / "panel.json").read_text())["targets"]
        rows_ = []
        for k_ in panel_keys:
            rr = torch.load(RECORDS / (k_ + ".pt"), weights_only=False)[k_]
            rows_ += [d["pos"] for d in (rr["strong"], rr.get("mid")) if d is not None] + [rr["neg"]]
        ref = torch.cat(rows_ + [pool_eval]).numpy()
        keep, kept_ids = [], []
        for j, row in enumerate(toks.numpy()):
            if int((toks[j] != 0).sum()) < 60:
                continue
            ex, _, lcr, c8 = HR.pair_stats(row[None], ref)
            _, _, _, c8r = HR.pair_stats(ref, row[None])
            if ex.any() or HR.near_of(lcr, c8).any() or (c8r >= 0.5).any():
                continue
            keep.append(j); kept_ids.append(got[j])
            if len(keep) == 48:
                break
        if len(keep) < 48:
            raise RuntimeError("only %d independent random training donors" % len(keep))
        tok = toks[keep]
        tmpp = E4POOL.with_name("e4_pool.%d.tmp" % os.getpid())
        E4POOL.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(tokens=tok, ids=kept_ids, seed=20261030, rule="073 random_pool draw; >=60 tokens; 099 exact/near "
                        "filter vs all panel record rows and 099's evaluation random pool"), tmpp)
        os.replace(tmpp, E4POOL)
        return tok

    def setup_e4(spec, key, rec, tr_arm):
        """Donor terms for an E4 arm. RESAMPLE = the C-side donor term (contrast-train 48, optionally mixed with
        mid-band training activating rows and random training contexts); for rsAC also E4 make2 = RA on the mid-band
        training rows. Every training donor is checked to be disjoint (exact row match) from the evaluation donors
        (A = strongest-train, C = held-out contrast, R = 099 pool) and the held-out bases."""
        from circuit.protocol_contexts import pick
        from eval.ablation_faithfulness import upstream_sites as _ups
        l_, kd_, _ = key.split(".")
        sites_ = sorted(_ups(bank, int(l_), kd_))
        contrast = R91.donor_tokens(rec).cpu()                    # asserted == the C term's contrast-train rows
        mid = pick(rec, "mid", rec["B_mid"])[0].cpu() if tr_arm == "B" else contrast[:0]
        parts = {"C": contrast}
        if spec["mix"] in ("CR", "CAR"):
            parts["Rtrain"] = train_random_pool()
        if spec["mix"] == "CAR":
            parts["Amid"] = mid
        donors = torch.cat(list(parts.values()))
        probe_ = eval_probe(rec, "strong", "cpu")
        ntr_ = int(probe_.metadata["n_train"])
        nneg_ = contrast_split(len(probe_.neg_tokens), probe_.metadata)
        evals = dict(A=probe_.pos_tokens[:ntr_], C=probe_.neg_tokens[nneg_:], R=pool_eval,
                     bases=probe_.pos_tokens[ntr_:])

        def overlap(x, y):
            if x.shape[1] != y.shape[1]:
                return 0
            return int((x[:, None, :] == y[None, :, :]).all(-1).any(-1).sum())
        train_sets = dict(parts, **({"RA_mid": mid} if spec["base"] == "rsAC" else {}))
        ov = {"%s|%s" % (a, b_): overlap(ta, tb) for a, ta in train_sets.items() for b_, tb in evals.items()}
        if any(ov.values()):
            raise RuntimeError("training donors overlap evaluation donors: %s" % ov)
        dev_ = G["device"]
        codes = SE.capture_codes(inf, bank, donors.to(dev_), sites_)
        codes = {s: (a.cpu(), i.cpu()) for s, (a, i) in codes.items()}
        floors = R91.rs_engine.DonorFloors(codes, bank.d_sae, dev_, seed=R91.TRAIN_POOL_SEED)
        mod.RESAMPLE["make"] = R91.rs_engine.make_factory(mod, floors)
        mod.RESAMPLE["weight"] = float(spec["w_R"])
        mod.E4.update(k=int(spec["k"]), amp_cap=spec["cap"])
        info = dict(parts={k_: int(v.shape[0]) for k_, v in parts.items()}, overlap_exact=ov)
        if spec["base"] == "rsAC":
            if len(mid) == 0:
                raise RuntimeError("rsAC needs mid-band training rows (thin target)")
            mod.E4.update(drop_pos=True)
            codes2 = SE.capture_codes(inf, bank, mid.to(dev_), sites_)
            codes2 = {s: (a.cpu(), i.cpu()) for s, (a, i) in codes2.items()}
            floors2 = R91.rs_engine.DonorFloors(codes2, bank.d_sae, dev_, seed=R91.TRAIN_POOL_SEED + 1)
            mod.E4["make2"] = R91.rs_engine.make_factory(mod, floors2)
            mod.E4["w2"] = GAMMA                                   # RA replaces A at A's weight; RC = RESAMPLE at w_R
            info["RA_mid"] = int(mid.shape[0])
        return floors, int(donors.shape[0]), info

    def fit_one(method, lam, key, rec, tr_arm, final, kind, ftr, ff, support=None):
        """Fit (cached at final via tmp + rename after its fits.jsonl row). Returns (circuit, fit row)."""
        fits = {(r["seed"], r["arm"], r["kind"]): r for r in load_jsonl(out / "fits.jsonl")}
        arm = final.parent.name                                # "<arm>" or "<arm>_rand"
        k3 = (key, arm, kind)
        if final.exists() and k3 in fits:
            return torch.load(final, weights_only=False), fits[k3]
        ENG.reset(mod)
        n_don = None
        floors = None
        spec = method_spec(method)
        e4info = {}
        if spec["e4"] and args.engine != "e4":
            raise RuntimeError("arm %s needs --engine e4" % method)
        if spec["base"] == "v2cr" and not spec["e4"]:
            mod.SPEED.update(drop_floor=False, sp_donor=True)
            floors, _sfn, n_don = R91.setup_terms(mod, R, rec, key, W_R, 0.0)
        elif spec["base"] in ("v2cr", "rsAC"):
            mod.SPEED.update(drop_floor=spec["base"] == "rsAC", sp_donor=True)
            floors, n_don, e4info = setup_e4(spec, key, rec, tr_arm)
        M = R.method(GAMMA, lam, True, steps=spec["steps"] or args.steps)   # built right before the fit (global config)
        assert (lm.train_fill, lm.sparse_grad, lm.train_means_source) == ("sparse", "ste", "eval_ref")
        if torch.cuda.is_available():
            sync()
            torch.cuda.reset_peak_memory_stats()
        tmp = out / "_tmp" / ("%d_%s" % (os.getpid(), time.strftime("%H%M%S"))) / ("%s.%s.pt" % (key, arm))
        SUPPORT.clear()
        if support is not None:
            SUPPORT.update(support)
        try:
            c, t_fit = R.fit(M, rec, key, tr_arm, tmp, train_fh=ftr,
                             label=dict(exp="100", arm=arm, kind=kind, lam=lam, method=method))
        finally:
            SUPPORT.clear()
        prov = M.last_mask_provenance or {}
        if support is not None and prov.get("support_n") is None:
            raise RuntimeError("support wrapper did not reach the engine")
        f = dict(seed=key, arm=arm, kind=kind, lam=lam, method=method, steps=spec["steps"] or args.steps,
                 t_fit=round(t_fit, 2), engine=args.engine, spec={k_: v_ for k_, v_ in spec.items()}, e4=e4info,
                 fit_peak_mb=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None),
                 n=driver.n_nodes(c) if c is not None else None, rejected=c is None, n_donors=n_don,
                 donor_draws=getattr(floors, "n_draws", None) if floors is not None else None,
                 donor_sp_calls=mod.DONOR_STATS["calls"], train_fill=prov.get("train_fill"),
                 train_means_source=prov.get("train_means_source"), support_n=prov.get("support_n"),
                 term_names=list(((prov.get("curve") or {}).get("terms") or {}).keys()),
                 source="fit100")
        ENG.reset(mod)
        ff.write(json.dumps(f) + "\n")
        ff.flush()
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, final)
        return c, f

    # ---- check mode: refit production for the listed targets, compare with 099's circuits
    if args.check_refit:
        res = []
        with open(out / "train_check.jsonl", "a") as ftr, open(out / "fits_check.jsonl", "a") as ff:
            for key in keys:
                rec = R.contexts(key)
                tr_arm, _ = R.train_arm(rec)
                if args.check_arm:                                    # E4: refit an arm through fit_one, compare
                    carm = args.check_arm
                    final = out / "check_refit" / carm / ("%s.pt" % key)
                    if final.exists():
                        print("check exists", final, flush=True)
                        continue
                    c, finfo = fit_one(parse_arm(carm)[0], parse_arm(carm)[1], key, rec, tr_arm, final, "circuit", ftr, ff)
                    t = finfo["t_fit"]
                    ref = torch.load(args.check_ref / carm / ("%s.pt" % key), weights_only=False)
                    a, b = circuit_sig(c), circuit_sig(ref)
                    ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
                    r = dict(seed=key, arm=carm, engine=args.engine, identical=a == b, n=len(a), n_ref=len(b),
                             jaccard=len(ka & kb) / max(len(ka | kb), 1), t_fit=t,
                             max_alpha_diff=(max(abs(x[3] - y[3]) for x, y in zip(a, b)) if ka == kb else None))
                    print("CHECK", json.dumps(r), flush=True)
                    ff.write(json.dumps(r) + "\n")
                    ff.flush()
                    continue
                final = out / "check_refit" / "wcm_1e-3" / ("%s.pt" % key)
                if final.exists():
                    print("check exists", final, flush=True)
                    continue
                ENG.reset(mod)
                M = R.method(GAMMA, PROD_LAM, True, steps=args.steps)
                sync()
                tmp = out / "_tmp" / ("check_%d" % os.getpid()) / ("%s.pt" % key)
                c, t = R.fit(M, rec, key, tr_arm, tmp, train_fh=ftr, label=dict(exp="100", check="refit"))
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp, final)
                ref = torch.load(CIRC99 / ("%s.wcm.pt" % key), weights_only=False)
                a, b = circuit_sig(c), circuit_sig(ref)
                ka, kb = {x[:3] for x in a}, {x[:3] for x in b}
                r = dict(seed=key, identical=a == b, n=len(a), n_099=len(b),
                         jaccard=len(ka & kb) / max(len(ka | kb), 1), t_fit=round(t, 2),
                         t_fit_099=train99.get(key, {}).get("t_fit"),
                         max_alpha_diff=(max(abs(x[3] - y[3]) for x, y in zip(a, b)) if ka == kb else None))
                print("CHECK", json.dumps(r), flush=True)
                res.append(r)
                ff.write(json.dumps(r) + "\n")
                ff.flush()
        return

    pool = torch.load(POOL99, weights_only=False)["tokens"].cpu()
    rows_path = out / "rows.jsonl"
    done = {(r["seed"], r["arm"]) for r in load_jsonl(rows_path) if "error" not in r}
    start_by = dt.datetime.fromisoformat(args.start_by) if args.start_by else None
    end_by = dt.datetime.fromisoformat(args.end_by) if args.end_by else None
    per_arm_secs = []
    t_run = time.time()
    split = {}
    panel = json.loads((E99 / "panel.json").read_text())
    for k in panel["development"]:
        split[k] = "dev"
    for k in panel["check"]:
        split[k] = "check"
    todo_keys = [k for k in keys if any((k, a) not in done for a in arms)]
    print("100: %d targets (%d with work), arms %s, out %s" % (len(keys), len(todo_keys), arms, out), flush=True)
    fh = open(rows_path, "a")
    ftr = open(out / "train.jsonl", "a")
    ff = open(out / "fits.jsonl", "a")
    for ti, key in enumerate(todo_keys):
        todo = [a for a in arms if (key, a) not in done]
        now = dt.datetime.now()
        if start_by and now > start_by:
            print("100: STOP (start-by %s passed) before %s" % (start_by, key), flush=True)
            break
        if end_by and per_arm_secs:
            est = sum(per_arm_secs) / len(per_arm_secs) * len(todo)
            if now + dt.timedelta(seconds=est) > end_by:
                print("100: STOP before %s: ~%.0fs needed, end-by %s" % (key, est, end_by), flush=True)
                break
        t_key = time.time()
        layer, kind, idx = key.split(".")
        layer, idx = int(layer), int(idx)
        site = (layer, kind)
        rec = R.contexts(key)
        tr_arm, thin = R.train_arm(rec)
        W, b = site_encoder(bank, layer, kind)
        sites = sorted(upstream_sites(bank, layer, kind))
        train_tokens = training_set(rec, tr_arm)[0].to(dev)

        # ---- fits (circuit + random control) per arm
        live = None
        arms_c = {}
        try:
            for arm in todo:
                method, lam = parse_arm(arm)
                if method == "wcm" and abs(lam - PROD_LAM) < 1e-15 and reuse_ok and (CIRC99 / ("%s.wcm.pt" % key)).exists():
                    c = torch.load(CIRC99 / ("%s.wcm.pt" % key), weights_only=False)
                    t99 = train99.get(key, {})
                    finfo = dict(seed=key, arm=arm, kind="circuit", lam=lam, method=method, source="099",
                                 t_fit=t99.get("t_fit"), n=driver.n_nodes(c), path=str((CIRC99 / ("%s.wcm.pt" % key)).relative_to(ROOT)),
                                 sha256=file_hash(CIRC99 / ("%s.wcm.pt" % key)))
                else:
                    c, finfo = fit_one(method, lam, key, rec, tr_arm, out / "circuits" / arm / ("%s.pt" % key),
                                       "circuit", ftr, ff)
                if c is None:
                    row = dict(seed=key, arm=arm, split=split.get(key), skip="rejected", fit=finfo)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    continue
                mem = members_by_site(c)
                if live is None:
                    live = {}
                    k2i = {k: n for n, k in enumerate(G["KINDS"])}
                    lv = {s: set() for s in sites}

                    def hook(layer_idx, acts):
                        for kd in G["KINDS"]:
                            s = (layer_idx, kd)
                            if s in lv:
                                ta, tix = bank.encode(acts[k2i[kd]], kd, layer_idx)
                                lv[s].update(tix.reshape(-1)[ta.reshape(-1) > 0].long().cpu().tolist())
                    inf.disable_compile()
                    try:
                        with torch.no_grad():
                            for s0 in range(0, int(train_tokens.shape[0]), 16):
                                inf.forward(train_tokens[s0:s0 + 16], activations_callback=hook,
                                            return_activations=False, tokenize_final=False)
                    finally:
                        inf.enable_compile()
                    live = lv
                rng = random.Random("%s|%s|rand100" % (key, arm))
                support, short = {}, {}
                for s in sorted(mem):
                    avail = sorted(live.get(s, set()) - set(mem[s]))
                    n_want = len(mem[s])
                    support[s] = sorted(rng.sample(avail, min(n_want, len(avail))))
                    if len(avail) < n_want:
                        short[site_name(s)] = [n_want, len(avail)]
                support = {s: v for s, v in support.items() if v}
                if args.no_rand:
                    rc, rinfo = None, None
                else:
                    if not support:
                        raise RuntimeError("no active non-member latents for the random control")
                    rc, rinfo = fit_one(method, lam, key, rec, tr_arm,
                                        out / "circuits" / (arm + "_rand") / ("%s.pt" % key), "random", ftr, ff,
                                        support=support)
                arms_c[arm] = dict(c=c, finfo=finfo, rc=rc, rinfo=rinfo, short=short, mem=mem,
                                   n_live={site_name(s): len(v) for s, v in live.items() if s in mem})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print("100: FIT ERROR %s: %s" % (key, e), flush=True)
            ENG.reset(mod)
            SUPPORT.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if not arms_c:
            continue

        # ---- production scores
        prod = {}
        try:
            for arm, d in arms_c.items():
                sync(); t0 = time.time()
                p = R.score(d["c"], rec, "strong")
                sync(); t_p = time.time() - t0
                ones = {s: {i: 1.0 for i in v} for s, v in d["mem"].items()}
                p1 = R.score(d["c"], rec, "strong", alphas=ones)
                pr = R.score(d["rc"], rec, "strong") if d["rc"] is not None else None
                prod[arm] = dict(fit=p, a1=p1, rand=pr, t_prod=t_p)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print("100: PROD ERROR %s: %s" % (key, e), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ---- donor test (099 run_baseline, extended with arms / variants)
        try:
            sync(); t_sh = time.time()

            def taps(tokens):
                return SE.run_full(inf, lambda a, z: NaturalTap(bank, site, W[idx], b[idx], idx), tokens.to(dev), pre=True)
            probe = eval_probe(rec, "strong", dev)
            ntr = int(probe.metadata["n_train"])
            nneg = contrast_split(len(probe.neg_tokens), probe.metadata)
            base, anchors = probe.pos_tokens[ntr:], probe.pos_argmax[ntr:]
            silent = probe.neg_tokens[nneg:]
            nb = len(base)
            bases = torch.cat([base, silent])
            anchors = torch.cat([anchors, torch.zeros(len(silent), dtype=anchors.dtype, device=dev)])
            natural, natural_pre = taps(bases)
            natural_anchors = SE.at(natural[:nb], anchors[:nb])
            natural_pre_anchors = SE.at(natural_pre[:nb], anchors[:nb])
            nat_mean = float(natural_anchors.mean())
            base_codes = SE.capture_codes(inf, bank, bases, sites)
            specs = {}       # (arm, variant) -> (kt, sc, pins or None)

            def kt_sc(c):
                al = circuit_alphas(c)
                kt = {s: torch.tensor(sorted(v), device=dev) for s, v in al.items() if v}
                sc = {s: torch.tensor([al[s][int(i)] for i in kt[s].tolist()], device=dev) for s in kt}
                return kt, sc

            from sae.dense import sparse_topk_to_dense
            D = int(bank.d_sae)

            class LazyPins:
                """099's pins (members at their clean natural post-Top-K value on the base context), built per batch
                and site on demand: identical values (Top-K indices are unique), memory O(batch x T x d_sae)."""

                def __init__(self, kt, a=0, z=None):
                    self.kt, self.a, self.z = kt, a, z

                def slice(self, a, z):
                    return LazyPins(self.kt, a, z)

                def __getitem__(self, s):
                    acts, ind = base_codes[s]
                    dense = sparse_topk_to_dense(acts[self.a:self.z], ind[self.a:self.z], D, dtype=torch.float32)
                    return dense[:, :, self.kt[s]]

            def pins_for(kt):
                return LazyPins(kt)
            for arm, d in arms_c.items():
                kt, sc = kt_sc(d["c"])
                specs[(arm, "fit")] = (kt, sc, None)
                specs[(arm, "a1")] = (kt, {}, None)
                specs[(arm, "pinned")] = (kt, sc, pins_for(kt))
                if d["rc"] is not None:
                    rkt, rsc = kt_sc(d["rc"])
                    specs[(arm, "rand")] = (rkt, rsc, None)
                    specs[(arm, "rand_pinned")] = (rkt, rsc, pins_for(rkt))

            def execute(donors, kt, sc, pins=None):
                return SE.run_full(inf, lambda a, z: SE.ResamplePatcher(
                    bank, set(sites), site, W[idx], b[idx], idx, kt, sc,
                    {s: (v[a:z], i[a:z]) for s, (v, i) in donors.items()}, bank.k,
                    pins=pins.slice(a, z) if pins is not None else None), bases, pre=True)

            def summarise(out_, pre_):
                at = SE.at(out_[:nb], anchors[:nb])
                atp = SE.at(pre_[:nb], anchors[:nb])
                return dict(mean=float(at.mean()), raw=[round(float(v), 6) for v in at.tolist()],
                            pre_mean=float(atp.mean()), pre_raw=[round(float(v), 6) for v in atp.tolist()],
                            silent_firing=float((out_[nb:] > 0).float().mean()) if len(silent) else None,
                            fire_recall=float((at[natural_anchors > 0] > 0).float().mean()) if (natural_anchors > 0).any() else None)
            t_shared = time.time() - t_sh
            t_var = defaultdict(float)
            n_fwd = defaultdict(int)
            draws_out = defaultdict(list)     # (arm, variant) -> list of per-draw dicts; arm None = empty
            empty_draws = []
            donor_sets = {"A": probe.pos_tokens[:ntr], "C": silent, "R": pool.to(dev)}
            rng = torch.Generator().manual_seed(SEED + layer * 1000 + idx)
            for ds, tokens in donor_sets.items():
                sync(); t0 = time.time()
                codes = SE.capture_codes(inf, bank, tokens, sites)
                donor_nat, _ = taps(tokens)        # [n_pool, T] target read on the donor contexts themselves
                sync(); t_shared += time.time() - t0
                for draw in range(DRAWS):
                    selection = torch.cat([torch.randperm(len(tokens), generator=rng)[:nb],
                                           torch.randperm(len(tokens), generator=rng)[:len(silent)]]).to(dev)
                    if len(selection) != len(bases):
                        raise RuntimeError("donor set too small")
                    donor = {s: (v[selection], i[selection]) for s, (v, i) in codes.items()}
                    sel_b = selection[:nb]
                    d_at = donor_nat[sel_b, anchors[:nb].clamp(0, donor_nat.shape[1] - 1)]
                    d_any = (donor_nat[sel_b] > 0).any(-1)
                    sync(); t0 = time.time()
                    o, pr_ = execute(donor, {}, {})
                    e = summarise(o, pr_)
                    sync(); t_shared += time.time() - t0
                    e.update(background=ds, draw=draw, donor_indices=selection.cpu().tolist(),
                             donor_fires_at_anchor=[bool(v) for v in (d_at > 0).tolist()],
                             donor_fires_anywhere=[bool(v) for v in d_any.tolist()],
                             donor_act_at_anchor=[round(float(v), 6) for v in d_at.tolist()])
                    empty_draws.append(e)
                    for (arm, var), (kt, sc, pins) in specs.items():
                        sync(); t0 = time.time()
                        o, pr_ = execute(donor, kt, sc, pins)
                        r = summarise(o, pr_)
                        sync(); t_var[(arm, var)] += time.time() - t0
                        n_fwd[(arm, var)] += 1
                        r.update(background=ds, draw=draw)
                        draws_out[(arm, var)].append(r)
                    del donor
                del codes, donor_nat
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # self donors (identity / own-background overshoot)
            selfd = {}
            sync(); t0 = time.time()
            o, _ = execute(base_codes, {}, {})
            selfd["empty"] = dict(max_abs=float((o - natural).abs().max()), mean=float(SE.at(o[:nb], anchors[:nb]).mean()))
            t_shared += time.time() - t0
            for (arm, var), (kt, sc, pins) in specs.items():
                if var not in ("fit", "a1", "rand"):
                    continue
                sync(); t0 = time.time()
                o, _ = execute(base_codes, kt, sc)
                sync(); t_var[(arm, var + "_self")] += time.time() - t0
                selfd.setdefault(arm, {})[var] = dict(max_abs=float((o - natural).abs().max()),
                                                      mean=float(SE.at(o[:nb], anchors[:nb]).mean()))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print("100: DONOR ERROR %s: %s" % (key, e), flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # ---- rows, one per arm (raw json first, then the summary row)
        e_by_bg = {bg: [d["mean"] for d in empty_draws if d["background"] == bg] for bg in BACKGROUNDS}
        n_arms = len(arms_c)
        for arm, d in arms_c.items():
            p, p1, pr = prod[arm]["fit"], prod[arm]["a1"], prod[arm]["rand"]
            rec_v = {}
            for var in ("fit", "a1", "pinned", "rand", "rand_pinned"):
                if (arm, var) not in draws_out:
                    continue
                rec_v[var] = {bg: pooled([x["mean"] for x in draws_out[(arm, var)] if x["background"] == bg],
                                         e_by_bg[bg], nat_mean) for bg in BACKGROUNDS}
            per_site = {site_name(s): len(v) for s, v in sorted(d["mem"].items())}
            t_donor_fit = sum(t_var[(arm, v)] for v in ("fit",))
            row = dict(seed=key, arm=arm, split=split.get(key), layer=layer, kind=kind, train_arm=tr_arm, thin=thin,
                       n=driver.n_nodes(d["c"]), per_site=per_site, n_rand=driver.n_nodes(d["rc"]) if d["rc"] else None,
                       rand_short=d["short"], n_live=d["n_live"], fit=d["finfo"], fit_rand=d["rinfo"],
                       alpha=alpha_stats(d["c"]), alpha_rand=alpha_stats(d["rc"]) if d["rc"] else None,
                       prod=scalars(p), prod_a1=scalars(p1), prod_rand=scalars(pr) if pr else None,
                       pass_rule=rule_pass(p), pass_pre=pre_pass(p), z_pre0=pre_reads(p)[0], nec_pre0=pre_reads(p)[1],
                       pass_rule_a1=rule_pass(p1), pass_pre_a1=pre_pass(p1),
                       pass_rule_rand=rule_pass(pr) if pr else None, pass_pre_rand=pre_pass(pr) if pr else None,
                       natural_mean=nat_mean, natural_pre_mean=float(natural_pre_anchors.mean()), n_base=nb,
                       n_silent=len(silent), recovery=rec_v,
                       empty_leak={bg: (sum(e_by_bg[bg]) / len(e_by_bg[bg]) / nat_mean if nat_mean > 1e-9 else None)
                                   for bg in BACKGROUNDS},
                       donor_pass={v: donor_pass(prod[arm]["rand"] if v.startswith("rand") else p, rec_v[v])
                                   for v in rec_v if prod[arm]["rand"] is not None or not v.startswith("rand")},
                       self_donor=dict(empty=selfd["empty"], **selfd.get(arm, {})),
                       t_prod=round(prod[arm]["t_prod"], 2),
                       t_donor={v: round(t_var[(arm, v)], 2) for v in
                                ("fit", "a1", "pinned", "rand", "rand_pinned", "fit_self", "a1_self", "rand_self")},
                       n_forward={v: n_fwd[(arm, v)] for v in ("fit", "a1", "pinned", "rand", "rand_pinned")},
                       t_donor_shared_target=round(t_shared, 2), n_arms_sharing=n_arms,
                       t_donor_fit_standalone=round(t_shared + t_donor_fit, 2),
                       run_py_sha256=inv["run_py_sha256"])
            raw = dict(seed=key, arm=arm, natural_raw=[round(float(v), 6) for v in natural_anchors.tolist()],
                       natural_pre_raw=[round(float(v), 6) for v in natural_pre_anchors.tolist()],
                       anchors=anchors[:nb].cpu().tolist(), empty=empty_draws,
                       variants={v: draws_out[(arm, v)] for (a_, v) in draws_out if a_ == arm})
            write_json(out / "targets" / ("%s.%s.json" % (key, arm)), raw)
            fh.write(json.dumps(clean(row), allow_nan=False) + "\n")
            fh.flush()
            done.add((key, arm))
            g = lambda v: "-" if v is None else "%.3f" % v  # noqa: E731
            rv = rec_v.get("fit", {})
            print("[%s] %s %-14s %-10s n %5d | Z %s nec %s rule %s pre %s | rec A %s C %s R %s | pin A %s C %s | "
                  "rand A %s C %s rule %s | a1 A %s | fit %ss donor %ss" % (
                      time.strftime("%H:%M:%S"), split.get(key), key, arm, row["n"], g(p.get("free0_tk")),
                      g(p.get("phi_sup_blind_tk")), row["pass_rule"], row["pass_pre"], g(rv.get("A")), g(rv.get("C")),
                      g(rv.get("R")), g(rec_v.get("pinned", {}).get("A")), g(rec_v.get("pinned", {}).get("C")),
                      g(rec_v.get("rand", {}).get("A")), g(rec_v.get("rand", {}).get("C")), row["pass_rule_rand"],
                      g(rec_v.get("a1", {}).get("A")), d["finfo"].get("t_fit"), row["t_donor_fit_standalone"]),
                  flush=True)
        el = time.time() - t_key
        per_arm_secs.extend([el / max(len(todo), 1)] * len(todo))
        left = sum(1 for k in todo_keys[ti + 1:] for a in arms if (k, a) not in done)
        est = sum(per_arm_secs) / len(per_arm_secs) * left
        print("100: target %d/%d %s done in %.0fs; ~%.0f min left (projected end %s)" % (
            ti + 1, len(todo_keys), key, el, est / 60, time.strftime("%H:%M", time.localtime(time.time() + est))),
            flush=True)
        del base_codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    changed = [k for k, v in src_hashes.items() if not (ROOT / k).exists() or file_hash(ROOT / k) != v]
    with open(out / "source_check.jsonl", "a") as fsc:
        fsc.write(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), changed_during_run=changed)) + "\n")
    print("100: DONE %.0fs; source changed during run: %s" % (time.time() - t_run, changed), flush=True)


if __name__ == "__main__":
    main()
