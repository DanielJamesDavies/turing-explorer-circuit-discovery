"""DAN-18: how stable is circuit membership, and does the shared core work on its own?

Fit 1 is the 15k run's circuit (H100). Fit 2 refits the same target under the same protocol with one change: the
training contexts are visited in a different order (a fixed per-target permutation of the training rows). The engine
walks the training set in fixed batches of 4, so this changes only the optimisation path: the training set, the
ablation values (means are order-free), the rank-keep ranks, and every hyperparameter are unchanged. On a different
GPU than fit 1, it also carries the hardware's float noise.

Then the core = members of both fits. Its coefficients are refitted on the core alone (the random-null recipe:
support-restricted, no sparsity penalty, gates fully open; random_null.install_support_wrapper), and it is scored with
the production scorer on held-out strongest. Fit 2 is scored too, so its pass rate can be set against fit 1's.

Targets: N_PER per (layer, kind) cell from the run's fitted, non-vacuous targets, cells interleaved so a run stopped
part-way is still stratified. Resumable; one row per (target, stage).

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/stable_core.py
  env: N_PER (5)  OUT_SC (default experiments/062-h100-protocol-v1/out_stable_core)  REPORT=1 (summary; no GPU)
       SC_TARGETS (a target list used as given, e.g. a pre-drawn public list, instead of the N_PER-per-cell draw;
       targets without a main circuit are skipped)
       SHARD=i/k (keys[i::k]; rows to stable_core.shard<i>.jsonl / train.shard<i>.jsonl when k > 1)
       FAMILY / DATA / STORES / LAM / STEPS as driver.py
"""
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
os.environ.setdefault("EVAL_SPLIT", "explicit")         # identical to as_run on held-out strongest (always 48 + held)
sys.path.insert(0, str(HERE))
import driver  # noqa: E402
import random_null as RN  # noqa: E402

N_PER = int(os.environ.get("N_PER", 5))
OUT_SC = Path(os.environ.get("OUT_SC", str(HERE / "out_stable_core")))
SC_TARGETS = os.environ.get("SC_TARGETS")
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
from merge import BAND, NEC_MIN  # noqa: E402  (the DAN-8 pass rule)
_PERMUTE = {}                                            # key -> permutation of the training rows, set per fit


def install_order_wrapper():
    """Wrap circuit.protocol_contexts.training_probe (driver.Runner.fit imports it at call time): when _PERMUTE holds
    a permutation, reorder the first n_train rows (the training set) of the fit's probe; the padded held-out rows,
    used only for the reported holdout loss, stay where they are."""
    import torch
    import circuit.protocol_contexts as PC
    orig = PC.training_probe

    def wrapped(rec, device, arm=None, holdout_frac=None):
        pd_ = orig(rec, device, arm, holdout_frac)
        perm = _PERMUTE.get("perm")
        if perm is None:
            return pd_
        n_tr = int(pd_.metadata["n_train"])
        p = torch.tensor(perm, dtype=torch.long, device=pd_.pos_tokens.device)
        assert len(perm) == n_tr
        for name in ("pos_tokens", "target_tokens", "pos_argmax"):
            t = getattr(pd_, name)
            setattr(pd_, name, torch.cat([t[:n_tr][p], t[n_tr:]]))
        return pd_
    PC.training_probe = wrapped


def pick_targets():
    """N_PER fitted, non-vacuous targets per (layer, kind), fixed seed, cells interleaved (or SC_TARGETS as given)."""
    import glob
    if SC_TARGETS:
        circ = driver.OUT / "main" / "circuits"
        return [k for k in Path(SC_TARGETS).read_text().split() if (circ / ("%s.pt" % k)).exists()]
    strong = {}
    for f in sorted(glob.glob(str(driver.OUT / "main" / "eval.shard*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if "error" not in r and r.get("held") == "strong":
                strong[r["seed"]] = r
    cells = defaultdict(list)
    for k, r in strong.items():
        if not r.get("vacuous_tk"):
            l, kind, _ = k.split(".")
            cells[(int(l), kind)].append(k)
    rng = random.Random(20261009)
    picks = [rng.sample(sorted(cells[c]), min(N_PER, len(cells[c]))) for c in sorted(cells)]
    return [p[i] for i in range(N_PER) for p in picks if i < len(p)]


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def member_set(c):
    return {(s[0], s[1], i) for s, v in RN.members_by_site(c).items() for i in v}


def main():
    import torch
    OUT_SC.mkdir(parents=True, exist_ok=True)
    path = driver.shard_jsonl(OUT_SC, "stable_core")
    done = set()
    for r in driver.read_jsonl_all(OUT_SC, "stable_core"):
        if "error" not in r:
            done.add((r["seed"], r["stage"]))
    keys = pick_targets()[driver.SHARD_I::driver.SHARD_K]
    print("stable core: %d targets (%s), %d rows done; shard %d/%d" % (
        len(keys), "SC_TARGETS" if SC_TARGETS else "%d per cell" % N_PER, len(done), driver.SHARD_I, driver.SHARD_K),
        flush=True)
    RN.install_support_wrapper()
    install_order_wrapper()
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    fh, ftr = open(path, "a"), open(driver.shard_jsonl(OUT_SC, "train"), "a")
    t0 = time.time()
    for n_, key in enumerate(keys):
        try:
            c1 = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            rec = R.contexts(key)
            arm, thin = R.train_arm(rec)
            # ---- fit 2: same protocol, training rows in a different order
            from circuit.protocol_contexts import training_set
            perm = list(range(int(training_set(rec, arm)[0].shape[0])))
            random.Random("%s-order" % key).shuffle(perm)
            _PERMUTE["perm"] = perm
            try:
                c2, t2 = R.fit(M, rec, key, arm, OUT_SC / "circuits" / "fit2" / ("%s.pt" % key), train_fh=ftr,
                               label=dict(stage="fit2"))
            finally:
                _PERMUTE.clear()
            m1, m2 = member_set(c1), member_set(c2) if c2 is not None else set()
            core = m1 & m2
            base = dict(seed=key, arm=arm, thin=thin, n1=len(m1), n2=len(m2), n_core=len(core), jaccard=jaccard(m1, m2))
            if (key, "fit2") not in done:
                row = dict(base, stage="fit2", t_fit=round(t2, 1))
                if c2 is not None:
                    r = R.score(c2, rec, "strong")
                    row.update({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
                fh.write(json.dumps(row) + "\n"); fh.flush()
            # ---- the core alone, coefficients refitted on it
            if (key, "core") not in done:
                row = dict(base, stage="core")
                if core:
                    sup = defaultdict(list)
                    for l, kd, i in sorted(core):
                        sup[(l, kd)].append(i)
                    RN._SUPPORT.clear(); RN._SUPPORT.update(sup)
                    try:
                        cc, tc = R.fit(M, rec, key, arm, OUT_SC / "circuits" / "core" / ("%s.pt" % key), train_fh=ftr,
                                       label=dict(stage="core"))
                    finally:
                        RN._SUPPORT.clear()
                    r = R.score(cc, rec, "strong")
                    row.update({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
                    row.update(t_fit=round(tc, 1))
                else:
                    row["skip"] = "empty_core"
                fh.write(json.dumps(row) + "\n"); fh.flush()
            print("%3d/%d %-16s n1 %4d n2 %4d core %4d J %.2f  %.0fs" % (
                n_ + 1, len(keys), key, base["n1"], base["n2"], base["n_core"], base["jaccard"], time.time() - t0),
                flush=True)
        except Exception as e:  # noqa: BLE001
            _PERMUTE.clear(); RN._SUPPORT.clear()
            fh.write(json.dumps(dict(seed=key, stage="error", error="%s: %s" % (type(e).__name__, str(e)[:300]))) + "\n")
            fh.flush()
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("DONE", flush=True)


def report():
    import glob
    import pandas as pd
    d = pd.DataFrame(driver.read_jsonl_all(OUT_SC, "stable_core"))
    d = d[d.stage != "error"].drop_duplicates(["seed", "stage"], keep="last")
    ev = {}
    for f in glob.glob(str(driver.OUT / "main" / "eval.shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if "error" not in r and r.get("held") == "strong":
                ev[r["seed"]] = r
    fit1 = pd.DataFrame([ev[k] for k in d.seed.unique() if k in ev]).assign(stage="fit1")
    for x in (d, fit1):
        x["band"] = x.seed.map(lambda k: RN.band(int(k.split(".")[0])))
        x["kind"] = x.seed.map(lambda k: k.split(".")[1])
        ok = x[HEAD].notna().all(axis=1)
        x["passed"] = ok & ((x[HEAD] >= BAND[0]) & (x[HEAD] <= BAND[1])).all(axis=1) & (x["phi_sup_blind_tk"] >= NEC_MIN)
    f2, core = d[d.stage == "fit2"], d[d.stage == "core"]
    print("targets: %d | fit 2 rows %d | core rows %d" % (d.seed.nunique(), len(f2), len(core)))
    print("\n== membership overlap between the two fits (by band)")
    print(f2.groupby("band").agg(n=("seed", "size"), J=("jaccard", "median"), n1=("n1", "median"), n2=("n2", "median"),
                                 core=("n_core", "median")).round(2).to_string())
    print("core / fit-1 size: median %.2f" % (f2.n_core / f2.n1).median())
    print("\n== pass rates (DAN-8 rule, held-out strongest)")
    for name, x in (("fit 1 (15k run)", fit1), ("fit 2 (reordered)", f2), ("core alone", core)):
        print("%-18s %5.1f%%  %s" % (name, 100 * x.passed.mean(),
                                     x.groupby("band").passed.mean().mul(100).round(1).to_dict()))
    print("\n== core alone: median scores by band")
    print(core.groupby("band")[HEAD + ["phi_sup_blind_tk", "phi_cf_alpha_blind_tk"]].median().round(2).to_string())
    both = set(fit1[fit1.passed].seed) & set(f2[f2.passed].seed)
    print("\ntargets where both fits pass: %d; of those the core passes: %d" % (
        len(both), int(core[core.seed.isin(both)].passed.sum())))


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
