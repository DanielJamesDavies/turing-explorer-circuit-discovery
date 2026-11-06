"""089: is the A / C failure of WCM circuits a coefficient trade-off against Z, or are the members insufficient?

For each target in targets.txt (select_targets.py: failing 062 production circuits, Z in band, necessity >= 0.9,
A and C < 0.8) fit these arms, 400 steps, protocol v1 otherwise (contexts, train arm, ablation values from the training
contexts, AdamW lr 0.05 / wd 0.05, batch 4, rank-keep gamma_R 3e-3):

  prod_refit      full WCM (gates + coefficients), lambda 1e-3, gamma_Z 1,    gamma_A = gamma_C 0.25  (production)
  AC_only         full WCM,                       lambda 1e-3, gamma_Z 0,    gamma_A = gamma_C 1
  alpha_AC        coefficients only on the production members,  gamma_Z 0,    gamma_A = gamma_C 1
  alpha_prod      coefficients only on the production members,  gamma_Z 1,    gamma_A = gamma_C 0.25  (control)
  alpha_AC_Zhalf  coefficients only on the production members,  gamma_Z 0.25, gamma_A = gamma_C 1

"Coefficients only" = 062 random_null's support wrapper: the engine's `support` restricted to the production circuit's
members, sparsity penalty 0, gates fully open (theta_init 40, binarize none); only the coefficients move.

gamma_Z: the engine weights the zero-ablation (Z) term at a fixed 1.0. src is NOT edited: as in 079's rs_engine, the
engine source is read, one exact text edit is applied in memory (the Z term's 1.0 becomes ZW["w"], and the term is
dropped when ZW["w"] == 0), and the result is exec'd as circuit.instrument.learned_mask. With ZW["w"] == 1 the patched
engine is the production engine. Rank-keep is computed inside each data term and so is scaled by that term's weight
(as in production); its config value stays 3e-3.

Scoring: driver.Runner.score (the production scorer) on the held-out strongest 16 (as-run split, as the 062 rows), and
the same scorer on a TRAIN probe: the 48 strongest-train reference rows (the A ablation value, as on held-out),
followed by the fit's strongest training rows (D_train, 32; arm A: strongest-train) as the scored slice. The fit's 16
mid-band training rows are not in the train probe. The production circuit itself is rescored the same way (arm "prod").

  wsl: PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/089-objective-tradeoff/run.py
  env: ARMS (comma list, default all, in the order below)  OUT89 (default experiments/089-objective-tradeoff/out)
       STEPS (driver: smoke only)  LIMIT (first N targets)  DEADLINE (unix time: start no new fit after it)
"""
import json
import os
import sys
import time
import types
import traceback
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
os.environ.setdefault("OUT", str(EXP / "062-h100-protocol-v1" / "out_full" / "out"))
os.environ.setdefault("CURVE_STRIDE", "20")
sys.path.insert(0, str(EXP / "062-h100-protocol-v1"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT89 = Path(os.environ.get("OUT89", str(HERE / "out")))
LIMIT = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
DEADLINE = float(os.environ.get("DEADLINE", "0")) or None

# name: (coefficients only?, gamma_Z, gamma_A = gamma_C, lambda)
ARMS = {
    "alpha_AC": (True, 0.0, 1.0, 0.0),
    "alpha_prod": (True, 1.0, 0.25, 0.0),
    "prod_refit": (False, 1.0, 0.25, 1e-3),
    "AC_only": (False, 0.0, 1.0, 1e-3),
    "alpha_AC_Zhalf": (True, 0.25, 1.0, 0.0),
}
WANT = [a for a in os.environ.get("ARMS", ",".join(ARMS)).split(",") if a]

# ------------------------------------------------------------------------------------------------ patched engine
SRC = REPO / "src" / "circuit" / "instrument" / "learned_mask.py"
MODNAME = "circuit.instrument.learned_mask"
EDITS = [
    ("                        terms = [(pt_tr, pa_tr, ptgt_tr, 1.0 / dn,\n"
     "                                  patcher_zero),\n"
     "                                 (pt_tr, pa_tr, ptgt_tr,\n"
     "                                  float(dual_floor_weight) / dn, patcher)]\n",
     "                        terms = ([(pt_tr, pa_tr, ptgt_tr, float(ZW[\"w\"]) / dn,\n"
     "                                   patcher_zero)] if float(ZW[\"w\"]) > 0 else []) + [\n"
     "                                 (pt_tr, pa_tr, ptgt_tr,\n"
     "                                  float(dual_floor_weight) / dn, patcher)]\n"),
]
FOOTER = "\n\n# 089: the Z (zero-ablation) term's weight, read at fit time\nZW = {\"w\": 1.0}\n"
SUPPORT = {}                                                # set per fit; read by the wrapped engine call


def install_engine():
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("089: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:160]))
        text = text.replace(anchor, repl)
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(text + FOOTER, "<089 patched %s>" % SRC.name, "exec"), mod.__dict__)
    pkg.learned_mask = mod
    orig = mod.run_learned_mask

    def wrapped(inference, bank, **kw):
        if SUPPORT:                                         # 062 random_null's coefficients-only fit
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in SUPPORT.items()},
                      l1_lambda=0.0, binarize="none", theta_init=40.0)
        return orig(inference, bank, **kw)
    mod.run_learned_mask = wrapped
    return mod


# ------------------------------------------------------------------------------------------------ helpers
def members_by_site(c):
    out = defaultdict(list)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))].append(int(f.index))
    return dict(out)


def alpha_summary(c):
    a = np.array([float(n.metadata.get("amplitude", 1.0)) for n in c.nodes.values() if n.metadata.get("role") != "seed"])
    if a.size == 0:
        return dict(alpha_med=None, alpha_p90=None, alpha_max=None, alpha_min=None)
    return dict(alpha_med=float(np.median(a)), alpha_p90=float(np.percentile(a, 90)), alpha_max=float(a.max()),
                alpha_min=float(a.min()))


KEEP = ("n", "a_pos_tk", "a_pos_pre", "free0_tk", "freeM_topk_tk", "freeN_topk_tk", "phi_sup_blind_tk", "free0_pre",
        "freeM_topk_pre", "freeN_topk_pre", "phi_sup_blind_pre", "freeM_dense_tk", "freeN_dense_tk", "free0_tr_tk",
        "free0_a1_tk", "phi_cf_alpha_blind_tk", "vacuous_tk", "alpha_median", "alpha_p90", "n_alpha_off")


def passes(r):
    v = [r.get(k) for k in ("free0_tk", "freeM_topk_tk", "freeN_topk_tk")]
    nec = r.get("phi_sup_blind_tk")
    if None in v or nec is None:
        return False
    return all(0.8 <= x <= 1.5 for x in v) and nec >= 0.9


def score_train(R, c, rec):
    """The production scorer on the training split: reference = the 48 strongest-train rows, scored slice = the fit's
    strongest training rows (D_train; arm A: strongest-train)."""
    import circuit.protocol_contexts as PC
    V, G = R.V, R.G
    arm, _ = R.train_arm(rec)
    ref = PC.pick(rec, "strong", rec["strong"]["train"])
    tr_idx = rec["D_train"] if arm == "B" else rec["strong"]["train"]
    trn = PC.pick(rec, "strong", tr_idx)
    pos, tgt, arg = PC.cat([ref, trn])
    dev = R.dev
    pd_ = PC.ProbeDataset(pos_tokens=pos.to(dev), target_tokens=tgt.to(dev), neg_tokens=rec["neg"].to(dev),
                          pos_argmax=arg.to(dev), metadata=dict(n_train=int(ref[0].shape[0]), n=int(pos.shape[0])))
    G["M0"].build_probe_dataset = lambda comp, i, _p=pd_: _p
    G["M0"]._neg_context_selector = lambda _s=PC.FixedSelector(pd_.neg_tokens): _s
    old = (V.N_SEQ, V.EVAL_SPLIT)
    V.N_SEQ, V.EVAL_SPLIT = int(pos.shape[0]), "explicit"
    try:
        row = V.score_circuit(c, {}, skip_roles=True)
    finally:
        V.N_SEQ, V.EVAL_SPLIT = old
    row["n_train_scored"] = int(trn[0].shape[0])
    return row


def slim(r, prefix):
    return {prefix + k: r.get(k) for k in KEEP if k in r}


def main():
    import driver
    keys = [t for t in (HERE / "targets.txt").read_text().split() if t][:LIMIT]
    OUT89.mkdir(parents=True, exist_ok=True)
    rows_path = OUT89 / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in open(rows_path):
            try:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["seed"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    mod = install_engine()
    R = driver.Runner()
    assert sys.modules[MODNAME] is mod
    fh, fr = open(rows_path, "a"), open(OUT89 / "train.jsonl", "a")
    t0 = time.time()
    print("089: %d targets, arms %s, steps %d, out %s" % (len(keys), WANT, driver.STEPS, OUT89), flush=True)

    def write(row):
        fh.write(json.dumps(row) + "\n"); fh.flush()
        f = lambda x: "%.2f" % x if isinstance(x, (int, float)) and x is not None else "-"
        print("[%5.0fs] %-14s %-16s n %s | ho Z %s A %s C %s nec %s pass %s | tr Z %s A %s C %s | %s" % (
            time.time() - t0, row["arm"], row["seed"], row.get("ho_n"), f(row.get("ho_free0_tk")),
            f(row.get("ho_freeM_topk_tk")), f(row.get("ho_freeN_topk_tk")), f(row.get("ho_phi_sup_blind_tk")),
            row.get("ho_pass"), f(row.get("tr_free0_tk")), f(row.get("tr_freeM_topk_tk")),
            f(row.get("tr_freeN_topk_tk")), row.get("error", "")), flush=True)

    def score_all(c, rec, base):
        ho = R.score(c, rec, "strong")
        tr = score_train(R, c, rec)
        row = dict(base, **slim(ho, "ho_"), **slim(tr, "tr_"), ho_pass=passes(ho), tr_pass=passes(tr),
                   tr_n_scored=tr.get("n_train_scored"), **alpha_summary(c))
        return row

    # the production circuits themselves, rescored (held-out reproduces 062 on this GPU; train split is new)
    for key in keys:
        if (key, "prod") in done:
            continue
        try:
            rec = R.contexts(key)
            c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
            write(score_all(c, rec, dict(seed=key, arm="prod")))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            write(dict(seed=key, arm="prod", error="%s: %s" % (type(e).__name__, str(e)[:300])))

    for arm in WANT:
        coef_only, gz, gac, lam = ARMS[arm]
        M = R.method(gac, lam if not coef_only else driver.LAM, True)   # coef-only: the wrapper sets lambda 0
        for key in keys:
            if (key, arm) in done:
                continue
            if DEADLINE and time.time() > DEADLINE:
                print("089: deadline reached, %s %s not started" % (arm, key), flush=True)
                return
            base = dict(seed=key, arm=arm, coef_only=coef_only, gamma_Z=gz, gamma_AC=gac,
                        lam=0.0 if coef_only else lam, rank_weight=driver.RANK, steps=driver.STEPS)
            try:
                rec = R.contexts(key)
                tr_arm, _ = R.train_arm(rec)
                ref = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
                SUPPORT.clear()
                if coef_only:
                    SUPPORT.update(members_by_site(ref))
                mod.ZW["w"] = float(gz)
                c, t_fit = R.fit(M, rec, key, tr_arm, OUT89 / "circuits" / arm / ("%s.pt" % key), train_fh=fr,
                                 label=dict(arm89=arm, gamma_Z=gz, gamma_AC=gac))
                SUPPORT.clear(); mod.ZW["w"] = 1.0
                if c is None:
                    write(dict(base, skip="rejected"))
                    continue
                row = score_all(c, rec, base)
                row.update(t_fit=round(t_fit, 1), n_ref=driver.n_nodes(ref))
                write(row)
            except Exception as e:  # noqa: BLE001
                SUPPORT.clear(); mod.ZW["w"] = 1.0
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                write(dict(base, error="%s: %s" % (type(e).__name__, str(e)[:300])))
    print("089: DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
