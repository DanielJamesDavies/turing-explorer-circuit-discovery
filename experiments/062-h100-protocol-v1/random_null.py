"""DAN-17: the random-circuit baseline with fitted coefficients, under protocol v1.

For each target, draw a random circuit matched to the target's production circuit (same number of members at every
upstream site), with members drawn from the latents active on the target's training contexts, then fit its
coefficients with the production objective and budget and score it with the production scorer on the held-out
strongest contexts. A passing random circuit would mean faithfulness comes from coefficient fitting, not node choice.

Coefficient-only fitting follows the pre-v1 panel's null (experiments/029-panel/runner.py): the mask engine's
`support` restricts the search to the random members, the sparsity penalty is 0 and gates start fully open
(theta_init 40, binarize none), so only the coefficients move. Everything else (contexts, ablation values, term
weights, rank-keep, steps, lr) is the v1 production configuration, because the fit goes through the same
driver.Runner.fit path with the engine call wrapped.

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/random_null.py
  env: TARGETS (default targets_sweep.txt; or out_full/out/targets.csv for the whole run)
       N_TARGETS (64, stratified by depth band and kind, cells interleaved)  N_DRAWS (2)
       OUT_RN (default experiments/062-h100-protocol-v1/out_random)  REPORT=1 (summarise; no GPU)
"""
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

N_TARGETS = int(os.environ.get("N_TARGETS", 64))
N_DRAWS = int(os.environ.get("N_DRAWS", 2))
OUT_RN = Path(os.environ.get("OUT_RN", str(HERE / "out_random")))
TARGETS = Path(os.environ.get("TARGETS", str(HERE / "targets_sweep.txt")))
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]

_SUPPORT = {}                                              # set per fit; read by the wrapped engine call


def install_support_wrapper():
    """Wrap circuit.instrument.learned_mask.run_learned_mask (ablation_gradient imports it at call time) so that, when
    _SUPPORT is set, the fit is restricted to it with no sparsity penalty and fully open gates."""
    import circuit.instrument.learned_mask as LM
    orig = LM.run_learned_mask

    def wrapped(inference, bank, **kw):
        if _SUPPORT:
            kw.update(support={s: torch.tensor(v, dtype=torch.long) for s, v in _SUPPORT.items()},
                      l1_lambda=0.0, binarize="none", theta_init=40.0)
        return orig(inference, bank, **kw)
    LM.run_learned_mask = wrapped


def members_by_site(c):
    out = defaultdict(list)
    for n in c.nodes.values():
        md = n.metadata
        if md.get("role") == "seed":
            continue
        f = md["feature_id"]
        out[(int(f.layer), str(f.kind))].append(int(f.index))
    return out


def live_latents(R, tokens, sites):
    """Latents with a nonzero (post-Top-K) activation anywhere on the given contexts, per upstream site."""
    G = R.G
    k2i = {k: n for n, k in enumerate(G["KINDS"])}
    live = {s: set() for s in sites}

    def hook(layer_idx, activations):
        for kd in G["KINDS"]:
            s = (layer_idx, kd)
            if s in live:
                ta, ti = G["bank"].encode(activations[k2i[kd]], kd, layer_idx)
                live[s].update(ti.reshape(-1)[ta.reshape(-1) > 0].long().cpu().tolist())
    G["inference"].disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(tokens.shape[0]), 16):
                G["inference"].forward(tokens[s0:s0 + 16].to(G["device"]), activations_callback=hook,
                                       return_activations=False, tokenize_final=False)
    finally:
        G["inference"].enable_compile()
    return live


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def pick_targets():
    """Stratified: equal numbers per (depth band, kind) from the target list, fixed seed. TARGETS may be a text list
    or the run's targets.csv (every full-run target). The cells are interleaved (round-robin) so that a run stopped
    part-way still covers every cell evenly; the sampled set is the same as cell-by-cell order."""
    if TARGETS.suffix == ".csv":
        import pandas as pd
        keys = list(pd.read_csv(TARGETS, index_col="seed").index)
    else:
        keys = [t for t in TARGETS.read_text().split() if t]
    cells = defaultdict(list)
    for k in keys:
        l, kind, _ = k.split(".")
        cells[(band(int(l)), kind)].append(k)
    rng = random.Random(20261003)
    per = max(1, N_TARGETS // max(1, len(cells)))
    picks = [rng.sample(cells[cell], min(per, len(cells[cell]))) for cell in sorted(cells)]
    out = [p[i] for i in range(max(map(len, picks))) for p in picks if i < len(p)]
    return out[:N_TARGETS]


def main():
    OUT_RN.mkdir(parents=True, exist_ok=True)
    path = OUT_RN / "random.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line); done.add((r["seed"], r["draw"]))
            except Exception:  # noqa: BLE001
                pass
    keys = pick_targets()
    print("random null: %d targets x %d draws (%d done)" % (len(keys), N_DRAWS, len(done)), flush=True)
    install_support_wrapper()
    R = driver.Runner()
    M = R.method(driver.GAMMA, driver.LAM, True)
    fh = open(path, "a")
    t0 = time.time()
    for n_, key in enumerate(keys):
        main_c = driver.OUT / "main" / "circuits" / ("%s.pt" % key)
        if not main_c.exists():
            continue
        ref = torch.load(main_c, weights_only=False)
        if ref is None:
            continue
        ref_sites = members_by_site(ref)
        rec = R.contexts(key)
        arm, thin = R.train_arm(rec)
        live = None
        rng = random.Random("%s-null" % key)
        for draw in range(N_DRAWS):
            # sample before the skip so a resumed run draws identical sets
            if live is None:
                tr = rec["strong"]["pos"][rec["strong"]["train"]] if "train" in rec["strong"] else rec["strong"]["pos"]
                live = live_latents(R, tr, set(ref_sites))
            support = {s: rng.sample(sorted(live.get(s, ())), min(len(v), len(live.get(s, ()))))
                       for s, v in ref_sites.items()}
            if (key, draw) in done:
                continue
            row = dict(seed=key, draw=draw, n_ref=sum(len(v) for v in ref_sites.values()),
                       n_drawn=sum(len(v) for v in support.values()),
                       short_sites=sum(1 for s, v in ref_sites.items() if len(live.get(s, ())) < len(v)))
            try:
                _SUPPORT.clear(); _SUPPORT.update(support)
                c, t_fit = R.fit(M, rec, key, arm, OUT_RN / "circuits" / ("%s.d%d.pt" % (key, draw)))
                _SUPPORT.clear()
                if c is None:
                    row["skip"] = "rejected"
                else:
                    r = R.score(c, rec, "strong")
                    row.update({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
                    row.update(held="strong", t_fit=round(t_fit, 1))
            except Exception as e:  # noqa: BLE001
                _SUPPORT.clear()
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("%3d/%d %-16s draw %d  n %s/%s  %s  %.0fs" % (
                n_ + 1, len(keys), key, draw, row.get("n"), row["n_ref"], row.get("error") or row.get("skip") or "ok",
                time.time() - t0), flush=True)
    print("DONE", flush=True)


def report():
    import glob
    import pandas as pd
    rn = pd.DataFrame([json.loads(l) for l in open(OUT_RN / "random.jsonl")])
    rn = rn[rn.get("error", pd.Series(index=rn.index, dtype=object)).isna()]
    rn = rn[rn.get("skip", pd.Series(index=rn.index, dtype=object)).isna()]
    passes = lambda d: (((d[HEAD] >= 0.8) & (d[HEAD] <= 1.5)).all(axis=1)) & (d.phi_sup_blind_tk >= 0.9)
    ev = [json.loads(l) for f in glob.glob(str(driver.OUT / "main" / "eval.shard*.jsonl")) for l in open(f)]
    ev = pd.DataFrame([r for r in ev if r.get("held") == "strong" and "error" not in r
                       and r.get("seed") in set(rn.seed)]).drop_duplicates("seed", keep="last")
    rn["pass"] = passes(rn)
    ev["pass"] = passes(ev)
    rn["band"] = rn.seed.map(lambda k: band(int(k.split(".")[0])))
    ev["band"] = ev.seed.map(lambda k: band(int(k.split(".")[0])))
    print("random draws: %d over %d targets | members drawn / reference: median %.2f" % (
        len(rn), rn.seed.nunique(), (rn.n_drawn / rn.n_ref).median()))
    for name, d in (("fitted WCM circuits", ev), ("random, fitted coefficients", rn)):
        g = d.groupby("band").agg(p=("pass", "mean"), Z=("free0_tk", "median"), A=("freeM_topk_tk", "median"),
                                  C=("freeN_topk_tk", "median"), nec=("phi_sup_blind_tk", "median"),
                                  ind=("phi_cf_alpha_blind_tk", "median"))
        print("== %s (all: pass %.1f%%)" % (name, 100 * d["pass"].mean()))
        print(g.round(2).to_string())


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
