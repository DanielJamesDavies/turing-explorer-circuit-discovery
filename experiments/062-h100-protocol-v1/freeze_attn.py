"""WHY DO ATTENTION-OUTPUT CIRCUITS FAIL? Test 1: freeze attention patterns at their clean values (2026-10-03).

Hypothesis: an attention output depends on where its heads look (the QK pattern over all positions). Mean-ablating
every non-circuit latent at every position can move that pattern, and a position-agnostic circuit cannot restore it.
If so, holding the attention pattern at its clean value during the circuit-only runs should rescue attention targets
under the mean ablations (A, C) far more than it changes MLP / resid targets.

Each sampled full-run circuit is scored three times with the production scorer (held-out strongest, activation read):
  arm "none"    normal scoring (recomputed through the same code path, the baseline)
  arm "target"  the attention pattern of the TARGET's layer is replayed from the clean run in circuit-only runs
  arm "all"     every layer's attention pattern is replayed from the clean run in circuit-only runs
Patterns are recorded on clean forwards (the scorer's natural-stream and floor passes, keyed by the token sequence;
first record wins) and replayed only on circuit-only runs (TapCO). The sufficiency-to-induce runs (TapCF) are left
unfrozen. Values (V) and everything else are recomputed from the ablated stream, so the test isolates the pattern.

Sample: non-near-threshold attention targets (N_ATTN per depth band) plus MLP and resid controls (N_CTRL per band per
kind), from the full run, fixed seed. Resumable (one row per target per arm).

  OUT=experiments/062-h100-protocol-v1/out_full/out PYTHONPATH=src python experiments/062-h100-protocol-v1/freeze_attn.py
  env: N_ATTN (50)  N_CTRL (8)  OUT_FZ (default experiments/062-h100-protocol-v1/out_freeze)  REPORT=1
"""
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
os.environ.setdefault("OUT", str(HERE / "out_full" / "out"))
sys.path.insert(0, str(HERE))
import driver  # noqa: E402

N_ATTN = int(os.environ.get("N_ATTN", 50))
N_CTRL = int(os.environ.get("N_CTRL", 8))
OUT_FZ = Path(os.environ.get("OUT_FZ", str(HERE / "out_freeze")))
HEAD = ["free0_tk", "freeM_topk_tk", "freeN_topk_tk"]
ARMS = ("none", "target", "all")

STATE = dict(arm=None, mode=None, keys=None, layers=set(), store={}, hits=0, misses=0)


def band(layer):
    return "L0-4" if layer <= 4 else ("L5-7" if layer <= 7 else "L8-11")


def install(R):
    """Patch the attention module (record / replay its softmax pattern) and wrap inference.forward to choose the mode
    from the patcher: circuit-only (TapCO) -> replay; counterfactual induce (TapCF) -> untouched; else -> record."""
    import model.turingllm as T
    inference = R.G["inference"]
    for i, block in enumerate(inference.model.transformer.h):
        block.attn._layer_idx = i
    orig_impl = T.CausalSelfAttention._forward_impl

    def impl(self, x):
        mode, L = STATE["mode"], getattr(self, "_layer_idx", None)
        if mode is None or L is None or (mode == "replay" and L not in STATE["layers"]):
            return orig_impl(self, x)
        B, Tn, C = x.size()
        hd = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, Tn, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
        att = None
        if mode == "replay":
            got = [STATE["store"].get((L, key)) for key in STATE["keys"]]
            if all(a is not None and a.shape[-1] == Tn for a in got):
                att = torch.stack(got).to(v.dtype)
                STATE["hits"] += B
            else:
                STATE["misses"] += B
        if att is None:
            s = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            mask = torch.ones(Tn, Tn, dtype=torch.bool, device=x.device).tril()
            att = F.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
            if mode == "record":
                for b, key in enumerate(STATE["keys"]):
                    STATE["store"].setdefault((L, key), att[b].detach().half())
        y = (att @ v).transpose(1, 2).contiguous().view(B, Tn, C)
        return self.c_proj(y)
    T.CausalSelfAttention._forward_impl = impl

    TapCO, TapCF = R.V._engine_classes()
    orig_fwd = inference.forward

    def fwd(tokens, *a, patcher=None, **kw):
        if STATE["arm"] in (None, "none"):
            STATE["mode"] = None
        elif isinstance(patcher, TapCO):
            STATE["mode"] = "replay"
        elif isinstance(patcher, TapCF):
            STATE["mode"] = None
        else:
            STATE["mode"] = "record"
        if STATE["mode"] is not None:
            STATE["keys"] = [hash(tuple(r)) for r in tokens.detach().cpu().tolist()]
        try:
            return orig_fwd(tokens, *a, patcher=patcher, **kw)
        finally:
            STATE["mode"] = None
    inference.forward = fwd
    inference.disable_compile()
    inference.enable_compile = lambda: None                # keep eager: the patch lives in the eager attention code


def sample():
    import pandas as pd
    tg = pd.read_csv(driver.OUT / "targets.csv", index_col="seed")
    have = {p.stem for p in (driver.OUT / "main" / "circuits").glob("*.pt")}
    tg = tg[tg.index.isin(have)]
    tg["layer"] = [int(k.split(".")[0]) for k in tg.index]
    tg["kind"] = [k.split(".")[1] for k in tg.index]
    tg["band"] = tg.layer.map(band)
    near = tg.rank_clean >= 64
    rng = random.Random(20261003)
    out = []
    for b in ("L0-4", "L5-7", "L8-11"):
        pool = sorted(tg[(tg.kind == "attn") & (tg.band == b) & ~near].index)
        out += rng.sample(pool, min(N_ATTN, len(pool)))
        for k in ("mlp", "resid"):
            pool = sorted(tg[(tg.kind == k) & (tg.band == b) & ~near].index)
            out += rng.sample(pool, min(N_CTRL, len(pool)))
    return out


def main():
    OUT_FZ.mkdir(parents=True, exist_ok=True)
    path = OUT_FZ / "freeze.jsonl"
    done = set()
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line); done.add((r["seed"], r["arm"]))
            except Exception:  # noqa: BLE001
                pass
    keys = sample()
    print("freeze test: %d targets x %d arms (%d rows done)" % (len(keys), len(ARMS), len(done)), flush=True)
    R = driver.Runner()
    install(R)
    fh = open(path, "a")
    t0 = time.time()
    for n, key in enumerate(keys):
        c = torch.load(driver.OUT / "main" / "circuits" / ("%s.pt" % key), weights_only=False)
        if c is None:
            continue
        rec = R.contexts(key)
        layer = int(key.split(".")[0])
        for arm in ARMS:
            if (key, arm) in done:
                continue
            STATE.update(arm=arm, store={}, hits=0, misses=0,
                         layers={layer} if arm == "target" else set(range(len(R.G["inference"].model.transformer.h))))
            row = dict(seed=key, arm=arm)
            try:
                R.H.patch_eval_contexts(R.G, rec, "strong")
                r = R.V.score_circuit(c, {}, skip_roles=True)
                row.update({k: v for k, v in r.items() if isinstance(v, (int, float, str, bool)) or v is None})
                row.update(replay_hits=STATE["hits"], replay_misses=STATE["misses"])
            except Exception as e:  # noqa: BLE001
                row["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
                traceback.print_exc()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            STATE.update(arm=None, store={})
            fh.write(json.dumps(row) + "\n"); fh.flush()
        el = time.time() - t0
        print("%3d/%d %-16s %.0fs elapsed, ~%.0f min left" % (n + 1, len(keys), key, el,
                                                             el / (n + 1) * (len(keys) - n - 1) / 60), flush=True)
    print("DONE", flush=True)


def report():
    import pandas as pd
    d = pd.DataFrame([json.loads(l) for l in open(OUT_FZ / "freeze.jsonl")])
    d = d[d.get("error", pd.Series(index=d.index, dtype=object)).isna()]
    d = d[~d.get("vacuous_tk", pd.Series(False, index=d.index)).fillna(False).astype(bool)]
    d["pass"] = ((d[HEAD] >= 0.8) & (d[HEAD] <= 1.5)).all(axis=1) & (d.phi_sup_blind_tk >= 0.9)
    d["kind"] = d.seed.str.split(".").str[1]
    d["band"] = d.seed.map(lambda k: band(int(k.split(".")[0])))
    print("replay misses (should be 0): %d of %d" % (d.replay_misses.fillna(0).sum(),
                                                     (d.replay_hits.fillna(0) + d.replay_misses.fillna(0)).sum()))
    g = d.groupby(["kind", "arm"]).agg(n=("seed", "nunique"), pass_=("pass", "mean"), Z=("free0_tk", "median"),
                                       A=("freeM_topk_tk", "median"), C=("freeN_topk_tk", "median"))
    print(g.round(2).to_string())
    a = d[d.kind == "attn"]
    print("\nattention by band:")
    print(a.groupby(["band", "arm"]).agg(pass_=("pass", "mean"), A=("freeM_topk_tk", "median"),
                                         C=("freeN_topk_tk", "median")).round(2).to_string())
    p = a.pivot_table(index="seed", columns="arm", values="pass")
    if {"none", "all"} <= set(p.columns):
        print("\nattention targets rescued by freezing all patterns: %d of %d failing; broken: %d of %d passing" % (
            ((p["none"] == 0) & (p["all"] == 1)).sum(), (p["none"] == 0).sum(),
            ((p["none"] == 1) & (p["all"] == 0)).sum(), (p["none"] == 1).sum()))


if __name__ == "__main__":
    report() if os.environ.get("REPORT") else main()
