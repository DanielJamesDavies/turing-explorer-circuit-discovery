"""CENSUS: which of the production circuits are ALGORITHMS?

Greater-than (gt_variables.py) taught us what an algorithmic circuit
looks like in this data: the seed's activation depends on DISTAL
context (not just its own token / local frame), and the circuit has
members at an EARLIER position whose activation VARIES across contexts
in a way that predicts the seed (they carry a variable), rather than
only firing at the seed's own position.

For every non-lexical, well-fitted seed (top-token consistency < 0.5,
posctx suppression >= 0.9, >= 30 members) on its 64 stored contexts:
  ctx_dep   1 - act(prefix replaced by another context's prefix) / act(full)
            keeping the last two tokens; = dependence on distal CONTENT
  ord_dep   1 - act(prefix shuffled) / act(full); = dependence on ORDER
  distal members: for each member, its modal firing offset relative to
            the seed's peak (<= 0), the consistency of that offset, and
            corr(member act at its offset, seed act) across contexts.
            A "distal reader" = offset <= -2, consistency >= 0.5, fires
            on >= 50% of contexts, |corr| >= 0.4.
  n_distal_readers, distal_share (amp x act mass of distal readers /
            all members), n_offsets (distinct distal offsets used)
score = ctx_dep * sqrt(n_distal_readers) * (1 + n_offsets) — ranks
circuits that need distal context AND have members carrying it.

Positive controls: greater-than seeds; negatives: the confirmed concept
seeds (9.resid.37056, 8.resid.16415). Both are forced into the run and
reported separately so the score is calibrated.

  PYTHONPATH=src python experiments/049-circuit-graph/algorithm_census.py
Env: N_CTX (64), LIMIT (0 = all), OUT
"""
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.component_index import component_idx as comp_of
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

HERE = Path(__file__).parent
T = HERE / "tables_full"; R = Path(os.environ.get("OUT", str(HERE / "results_full")))
N_CTX = int(os.environ.get("N_CTX", 64)); LIMIT = int(os.environ.get("LIMIT", 0))
BS = 16
rng = np.random.default_rng(0)
torch.set_float32_matmul_precision("high")
load_discovery_artifacts("outputs", candidates_path="outputs/candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
KINDS = list(bank.kinds); NK = len(KINDS); D = bank.d_sae
avg_acts = torch.zeros((bank.n_layer * NK, D), device=bank.device)
config.discovery.probe_sequence_count = N_CTX; config.discovery.probe_batch_size = 4
M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)
tok = Tokenizer()
inference.disable_compile()

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet"); M["kind"] = M["kind"].astype(str)
rank = pd.read_csv(R / "interesting_circuits.csv")
cand = rank[(rank["tok_cons"] < 0.5) & (rank["posctx_sup"] >= 0.9) & (rank["n_members"] >= 30)].copy()
CONTROLS = {"2.mlp.6540": "gt frame", "9.attn.7712": "gt transport", "3.resid.6422": "gt tens detector", "2.resid.687": "gt frame",
            "9.resid.37056": "concept: spatial", "8.resid.16415": "concept: human agent"}
forced = rank[rank["skey"].isin(CONTROLS)]
cand = pd.concat([forced, cand[~cand["skey"].isin(CONTROLS)]]).drop_duplicates("skey")
if LIMIT:
    cand = cand.head(LIMIT)
print("census over %d seeds (%d controls)" % (len(cand), len(forced)))
members = defaultdict(lambda: defaultdict(set)); amp = {}
for cid, l, k, i, a in zip(M["cid"].values, M["layer"].values, M["kind"].values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(l), k)].add(int(i)); amp[(int(cid), int(l), k, int(i))] = float(a) if a == a else 1.0


def probes(l, k, i):
    d = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
    return (d.pos_tokens.cpu(), d.pos_argmax.cpu()) if d is not None and d.pos_tokens.shape[0] else (None, None)


class Cap:
    def __init__(self, sites):
        self.sites = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in sites.items()}
        self.out = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.sites.get((layer_idx, kind))
        if idx is not None:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
            self.out[(layer_idx, kind)] = dense[..., idx.to(dense.device)].float().cpu().numpy()
        return x


def run(tokens, sites):
    outs = defaultdict(list)
    with torch.no_grad():
        for s0 in range(0, tokens.shape[0], BS):
            cap = Cap(sites)
            inference.forward(tokens[s0:s0 + BS].to(device), patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
            for s, a in cap.out.items():
                outs[s].append(a)
    return {s: np.concatenate(v) for s, v in outs.items()}


rows = []; t0 = time.time()
existing = set()
out_path = R / "algorithm_census.jsonl"
if out_path.exists():
    for ln in open(out_path):
        existing.add(json.loads(ln)["skey"])
fout = open(out_path, "a")
for n_, r in enumerate(cand.itertuples()):
    skey = r.skey
    if skey in existing:
        continue
    cid = int(r.cid); l, k, i = int(r.seed_layer), str(r.seed_kind), int(r.seed_index)
    pt, pa = probes(l, k, i)
    if pt is None:
        print("  skip %s: no contexts" % skey); continue
    keep = [b for b in range(pt.shape[0]) if int(pa[b]) >= 4]
    if len(keep) < 8:
        print("  skip %s: only %d/%d contexts with peak >= 4 (peaks %s)" % (skey, len(keep), pt.shape[0], sorted(Counter(int(x) for x in pa).items())[:6])); continue
    pt = pt[keep]; pa = pa[keep]; B, L = pt.shape
    sites = defaultdict(set)
    for s, v in members[cid].items():
        sites[s] |= v
    sites[(l, k)].add(i)
    # full run with member capture
    full = run(pt, sites)
    seed_full = np.array([full[(l, k)][b, int(pa[b]), sorted(sites[(l, k)]).index(i)] for b in range(B)])
    # prefix replaced / shuffled, seed only
    rep = pt.clone(); shf = pt.clone()
    perm = rng.permutation(B)
    for b in range(B):
        p = int(pa[b]); n = p - 1
        src = pt[perm[b], :n] if perm[b] != b else pt[(b + 1) % B, :n]
        rep[b, :n] = src[:n] if src.shape[0] >= n else torch.cat([src, pt[b, src.shape[0]:n]])
        shf[b, :n] = pt[b, torch.tensor(rng.permutation(n))]
    seed_site = {(l, k): {i}}
    s_rep = np.array([run(rep, seed_site)[(l, k)][b, int(pa[b]), 0] for b in range(B)]) if True else None
    s_shf = np.array([run(shf, seed_site)[(l, k)][b, int(pa[b]), 0] for b in range(B)])
    ok = seed_full > 0
    ctx_dep = float(np.clip(1 - s_rep[ok].sum() / max(seed_full[ok].sum(), 1e-6), 0, 1))
    ord_dep = float(np.clip(1 - s_shf[ok].sum() / max(seed_full[ok].sum(), 1e-6), 0, 1))
    # positional occlusion (gated): replace ONE distal token at offset -2..-OCC with the token another
    # context has at that position; drop = 1 - act/act_full. Sharp profile = a variable read at a position.
    OCC = 9
    occ = [0.0] * (OCC - 1); occ_max = 0.0; occ_off = 0; occ_conc = 0.0
    if ord_dep >= 0.4 and ok.sum() >= 8:
        for o in range(2, OCC + 1):
            x = pt.clone()
            for b in range(B):
                p = int(pa[b]) - o
                if p >= 0:
                    x[b, p] = pt[perm[b], p] if int(pa[perm[b]]) - o != p or perm[b] == b else pt[(b + 1) % B, p]
            s_o = run(x, seed_site)[(l, k)]
            s_o = np.array([s_o[b, int(pa[b]), 0] for b in range(B)])
            occ[o - 2] = float(np.clip(1 - s_o[ok].sum() / max(seed_full[ok].sum(), 1e-6), 0, 1))
        occ_max = max(occ); occ_off = -(int(np.argmax(occ)) + 2)
        occ_conc = float(occ_max / max(sum(occ), 1e-6))
    # members: activation profile over offsets -W..0 relative to the seed's peak (mean over contexts);
    # a member is DISTAL if >= 50% of its profile mass is at offsets <= -2; its offset = the profile's peak
    # among those; consistency = fraction of contexts in which the member fires at that offset; corr =
    # corr(member act at that offset, seed act at the peak) across contexts.
    W = 12
    mem_rows = []
    for s, a in full.items():
        idx = sorted(sites[s])
        for j, li in enumerate(idx):
            if s == (l, k) and li == i:
                continue
            prof = np.zeros((B, W + 1), dtype=np.float32)          # column w = offset -(W-w)
            for b in range(B):
                p = int(pa[b]); lo = max(0, p - W)
                seg = a[b, lo:p + 1, j]
                prof[b, W + 1 - len(seg):] = seg
            mean = prof.mean(0)
            if mean.sum() <= 0:
                continue
            distal_mass = mean[:W - 1].sum() / mean.sum()            # offsets <= -2
            w = int(mean[:W - 1].argmax()); off = w - W
            at = prof[:, w]
            fire = float((at > 0).mean())
            c = np.corrcoef(at, seed_full)[0, 1] if at.std() > 0 and seed_full.std() > 0 else 0.0
            g = amp.get((cid, s[0], s[1], li), 1.0)
            mem_rows.append(dict(site="%d.%s.%d" % (s[0], s[1], li), off=off, cons=float(distal_mass), fire=fire, corr=float(c),
                                 mass=g * float(mean.sum()), distal=bool(distal_mass >= 0.5 and fire >= 0.5)))
    mass_all = sum(m["mass"] for m in mem_rows) or 1e-6
    distal_any = [m for m in mem_rows if m["distal"]]
    distal = [m for m in distal_any if abs(m["corr"]) >= 0.4]
    n_off = len({m["off"] for m in distal})
    readers_at = [m for m in distal if m["off"] == occ_off]            # readers sitting at the occlusion-critical position
    score = ord_dep * occ_max * occ_conc * np.sqrt(1 + len(readers_at))
    rec = dict(skey=skey, cid=cid, layer=l, kind=k, n_members=int(r.n_members), n_ctx=B, ctx_dep=ctx_dep, ord_dep=ord_dep,
               n_distal_readers=len(distal), n_distal_any=len(distal_any), distal_share=float(sum(m["mass"] for m in distal) / mass_all),
               n_offsets=n_off, occ=occ, occ_max=occ_max, occ_off=occ_off, occ_conc=occ_conc, n_readers_at=len(readers_at),
               score=float(score), control=CONTROLS.get(skey, ""),
               top_distal=sorted(distal, key=lambda m: -abs(m["corr"]) * m["mass"])[:8],
               readers_at=sorted(readers_at, key=lambda m: -abs(m["corr"]) * m["mass"])[:8])
    fout.write(json.dumps(rec) + "\n"); fout.flush()
    rows.append(rec)
    if n_ % 25 == 0 or LIMIT:
        print("  %4d/%d %-14s ctx_dep %.2f ord %.2f distal %3d/%3d off %d | occ max %.2f @%d conc %.2f readers@ %d | score %5.2f  [%.0fs]"
              % (n_, len(cand), skey, ctx_dep, ord_dep, len(distal), len(distal_any), n_off, occ_max, occ_off, occ_conc, len(readers_at), score, time.time() - t0), flush=True)
fout.close()
df = pd.DataFrame([json.loads(ln) for ln in open(out_path)])
df = df.drop(columns=["top_distal", "readers_at"]).sort_values("score", ascending=False)
df.to_csv(R / "algorithm_census.csv", index=False)


def line(r):
    return ("%-14s n %4d ctx %.2f ord %.2f distal %3d/%3d | occ max %.2f @%d conc %.2f readers@ %2d | profile -2..-9 %s"
            % (r["skey"], r["n_members"], r["ctx_dep"], r["ord_dep"], r["n_distal_readers"], r["n_distal_any"], r["occ_max"], r["occ_off"],
               r["occ_conc"], r["n_readers_at"], " ".join("%.2f" % v for v in r["occ"])))


print("\nCONTROLS:")
for _, r in df[df["control"] != ""].iterrows():
    print("  %5.2f %-20s " % (r["score"], r["control"]) + line(r))
print("\nTOP 40 BY ALGORITHM SCORE (order-dependence x sharpest single-position occlusion x readers there):")
for _, r in df.head(40).iterrows():
    print("  %5.2f " % r["score"] + line(r))
print("\nby seed kind: median score", df.groupby("kind")["score"].median().to_dict())
print("->", R / "algorithm_census.csv")
