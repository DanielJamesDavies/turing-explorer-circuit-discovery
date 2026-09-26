"""INSPECT census hits: is the distal dependence a VARIABLE, and do the
members carry it?

For each seed (SEEDS env, or the top N of algorithm_census.csv):
  1. its contexts, with the peak token in [[ ]] and the occlusion-critical
     positions (drop >= 0.25) in { }; what the seed promotes (top tokens
     by direct logit effect)
  2. at each critical offset: the tokens found there across contexts —
     ONE token = a frame word, MANY = a variable
  3. the readers at the critical offset (members whose activation there
     predicts the seed), each with its own label
  4. THE SWAP: for pairs of contexts (a, b) install b's reader states at
     a's critical position and read a's seed:
        score = (act_swap - act_a) / (act_b - act_a)   (median over pairs)
     against site-matched random latents given the same values. If the
     readers carry the variable, the seed follows the donor.

  PYTHONPATH=src python experiments/049-circuit-graph/algorithm_inspect.py
Env: SEEDS (comma list) | TOP (12) | N_PAIRS (40) | MIN_DROP (0.25)
"""
import json
import os
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
TOP = int(os.environ.get("TOP", 12)); N_PAIRS = int(os.environ.get("N_PAIRS", 40)); MIN_DROP = float(os.environ.get("MIN_DROP", 0.25))
SEEDS = [s for s in os.environ.get("SEEDS", "").split(",") if s]
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
config.discovery.probe_sequence_count = 64; config.discovery.probe_batch_size = 4
M0 = _build_mode_method("counterfactual_gradient", "local", inference, bank, avg_acts, pb)
tok = Tokenizer()
inference.disable_compile()
W_U = inference.model.lm_head.weight.detach().float() * inference.model.transformer.norm_f.scale.detach().float()[None, :]

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet"); M["kind"] = M["kind"].astype(str)
census = {json.loads(l)["skey"]: json.loads(l) for l in open(R / "algorithm_census.jsonl")}
if not SEEDS:
    SEEDS = list(pd.read_csv(R / "algorithm_census.csv").sort_values("score", ascending=False)["skey"].head(TOP))
members = defaultdict(lambda: defaultdict(set)); amp = {}
for cid, l, k, i, a in zip(M["cid"].values, M["layer"].values, M["kind"].values, M["index"].values, M["amplitude"].values):
    members[int(cid)][(int(l), k)].add(int(i)); amp[(int(cid), int(l), k, int(i))] = float(a) if a == a else 1.0


def parse(skey):
    l, k, i = skey.split("."); return int(l), k, int(i)


def probes(l, k, i):
    d = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
    return (d.pos_tokens.cpu(), d.pos_argmax.cpu()) if d is not None and d.pos_tokens.shape[0] else (None, None)


_LAB = {}


def label(l, k, i):
    if (l, k, i) not in _LAB:
        pt, pa = probes(l, k, i)
        if pt is None:
            _LAB[(l, k, i)] = "(no contexts)"
        else:
            c = Counter(tok.decode([int(pt[b, int(pa[b])])]) for b in range(pt.shape[0]))
            _LAB[(l, k, i)] = " ".join("%r" % t for t, _ in c.most_common(3)) + " (%.0f%%)" % (100 * c.most_common(1)[0][1] / pt.shape[0])
    return _LAB[(l, k, i)]


def promotes(l, k, i, n=8):
    dvec = bank.saes[k][l].decoder.weight[:, i].detach().float().to(W_U.device)
    lg = W_U @ dvec
    top = torch.topk(lg, n).indices.tolist(); bot = torch.topk(-lg, 4).indices.tolist()
    return " ".join("%r" % tok.decode([t]) for t in top), " ".join("%r" % tok.decode([t]) for t in bot)


class Cap:
    def __init__(self, sites):
        self.sites = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in sites.items()}; self.out = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        idx = self.sites.get((layer_idx, kind))
        if idx is not None:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
            self.out[(layer_idx, kind)] = dense[..., idx.to(dense.device)].float().cpu().numpy()
        return x


class Inject:
    """per batch row: spec[b] = {site: {pos: (idx, vals)}}; also captures `read` sites."""

    def __init__(self, spec, read):
        self.spec = spec; self.read = {s: torch.tensor(sorted(v), dtype=torch.long) for s, v in read.items()}; self.out = {}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        hit = any(s in sp for sp in self.spec)
        if not hit and s not in self.read:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, D, dtype=x.dtype)
        if s in self.read:
            self.out[s] = dense[..., self.read[s].to(dense.device)].float().cpu().numpy()
        if not hit:
            return x
        code = dense.clone()
        for b, sp in enumerate(self.spec):
            for p, (idx, vals) in sp.get(s, {}).items():
                code[b, p, idx.to(code.device)] = vals.to(code.device, code.dtype)
        return x + bank.decode(code - dense, kind, layer_idx, add_bias=False).to(x.dtype)


def run(tokens, sites, specs=None):
    outs = defaultdict(list)
    with torch.no_grad():
        for s0 in range(0, tokens.shape[0], BS):
            p = Inject(specs[s0:s0 + BS], sites) if specs else Cap(sites)
            inference.forward(tokens[s0:s0 + BS].to(device), patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
            for s, a in p.out.items():
                outs[s].append(a)
    return {s: np.concatenate(v) for s, v in outs.items()}


for skey in SEEDS:
    l, k, i = parse(skey); rec = census.get(skey)
    row = C[(C["seed_layer"] == l) & (C["seed_kind"] == k) & (C["seed_index"] == i)].iloc[0]; cid = int(row["cid"])
    pt, pa = probes(l, k, i)
    keep = [b for b in range(pt.shape[0]) if int(pa[b]) >= 4]; pt = pt[keep]; pa = pa[keep]; B = pt.shape[0]
    crit = [-(j + 2) for j, v in enumerate(rec["occ"]) if v >= MIN_DROP] if rec else []
    print("\n" + "=" * 120)
    print("SEED %s | %d members | census: ord_dep %.2f occ %s | critical offsets %s" % (skey, row["n_members"], rec["ord_dep"] if rec else -1,
          " ".join("%.2f" % v for v in rec["occ"]) if rec else "-", crit))
    up, down = promotes(l, k, i)
    print("  seed label %s | promotes %s | suppresses %s" % (label(l, k, i), up, down))
    for b in range(min(8, B)):
        p = int(pa[b]); ids = pt[b].tolist(); words = []
        for q in range(max(0, p - 12), min(len(ids), p + 3)):
            w = tok.decode([ids[q]]).replace("\n", "\\n")
            words.append("[[%s]]" % w if q == p else ("{%s}" % w if (q - p) in crit else w))
        print("    " + "".join(words))
    # 2. tokens at critical offsets
    for o in crit:
        c = Counter(tok.decode([int(pt[b, int(pa[b]) + o])]) for b in range(B) if int(pa[b]) + o >= 0)
        print("  offset %d: %d distinct tokens over %d contexts; top: %s" % (o, len(c), B, " ".join("%r×%d" % kv for kv in c.most_common(6))))
    # 3. readers at the critical offset (from the census) + labels
    if not rec or not rec["readers_at"]:
        print("  no readers at the critical position in the census -> no swap"); continue
    o = rec["occ_off"]
    readers = [(m["site"], m["corr"]) for m in rec["readers_at"]]
    print("  readers at offset %d (%d in census; showing %d):" % (o, rec["n_readers_at"], len(readers)))
    for site, corr in readers:
        ll, kk, ii = parse(site)
        print("    %-14s gain %.2f corr %+.2f | %s" % (site, amp.get((cid, ll, kk, ii), 1.0), corr, label(ll, kk, ii)))
    # 4. swap at the critical offset: all readers at that offset (recompute the set from the members: distal at o with |corr|>=0.4)
    sites = defaultdict(set)
    for s, v in members[cid].items():
        sites[s] |= v
    sites[(l, k)].add(i)
    full = run(pt, sites)
    seed_full = np.array([full[(l, k)][b, int(pa[b]), sorted(sites[(l, k)]).index(i)] for b in range(B)])
    rd = []
    for s, a in full.items():
        idx = sorted(sites[s])
        for j, li in enumerate(idx):
            if s == (l, k) and li == i:
                continue
            at = np.array([a[b, int(pa[b]) + o, j] if int(pa[b]) + o >= 0 else 0.0 for b in range(B)])
            if (at > 0).mean() >= 0.5 and at.std() > 0 and seed_full.std() > 0 and abs(np.corrcoef(at, seed_full)[0, 1]) >= 0.4:
                rd.append((s, li, at))
    if len(rd) < 2:
        print("  fewer than 2 readers recomputed -> no swap"); continue
    order = np.argsort(seed_full)
    pairs = []
    for _ in range(N_PAIRS):
        a_, b_ = int(rng.choice(order[:B // 3])), int(rng.choice(order[-B // 3:]))
        if int(pa[a_]) + o >= 0:
            pairs.append((a_, b_))
    recv = pt[[a_ for a_, _ in pairs]]; pa_r = pa[[a_ for a_, _ in pairs]]

    def specs_for(rand=False):
        sp = []
        for a_, b_ in pairs:
            d_ = {}
            by = defaultdict(lambda: ([], []))
            for s, li, at in rd:
                ii = li
                if rand:
                    while True:
                        ii = int(rng.integers(D))
                        if ii not in sites[s]:
                            break
                by[s][0].append(ii); by[s][1].append(float(at[b_]))
            for s, (ii, vv) in by.items():
                d_[s] = {int(pa[a_]) + o: (torch.tensor(ii, dtype=torch.long), torch.tensor(vv))}
            sp.append(d_)
        return sp

    seed_site = {(l, k): {i}}
    out = {}
    for name, rand in (("readers", False), ("random matched", True)):
        r_ = run(recv, seed_site, specs_for(rand))[(l, k)]
        sw = np.array([r_[n_, int(pa_r[n_]), 0] for n_ in range(len(pairs))])
        a_act = seed_full[[a_ for a_, _ in pairs]]; b_act = seed_full[[b_ for _, b_ in pairs]]
        sc = (sw - a_act) / np.where(np.abs(b_act - a_act) < 1e-6, np.nan, b_act - a_act)
        out[name] = (float(np.nanmedian(sc)), float(a_act.mean()), float(sw.mean()), float(b_act.mean()))
    print("  SWAP at offset %d, %d readers, %d low->high pairs: receiver %.2f -> swapped %.2f (donor %.2f) | score readers %.2f vs random %.2f"
          % (o, len(rd), len(pairs), out["readers"][1], out["readers"][2], out["readers"][3], out["readers"][0], out["random matched"][0]))
