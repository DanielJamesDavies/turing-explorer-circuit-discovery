"""CASE STUDIES of TuringLLM concept circuits, member by member.

For each seed (interesting_report.py's semantic-category shortlist):
  1. the seed's own label: peak tokens over its 64 stored contexts
  2. its members, ranked by amplitude x activation on those contexts,
     each with ITS OWN label (peak token over its own contexts) and the
     position it fires at relative to the seed's peak
  3. a GENERALISATION test on hand-written prompts the circuit has never
     seen: the seed's activation on fresh instances of the hypothesised
     class vs matched non-instances in the same frame. This is what
     separates "a concept" from "a list of memorised tokens".

  PYTHONPATH=src python experiments/049-circuit-graph/tl_case_study.py
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
T = HERE / "tables_full"; R = HERE / "results_full"
N_MEM = int(os.environ.get("N_MEM", 14))

ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
CASES = {
    "8.resid.16415": dict(name="person nouns",
                          pos=["teacher", "nurse", "pilot", "lawyer", "farmer", "soldier", "painter", "waiter", "driver", "judge"],
                          neg=["table", "river", "bridge", "engine", "mountain", "garden", "window", "letter", "bottle", "market"],
                          frame=["She spoke quietly to the {}", "The {}", 'someone writes "the {}',
                                 "Everyone in the village respected the old {}"]),
    "8.resid.26994": dict(name="temporal span",
                          pos=["time", "decades", "centuries", "years", "generations", "history", "millennia", "months", "seasons", "eras"],
                          neg=["Europe", "water", "land", "mountains", "budgets", "distances", "costs", "space", "borders", "oceans"],
                          frame=["These patterns have shifted considerably over {}",
                                 "The data reveal clear patterns and seasonality over {}",
                                 "researchers track how usage changes across {}"]),
    "8.resid.1629": dict(name="place names",
                         pos=["Berlin", "Tokyo", "Madrid", "Cairo", "Lima", "Oslo", "Delhi", "Seoul", "Vienna", "Lagos"],
                         neg=["silence", "private", "detail", "general", "theory", "practice", "writing", "person", "advance", "vain"],
                         frame="They opened a new office in {}"),
    "9.resid.37056": dict(name="spatial / world",
                          pos=["world", "space", "universe", "landscape", "terrain", "environment", "surface", "region", "territory", "cosmos"],
                          neg=["argument", "budget", "recipe", "melody", "opinion", "schedule", "grammar", "salary", "menu", "verdict"],
                          frame="This gives us a richer understanding of the {}"),
}

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

C = pd.read_parquet(T / "circuits.parquet")
M = pd.read_parquet(T / "members.parquet"); M["kind"] = M["kind"].astype(str)
C["skey"] = C["seed_layer"].astype(str) + "." + C["seed_kind"].astype(str) + "." + C["seed_index"].astype(str)
rep = pd.read_csv(R / "interesting_report.csv").set_index("skey")


def probes(l, k, i):
    d = M0.build_probe_dataset(comp_of(l, KINDS.index(k), NK), i)
    return (d.pos_tokens.cpu(), d.pos_argmax.cpu()) if d is not None and d.pos_tokens.shape[0] else (None, None)


def label(l, k, i, pt=None, pa=None):
    if pt is None:
        pt, pa = probes(l, k, i)
    if pt is None:
        return "(no contexts)", 0.0
    c = Counter(tok.decode([int(pt[b, int(pa[b])])]) for b in range(pt.shape[0]))
    top = c.most_common(3)
    return " ".join("%r" % t for t, _ in top), top[0][1] / pt.shape[0]


class Cap:
    """capture SAE activations of chosen latents at chosen sites: {site: idx tensor}."""

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
            self.out[(layer_idx, kind)] = dense[..., idx.to(dense.device)].float().cpu()   # [B, T, n]
        return x


def run(tokens, sites):
    cap = Cap(sites)
    with torch.no_grad():
        inference.forward(tokens.to(device), patcher=cap, grad_enabled=False, return_activations=False, tokenize_final=False)
    return cap.out


for skey, spec in CASES.items():
    if ONLY and skey not in ONLY:
        continue
    row = C[C["skey"] == skey].iloc[0]; cid = int(row["cid"])
    l, k, i = int(row["seed_layer"]), str(row["seed_kind"]), int(row["seed_index"])
    print("\n" + "=" * 110)
    print("CIRCUIT %s  (%s) | %d members | interest score %.2f | held-out F0 with gains %.2f, a=1 %.2f"
          % (skey, spec["name"], row["n_members"], rep.loc[skey, "score"], rep.loc[skey, "F0_amp_b"], rep.loc[skey, "F0_a1_b"]))
    pt, pa = probes(l, k, i)
    lab, cons = label(l, k, i, pt, pa)
    print("  seed peak tokens over its %d contexts: %s (top consistency %.0f%%)" % (pt.shape[0], lab, 100 * cons))
    for b in range(3):
        p = int(pa[b]); ids = pt[b].tolist()
        print("    e.g. ...%s [[%s]]" % (tok.decode(ids[max(0, p - 10):p]).replace("\n", " "), tok.decode([ids[p]])))

    # ---- members on the seed's own contexts ----------------------------------------
    mem = M[M["cid"] == cid]
    sites = defaultdict(set)
    for _, m in mem.iterrows():
        sites[(int(m["layer"]), m["kind"])].add(int(m["index"]))
    acts = defaultdict(list); at_peak = defaultdict(list)
    for s0 in range(0, pt.shape[0], 16):
        out = run(pt[s0:s0 + 16], sites)
        for (ll, kk), a in out.items():                 # a: [B, T, n]
            idx = sorted(sites[(ll, kk)])
            for b in range(a.shape[0]):
                p = int(pa[s0 + b])
                for j, li in enumerate(idx):
                    v = a[b, :p + 1, j]
                    acts[(ll, kk, li)].append(float(v.max()))
                    at_peak[(ll, kk, li)].append(float(a[b, p, j]) / max(float(v.max()), 1e-6) if float(v.max()) > 0 else 0.0)
    amp = {(int(m["layer"]), m["kind"], int(m["index"])): float(m["amplitude"]) for _, m in mem.iterrows()}
    score = {key: amp[key] * float(np.mean(v)) for key, v in acts.items()}
    top = sorted(score, key=lambda x: -score[x])[:N_MEM]
    kinds = Counter("%s%d" % (kk[0], ll) for (ll, kk, _) in amp)
    print("  members by site: %s" % dict(sorted(kinds.items(), key=lambda kv: (int(kv[0][1:]), kv[0]))))
    print("  %-14s %5s %7s %7s  %s" % ("member", "gain", "act", "@peak", "the member's OWN peak tokens"))
    for (ll, kk, li) in top:
        mlab, mcons = label(ll, kk, li)
        print("  %-14s %5.2f %7.2f %6.0f%%  %s (%.0f%%)"
              % ("%d.%s.%d" % (ll, kk, li), amp[(ll, kk, li)], float(np.mean(acts[(ll, kk, li)])),
                 100 * float(np.mean(at_peak[(ll, kk, li)])), mlab, 100 * mcons))

    # ---- generalisation on unseen prompts -------------------------------------------
    seed_site = {(l, k): {i}}
    frames = spec["frame"] if isinstance(spec["frame"], list) else [spec["frame"]]
    for frame in frames:
        print("  GENERALISATION — seed activation at the final token of '%s':" % frame)
        res = {}
        for tag in ("pos", "neg"):
            vals = []
            for w in spec[tag]:
                ids = tok.encode(frame.format(w))
                out = run(torch.tensor([ids], dtype=torch.long), seed_site)
                vals.append((w, float(out[(l, k)][0, -1, 0])))
            res[tag] = vals
        print("    class instances:  " + "  ".join("%s %.2f" % v for v in res["pos"]))
        print("    non-instances:    " + "  ".join("%s %.2f" % v for v in res["neg"]))
        mp, mn = np.mean([v for _, v in res["pos"]]), np.mean([v for _, v in res["neg"]])
        print("    mean %.2f vs %.2f | instances firing: %d/10 | non-instances firing: %d/10"
              % (mp, mn, sum(v > 0 for _, v in res["pos"]), sum(v > 0 for _, v in res["neg"])))
