"""Matched null for the amplitude-aware scores: random member ids per
site, SAME membership count per site, and the circuit's OWN amplitude
values permuted onto them. Zero-fill frame, held-out contexts. If the
real circuits' EF ~0.94-0.99 collapses here, membership identity (not
amplitude mass) carries the recovery.

  PYTHONPATH=src python .../null_amp.py
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import CircuitOnlyPatcher
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
EVAL_BS = 16
N_NULL = 2

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
_apply_sweep_config(max_per_site=24)
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in bank.kinds)

data = torch.load(HERE / "behaviour_clusters.pt", weights_only=False)
ANCHOR = data["anchor"]
assign = data["assign"]

circuits = {}
for line in open(HERE / "behaviour_members.jsonl"):
    r = json.loads(line)
    circuits[r["cluster"]] = {
        tuple([int(s.split("/")[0]), s.split("/")[1]]):
            {int(i): float(a) for i, a in d.items()}
        for s, d in r["alphas"].items()}


def metric(keep, scales, tokens, anchors, targets):
    tot, m = 0.0, int(tokens.shape[0])
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s in range(0, m, EVAL_BS):
                tk = tokens[s:s + EVAL_BS]
                p = (CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                        in_scope=set(ALL_SITES),
                                        seed_layer=-1, seed_kind="",
                                        seed_latent_idx=0, site_means=None,
                                        keep_scales=scales)
                     if keep is not None else None)
                out = inference.forward(tk, patcher=p, all_logits=True,
                                        grad_enabled=False,
                                        return_activations=False,
                                        tokenize_final=False)
                lg = out[1] if isinstance(out, (tuple, list)) else out
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(
                    lg[b, anchors[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, targets[s:s + EVAL_BS].to(device)].sum())
    finally:
        inference.enable_compile()
    return tot / max(m, 1)


for ck in (5, 95, 99):
    al = circuits[ck]
    idx = (assign == ck).nonzero(as_tuple=True)[0].tolist()
    wins = [data["windows"][i] for i in idx]
    tgts = [data["targets"][i] for i in idx]
    n = len(wins)
    n_tr = max(8, int(n * 0.75))
    pt = torch.tensor([[max(t, 0) for t in w[:ANCHOR + 1]] for w in wins],
                      dtype=torch.long, device=device)
    pa = torch.full((n,), ANCHOR, dtype=torch.long)
    tgt = torch.tensor(tgts, dtype=torch.long)
    pt_ho, pa_ho, tgt_ho = pt[n_tr:], pa[n_tr:], tgt[n_tr:]
    m_full = metric(None, None, pt_ho, pa_ho, tgt_ho)
    m_empty = metric({}, None, pt_ho, pa_ho, tgt_ho)
    den = m_full - m_empty
    for j in range(N_NULL):
        rng = random.Random(101 + j)
        keep_n, scales_n = {}, {}
        for st, d in al.items():
            ids = rng.sample(range(bank.d_sae), len(d))
            keep_n[st] = set(ids)
            v = torch.ones(bank.d_sae)
            amps = list(d.values())
            rng.shuffle(amps)
            for i, a in zip(ids, amps):
                v[i] = a
            scales_n[st] = v
        mn = metric(keep_n, scales_n, pt_ho, pa_ho, tgt_ho)
        print("cluster %-3d amp-null%d: EF %.3f (logp %.3f | empty %.3f)"
              % (ck, j, (mn - m_empty) / den, mn, m_empty), flush=True)
