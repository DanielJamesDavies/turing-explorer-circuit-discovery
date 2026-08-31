"""Re-score behaviour circuits WITH their fitted amplitudes
(keep_scales) -- behaviour_runner.metric() dropped them, scoring a
tri-amp circuit as if every alpha were 1.0. Scores mean-fill and
zero-fill frames, held-out contexts, v2 clusters only.

  PYTHONPATH=src python .../score_amp.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import CircuitOnlyPatcher
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
EVAL_BS = 16

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
    keep, scales = {}, {}
    for site, d in r["alphas"].items():
        lyr, knd = site.split("/")
        st = (int(lyr), knd)
        keep[st] = set(int(i) for i in d)
        v = torch.ones(bank.d_sae)
        for i, a in d.items():
            v[int(i)] = float(a)
        scales[st] = v
    circuits[r["cluster"]] = (keep, scales)   # latest write (v2) wins


def metric(keep, scales, site_means, tokens, anchors, targets):
    tot, m = 0.0, int(tokens.shape[0])
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s in range(0, m, EVAL_BS):
                tk = tokens[s:s + EVAL_BS]
                p = (CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                        in_scope=set(ALL_SITES),
                                        seed_layer=-1, seed_kind="",
                                        seed_latent_idx=0,
                                        site_means=site_means,
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
    if ck not in circuits:
        continue
    keep, scales = circuits[ck]
    idx = (assign == ck).nonzero(as_tuple=True)[0].tolist()
    wins = [data["windows"][i] for i in idx]
    tgts = [data["targets"][i] for i in idx]
    n = len(wins)
    n_tr = max(8, int(n * 0.75))
    pt = torch.tensor([[max(t, 0) for t in w[:ANCHOR + 1]] for w in wins],
                      dtype=torch.long, device=device)
    pa = torch.full((n,), ANCHOR, dtype=torch.long)
    tgt = torch.tensor(tgts, dtype=torch.long)
    pt_tr = pt[:n_tr]
    pt_ho, pa_ho, tgt_ho = pt[n_tr:], pa[n_tr:], tgt[n_tr:]
    means = collect_site_means(inference, bank, pt_tr, set(ALL_SITES))
    m_full = metric(None, None, None, pt_ho, pa_ho, tgt_ho)
    for tag, sm in (("zero", None), ("mean", means)):
        m_empty = metric({}, None, sm, pt_ho, pa_ho, tgt_ho)
        den = m_full - m_empty
        m_plain = metric(keep, None, sm, pt_ho, pa_ho, tgt_ho)
        m_amp = metric(keep, scales, sm, pt_ho, pa_ho, tgt_ho)
        print("cluster %-3d %s-fill: EF plain %.3f -> WITH AMPS %.3f "
              "(circ %.3f, empty %.3f)"
              % (ck, tag, (m_plain - m_empty) / den,
                 (m_amp - m_empty) / den, m_amp, m_empty), flush=True)
