"""Make the two arms comparable: every circuit scored under BOTH scopes.

The main run scores each arm against its own scope's empty baseline —
`pos` against upstream-only mean-ablation (the upper stack left intact),
`logit` against whole-model mean-ablation. Those are the right natives,
but they are DIFFERENT DENOMINATORS, so the two arms' logit_faith
columns cannot be read side by side.

This re-scores every stored circuit on both:
  faith_up    circuit restored, only the 8 upstream sites mean-ablated
  faith_all   circuit restored, ALL 36 sites mean-ablated
A logit-arm circuit evaluated at faith_up has its non-upstream members
simply left in place (they are in scope for neither ablation), and a
pos-arm circuit at faith_all must carry the whole model from means.

  PYTHONPATH=src python experiments/025-logit-endpoint/cross_scope.py
"""
import gzip
import json
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    CircuitOnlyPatcher, collect_site_means, upstream_sites)
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX, N_SEQ, EVAL_BS, D_SAE = 8, 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = set(upstream_sites(bank, LAYER, KIND))
ALL_SITES = {(l, k) for l in range(bank.n_layer) for k in bank.kinds
             if bank.saes[k][l] is not None}

circuits = {}
with gzip.open(HERE / "members.jsonl.gz", "rt") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            circuits[(r["latent"], r["arm"], r["l1"])] = [
                (a, b, c) for a, b, c in r["members"]]
print("%d circuits" % len(circuits), flush=True)

out = (HERE / "cross_scope.jsonl").open("a")
for sl in sorted({k[0] for k in circuits}):
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    tgt = pd_.target_tokens[:N_SEQ][torch.arange(pt.shape[0]), pa]
    means_all = collect_site_means(inference, bank, pt, ALL_SITES)
    means_up = {s: v for s, v in means_all.items() if s in UP}

    def metric(keep, scope, site_means):
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                p = CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                       in_scope=scope, seed_layer=-1,
                                       seed_kind="", seed_latent_idx=0,
                                       site_means=site_means)
                _, lg, _ = inference.forward(tk, patcher=p, all_logits=True,
                                             grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(
                    lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def full():
        tot, n = 0.0, int(pt.shape[0])
        inference.disable_compile()
        try:
            for s in range(0, n, EVAL_BS):
                tk = pt[s:s + EVAL_BS]
                _, lg, _ = inference.forward(tk, all_logits=True,
                                             grad_enabled=False,
                                             return_activations=False,
                                             tokenize_final=False)
                b = torch.arange(tk.shape[0], device=device)
                lp = torch.log_softmax(
                    lg[b, pa[s:s + EVAL_BS].to(device)].float(), dim=-1)
                tot += float(lp[b, tgt[s:s + EVAL_BS].to(device)].sum())
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    m_full = full()
    e_up = metric({}, UP, means_up)
    e_all = metric({}, ALL_SITES, means_all)
    print("\n[%d] full %.3f | empty up %.3f all %.3f" % (sl, m_full, e_up, e_all),
          flush=True)

    for (s2, arm, lam), mem in sorted(circuits.items()):
        if s2 != sl:
            continue
        keep = {}
        for l, k, i in mem:
            keep.setdefault((l, k), set()).add(i)
        keep_up = {s: v for s, v in keep.items() if s in UP}
        f_up = (metric(keep_up, UP, means_up) - e_up) / (m_full - e_up)
        f_all = (metric(keep, ALL_SITES, means_all) - e_all) / (m_full - e_all)
        row = {"latent": sl, "arm": arm, "l1": lam, "n": len(mem),
               "n_up": sum(len(v) for v in keep_up.values()),
               "faith_up": round(float(f_up), 4),
               "faith_all": round(float(f_all), 4)}
        out.write(json.dumps(row) + "\n"); out.flush()
        print("  %-6s l1=%-7g n=%-7d up=%-7d faith_up=%-8.4f faith_all=%.4f"
              % (arm, lam, len(mem), row["n_up"], f_up, f_all), flush=True)
        torch.cuda.empty_cache()

out.close()
print("ALL DONE", flush=True)
