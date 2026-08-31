"""THE SAE CEILING: score a circuit that keeps EVERY latent at every
site. Circuit-only execution then equals "replace the stream with the
SAE reconstruction (post top-k) at all 36 sites" -- pure
reconstruction-error damage, no membership choice involved. If this
ceiling is itself low, behaviour-level EF was never achievable by any
member set (Marks et al. include SAE error nodes for this reason).

Also: the same ceiling restricted to the 12 resid sites only.

  PYTHONPATH=src python .../ceiling_test.py
"""
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

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
_apply_sweep_config(max_per_site=24)
ALL_SITES = sorted((l, k) for l in range(bank.n_layer) for k in bank.kinds)
RESID_SITES = [s for s in ALL_SITES if s[1] == "resid"]

data = torch.load(HERE / "behaviour_clusters.pt", weights_only=False)
ANCHOR = data["anchor"]
assign = data["assign"]
EVERY = set(range(bank.d_sae))


def metric(keep, scope, tokens, anchors, targets):
    tot, m = 0.0, int(tokens.shape[0])
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s in range(0, m, EVAL_BS):
                tk = tokens[s:s + EVAL_BS]
                p = (CircuitOnlyPatcher(bank=bank, keep_indices=keep,
                                        in_scope=set(scope), seed_layer=-1,
                                        seed_kind="", seed_latent_idx=0,
                                        site_means=None)
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
    m_full = metric(None, ALL_SITES, pt_ho, pa_ho, tgt_ho)
    m_empty = metric({}, ALL_SITES, pt_ho, pa_ho, tgt_ho)
    keep_all = {s: EVERY for s in ALL_SITES}
    m_ceil = metric(keep_all, ALL_SITES, pt_ho, pa_ho, tgt_ho)
    keep_res = {s: EVERY for s in RESID_SITES}
    m_ceil_r = metric(keep_res, RESID_SITES, pt_ho, pa_ho, tgt_ho)
    den = m_full - m_empty
    print("cluster %-3d full %7.3f | empty %7.3f || ALL-latent ceiling: "
          "36-site %7.3f (EF %.3f) | resid-only %7.3f (EF %.3f)"
          % (ck, m_full, m_empty, m_ceil, (m_ceil - m_empty) / den,
             m_ceil_r, (m_ceil_r - m_empty) / den), flush=True)
