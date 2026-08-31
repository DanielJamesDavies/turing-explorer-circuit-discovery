"""Diagnose the AmpCircuitPatcher / circuit_only_activation divergence
on boundary configs the runner never evaluates."""
import json
import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, "src")
os.environ.setdefault("COMP", "35"); os.environ.setdefault("LAT", "13633")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ea", "experiments/043-relativity/edge_audit.py")
ea = importlib.util.module_from_spec(spec)
sys.modules["ea"] = ea
spec.loader.exec_module(ea)          # module-level setup runs (no main())

# rebuild the pieces main() builds
from pipeline.component_index import split_component_idx
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
HERE = Path("experiments/043-relativity")
COMP, LAT = 35, 13633
alphas = None
for line in open(HERE / "members.jsonl"):
    r = json.loads(line)
    if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, "triamp400"):
        alphas = {}
        for site, d in r["alphas"].items():
            lyr, knd = site.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(a) for i, a in d.items()}
layer, ki = split_component_idx(COMP, ea.n_kinds)
kind = ea.bank.kinds[ki]
avg = torch.zeros((ea.bank.n_layer * ea.n_kinds, ea.bank.d_sae),
                  device=ea.bank.device)
from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
m0 = _build_mode_method("counterfactual_gradient", "local", ea.inference,
                        ea.bank, avg, ea.pb)
pd_ = m0.build_probe_dataset(COMP, LAT); del m0
pt, pa = pd_.pos_tokens[:64], pd_.pos_argmax[:64]
pt_ho, pa_ho = pt[48:], pa[48:]
sae = ea.bank.saes[kind][layer]
w_seed = sae.encoder.weight[LAT].detach()
b_seed = sae._get_bias_eff()[LAT].detach()
UP = sorted(upstream_sites(ea.bank, layer, kind))

def read(al):
    p = ea.AmpCircuitPatcher(al, (layer, kind), w_seed, b_seed)
    tot, n = 0.0, 0
    ea.inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(pt_ho.shape[0]), 16):
                p.seed_pre = None
                ea.inference.forward(pt_ho[s0:s0 + 16], patcher=p,
                                     grad_enabled=False,
                                     return_activations=False,
                                     tokenize_final=False)
                pre = p.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = pa_ho[s0:s0 + B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum()); n += B
    finally:
        ea.inference.enable_compile()
    return tot / max(n, 1)

a_pos = float(measure_seed_activation(ea.inference, ea.bank, pt_ho, layer,
                                      kind, LAT, pa_ho, batch_size=16))
e0 = float(circuit_only_activation(ea.inference, ea.bank, {}, UP, pt_ho,
                                   layer, kind, LAT, pos_argmax=pa_ho,
                                   batch_size=16))
print("a_pos_ho        %.3f" % a_pos)
print("e0 (canonical)  %.3f" % e0)
print("read(full)      %.3f" % read(alphas))
print("read({} = no keys, all live)   %.3f" % read({}))
print("read(all sites keyed, empty)   %.3f"
      % read({s: {} for s in alphas}))
print("read(UP sites keyed, empty)    %.3f"
      % read({s: {} for s in UP}))
one = (0, "attn")
print("read(only 0/attn keyed empty)  %.3f" % read({one: {}}))
