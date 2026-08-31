"""Show ONE edge end to end, per held-out probe: the seed's
pre-activation with the full fitted circuit vs the same circuit with a
single member knocked out (mean-filled). The per-sequence deltas are
the edge, shown rather than summarised.

  COMP=29 LAT=3736 M_SITE=3/resid M_LAT=18699 PYTHONPATH=src python
      experiments/043-relativity/edge_demo.py
"""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ea", str(Path(__file__).parent / "edge_audit.py"))
ea = importlib.util.module_from_spec(spec)
sys.modules["ea"] = ea
spec.loader.exec_module(ea)              # module-level setup only

from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
from eval.ablation_faithfulness import circuit_only_activation, upstream_sites
from eval.floors import collect_site_anchors
from model.tokenizer import Tokenizer
from pipeline.component_index import split_component_idx

HERE = Path(__file__).parent
COMP, LAT = int(os.environ["COMP"]), int(os.environ["LAT"])
lyr_s, knd_s = os.environ["M_SITE"].split("/")
M_SITE, M_LAT = (int(lyr_s), knd_s), int(os.environ["M_LAT"])

alphas = None
for line in open(HERE / "members.jsonl"):
    r = json.loads(line)
    if (r["comp_idx"], r["latent"], r["arm"]) == (COMP, LAT, "triamp400"):
        alphas = {}
        for site, d in r["alphas"].items():
            l_, k_ = site.split("/")
            alphas[(int(l_), k_)] = {int(i): float(a) for i, a in d.items()}
layer, ki = split_component_idx(COMP, ea.n_kinds)
kind = ea.bank.kinds[ki]
avg = torch.zeros((ea.bank.n_layer * ea.n_kinds, ea.bank.d_sae),
                  device=ea.bank.device)
m0 = _build_mode_method("counterfactual_gradient", "local", ea.inference,
                        ea.bank, avg, ea.pb)
pd_ = m0.build_probe_dataset(COMP, LAT)
del m0
pt, pa = pd_.pos_tokens[:64], pd_.pos_argmax[:64]
pt_tr, pa_tr = pt[:48], pa[:48]
pt_ho, pa_ho = pt[48:], pa[48:]
UP = sorted(upstream_sites(ea.bank, layer, kind))
means_tr, _ = collect_site_anchors(ea.inference, ea.bank, pt_tr, set(UP),
                                   pa_tr, pin_position_specific=False)


def read_perseq(al):
    """Seed pre-activation at the anchor, one value per held-out probe."""
    keep = {st: set(d) for st, d in al.items() if d}
    scales = {}
    for st, d in al.items():
        if not d:
            continue
        v = torch.ones(ea.bank.d_sae)
        for i, a in d.items():
            v[int(i)] = float(a)
        scales[st] = v
    vals = []
    for b in range(int(pt_ho.shape[0])):
        vals.append(float(circuit_only_activation(
            ea.inference, ea.bank, keep, UP, pt_ho[b:b + 1], layer, kind,
            LAT, pos_argmax=pa_ho[b:b + 1], site_means=means_tr,
            keep_scales=scales, preact=True)))
    return vals


drop = {s: dict(d) for s, d in alphas.items()}
drop[M_SITE].pop(M_LAT)
full = read_perseq(alphas)
without = read_perseq(drop)

tok = Tokenizer()
print("\nEDGE: L%d %s latent %d  ->  seed c%d/%d" %
      (M_SITE[0], M_SITE[1], M_LAT, COMP, LAT))
print("seed pre-activation at its anchor token, per HELD-OUT probe:")
print("%-4s %9s %11s %8s  %s" % ("seq", "full", "w/o member", "drop",
                                 "probe text at the anchor"))
for b, (f, w) in enumerate(zip(full, without)):
    s = pt_ho[b].tolist()
    p = int(pa_ho[b])
    lo, hi = max(0, p - 6), min(len(s), p + 4)
    cl = lambda a, c: tok.decode([t for t in s[a:c] if t >= 0])
    txt = "%s[[%s]]%s" % (cl(lo, p), cl(p, p + 1), cl(p + 1, hi))
    print("%-4d %9.2f %11.2f %8.2f  %s"
          % (b, f, w, f - w, txt.replace("\n", " ")[:60]))
mf, mw = sum(full) / len(full), sum(without) / len(without)
print("\nmean: full %.2f -> without %.2f  (drop %.2f, %.0f%% of the "
      "circuit's drive above the mean-fill baseline)"
      % (mf, mw, mf - mw, 100 * (mf - mw) / max(mf - mw + 1e-9, 1e-9)))
