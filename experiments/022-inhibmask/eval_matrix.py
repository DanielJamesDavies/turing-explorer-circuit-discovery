"""Full eval matrix for the inhibitor-mask circuits (lambda 1e-3 arm).

Scores the ARCHIVED brake sets on every standing eval, so the object can
be placed against abl-mask (closure), AMPC (drive) and R:
  free0        members kept, everything else ablated to zero
  freeN_topk   members kept, rest at the negctx floor, top-k respected
  cf / sup     the counterfactual pair, members labelled INHIBITOR
  raise/drop   the brake-specific pair (from the run, re-derived here)
  + size-matched RANDOM control on every column.

  PYTHONPATH=src python experiments/022-inhibmask/eval_matrix.py
"""
import gzip
import json
import random
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.circuits import Circuit, CircuitNode
from circuit.discovery.counterfactual_gradient import SeedPreActCapture

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
ARM = "lam0.001"
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053),
         (20, 35678), (26, 17432), (27, 6859), (35, 6599)]

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24)
config.discovery.eval_batch_size = EVAL_BS


class SetPatcher:
    def __init__(self, values, seed_site, seed_idx):
        self.values, self.seed_site, self.seed_idx = values, seed_site, seed_idx
        self.argmax_chunk = None
        self.seed_capture = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            ta, ti = bank.encode(x, kind, layer_idx)
            s_dense = target_latent_activations(ta, ti, self.seed_idx)
            pa_c = self.argmax_chunk
            if pa_c is not None:
                B = min(s_dense.shape[0], pa_c.shape[0])
                pa_c = pa_c[:B].to(s_dense.device).clamp(0, s_dense.shape[1] - 1)
                rows = torch.arange(B, device=s_dense.device)
                self.seed_capture = float(s_dense[rows, pa_c].mean())
            else:
                self.seed_capture = float(s_dense.mean())
            return x
        v = self.values.get((layer_idx, kind))
        if not v:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        c_new = dense.clone()
        zi = torch.tensor(sorted(v), device=dense.device, dtype=torch.long)
        zv = torch.tensor([v[int(i)] for i in zi], device=dense.device,
                          dtype=dense.dtype)
        c_new[..., zi] = zv
        out = bank.decode(c_new - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


rows = []
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    mp = HERE / ("members_v2_%s_%d_%d.jsonl.gz" % (ARM, sc_idx, sl))
    if not mp.exists():
        print("missing %s" % mp.name, flush=True); continue
    members = []
    with gzip.open(mp, "rt", encoding="utf-8") as gz:
        for line in gz:
            l_, kd_, idx_, v_ = json.loads(line)
            members.append(((l_, kd_), int(idx_)))

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, nt_tr = pt[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den_free = a_pos_ev - a_e0
    _, pins_pos = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                       pin_position_specific=False)
    # negctx anchors for the freeN floor
    sae_ = bank.saes[kind][layer]
    cap = SeedPreActCapture(layer, kind, sae_.encoder.weight[sl].detach(),
                            sae_._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        chunks = []
        with torch.no_grad():
            for s0 in range(0, int(nt_tr.shape[0]), EVAL_BS):
                cap.seed_pre_act = None
                inference.forward(nt_tr[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                chunks.append(cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    na_tr = torch.cat(chunks, 0).argmax(dim=1).cpu()
    _, neg_means = collect_site_anchors(inference, bank, nt_tr, up, na_tr,
                                        pin_position_specific=False)

    def act_vals(values, tokens, argmax):
        p = SetPatcher(values, (layer, kind), sl)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                p.seed_capture = None
                p.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                tot += float(p.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def score(mem, label):
        keep = {}
        for s, i in mem:
            keep.setdefault(s, set()).add(i)
        f0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        fn = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=neg_means, respect_topk=True))
        c = Circuit(name="inh")
        for (l, kd), idx in mem:
            c.add_node(CircuitNode(metadata={
                "layer_idx": l, "kind": kd, "latent_idx": idx,
                "role": "counterfactual_inhibitor"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev, circuit_layers={l for (l, _), _ in mem})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception:
            cf_v = sup_v = None
        sil = act_vals({s: {i: 0.0} for s, i in mem} if False else
                       {s: {i: 0.0 for (ss, i) in mem if ss == s}
                        for s, _ in mem}, pt_ev, pa_ev)
        amp = act_vals({s: {i: 4.0 * float(pins_pos[s][i])
                            for (ss, i) in mem if ss == s}
                        for s, _ in mem}, pt_ev, pa_ev)
        return {"label": label, "n": len(mem),
                "free0": round((f0 - a_e0) / den_free, 4),
                "freeN_topk": round((fn - a_e0) / den_free, 4),
                "cf": cf_v, "sup": sup_v,
                "raise": round((sil - a_pos_ev) / max(a_pos_ev, 1e-9), 4),
                "drop_x4": round((a_pos_ev - amp) / max(a_pos_ev, 1e-9), 4)}

    rng = random.Random(11)
    rnd = [(up_sorted[rng.randrange(len(up_sorted))], rng.randrange(D_SAE))
           for _ in range(len(members))]
    for mem, lab in ((members, "inhib"), (rnd, "random")):
        r = score(mem, lab)
        r.update({"seed": seed_key, "layer": layer, "kind": kind})
        rows.append(r)
        print("%-10s L%-2d %-5s %-7s n=%-4d free0=%-8s freeN=%-8s cf=%-8s "
              "sup=%-8s raise=%-8s drop4=%s"
              % (seed_key, layer, kind, lab, r["n"], r["free0"], r["freeN_topk"],
                 r["cf"], r["sup"], r["raise"], r["drop_x4"]), flush=True)
    torch.cuda.empty_cache()

with (HERE / "eval_matrix.jsonl").open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("ALL DONE", flush=True)
