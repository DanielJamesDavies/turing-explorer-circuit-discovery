"""RECURSIVE LATENT MAP — does the internal explanation close?

"Latents explain latents": expand a seed into its top-K drivers, then
expand each DRIVER into ITS top-K drivers, recursively. Two questions:

 (Q1) STRUCTURE — does the frontier saturate? If drivers-of-drivers
      overlap heavily the graph closes over a readable number of nodes;
      if not, the map is irreducibly wide. Measured per hop: new nodes,
      cumulative unique, and the SHARING rate (fraction of a node's
      drivers already seen).

 (Q2) EVALS — is the depth-d union a CIRCUIT for the seed? Scored at
      every depth on the full matrix: free0 (closure semantics), pin0,
      and cf under the AMPC alpha-fit (drive semantics), each against a
      size-matched random control. The hypothesis worth killing or
      keeping: the transitive closure of DRIVERS may reconstruct the
      CLOSURE object that abl-mask finds directly — i.e. (P2) as the
      recursive completion of (P1).

Edges are VALUE-edges (D3.3): d(node value)/du * (u_nat - u_floor) —
the effect of an upstream latent on ANOTHER LATENT's value, not on a
behavioural metric. One forward per level, one backward per node
(retain_graph), so cost is ~linear in expanded nodes.

  PYTHONPATH=src python experiments/023-recursive-map/runner.py
"""
import gzip
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.sae_graph import SAEGraphInstrument
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

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
N_SEQ, N_TR, EVAL_BS, GRAD_B = 64, 48, 16, 8
BRANCH = int(os.environ.get("BRANCH", 8))       # top-K drivers per node
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 3))
MAX_EXPAND = int(os.environ.get("MAX_EXPAND", 220))   # runtime guard
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(8, 20333), (20, 35678), (26, 17432), (35, 6599)]

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

ALL_SITES = [(l, k) for l in range(bank.n_layer) for k in bank.kinds
             if bank.saes[k][l] is not None]


def site_rank(site):
    """Total order on sites: deeper = later. Upstream means strictly lower."""
    return ALL_SITES.index(site)


class InjectPatcher:
    def __init__(self, targets, seed_site, seed_idx):
        self.targets, self.seed_site, self.seed_idx = targets, seed_site, seed_idx
        self.argmax_chunk = None
        self.seed_capture = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            ta, ti = bank.encode(x, kind, layer_idx)
            s = target_latent_activations(ta, ti, self.seed_idx)
            pa_c = self.argmax_chunk
            if pa_c is not None:
                B = min(s.shape[0], pa_c.shape[0])
                pa_c = pa_c[:B].to(s.device).clamp(0, s.shape[1] - 1)
                rows = torch.arange(B, device=s.device)
                self.seed_capture = float(s[rows, pa_c].mean())
            else:
                self.seed_capture = float(s.mean())
            return x
        t = self.targets.get((layer_idx, kind))
        if not t:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        c_new = dense.clone()
        idxs = torch.tensor(sorted(t), device=dense.device, dtype=torch.long)
        vals = torch.tensor([t[int(i)] for i in idxs], device=dense.device,
                            dtype=dense.dtype)
        c_new[..., idxs] = vals
        out = bank.decode(c_new - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


OUT = HERE / "rows.jsonl"
fh = OUT.open("a")

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    seed_site = (layer, kind)
    up = upstream_sites(bank, layer, kind)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den_free = a_pos_ev - a_e0
    _, pins_ev = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                      pin_position_specific=False)
    print("\n[%s] L%d %s | %d upstream sites | a_pos %.3f | branch=%d depth=%d"
          % (seed_key, layer, kind, len(up), a_pos_ev, BRANCH, MAX_DEPTH),
          flush=True)

    # ---------------- recursive expansion ----------------
    # one forward per LEVEL; one backward per node (retain_graph)
    visited = set()                    # (site, idx) already expanded
    nodes_by_depth = {0: [(seed_site, sl)]}
    edges = []
    t0 = time.time()
    n_expanded = 0
    for depth in range(1, MAX_DEPTH + 1):
        frontier = [n for n in nodes_by_depth[depth - 1] if n not in visited]
        frontier = frontier[:max(0, MAX_EXPAND - n_expanded)]
        if not frontier:
            break
        instrument = SAEGraphInstrument(bank)
        inference.disable_compile()
        try:
            inference.forward(pt_tr[:GRAD_B], patcher=instrument,
                              grad_enabled=True, return_activations=False,
                              tokenize_final=False)
        finally:
            inference.enable_compile()
        graph = instrument.graph
        avail = [s for s in sorted(graph.activations)]
        anchors = [graph.get_latents(*s)[0].act for s in avail]
        new_nodes, shared_hits, total_hits = [], 0, 0
        for (nsite, nidx) in frontier:
            if nsite not in graph.activations:
                continue
            _, conn, _ = graph.get_latents(*nsite)
            v = conn.act[..., nidx]
            B = min(v.shape[0], pa_tr.shape[0])
            rows = torch.arange(B, device=v.device)
            val = v[:B][rows, pa_tr[:B].to(v.device).clamp(0, v.shape[1] - 1)].mean()
            grads = torch.autograd.grad(val, anchors, allow_unused=True,
                                        retain_graph=True)
            cand = []
            for s, a, g in zip(avail, anchors, grads):
                if g is None or site_rank(s) >= site_rank(nsite):
                    continue           # strictly upstream only
                w = (g * a.detach()).sum(dim=1).mean(dim=0).abs().float().cpu()
                kk = min(BRANCH, w.numel())
                vv, ix = torch.topk(w, k=kk)
                cand += [(float(x), s, int(i)) for x, i in zip(vv, ix)]
            cand.sort(key=lambda x: -x[0])
            for wgt, s, i in cand[:BRANCH]:
                edges.append([list(nsite), nidx, list(s), i, round(wgt, 5), depth])
                total_hits += 1
                if (s, i) in visited or any((s, i) == n for lvl in nodes_by_depth
                                            for n in nodes_by_depth[lvl]):
                    shared_hits += 1
                else:
                    new_nodes.append((s, i))
            visited.add((nsite, nidx))
            n_expanded += 1
        instrument.release()
        del instrument, graph, anchors
        torch.cuda.empty_cache()
        # dedupe preserving order
        seen = set()
        uniq = []
        for n in new_nodes:
            if n not in seen:
                seen.add(n); uniq.append(n)
        nodes_by_depth[depth] = uniq
        cum = set()
        for d in range(1, depth + 1):
            cum |= set(nodes_by_depth.get(d, []))
        share = shared_hits / max(total_hits, 1)
        print("  depth %d: expanded %d nodes -> %d new (%d cumulative) | "
              "sharing %.2f | %.0fs"
              % (depth, len(frontier), len(uniq), len(cum), share,
                 time.time() - t0), flush=True)
        fh.write(json.dumps({
            "seed": seed_key, "layer": layer, "kind": kind, "task": "structure",
            "depth": depth, "branch": BRANCH, "expanded": len(frontier),
            "new_nodes": len(uniq), "cumulative": len(cum),
            "sharing_rate": round(share, 4)}) + "\n")
        fh.flush()

    # ---------------- evals of the depth-d union ----------------
    rng = random.Random(31)
    up_sorted = sorted(up)

    def keep_of(mem):
        k = {}
        for s, i in mem:
            k.setdefault(s, set()).add(i)
        return k

    def act_under(patcher, tokens, argmax):
        if patcher is None:
            return float(measure_seed_activation(
                inference, bank, tokens, layer, kind, sl, argmax,
                batch_size=EVAL_BS))
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_capture = None
                patcher.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                tot += float(patcher.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    a_base = act_under(None, nt_ev, pa_ev)
    den_cf = a_pos_ev - a_base

    def alpha_fit(mem):
        targets = {}
        for s, i in mem:
            v = float(pins_ev[s][i]) if s in pins_ev else 0.0
            if v > 0:
                targets.setdefault(s, {})[i] = v
        if not targets:
            return None, None
        base = {s: dict(t) for s, t in targets.items()}

        def at(alpha, tk, am):
            sc = {s: {i: alpha * v for i, v in t.items()} for s, t in base.items()}
            return act_under(InjectPatcher(sc, seed_site, sl), tk, am)
        lo, hi = 0.25, 8.0
        if at(hi, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
            alpha = hi
        elif at(lo, nt_tr[:16], pa_tr[:16]) > a_pos_tr:
            alpha = lo
        else:
            for _ in range(6):
                mid = (lo + hi) / 2
                if at(mid, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                    lo = mid
                else:
                    hi = mid
            alpha = (lo + hi) / 2
        a_int = at(alpha, nt_ev, pa_ev)
        return round(alpha, 3), (round((a_int - a_base) / den_cf, 4)
                                 if abs(den_cf) > 1e-9 else None)

    def evaluate(mem, tag, depth):
        mem = [m for m in mem if m[0] in up]
        if not mem:
            return
        f0 = float(circuit_only_activation(
            inference, bank, keep_of(mem), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        p0 = float(circuit_only_activation(
            inference, bank, keep_of(mem), up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins_ev))
        al, cfa = alpha_fit(mem)
        c = Circuit(name="rec")
        for (l, kd), idx in mem:
            c.add_node(CircuitNode(metadata={"layer_idx": l, "kind": kd,
                                             "latent_idx": idx,
                                             "role": "ablation_support"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev, circuit_layers={l for (l, _), _ in mem})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception:
            cf_v = sup_v = None
        row = {"seed": seed_key, "layer": layer, "kind": kind, "task": "eval",
               "depth": depth, "set": tag, "n": len(mem),
               "pct_dict": round(100.0 * len(mem) / (len(up) * D_SAE), 5),
               "free0": round((f0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "pin0": round((p0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "alpha": al, "cf_alpha": cfa, "cf_raw": cf_v, "sup": sup_v}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("    eval d%d %-8s n=%-6d free0=%-8s pin0=%-8s cf_a=%-8s (a*=%s) sup=%s"
              % (depth, tag, len(mem), row["free0"], row["pin0"], cfa, al, sup_v),
              flush=True)

    cum = set()
    for d in sorted(k for k in nodes_by_depth if k >= 1):
        cum |= set(nodes_by_depth[d])
        evaluate(sorted(cum), "recursive", d)
        evaluate([(up_sorted[rng.randrange(len(up_sorted))], rng.randrange(D_SAE))
                  for _ in range(len(cum))], "random", d)

    with gzip.open(HERE / ("edges_%d_%d.jsonl.gz" % (sc_idx, sl)), "wt",
                   encoding="utf-8") as gz:
        for e in edges:
            gz.write(json.dumps(e) + chr(10))
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
