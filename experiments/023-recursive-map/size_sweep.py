"""Recursive map SIZE SWEEP — does free0 ever rise?

Pushes the recursive latent map to large node counts by widening the
branch and deepening, then back down, recording the full closure/drive
matrix at every size. Critically, every recursive set is compared to a
SIZE-MATCHED FLAT baseline: the top-n latents by value-edge weight taken
DIRECTLY from the seed (no recursion). If recursion buys nothing for
closure, the two curves coincide and the recursive structure is only a
drive/necessity object; if recursion wins, the compositional structure
is doing work the flat ranking cannot.

Grid: (branch, depth) in {(8,3), (32,3), (128,2), (128,3), (512,2)}
with an expansion cap for runtime. Random control at every size.

  PYTHONPATH=src python experiments/023-recursive-map/size_sweep.py
"""
import json
import os
import random
import time
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
N_SEQ, N_TR, EVAL_BS, GRAD_B = 64, 48, 16, 8
D_SAE = 40960
MAX_EXPAND = int(os.environ.get("MAX_EXPAND", 260))
GRID = [(8, 3), (32, 3), (128, 2), (128, 3), (512, 2)]
torch.set_float32_matmul_precision("high")
SEEDS = [(8, 20333), (26, 17432), (35, 6599)]

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


def site_rank(s):
    return ALL_SITES.index(s)


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


fh = (HERE / "size_sweep.jsonl").open("a")
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    seed_site = (layer, kind)
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                          sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference, bank, {}, up, pt_ev, layer,
                                         kind, sl, pos_argmax=pa_ev,
                                         batch_size=EVAL_BS))
    den_free = a_pos - a_e0
    _, pins = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                   pin_position_specific=False)
    print("\n[%s] L%d %s | %d sites | a_pos %.3f" % (seed_key, layer, kind,
                                                     len(up), a_pos), flush=True)

    def act(values, tokens, argmax):
        p = SetPatcher(values, seed_site, sl)
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

    a_base = act({}, nt_ev, pa_ev)
    den_cf = a_pos - a_base

    def expand(branch, depth):
        """BFS with value-edges; returns (nodes, flat_rank_from_seed)."""
        visited, by_depth, flat = set(), {0: [(seed_site, sl)]}, []
        n_exp = 0
        for d in range(1, depth + 1):
            frontier = [x for x in by_depth[d - 1] if x not in visited]
            frontier = frontier[:max(0, MAX_EXPAND - n_exp)]
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
            new = []
            for (nsite, nidx) in frontier:
                if nsite not in graph.activations:
                    continue
                _, conn, _ = graph.get_latents(*nsite)
                v = conn.act[..., nidx]
                B = min(v.shape[0], pa_tr.shape[0])
                rows = torch.arange(B, device=v.device)
                val = v[:B][rows, pa_tr[:B].to(v.device)
                            .clamp(0, v.shape[1] - 1)].mean()
                grads = torch.autograd.grad(val, anchors, allow_unused=True,
                                            retain_graph=True)
                is_seed = (nsite, nidx) == (seed_site, sl)
                # the seed's own ranking is kept DEEP so the flat baseline can
                # be matched to arbitrarily large recursive sets
                per_site = 8192 if is_seed else branch
                cand = []
                for s, a, g in zip(avail, anchors, grads):
                    if g is None or site_rank(s) >= site_rank(nsite):
                        continue
                    w = (g * a.detach()).sum(dim=1).mean(dim=0).abs().float().cpu()
                    kk = min(per_site, w.numel())
                    vv, ix = torch.topk(w, k=kk)
                    cand += [(float(x), s, int(i)) for x, i in zip(vv, ix)]
                cand.sort(key=lambda x: -x[0])
                picked = cand[:branch]
                if is_seed:
                    flat.extend([(s, i) for _, s, i in cand])   # deep ranking
                for _, s, i in picked:
                    new.append((s, i))
                visited.add((nsite, nidx))
                n_exp += 1
            instrument.release()
            del instrument, graph, anchors
            torch.cuda.empty_cache()
            seen, uniq = set(), []
            allprev = {x for lv in by_depth for x in by_depth[lv]}
            for x in new:
                if x not in seen and x not in allprev:
                    seen.add(x); uniq.append(x)
            by_depth[d] = uniq
        nodes = set()
        for d in by_depth:
            if d >= 1:
                nodes |= set(by_depth[d])
        return sorted(nodes), flat, n_exp

    def score(mem, tag, branch, depth, n_exp):
        mem = [m for m in mem if m[0] in up]
        if not mem:
            return
        keep = {}
        for s, i in mem:
            keep.setdefault(s, set()).add(i)
        f0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        f0p = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, preact=True))
        p0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins))
        tg = {}
        for s, i in mem:
            v = float(pins[s][i]) if s in pins else 0.0
            if v > 0:
                tg.setdefault(s, {})[i] = v
        cf_a = al = None
        if tg:
            base = {s: dict(t) for s, t in tg.items()}

            def at(alpha, tk, am):
                return act({s: {i: alpha * v for i, v in t.items()}
                            for s, t in base.items()}, tk, am)
            lo, hi = 0.25, 8.0
            if at(hi, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                al = hi
            elif at(lo, nt_tr[:16], pa_tr[:16]) > a_pos_tr:
                al = lo
            else:
                for _ in range(6):
                    mid = (lo + hi) / 2
                    if at(mid, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                        lo = mid
                    else:
                        hi = mid
                al = (lo + hi) / 2
            cf_a = round((at(al, nt_ev, pa_ev) - a_base) / den_cf, 4)
        c = Circuit(name="sw")
        for (l, kd), idx in mem:
            c.add_node(CircuitNode(metadata={"layer_idx": l, "kind": kd,
                                             "latent_idx": idx,
                                             "role": "ablation_support"}))
        try:
            _cf, supv = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev, circuit_layers={l for (l, _), _ in mem})
            supv = round(float(supv), 4)
        except Exception:
            supv = None
        row = {"seed": seed_key, "layer": layer, "kind": kind, "set": tag,
               "branch": branch, "depth": depth, "n": len(mem),
               "expanded": n_exp,
               "pct_dict": round(100.0 * len(mem) / (len(up) * D_SAE), 4),
               "free0": round((f0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "free0_pre_raw": round(f0p, 2),
               "pin0": round((p0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "cf_alpha": cf_a, "alpha": al, "sup": supv}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  b%-4d d%d %-10s n=%-6d free0=%-8s pin0=%-8s cf_a=%-8s sup=%-8s"
              % (branch, depth, tag, len(mem), row["free0"], row["pin0"],
                 cf_a, supv), flush=True)

    rng = random.Random(43)
    for branch, depth in GRID:
        t0 = time.time()
        nodes, flat, n_exp = expand(branch, depth)
        if not nodes:
            continue
        score(nodes, "recursive", branch, depth, n_exp)
        # matched-size FLAT baseline: top-n straight from the seed
        seen, flat_u = set(), []
        for x in flat:
            if x not in seen:
                seen.add(x); flat_u.append(x)
        score(flat_u[:len(nodes)], "flat_matched", branch, depth, n_exp)
        score([(up_sorted[rng.randrange(len(up_sorted))], rng.randrange(D_SAE))
               for _ in range(len(nodes))], "random", branch, depth, n_exp)
        print("    (%.0fs, expanded %d)" % (time.time() - t0, n_exp), flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
