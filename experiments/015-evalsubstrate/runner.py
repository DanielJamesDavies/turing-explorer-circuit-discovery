"""Exam-substrate sensitivity: FIXED circuits, cf/sup re-evaluated on all
four selector negative modes (close, random, distant, fused), vs the
frozen exam's store rows (already on disk in D2.2/D2.3).

Sets per seed (no rediscovery):
  R64, R1024  R's top-K include, from the D2.2 archive
  AR16        16 latents at alpha* (from D2.3's rows) — the headline object
For each substrate: 16 negatives selected by mode (selection_seed fixed),
then evaluate_counterfactual_faithfulness for R-sets (cf AND sup — the
inhibitor sup-targets are negctx means, so sup is substrate-dependent
too), and the calibrated injection cf for AR16 (a_base re-measured per
substrate). a_pos / posctx side is fixed by the frozen split.

  PYTHONPATH=src python experiments/015-evalsubstrate/runner.py
"""
import gzip
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from utils.neg_context_selector import NegContextSelector

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D22 = HERE.parent / "019-roles-drivers"
D23 = HERE.parent / "013-amplitude-drivers"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
MODES = ("close", "random", "distant", "fused")
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(2, 19766), (8, 20333), (9, 38734), (13, 30053), (17, 38268),
         (20, 35678), (25, 10628), (26, 17432), (27, 6859), (29, 2753),
         (35, 6599)]

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

# alpha* per seed for AR16, from D2.3
ALPHA16 = {}
for line in open(D23 / "rows.jsonl"):
    r = json.loads(line)
    if r["arm"] == "AR" and r["K"] == 16 and r.get("alpha_star") is not None:
        ALPHA16[r["seed"]] = float(r["alpha_star"])

from store.context import mid_ctx, neg_ctx, top_ctx
from store.seq_repr import seq_repr
selector = NegContextSelector(inference, bank, loader, neg_ctx, seq_repr,
                              top_ctx, mid_ctx)
ncs = config.discovery.neg_context_selection


class InjectPatcher:
    def __init__(self, targets, seed_site, seed_idx):
        self.targets = targets
        self.seed_site = seed_site
        self.seed_idx = seed_idx
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


from store.circuits import Circuit, CircuitNode

OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["set"], r["neg_mode"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    if all((seed_key, st, m) in done for st in ("R64", "R1024", "AR16")
           for m in MODES):
        continue

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    pt_ev, pa_ev = pt[N_TR:], pa[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt[:N_TR], layer,
                                             kind, sl, pa[:N_TR],
                                             batch_size=EVAL_BS))
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)

    # sets from archives
    rank = []
    with gzip.open(D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl)), "rt",
                   encoding="utf-8") as gz:
        for line in gz:
            s, l, kd, idx, role, rr = json.loads(line)
            rank.append(((l, kd), int(idx), role))
            if len(rank) >= 1024:
                break
    sets = {"R64": rank[:64], "R1024": rank[:1024], "AR16": rank[:16]}
    alpha16 = ALPHA16.get(seed_key)

    print("\n[%s] L%d %s | a_pos %.3f | alpha16 %s"
          % (seed_key, layer, kind, a_pos_ev, alpha16), flush=True)

    def seed_act_under(patcher, tokens, argmax):
        if patcher is None:
            return float(measure_seed_activation(
                inference, bank, tokens, layer, kind, sl, argmax,
                batch_size=EVAL_BS))
        total, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_capture = None
                patcher.argmax_chunk = argmax[s0:s0 + EVAL_BS]
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
                total += float(patcher.seed_capture or 0.0) * tk.shape[0]
                n += int(tk.shape[0])
        finally:
            inference.enable_compile()
        return total / max(n, 1)

    for mode in MODES:
        if all((seed_key, st, mode) in done for st in sets):
            continue
        try:
            sel = selector.select(
                sc_idx, sl, mode, EVAL_BS, EVAL_BS,
                selection_seed=int(ncs.selection_seed),
                filter_batch_size=int(ncs.filter_batch_size),
                load_window_size=int(ncs.load_window_size))
        except Exception as exc:
            print("  [%s] selector error: %s" % (mode, str(exc)[:80]), flush=True)
            continue
        if sel is None or int(sel.tokens.shape[0]) == 0:
            print("  [%s] no negatives" % mode, flush=True)
            continue
        nt_sub = sel.tokens[:EVAL_BS]

        for st in ("R64", "R1024"):
            if (seed_key, st, mode) in done:
                continue
            c = Circuit(name="sub")
            for (l, kd), idx, role in sets[st]:
                c.add_node(CircuitNode(metadata={
                    "layer_idx": l, "kind": kd, "latent_idx": idx,
                    "role": role}))
            t0 = time.time()
            try:
                cf_v, sup_v = evaluate_counterfactual_faithfulness(
                    inference, bank, avg_acts, c, neg_tokens=nt_sub,
                    pos_tokens=pt_ev, seed_layer=layer, seed_kind=kind,
                    seed_latent_idx=sl, pos_argmax=pa_ev,
                    circuit_layers={l for (l, _), _, _ in sets[st]})
                cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
            except Exception as exc:
                print("    %s cf error: %s" % (st, str(exc)[:80]), flush=True)
                cf_v = sup_v = None
            row = {"seed": seed_key, "layer": layer, "kind": kind, "set": st,
                   "neg_mode": mode, "n_negs": int(nt_sub.shape[0]),
                   "cf": cf_v, "sup": sup_v,
                   "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  [%-7s] %-5s cf=%-8s sup=%-8s" % (mode, st, cf_v, sup_v),
                  flush=True)

        if (seed_key, "AR16", mode) not in done and alpha16 is not None:
            targets = {}
            for (site, idx, role) in sets["AR16"]:
                v = float(pins_c[site][idx]) if site in pins_c else 0.0
                if v > 0:
                    targets.setdefault(site, {})[idx] = alpha16 * v
            t0 = time.time()
            a_int = seed_act_under(InjectPatcher(targets, (layer, kind), sl),
                                   nt_sub, pa_ev)
            a_base = seed_act_under(None, nt_sub, pa_ev)
            den = a_pos_ev - a_base
            cf_a = round((a_int - a_base) / den, 4) if abs(den) > 1e-9 else None
            row = {"seed": seed_key, "layer": layer, "kind": kind,
                   "set": "AR16", "neg_mode": mode,
                   "n_negs": int(nt_sub.shape[0]), "alpha_star": alpha16,
                   "a_base": round(a_base, 4), "cf": cf_a, "sup": None,
                   "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  [%-7s] AR16  cf_a=%-8s a_base=%-8s" % (mode, cf_a,
                                                            row["a_base"]),
                  flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
