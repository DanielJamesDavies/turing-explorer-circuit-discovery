"""Full eval matrix for the depth-3 recursive latent maps.

Columns (all vs a size-matched random control):
  free0        members natural, everything else ablated to zero
  freeN_topk   members natural, rest at the negctx floor, top-k respected
  pin0         members pinned to clean posctx values
  cf_raw       members injected at posctx means (NO alpha fit)
  cf_alpha     members injected at alpha* (fit on train)
  sup          members zeroed on posctx / inhibitors injected (phi-sup)
  raise        members SILENCED on posctx -> does the seed rise?
  drop_x4      members amplified 4x on posctx -> does the seed fall?
Plus the pre-activation read alongside the censored post-top-k one.

Node sets are reconstructed from edges_*.jsonl.gz (children at depth<=3).

  PYTHONPATH=src python experiments/023-recursive-map/full_matrix.py
"""
import gzip
import json
import random
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.discovery.counterfactual_gradient import SeedPreActCapture
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors, collect_site_means
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
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
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


class SetPatcher:
    """values: {(layer,kind): {idx: value}}; captures the seed."""

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


fh = (HERE / "full_matrix_v2.jsonl").open("w")
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    seed_site = (layer, kind)
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)

    ep = HERE / ("edges_%d_%d.jsonl.gz" % (sc_idx, sl))
    if not ep.exists():
        print("missing %s" % ep.name, flush=True); continue
    nodes = set()
    with gzip.open(ep, "rt", encoding="utf-8") as gz:
        for line in gz:
            psite, pidx, csite, cidx, w, d = json.loads(line)
            nodes.add(((csite[0], csite[1]), int(cidx)))
    nodes = sorted(n for n in nodes if n[0] in up)

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
    # negctx floor for freeN
    sae0 = bank.saes[kind][layer]
    cap = SeedPreActCapture(layer, kind, sae0.encoder.weight[sl].detach(),
                            sae0._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        ch = []
        with torch.no_grad():
            for s0 in range(0, int(nt_tr.shape[0]), EVAL_BS):
                cap.seed_pre_act = None
                inference.forward(nt_tr[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
                ch.append(cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    na_tr = torch.cat(ch, 0).argmax(dim=1).cpu()
    _, neg_means = collect_site_anchors(inference, bank, nt_tr, up, na_tr,
                                        pin_position_specific=False)
    # freeM floor: MEAN dense latent vector per site over a broad corpus
    # draw (the SFC-style mean-ablation baseline), dense and top-k variants.
    _rngm = random.Random(77)
    _ids = [_rngm.randrange(1, len(loader)) for _ in range(32)]
    _corpus = probe_builder._load_all_ids(_ids, max_length=64).to(pt_ev.device)
    mean_floor = collect_site_means(inference, bank, _corpus, up)

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

    def score(mem, tag):
        keep = {}
        for s, i in mem:
            keep.setdefault(s, set()).add(i)
        f0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS))
        f0p = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, preact=True))
        fn = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=neg_means, respect_topk=True))
        fn_d = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, site_means=neg_means))
        fm_d = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, site_means=mean_floor))
        fm_t = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS,
            site_means=mean_floor, respect_topk=True))
        p0 = float(circuit_only_activation(
            inference, bank, keep, up, pt_ev, layer, kind, sl,
            pos_argmax=pa_ev, batch_size=EVAL_BS, pin_values=pins))
        # cf_raw: inject at posctx means, no alpha
        tg = {}
        for s, i in mem:
            v = float(pins[s][i]) if s in pins else 0.0
            if v > 0:
                tg.setdefault(s, {})[i] = v
        cf_raw = (round((act(tg, nt_ev, pa_ev) - a_base) / den_cf, 4)
                  if tg and abs(den_cf) > 1e-9 else None)
        # cf_alpha
        cf_a, al = None, None
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
        # sup
        c = Circuit(name="rec")
        for (l, kd), idx in mem:
            c.add_node(CircuitNode(metadata={"layer_idx": l, "kind": kd,
                                             "latent_idx": idx,
                                             "role": "ablation_support"}))
        try:
            _cfv, supv = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, c, neg_tokens=nt_ev, pos_tokens=pt_ev,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_argmax=pa_ev, circuit_layers={l for (l, _), _ in mem})
            supv = round(float(supv), 4)
        except Exception:
            supv = None
        # brake-native: silence / amplify on POSCTX
        sil = act({s: {i: 0.0 for (ss, i) in mem if ss == s} for s, _ in mem},
                  pt_ev, pa_ev)
        amp = act({s: {i: 4.0 * float(pins[s][i]) for (ss, i) in mem if ss == s}
                   for s, _ in mem}, pt_ev, pa_ev)
        row = {"seed": seed_key, "layer": layer, "kind": kind, "set": tag,
               "n": len(mem),
               "free0": round((f0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "free0_pre_raw": round(f0p, 3), "a_pos": round(a_pos, 4),
               "freeN_topk": round((fn - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "freeN_dense": round((fn_d - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "freeM_dense": round((fm_d - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "freeM_topk": round((fm_t - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "pin0": round((p0 - a_e0) / den_free, 4) if abs(den_free) > 1e-9 else None,
               "cf_raw": cf_raw, "cf_alpha": cf_a, "alpha": al, "sup": supv,
               "raise": round((sil - a_pos) / max(a_pos, 1e-9), 4),
               "drop_x4": round((a_pos - amp) / max(a_pos, 1e-9), 4)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-9s n=%-4d free0=%-7s fM=%-7s fM_tk=%-7s fN=%-7s fN_tk=%-7s "
              "pin0=%-6s cf_raw=%-7s cf_a=%-7s sup=%-7s raise=%-7s drop4=%s"
              % (tag, len(mem), row["free0"], row["freeM_dense"],
                 row["freeM_topk"], row["freeN_dense"], row["freeN_topk"],
                 row["pin0"], cf_raw, cf_a, supv, row["raise"],
                 row["drop_x4"]), flush=True)

    print("\n[%s] L%d %s | recursive map n=%d | a_pos %.3f"
          % (seed_key, layer, kind, len(nodes), a_pos), flush=True)
    rng = random.Random(19)
    score(nodes, "recursive")
    score([(up_sorted[rng.randrange(len(up_sorted))], rng.randrange(D_SAE))
           for _ in range(len(nodes))], "random")
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
