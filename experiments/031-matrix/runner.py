"""The definitive cross-method matrix — attribution arms on the panel's
22 seeds under the panel's held-out protocol, so every method row in the
paper's comparison shares seeds, probes, and evaluation semantics.

Arms run here (discovery on the 48 TRAIN contexts only; all metrics on
the 16 HELD-OUT contexts):

  abl_ig    ablation gradient, ig_mean, position-aware abs-p50
  cf_ig     counterfactual gradient, ig_mean, position-aware abs-p50
  resto     ablation gradient, restoration, rounds = seed's upstream
            site count (the July "rounds=sites" recipe), PA abs-p50

Each is scored at TWO operating points: its full discovered set, and
the global top-n_ref members by |attribution| where n_ref = the seed's
triamp400 size from 029-panel (the matched-size comparison).
The mask arms (gate400 / triamp400 / triamp100) join from the panel's
rows at analysis time. Scores: free0 / freeM_dense (members live at
natural amplitude, others zero / train-mean filled), cf_bare, sup,
plus discovery wall-clock (the cost column).

Resumable via rows.jsonl keys (comp_idx, latent, arm).

  PYTHONPATH=src python experiments/031-matrix/runner.py
"""
import json
import os
import time
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
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from store.circuits import Circuit, CircuitNode

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
PANEL_ROWS = HERE.parent / "029-panel" / "rows.jsonl"
SMOKE = os.environ.get("SMOKE") == "1"
N_SEQ, N_TRAIN, EVAL_BS, D_SAE = 64, 48, 16, 40960
torch.set_float32_matmul_precision("high")

# seed list + n_ref straight from the panel
panel = [json.loads(l) for l in PANEL_ROWS.open()]
SEEDS = []
for r in panel:
    if r["arm"] == "triamp400":
        SEEDS.append((r["comp_idx"], r["band"], r["latent"], r["n"]))
if SMOKE:
    SEEDS = SEEDS[:1]
print("matrix over %d panel seeds" % len(SEEDS), flush=True)

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=200000)
disc = config.discovery
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.floor_source = "posctx"
disc.position_aware = True
disc.position_aware_select = "abs_pctl"
disc.position_aware_threshold = 50.0

ROWS_PATH = HERE / "rows.jsonl"
done = set()
if ROWS_PATH.exists():
    for line in ROWS_PATH.open():
        try:
            r = json.loads(line)
            done.add((r["comp_idx"], r["latent"], r["arm"]))
        except Exception:
            pass
fh = ROWS_PATH.open("a")


class SetPatcher:
    """Members live at natural amplitude, non-members at floor (zero when
    floors is None) — panel AmpCircuitPatcher with alpha == 1."""

    def __init__(self, members, floors, seed_site, w_seed, b_seed):
        self.members, self.floors = members, floors or {}
        self.seed_site = seed_site
        self.w_seed, self.b_seed = w_seed, b_seed
        self.seed_pre = None

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.seed_site:
            w = self.w_seed.to(device=x.device, dtype=x.dtype)
            b = self.b_seed.to(device=x.device, dtype=x.dtype)
            self.seed_pre = x @ w + b
            return x
        mem = self.members.get((layer_idx, kind))
        if mem is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        fl = self.floors.get((layer_idx, kind))
        code = (fl.to(device=dense.device, dtype=dense.dtype)
                .expand_as(dense).clone() if fl is not None
                else torch.zeros_like(dense))
        if mem:
            idx = torch.tensor(sorted(mem), device=dense.device,
                               dtype=torch.long)
            code[..., idx] = dense[..., idx]
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


for comp_idx, band, sl, n_ref in SEEDS:
    if (comp_idx, sl, "resto@n") in done:
        print("[%s %d] complete, skipping" % (band, sl), flush=True)
        continue
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    # eval probes at 64, then discovery configured to see only the 48 train
    disc.probe_sequence_count = N_SEQ
    m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                            bank, avg_acts, pb)
    try:
        pd_ = m0.build_probe_dataset(comp_idx, sl)
    except Exception as e:
        print("[%s %d] probes FAILED %s" % (band, sl, e), flush=True)
        del m0
        continue
    del m0
    if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
        print("[%s %d] thin probes, skipping" % (band, sl), flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_ho, pa_ho, nt_ho = pt[N_TRAIN:], pa[N_TRAIN:], nt[N_TRAIN:]
    pt_tr = pt[:N_TRAIN]

    sae = bank.saes[kind][layer]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    UP = sorted(upstream_sites(bank, layer, kind))
    a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer,
                                             kind, sl, pa_ho,
                                             batch_size=EVAL_BS))
    if a_pos_ho < 0.05:
        print("[%s %d] held-out a_pos too small, skipping" % (band, sl),
              flush=True)
        continue
    means_tr = collect_site_means(inference, bank, pt_tr, set(UP))
    e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                          layer, kind, sl, pos_argmax=pa_ho,
                                          batch_size=EVAL_BS))
    eMd_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                           layer, kind, sl, pos_argmax=pa_ho,
                                           site_means=means_tr,
                                           batch_size=EVAL_BS))
    print("[%s %d] a_pos_ho %.3f | e0 %.3f eMd %.3f | n_ref %d"
          % (band, sl, a_pos_ho, e0_ho, eMd_ho, n_ref), flush=True)

    def read(patcher, tokens, anchors):
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(tokens.shape[0]), EVAL_BS):
                tk = tokens[s0:s0 + EVAL_BS]
                patcher.seed_pre = None
                inference.forward(tk, patcher=patcher, grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
            # per-chunk anchored read
                pre = patcher.seed_pre
                B = pre.shape[0]
                rr = torch.arange(B, device=pre.device)
                anc = anchors[s0:s0 + B].to(pre.device).clamp(
                    0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[rr, anc]).sum()); n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def score_set(members, tag, secs):
        n_mem = sum(len(v) for v in members.values())
        f0 = ((read(SetPatcher(members, None, (layer, kind), w_seed, b_seed),
                    pt_ho, pa_ho) - e0_ho) / (a_pos_ho - e0_ho)
              if abs(a_pos_ho - e0_ho) > 1e-9 else None)
        fM = ((read(SetPatcher(members, means_tr, (layer, kind), w_seed,
                               b_seed), pt_ho, pa_ho) - eMd_ho)
              / (a_pos_ho - eMd_ho)
              if abs(a_pos_ho - eMd_ho) > 1e-9 else None)
        circ = Circuit(name=tag)
        for (l, kd), idxs in members.items():
            for i in idxs:
                circ.add_node(CircuitNode(metadata={
                    "layer_idx": l, "kind": kd, "latent_idx": int(i),
                    "role": "ablation_support"}))
        try:
            cf_v, sup_v = evaluate_counterfactual_faithfulness(
                inference, bank, avg_acts, circ, neg_tokens=nt_ho,
                pos_tokens=pt_ho, seed_layer=layer, seed_kind=kind,
                seed_latent_idx=sl, pos_argmax=pa_ho,
                circuit_layers={l for (l, _) in members})
            cf_v, sup_v = round(float(cf_v), 4), round(float(sup_v), 4)
        except Exception as e:
            print("  cf/sup failed: %s" % e, flush=True)
            cf_v = sup_v = None
        row = {"comp_idx": comp_idx, "band": band, "latent": sl, "arm": tag,
               "n": n_mem,
               "free0": round(f0, 4) if f0 is not None else None,
               "freeM": round(fM, 4) if fM is not None else None,
               "cf": cf_v, "sup": sup_v, "secs": round(secs, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-10s n=%-7d free0=%-8s freeM=%-8s cf=%-8s sup=%-8s %.0fs"
              % (tag, n_mem, row["free0"], row["freeM"], cf_v, sup_v, secs),
              flush=True)

    def members_of(circuit, top_n=None):
        scored = []
        for node in circuit.nodes.values():
            fid = node.feature_id
            if fid is None:
                continue
            if (fid.layer, fid.kind, int(fid.index)) == (layer, kind, sl):
                continue
            s = node.metadata.get("attribution_score")
            scored.append((abs(float(s)) if s is not None else 0.0,
                           fid.layer, fid.kind, int(fid.index)))
        if top_n is not None:
            scored.sort(reverse=True)
            scored = scored[:top_n]
        members = {}
        for _, l, kd, i in scored:
            members.setdefault((l, kd), set()).add(i)
        return members

    ARMS = [("abl_ig", "ablation_gradient", "ig_mean"),
            ("cf_ig", "counterfactual_gradient", "ig_mean"),
            ("resto", "ablation_gradient", "restoration")]
    for tag, mname, mode in ARMS:
        if (comp_idx, sl, tag + "@n") in done:
            continue
        disc.probe_sequence_count = N_TRAIN     # discover on TRAIN only
        saved_rounds = disc.ablation_gradient.restoration.rounds
        saved_k = disc.ablation_gradient.restoration.per_round_k
        if tag == "resto":
            disc.ablation_gradient.restoration.rounds = len(UP)
            disc.ablation_gradient.restoration.per_round_k = 128
        t0 = time.time()
        try:
            meth = _build_mode_method(mname, mode, inference, bank, avg_acts,
                                      pb)
            circuit = meth.discover(comp_idx, sl)
            del meth
        except Exception as e:
            print("  %s FAILED %s: %s" % (tag, type(e).__name__, e),
                  flush=True)
            circuit = None
        finally:
            disc.ablation_gradient.restoration.rounds = saved_rounds
            disc.ablation_gradient.restoration.per_round_k = saved_k
            disc.probe_sequence_count = N_SEQ
        if circuit is None or len(circuit.nodes) == 0:
            fh.write(json.dumps({"comp_idx": comp_idx, "band": band,
                                 "latent": sl, "arm": tag + "@n",
                                 "n": 0, "free0": None, "freeM": None,
                                 "cf": None, "sup": None,
                                 "secs": round(time.time() - t0, 1),
                                 "error": "no_circuit"}) + "\n")
            fh.flush()
            continue
        disc_secs = time.time() - t0
        if (comp_idx, sl, tag) not in done:
            score_set(members_of(circuit), tag, disc_secs)
        score_set(members_of(circuit, top_n=n_ref), tag + "@n", disc_secs)
        del circuit
        torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
