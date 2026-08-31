"""D3.7 gate test — does PER-LATENT amplitude headroom beat uniform alpha?

Daniel's gain-mask proposal (m in [0,2]; 1..2 interpolates current ->
topctx-max) justified only if per-latent ceilings can do what uniform
alpha cannot. Zero new machinery: the AMPC member sets (D1 direct-mass
top-K, same pin>0 inclusion rule as AMPC so sets are IDENTICAL), with
per-latent value anchors measured as each member's MAX activation over
the seed's own train posctx probes (in-distribution ceiling; the m=2
endpoint). Arms per (seed, K in {8,16,64}):

  MAXINJ  every member at its ceiling (beta=1, no fitting)
  BETA    one scalar beta in [0,1] bisected on train to hit a_pos:
          value = pin + beta*(pmax - pin). Same fitted-param count as
          AMPC but per-latent-SHAPED. beta=1 censoring means "not
          reachable inside the latents' own posctx range".

Reference: AMPC's uniform-alpha cf_alpha from D1/D2.3 rows (on disk).
cf on the held-out split as everywhere.

  PYTHONPATH=src python experiments/016-gainmask-gate/runner.py
"""
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
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
KS = (8, 16, 64)
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


class MaxCapture:
    """Per-site scatter-max of top-k codes over (batch, pos): the ceiling."""

    def __init__(self, sites):
        self.pmax = {s: torch.zeros(D_SAE, device=device) for s in sites}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        site = (layer_idx, kind)
        if site in self.pmax:
            ta, ti = bank.encode(x, kind, layer_idx)
            self.pmax[site].scatter_reduce_(
                0, ti.reshape(-1).to(device=device, dtype=torch.long),
                ta.reshape(-1).to(device=device, dtype=self.pmax[site].dtype),
                reduce="amax")
        return x


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


OUT = HERE / "rows.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"], r["K"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    if all((seed_key, a, K) in done for a in ("MAXINJ", "BETA") for K in KS):
        continue

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    if pd_.pos_tokens.shape[0] == 0:
        print("[%s] no positives — skip" % seed_key, flush=True)
        continue
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)

    # per-latent ceilings on the TRAIN posctx probes
    cap = MaxCapture(up)
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(pt_tr.shape[0]), EVAL_BS):
                inference.forward(pt_tr[s0:s0 + EVAL_BS], patcher=cap,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    pmax = {s: v.cpu() for s, v in cap.pmax.items()}

    dw = torch.load(D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl)),
                    map_location="cpu", weights_only=False)["direct"]
    triples = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(64, w.numel()))
        triples += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
    triples.sort(key=lambda x: -x[0])
    rank_c = [(site, idx) for _, site, idx in triples]

    print("\n[%s] L%d %s | a_pos %.3f" % (seed_key, layer, kind, a_pos_ev),
          flush=True)

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

    a_base_ev = seed_act_under(None, nt_ev, pa_ev)
    den = a_pos_ev - a_base_ev

    for K in KS:
        # AMPC's inclusion rule (pin > 0) so the SETS match AMPC exactly
        members = []
        for site, idx in rank_c[:K]:
            pin = float(pins_c[site][idx]) if site in pins_c else 0.0
            if pin > 0:
                members.append((site, idx, pin, float(pmax[site][idx])))
        if not members:
            continue
        headroom = [(mx / pin) for _, _, pin, mx in members if pin > 0]
        med_head = sorted(headroom)[len(headroom) // 2]

        def targets_at(beta):
            t = {}
            for site, idx, pin, mx in members:
                t.setdefault(site, {})[idx] = pin + beta * (mx - pin)
            return t

        def act_at(beta, tokens, argmax):
            return seed_act_under(
                InjectPatcher(targets_at(beta), (layer, kind), sl),
                tokens, argmax)

        for arm in ("MAXINJ", "BETA"):
            if (seed_key, arm, K) in done:
                continue
            t0 = time.time()
            if arm == "MAXINJ":
                beta = 1.0
            else:
                lo, hi = 0.0, 1.0
                if act_at(1.0, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                    beta = 1.0        # censored at the in-distribution cap
                else:
                    for _ in range(6):
                        mid = (lo + hi) / 2
                        if act_at(mid, nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                            lo = mid
                        else:
                            hi = mid
                    beta = (lo + hi) / 2
            a_int = act_at(beta, nt_ev, pa_ev)
            cf = round((a_int - a_base_ev) / den, 4) if abs(den) > 1e-9 else None
            row = {"seed": seed_key, "layer": layer, "kind": kind, "arm": arm,
                   "K": K, "n_injectable": len(members),
                   "beta": round(beta, 3), "med_headroom": round(med_head, 2),
                   "cf": cf, "secs": round(time.time() - t0, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-6s K=%3d beta=%-5s cf=%-8s (med ceiling/pin %.1fx, "
                  "%d inj)" % (arm, K, row["beta"], cf, med_head, len(members)),
                  flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
