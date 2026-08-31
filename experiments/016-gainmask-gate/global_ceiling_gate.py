"""D3.7, the LAST untested variant: the GLOBAL topctx-max ceiling.

The 2026-08-01 gate refuted the capped gain-mask using an IN-MECHANISM
ceiling (each member's max activation on the SEED's own contexts, which
sits only 1.2-2.9x above its posctx mean — far below the 2.5-8x gains
that make small drivers work). Daniel's original proposal named a
LOOSER anchor: the latent's own GLOBAL topctx max, i.e. its peak over
the contexts where IT fires hardest, corpus-wide. That ceiling can be
far higher, so the gate deserves a second run before D3.7 is closed.

Method: for each of the AMPC K=64 members, load ITS OWN top contexts
(top_ctx) and measure its max activation there -> the m=2 endpoint.
Then re-run the gate: beta in [0,1] interpolating pin -> global ceiling,
fitted on train, cf on the frozen held-out exam. Compared against
uniform-alpha AMPC.

  PYTHONPATH=src python experiments/016-gainmask-gate/global_ceiling_gate.py
"""
import glob
import json
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import measure_seed_activation, upstream_sites
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.context import top_ctx

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
KS = (8, 16, 64)
D_SAE = 40960
N_CTX = 4          # own-top contexts per member latent
torch.set_float32_matmul_precision("high")

SEEDS = [(13, 30053), (25, 10628), (26, 17432), (35, 6599)]

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

ALPHA64 = {}
for p in glob.glob(str(D1 / "rows_s*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if r.get("arm") == "C" and r.get("K") == 64 and r.get("alpha_star"):
            ALPHA64[r["seed"]] = float(r["alpha_star"])


class MaxProbe:
    """Capture the max activation of ONE latent at ONE site."""

    def __init__(self, site, idx):
        self.site, self.idx = site, idx
        self.val = 0.0

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        if (layer_idx, kind) == self.site:
            ta, ti = bank.encode(x, kind, layer_idx)
            d = target_latent_activations(ta, ti, self.idx)
            self.val = max(self.val, float(d.max()))
        return x


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


OUT = HERE / "global_ceiling_rows.jsonl"
fh = OUT.open("a")
for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    pt_tr, pa_tr, nt_tr = pt[:N_TR], pa[:N_TR], nt[:N_TR]
    pt_ev, pa_ev, nt_ev = pt[N_TR:], pa[N_TR:], nt[N_TR:]
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    _, pins = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                   pin_position_specific=False)
    dw = torch.load(D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl)),
                    map_location="cpu", weights_only=False)["direct"]
    tri = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(512, w.numel()))
        tri += [(float(a), s, int(i)) for a, i in zip(v, ix)]
    tri.sort(key=lambda x: -x[0])
    rank_c = [(s, i) for _, s, i in tri]

    members = []
    for site, idx in rank_c:
        if len(members) >= max(KS):
            break
        p = float(pins[site][idx]) if site in pins else 0.0
        if p > 0:
            members.append((site, idx, p))

    # ---- GLOBAL ceilings: each member's max on ITS OWN top contexts ----
    t0 = time.time()
    ceilings = {}
    inference.disable_compile()
    try:
        for site, idx, p in members:
            comp = site[0] * n_kinds + bank.kinds.index(site[1])
            try:
                ids = [int(x) for x in top_ctx.ctx_seq_idx[comp, idx].tolist()
                       if int(x) > 0][:N_CTX]
            except Exception:
                ids = []
            if not ids:
                ceilings[(site, idx)] = p
                continue
            tk = probe_builder._load_all_ids(ids, max_length=64)
            mp = MaxProbe(site, idx)
            with torch.no_grad():
                inference.forward(tk.to(pt_ev.device), patcher=mp,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
            ceilings[(site, idx)] = max(mp.val, p)
    finally:
        inference.enable_compile()
    head = [(ceilings[(s, i)] / p) for s, i, p in members if p > 0]
    med_head = sorted(head)[len(head) // 2] if head else None
    print("\n[%s] L%d %s | %d members | GLOBAL ceiling/pin median %.1fx "
          "(%.0fs)" % (seed_key, layer, kind, len(members), med_head,
                       time.time() - t0), flush=True)

    def act_under(targets, tokens, argmax):
        p = InjectPatcher(targets, (layer, kind), sl)
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

    a_base = act_under({}, nt_ev, pa_ev)
    den = a_pos_ev - a_base

    for K in KS:
        mem = members[:K]
        if not mem:
            continue

        def targets_at(beta):
            t = {}
            for site, idx, p in mem:
                c = ceilings[(site, idx)]
                t.setdefault(site, {})[idx] = p + beta * (c - p)
            return t

        lo, hi = 0.0, 1.0
        if act_under(targets_at(1.0), nt_tr[:16], pa_tr[:16]) < a_pos_tr:
            beta = 1.0
        else:
            for _ in range(6):
                mid = (lo + hi) / 2
                if act_under(targets_at(mid), nt_tr[:16], pa_tr[:16]) < a_pos_tr:
                    lo = mid
                else:
                    hi = mid
            beta = (lo + hi) / 2
        a_int = act_under(targets_at(beta), nt_ev, pa_ev)
        cf_beta = round((a_int - a_base) / den, 4) if abs(den) > 1e-9 else None
        a_max = act_under(targets_at(1.0), nt_ev, pa_ev)
        cf_max = round((a_max - a_base) / den, 4) if abs(den) > 1e-9 else None
        row = {"seed": seed_key, "layer": layer, "kind": kind, "K": K,
               "n": len(mem), "median_ceiling_ratio": round(med_head, 3),
               "beta": round(beta, 3), "cf_beta": cf_beta, "cf_at_ceiling": cf_max,
               "ampc_alpha": ALPHA64.get(seed_key)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  K=%-3d beta=%-6s cf_beta=%-8s cf@ceiling=%-8s "
              "(AMPC alpha %s)" % (K, row["beta"], cf_beta, cf_max,
                                   row["ampc_alpha"]), flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
