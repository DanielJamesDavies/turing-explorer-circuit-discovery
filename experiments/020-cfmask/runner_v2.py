"""cf-mask v2 (D3.2-lite) — 4-seed probe: warm-start + scale-normalized
pricing, aimed at the AMPC failure cells.

Seeds: 13/30053 (L4-mlp, AMPC 0.84@ceiling), 25/10628 (L8-mlp, small-K
wall), 27/6859 (L9-attn, identity failure), 35/6599 (L11-resid, healthy
control). Arms per seed (all objective="inject", scale_normalize=True,
zero floor, house steps/lr/anneal):

  warm-l{0,1,2}  delta_init = AMPC K=64 intervention (top direct-mass
                 latents at alpha* x posctx pin), lambda_inj in
                 {3e-4, 3e-3, 3e-2} (target-relative units)
  cold           no warm start, middle lambda_inj
  warm-x2        warm, middle lambda_inj, inject_exclude_sites=2
                 (mediated-drive ablation)

Eval per arm (held-out store negctx, D2.3's convention so rows are
comparable with AMPC's cf_alpha): binarised intervention — member
inhibitors removed (m=0), member activators injected at their learned
delta values — cf = (a_int - a_base) / (a_pos - a_base). Plus the
engine's train-split decomposition (p_gate/p_inject/p_both), delta
concentration, and the mechanism check: fraction of learned activators
inside direct-mass top-1024.

  PYTHONPATH=src python experiments/020-cfmask/runner.py
"""
import glob
import json
import math
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from eval.ablation_faithfulness import circuit_only_activation
from circuit.discovery.counterfactual_gradient import SeedPreActCapture
import gzip
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
K_WARM = 64
LAMBDAS = (3e-4, 3e-3, 3e-2)
D_SAE = 40960
torch.set_float32_matmul_precision("high")

SEEDS = [(13, 30053), (25, 10628), (27, 6859), (35, 6599)]

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
lm = config.discovery.learned_mask

# AMPC alpha* at K=64 per seed, from D1 rows
ALPHA64 = {}
for p in glob.glob(str(D1 / "rows_s*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if r.get("arm") == "C" and r.get("K") == 64 and r.get("alpha_star"):
            ALPHA64[r["seed"]] = float(r["alpha_star"])


class InjectPatcher:
    """Eval patcher: member inhibitors -> 0, member activators -> +delta
    (additive, position-uniform) — the learned intervention, binarised."""

    def __init__(self, zero_idx, add_vals, seed_site, seed_idx):
        self.zero_idx = zero_idx        # {(l,kind): [idx,...]}
        self.add_vals = add_vals        # {(l,kind): {idx: delta}}
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
        z = self.zero_idx.get((layer_idx, kind))
        a = self.add_vals.get((layer_idx, kind))
        if not z and not a:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        c_new = dense.clone()
        if z:
            zi = torch.tensor(sorted(z), device=dense.device, dtype=torch.long)
            c_new[..., zi] = 0.0
        if a:
            ai = torch.tensor(sorted(a), device=dense.device, dtype=torch.long)
            av = torch.tensor([a[int(i)] for i in ai], device=dense.device,
                              dtype=dense.dtype)
            c_new[..., ai] = c_new[..., ai] + av
        out = bank.decode(c_new - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


OUT = HERE / "rows2.jsonl"
fh = OUT.open("a")
done = set()
if OUT.exists():
    for line in open(OUT):
        if line.strip():
            r = json.loads(line)
            done.add((r["seed"], r["arm"]))

for sc_idx, sl in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
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
    a_pos_ev = float(measure_seed_activation(inference, bank, pt_ev, layer, kind,
                                             sl, pa_ev, batch_size=EVAL_BS))
    a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer, kind,
                                             sl, pa_tr, batch_size=EVAL_BS))
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)

    # direct-mass ranking (mechanism reference + warm-start members)
    dw = torch.load(D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl)),
                    map_location="cpu", weights_only=False)["direct"]
    triples = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(2048, w.numel()))
        triples += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
    triples.sort(key=lambda x: -x[0])
    rank_c = [(s, i) for _, s, i in triples]
    direct_top1024 = set(rank_c[:1024])

    alpha = ALPHA64.get(seed_key, 2.0)
    warm = {}
    n_w = 0
    for site, idx in rank_c:
        if n_w >= K_WARM:
            break
        pin = float(pins_c[site][idx]) if site in pins_c else 0.0
        if pin > 0:
            warm.setdefault(site, {})[idx] = alpha * pin
            n_w += 1

    print("\n[%s] L%d %s — %d sites | a_pos %.3f | warm %d latents @ a*%.2f"
          % (seed_key, layer, kind, len(up), a_pos_ev, n_w, alpha), flush=True)

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

    # closure-eval anchors: a_e0 (empty circuit) + negctx anchor means for
    # the freeN floor (negctx anchors = pre-act argmax on the neg split,
    # the restoration-negctx floor construction)
    a_e0_ev = float(circuit_only_activation(
        inference, bank, {}, up, pt_ev, layer, kind, sl,
        pos_argmax=pa_ev, batch_size=EVAL_BS))
    den_free = a_pos_ev - a_e0_ev
    sae_ = bank.saes[kind][layer]
    _cap = SeedPreActCapture(layer, kind,
                             sae_.encoder.weight[sl].detach(),
                             sae_._get_bias_eff()[sl].detach())
    inference.disable_compile()
    try:
        _pre_chunks = []
        with torch.no_grad():
            for s0 in range(0, int(nt_tr.shape[0]), EVAL_BS):
                _cap.seed_pre_act = None
                inference.forward(nt_tr[s0:s0 + EVAL_BS], patcher=_cap,
                                  grad_enabled=False,
                                  return_activations=False,
                                  tokenize_final=False)
                _pre_chunks.append(_cap.seed_pre_act.detach())
    finally:
        inference.enable_compile()
    na_tr = torch.cat(_pre_chunks, 0).argmax(dim=1).cpu()
    _, neg_means = collect_site_anchors(inference, bank, nt_tr, up, na_tr,
                                        pin_position_specific=False)

    ARMS = ([("warm-l%d" % i, dict(delta_init=warm, inject_lambda=lam))
             for i, lam in enumerate(LAMBDAS)]
            + [("cold", dict(delta_init=None, inject_lambda=LAMBDAS[1])),
               ("warm-x2", dict(delta_init=warm, inject_lambda=LAMBDAS[1],
                                inject_exclude_sites=2))])

    for arm, kw in ARMS:
        if (seed_key, arm) in done:
            continue
        t0 = time.time()
        try:
            scores, prov = run_learned_mask(
                inference, bank, objective="inject", sites=up_sorted,
                seed_layer=layer, seed_kind=kind, seed_latent_idx=sl,
                pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
                target_act=a_pos_tr, scale_normalize=True,
                mask_floor_source="zero",
                binarize=lm.binarize, steps=lm.steps, lr=lm.lr,
                l1_lambda=lm.l1_lambda, keep_threshold=lm.keep_threshold,
                batch_size=4, holdout_frac=lm.holdout_frac,
                theta_init=lm.theta_init, log_every=0,
                deep_site_threshold=lm.deep_site_threshold,
                deep_batch_size=lm.deep_batch_size,
                optimizer=lm.optimizer, weight_decay=lm.weight_decay,
                code_dtype=lm.code_dtype, lr_schedule=lm.lr_schedule,
                lr_min_frac=lm.lr_min_frac, warmup_frac=lm.warmup_frac,
                **kw)
        except Exception as exc:
            print("  %-8s ERROR %s: %s" % (arm, type(exc).__name__,
                                           str(exc)[:110]), flush=True)
            continue
        secs = round(time.time() - t0, 1)

        zero_idx, add_vals = {}, {}
        n_inh = n_act = 0
        for f, v in scores.items():
            site = (f.layer, f.kind)
            if v < 0:
                zero_idx.setdefault(site, []).append(f.index)
                n_inh += 1
            else:
                add_vals.setdefault(site, {})[f.index] = float(v)
                n_act += 1
        if n_inh + n_act:
            t1 = time.time()
            a_int = seed_act_under(
                InjectPatcher(zero_idx, add_vals, (layer, kind), sl),
                nt_ev, pa_ev)
            cf = round((a_int - a_base_ev) / den, 4) if abs(den) > 1e-9 else None
        else:
            a_int, cf = None, None
        # closure evals: members (both roles) kept live, rest ablated
        keep = {}
        for s, lst in zero_idx.items():
            keep.setdefault(s, set()).update(lst)
        for s, d in add_vals.items():
            keep.setdefault(s, set()).update(d)
        if keep and abs(den_free) > 1e-9:
            a_f0 = float(circuit_only_activation(
                inference, bank, keep, up, pt_ev, layer, kind, sl,
                pos_argmax=pa_ev, batch_size=EVAL_BS))
            free0 = round((a_f0 - a_e0_ev) / den_free, 4)
            a_fn = float(circuit_only_activation(
                inference, bank, keep, up, pt_ev, layer, kind, sl,
                pos_argmax=pa_ev, batch_size=EVAL_BS,
                site_means=neg_means, respect_topk=True))
            freen_topk = round((a_fn - a_e0_ev) / den_free, 4)
        else:
            free0 = freen_topk = None
        with gzip.open(HERE / ("members_%s_%d_%d.jsonl.gz"
                               % (arm, sc_idx, sl)), "wt",
                       encoding="utf-8") as gz:
            for f, v in scores.items():
                gz.write(json.dumps([f.layer, f.kind, f.index,
                                     round(float(v), 4)]) + chr(10))
        acts_all = [(s, i) for s, d in add_vals.items() for i in d]
        in_direct = (round(sum(1 for m in acts_all if m in direct_top1024)
                           / max(len(acts_all), 1), 4) if acts_all else None)
        row = {"seed": seed_key, "layer": layer, "kind": kind, "arm": arm,
               "inject_lambda": kw.get("inject_lambda"),
               "warm": kw.get("delta_init") is not None,
               "exclude": kw.get("inject_exclude_sites", 0),
               "n_inh": n_inh, "n_act": n_act,
               "cf": cf, "a_int": (round(a_int, 4) if a_int is not None else None),
               "p_both": prov.get("p_both"),
               "p_gate_only": prov.get("p_gate_only"),
               "p_inject_only": prov.get("p_inject_only"),
               "delta_sum": prov.get("delta_sum"),
               "delta_top1pct_share": prov.get("delta_top1pct_share"),
               "n_delta_gt_0p1": prov.get("n_delta_gt_0p1"),
               "act_in_direct1024": in_direct,
               "free0": free0, "freen_topk": freen_topk,
               "holdout_loss": prov.get("holdout_data_loss"),
               "secs": secs}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-8s cf=%-8s free0=%-8s freeN=%-8s n=%d+%d  dir1k=%s (%ss)"
              % (arm, cf, free0, freen_topk, n_inh, n_act, in_direct, secs),
              flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
fh.close()
