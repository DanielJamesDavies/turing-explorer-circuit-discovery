"""cf-mask training dynamics — why does it win cf and lose free0?

Instruments run_learned_mask's new step_hook to record, for every one of
the 400 steps: data loss, penalty, lr, temperature, gradient norms
(theta and psi, split by whether the latent is currently a member /
a warm-start latent), mean m, membership size, per-step ADD/REMOVE
churn, and delta mass. Every SNAP_EVERY steps it additionally evaluates
the live circuit: held-out cf (binarised intervention), free0, and the
ALIGNMENT of current membership against four reference sets —

  direct  : direct-mass top-1024 (D1 saved weights) — the drive mechanism
  R_act   : restoration PA activators   (D2.2 archive)
  R_inh   : restoration PA inhibitors   (D2.2 archive)
  maskMF  : the D3.6 abl-mask (pos objective) member list — the CLOSURE
            object for the same seed
  coact   : the seed's top-coactivation neighbours (store)

Arms (one per run, chosen from the 020-cfmask probe):
  L4-mlp/L8-mlp  inject @ lambda_inj 3e-4 (their best)
  L9-attn/L11    inject @ lambda_inj 3e-3 (their best)
  L11-cold       inject, no warm start   (the diffuse-degeneracy case)
  L11-pos        the CLOSURE objective, same seed/recipe as D3.6's MF
                 (the contrast run: same engine, other semantics)

  PYTHONPATH=src python experiments/020-cfmask/runner.py
"""
import gzip
import json
import math
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.types.feature_id import FeatureID
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.top_coactivation import top_coactivation

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
D1 = HERE.parent / "012-driver-bakeoff"
D22 = HERE.parent / "019-roles-drivers"
D36 = HERE.parent / "018-maskrefine"
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
K_WARM = 64
SNAP_EVERY = 25
D_SAE = 40960
torch.set_float32_matmul_precision("high")

RUNS = [
    ("L4mlp",   (13, 30053), "inject", 3e-4, True),
    ("L8mlp",   (25, 10628), "inject", 3e-4, True),
    ("L9attn",  (27, 6859),  "inject", 3e-3, True),
    ("L11res",  (35, 6599),  "inject", 3e-3, True),
    ("L11cold", (35, 6599),  "inject", 3e-3, False),
    ("L11pos",  (35, 6599),  "pos",    None,  False),
]

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

ALPHA64 = {}
import glob
for p in glob.glob(str(D1 / "rows_s*.jsonl")):
    for line in open(p):
        r = json.loads(line)
        if r.get("arm") == "C" and r.get("K") == 64 and r.get("alpha_star"):
            ALPHA64[r["seed"]] = float(r["alpha_star"])


class InjectPatcher:
    def __init__(self, zero_idx, add_vals, seed_site, seed_idx):
        self.zero_idx, self.add_vals = zero_idx, add_vals
        self.seed_site, self.seed_idx = seed_site, seed_idx
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


for tag, (sc_idx, sl), objective, lam_inj, warm_on in RUNS:
    tpath = HERE / ("trace_%s.jsonl" % tag)
    if tpath.exists():
        print("[%s] exists — skip" % tag, flush=True)
        continue
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind)
    up_sorted = sorted(up)
    seed_key = "%d/%d" % (sc_idx, sl)

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
    a_e0_ev = float(circuit_only_activation(inference, bank, {}, up, pt_ev,
                                            layer, kind, sl, pos_argmax=pa_ev,
                                            batch_size=EVAL_BS))
    _, pins_c = collect_site_anchors(inference, bank, pt_ev, up, pa_ev,
                                     pin_position_specific=False)
    den_free = a_pos_ev - a_e0_ev

    # ---- reference sets -------------------------------------------------
    dw = torch.load(D1 / ("direct_full_%d_%d.pt" % (sc_idx, sl)),
                    map_location="cpu", weights_only=False)["direct"]
    tri = []
    for s, w in dw.items():
        v, ix = torch.topk(w, k=min(2048, w.numel()))
        tri += [(float(vv), s, int(ii)) for vv, ii in zip(v, ix)]
    tri.sort(key=lambda x: -x[0])
    rank_c = [(s, i) for _, s, i in tri]
    ref_direct = set(rank_c[:1024])

    ref_r_act, ref_r_inh = set(), set()
    rp = D22 / ("ranking_R_%d_%d.jsonl.gz" % (sc_idx, sl))
    if rp.exists():
        with gzip.open(rp, "rt", encoding="utf-8") as gz:
            for n, line in enumerate(gz):
                if n >= 20000:
                    break
                s_, l_, kd_, idx_, role_, rr_ = json.loads(line)
                (ref_r_inh if role_ == "counterfactual_inhibitor"
                 else ref_r_act).add(((l_, kd_), int(idx_)))

    ref_mask = set()
    mp = D36 / ("members_MF_%d_%d.jsonl.gz" % (sc_idx, sl))
    if mp.exists():
        with gzip.open(mp, "rt", encoding="utf-8") as gz:
            for line in gz:
                l_, kd_, idx_, m_ = json.loads(line)
                ref_mask.add(((l_, kd_), int(idx_)))

    ref_coact = set()
    try:
        gids = top_coactivation.top_indices[sc_idx, sl, :].tolist()
        ref_coact = {int(g) for g in gids if int(g) > 0}
    except Exception as exc:
        print("  coact unavailable: %s" % str(exc)[:60], flush=True)

    def gid_of(site, idx):
        return FeatureID(layer=site[0], kind=site[1],
                         index=idx).to_global_id(n_kinds, D_SAE, bank.kinds)

    warm = {}
    if warm_on:
        alpha = ALPHA64.get(seed_key, 2.0)
        n_w = 0
        for site, idx in rank_c:
            if n_w >= K_WARM:
                break
            pin = float(pins_c[site][idx]) if site in pins_c else 0.0
            if pin > 0:
                warm.setdefault(site, {})[idx] = alpha * pin
                n_w += 1
    warm_set = {(s, i) for s, d in warm.items() for i in d}

    print("\n[%s] %s L%d %s | obj=%s lam_inj=%s warm=%d | refs: direct %d, "
          "R %d/%d, maskMF %d, coact %d"
          % (tag, seed_key, layer, kind, objective, lam_inj, len(warm_set),
             len(ref_direct), len(ref_r_act), len(ref_r_inh), len(ref_mask),
             len(ref_coact)), flush=True)

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
    den_cf = a_pos_ev - a_base_ev

    tfh = tpath.open("w")
    state = {"prev": set(), "t0": time.time()}

    def members_now(ctx):
        """(inhibitors_by_site, activators_by_site, member_set) at the
        engine's own selection rule."""
        kt = ctx["keep_threshold"]
        zero_idx, add_vals, mem = {}, {}, set()
        with torch.no_grad():
            for site, th in ctx["thetas"].items():
                m = torch.sigmoid(th)
                if objective == "inject":
                    edit = 1.0 - m
                    idx = (edit > kt).nonzero(as_tuple=True)[0]
                    for i in idx.tolist():
                        zero_idx.setdefault(site, []).append(i)
                        mem.add((site, i))
                else:
                    idx = (m > kt).nonzero(as_tuple=True)[0]
                    for i in idx.tolist():
                        mem.add((site, i))
            for site, psi in ctx["deltas"].items():
                d = torch.nn.functional.softplus(psi)
                idx = (d > kt).nonzero(as_tuple=True)[0]
                vals = d[idx].tolist()
                for i, v in zip(idx.tolist(), vals):
                    add_vals.setdefault(site, {})[i] = v
                    mem.add((site, i))
        return zero_idx, add_vals, mem

    def hook(step, ctx):
        zero_idx, add_vals, mem = members_now(ctx)
        prev = state["prev"]
        added, removed = mem - prev, prev - mem
        state["prev"] = mem
        # gradient norms, split by membership / warm-start
        g = ctx["grads"] or {}
        def gnorm(d, subset=None):
            tot, n = 0.0, 0
            for site, gr in (d or {}).items():
                if gr is None:
                    continue
                if subset is None:
                    tot += float((gr ** 2).sum()); n += gr.numel()
                else:
                    ix = [i for (s, i) in subset if s == site]
                    if ix:
                        t = gr[torch.tensor(ix, device=gr.device,
                                            dtype=torch.long)]
                        tot += float((t ** 2).sum()); n += t.numel()
            return (math.sqrt(tot), n)
        gt_all, _ = gnorm(g.get("theta"))
        gp_all, _ = gnorm(g.get("psi"))
        gp_warm, nw = gnorm(g.get("psi"), warm_set) if warm_set else (0.0, 0)
        gt_mem, _ = gnorm(g.get("theta"), mem) if mem else (0.0, 0)
        with torch.no_grad():
            mm = float(torch.stack([torch.sigmoid(t).mean()
                                    for t in ctx["thetas"].values()]).mean())
            if ctx["deltas"]:
                dall = torch.cat([torch.nn.functional.softplus(p).flatten()
                                  for p in ctx["deltas"].values()])
                d_sum, d_max = float(dall.sum()), float(dall.max())
                d_warm = (float(sum(
                    float(torch.nn.functional.softplus(
                        ctx["deltas"][s][i])) for s, i in warm_set
                    if s in ctx["deltas"])) if warm_set else 0.0)
                d_sub = float(dall[(dall > 1e-3) & (dall <= ctx["keep_threshold"])].sum())
            else:
                d_sum = d_max = d_warm = d_sub = 0.0
        row = {"step": step, "data_loss": ctx["data_loss"],
               "penalty": ctx["penalty"], "lr": ctx["lr"],
               "temp": ctx["temperature"],
               "n_mem": len(mem), "n_inh": sum(len(v) for v in zero_idx.values()),
               "n_act": sum(len(v) for v in add_vals.values()),
               "added": len(added), "removed": len(removed),
               "mean_m": round(mm, 5),
               "g_theta": round(gt_all, 6), "g_theta_mem": round(gt_mem, 6),
               "g_psi": round(gp_all, 6), "g_psi_warm": round(gp_warm, 6),
               "d_sum": round(d_sum, 3), "d_max": round(d_max, 3),
               "d_warm_sum": round(d_warm, 3), "d_subthresh_sum": round(d_sub, 3),
               "warm_kept": (len(mem & warm_set) if warm_set else None)}
        if step % SNAP_EVERY == 0 or step == int(lm.steps) - 1:
            keep = {}
            for s, lst in zero_idx.items():
                keep.setdefault(s, set()).update(lst)
            for s, dd in add_vals.items():
                keep.setdefault(s, set()).update(dd)
            if mem:
                a_int = seed_act_under(
                    InjectPatcher(zero_idx, add_vals, (layer, kind), sl),
                    nt_ev, pa_ev)
                row["cf"] = (round((a_int - a_base_ev) / den_cf, 4)
                             if abs(den_cf) > 1e-9 else None)
                a_f0 = float(circuit_only_activation(
                    inference, bank, keep, up, pt_ev, layer, kind, sl,
                    pos_argmax=pa_ev, batch_size=EVAL_BS))
                row["free0"] = (round((a_f0 - a_e0_ev) / den_free, 4)
                                if abs(den_free) > 1e-9 else None)
                gids = {gid_of(s, i) for (s, i) in mem}
                row["al_direct"] = round(len(mem & ref_direct) / len(mem), 4)
                row["al_R_act"] = round(len(mem & ref_r_act) / len(mem), 4)
                row["al_R_inh"] = round(len(mem & ref_r_inh) / len(mem), 4)
                row["al_maskMF"] = (round(len(mem & ref_mask) / len(mem), 4)
                                    if ref_mask else None)
                row["al_coact"] = (round(len(gids & ref_coact) / len(gids), 4)
                                   if ref_coact else None)
            row["snapshot"] = True
        tfh.write(json.dumps(row) + "\n")
        if step % 50 == 0:
            print("   step %3d loss %.4f n=%d (+%d/-%d) g_th %.3g g_psi %.3g "
                  "d_sum %.1f cf=%s free0=%s"
                  % (step, ctx["data_loss"], len(mem), len(added), len(removed),
                     gt_all, gp_all, d_sum, row.get("cf"), row.get("free0")),
                  flush=True)

    kw = dict(objective=objective, sites=up_sorted, seed_layer=layer,
              seed_kind=kind, seed_latent_idx=sl,
              pos_tokens=pt_tr, pos_argmax=pa_tr, neg_tokens=nt_tr,
              binarize=lm.binarize, steps=lm.steps, lr=lm.lr,
              l1_lambda=lm.l1_lambda, keep_threshold=lm.keep_threshold,
              batch_size=4, holdout_frac=lm.holdout_frac,
              theta_init=lm.theta_init, log_every=0,
              deep_site_threshold=lm.deep_site_threshold,
              deep_batch_size=lm.deep_batch_size,
              optimizer=lm.optimizer, weight_decay=lm.weight_decay,
              code_dtype=lm.code_dtype, lr_schedule=lm.lr_schedule,
              lr_min_frac=lm.lr_min_frac, warmup_frac=lm.warmup_frac,
              step_hook=hook)
    if objective == "inject":
        kw.update(target_act=a_pos_tr, scale_normalize=True,
                  mask_floor_source="zero", inject_lambda=lam_inj,
                  delta_init=(warm if warm_on else None))
    else:
        kw.update(mask_floor_source=lm.mask_floor_source,
                  dual_floor_weight=lm.dual_floor_weight)
    t0 = time.time()
    scores, prov = run_learned_mask(inference, bank, **kw)
    tfh.close()
    summ = {"tag": tag, "seed": seed_key, "layer": layer, "kind": kind,
            "objective": objective, "inject_lambda": lam_inj,
            "warm": warm_on, "n_final": len(scores),
            "secs": round(time.time() - t0, 1),
            "holdout_loss": prov.get("holdout_data_loss"),
            "loss_final": prov.get("loss_final"),
            "p_both": prov.get("p_both"), "p_gate_only": prov.get("p_gate_only"),
            "p_inject_only": prov.get("p_inject_only"),
            "a_pos_ev": round(a_pos_ev, 4), "a_pos_tr": round(a_pos_tr, 4)}
    (HERE / ("summary_%s.json" % tag)).write_text(json.dumps(summ, indent=1))
    with gzip.open(HERE / ("members_%s.jsonl.gz" % tag), "wt",
                   encoding="utf-8") as gz:
        for f, v in scores.items():
            gz.write(json.dumps([f.layer, f.kind, f.index,
                                 round(float(v), 4)]) + chr(10))
    print("  done %s: %d members, %.0fs" % (tag, len(scores), summ["secs"]),
          flush=True)
    torch.cuda.empty_cache()

print("ALL DONE", flush=True)
