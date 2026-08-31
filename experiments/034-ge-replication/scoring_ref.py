"""The depth-stratified tri-amp panel — the run that takes the paper's
weighted-circuit claims from 3-4 seeds at two layers to a stratified
panel with held-out probes and a properly-drawn null.

Per seed (20 seeds: L2/L5/L8/L9/L11 resid, L3/L11 mlp, L5 attn, L7
resid, L8 mlp):

  ARMS   triamp400   triple floor + free amplitudes, 400 joint steps,
                     lambda 1e-3 (the compact claim)
         triamp100   100 steps at raised lambda (the drive claim; R20:
                     4x shallow / 2x deep)
         gate400     triple floor, gates only (the set-only reference)
  NULLS  amplitude-fitted random same-size sets (live pool), scored on
         ampF0/ampFMd/cf_amp — 10 draws at the L2/L9 headline cells,
         4 elsewhere.

HELD-OUT SPLIT (the paper's named protocol debt): 48 train / 16
held-out positive contexts (and negctx likewise). Everything fitted or
derived — membership, alphas, floors, pins — comes from TRAIN; every
reported metric is read on the HELD-OUT 16 (ampF0_tr kept for
continuity with R15's in-sample numbers).

Resumable: rows keyed (comp_idx, latent, arm) are skipped if present
in rows.jsonl. SMOKE=1 runs one seed of comp 8 with 1 null draw.

  PYTHONPATH=src python experiments/029-panel/runner.py
"""
import json
import os
import random
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
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from eval.floors import collect_site_anchors, collect_site_means
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
SMOKE = os.environ.get("SMOKE") == "1"
N_SEQ, N_TRAIN, EVAL_BS, D_SAE = 64, 48, 16, 40960
# (comp_idx, band, n_seeds, triple_w, lam100, n_null)
PANEL = [
    (8,  "L2 resid",  3, 0.10, 4e-3, 10),
    (10, "L3 mlp",    2, 0.10, 4e-3, 4),
    (15, "L5 attn",   2, 0.10, 4e-3, 4),
    (17, "L5 resid",  2, 0.10, 4e-3, 4),
    (23, "L7 resid",  2, 0.05, 2e-3, 4),
    (26, "L8 resid",  2, 0.05, 2e-3, 4),
    (25, "L8 mlp",    2, 0.05, 2e-3, 4),
    (29, "L9 resid",  3, 0.05, 2e-3, 10),
    (34, "L11 mlp",   2, 0.05, 2e-3, 4),
    (35, "L11 resid", 2, 0.05, 2e-3, 4),
]
if SMOKE:
    PANEL = [(8, "L2 resid", 1, 0.10, 4e-3, 1)]
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
n_kinds = len(bank.kinds)
avg_acts = torch.zeros((bank.n_layer * n_kinds, D_SAE), device=bank.device)
_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4
disc.position_aware = False
disc.floor_source = "posctx"
cfg = disc.learned_mask

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)

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


class AmpCircuitPatcher:
    """Members at alpha * live value, non-members at floor (zero when
    floors is None). Verbatim semantics from amp_null.py."""

    def __init__(self, alphas, floors, seed_site, w_seed, b_seed):
        self.alphas, self.floors = alphas, floors or {}
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
        al = self.alphas.get((layer_idx, kind))
        if al is None:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        fl = self.floors.get((layer_idx, kind))
        code = (fl.to(device=dense.device, dtype=dense.dtype)
                .expand_as(dense).clone() if fl is not None
                else torch.zeros_like(dense))
        if al:
            idx = torch.tensor(sorted(al), device=dense.device, dtype=torch.long)
            av = torch.tensor([al[int(i)] for i in idx], device=dense.device,
                              dtype=dense.dtype)
            code[..., idx] = dense[..., idx] * av
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class AmpInjectPatcher:
    """Members SET to alpha_i * pin_i in the otherwise-live stream.
    Verbatim semantics from amp_cfsup.py."""

    def __init__(self, inject, seed_site, w_seed, b_seed):
        self.inject = inject
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
        inj = self.inject.get((layer_idx, kind))
        if not inj:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
        code = dense.clone()
        idx = torch.tensor(sorted(inj), device=dense.device, dtype=torch.long)
        vals = torch.tensor([inj[int(i)] for i in idx], device=dense.device,
                            dtype=dense.dtype)
        code[..., idx] = vals
        out = bank.decode(code - dense, kind, layer_idx, add_bias=False)
        return x + out.to(x.dtype)


class LiveCounter:
    def __init__(self, sites):
        self.sites = set(sites)
        self.live = {s: torch.zeros(D_SAE, dtype=torch.bool) for s in sites}

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s in self.sites:
            ta, ti = bank.encode(x, kind, layer_idx)
            idx = ti.reshape(-1)[ta.reshape(-1) > 0].to(torch.long).cpu()
            self.live[s][idx] = True
        return x


for comp_idx, band, n_seeds, triple_w, lam100, n_null in PANEL:
    layer, ki = split_component_idx(comp_idx, n_kinds)
    kind = bank.kinds[ki]
    pool = [int(c["latent_idx"]) for c in _cand
            if int(c["comp_idx"]) == comp_idx]
    random.Random(42).shuffle(pool)
    seeds = sorted(pool[:32])
    if comp_idx == 29 and 1639 in seeds:
        seeds = [s for s in seeds if s != 1639]
    seeds = seeds[:n_seeds]
    print("\n=== %s | comp %d | seeds %s | w %g | nulls %d"
          % (band, comp_idx, seeds, triple_w, n_null), flush=True)

    for sl in seeds:
        if (comp_idx, sl, "null%d" % (n_null - 1)) in done:
            print("[%s %d] already complete, skipping" % (band, sl), flush=True)
            continue
        m0 = _build_mode_method("counterfactual_gradient", "local", inference,
                                bank, avg_acts, pb)
        try:
            pd_ = m0.build_probe_dataset(comp_idx, sl)
        except Exception as e:
            print("[%s %d] probes FAILED %s: %s" % (band, sl,
                  type(e).__name__, e), flush=True)
            del m0
            continue
        del m0
        if pd_ is None or int(pd_.pos_tokens.shape[0]) < N_SEQ:
            print("[%s %d] thin probes, skipping" % (band, sl), flush=True)
            continue
        pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
        nt = pd_.neg_tokens[:N_SEQ]
        pt_tr, pa_tr, nt_tr = pt[:N_TRAIN], pa[:N_TRAIN], nt[:N_TRAIN]
        pt_ho, pa_ho, nt_ho = pt[N_TRAIN:], pa[N_TRAIN:], nt[N_TRAIN:]

        sae = bank.saes[kind][layer]
        w_seed = sae.encoder.weight[sl].detach()
        b_seed = sae._get_bias_eff()[sl].detach()
        UP = sorted(upstream_sites(bank, layer, kind))
        a_pos_tr = float(measure_seed_activation(inference, bank, pt_tr, layer,
                                                 kind, sl, pa_tr,
                                                 batch_size=EVAL_BS))
        a_pos_ho = float(measure_seed_activation(inference, bank, pt_ho, layer,
                                                 kind, sl, pa_ho,
                                                 batch_size=EVAL_BS))
        if a_pos_ho < 0.05:
            print("[%s %d] held-out a_pos %.4f too small, skipping"
                  % (band, sl, a_pos_ho), flush=True)
            continue
        # everything derived comes from TRAIN
        means_tr, pins_tr = collect_site_anchors(inference, bank, pt_tr,
                                                 set(UP), pa_tr,
                                                 pin_position_specific=False)
        e0_tr = float(circuit_only_activation(inference, bank, {}, UP, pt_tr,
                                              layer, kind, sl, pos_argmax=pa_tr,
                                              batch_size=EVAL_BS))
        e0_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                              layer, kind, sl, pos_argmax=pa_ho,
                                              batch_size=EVAL_BS))
        eMd_ho = float(circuit_only_activation(inference, bank, {}, UP, pt_ho,
                                               layer, kind, sl, pos_argmax=pa_ho,
                                               site_means=means_tr,
                                               batch_size=EVAL_BS))
        # negctx anchors + baseline on the HELD-OUT negatives
        p0 = AmpInjectPatcher({}, (layer, kind), w_seed, b_seed)
        chunks = []
        inference.disable_compile()
        try:
            with torch.no_grad():
                for s0 in range(0, int(nt_ho.shape[0]), EVAL_BS):
                    p0.seed_pre = None
                    inference.forward(nt_ho[s0:s0 + EVAL_BS], patcher=p0,
                                      grad_enabled=False,
                                      return_activations=False,
                                      tokenize_final=False)
                    chunks.append(p0.seed_pre.detach())
        finally:
            inference.enable_compile()
        neg_pre = torch.cat(chunks, 0)
        na_ho = neg_pre.argmax(dim=1).cpu()
        a_base = float(torch.relu(
            neg_pre[torch.arange(neg_pre.shape[0], device=neg_pre.device),
                    na_ho.to(neg_pre.device)]).mean())
        vac_f0 = abs(a_pos_ho - e0_ho) < 0.05 * abs(e0_ho)
        print("[%s %d] a_pos tr %.3f ho %.3f | e0_ho %.3f | a_base %.3f%s"
              % (band, sl, a_pos_tr, a_pos_ho, e0_ho, a_base,
                 " | F0 VACUOUS" if vac_f0 else ""), flush=True)

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
                    pre = patcher.seed_pre
                    B = pre.shape[0]
                    rr = torch.arange(B, device=pre.device)
                    anc = anchors[s0:s0 + B].to(pre.device).clamp(
                        0, pre.shape[1] - 1)
                    tot += float(torch.relu(pre[rr, anc]).sum()); n += B
            finally:
                inference.enable_compile()
            return tot / max(n, 1)

        def fit(free_amp, steps, lam, support_members=None):
            kw = dict(sites=UP, seed_layer=layer, seed_kind=kind,
                      seed_latent_idx=sl, pos_tokens=pt_tr, pos_argmax=pa_tr,
                      neg_tokens=nt_tr, mask_floor_source="triple",
                      dual_floor_weight=cfg.dual_floor_weight,
                      triple_floor_weight=triple_w, free_amplitude=free_amp,
                      steps=int(steps), lr=cfg.lr,
                      keep_threshold=cfg.keep_threshold,
                      batch_size=disc.probe_batch_size,
                      holdout_frac=cfg.holdout_frac, log_every=0,
                      deep_site_threshold=cfg.deep_site_threshold,
                      deep_batch_size=cfg.deep_batch_size,
                      optimizer=cfg.optimizer, weight_decay=cfg.weight_decay,
                      code_dtype=cfg.code_dtype, lr_schedule=cfg.lr_schedule,
                      lr_min_frac=cfg.lr_min_frac, warmup_frac=cfg.warmup_frac)
            if support_members is None:
                kw.update(l1_lambda=lam, binarize=cfg.binarize,
                          theta_init=cfg.theta_init)
            else:
                support = {}
                for s, i in support_members:
                    support.setdefault(s, []).append(i)
                kw.update(l1_lambda=0.0, binarize="none", theta_init=40.0,
                          support={s: torch.tensor(v, dtype=torch.long)
                                   for s, v in support.items()})
            scores, prov = run_learned_mask(inference, bank, objective="pos",
                                            **kw)
            if free_amp:
                ak = prov.get("amp_kept") or {}
                alphas = {}
                for k, d in ak.items():
                    lyr, knd = k.split("/")
                    alphas[(int(lyr), knd)] = {int(i): float(v)
                                               for i, v in d.items()}
            else:
                alphas = {}
                for fid in scores:
                    alphas.setdefault((fid.layer, fid.kind), {})[
                        int(fid.index)] = 1.0
            return alphas, (prov.get("amp_stats") or {})

        def score(alphas, st, tag, secs):
            f0_tr = ((read(AmpCircuitPatcher(alphas, None, (layer, kind),
                                             w_seed, b_seed), pt_tr, pa_tr)
                      - e0_tr) / (a_pos_tr - e0_tr)
                     if abs(a_pos_tr - e0_tr) > 1e-9 else None)
            aw0 = read(AmpCircuitPatcher(alphas, None, (layer, kind),
                                         w_seed, b_seed), pt_ho, pa_ho)
            awM = read(AmpCircuitPatcher(alphas, means_tr, (layer, kind),
                                         w_seed, b_seed), pt_ho, pa_ho)
            inject = {s: {i: float(a * float(pins_tr[s][i]))
                          for i, a in d.items()}
                      for s, d in alphas.items() if s in pins_tr}
            cfa = read(AmpInjectPatcher(inject, (layer, kind), w_seed, b_seed),
                       nt_ho, na_ho)
            n_mem = sum(len(d) for d in alphas.values())
            circ = Circuit(name=tag)
            for (l, kd), dd in alphas.items():
                for i in dd:
                    circ.add_node(CircuitNode(metadata={
                        "layer_idx": l, "kind": kd, "latent_idx": int(i),
                        "role": "ablation_support"}))
            try:
                cf_bare, sup_v = evaluate_counterfactual_faithfulness(
                    inference, bank, avg_acts, circ, neg_tokens=nt_ho,
                    pos_tokens=pt_ho, seed_layer=layer, seed_kind=kind,
                    seed_latent_idx=sl, pos_argmax=pa_ho,
                    circuit_layers={l for (l, _) in alphas})
                cf_bare, sup_v = round(float(cf_bare), 4), round(float(sup_v), 4)
            except Exception as e:
                print("  cf_bare/sup failed: %s" % e, flush=True)
                cf_bare = sup_v = None
            row = {"comp_idx": comp_idx, "band": band, "latent": sl,
                   "arm": tag, "n": n_mem,
                   "ampF0_tr": round(f0_tr, 4) if f0_tr is not None else None,
                   "ampF0": (round((aw0 - e0_ho) / (a_pos_ho - e0_ho), 4)
                             if abs(a_pos_ho - e0_ho) > 1e-9 else None),
                   "ampFMd": (round((awM - eMd_ho) / (a_pos_ho - eMd_ho), 4)
                              if abs(a_pos_ho - eMd_ho) > 1e-9 else None),
                   "cf_amp": (round((cfa - a_base) / (a_pos_ho - a_base), 4)
                              if abs(a_pos_ho - a_base) > 1e-9 else None),
                   "cf_bare": cf_bare, "sup": sup_v,
                   "f0_vacuous": bool(vac_f0),
                   "alpha_med": st.get("median"), "alpha_p90": st.get("p90"),
                   "a_pos_ho": round(a_pos_ho, 3),
                   "secs": round(secs, 1)}
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print("  %-10s n=%-6d F0=%-8s FMd=%-8s cf_amp=%-8s cf_bare=%-8s "
                  "sup=%-8s" % (tag, n_mem, row["ampF0"], row["ampFMd"],
                                row["cf_amp"], cf_bare, sup_v), flush=True)
            return row

        arm_specs = [("triamp400", True, cfg.steps, 1e-3),
                     ("triamp100", True, 100, lam100),
                     ("gate400", False, cfg.steps, 1e-3)]
        n_ref = None
        ref_members = None
        for tag, fa, steps, lam in arm_specs:
            if (comp_idx, sl, tag) in done:
                continue
            t0 = time.time()
            alphas, st = fit(fa, steps, lam)
            r = score(alphas, st, tag, time.time() - t0)
            if tag == "triamp400":
                n_ref = r["n"]
                ref_members = [(s, i) for s, d in alphas.items() for i in d]

        if n_ref is None:
            # resumed: reconstruct the reference size from disk
            for line in ROWS_PATH.open():
                r = json.loads(line)
                if (r["comp_idx"], r["latent"], r["arm"]) == (comp_idx, sl,
                                                              "triamp400"):
                    n_ref = r["n"]
        if n_ref:
            lc = LiveCounter(UP)
            inference.disable_compile()
            try:
                with torch.no_grad():
                    for s0 in range(0, int(pt_tr.shape[0]), EVAL_BS):
                        inference.forward(pt_tr[s0:s0 + EVAL_BS], patcher=lc,
                                          grad_enabled=False,
                                          return_activations=False,
                                          tokenize_final=False)
            finally:
                inference.enable_compile()
            live = [(s, i) for s in UP
                    for i in lc.live[s].nonzero(as_tuple=True)[0].tolist()]
            rng = random.Random(1000 + sl)
            for draw in range(n_null):
                tag = "null%d" % draw
                # sample BEFORE the skip so resumed runs draw identical sets
                members = rng.sample(live, min(n_ref, len(live)))
                if (comp_idx, sl, tag) in done:
                    continue
                t0 = time.time()
                alphas, st = fit(True, cfg.steps, 0.0, support_members=members)
                score(alphas, st, tag, time.time() - t0)
        torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
