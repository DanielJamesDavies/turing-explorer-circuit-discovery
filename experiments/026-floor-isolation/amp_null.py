"""The null that decides whether ampF0 is defendable: can a RANDOM set
of the same size, given the same amplitude-fitting budget, also reach
ampF0 ~ 1?

If yes, the metric measures the FIT (any spanning set + enough free
coefficients reconstructs the seed) and must be retracted. If random
sets fail while discovered sets succeed, the amplitudes are a
re-weighting of a genuinely selected circuit and the metric stands.

Method: for each seed, take the discovered amp circuit's size n at the
decision cell (dual @1e-3), draw a random set of n latents from the LIVE
pool (latents active anywhere on the probe batch — the hardest fair
null; dead latents can carry no signal at any alpha), then fit ONLY the
amplitudes: gates frozen wide open on the random set (theta=+40),
closed elsewhere (theta=-40, excluded from the optimiser), psi trained
with the same objective/floor/steps/lr as the discovered run. Score
ampF0/ampFMd identically.

Three random draws per seed. COMP_IDX=8 (L2) and 29 (L9) both matter:
L2 has a small scope where spanning is plausible, L9 is where the claim
is largest.

  COMP_IDX=29 PYTHONPATH=src python experiments/026-floor-isolation/amp_null.py
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
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX = int(os.environ.get("COMP_IDX", 29))
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_DRAWS = int(os.environ.get("N_DRAWS", 3))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
LAM = 1e-3
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

LAYER, KI = split_component_idx(COMP_IDX, n_kinds)
KIND = bank.kinds[KI]
UP = sorted(upstream_sites(bank, LAYER, KIND))

_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
_pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == COMP_IDX]
random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:32])[:N_SEEDS]
del _cand
# exclude the vacuous-free0 seed at L9
if COMP_IDX == 29 and 1639 in SEEDS:
    SEEDS = [s for s in SEEDS if s != 1639][:N_SEEDS]
print("L%d %s | seeds %s | %d random draws each | lambda %g"
      % (LAYER, KIND, SEEDS, N_DRAWS, LAM), flush=True)


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


class AmpCircuitPatcher:
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


fh = (HERE / ("amp_null_c%d.jsonl" % COMP_IDX)).open("a")
for sl in SEEDS:
    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, pb)
    pd_ = m0.build_probe_dataset(COMP_IDX, sl)
    del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    nt = pd_.neg_tokens[:N_SEQ]
    sae = bank.saes[KIND][LAYER]
    w_seed = sae.encoder.weight[sl].detach()
    b_seed = sae._get_bias_eff()[sl].detach()
    a_pos = float(measure_seed_activation(inference, bank, pt, LAYER, KIND, sl,
                                          pa, batch_size=EVAL_BS))
    means = collect_site_means(inference, bank, pt, set(UP))
    e0 = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                       KIND, sl, pos_argmax=pa,
                                       batch_size=EVAL_BS))
    eMd = float(circuit_only_activation(inference, bank, {}, UP, pt, LAYER,
                                        KIND, sl, pos_argmax=pa,
                                        site_means=means, batch_size=EVAL_BS))
    # live pool over the probe batch
    lc = LiveCounter(UP)
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(pt.shape[0]), EVAL_BS):
                inference.forward(pt[s0:s0 + EVAL_BS], patcher=lc,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    live = [(s, i) for s in UP for i in lc.live[s].nonzero(as_tuple=True)[0].tolist()]
    print("\n[%d] a_pos %.3f | live pool %d" % (sl, a_pos, len(live)), flush=True)

    def amp_act(alphas, floors):
        p = AmpCircuitPatcher(alphas, floors, (LAYER, KIND), w_seed, b_seed)
        tot, n = 0.0, 0
        inference.disable_compile()
        try:
            for s0 in range(0, int(pt.shape[0]), EVAL_BS):
                tk = pt[s0:s0 + EVAL_BS]
                p.seed_pre = None
                inference.forward(tk, patcher=p, grad_enabled=False,
                                  return_activations=False, tokenize_final=False)
                pre = p.seed_pre
                B = min(pre.shape[0], pa[s0:s0 + EVAL_BS].shape[0])
                rows = torch.arange(B, device=pre.device)
                anc = pa[s0:s0 + EVAL_BS][:B].to(pre.device).clamp(0, pre.shape[1] - 1)
                tot += float(torch.relu(pre[:B][rows, anc]).sum()); n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    def fit_and_score(members, tag):
        support = {}
        for s, i in members:
            support.setdefault(s, []).append(i)
        support = {s: torch.tensor(v, dtype=torch.long) for s, v in support.items()}
        scores, prov = run_learned_mask(
            inference, bank, objective="pos", sites=UP,
            seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
            pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
            mask_floor_source="dual", dual_floor_weight=cfg.dual_floor_weight,
            support=support,                       # restrict to this set
            theta_init=40.0,                       # gates wide open on it
            l1_lambda=0.0,                         # NO pruning: fit alphas only
            free_amplitude=True,
            binarize="none", steps=cfg.steps, lr=cfg.lr,
            keep_threshold=cfg.keep_threshold,
            batch_size=disc.probe_batch_size, holdout_frac=cfg.holdout_frac,
            log_every=0, deep_site_threshold=cfg.deep_site_threshold,
            deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
            weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
            lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
            warmup_frac=cfg.warmup_frac)
        ak = prov.get("amp_kept") or {}
        alphas = {}
        for k, d in ak.items():
            lyr, knd = k.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
        n_mem = sum(len(v) for v in alphas.values())
        aw0 = amp_act(alphas, None)
        awM = amp_act(alphas, means)
        st = prov.get("amp_stats") or {}
        return {"n": n_mem,
                "ampF0": round((aw0 - e0) / (a_pos - e0), 4),
                "ampFMd": round((awM - eMd) / (a_pos - eMd), 4) if abs(a_pos - eMd) > 1e-9 else None,
                "alpha_p90": st.get("p90"), "alpha_max": st.get("max")}

    # discovered circuit at the decision cell
    t0 = time.time()
    disc_scores, disc_prov = run_learned_mask(
        inference, bank, objective="pos", sites=UP,
        seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source="dual", dual_floor_weight=cfg.dual_floor_weight,
        l1_lambda=LAM, free_amplitude=True, binarize=cfg.binarize,
        steps=cfg.steps, lr=cfg.lr, keep_threshold=cfg.keep_threshold,
        batch_size=disc.probe_batch_size, holdout_frac=cfg.holdout_frac,
        log_every=0, deep_site_threshold=cfg.deep_site_threshold,
        deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
        lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
        warmup_frac=cfg.warmup_frac)
    ak = disc_prov.get("amp_kept") or {}
    alphas = {}
    for k, d in ak.items():
        lyr, knd = k.split("/")
        alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
    n_disc = sum(len(v) for v in alphas.values())
    aw0 = amp_act(alphas, None); awM = amp_act(alphas, means)
    st = disc_prov.get("amp_stats") or {}
    row = {"latent": sl, "set": "discovered", "n": n_disc,
           "ampF0": round((aw0 - e0) / (a_pos - e0), 4),
           "ampFMd": round((awM - eMd) / (a_pos - eMd), 4) if abs(a_pos - eMd) > 1e-9 else None,
           "alpha_p90": st.get("p90"), "alpha_max": st.get("max"),
           "secs": round(time.time() - t0, 1)}
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print("  discovered n=%-6d ampF0=%-8s ampFMd=%-8s" % (n_disc, row["ampF0"],
                                                          row["ampFMd"]), flush=True)

    rng = random.Random(1000 + sl)
    for draw in range(N_DRAWS):
        t0 = time.time()
        members = rng.sample(live, min(n_disc, len(live)))
        r = fit_and_score(members, "random%d" % draw)
        r.update({"latent": sl, "set": "random%d" % draw,
                  "secs": round(time.time() - t0, 1)})
        fh.write(json.dumps(r) + "\n"); fh.flush()
        print("  random%d   n=%-6d ampF0=%-8s ampFMd=%-8s alpha_max=%s"
              % (draw, r["n"], r["ampF0"], r["ampFMd"], r["alpha_max"]), flush=True)
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
