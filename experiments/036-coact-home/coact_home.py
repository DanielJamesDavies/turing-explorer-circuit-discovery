"""RUNG 3 ON HOME TURF: the greedy co-activation baseline on TuringLLM
and the home SAE bank (k=128 / 40,960).

On both cross-architecture banks (Llama TopK-32 skip-transcoders,
GemmaScope JumpReLU) `coact_amp` — top-n latents by summed anchor
activation, size-matched to the discovered circuit, with amplitudes
fitted by the SAME machinery — landed at rough parity with the tri-amp
mask on faithfulness, while `coact_raw` (alpha=1) collapsed. This run
asks whether that holds on the original setting, where the discovered
circuits and their nulls are best understood (amp_null: random
same-size fitted sets fail by 3.6-20,000x at both depths).

Derived from amp_null.py (026-floor-isolation), which supplies
the support-restricted amplitude fit; the only new ingredient is the
SELECTION: per-latent activation mass at the anchor positions over the
probe batch, top-n across all upstream sites.

Sizes n are read from amp_null's logged "discovered" rows, so the
discovered fits are not re-run and the baseline is size-matched to the
same objects the null was.

Arms per seed:
  coact_raw   the selection at alpha = 1.0 (the method's own output)
  coact_amp   the selection with fitted amplitudes (same budget as ours)

  COMP_IDX=29 PYTHONPATH=src python experiments/036-coact-home/coact_home.py
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
NULL_DIR = HERE.parent / "026-floor-isolation"
COMP_IDX = int(os.environ.get("COMP_IDX", 29))
N_SEEDS = int(os.environ.get("N_SEEDS", 3))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
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

# the same seed list amp_null used (same pool RNG), and its logged sizes
_cand = torch.load(RUN_ROOT / "candidates.pt", map_location="cpu",
                   weights_only=False)
_pool = [int(c["latent_idx"]) for c in _cand if int(c["comp_idx"]) == COMP_IDX]
random.Random(42).shuffle(_pool)
SEEDS = sorted(_pool[:32])[:N_SEEDS]
del _cand
if COMP_IDX == 29 and 1639 in SEEDS:
    SEEDS = [s for s in SEEDS if s != 1639][:N_SEEDS]

N_DISC = {}
for line in (NULL_DIR / ("amp_null_c%d.jsonl" % COMP_IDX)).open():
    r = json.loads(line)
    if r.get("set") == "discovered":
        N_DISC[int(r["latent"])] = int(r["n"])
print("L%d %s | seeds %s | sizes %s"
      % (LAYER, KIND, SEEDS, {s: N_DISC.get(s) for s in SEEDS}), flush=True)


class AnchorMass:
    """Per-latent summed activation at the anchor positions, per site."""

    def __init__(self, sites):
        self.sites = set(sites)
        self.mass = {s: torch.zeros(D_SAE) for s in sites}
        self.anchors = None          # set per batch before forward

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        s = (layer_idx, kind)
        if s in self.sites:
            ta, ti = bank.encode(x, kind, layer_idx)
            dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=x.dtype)
            B = min(dense.shape[0], self.anchors.shape[0])
            rows = torch.arange(B, device=dense.device)
            anc = self.anchors[:B].to(dense.device).clamp(
                0, dense.shape[1] - 1)
            self.mass[s] += dense[rows, anc].sum(0).float().cpu()
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


fh = (HERE / ("coact_home_c%d.jsonl" % COMP_IDX)).open("a")
for sl in SEEDS:
    n_disc = N_DISC.get(sl)
    if not n_disc:
        print("[%d] no discovered size logged; skipping" % sl, flush=True)
        continue
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

    # SELECTION: anchor mass over the probe batch, top-n across all sites
    am = AnchorMass(UP)
    inference.disable_compile()
    try:
        with torch.no_grad():
            for s0 in range(0, int(pt.shape[0]), EVAL_BS):
                am.anchors = pa[s0:s0 + EVAL_BS]
                inference.forward(pt[s0:s0 + EVAL_BS], patcher=am,
                                  grad_enabled=False, return_activations=False,
                                  tokenize_final=False)
    finally:
        inference.enable_compile()
    flat = torch.cat([am.mass[s] for s in UP])
    top = flat.argsort(descending=True)[:n_disc]
    members = [(UP[int(i) // D_SAE], int(i) % D_SAE) for i in top.tolist()]
    n_anchor_active = int((flat > 0).sum())
    print("\n[%d] a_pos %.3f | n=%d | anchor-active pool %d"
          % (sl, a_pos, n_disc, n_anchor_active), flush=True)

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

    def score(alphas, tag, secs, extra=None):
        aw0 = amp_act(alphas, None)
        awM = amp_act(alphas, means)
        row = {"latent": sl, "set": tag,
               "n": sum(len(v) for v in alphas.values()),
               "ampF0": round((aw0 - e0) / (a_pos - e0), 4),
               "ampFMd": round((awM - eMd) / (a_pos - eMd), 4)
               if abs(a_pos - eMd) > 1e-9 else None,
               "secs": round(secs, 1)}
        row.update(extra or {})
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-10s n=%-6d ampF0=%-8s ampFMd=%-8s %s"
              % (tag, row["n"], row["ampF0"], row["ampFMd"],
                 "alpha_max=%s" % row.get("alpha_max") if extra else ""),
              flush=True)

    # coact_raw: the selection at alpha = 1.0
    t0 = time.time()
    raw = {}
    for s, i in members:
        raw.setdefault(s, {})[i] = 1.0
    score(raw, "coact_raw", time.time() - t0)

    # coact_amp: same selection, fitted amplitudes (amp_null's exact call)
    t0 = time.time()
    support = {}
    for s, i in members:
        support.setdefault(s, []).append(i)
    support = {s: torch.tensor(v, dtype=torch.long) for s, v in support.items()}
    scores, prov = run_learned_mask(
        inference, bank, objective="pos", sites=UP,
        seed_layer=LAYER, seed_kind=KIND, seed_latent_idx=sl,
        pos_tokens=pt, pos_argmax=pa, neg_tokens=nt,
        mask_floor_source="dual", dual_floor_weight=cfg.dual_floor_weight,
        support=support, theta_init=40.0, l1_lambda=0.0,
        free_amplitude=True, binarize="none", steps=cfg.steps, lr=cfg.lr,
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
    st = prov.get("amp_stats") or {}
    score(alphas, "coact_amp", time.time() - t0,
          {"alpha_p90": st.get("p90"), "alpha_max": st.get("max")})
    torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
