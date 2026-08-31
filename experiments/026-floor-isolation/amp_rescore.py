"""Re-score the amp arms WITH their learned amplitudes applied.

The standard panel evaluates membership at NATURAL values, which throws
away exactly what free_amplitude learned — at lambda=1e-3 the mask kept
206 latents with 42% of them elevated (median alpha 1.05, p90 1.65), and
the set-only eval reads free0 0.058 because the compensation is missing.

This rescoring applies the archived per-member alphas: kept latents fire
at alpha * (their live value), non-members at the floor. That is the
counterfactual the mask was actually trained to satisfy, evaluated under
each floor semantics:

    ampF0   non-members zeroed          (free0 analogue)
    ampFMd  non-members at posctx mean  (freeM_dense analogue)

Both normalised as (a_c - empty)/(a_pos - empty) with the same empty
baselines as the main panel. Membership sets and alphas come from
provenance archived in rows.jsonl (amp_stats) and members.jsonl.gz +
amp_kept — but amp_kept was only stored in provenance, not archived per
row, so this script RERUNS discovery at the settings of interest and
scores in-process. Settings: the (arm, lambda) cells worth deciding on.

  PYTHONPATH=src python experiments/026-floor-isolation/amp_rescore.py
"""
import json
import os
import random
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.instrument.learned_mask import LearnedMaskPatcher, run_learned_mask
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_means
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense
from model.hooks import multi_patch

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
HERE = Path(__file__).parent
COMP_IDX = int(os.environ.get("COMP_IDX", 8))
N_SEEDS = int(os.environ.get("N_SEEDS", 4))
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
# (floor, triple_weight, lambda) cells to decide on. Depth-aware: at L9
# the gate-only optima sit at 1e-5/1e-4 and NOTHING passes 1e-3+ (R13), so
# the L9 cells probe whether amplitude unlocks exactly the lambdas that
# are closed to gates.
if COMP_IDX == 8:
    CELLS = [("dual", None, 1e-4), ("dual", None, 1e-3),
             ("triple", 0.10, 1e-3), ("triple", 2.0, 3e-3)]
else:
    CELLS = [("dual", None, 1e-4), ("dual", None, 1e-3),
             ("triple", 0.05, 1e-3), ("triple", 0.05, 1e-4)]
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


class AmpCircuitPatcher:
    """Kept latents at alpha * live value; everything else at the floor
    (None = zero). The with-amplitude counterfactual the mask trained on,
    with a HARD membership set."""

    def __init__(self, alphas, floors, seed_site, w_seed, b_seed):
        self.alphas = alphas          # {site: {idx: alpha}}
        self.floors = floors or {}    # {site: [d_sae]} or empty for zero
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


TAG = "" if COMP_IDX == 8 else "_c%d" % COMP_IDX
fh = (HERE / ("amp_rescore%s.jsonl" % TAG)).open("a")
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
    print("\n[%d] a_pos %.3f" % (sl, a_pos), flush=True)

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
                v = torch.relu(pre[:B][rows, anc])
                tot += float(v.sum()); n += B
        finally:
            inference.enable_compile()
        return tot / max(n, 1)

    for floor, tw, lam in CELLS:
        t0 = time.time()
        kw = dict(sites=UP, seed_layer=LAYER, seed_kind=KIND,
                  seed_latent_idx=sl, pos_tokens=pt, pos_argmax=pa,
                  neg_tokens=nt, binarize=cfg.binarize, steps=cfg.steps,
                  lr=cfg.lr, l1_lambda=lam, keep_threshold=cfg.keep_threshold,
                  batch_size=disc.probe_batch_size,
                  holdout_frac=cfg.holdout_frac, theta_init=cfg.theta_init,
                  log_every=0, deep_site_threshold=cfg.deep_site_threshold,
                  deep_batch_size=cfg.deep_batch_size, optimizer=cfg.optimizer,
                  weight_decay=cfg.weight_decay, code_dtype=cfg.code_dtype,
                  lr_schedule=cfg.lr_schedule, lr_min_frac=cfg.lr_min_frac,
                  warmup_frac=cfg.warmup_frac, mask_floor_source=floor,
                  free_amplitude=True)
        if floor in ("dual", "triple"):
            kw["dual_floor_weight"] = cfg.dual_floor_weight
        if floor == "triple" and tw is not None:
            kw["triple_floor_weight"] = tw
        scores, prov = run_learned_mask(inference, bank, objective="pos", **kw)
        ak = prov.get("amp_kept") or {}
        alphas = {}
        for k, d in ak.items():
            lyr, knd = k.split("/")
            alphas[(int(lyr), knd)] = {int(i): float(v) for i, v in d.items()}
        n_mem = sum(len(v) for v in alphas.values())
        # natural-value membership eval (the standard panel's view)
        keep = {s: set(d) for s, d in alphas.items()}
        nat0 = float(circuit_only_activation(inference, bank, keep, UP, pt,
                                             LAYER, KIND, sl, pos_argmax=pa,
                                             batch_size=EVAL_BS))
        # with-amplitude eval under both floors
        aw0 = amp_act(alphas, None)
        awM = amp_act(alphas, means)
        st = prov.get("amp_stats") or {}
        row = {"latent": sl, "floor": floor, "triple_w": tw, "l1": lam,
               "n": n_mem,
               "nat_free0": round((nat0 - e0) / (a_pos - e0), 4),
               "amp_free0": round((aw0 - e0) / (a_pos - e0), 4),
               "amp_freeMd": round((awM - eMd) / (a_pos - eMd), 4) if abs(a_pos - eMd) > 1e-9 else None,
               "alpha_median": st.get("median"), "alpha_p90": st.get("p90"),
               "frac_elev": st.get("frac_elevated"),
               "holdout": prov.get("holdout_data_loss"),
               "secs": round(time.time() - t0, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-7s w=%-5s l1=%-7g n=%-6d natF0=%-8s ampF0=%-8s ampFMd=%-8s "
              "alpha(med %.2f p90 %.2f) %.0fs"
              % (floor, tw, lam, n_mem, row["nat_free0"], row["amp_free0"],
                 row["amp_freeMd"], st.get("median", 0), st.get("p90", 0),
                 row["secs"]), flush=True)
        torch.cuda.empty_cache()

fh.close()
print("ALL DONE", flush=True)
