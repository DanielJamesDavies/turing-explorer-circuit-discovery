"""END-TO-END CPU SMOKE: the production engine on a public model (DAN-34/35).

Proves the engine RUNS on a public model through the adapters. It is not an
experiment: a handful of steps, a dozen contexts, one target.

  1. Inference / SAEBank built for the family (the config-free family= path).
  2. A target is picked from cached wikitext: a latent at SITE that fires on a
     few percent of 64-token windows. Activating contexts = its strongest
     windows (anchor = peak position); contrast contexts = windows where it
     never fires (verified silent: zero post-Top-K activation at every token).
  3. One WCM[Z+C+A] fit through _build_mode_method("ablation_gradient", "mask",
     ...) -> AblationGradientDiscovery.discover -> run_learned_mask, with the
     protocol-v1 settings (triple floor, free amplitudes, gamma 0.25 / 0.25,
     lambda 1e-3, rank-keep 3e-3, train-only ablation values) except STEPS.
     Contexts reach the engine by the same injection 062's driver uses.
  4. One scorer pass: src/eval/wcm_eval.score_circuit on the same contexts.

  wsl: PYTHONPATH=src CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
       .venv/bin/python experiments/070-public-model-adapters/smoke.py
  env: FAMILY (pythia-70m | gpt2-small)  SITE (e.g. 2.resid)  STEPS (20)
       N_WIN (512 scanned windows)  N_POS (16)  N_NEG (16)  DEVICE (cpu | cuda)
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from common import load_tokenizer, wikitext_windows  # noqa: E402

FAMILY = os.environ.get("FAMILY", "pythia-70m")
SITE = os.environ.get("SITE", "2.resid")
STEPS = int(os.environ.get("STEPS", 20))
N_WIN = int(os.environ.get("N_WIN", 512))
N_POS = int(os.environ.get("N_POS", 16))
N_NEG = int(os.environ.get("N_NEG", 16))
SEQ = 64
DEV = torch.device(os.environ.get("DEVICE", "cpu"))
torch.manual_seed(0)


def log(*a):
    print(*a, flush=True)


def configure(steps):
    """062's protocol_harness.configure (WCM[Z+C+A], gamma 0.25, lambda 1e-3, rank-keep 3e-3) + the scorer's
    batch, with the training budget cut to `steps`."""
    from config import config
    disc = config.discovery
    lm = disc.learned_mask
    lm.mask_floor_source = "triple"; lm.free_amplitude = True; lm.l1_lambda = 1e-3
    lm.deep_site_threshold = 99
    lm.dual_floor_weight = 0.25; lm.triple_floor_weight = 0.25; lm.offtarget_weight = 0.0
    lm.rank_weight = 3e-3; lm.rank_mode = "keep"; lm.floors_train_only = True; lm.steps = int(steps)
    lm.log_every = max(1, int(steps) // 4)
    disc.floor_negctx_mode = "close"; disc.eval_batch_size = 64
    disc.probe_sequence_count = 128; disc.eval_sequence_count = 128
    disc.probe_batch_size = 4
    disc.context_protocol = "legacy"
    disc.neg_context_selection.preact_filter = False
    return config


def site_acts(inference, bank, tokens, layer, kind, bs=32):
    """Post-Top-K dense code of the site over `tokens` -> [N, T, d_sae] (sparse kept as (vals, idx))."""
    ki = bank.kinds.index(kind)
    vals, idxs = [], []
    for s in range(0, len(tokens), bs):
        cap = {}
        inference.forward(tokens[s:s + bs], activations_callback=lambda l, a: cap.__setitem__(l, a[ki]),
                          return_activations=False, tokenize_final=False)
        ta, ti = bank.encode(cap[layer], kind, layer)
        vals.append(ta.float().cpu()); idxs.append(ti.cpu())
    return torch.cat(vals), torch.cat(idxs)


def pick_target(vals, idxs, d_sae):
    """A latent firing on 2-20% of windows with the largest peak; its strongest / never-firing windows."""
    n = vals.shape[0]
    win_max = torch.zeros(n, d_sae)
    for i in range(n):
        dense = torch.zeros(vals.shape[1], d_sae).scatter_add_(-1, idxs[i], vals[i])
        win_max[i] = dense.amax(0)
    frac = (win_max > 0).float().mean(0)
    band = ((frac >= 0.02) & (frac <= 0.20)).nonzero(as_tuple=True)[0]
    if not len(band):
        raise RuntimeError("no latent in the firing band")
    peak = win_max[:, band].topk(N_POS, dim=0).values[-1]          # the N_POS-th strongest window
    latent = int(band[peak.argmax()])
    return latent, win_max[:, latent], float(frac[latent])


def main():
    from model.inference import Inference
    from sae.bank import SAEBank
    from circuit.probe_dataset import ProbeDataset
    from analysis.circuits.gradient_size_sweep_runner import _build_mode_method
    from eval import wcm_eval

    t0 = time.time()
    layer, kind = int(SITE.split(".")[0]), SITE.split(".")[1]
    inference = Inference(DEV, family=FAMILY)
    bank = SAEBank(device=DEV, family=FAMILY, preload=False)
    log("[smoke] %s: %s + %s | kinds %s | k=%d d_sae=%d | %.0fs"
        % (FAMILY, type(inference).__name__, type(bank).__name__, bank.kinds, bank.k, bank.d_sae, time.time() - t0))

    tok = load_tokenizer(FAMILY)
    windows = wikitext_windows(tok, N_WIN, SEQ).to(DEV)
    vals, idxs = site_acts(inference, bank, windows, layer, kind)
    latent, wmax, frac = pick_target(vals, idxs, bank.d_sae)
    order = wmax.argsort(descending=True)
    pos_ids = order[:N_POS]
    silent = (wmax == 0).nonzero(as_tuple=True)[0]
    neg_ids = silent[torch.randperm(len(silent), generator=torch.Generator().manual_seed(0))[:N_NEG]]
    pos = windows[pos_ids]
    dense_pos = torch.zeros(N_POS, SEQ)
    for j, w in enumerate(pos_ids.tolist()):
        hit = idxs[w] == latent
        dense_pos[j] = torch.where(hit, vals[w], torch.zeros_like(vals[w])).amax(-1)
    pos_argmax = dense_pos.argmax(-1)
    neg = windows[neg_ids]
    # verify silence on the contrast windows through a fresh pass (the protocol's "verified silent")
    nv, ni = site_acts(inference, bank, neg, layer, kind)
    assert not bool(((ni == latent) & (nv > 0)).any()), "contrast window fires"
    log("[smoke] target %d.%s.%d: fires on %.1f%% of %d windows; pos peak %.2f..%.2f; %d silent windows"
        % (layer, kind, latent, 100 * frac, N_WIN, float(wmax[pos_ids].max()), float(wmax[pos_ids].min()),
           len(silent)))

    n_train = N_POS - N_POS // 4
    probe = ProbeDataset(pos_tokens=pos, target_tokens=torch.roll(pos, -1, dims=1), neg_tokens=neg,
                         pos_argmax=pos_argmax, metadata={"n_train": n_train})

    configure(STEPS)
    avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae))
    M = _build_mode_method("ablation_gradient", "mask", inference, bank, avg_acts, None)
    M.build_probe_dataset = lambda comp, i, _p=probe: _p
    M._floor_negatives = lambda probe_data, comp, i, logger: probe_data.neg_tokens
    comp = layer * len(bank.kinds) + bank.kinds.index(kind)
    t1 = time.time()
    circuit = M.discover(comp, latent)
    t_fit = time.time() - t1
    prov = M.last_mask_provenance or {}
    if circuit is None:
        raise RuntimeError("discover() returned no circuit")
    members = [nd for nd in circuit.nodes.values() if nd.metadata.get("role") != "seed"]
    by_site = {}
    for nd in members:
        f = nd.metadata["feature_id"]
        by_site["%d.%s" % (f.layer, f.kind)] = by_site.get("%d.%s" % (f.layer, f.kind), 0) + 1
    log("[smoke] fit %d steps in %.0fs: loss %.4g -> %.4g | %d members %s | cf %.3f sup %.3f"
        % (STEPS, t_fit, prov.get("loss_initial") or 0, prov.get("loss_final") or 0, len(members), by_site,
           circuit.metadata.get("counterfactual_faithfulness", float("nan")),
           circuit.metadata.get("posctx_suppression_score", float("nan"))))

    t2 = time.time()
    row = wcm_eval.score_circuit(inference, bank, circuit, probe, batch_size=16)
    log("[smoke] scorer pass %.0fs" % (time.time() - t2))
    keep = {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
    log(json.dumps(keep, indent=1, default=str))
    out = HERE / ("smoke_%s.json" % FAMILY)
    json.dump({"family": FAMILY, "target": "%d.%s.%d" % (layer, kind, latent), "fire_frac": frac,
               "steps": STEPS, "t_fit_s": round(t_fit, 1), "n_members": len(members), "members_by_site": by_site,
               "loss_initial": prov.get("loss_initial"), "loss_final": prov.get("loss_final"),
               "cf": circuit.metadata.get("counterfactual_faithfulness"),
               "sup": circuit.metadata.get("posctx_suppression_score"), "score_row": keep},
              open(out, "w"), indent=1, default=str)
    log("[smoke] OK (%.0fs total) -> %s" % (time.time() - t0, out.name))


if __name__ == "__main__":
    main()
