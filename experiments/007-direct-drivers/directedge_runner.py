"""Direct-edge circuits: keep only the latents that DIRECTLY drive the seed.

In a residual architecture every upstream latent reaches the seed via
  (a) the IDENTITY path — its decoder write persists in the residual stream
      and the seed's encoder reads it: strength = a_j * (W_dec[:,j] . w_seed)
      — pure geometry x clean activation, no forward pass needed; and
  (b) MEDIATED paths — intermediate attn/mlp react to it.

Attribution conflates the two. This experiment selects members by the DIRECT
strength alone and asks the user's question: pin those members to their clean
values, ablate everything else (zero fill), and see what the evals say.

Selectors at matched K (16..4096):
  direct — top-K by |pins_j * (W_dec_j . w_seed)|   (no discovery involved)
  attr   — top-K by |attribution| from the saved abl-ig_mean PA raw circuit
  rand   — K uniform upstream latents (null), one draw, seed=0

Evals per set: pinned0 (pin members, zero-fill rest — the DIRECT question) and
free0 (members re-encode live — do direct edges have any closure?).

Also reported per seed: the ANALYTIC direct sum — sum over ALL upstream
latents of pins_j * dot_j, vs a_pos. If the identity path were the whole
story, that sum (plus error/embedding terms we do not model) would predict
a_pos. The gap is the mediated share.

  PYTHONPATH=src python experiments/007-direct-drivers/runner.py
"""
import json
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config, _build_mode_method)
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation, measure_seed_activation, upstream_sites)
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/"
                "20260531-152059-37117a33/20260531-152059-37117a33")
CIRC = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2/experiments/"
            "007-free0-cf-32seed/circuits")
HERE = Path(__file__).parent
OUT = HERE / "rows.jsonl"
SEEDS = [(8, 30122, "L2"), (25, 10628, "L8"), (27, 6859, "L9"), (32, 3021, "L10")]
KS = [16, 64, 256, 1024, 4096]
N_SEQ, EVAL_BS, D_SAE = 64, 16, 40960
torch.set_float32_matmul_precision("high")

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
probe_builder = ProbeDatasetBuilder(inference, bank, loader)
avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), D_SAE), device=bank.device)
n_kinds = len(bank.kinds)

_apply_sweep_config(max_per_site=24)
disc = config.discovery
disc.probe_sequence_count = N_SEQ
disc.eval_sequence_count = N_SEQ
disc.eval_batch_size = EVAL_BS
disc.probe_batch_size = 4

fh = OUT.open("a")
for sc_idx, sl, label in SEEDS:
    seed_key = "%d/%d" % (sc_idx, sl)
    layer, ki = split_component_idx(sc_idx, n_kinds)
    kind = bank.kinds[ki]
    up = sorted(upstream_sites(bank, layer, kind))

    m0 = _build_mode_method("counterfactual_gradient", "local", inference, bank,
                            avg_acts, probe_builder)
    pd_ = m0.build_probe_dataset(sc_idx, sl)
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]

    a_pos = measure_seed_activation(inference, bank, pt, layer, kind, sl, pa,
                                    batch_size=EVAL_BS)
    a_e0 = circuit_only_activation(inference, bank, {}, set(up), pt, layer, kind,
                                   sl, pos_argmax=pa, batch_size=EVAL_BS)
    _, pins = collect_site_anchors(inference, bank, pt, set(up), pa,
                                   pin_position_specific=False)

    sae_seed = bank.saes[kind][layer]
    w_seed = sae_seed.encoder.weight[sl].detach().float().cpu()

    # direct strength per upstream latent: pins_j * (W_dec_j . w_seed)
    contribs = []            # (|c|, signed c, site, idx)
    direct_sum = 0.0
    for site in up:
        l, k = site
        sae = bank.saes[k][l]
        dots = (w_seed.to(sae.decoder.weight.device).float()
                @ sae.decoder.weight.float()).cpu()          # [d_sae]
        p = pins[site].float().cpu()
        c = p * dots
        direct_sum += float(c.sum())
        vals, idxs = c.abs().topk(min(8192, c.numel()))
        for v, i in zip(vals.tolist(), idxs.tolist()):
            contribs.append((v, float(c[i]), site, i))
    contribs.sort(key=lambda x: -x[0])

    # attribution ranking from the saved raw abl-ig_mean circuit
    sd = "%d_%d" % (sc_idx, sl)
    ck = torch.load(CIRC / sd / "abl-ig_mean_PA__raw.pt", map_location="cpu",
                    weights_only=False)
    roles = [ck["roles_legend"][i] for i in ck["role"].tolist()]
    attr_ranked = sorted(
        [((l, ck["kinds_legend"][k]), i, abs(float(s)))
         for (l, k, i, s), r in zip(zip(ck["layer"].tolist(), ck["kind_idx"].tolist(),
                                        ck["index"].tolist(), ck["score"].tolist()), roles)
         if r != "seed"], key=lambda x: -x[2])

    g = torch.Generator().manual_seed(0)
    n_up_lat = len(up) * D_SAE
    rand_flat = torch.randperm(n_up_lat, generator=g)[:max(KS)].tolist()

    def keep_direct(K):
        d = {}
        for _, _, site, i in contribs[:K]:
            d.setdefault(site, set()).add(i)
        return d

    def keep_attr(K):
        d = {}
        for site, i, _ in attr_ranked[:K]:
            d.setdefault(site, set()).add(i)
        return d

    def keep_rand(K):
        d = {}
        for f in rand_flat[:K]:
            site = up[f // D_SAE]
            d.setdefault(site, set()).add(f % D_SAE)
        return d

    def phi(keep, pinned):
        a_c = circuit_only_activation(
            inference, bank, keep, set(up), pt, layer, kind, sl, pos_argmax=pa,
            pin_values=pins if pinned else None, batch_size=EVAL_BS)
        den = a_pos - a_e0
        return round(float((a_c - a_e0) / den), 4) if abs(den) > 1e-9 else None

    print("\n[%s] %s | a_pos %.3f | analytic direct sum (all upstream) = %.3f "
          "(%.0f%% of a_pos)" % (seed_key, label, a_pos, direct_sum,
                                 100 * direct_sum / max(a_pos, 1e-9)), flush=True)
    print("  %-6s %10s %10s | %10s %10s | %10s %10s"
          % ("K", "dir-pin", "dir-free", "attr-pin", "attr-free", "rand-pin", "rand-free"),
          flush=True)
    for K in KS:
        row = {"seed": seed_key, "label": label, "K": K,
               "a_pos": round(float(a_pos), 4), "a_e0": round(float(a_e0), 4),
               "direct_sum_full": round(direct_sum, 4)}
        cells = []
        for name, kf in (("dir", keep_direct), ("attr", keep_attr), ("rand", keep_rand)):
            kp = kf(K)
            row[name + "_pin"] = phi(kp, True)
            row[name + "_free"] = phi(kp, False)
            cells += [row[name + "_pin"], row[name + "_free"]]
        # analytic direct sum of the direct top-K alone
        row["direct_sum_topK"] = round(sum(c for _, c, _, _ in contribs[:K]), 4)
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print("  %-6d %10s %10s | %10s %10s | %10s %10s"
              % (K, *cells), flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

fh.close()
print("\nwrote %s" % OUT)
