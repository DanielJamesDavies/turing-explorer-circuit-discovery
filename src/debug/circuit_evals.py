"""
Debug script: evaluates the SAE reconstruction patching framework under a
"full circuit" baseline — every latent is treated as a circuit member, making
the patcher a mathematical identity (recon + error = x).

This serves as a sanity check / ceiling test for the eval metrics.  If the
framework is correct, all three scores should be ≈ 1.0:

    faithfulness   ≈ 1.0   (MSE(circuit, orig) ≈ 0)
    sufficiency    ≈ 1.0   (log-prob ratio ≈ 0)
    completeness   ≈ 1.0   (complement = empty = baseline ≈ 0 faithfulness)

Any meaningful deviation from 1.0 indicates a bug in the patcher or eval
pipeline rather than a property of the circuit.

Run from the repo root:
    python -m debug.circuit_evals [--comp COMP_IDX] [--lat LAT_IDX]
    python -m debug.circuit_evals                         # uses first candidate
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from hardware import detect_devices
from model.inference import Inference
from sae.bank import SAEBank
from data.loader import DataLoader
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation
from store.context import top_ctx, mid_ctx, neg_ctx
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.neg_ctx_baseline import compute_neg_ctx_means
from circuit.patcher import CircuitPatcher
from eval.faithfulness import _calculate_faithfulness_score
from config import config
from pipeline.component_index import split_component_idx


# ──────────────────────────────────────────────────────────────────────────────
# Three-pass eval
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_full_circuit_eval(
    inference,
    bank,
    avg_acts: torch.Tensor,
    tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    pos_argmax: torch.Tensor,
    label: str = "",
) -> dict:
    """
    Runs faithfulness, sufficiency, and completeness for a full-circuit patcher.

    The full-circuit patcher has every latent in the "circuit", so:
        circuit pass  → identity (live_acts = top_acts, bg = 0)
        complement    → empty circuit = same as baseline ablation
    """
    use_all = True
    B = tokens.shape[0]
    batch_idx = torch.arange(B, device=tokens.device)

    prefix = f"  [{label}] " if label else "  "

    # ── Pass 1: original ──────────────────────────────────────────────────────
    _, orig_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        return_activations=False, all_logits=use_all,
    )

    # ── Pass 2: full-circuit (identity) ───────────────────────────────────────
    patcher_full = CircuitPatcher(bank, None, avg_acts,
                                  pos_argmax=pos_argmax, full_circuit=True)
    _, full_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_full, return_activations=False, all_logits=use_all,
    )

    # ── Pass 3: baseline (total ablation) ────────────────────────────────────
    patcher_base = CircuitPatcher(bank, None, avg_acts, pos_argmax=pos_argmax)
    _, base_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_base, return_activations=False, all_logits=use_all,
    )

    # ── Pass 4: complement (inverse full-circuit = empty circuit = baseline) ──
    patcher_comp = CircuitPatcher(bank, None, avg_acts,
                                  inverse=True, pos_argmax=pos_argmax,
                                  full_circuit=True)
    _, comp_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_comp, return_activations=False, all_logits=use_all,
    )

    # ── Faithfulness ──────────────────────────────────────────────────────────
    faithfulness = _calculate_faithfulness_score(
        orig_logits, full_logits, base_logits, pos_argmax
    )

    # ── MSE diagnostics ───────────────────────────────────────────────────────
    pa = pos_argmax.to(orig_logits.device)
    orig_at = orig_logits[batch_idx, pa].float()
    full_at = full_logits[batch_idx, pa].float()
    base_at = base_logits[batch_idx, pa].float()

    mse_circuit  = float(F.mse_loss(full_at, orig_at).item())
    mse_baseline = float(F.mse_loss(base_at, orig_at).item())

    # ── Sufficiency ───────────────────────────────────────────────────────────
    target_at = target_tokens[batch_idx, pa]
    orig_lp   = F.log_softmax(orig_at, dim=-1).gather(-1, target_at.unsqueeze(-1)).squeeze(-1)
    full_lp   = F.log_softmax(full_at, dim=-1).gather(-1, target_at.unsqueeze(-1)).squeeze(-1)
    sufficiency = float(torch.exp(full_lp - orig_lp).mean().item())

    # ── Completeness ──────────────────────────────────────────────────────────
    f_complement = _calculate_faithfulness_score(
        orig_logits, comp_logits, base_logits, pos_argmax
    )
    completeness = 1.0 - f_complement

    print(f"{prefix}MSE(circuit, orig)  = {mse_circuit:.3e}  (expected: ~0)")
    print(f"{prefix}MSE(baseline, orig) = {mse_baseline:.6f}")
    print(f"{prefix}faithfulness        = {faithfulness:.6f}  (expected: 1.0)")
    print(f"{prefix}sufficiency         = {sufficiency:.6f}  (expected: 1.0)")
    print(f"{prefix}completeness        = {completeness:.6f}  (expected: 1.0)")
    print(f"{prefix}f(complement)       = {f_complement:.6f}  (expected: 0.0)")

    return {
        "faithfulness":         faithfulness,
        "sufficiency":          sufficiency,
        "completeness":         completeness,
        "f_complement":         f_complement,
        "mse_circuit_vs_orig":  mse_circuit,
        "mse_baseline_vs_orig": mse_baseline,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full-circuit eval: sanity check for the SAE patching framework."
    )
    parser.add_argument("--comp", type=int, default=None, help="comp_idx override")
    parser.add_argument("--lat",  type=int, default=None, help="latent_idx override")
    args = parser.parse_args()

    # ── Stores ────────────────────────────────────────────────────────────────
    print("Loading stores...")
    latent_stats.load("outputs/latent_stats.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    top_ctx.load("outputs/top_ctx.pt")
    mid_ctx.load("outputs/mid_ctx.pt")
    neg_ctx.load("outputs/neg_ctx.pt")

    # ── Model + bank ──────────────────────────────────────────────────────────
    devices = detect_devices()
    loader  = DataLoader(device=devices[0], pin_memory=False)
    model   = Inference(device=devices[0], compile=False)
    bank    = SAEBank(devices=devices, load_decoders=True, compile=False)

    # ── Candidate ─────────────────────────────────────────────────────────────
    if args.comp is not None and args.lat is not None:
        comp_idx   = args.comp
        latent_idx = args.lat
        print(f"Using manual candidate: comp={comp_idx}  lat={latent_idx}")
    else:
        candidates = torch.load("outputs/candidates.pt", weights_only=False)
        comp_idx   = candidates[0]["comp_idx"]
        latent_idx = candidates[0]["latent_idx"]
        print(f"Using first candidate from candidates.pt: comp={comp_idx}  lat={latent_idx}")

    seed_layer, seed_kind_idx = split_component_idx(comp_idx, len(bank.kinds))
    seed_kind = bank.kinds[seed_kind_idx]
    print(f"Seed: layer={seed_layer}  kind={seed_kind}  latent={latent_idx}")

    # ── Probe dataset ─────────────────────────────────────────────────────────
    probe_builder = ProbeDatasetBuilder(model, bank, loader)
    probe_data    = probe_builder.build_for_latent(
        comp_idx, latent_idx, top_ctx, mid_ctx, neg_ctx,
    )

    if probe_data.pos_tokens.shape[0] == 0:
        print("ERROR: empty probe dataset — no positive contexts found.")
        return

    n_probe      = min(int(config.discovery.probe_batch_size or 32), probe_data.pos_tokens.shape[0])
    tokens       = probe_data.pos_tokens[:n_probe].to(devices[0])
    target_tokens = probe_data.target_tokens[:n_probe].to(devices[0])
    pos_argmax   = probe_data.pos_argmax[:n_probe].to(devices[0])

    print(f"Probe sequences: {n_probe}  (of {probe_data.pos_tokens.shape[0]} available)")
    print(f"pos_argmax: min={pos_argmax.min()}  max={pos_argmax.max()}  "
          f"mean={pos_argmax.float().mean():.1f}")

    n_components  = bank.n_layer * len(bank.kinds)
    avg_acts_zero = torch.zeros((n_components, bank.d_sae), dtype=torch.float32, device=bank.device)

    # ── Eval A: zero ablation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVAL A — zero-ablation baseline (avg_acts = 0)")
    print("=" * 60)
    scores_zero = run_full_circuit_eval(
        model, bank, avg_acts_zero, tokens, target_tokens, pos_argmax,
        label="zero-abl",
    )

    # ── Eval B: neg-ctx baseline ───────────────────────────────────────────────
    max_neg    = int(config.discovery.neg_ctx_eval_max or 8)
    neg_avg    = compute_neg_ctx_means(model, bank, probe_data.neg_tokens, max_neg=max_neg)
    nonzero_ct = int((neg_avg != 0).sum().item())

    print(f"\n" + "=" * 60)
    print(f"EVAL B — neg-ctx baseline (avg_acts from {min(max_neg, probe_data.neg_tokens.shape[0])} neg seqs, "
          f"{nonzero_ct} nonzero latents)")
    print("=" * 60)
    scores_neg = run_full_circuit_eval(
        model, bank, neg_avg, tokens, target_tokens, pos_argmax,
        label="neg-ctx",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  comp={comp_idx}  lat={latent_idx}  layer={seed_layer}  kind={seed_kind}")
    print(f"  probe sequences: {n_probe}")
    print()
    print(f"  {'metric':<20} {'zero-ablation':>16} {'neg-ctx':>16} {'expected':>10}")
    print(f"  {'-'*20} {'-'*16} {'-'*16} {'-'*10}")
    for key, label in [
        ("faithfulness",  "faithfulness"),
        ("sufficiency",   "sufficiency"),
        ("completeness",  "completeness"),
        ("f_complement",  "f(complement)"),
    ]:
        expected = "0.0" if key == "f_complement" else "1.0"
        print(f"  {label:<20} {scores_zero[key]:>16.6f} {scores_neg[key]:>16.6f} {expected:>10}")

    print()
    print("  (minimality: N/A — LOO ablation is undefined over a full-feature circuit)")

    all_ok = all(
        abs(scores_zero[k] - v) < 0.01 and abs(scores_neg[k] - v) < 0.01
        for k, v in [("faithfulness", 1.0), ("sufficiency", 1.0), ("completeness", 1.0)]
    )
    if all_ok:
        print("\n  ✓ All scores within 0.01 of expected — patcher identity holds.")
    else:
        print("\n  ✗ One or more scores deviate from expected — investigate patcher.")


if __name__ == "__main__":
    main()
