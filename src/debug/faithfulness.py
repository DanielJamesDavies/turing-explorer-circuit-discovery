"""
Debug script: runs ONE candidate through SFCAttributionPatching and prints
verbose diagnostics at every stage, including the raw logit tensors and MSE
values from the three evaluation passes.

Run from the repo root:
    python -m debug.faithfulness [--comp COMP_IDX] [--lat LAT_IDX]
    # or: python src/debug/faithfulness.py [--comp COMP_IDX] [--lat LAT_IDX]

If --comp / --lat are omitted the first candidate from outputs/candidates.pt
is used.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F

from hardware import detect_devices
from model.inference import Inference
from sae.bank import SAEBank
from data.loader import DataLoader
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation
from store.context import top_ctx, neg_ctx
from store.logit_context import logit_ctx
from circuit.discovery_window import DiscoveryWindow
from circuit.discovery.sfc_attribution_patching import SFCAttributionPatching
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.patcher import CircuitPatcher
from circuit.neg_ctx_baseline import compute_neg_ctx_means
from config import config
from pipeline.component_index import split_component_idx


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _stats(t: torch.Tensor, label: str):
    t = t.float()
    nonzero = t[t != 0]
    print(f"  {label}: shape={tuple(t.shape)}  "
          f"min={t.min():.6f}  max={t.max():.6f}  "
          f"mean={t.mean():.6f}  "
          f"nonzero={nonzero.numel()} / {t.numel()}  "
          + (f"nz_mean={nonzero.mean():.6f}" if nonzero.numel() > 0 else "nz_mean=n/a"))


def _logit_diff(orig, other, label, batch_idx, pos_argmax):
    """Print per-sequence logit diff at the probe positions."""
    o = orig[batch_idx, pos_argmax].float()
    x = other[batch_idx, pos_argmax].float()
    mse = F.mse_loss(x, o)
    diff = (x - o).abs()
    print(f"  {label}: MSE={mse:.6f}  max_abs_diff={diff.max():.6f}  "
          f"mean_abs_diff={diff.mean():.6f}")
    return float(mse.item())


@torch.no_grad()
def debug_baseline_mse(inference, sae_bank, avg_acts, tokens, pos_argmax, label="neg-ctx"):
    """
    Runs only the original and total-ablation baseline passes and prints the
    faithfulness DENOMINATOR: MSE(baseline, original) at probe positions.

    This tells us whether the ablation is actually disrupting the logits
    (large MSE = good signal) or not (near-zero MSE = uninformative baseline).
    """
    use_all = pos_argmax is not None
    B = tokens.shape[0]

    _, orig_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        return_activations=False, all_logits=use_all,
    )

    patcher_b = CircuitPatcher(sae_bank, None, avg_acts, pos_argmax=pos_argmax)
    _, base_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_b, return_activations=False, all_logits=use_all,
    )

    if pos_argmax is not None:
        batch_idx = torch.arange(B, device=orig_logits.device)
        pa = pos_argmax.to(orig_logits.device)
        o = orig_logits[batch_idx, pa].float()
        b = base_logits[batch_idx, pa].float()
    else:
        o = orig_logits[:, -1, :].float()
        b = base_logits[:, -1, :].float()

    mse_b    = float(F.mse_loss(b, o).item())
    per_seq  = F.mse_loss(b, o, reduction="none").mean(dim=-1)   # [B]
    abs_diff = (b - o).abs()

    print(f"  [{label} baseline vs original]")
    print(f"    MSE(baseline, orig) = {mse_b:.8f}   ← faithfulness denominator")
    print(f"    per-seq MSE : min={per_seq.min():.6f}  max={per_seq.max():.6f}  "
          f"mean={per_seq.mean():.6f}")
    print(f"    orig  logits: mean_abs={o.abs().mean():.4f}  std={o.std():.4f}")
    print(f"    base  logits: mean_abs={b.abs().mean():.4f}  std={b.std():.4f}")
    print(f"    abs diff    : mean={abs_diff.mean():.6f}  max={abs_diff.max():.6f}")

    if mse_b < 1e-6:
        print(f"  ⚠ VERY SMALL denominator — baseline is nearly identical to original.")
        print(f"    Faithfulness will be ~0 regardless of circuit quality.")
    elif mse_b < 0.01:
        print(f"  ⚠ SMALL denominator — consider more neg sequences or zero ablation.")
    else:
        print(f"  ✓ Denominator is meaningful (MSE={mse_b:.4f})")

    return mse_b


@torch.no_grad()
def debug_eval(inference, sae_bank, avg_acts, circuit, tokens, pos_argmax):
    """
    Replicates evaluate_faithfulness but with verbose intermediate printing.
    """
    use_all = pos_argmax is not None
    B = tokens.shape[0]
    device = next(inference.model.parameters()).device

    print("\n  [Pass 1] Original forward (no patcher)...")
    _, orig_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        return_activations=False, all_logits=use_all,
    )
    print(f"    orig_logits shape: {orig_logits.shape}  "
          f"dtype: {orig_logits.dtype}  device: {orig_logits.device}")
    _stats(orig_logits.float(), "orig_logits")

    print("\n  [Pass 2] Circuit forward (position-selective)...")
    patcher_c = CircuitPatcher(sae_bank, circuit, avg_acts, pos_argmax=pos_argmax)
    _, circ_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_c, return_activations=False, all_logits=use_all,
    )
    print(f"    circ_logits shape: {circ_logits.shape}")
    _stats(circ_logits.float(), "circ_logits")

    print("\n  [Pass 3] Baseline forward (position-selective, circuit=None)...")
    patcher_b = CircuitPatcher(sae_bank, None, avg_acts, pos_argmax=pos_argmax)
    _, base_logits, _ = inference.forward(
        tokens, num_gen=1, tokenize_final=False,
        patcher=patcher_b, return_activations=False, all_logits=use_all,
    )
    print(f"    base_logits shape: {base_logits.shape}")
    _stats(base_logits.float(), "base_logits")

    batch_idx = torch.arange(B, device=orig_logits.device)
    if pos_argmax is not None:
        pa = pos_argmax.to(orig_logits.device)
        print(f"\n  pos_argmax (first 8): {pa[:8].tolist()}")
        o = orig_logits[batch_idx, pa]
        c = circ_logits[batch_idx, pa]
        b = base_logits[batch_idx, pa]
    else:
        o = orig_logits[:, -1, :]
        c = circ_logits[:, -1, :]
        b = base_logits[:, -1, :]

    print("\n  [MSE at probe positions]")
    mse_c = float(F.mse_loss(c.float(), o.float()).item())
    mse_b = float(F.mse_loss(b.float(), o.float()).item())
    diff_cb = (c.float() - b.float()).abs()
    print(f"    MSE(circuit,  orig)  = {mse_c:.8f}")
    print(f"    MSE(baseline, orig)  = {mse_b:.8f}")
    print(f"    circ vs base  max_abs_diff = {diff_cb.max():.8f}  "
          f"mean_abs_diff = {diff_cb.mean():.8f}")

    if mse_b < 1e-9:
        score = 1.0 if mse_c < 1e-9 else 0.0
        print(f"\n  *** mse_base < 1e-9 guard triggered → faithfulness = {score}")
    else:
        score = 1.0 - mse_c / mse_b
        print(f"\n  faithfulness = 1 - {mse_c:.6f} / {mse_b:.6f} = {score:.6f}")

    return score


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=int, default=None, help="comp_idx override")
    parser.add_argument("--lat",  type=int, default=None, help="latent_idx override")
    args = parser.parse_args()

    # 1. Stores
    print("Loading stores...")
    latent_stats.load("outputs/latent_stats.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    top_ctx.load("outputs/top_ctx.pt")
    neg_ctx.load("outputs/neg_ctx.pt")
    logit_ctx.load("outputs/logit_ctx.pt")

    # 2. Model + bank
    devices = detect_devices()
    device  = devices[0]
    loader  = DataLoader(device=device, pin_memory=False)
    model   = Inference(device=device, compile=False)
    bank    = SAEBank(devices=devices, load_decoders=True, compile=False)

    # 3. Pick candidate
    if args.comp is not None and args.lat is not None:
        comp_idx   = args.comp
        latent_idx = args.lat
        print(f"\nUsing manual candidate: comp={comp_idx} lat={latent_idx}")
    else:
        candidates = torch.load("outputs/candidates.pt", weights_only=False)
        cand       = candidates[0]
        comp_idx   = cand["comp_idx"]
        latent_idx = cand["latent_idx"]
        print(f"\nUsing first candidate from candidates.pt: comp={comp_idx}  lat={latent_idx}")

    seed_layer, seed_kind_idx = split_component_idx(comp_idx, len(bank.kinds))
    seed_kind = bank.kinds[seed_kind_idx]
    print(f"Seed: layer={seed_layer}  kind={seed_kind}  latent={latent_idx}")

    # 4. Build probe dataset
    n_components = bank.n_layer * len(bank.kinds)
    avg_acts_zero = torch.zeros((n_components, bank.d_sae), dtype=torch.float32, device=bank.device)

    probe_builder = ProbeDatasetBuilder(model, bank, loader)
    method = SFCAttributionPatching(model, bank, avg_acts_zero, probe_builder)

    print("\nBuilding probe dataset...")
    probe_data = method.build_probe_dataset(comp_idx, latent_idx)
    print(f"  pos_tokens:  {probe_data.pos_tokens.shape}")
    print(f"  neg_tokens:  {probe_data.neg_tokens.shape}")
    print(f"  pos_argmax (first 8): {probe_data.pos_argmax[:8].tolist()}")
    print(f"  pos_argmax stats: min={probe_data.pos_argmax.min()}  "
          f"max={probe_data.pos_argmax.max()}  "
          f"mean={probe_data.pos_argmax.float().mean():.2f}")

    if probe_data.pos_tokens.shape[0] == 0:
        print("ERROR: empty probe dataset — no positive contexts found.")
        return

    # 5. Compute neg-ctx baseline
    max_neg = int(config.discovery.neg_ctx_eval_max or 8)
    print(f"\nComputing neg-ctx means (max_neg={max_neg})...")
    neg_avg_acts = compute_neg_ctx_means(model, bank, probe_data.neg_tokens, max_neg=max_neg)
    _stats(neg_avg_acts, "neg_avg_acts")

    # 5b. Baseline MSE check — always runs so we can see the faithfulness denominator
    #     even if the circuit is later rejected.
    print("\n" + "="*60)
    print("BASELINE MSE CHECK (faithfulness denominator)")
    print("="*60)
    print(f"  Evaluating on ALL {probe_data.pos_tokens.shape[0]} positive sequences")
    mse_neg  = debug_baseline_mse(
        model, bank, neg_avg_acts,
        probe_data.pos_tokens, probe_data.pos_argmax,
        label="neg-ctx",
    )
    print()
    mse_zero = debug_baseline_mse(
        model, bank, avg_acts_zero,
        probe_data.pos_tokens, probe_data.pos_argmax,
        label="zero-ablation",
    )

    # 6. Run SFC to get a circuit
    print("\nRunning SFCAttributionPatching.discover()...")
    circuit = method.discover(comp_idx, latent_idx)
    if circuit is None:
        print("\n  SFC returned None — checking why from log file...")
        log_dir = "outputs/discovery_logs"
        if os.path.isdir(log_dir):
            logs = sorted(
                [f for f in os.listdir(log_dir) if f.endswith(".txt")],
                key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
                reverse=True,
            )
            if logs:
                latest = os.path.join(log_dir, logs[0])
                print(f"\n  Latest log ({logs[0]}):")
                with open(latest) as fh:
                    print(fh.read())
        print("\n  (Circuit was not returned — MSE denominator diagnostics printed above.)")
        print(f"  neg-ctx  MSE(baseline, orig) = {mse_neg:.8f}")
        print(f"  zero-abl MSE(baseline, orig) = {mse_zero:.8f}")
        if mse_neg < 1e-4:
            print("  ⚠ Denominator is near-zero with neg-ctx baseline.")
            print("    Try: increase neg_ctx_eval_max, or switch to zero ablation for eval.")
        return

    print(f"\nCircuit built: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
    for n in circuit.nodes.values():
        m = n.metadata
        print(f"  node  layer={m.get('layer_idx')}  kind={m.get('kind')}  "
              f"lat={m.get('latent_idx')}  role={m.get('role')}  "
              f"score={m.get('effect_score', 'n/a')}")
    for e in circuit.edges[:10]:
        print(f"  edge  {e.source_uuid[:8]}→{e.target_uuid[:8]}  w={e.weight:.6f}")
    if len(circuit.edges) > 10:
        print(f"  ... and {len(circuit.edges)-10} more edges")

    # 7. Evaluate with ZERO ablation (old baseline)
    n_probe = min(method.probe_batch_size, probe_data.pos_tokens.shape[0])
    probe_tokens  = probe_data.pos_tokens[:n_probe]
    probe_argmax  = probe_data.pos_argmax[:n_probe]

    print("\n" + "="*60)
    print("EVAL A — zero-ablation baseline (avg_acts = 0)")
    print("="*60)
    score_zero = debug_eval(model, bank, avg_acts_zero, circuit, probe_tokens, probe_argmax)

    # 8. Evaluate with neg-ctx baseline
    print("\n" + "="*60)
    print("EVAL B — neg-ctx baseline (avg_acts = neg means)")
    print("="*60)
    score_neg = debug_eval(model, bank, neg_avg_acts, circuit, probe_tokens, probe_argmax)

    # 9. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  comp={comp_idx}  lat={latent_idx}  "
          f"seed layer={seed_layer}  kind={seed_kind}")
    print(f"  circuit: {len(circuit.nodes)} nodes  {len(circuit.edges)} edges")
    print(f"  neg_avg_acts nonzero: {(neg_avg_acts != 0).sum().item()} / {neg_avg_acts.numel()}")
    print(f"  faithfulness (zero ablation):    {score_zero:.6f}")
    print(f"  faithfulness (neg-ctx baseline): {score_neg:.6f}")
    print(f"  min_faithfulness threshold:      {method.min_faithfulness}")


if __name__ == "__main__":
    main()
