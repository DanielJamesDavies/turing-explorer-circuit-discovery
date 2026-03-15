"""
Debug script for one-seed MLP sparse expansion.

Runs the same expansion logic as MlpSparseExpansion for a single latent and prints:
  1) raw top-coactivation neighbors (display-style decode),
  2) per-parent filtering decisions at each depth,
  3) accepted neighbors per depth,
  4) passthrough capture counts (attn/resid).

Usage:
  python -m debug.mlp_sparse_expansion_step --comp 22 --lat 35295
  python src/debug/mlp_sparse_expansion_step.py --comp 22 --lat 35295
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, cast

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import config
from hardware import detect_devices
from data.loader import DataLoader
from model.inference import Inference
from sae.bank import SAEBank

from store.context import top_ctx, mid_ctx, neg_ctx
from store.latent_stats import latent_stats
from store.top_coactivation import top_coactivation

from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.sae_graph import SAEGraphInstrument

from pipeline.component_index import split_component_idx


Neighbor = Dict[str, Any]


def _decode_global_latent(global_idx: int, d_sae: int, n_kinds: int, kinds: List[str]) -> Tuple[int, str, int, int]:
    comp_idx = int(global_idx) // d_sae
    lat_idx = int(global_idx) % d_sae
    layer_idx, kind_idx = split_component_idx(comp_idx, n_kinds)
    return comp_idx, kinds[kind_idx], layer_idx, lat_idx


def _print_top_coactivation(
    comp_idx: int,
    latent_idx: int,
    bank: SAEBank,
) -> List[Neighbor]:
    """
    Print full stored top-coactivation list for the seed, decoded like display.py.
    """
    indices = top_coactivation.top_indices[comp_idx, latent_idx]
    values = top_coactivation.top_values[comp_idx, latent_idx]
    d_sae = bank.d_sae
    kinds = list(bank.kinds)
    n_kinds = len(kinds)

    decoded: List[Neighbor] = []
    for rank, (g_idx, w) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        gid = int(g_idx)
        weight = float(w)
        if gid == 0 and weight == 0:
            continue
        n_comp, n_kind, n_layer, n_lat = _decode_global_latent(gid, d_sae, n_kinds, kinds)
        decoded.append(
            {
                "rank": rank,
                "global_idx": gid,
                "weight": weight,
                "comp_idx": n_comp,
                "kind": n_kind,
                "layer": n_layer,
                "latent_idx": n_lat,
            }
        )

    print("\n=== Top Co-activation (stored list) ===")
    if not decoded:
        print("No stored neighbors for this seed.")
        return decoded

    for item in decoded:
        print(
            f"{item['rank']:>2}. "
            f"layer={item['layer']:>2} kind={item['kind']:<5} latent={item['latent_idx']:>5} "
            f"weight={item['weight']:.4f}"
        )
    return decoded


def _filter_neighbors_mlp_only(
    parent_comp: int,
    parent_lat: int,
    limit: int,
    in_circuit: Set[Tuple[int, str, int]],
    bank: SAEBank,
    min_active_count: int,
) -> Tuple[List[Neighbor], Counter]:
    """
    Mirror MlpSparseExpansion._expand_mlp_neighbors logic with reason accounting.
    """
    indices = top_coactivation.top_indices[parent_comp, parent_lat]
    values = top_coactivation.top_values[parent_comp, parent_lat]

    d_sae = bank.d_sae
    kinds = list(bank.kinds)
    n_kinds = len(kinds)

    accepted: List[Neighbor] = []
    reasons: Counter = Counter()

    for rank, (g_idx, w) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        if len(accepted) >= limit:
            reasons["limit_reached"] += 1
            break

        gid = int(g_idx)
        weight = float(w)
        n_comp, n_kind, n_layer, n_lat = _decode_global_latent(gid, d_sae, n_kinds, kinds)
        key = (n_layer, n_kind, n_lat)
        active_count = int(latent_stats.active_count[n_comp, n_lat].item())

        if gid == 0 and weight == 0:
            reasons["empty_slot"] += 1
            continue
        if n_kind != "mlp":
            reasons["filtered_non_mlp"] += 1
            continue
        if key in in_circuit:
            reasons["already_in_circuit"] += 1
            continue
        if active_count < min_active_count:
            reasons["below_min_active_count"] += 1
            continue

        accepted.append(
            {
                "rank": rank,
                "global_idx": gid,
                "weight": weight,
                "comp_idx": n_comp,
                "kind": n_kind,
                "layer": n_layer,
                "latent_idx": n_lat,
                "active_count": active_count,
            }
        )
        reasons["accepted"] += 1

    return accepted, reasons


@torch.no_grad()
def _capture_passthrough_nodes(
    inference: Inference,
    bank: SAEBank,
    probe_tokens: torch.Tensor,
) -> Dict[Tuple[int, str], Set[int]]:
    """
    Mirror MlpSparseExpansion._capture_passthrough_nodes.
    """
    instrument = SAEGraphInstrument(bank)
    was_compiled = inference._compiled
    inference.disable_compile()
    try:
        inference.forward(
            probe_tokens,
            num_gen=1,
            tokenize_final=False,
            return_activations=False,
            all_logits=False,
            patcher=instrument,
        )
    finally:
        if was_compiled:
            inference.enable_compile()

    passthrough: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for (layer, kind), steps in instrument.graph.activations.items():
        if kind not in ("attn", "resid"):
            continue
        for _, _, top_indices in steps:
            passthrough[(layer, kind)].update(int(v) for v in top_indices.flatten().tolist())

    return passthrough


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=int, required=True, help="seed component index")
    parser.add_argument("--lat", type=int, required=True, help="seed latent index")
    parser.add_argument(
        "--coact-depth",
        type=str,
        default=None,
        help="override coact depth list, e.g. '128,32'",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=None,
        help="override probe batch size",
    )
    parser.add_argument(
        "--min-active-count",
        type=int,
        default=None,
        help="override minimum active count filter",
    )
    args = parser.parse_args()

    print("Loading stores...")
    latent_stats.load("outputs/latent_stats.pt")
    top_coactivation.load("outputs/top_coactivation.pt")
    top_ctx.load("outputs/top_ctx.pt")
    mid_ctx.load("outputs/mid_ctx.pt")
    neg_ctx.load("outputs/neg_ctx.pt")

    devices = detect_devices()
    device = devices[0]
    loader = DataLoader(device=device, pin_memory=False)
    inference = Inference(device=device, compile=False)
    bank = SAEBank(devices=devices, load_decoders=True, compile=False)

    seed_comp = int(args.comp)
    seed_lat = int(args.lat)
    seed_layer, seed_kind_idx = split_component_idx(seed_comp, len(bank.kinds))
    seed_kind = bank.kinds[seed_kind_idx]

    if seed_kind != "mlp":
        print(f"Seed kind is '{seed_kind}', but this script mirrors mlp_sparse_expansion (MLP seeds only).")
        return

    cfg = config.discovery.mlp_sparse_expansion
    if args.coact_depth:
        coact_depth = [int(x.strip()) for x in args.coact_depth.split(",") if x.strip()]
    elif cfg.coact_depth is not None:
        coact_depth = list(cast(List[int], cfg.coact_depth))
    else:
        coact_depth = [32, 32]

    min_active_count = (
        int(args.min_active_count)
        if args.min_active_count is not None
        else int(cast(int, config.discovery.min_active_count or 50))
    )
    probe_batch_size = (
        int(args.probe_batch_size)
        if args.probe_batch_size is not None
        else int(cast(int, config.discovery.probe_batch_size or 16))
    )

    print("\n=== Seed ===")
    print(f"comp={seed_comp} layer={seed_layer} kind={seed_kind} latent={seed_lat}")
    print(f"coact_depth={coact_depth}")
    print(f"min_active_count={min_active_count}")
    print(f"probe_batch_size={probe_batch_size}")
    print(f"stored top_coactivation slots={top_coactivation.top_indices.shape[-1]}")

    _print_top_coactivation(seed_comp, seed_lat, bank)

    probe_builder = ProbeDatasetBuilder(inference, bank, loader)
    probe_data = probe_builder.build_for_latent(
        seed_comp, seed_lat, top_ctx, mid_ctx, neg_ctx, n_pos=64, n_neg=64
    )
    if probe_data.pos_tokens.shape[0] == 0:
        print("\nNo positive probe contexts for this latent.")
        return
    probe_tokens = probe_data.pos_tokens[:probe_batch_size]
    print(
        f"\nProbe dataset: n_pos={probe_data.pos_tokens.shape[0]} "
        f"n_neg={probe_data.neg_tokens.shape[0]} "
        f"probe_tokens={probe_tokens.shape[0]}"
    )

    seed_key = (seed_layer, seed_kind, seed_lat)
    in_circuit: Set[Tuple[int, str, int]] = {seed_key}
    frontier: List[Tuple[int, int, Tuple[int, str, int]]] = [(seed_comp, seed_lat, seed_key)]
    total_edges = 0

    print("\n=== Sparse Expansion Trace ===")
    for depth_idx, n_coacts in enumerate(coact_depth, start=1):
        role = f"hop{depth_idx}"
        next_frontier: List[Tuple[int, int, Tuple[int, str, int]]] = []
        depth_added = 0
        depth_reason_totals: Counter = Counter()

        print(f"\n[depth-{depth_idx}] role={role} limit={n_coacts} frontier={len(frontier)}")

        for parent_comp, parent_lat, parent_key in frontier:
            print(
                f"  parent layer={parent_key[0]} kind={parent_key[1]} latent={parent_key[2]}"
            )

            accepted, reasons = _filter_neighbors_mlp_only(
                parent_comp,
                parent_lat,
                n_coacts,
                in_circuit,
                bank,
                min_active_count,
            )
            depth_reason_totals.update(reasons)

            if accepted:
                print("    accepted:")
                for n in accepted:
                    print(
                        f"      rank={n['rank']:>2} -> layer={n['layer']:>2} "
                        f"kind={n['kind']:<5} latent={n['latent_idx']:>5} "
                        f"weight={n['weight']:.4f} active_count={n['active_count']}"
                    )
                    key = (int(n["layer"]), str(n["kind"]), int(n["latent_idx"]))
                    if key not in in_circuit:
                        in_circuit.add(key)
                        next_frontier.append((int(n["comp_idx"]), int(n["latent_idx"]), key))
                        total_edges += 1
                        depth_added += 1
            else:
                print("    accepted: none")

            print(f"    reasons: {dict(reasons)}")

        frontier = next_frontier
        print(
            f"  depth-{depth_idx} summary: added={depth_added} "
            f"nodes={len(in_circuit)} edges={total_edges}"
        )
        print(f"  depth-{depth_idx} aggregate reasons: {dict(depth_reason_totals)}")

    print("\n=== Passthrough Capture (attn/resid) ===")
    passthrough = _capture_passthrough_nodes(inference, bank, probe_tokens)
    by_kind: Counter = Counter()
    total_passthrough = 0

    for (_, kind), lat_set in passthrough.items():
        count = len(lat_set)
        by_kind[kind] += count
        total_passthrough += count

    print(f"passthrough total={total_passthrough} by_kind={dict(by_kind)}")
    print(
        f"final estimated nodes = sparse({len(in_circuit)}) + passthrough({total_passthrough}) "
        f"= {len(in_circuit) + total_passthrough}"
    )


if __name__ == "__main__":
    main()
