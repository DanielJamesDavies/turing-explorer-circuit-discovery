"""Retro-wiring pilot: attach direct-effect edges (SFC's edge weights) to
freshly discovered circuits and measure chain depth.

For a small seed sample, discovers circuits with local and ig_baseline
attribution (gates off, as in the sweeps), truncates to the deployed
configuration (top-12 per site per role), wires members with
attach_direct_edges, and records longest-path depth stats per circuit.

Run (GPU):
    PYTHONPATH=src python -m debug.edge_wiring_pilot [--sample-size 8]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from analysis.circuits.gradient_size_sweep_runner import (
    _apply_sweep_config,
    _balanced_candidate_indices,
    _build_mode_method,
    _candidate_with_index,
    _circuit_depth_stats,
    _restore_sweep_config,
    _site_role_groups,
    _truncated_circuit_per_site,
)
from circuit.instrument.edge_attribution import attach_direct_edges
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import collect_site_anchors, upstream_sites
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank

FIELDS = [
    "method", "attribution_mode", "comp_idx", "latent_idx", "kind", "layer",
    "nodes_wired", "n_edges_added", "n_downstream_nodes", "node_depth_max", "node_depth_mean",
    "n_internal_edges", "duration_s", "error",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("analysis-restyled/edge-pilot"))
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--truncate-m", type=int, default=12)
    args = parser.parse_args(argv)

    root = args.run_root
    load_discovery_artifacts(root, candidates_path=root / "candidates.pt")
    all_candidates = torch.load(root / "candidates.pt", map_location="cpu", weights_only=False)
    # Depth needs depth: shallow seeds have no upstream-of-upstream sites, so
    # chains are impossible by construction. One seed per layer, deepest first.
    n_kinds = 3
    by_layer: dict[int, int] = {}
    for index, candidate in enumerate(all_candidates):
        layer = int(candidate["comp_idx"]) // n_kinds
        by_layer.setdefault(layer, index)
    layers = sorted(by_layer, reverse=True)[: args.sample_size]
    candidates = [_candidate_with_index(all_candidates[by_layer[l]], by_layer[l]) for l in layers]
    print("[edge-pilot] seed layers:", layers, flush=True)

    devices = detect_devices()
    device = devices[0]
    loader = DataLoader(device=device, pin_memory=is_fast_memory())
    inference = Inference(device=device, compile=should_compile())
    bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
    probe_builder = ProbeDatasetBuilder(inference, bank, loader)
    avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae), dtype=torch.float32, device=bank.device)

    rows = []
    out_csv = args.out / "edge-pilot.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    original = _apply_sweep_config(max_per_site=128)
    try:
        methods = {}
        for name in ("counterfactual_gradient", "ablation_gradient"):
            for mode in ("local", "ig_baseline"):
                methods[(name, mode)] = _build_mode_method(name, mode, inference, bank, avg_acts, probe_builder)
        probe_batch = int(config.discovery.probe_batch_size)
        CircuitLogger._LOG_DIR = str(args.out / "logs")

        for candidate in candidates:
            comp_idx, latent_idx = int(candidate["comp_idx"]), int(candidate["latent_idx"])
            layer, kind_idx = split_component_idx(comp_idx, len(bank.kinds))
            kind = bank.kinds[kind_idx]
            probe = methods[("counterfactual_gradient", "local")].build_probe_dataset(comp_idx, latent_idx)
            if probe.pos_tokens.shape[0] == 0:
                continue
            pos_tokens = probe.pos_tokens[:probe_batch]
            pos_argmax = probe.pos_argmax[:probe_batch]
            in_scope = upstream_sites(bank, int(layer), kind)
            site_means, _ = collect_site_anchors(inference, bank, pos_tokens, in_scope, pos_argmax)

            for (name, mode), method in methods.items():
                start = time.perf_counter()
                row = {
                    "method": name, "attribution_mode": mode, "comp_idx": comp_idx,
                    "latent_idx": latent_idx, "kind": kind, "layer": int(layer), "error": "",
                }
                try:
                    circuit = method.discover(comp_idx, latent_idx)
                    if circuit is None:
                        row["error"] = "no circuit"
                    else:
                        groups = _site_role_groups(circuit)
                        sub, n_use = _truncated_circuit_per_site(circuit, groups, args.truncate_m)
                        stats = attach_direct_edges(
                            sub, inference, bank,
                            pos_tokens=pos_tokens, pos_argmax=pos_argmax,
                            seed_layer=int(layer), seed_kind=kind, seed_latent_idx=latent_idx,
                            site_baselines=site_means,
                            # Batched grad tensors are K x B x T x d_sae PER upstream
                            # anchor; chunk 32 overflows 16GB VRAM into shared memory
                            # (~10x slowdown). 4 keeps the pass resident.
                            chunk_size=4,
                        )
                        row.update({"nodes_wired": n_use, **stats, **_circuit_depth_stats(sub)})
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["duration_s"] = time.perf_counter() - start
                rows.append(row)
                with out_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
                    writer.writeheader()
                    writer.writerows(rows)
                print("[edge-pilot]", name, mode, f"seed {comp_idx}/{latent_idx}",
                      row.get("node_depth_max", "err"), flush=True)
    finally:
        _restore_sweep_config(original)
    print("wrote", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
