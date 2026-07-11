"""Faithfulness-by-circuit-size sweep: discover once per seed x method with the
acceptance gates disabled, then evaluate nested attribution-ranked truncations.

Mirrors the SFC-style faithfulness-vs-nodes curve: nodes are ranked once by
attribution and each nested subset is evaluated with the standard 4-pass
seed-intervention evaluation. Truncation is per site per role: the top-m
nodes of each (layer, kind, role) group, matching how discovery itself caps
candidates (top_k_scope="layer_kind"), so m=12 reproduces the deployed
configuration. Discovery runs with per-site caps raised to max(m) and
attribution thresholds at zero so the ranking extends beyond the default
circuit size. Uses the same 128 component-balanced seeds as the gradient
grid, the random negative mode, no pruning, and gates at -100 so the
resulting curves are not conditioned on acceptance.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from analysis.io import analysis_output_dirs, resolve_run_root
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import (
    circuit_only_activation,
    collect_site_anchors,
    evaluate_ablation_faithfulness,
    measure_seed_activation,
    upstream_sites,
)
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit
from circuit.probe_dataset import ProbeDatasetBuilder

from .gradient_method_neg_mode_grid_runner import (
    GRID_METHODS,
    _balanced_candidate_indices,
    _build_method,
    _candidate_with_index,
    _snapshot_dir,
    _snapshot_root_without_analysis,
)

SUITE_NAME = "gradient-size-sweep"
M_VALUES = (2, 4, 8, 12, 16, 32, 64, 128)
SWEEP_NEG_MODE = "random"
ROW_FIELDS = [
    "method",
    "attribution_mode",
    "run_index",
    "candidate_index",
    "kind",
    "layer",
    "comp_idx",
    "latent_idx",
    "status",
    "circuit_nodes_total",
    "circuit_nodes_ranked",
    "m",
    "nodes_used",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "ablation_faithfulness",
    "pinned_ablation_faithfulness",
    "a_empty",
    "node_depth_max",
    "node_depth_mean",
    "n_internal_edges",
    "duration_s",
    "error",
]


def run_gradient_size_sweep(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    sample_size: int = 128,
    methods: Sequence[str] = GRID_METHODS,
    m_values: Sequence[int] = M_VALUES,
    attribution_modes: Sequence[str] = ("local",),
) -> dict[str, Path]:
    root = resolve_run_root(run_root)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    rows_path = output_dirs["tables"] / "gradient-size-sweep.csv"
    summary_path = output_dirs["summaries"] / "gradient-size-sweep.json"
    logs_root = output_dirs["root"] / "discovery_logs"

    before_root = _snapshot_root_without_analysis(root)
    before_circuits = _snapshot_dir(root / "circuits")

    load_discovery_artifacts(root, candidates_path=root / "candidates.pt")
    if before_root != _snapshot_root_without_analysis(root) or before_circuits != _snapshot_dir(root / "circuits"):
        raise RuntimeError("artifact loading changed protected run artifacts")

    all_candidates = torch.load(root / "candidates.pt", map_location="cpu", weights_only=False)
    selected_indices = _balanced_candidate_indices(all_candidates, sample_size)
    candidates = [_candidate_with_index(all_candidates[index], index) for index in selected_indices]

    devices = detect_devices()
    device = devices[0]
    fast = is_fast_memory()
    compile_model = should_compile()
    loader = DataLoader(device=device, pin_memory=fast)
    inference = Inference(device=device, compile=compile_model)
    bank = SAEBank(devices=devices, load_decoders=fast, compile=compile_model)
    probe_builder = ProbeDatasetBuilder(inference, bank, loader)
    avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae), dtype=torch.float32, device=bank.device)

    # Resume: the CSV only ever contains complete per-(seed, method) blocks
    # (rows are flushed after each method finishes its m-loop), so existing
    # rows are a safe checkpoint after an interrupted run.
    rows: List[Dict[str, Any]] = []
    done: set[tuple[str, str, str]] = set()
    if rows_path.exists():
        # utf-8-sig tolerates a BOM (e.g. from external CSV edits on Windows).
        with rows_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        done = {
            (
                str(row["comp_idx"]),
                str(row["latent_idx"]),
                str(row["method"]),
                str(row.get("attribution_mode") or "local"),
            )
            for row in rows
        }
        if rows:
            print(f"[size-sweep] resuming: {len(rows)} rows, {len(done)} seed-method blocks done", flush=True)

    original = _apply_sweep_config(max_per_site=max(int(m) for m in m_values))
    try:
        # Config must be mutated before construction: methods (and hybrid's
        # internal sub-methods) read gate/pruning settings at __init__ time.
        # One instance per (method, attribution_mode) combination; hybrid has
        # no attribution_mode override and only runs in "local".
        method_instances: Dict[tuple[str, str], Any] = {}
        for name in methods:
            for mode in attribution_modes:
                if name == "hybrid_gradient" and mode != "local":
                    continue
                method_instances[(name, mode)] = _build_mode_method(
                    name, mode, inference, bank, avg_acts, probe_builder
                )
        probe_batch_size = int(config.discovery.probe_batch_size)

        for run_index, candidate in enumerate(candidates):
            comp_idx = int(candidate["comp_idx"])
            latent_idx = int(candidate["latent_idx"])
            combos_todo = [
                (name, mode)
                for (name, mode) in method_instances
                if (str(comp_idx), str(latent_idx), name, mode) not in done
            ]
            if not combos_todo:
                continue
            layer, kind_idx = split_component_idx(comp_idx, len(bank.kinds))
            kind = bank.kinds[kind_idx]
            base = {
                "run_index": run_index,
                "candidate_index": int(candidate["candidate_index"]),
                "kind": kind,
                "layer": int(layer),
                "comp_idx": comp_idx,
                "latent_idx": latent_idx,
            }

            # One canonical evaluation context per seed, shared by all methods,
            # so the truncation curves compare methods on identical probes.
            cf_method = next(
                (
                    instance
                    for (name, _mode), instance in method_instances.items()
                    if name == "counterfactual_gradient"
                ),
                None,
            ) or _build_method("counterfactual_gradient", inference, bank, avg_acts, probe_builder)
            eval_contexts, context_error = _build_eval_contexts(
                cf_method, comp_idx, latent_idx, probe_batch_size, logs_root
            )
            if eval_contexts is None:
                for method_name, attr_mode in combos_todo:
                    rows.append(
                        {
                            **base,
                            "method": method_name,
                            "attribution_mode": attr_mode,
                            "status": "no_contexts",
                            "error": context_error,
                        }
                    )
                _write_rows(rows_path, rows)
                continue
            pos_tokens_eval, pos_argmax_eval, neg_tokens_eval = eval_contexts

            # Circuit-independent anchors for ablation faithfulness (SFC
            # protocol: mean ablation, empty-circuit floor), measured once
            # per seed and shared across methods and truncations.
            a_posctx = measure_seed_activation(
                inference, bank, pos_tokens_eval, int(layer), kind, latent_idx, pos_argmax_eval
            )
            in_scope = upstream_sites(bank, int(layer), kind)
            site_means, pin_values = collect_site_anchors(
                inference, bank, pos_tokens_eval, in_scope, pos_argmax_eval
            )
            a_empty = circuit_only_activation(
                inference,
                bank,
                {},
                in_scope,
                pos_tokens_eval,
                int(layer),
                kind,
                latent_idx,
                pos_argmax_eval,
                site_means,
            )

            for method_name, attr_mode in combos_todo:
                CircuitLogger._LOG_DIR = str(logs_root / f"{method_name}__{attr_mode}")
                start = time.perf_counter()
                try:
                    circuit = method_instances[(method_name, attr_mode)].discover(comp_idx, latent_idx)
                except Exception as exc:  # noqa: BLE001 - record and continue the sweep
                    rows.append(
                        {
                            **base,
                            "method": method_name,
                            "attribution_mode": attr_mode,
                            "status": "error",
                            "duration_s": time.perf_counter() - start,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    _write_rows(rows_path, rows)
                    continue
                if circuit is None:
                    rows.append(
                        {
                            **base,
                            "method": method_name,
                            "attribution_mode": attr_mode,
                            "status": "none",
                            "duration_s": time.perf_counter() - start,
                            "error": "",
                        }
                    )
                    _write_rows(rows_path, rows)
                    continue

                depth_stats = _circuit_depth_stats(circuit)
                site_groups = _site_role_groups(circuit)
                n_ranked = sum(len(group) for group in site_groups.values())
                previous_n = None
                previous_scores = None
                for m in m_values:
                    sub_circuit, n_use = _truncated_circuit_per_site(circuit, site_groups, int(m))
                    eval_start = time.perf_counter()
                    if n_use == previous_n and previous_scores is not None:
                        cf_faith, sup_score, abl_faith, pinned_abl = previous_scores
                    else:
                        circuit_layers = {
                            node.feature_id.layer
                            for node in sub_circuit.nodes.values()
                            if node.feature_id is not None
                        }
                        cf_faith, sup_score = evaluate_counterfactual_faithfulness(
                            inference,
                            bank,
                            avg_acts,
                            sub_circuit,
                            neg_tokens=neg_tokens_eval,
                            pos_tokens=pos_tokens_eval,
                            seed_layer=int(layer),
                            seed_kind=kind,
                            seed_latent_idx=latent_idx,
                            pos_argmax=pos_argmax_eval,
                            circuit_layers=circuit_layers,
                        )
                        abl_faith, _ = evaluate_ablation_faithfulness(
                            inference,
                            bank,
                            avg_acts,
                            sub_circuit,
                            pos_tokens=pos_tokens_eval,
                            seed_layer=int(layer),
                            seed_kind=kind,
                            seed_latent_idx=latent_idx,
                            pos_argmax=pos_argmax_eval,
                            ablation="mean",
                            site_means=site_means,
                            a_posctx=a_posctx,
                            a_empty=a_empty,
                        )
                        pinned_abl, _ = evaluate_ablation_faithfulness(
                            inference,
                            bank,
                            avg_acts,
                            sub_circuit,
                            pos_tokens=pos_tokens_eval,
                            seed_layer=int(layer),
                            seed_kind=kind,
                            seed_latent_idx=latent_idx,
                            pos_argmax=pos_argmax_eval,
                            ablation="mean",
                            site_means=site_means,
                            pin_values=pin_values,
                            a_posctx=a_posctx,
                            a_empty=a_empty,
                        )
                        previous_n = n_use
                        previous_scores = (cf_faith, sup_score, abl_faith, pinned_abl)
                    rows.append(
                        {
                            **base,
                            "method": method_name,
                            "attribution_mode": attr_mode,
                            "status": "ok",
                            **depth_stats,
                            "circuit_nodes_total": len(circuit.nodes),
                            "circuit_nodes_ranked": n_ranked,
                            "m": int(m),
                            "nodes_used": n_use,
                            "counterfactual_faithfulness": cf_faith,
                            "posctx_suppression_score": sup_score,
                            "ablation_faithfulness": abl_faith,
                            "pinned_ablation_faithfulness": pinned_abl,
                            "a_empty": a_empty,
                            "duration_s": time.perf_counter() - eval_start,
                            "error": "",
                        }
                    )
                _write_rows(rows_path, rows)
                print(
                    "[size-sweep]",
                    method_name,
                    attr_mode,
                    run_index + 1,
                    "/",
                    len(candidates),
                    f"nodes={n_ranked}",
                    f"depth_max={depth_stats['node_depth_max']}",
                    flush=True,
                )
    finally:
        _restore_sweep_config(original)

    summary = {
        "run_root": str(root),
        "output_root": str(output_dirs["root"]),
        "sample_size": sample_size,
        "neg_mode": SWEEP_NEG_MODE,
        "m_values": list(m_values),
        "truncation": "per_site_per_role",
        "methods": list(methods),
        "attribution_modes": list(attribution_modes),
        "n_rows": len(rows),
        "gates_disabled": True,
        "pruning_disabled": True,
        "ablation_faithfulness_protocol": "mean ablation over posctx batch, empty-circuit floor (SFC-aligned)",
        "protected_root_unchanged": before_root == _snapshot_root_without_analysis(root),
        "circuits_unchanged": before_circuits == _snapshot_dir(root / "circuits"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["protected_root_unchanged"] or not summary["circuits_unchanged"]:
        raise RuntimeError("size sweep changed protected run artifacts")
    return {"rows": rows_path, "summary": summary_path}


def _build_mode_method(
    name: str,
    attribution_mode: str,
    inference: Any,
    bank: Any,
    avg_acts: torch.Tensor,
    probe_builder: Any,
):
    """Build a discovery method pinned to an attribution mode. Hybrid takes
    no override (its sub-methods read config) and only runs in "local"."""

    from circuit.discovery.ablation_gradient import AblationGradientDiscovery
    from circuit.discovery.counterfactual_gradient import CounterfactualGradientDiscovery

    if name == "counterfactual_gradient":
        return CounterfactualGradientDiscovery(
            inference, bank, avg_acts, probe_builder, attribution_mode=attribution_mode
        )
    if name == "ablation_gradient":
        return AblationGradientDiscovery(
            inference, bank, avg_acts, probe_builder, attribution_mode=attribution_mode
        )
    if attribution_mode != "local":
        raise ValueError(f"{name} does not support attribution_mode={attribution_mode!r}")
    return _build_method(name, inference, bank, avg_acts, probe_builder)


def _circuit_depth_stats(circuit: Circuit) -> Dict[str, Any]:
    """Graph-hop depth of nodes from the seed along circuit edges.

    Depth = shortest hop count from a node to the seed following edge
    direction (source -> target). A pure star circuit measures depth 1 for
    every node; closure modes that add node-to-node edges will show deeper
    chains. n_internal_edges counts edges whose target is not the seed."""

    seed_uuid = next(
        (uuid for uuid, node in circuit.nodes.items() if node.metadata.get("role") == "seed"),
        None,
    )
    if seed_uuid is None or not circuit.edges:
        return {"node_depth_max": 0, "node_depth_mean": 0.0, "n_internal_edges": 0}

    # Reverse adjacency: for BFS outward from the seed, walk target -> sources.
    sources_by_target: Dict[str, list[str]] = {}
    n_internal = 0
    for edge in circuit.edges:
        sources_by_target.setdefault(edge.target_uuid, []).append(edge.source_uuid)
        if edge.target_uuid != seed_uuid:
            n_internal += 1

    depths: Dict[str, int] = {seed_uuid: 0}
    frontier = [seed_uuid]
    while frontier:
        next_frontier: list[str] = []
        for uuid in frontier:
            for source in sources_by_target.get(uuid, ()):
                if source not in depths:
                    depths[source] = depths[uuid] + 1
                    next_frontier.append(source)
        frontier = next_frontier

    non_seed_depths = [depth for uuid, depth in depths.items() if uuid != seed_uuid]
    if not non_seed_depths:
        return {"node_depth_max": 0, "node_depth_mean": 0.0, "n_internal_edges": n_internal}
    return {
        "node_depth_max": max(non_seed_depths),
        "node_depth_mean": sum(non_seed_depths) / len(non_seed_depths),
        "n_internal_edges": n_internal,
    }


def _apply_sweep_config(*, max_per_site: int) -> dict[str, Any]:
    cf = config.discovery.counterfactual_gradient
    ab = config.discovery.ablation_gradient
    hy = config.discovery.hybrid_gradient
    original = {
        "cf_neg_mode": cf.neg_mode,
        "cf_min_faithfulness": cf.min_faithfulness,
        "cf_pruning_threshold": cf.pruning_threshold,
        "cf_top_k_activators": cf.top_k_activators,
        "cf_top_k_inhibitors": cf.top_k_inhibitors,
        "cf_activator_threshold": cf.activator_threshold,
        "cf_inhibitor_threshold": cf.inhibitor_threshold,
        "ab_neg_mode": ab.neg_mode,
        "ab_min_suppression_score": ab.min_suppression_score,
        "ab_pruning_threshold": ab.pruning_threshold,
        "ab_top_k_supports": ab.top_k_supports,
        "ab_support_threshold": ab.support_threshold,
        "hy_min_counterfactual_faithfulness": hy.min_counterfactual_faithfulness,
        "hy_min_suppression_score": hy.min_suppression_score,
        "hy_pruning_enabled": hy.pruning_enabled,
    }
    cf.neg_mode = SWEEP_NEG_MODE
    cf.min_faithfulness = -100.0
    cf.pruning_threshold = 0.0
    # Per-site caps raised to the sweep maximum with thresholds at zero so the
    # per-site ranking extends beyond the deployed configuration (m=12).
    cf.top_k_activators = max_per_site
    cf.top_k_inhibitors = max_per_site
    cf.activator_threshold = 0.0
    cf.inhibitor_threshold = 0.0
    ab.neg_mode = SWEEP_NEG_MODE
    ab.min_suppression_score = -100.0
    ab.pruning_threshold = 0.0
    ab.top_k_supports = max_per_site
    ab.support_threshold = 0.0
    hy.min_counterfactual_faithfulness = -100.0
    hy.min_suppression_score = -100.0
    hy.pruning_enabled = False
    return original


def _restore_sweep_config(original: Mapping[str, Any]) -> None:
    cf = config.discovery.counterfactual_gradient
    ab = config.discovery.ablation_gradient
    hy = config.discovery.hybrid_gradient
    cf.neg_mode = original["cf_neg_mode"]
    cf.min_faithfulness = original["cf_min_faithfulness"]
    cf.pruning_threshold = original["cf_pruning_threshold"]
    cf.top_k_activators = original["cf_top_k_activators"]
    cf.top_k_inhibitors = original["cf_top_k_inhibitors"]
    cf.activator_threshold = original["cf_activator_threshold"]
    cf.inhibitor_threshold = original["cf_inhibitor_threshold"]
    ab.neg_mode = original["ab_neg_mode"]
    ab.min_suppression_score = original["ab_min_suppression_score"]
    ab.pruning_threshold = original["ab_pruning_threshold"]
    ab.top_k_supports = original["ab_top_k_supports"]
    ab.support_threshold = original["ab_support_threshold"]
    hy.min_counterfactual_faithfulness = original["hy_min_counterfactual_faithfulness"]
    hy.min_suppression_score = original["hy_min_suppression_score"]
    hy.pruning_enabled = original["hy_pruning_enabled"]


def _build_eval_contexts(
    cf_method: Any,
    comp_idx: int,
    latent_idx: int,
    probe_batch_size: int,
    logs_root: Path,
):
    CircuitLogger._LOG_DIR = str(logs_root / "eval_contexts")
    logger = CircuitLogger(comp_idx, latent_idx, "size_sweep_contexts")
    try:
        probe_data = cf_method.build_probe_dataset(comp_idx, latent_idx)
        if probe_data.pos_tokens.shape[0] == 0:
            return None, "empty probe dataset (no positive contexts)"
        selection = cf_method._select_neg_context(
            comp_idx,
            latent_idx,
            SWEEP_NEG_MODE,
            cf_method.max_neg_sequences,
            cf_method.neg_batch_size,
            logger,
        )
        if selection is None:
            return None, "negative-context selection returned nothing"
        pos_tokens_eval = probe_data.pos_tokens[:probe_batch_size]
        pos_argmax_eval = probe_data.pos_argmax[:probe_batch_size]
        return (pos_tokens_eval, pos_argmax_eval, selection.tokens), ""
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        logger.save()


def _site_role_groups(circuit: Circuit) -> dict[tuple, list]:
    """Non-seed nodes grouped by (layer, kind, role), each ranked by |edge
    weight into the seed| (desc) — the per-site ranking discovery itself uses."""

    weight_by_uuid: dict[str, float] = {}
    for edge in circuit.edges:
        weight = edge.weight
        if weight is None:
            continue
        magnitude = abs(float(weight))
        if magnitude > weight_by_uuid.get(edge.source_uuid, -1.0):
            weight_by_uuid[edge.source_uuid] = magnitude

    def node_score(node) -> float:
        score = weight_by_uuid.get(node.uuid)
        if score is not None:
            return score
        attribution = node.metadata.get("attribution_score")
        return abs(float(attribution)) if attribution is not None else 0.0

    groups: dict[tuple, list] = {}
    for node in circuit.nodes.values():
        role = node.metadata.get("role")
        if role == "seed":
            continue
        fid = node.feature_id
        if fid is None:
            continue
        groups.setdefault((fid.layer, fid.kind, role), []).append(node)
    for members in groups.values():
        members.sort(key=node_score, reverse=True)
    return groups


def _truncated_circuit_per_site(
    circuit: Circuit, site_groups: Mapping[tuple, Sequence[Any]], m: int
) -> tuple[Circuit, int]:
    sub = Circuit(name=f"{circuit.name}_m{m}")
    for node in circuit.nodes.values():
        if node.metadata.get("role") == "seed":
            sub.nodes[node.uuid] = node
    n_use = 0
    for members in site_groups.values():
        for node in members[:m]:
            sub.nodes[node.uuid] = node
            n_use += 1
    kept = set(sub.nodes)
    sub.edges = [edge for edge in circuit.edges if edge.source_uuid in kept and edge.target_uuid in kept]
    return sub, n_use


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the faithfulness-by-circuit-size sweep.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=list(GRID_METHODS), choices=GRID_METHODS)
    parser.add_argument("--m-values", nargs="+", type=int, default=list(M_VALUES))
    parser.add_argument(
        "--attribution-modes", nargs="+", default=["local"], choices=["local", "ig_baseline"]
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_gradient_size_sweep(
        args.run_root,
        output_root=args.output_root,
        sample_size=args.sample_size,
        methods=args.methods,
        m_values=args.m_values,
        attribution_modes=args.attribution_modes,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
