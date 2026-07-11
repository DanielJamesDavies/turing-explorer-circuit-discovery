"""Run the gradient-method x negative-context-mode discovery grid."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from analysis.io import analysis_output_dirs, resolve_run_root
from circuit.discovery.ablation_gradient import AblationGradientDiscovery
from circuit.discovery.counterfactual_gradient import CounterfactualGradientDiscovery
from circuit.discovery.hybrid_gradient import HybridGradientDiscovery
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from observability.circuit_logger import CircuitLogger
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank


SUITE_NAME = "gradient-method-neg-mode-grid"
GRID_METHODS = ("counterfactual_gradient", "ablation_gradient", "hybrid_gradient")
GRID_NEG_MODES = ("close", "random", "distant")
ROW_FIELDS = [
    "method",
    "neg_mode",
    "run_index",
    "candidate_index",
    "kind",
    "layer",
    "comp_idx",
    "latent_idx",
    "status",
    "n_nodes",
    "n_edges",
    "duration_s",
    "peak_gb",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "source_counterfactual_returned",
    "source_ablation_returned",
    "source_cf_node_count",
    "source_ablation_node_count",
    "source_intersection_node_count",
    "source_union_node_count",
    "source_cf_only_node_count",
    "source_ablation_only_node_count",
    "source_jaccard",
    "post_prune_cf_only_node_count",
    "post_prune_ablation_only_node_count",
    "post_prune_intersection_node_count",
    "post_prune_union_node_count",
    "post_prune_jaccard",
    "error",
]


def run_gradient_method_neg_mode_grid(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    sample_size: int = 128,
    methods: Sequence[str] = GRID_METHODS,
    neg_modes: Sequence[str] = GRID_NEG_MODES,
) -> dict[str, Path]:
    root = resolve_run_root(run_root)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    rows_path = output_dirs["tables"] / "gradient-method-neg-mode-grid.csv"
    summary_path = output_dirs["summaries"] / "gradient-method-neg-mode-grid.json"
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

    rows: List[Dict[str, Any]] = []
    original_cf_neg_mode = config.discovery.counterfactual_gradient.neg_mode
    original_ablation_neg_mode = config.discovery.ablation_gradient.neg_mode
    try:
        for method_name in methods:
            if method_name not in GRID_METHODS:
                raise ValueError(f"unsupported grid method: {method_name}")
            for neg_mode in neg_modes:
                if neg_mode not in GRID_NEG_MODES:
                    raise ValueError(f"unsupported neg_mode: {neg_mode}")
                config.discovery.counterfactual_gradient.neg_mode = neg_mode
                config.discovery.ablation_gradient.neg_mode = neg_mode
                CircuitLogger._LOG_DIR = str(logs_root / f"{method_name}__{neg_mode}")
                method = _build_method(method_name, inference, bank, avg_acts, probe_builder)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                for run_index, candidate in enumerate(candidates):
                    row = _run_candidate(
                        method,
                        method_name=method_name,
                        neg_mode=neg_mode,
                        run_index=run_index,
                        candidate=candidate,
                        bank=bank,
                        device=device,
                    )
                    rows.append(row)
                    _write_rows(rows_path, rows)
                    print(
                        "[gradient-grid]",
                        method_name,
                        neg_mode,
                        run_index + 1,
                        "/",
                        len(candidates),
                        row["status"],
                        flush=True,
                    )
    finally:
        config.discovery.counterfactual_gradient.neg_mode = original_cf_neg_mode
        config.discovery.ablation_gradient.neg_mode = original_ablation_neg_mode

    summary = _build_summary(
        rows,
        run_root=root,
        output_root=output_dirs["root"],
        sample_size=sample_size,
        selected_indices=selected_indices,
        protected_root_unchanged=before_root == _snapshot_root_without_analysis(root),
        circuits_unchanged=before_circuits == _snapshot_dir(root / "circuits"),
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not summary["protected_root_unchanged"] or not summary["circuits_unchanged"]:
        raise RuntimeError("grid run changed protected run artifacts")
    return {"rows": rows_path, "summary": summary_path}


def _build_method(
    method_name: str,
    inference: Inference,
    bank: SAEBank,
    avg_acts: torch.Tensor,
    probe_builder: ProbeDatasetBuilder,
):
    if method_name == "counterfactual_gradient":
        return CounterfactualGradientDiscovery(inference, bank, avg_acts, probe_builder)
    if method_name == "ablation_gradient":
        return AblationGradientDiscovery(inference, bank, avg_acts, probe_builder)
    if method_name == "hybrid_gradient":
        return HybridGradientDiscovery(inference, bank, avg_acts, probe_builder)
    raise ValueError(f"unsupported grid method: {method_name}")


def _run_candidate(
    method,
    *,
    method_name: str,
    neg_mode: str,
    run_index: int,
    candidate: Mapping[str, Any],
    bank: SAEBank,
    device: torch.device,
) -> Dict[str, Any]:
    comp_idx = int(candidate["comp_idx"])
    latent_idx = int(candidate["latent_idx"])
    source_index = int(candidate["candidate_index"])
    layer, kind_idx = split_component_idx(comp_idx, len(bank.kinds))
    kind = bank.kinds[kind_idx]
    start = time.perf_counter()
    try:
        circuit = method.discover(comp_idx, latent_idx)
        status = "accepted" if circuit is not None else "none"
        metadata = dict(circuit.metadata) if circuit is not None else {}
        n_nodes = len(circuit.nodes) if circuit is not None else 0
        n_edges = len(circuit.edges) if circuit is not None else 0
        error = ""
    except Exception as exc:
        status = "error"
        metadata = {}
        n_nodes = 0
        n_edges = 0
        error = f"{type(exc).__name__}: {exc}"
    return {
        "method": method_name,
        "neg_mode": neg_mode,
        "run_index": run_index,
        "candidate_index": source_index,
        "kind": kind,
        "layer": int(layer),
        "comp_idx": comp_idx,
        "latent_idx": latent_idx,
        "status": status,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "duration_s": time.perf_counter() - start,
        "peak_gb": (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else None,
        "counterfactual_faithfulness": metadata.get("counterfactual_faithfulness"),
        "posctx_suppression_score": metadata.get("posctx_suppression_score"),
        "source_counterfactual_returned": metadata.get("source_counterfactual_returned"),
        "source_ablation_returned": metadata.get("source_ablation_returned"),
        "source_cf_node_count": metadata.get("source_cf_node_count"),
        "source_ablation_node_count": metadata.get("source_ablation_node_count"),
        "source_intersection_node_count": metadata.get("source_intersection_node_count"),
        "source_union_node_count": metadata.get("source_union_node_count"),
        "source_cf_only_node_count": metadata.get("source_cf_only_node_count"),
        "source_ablation_only_node_count": metadata.get("source_ablation_only_node_count"),
        "source_jaccard": metadata.get("source_jaccard"),
        "post_prune_cf_only_node_count": metadata.get("post_prune_cf_only_node_count"),
        "post_prune_ablation_only_node_count": metadata.get("post_prune_ablation_only_node_count"),
        "post_prune_intersection_node_count": metadata.get("post_prune_intersection_node_count"),
        "post_prune_union_node_count": metadata.get("post_prune_union_node_count"),
        "post_prune_jaccard": metadata.get("post_prune_jaccard"),
        "error": error,
    }


def _balanced_candidate_indices(candidates: Sequence[Mapping[str, Any]], sample_size: int) -> List[int]:
    by_comp: dict[int, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_comp[int(candidate["comp_idx"])].append(index)
    selected: List[int] = []
    max_len = max(len(indices) for indices in by_comp.values())
    for rank in range(max_len):
        for comp_idx in sorted(by_comp):
            indices = by_comp[comp_idx]
            if rank < len(indices):
                selected.append(indices[rank])
                if len(selected) >= sample_size:
                    return selected
    return selected


def _candidate_with_index(candidate: Mapping[str, Any], index: int) -> Dict[str, Any]:
    copied = dict(candidate)
    copied["candidate_index"] = index
    return copied


def _write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_root: Path,
    output_root: Path,
    sample_size: int,
    selected_indices: Sequence[int],
    protected_root_unchanged: bool,
    circuits_unchanged: bool,
) -> Dict[str, Any]:
    by_combo: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['method']}__{row['neg_mode']}"
        summary = by_combo.setdefault(key, {"count": 0, "accepted": 0, "errors": 0})
        summary["count"] += 1
        summary["accepted"] += int(row["status"] == "accepted")
        summary["errors"] += int(row["status"] == "error")
    return {
        "run_root": str(run_root),
        "output_root": str(output_root),
        "sample_size": sample_size,
        "selected_indices_head": list(selected_indices[:20]),
        "n_rows": len(rows),
        "accepted_count": sum(row["status"] == "accepted" for row in rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "by_combo": by_combo,
        "by_method": dict(Counter(str(row["method"]) for row in rows)),
        "by_neg_mode": dict(Counter(str(row["neg_mode"]) for row in rows)),
        "protected_root_unchanged": protected_root_unchanged,
        "circuits_unchanged": circuits_unchanged,
    }


def _snapshot_root_without_analysis(path: Path):
    rows = []
    for child in sorted(path.iterdir(), key=lambda p: p.name):
        if child.name == "analysis":
            continue
        st = child.stat()
        rows.append((child.name, "dir" if child.is_dir() else "file", st.st_size, st.st_mtime_ns))
    return rows


def _snapshot_dir(path: Path):
    if not path.exists():
        return None
    rows = []
    for child in sorted(path.iterdir(), key=lambda p: p.name):
        st = child.stat()
        rows.append((child.name, "dir" if child.is_dir() else "file", st.st_size, st.st_mtime_ns))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the gradient method x neg-mode discovery grid.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--methods", nargs="+", default=list(GRID_METHODS), choices=GRID_METHODS)
    parser.add_argument("--neg-modes", nargs="+", default=list(GRID_NEG_MODES), choices=GRID_NEG_MODES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    outputs = run_gradient_method_neg_mode_grid(
        args.run_root,
        output_root=args.output_root,
        sample_size=args.sample_size,
        methods=args.methods,
        neg_modes=args.neg_modes,
    )
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
