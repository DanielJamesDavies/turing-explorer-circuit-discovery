"""Circuit-level analysis suites."""

from __future__ import annotations

from pathlib import Path

from .coact_overlap import plot_circuit_coact_overlap
from .gradient_distribution import plot_gradient_distribution
from .latent_commonality import plot_circuit_latent_commonality
from .node_hop_overlap import plot_circuit_node_hop_overlap
from .pruned_hop_eval_runner import run_pruned_hop_evals
from .pruned_hop_eval_results import plot_pruned_hop_eval_results
from .pruned_hop_eval_spec import plot_pruned_hop_eval_spec
from .seed_coact_hops import plot_circuit_seed_coact_hops
from .top_ctx_frequency import plot_top_ctx_circuit_vs_coact_frequency


def run_circuit_suite(run_root: str | Path, *, output_root: str | Path | None = None) -> list[Path]:
    """Generate the default circuit analysis figures and summaries."""

    results = [
        plot_circuit_coact_overlap(run_root, output_root=output_root),
        plot_circuit_seed_coact_hops(run_root, output_root=output_root),
    ]
    paths: list[Path] = []
    for result in results:
        paths.extend([result.figure_path, result.summary_path, result.table_path])
    return paths


__all__ = [
    "plot_circuit_coact_overlap",
    "plot_circuit_latent_commonality",
    "plot_circuit_node_hop_overlap",
    "plot_circuit_seed_coact_hops",
    "plot_gradient_distribution",
    "plot_pruned_hop_eval_results",
    "plot_pruned_hop_eval_spec",
    "plot_top_ctx_circuit_vs_coact_frequency",
    "run_pruned_hop_evals",
    "run_circuit_suite",
]

