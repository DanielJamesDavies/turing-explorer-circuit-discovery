"""Circuit-level analysis suites."""

from __future__ import annotations

from pathlib import Path

from .circuit_motifs import plot_circuit_motifs
from .coact_overlap import plot_circuit_coact_overlap
from .gradient_distribution import plot_gradient_distribution
from .gradient_grid_breakdown import plot_gradient_grid_seed_breakdown
from .gradient_grid_descriptives import compute_gradient_grid_descriptives
from .gradient_grid_medians import plot_gradient_grid_median_faithfulness
from .gradient_method_eval_distribution import plot_gradient_method_eval_distribution
from .gradient_method_eval_top_n_distribution import plot_gradient_method_eval_top_n_distribution
from .gradient_method_neg_mode_grid_runner import run_gradient_method_neg_mode_grid
from .gradient_neg_mode_comparison import plot_gradient_neg_mode_comparison
from .gradient_size_curve import plot_gradient_size_curve
from .gradient_size_sweep_runner import run_gradient_size_sweep
from .hybrid_source_overlap import plot_hybrid_source_overlap
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
        paper_figure_path = getattr(result, "paper_figure_path", None)
        if paper_figure_path is not None:
            paths.append(paper_figure_path)
    return paths


__all__ = [
    "plot_circuit_motifs",
    "plot_circuit_coact_overlap",
    "plot_circuit_latent_commonality",
    "plot_circuit_node_hop_overlap",
    "plot_circuit_seed_coact_hops",
    "compute_gradient_grid_descriptives",
    "plot_gradient_distribution",
    "plot_gradient_grid_median_faithfulness",
    "plot_gradient_grid_seed_breakdown",
    "plot_gradient_method_eval_distribution",
    "plot_gradient_method_eval_top_n_distribution",
    "plot_gradient_neg_mode_comparison",
    "plot_hybrid_source_overlap",
    "plot_pruned_hop_eval_results",
    "plot_pruned_hop_eval_spec",
    "plot_top_ctx_circuit_vs_coact_frequency",
    "run_gradient_method_neg_mode_grid",
    "run_gradient_size_sweep",
    "plot_gradient_size_curve",
    "run_pruned_hop_evals",
    "run_circuit_suite",
]

