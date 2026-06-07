"""Coactivation analysis suite."""

from __future__ import annotations

from pathlib import Path

from .coact_degrees import plot_coact_degrees
from .component_bipartite_graph import plot_component_bipartite_graph
from .component_pair_heatmap import plot_component_pair_heatmap
from .component_signature_similarity import plot_component_signature_similarity
from .hub_corrected_coacts import plot_hub_corrected_coacts
from .jaccard_overlap import plot_jaccard_overlap
from .latent_profile_clustering import plot_latent_profile_clustering
from .latent_profile_pca import plot_latent_profile_pca
from .mutual_coact_graph import plot_mutual_coact_graph
from .mutual_neighbor_similarity import plot_mutual_neighbor_similarity
from .pmi_histogram import plot_pmi_histogram
from .profile_similarity_distribution import plot_profile_similarity_distribution
from .same_cross_distribution import plot_same_cross_distribution
from .sorted_pmi_decay import plot_sorted_pmi_decay
from .threshold_counts import plot_threshold_counts
from .top_ctx_logit_effect import plot_top_ctx_logit_effect
from .top_ctx_sequence_overlap import plot_top_ctx_sequence_overlap
from .top_coact_hubs import plot_top_coact_hubs


def run_coactivation_suite(run_root: str | Path, *, output_root: str | Path | None = None) -> list[Path]:
    """Generate the default coactivation analysis figures and summaries."""

    results = [
        plot_sorted_pmi_decay(run_root, output_root=output_root),
        plot_pmi_histogram(run_root, output_root=output_root),
        plot_threshold_counts(run_root, output_root=output_root),
        plot_component_pair_heatmap(run_root, output_root=output_root),
        plot_same_cross_distribution(run_root, output_root=output_root),
        plot_component_signature_similarity(run_root, output_root=output_root),
        plot_latent_profile_pca(run_root, output_root=output_root),
        plot_profile_similarity_distribution(run_root, output_root=output_root),
        plot_top_coact_hubs(run_root, output_root=output_root),
        plot_component_bipartite_graph(run_root, output_root=output_root),
        plot_latent_profile_clustering(run_root, output_root=output_root),
        plot_jaccard_overlap(run_root, output_root=output_root),
        plot_mutual_coact_graph(run_root, output_root=output_root),
        plot_hub_corrected_coacts(run_root, output_root=output_root),
        plot_mutual_neighbor_similarity(run_root, output_root=output_root),
        plot_top_ctx_sequence_overlap(run_root, output_root=output_root),
        plot_coact_degrees(run_root, output_root=output_root),
    ]
    paths: list[Path] = []
    for result in results:
        paths.extend([result.figure_path, result.summary_path, result.table_path])
    return paths


__all__ = [
    "plot_component_bipartite_graph",
    "plot_coact_degrees",
    "plot_component_pair_heatmap",
    "plot_component_signature_similarity",
    "plot_hub_corrected_coacts",
    "plot_jaccard_overlap",
    "plot_latent_profile_clustering",
    "plot_latent_profile_pca",
    "plot_mutual_coact_graph",
    "plot_mutual_neighbor_similarity",
    "plot_pmi_histogram",
    "plot_profile_similarity_distribution",
    "plot_same_cross_distribution",
    "plot_sorted_pmi_decay",
    "plot_threshold_counts",
    "plot_top_ctx_logit_effect",
    "plot_top_ctx_sequence_overlap",
    "plot_top_coact_hubs",
    "run_coactivation_suite",
]

