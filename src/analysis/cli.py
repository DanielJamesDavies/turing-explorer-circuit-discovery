"""Command line entrypoint for run artifact analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analysis.circuits import (
    plot_circuit_motifs,
    plot_circuit_coact_overlap,
    plot_circuit_latent_commonality,
    plot_circuit_node_hop_overlap,
    compute_gradient_grid_descriptives,
    plot_gradient_distribution,
    plot_gradient_grid_median_faithfulness,
    plot_gradient_grid_seed_breakdown,
    plot_gradient_method_eval_distribution,
    plot_gradient_method_eval_top_n_distribution,
    plot_gradient_neg_mode_comparison,
    plot_hybrid_source_overlap,
    plot_pruned_hop_eval_results,
    plot_pruned_hop_eval_spec,
    plot_circuit_seed_coact_hops,
    plot_top_ctx_circuit_vs_coact_frequency,
    plot_gradient_size_curve,
    run_circuit_suite,
    run_gradient_method_neg_mode_grid,
    run_gradient_size_sweep,
    run_pruned_hop_evals,
)
from analysis.coactivation import (
    plot_coact_degrees,
    plot_component_bipartite_graph,
    plot_component_pair_heatmap,
    plot_component_signature_similarity,
    plot_hub_corrected_coacts,
    plot_jaccard_overlap,
    plot_latent_profile_clustering,
    plot_latent_profile_pca,
    plot_mutual_coact_graph,
    plot_mutual_neighbor_similarity,
    plot_pmi_histogram,
    plot_profile_similarity_distribution,
    plot_same_cross_distribution,
    plot_sorted_pmi_decay,
    plot_threshold_counts,
    plot_top_ctx_logit_effect,
    plot_top_ctx_sequence_overlap,
    plot_top_coact_hubs,
    run_coactivation_suite,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate analysis plots for pipeline run artifacts.")
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help="Path to a pipeline run root containing canonical artifacts.",
    )
    parser.add_argument(
        "--suite",
        default="coactivation",
        choices=("coactivation", "circuits"),
        help="Analysis suite to run.",
    )
    parser.add_argument(
        "--plot",
        default="all",
        choices=(
            "all",
            "sorted-pmi-decay",
            "pmi-histogram",
            "strong-coact-counts",
            "component-pair-heatmap",
            "same-vs-cross-component",
            "component-signature-similarity",
            "latent-profile-pca",
            "profile-similarity-distribution",
            "top-coact-hubs",
            "component-bipartite-graph",
            "latent-profile-clustering",
            "jaccard-overlap",
            "mutual-coact-graph",
            "hub-corrected-coacts",
            "mutual-neighbor-similarity",
            "top-ctx-logit-effect",
            "top-ctx-sequence-overlap",
            "circuit-coact-overlap",
            "circuit-latent-commonality",
            "circuit-node-hop-overlap",
            "circuit-seed-coact-hops",
            "top-ctx-circuit-vs-coact-frequency",
            "gradient-distribution",
            "gradient-method-neg-mode-grid-run",
            "gradient-size-sweep-run",
            "gradient-size-curve",
            "gradient-grid-median-faithfulness",
            "gradient-grid-seed-breakdown",
            "gradient-grid-descriptives",
            "gradient-method-eval-distribution",
            "gradient-method-eval-top-n-distribution",
            "gradient-neg-mode-comparison",
            "hybrid-source-overlap",
            "pruned-hop-eval-results",
            "pruned-hop-eval-run",
            "pruned-hop-eval-spec",
            "coact-degrees",
            "circuit-motifs",
        ),
        help="Plot to generate within the selected suite.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root for generated analysis outputs. Defaults to --run-root.",
    )
    parser.add_argument(
        "--circuit-store",
        type=Path,
        default=None,
        help="Optional path to discovered_circuits.pt for exact circuit-node analyses.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Maximum coact-hop depth for exact circuit-node hop analyses.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=128,
        help="Sample size for sampled circuit experiments.",
    )
    parser.add_argument(
        "--top-ctx-batch-size",
        type=int,
        default=16,
        help="Top-context sequence count per latent for top-ctx logit-effect analysis.",
    )
    parser.add_argument(
        "--rare-max-pct",
        type=float,
        default=5.0,
        help="Maximum percent of circuits for the latent-commonality rare bucket.",
    )
    parser.add_argument(
        "--common-min-pct",
        type=float,
        default=15.0,
        help="Minimum percent of circuits for the latent-commonality common bucket.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional path to an eval results CSV for result-plotting commands.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="Optional path to a pruned-hop eval spec PT file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of new eval rows to write before exiting.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="For eval runners, retry rows already present with errors or non-finite scores.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="For top-N eval distribution plots, number of circuits per method to keep. Defaults to the smallest method count.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_paths: list[Path]

    if args.suite == "coactivation" and args.plot == "all":
        output_paths = run_coactivation_suite(args.run_root, output_root=args.output_root)
    elif args.suite == "circuits" and args.plot == "all":
        output_paths = run_circuit_suite(args.run_root, output_root=args.output_root)
    elif args.suite == "circuits" and args.plot == "circuit-coact-overlap":
        result = plot_circuit_coact_overlap(args.run_root, output_root=args.output_root)
        output_paths = [result.figure_path, result.summary_path, result.table_path]
        if result.paper_figure_path is not None:
            output_paths.append(result.paper_figure_path)
    elif args.suite == "circuits" and args.plot == "circuit-latent-commonality":
        result = plot_circuit_latent_commonality(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
            rare_max_pct=args.rare_max_pct,
            common_min_pct=args.common_min_pct,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "circuit-node-hop-overlap":
        result = plot_circuit_node_hop_overlap(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
            max_hops=args.max_hops,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "circuit-seed-coact-hops":
        result = plot_circuit_seed_coact_hops(args.run_root, output_root=args.output_root)
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "circuit-motifs":
        result = plot_circuit_motifs(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
        )
        output_paths = [
            result.figure_path,
            result.summary_path,
            result.motifs_table_path,
            result.membership_table_path,
            result.cohesion_table_path,
            result.family_table_path,
        ]
    elif args.suite == "circuits" and args.plot == "top-ctx-circuit-vs-coact-frequency":
        result = plot_top_ctx_circuit_vs_coact_frequency(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
            sample_size=args.sample_size,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-distribution":
        result = plot_gradient_distribution(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
            sample_size=args.sample_size,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-method-neg-mode-grid-run":
        result = run_gradient_method_neg_mode_grid(
            args.run_root,
            output_root=args.output_root,
            sample_size=args.sample_size,
        )
        output_paths = [result["rows"], result["summary"]]
    elif args.suite == "circuits" and args.plot == "gradient-size-sweep-run":
        result = run_gradient_size_sweep(
            args.run_root,
            output_root=args.output_root,
            sample_size=args.sample_size,
        )
        output_paths = [result["rows"], result["summary"]]
    elif args.suite == "circuits" and args.plot == "gradient-size-curve":
        result = plot_gradient_size_curve(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
        if result.ablation_figure_path is not None:
            output_paths.append(result.ablation_figure_path)
    elif args.suite == "circuits" and args.plot == "gradient-grid-median-faithfulness":
        result = plot_gradient_grid_median_faithfulness(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-grid-seed-breakdown":
        result = plot_gradient_grid_seed_breakdown(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-grid-descriptives":
        result = compute_gradient_grid_descriptives(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-method-eval-distribution":
        result = plot_gradient_method_eval_distribution(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-method-eval-top-n-distribution":
        result = plot_gradient_method_eval_top_n_distribution(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
            top_n=args.top_n,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "gradient-neg-mode-comparison":
        result = plot_gradient_neg_mode_comparison(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [
            *result.figure_paths,
            result.summary_path,
            result.aggregate_table_path,
            result.paired_delta_table_path,
        ]
    elif args.suite == "circuits" and args.plot == "hybrid-source-overlap":
        result = plot_hybrid_source_overlap(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [
            *result.figure_paths,
            result.summary_path,
            result.aggregate_table_path,
            result.paired_delta_table_path,
        ]
    elif args.suite == "circuits" and args.plot == "pruned-hop-eval-spec":
        result = plot_pruned_hop_eval_spec(
            args.run_root,
            circuit_store_path=args.circuit_store,
            output_root=args.output_root,
            sample_size=args.sample_size,
            max_hops=args.max_hops,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path, result.spec_path]
    elif args.suite == "circuits" and args.plot == "pruned-hop-eval-results":
        result = plot_pruned_hop_eval_results(
            args.run_root,
            results_path=args.results_path,
            output_root=args.output_root,
        )
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    elif args.suite == "circuits" and args.plot == "pruned-hop-eval-run":
        result_path = run_pruned_hop_evals(
            args.run_root,
            spec_path=args.spec_path,
            output_root=args.output_root,
            limit=args.limit,
            retry_errors=args.retry_errors,
        )
        output_paths = [result_path]
    elif args.suite == "coactivation":
        plotters = {
            "sorted-pmi-decay": plot_sorted_pmi_decay,
            "pmi-histogram": plot_pmi_histogram,
            "strong-coact-counts": plot_threshold_counts,
            "component-pair-heatmap": plot_component_pair_heatmap,
            "same-vs-cross-component": plot_same_cross_distribution,
            "component-signature-similarity": plot_component_signature_similarity,
            "latent-profile-pca": plot_latent_profile_pca,
            "profile-similarity-distribution": plot_profile_similarity_distribution,
            "top-coact-hubs": plot_top_coact_hubs,
            "component-bipartite-graph": plot_component_bipartite_graph,
            "latent-profile-clustering": plot_latent_profile_clustering,
            "jaccard-overlap": plot_jaccard_overlap,
            "mutual-coact-graph": plot_mutual_coact_graph,
            "hub-corrected-coacts": plot_hub_corrected_coacts,
            "mutual-neighbor-similarity": plot_mutual_neighbor_similarity,
            "top-ctx-logit-effect": plot_top_ctx_logit_effect,
            "top-ctx-sequence-overlap": plot_top_ctx_sequence_overlap,
            "coact-degrees": plot_coact_degrees,
        }
        if args.plot == "top-ctx-logit-effect":
            result = plot_top_ctx_logit_effect(
                args.run_root,
                output_root=args.output_root,
                sample_size=args.sample_size,
                top_ctx_batch_size=args.top_ctx_batch_size,
            )
        else:
            result = plotters[args.plot](args.run_root, output_root=args.output_root)
        output_paths = [result.figure_path, result.summary_path, result.table_path]
    else:
        raise ValueError(f"unsupported suite/plot combination: {args.suite}/{args.plot}")

    print("Generated analysis artifacts:")
    for path in output_paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

