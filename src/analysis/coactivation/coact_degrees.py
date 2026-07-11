"""Multi-hop "degrees of coactivation" graph analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import torch

from analysis.io import analysis_output_dirs, write_csv, write_json
from analysis.style import (
    BLUE,
    SERIES2,
    configure_matplotlib,
    integer_ticks,
    panel_figsize,
    round_bars,
    save_figure,
    style_suptitle,
    styled_legend,
)
from .data import TopCoactivationArtifact, load_top_coactivation
from .graph_utils import build_high_pmi_edges, high_pmi_in_degree
from .profile_utils import deterministic_sample_indices
from .sorted_pmi_decay import SUITE_NAME


@dataclass(frozen=True)
class CoactDegreesResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_coact_degrees(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    max_samples: int = 1_000,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
) -> CoactDegreesResult:
    """Generate sampled 1/2/3-hop coactivation reachability plots."""

    artifact = load_top_coactivation(run_root)
    if artifact.mode != "pmi":
        raise ValueError(f"coact degrees requires mode='pmi', got {artifact.mode!r}")

    stats = compute_coact_degrees(
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        max_samples=max_samples,
        top_out_degree=top_out_degree,
        max_frontier=max_frontier,
        hub_quantile=hub_quantile,
    )
    comparison_stats = None
    if hub_quantile < 1.0:
        comparison_stats = compute_coact_degrees(
            artifact.top_values,
            artifact.top_indices,
            d_sae=artifact.d_sae,
            threshold=threshold,
            max_samples=max_samples,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            hub_quantile=1.0,
        )
    output_dirs = analysis_output_dirs(run_root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "coact-degrees.png"
    comparison_figure_path = output_dirs["figures"] / "coact-degrees-hub-comparison.png"
    table_path = output_dirs["tables"] / "coact-degrees-sampled-sources.csv"
    summary_path = output_dirs["summaries"] / "coact-degrees.json"

    _write_plot(figure_path, stats)
    if comparison_stats is not None:
        _write_comparison_plot(comparison_figure_path, pruned_stats=stats, unpruned_stats=comparison_stats)
    _write_table(table_path, stats)
    summary = _build_summary(artifact, stats)
    if comparison_stats is not None:
        summary["hub_comparison_figure_path"] = str(comparison_figure_path)
        summary["unpruned_reach_summary"] = comparison_stats["reach_summary"]
        summary["unpruned_two_hop_same_summary"] = comparison_stats["two_hop_same_summary"]
        summary["unpruned_two_hop_cross_summary"] = comparison_stats["two_hop_cross_summary"]
    write_json(summary_path, summary)
    return CoactDegreesResult(figure_path, summary_path, table_path, summary)


def compute_coact_degrees(
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    max_samples: int = 1_000,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
) -> dict[str, object]:
    """Sample sources and expand high-PMI coact neighborhoods up to 3 hops."""

    if top_out_degree <= 0:
        raise ValueError("top_out_degree must be positive")
    if max_frontier <= 0:
        raise ValueError("max_frontier must be positive")
    if not 0.0 < hub_quantile <= 1.0:
        raise ValueError("hub_quantile must be in (0, 1]")

    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    num_latents = int(values.shape[0])
    num_components = int(top_values.shape[0])
    sample_indices = deterministic_sample_indices(num_latents, max_samples)
    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    in_degree = high_pmi_in_degree(edges)
    hub_cutoff = int(torch.quantile(in_degree.float(), torch.tensor(float(hub_quantile))).item())
    hub_cutoff = max(hub_cutoff, 1)
    cache: dict[int, torch.Tensor] = {}

    source_rows = []
    reach_by_hop: dict[int, list[int]] = {1: [], 2: [], 3: []}
    new_by_hop: dict[int, list[int]] = {1: [], 2: [], 3: []}
    same_component_by_hop: dict[int, list[float]] = {1: [], 2: [], 3: []}
    component_counts_by_hop = {hop: torch.zeros(num_components, dtype=torch.int64) for hop in (1, 2, 3)}
    hop_sets_by_source: list[dict[int, set[int]]] = []

    for source in sample_indices.tolist():
        source_id = int(source)
        source_component = source_id // int(d_sae)
        seen = {source_id}
        hop_sets: dict[int, set[int]] = {}
        frontier = {source_id}
        cumulative = set()

        for hop in (1, 2, 3):
            next_nodes = _expand_frontier(
                frontier,
                values,
                indices,
                in_degree,
                hub_cutoff=hub_cutoff,
                threshold=threshold,
                top_out_degree=top_out_degree,
                max_frontier=max_frontier,
                cache=cache,
            )
            next_nodes.difference_update(seen)
            hop_sets[hop] = set(next_nodes)
            seen.update(next_nodes)
            cumulative.update(next_nodes)
            frontier = set(list(next_nodes)[:max_frontier])

            components = [node // int(d_sae) for node in next_nodes]
            for component in components:
                component_counts_by_hop[hop][int(component)] += 1
            same_fraction = (
                sum(1 for component in components if component == source_component) / len(components)
                if components
                else 0.0
            )
            reach_by_hop[hop].append(len(cumulative))
            new_by_hop[hop].append(len(next_nodes))
            same_component_by_hop[hop].append(float(same_fraction))

        hop_sets_by_source.append(hop_sets)
        source_rows.append(
            {
                "source_global_id": source_id,
                "source_component": source_component,
                "source_latent": source_id % int(d_sae),
                "hop1_reachable": reach_by_hop[1][-1],
                "hop2_reachable": reach_by_hop[2][-1],
                "hop3_reachable": reach_by_hop[3][-1],
                "hop1_new": new_by_hop[1][-1],
                "hop2_new": new_by_hop[2][-1],
                "hop3_new": new_by_hop[3][-1],
            }
        )

    same_pair_scores, cross_pair_scores = _two_hop_pair_scores(sample_indices, hop_sets_by_source, in_degree, d_sae)
    return {
        "threshold": float(threshold),
        "sample_count": int(sample_indices.numel()),
        "top_out_degree": int(top_out_degree),
        "max_frontier": int(max_frontier),
        "hub_quantile": float(hub_quantile),
        "hub_cutoff_in_degree": int(hub_cutoff),
        "reach_by_hop": {str(hop): reach_by_hop[hop] for hop in (1, 2, 3)},
        "new_by_hop": {str(hop): new_by_hop[hop] for hop in (1, 2, 3)},
        "same_component_fraction_by_hop": {str(hop): same_component_by_hop[hop] for hop in (1, 2, 3)},
        "component_counts_by_hop": {str(hop): component_counts_by_hop[hop].tolist() for hop in (1, 2, 3)},
        "reach_summary": {str(hop): _summary(reach_by_hop[hop]) for hop in (1, 2, 3)},
        "new_summary": {str(hop): _summary(new_by_hop[hop]) for hop in (1, 2, 3)},
        "same_component_summary": {str(hop): _summary(same_component_by_hop[hop]) for hop in (1, 2, 3)},
        "two_hop_same_component_scores": same_pair_scores,
        "two_hop_cross_component_scores": cross_pair_scores,
        "two_hop_same_summary": _summary(same_pair_scores),
        "two_hop_cross_summary": _summary(cross_pair_scores),
        "sample_rows": source_rows,
        "limitation": (
            "This is a sampled graph-neighborhood analysis over high-PMI coacts. "
            "It is associative evidence, not a causal intervention result."
        ),
    }


def _expand_frontier(
    frontier: set[int],
    values: torch.Tensor,
    indices: torch.Tensor,
    in_degree: torch.Tensor,
    *,
    hub_cutoff: int,
    threshold: float,
    top_out_degree: int,
    max_frontier: int,
    cache: dict[int, torch.Tensor],
) -> set[int]:
    output: set[int] = set()
    for node in list(frontier)[:max_frontier]:
        neighbors = _neighbors(
            int(node),
            values,
            indices,
            in_degree,
            hub_cutoff=hub_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            cache=cache,
        )
        output.update(int(value) for value in neighbors.tolist())
        if len(output) >= max_frontier * top_out_degree:
            break
    return output


def _neighbors(
    node: int,
    values: torch.Tensor,
    indices: torch.Tensor,
    in_degree: torch.Tensor,
    *,
    hub_cutoff: int,
    threshold: float,
    top_out_degree: int,
    cache: dict[int, torch.Tensor],
) -> torch.Tensor:
    if node in cache:
        return cache[node]
    row_values = values[int(node)]
    row_indices = indices[int(node)]
    mask = (row_values > float(threshold)) & (in_degree[row_indices] <= int(hub_cutoff)) & (row_indices != int(node))
    if not bool(mask.any()):
        result = torch.empty(0, dtype=torch.int64)
    else:
        filtered_values = row_values[mask]
        filtered_indices = row_indices[mask]
        top_count = min(int(top_out_degree), int(filtered_indices.numel()))
        _, order = torch.topk(filtered_values, k=top_count)
        result = filtered_indices[order].to(torch.int64)
    cache[node] = result
    return result


def _two_hop_pair_scores(
    sample_indices: torch.Tensor,
    hop_sets_by_source: list[dict[int, set[int]]],
    in_degree: torch.Tensor,
    d_sae: int,
) -> tuple[list[float], list[float]]:
    components = (sample_indices // int(d_sae)).tolist()
    same_scores = []
    cross_scores = []
    for idx in range(max(0, len(hop_sets_by_source) - 1)):
        score = _hub_discounted_overlap(hop_sets_by_source[idx].get(2, set()), hop_sets_by_source[idx + 1].get(2, set()), in_degree)
        if components[idx] == components[idx + 1]:
            same_scores.append(score)
        else:
            cross_scores.append(score)
    if len(hop_sets_by_source) > 2:
        offset = max(len(hop_sets_by_source) // 3, 1)
        for idx in range(len(hop_sets_by_source)):
            other = (idx + offset) % len(hop_sets_by_source)
            score = _hub_discounted_overlap(hop_sets_by_source[idx].get(2, set()), hop_sets_by_source[other].get(2, set()), in_degree)
            if components[idx] == components[other]:
                same_scores.append(score)
            else:
                cross_scores.append(score)
    return same_scores, cross_scores


def _hub_discounted_overlap(left: set[int], right: set[int], in_degree: torch.Tensor) -> float:
    shared = left & right
    if not shared:
        return 0.0
    ids = torch.tensor(sorted(shared), dtype=torch.int64)
    weights = 1.0 / torch.log1p(in_degree[ids].float().clamp(min=1.0))
    return float(weights.sum().item())


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))

    reach_summary = stats["reach_summary"]
    new_summary = stats["new_summary"]
    same_component_summary = stats["same_component_summary"]
    assert isinstance(reach_summary, dict)
    assert isinstance(new_summary, dict)
    assert isinstance(same_component_summary, dict)
    hops = [1, 2, 3]
    labels = [f"{hop} hop" for hop in hops]
    reach_p50 = [reach_summary[str(hop)]["p50"] for hop in hops]
    reach_p90 = [reach_summary[str(hop)]["p90"] for hop in hops]
    x = range(len(hops))
    axes[0, 0].bar([pos - 0.18 for pos in x], reach_p50, width=0.3, color=SERIES2[0], label="p50")
    axes[0, 0].bar([pos + 0.18 for pos in x], reach_p90, width=0.3, color=SERIES2[1], label="p90")
    axes[0, 0].set_title("Unique Latents Reachable By Hop")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel("Cumulative reachable latents")
    styled_legend(axes[0, 0], loc="upper left")
    integer_ticks(axes[0, 0])

    new_p50 = [new_summary[str(hop)]["p50"] for hop in hops]
    new_p90 = [new_summary[str(hop)]["p90"] for hop in hops]
    axes[0, 1].bar([pos - 0.18 for pos in x], new_p50, width=0.3, color=SERIES2[0], label="p50")
    axes[0, 1].bar([pos + 0.18 for pos in x], new_p90, width=0.3, color=SERIES2[1], label="p90")
    axes[0, 1].set_title("New Latents Added At Each Hop")
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylabel("New reachable latents")
    styled_legend(axes[0, 1], loc="upper left")
    integer_ticks(axes[0, 1])

    same_component_mean = [same_component_summary[str(hop)]["mean"] for hop in hops]
    axes[1, 0].plot(labels, same_component_mean, marker="o", linewidth=2.0, color=BLUE)
    axes[1, 0].set_title("Component Locality By Hop")
    axes[1, 0].set_ylabel("Mean fraction in source component")
    axes[1, 0].set_ylim(bottom=0.0)

    same_scores = stats["two_hop_same_component_scores"]
    cross_scores = stats["two_hop_cross_component_scores"]
    assert isinstance(same_scores, list)
    assert isinstance(cross_scores, list)
    axes[1, 1].hist(same_scores, bins=50, alpha=0.6, color=SERIES2[0], label="same component")
    axes[1, 1].hist(cross_scores, bins=50, alpha=0.6, color=SERIES2[1], label="cross component")
    axes[1, 1].set_title("Hub-Discounted Shared 2-Hop Neighborhood")
    axes[1, 1].set_xlabel("Shared 2-hop neighbor score")
    axes[1, 1].set_ylabel("Sampled pair count")
    styled_legend(axes[1, 1], loc="upper right")

    round_bars(axes[0, 0])
    round_bars(axes[0, 1])
    style_suptitle(fig, "Degrees Of Coactivation")
    save_figure(fig, path)


def _write_comparison_plot(path: Path, *, pruned_stats: dict[str, object], unpruned_stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=panel_figsize(1, 3))
    hops = [1, 2, 3]
    labels = [f"{hop} hop" for hop in hops]
    x = range(len(hops))

    pruned_reach = pruned_stats["reach_summary"]
    unpruned_reach = unpruned_stats["reach_summary"]
    assert isinstance(pruned_reach, dict)
    assert isinstance(unpruned_reach, dict)
    axes[0].bar(
        [pos - 0.18 for pos in x],
        [pruned_reach[str(hop)]["p90"] for hop in hops],
        width=0.3,
        color=SERIES2[0],
        label="hub-pruned p90",
    )
    axes[0].bar(
        [pos + 0.18 for pos in x],
        [unpruned_reach[str(hop)]["p90"] for hop in hops],
        width=0.3,
        color=SERIES2[1],
        label="unpruned p90",
    )
    axes[0].set_title("Reachability P90")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Cumulative reachable latents")
    styled_legend(axes[0], loc="upper left")

    axes[1].bar(
        [pos - 0.18 for pos in x],
        [pruned_reach[str(hop)]["mean"] for hop in hops],
        width=0.3,
        color=SERIES2[0],
        label="hub-pruned mean",
    )
    axes[1].bar(
        [pos + 0.18 for pos in x],
        [unpruned_reach[str(hop)]["mean"] for hop in hops],
        width=0.3,
        color=SERIES2[1],
        label="unpruned mean",
    )
    axes[1].set_title("Reachability Mean")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Cumulative reachable latents")
    styled_legend(axes[1], loc="upper left")

    labels_2hop = ["same comp", "cross comp"]
    pruned_scores = [
        pruned_stats["two_hop_same_summary"]["max"],
        pruned_stats["two_hop_cross_summary"]["max"],
    ]
    unpruned_scores = [
        unpruned_stats["two_hop_same_summary"]["max"],
        unpruned_stats["two_hop_cross_summary"]["max"],
    ]
    x2 = range(len(labels_2hop))
    axes[2].bar([pos - 0.18 for pos in x2], pruned_scores, width=0.3, color=SERIES2[0], label="hub-pruned")
    axes[2].bar([pos + 0.18 for pos in x2], unpruned_scores, width=0.3, color=SERIES2[1], label="unpruned")
    axes[2].set_title("Strongest Shared 2-Hop Neighborhood")
    axes[2].set_xticks(list(x2))
    axes[2].set_xticklabels(labels_2hop)
    axes[2].set_ylabel("Max hub-discounted score")
    styled_legend(axes[2], loc="upper left")

    for axis in axes:
        round_bars(axis)
    style_suptitle(fig, "Hub-Pruned vs Unpruned Coactivation Degrees")
    save_figure(fig, path)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    write_csv(
        path,
        stats["sample_rows"],
        [
            "source_global_id",
            "source_component",
            "source_latent",
            "hop1_reachable",
            "hop2_reachable",
            "hop3_reachable",
            "hop1_new",
            "hop2_new",
            "hop3_new",
        ],
    )


def _build_summary(artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "sample_count": stats["sample_count"],
        "top_out_degree": stats["top_out_degree"],
        "max_frontier": stats["max_frontier"],
        "hub_quantile": stats["hub_quantile"],
        "hub_cutoff_in_degree": stats["hub_cutoff_in_degree"],
        "reach_summary": stats["reach_summary"],
        "new_summary": stats["new_summary"],
        "same_component_summary": stats["same_component_summary"],
        "two_hop_same_summary": stats["two_hop_same_summary"],
        "two_hop_cross_summary": stats["two_hop_cross_summary"],
        "component_counts_by_hop": stats["component_counts_by_hop"],
        "limitation": stats["limitation"],
    }


def _summary(values: list[float] | list[int]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(clean)
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "p50": float(median(clean)),
        "p90": float(_quantile(ordered, 0.90)),
        "max": float(max(clean)),
    }


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

