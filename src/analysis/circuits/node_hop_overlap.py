"""Exact overlap between circuit nodes and multi-hop coact neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch

from analysis.coactivation.coact_degrees import _expand_frontier, _summary
from analysis.coactivation.data import TopCoactivationArtifact, load_top_coactivation
from analysis.coactivation.graph_utils import build_high_pmi_edges, high_pmi_in_degree
from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import configure_matplotlib
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit
from .coact_overlap import SUITE_NAME

DEFAULT_KINDS = ("attn", "mlp", "resid")
_ACTIVATOR_ROLES = {"counterfactual_activator", "ablation_support"}
_INHIBITOR_ROLES = {"counterfactual_inhibitor"}


@dataclass(frozen=True)
class CircuitNodeHopOverlapResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_circuit_node_hop_overlap(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
    max_hops: int = 3,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> CircuitNodeHopOverlapResult:
    """Generate exact circuit-node recovery by multi-hop coact neighborhoods."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    artifact = load_top_coactivation(root)
    if artifact.mode != "pmi":
        raise ValueError(f"circuit node hop overlap requires mode='pmi', got {artifact.mode!r}")

    stats = compute_circuit_node_hop_overlap(
        circuits,
        artifact.top_values,
        artifact.top_indices,
        d_sae=artifact.d_sae,
        threshold=threshold,
        top_out_degree=top_out_degree,
        max_frontier=max_frontier,
        hub_quantile=hub_quantile,
        max_hops=max_hops,
        kinds=kinds,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "circuit-node-hop-overlap.png"
    table_path = output_dirs["tables"] / "circuit-node-hop-overlap.csv"
    summary_path = output_dirs["summaries"] / "circuit-node-hop-overlap.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(store_path, artifact, stats)
    write_json(summary_path, summary)
    return CircuitNodeHopOverlapResult(figure_path, summary_path, table_path, summary)


def resolve_circuit_store_path(run_root: str | Path, circuit_store_path: str | Path | None = None) -> Path:
    """Resolve the full circuit store path needed for exact node overlap."""

    if circuit_store_path is not None:
        path = Path(circuit_store_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"circuit store path does not exist: {path}")
        return path

    root = resolve_run_root(run_root)
    candidates = [
        root / "circuits" / "discovered_circuits.pt",
        root / "distributed" / "circuits" / "discovered_circuits.pt",
        root / "distributed" / "parts" / "discovery" / "circuits" / "discovered_circuits.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find discovered_circuits.pt under the run root. "
        "Pass --circuit-store PATH pointing to the full circuit store."
    )


def load_circuit_store(path: str | Path) -> dict[str, Circuit]:
    """Load a torch-saved circuit store mapping."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"circuit store must be a mapping, got {type(payload).__name__}")
    circuits = {str(key): value for key, value in payload.items() if isinstance(value, Circuit)}
    if not circuits:
        raise ValueError(f"no Circuit objects found in circuit store: {path}")
    return circuits


def compute_circuit_node_hop_overlap(
    circuits: Mapping[str, Circuit],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    d_sae: int,
    threshold: float = 2.0,
    top_out_degree: int = 12,
    max_frontier: int = 256,
    hub_quantile: float = 0.99,
    max_hops: int = 3,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Measure recovery of actual circuit latent nodes from seed coact hops."""

    if max_hops <= 0:
        raise ValueError("max_hops must be positive")
    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    edges = build_high_pmi_edges(top_values, top_indices, threshold=threshold)
    in_degree = high_pmi_in_degree(edges)
    hub_cutoff = int(torch.quantile(in_degree.float(), torch.tensor(float(hub_quantile))).item())
    hub_cutoff = max(hub_cutoff, 1)
    n_kinds = len(kinds)
    cache: dict[int, torch.Tensor] = {}
    hops = list(range(1, int(max_hops) + 1))

    rows = []
    all_pct_by_hop: dict[int, list[float]] = {hop: [] for hop in hops}
    activator_pct_by_hop: dict[int, list[float]] = {hop: [] for hop in hops}
    inhibitor_pct_by_hop: dict[int, list[float]] = {hop: [] for hop in hops}

    for circuit in circuits.values():
        seed_comp = circuit.metadata.get("seed_comp")
        seed_latent = circuit.metadata.get("seed_latent")
        if seed_comp is None or seed_latent is None:
            continue
        seed_global_id = int(seed_comp) * int(d_sae) + int(seed_latent)
        if seed_global_id < 0 or seed_global_id >= values.shape[0]:
            continue
        node_sets = _circuit_node_sets(circuit, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds, seed_global_id=seed_global_id)
        if not node_sets["all"]:
            continue
        reachable_by_hop = _reachable_sets(
            seed_global_id,
            values,
            indices,
            in_degree,
            hub_cutoff=hub_cutoff,
            threshold=threshold,
            top_out_degree=top_out_degree,
            max_frontier=max_frontier,
            max_hops=max_hops,
            cache=cache,
        )
        row: dict[str, Any] = {
            "uuid": circuit.uuid,
            "name": circuit.name,
            "seed_global_id": seed_global_id,
            "seed_comp": int(seed_comp),
            "seed_latent": int(seed_latent),
            "circuit_latent_count": len(node_sets["all"]),
            "activator_count": len(node_sets["activator"]),
            "inhibitor_count": len(node_sets["inhibitor"]),
            "counterfactual_faithfulness": _metadata_float(circuit.metadata, ("evals", "counterfactual_faithfulness")),
        }
        for hop in hops:
            reachable = reachable_by_hop[hop]
            _add_overlap_columns(row, "all", hop, reachable, node_sets["all"])
            _add_overlap_columns(row, "activator", hop, reachable, node_sets["activator"])
            _add_overlap_columns(row, "inhibitor", hop, reachable, node_sets["inhibitor"])
            all_pct_by_hop[hop].append(float(row[f"all_hop{hop}_pct"]))
            activator_pct_by_hop[hop].append(float(row[f"activator_hop{hop}_pct"]))
            inhibitor_pct_by_hop[hop].append(float(row[f"inhibitor_hop{hop}_pct"]))
        rows.append(row)

    return {
        "threshold": float(threshold),
        "top_out_degree": int(top_out_degree),
        "max_frontier": int(max_frontier),
        "max_hops": int(max_hops),
        "hub_quantile": float(hub_quantile),
        "hub_cutoff_in_degree": int(hub_cutoff),
        "circuit_count": len(rows),
        "all_pct_summary": {str(hop): _summary(all_pct_by_hop[hop]) for hop in hops},
        "activator_pct_summary": {str(hop): _summary(activator_pct_by_hop[hop]) for hop in hops},
        "inhibitor_pct_summary": {str(hop): _summary(inhibitor_pct_by_hop[hop]) for hop in hops},
        "rows": rows,
        "top_recovered": sorted(rows, key=lambda row: float(row[f"all_hop{max_hops}_pct"]), reverse=True)[:50],
    }


def _circuit_node_sets(
    circuit: Circuit,
    *,
    n_kinds: int,
    d_sae: int,
    kinds: Sequence[str],
    seed_global_id: int,
) -> dict[str, set[int]]:
    all_nodes: set[int] = set()
    activators: set[int] = set()
    inhibitors: set[int] = set()
    for node in circuit.nodes.values():
        fid = node.feature_id
        if not _eligible_feature(fid, kinds):
            continue
        assert fid is not None
        gid = fid.to_global_id(n_kinds, d_sae, kinds)
        if gid == int(seed_global_id):
            continue
        all_nodes.add(gid)
        role = str(node.metadata.get("role", ""))
        if role in _ACTIVATOR_ROLES:
            activators.add(gid)
        elif role in _INHIBITOR_ROLES:
            inhibitors.add(gid)
    return {"all": all_nodes, "activator": activators, "inhibitor": inhibitors}


def _eligible_feature(fid: FeatureID | None, kinds: Sequence[str]) -> bool:
    if fid is None:
        return False
    if fid.kind in ("logit", "token"):
        return False
    if fid.kind.endswith("_err"):
        return False
    return fid.kind in kinds


def _reachable_sets(
    source: int,
    values: torch.Tensor,
    indices: torch.Tensor,
    in_degree: torch.Tensor,
    *,
    hub_cutoff: int,
    threshold: float,
    top_out_degree: int,
    max_frontier: int,
    max_hops: int,
    cache: dict[int, torch.Tensor],
) -> dict[int, set[int]]:
    seen = {int(source)}
    frontier = {int(source)}
    cumulative: set[int] = set()
    reachable = {}
    for hop in range(1, int(max_hops) + 1):
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
        seen.update(next_nodes)
        cumulative.update(next_nodes)
        frontier = set(list(next_nodes)[:max_frontier])
        reachable[hop] = set(cumulative)
    return reachable


def _add_overlap_columns(row: dict[str, Any], prefix: str, hop: int, reachable: set[int], target: set[int]) -> None:
    count = len(reachable & target)
    total = len(target)
    row[f"{prefix}_hop{hop}_count"] = count
    row[f"{prefix}_hop{hop}_pct"] = (count / total * 100.0) if total else 0.0


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    max_hops = int(stats["max_hops"])
    hops = list(range(1, max_hops + 1))
    labels = [f"{hop} hop" for hop in hops]
    x = range(len(hops))

    all_summary = stats["all_pct_summary"]
    activator_summary = stats["activator_pct_summary"]
    inhibitor_summary = stats["inhibitor_pct_summary"]
    rows = stats["rows"]
    assert isinstance(all_summary, dict)
    assert isinstance(activator_summary, dict)
    assert isinstance(inhibitor_summary, dict)
    assert isinstance(rows, list)

    axes[0, 0].bar([pos - 0.18 for pos in x], [all_summary[str(hop)]["p50"] for hop in hops], width=0.36, color="#2f6f9f", alpha=0.85, label="p50")
    axes[0, 0].bar([pos + 0.18 for pos in x], [all_summary[str(hop)]["p90"] for hop in hops], width=0.36, color="#b45f06", alpha=0.85, label="p90")
    axes[0, 0].set_title("Circuit Latents Recovered By Coact Hops")
    axes[0, 0].set_ylabel("Circuit latents recovered (%)")
    axes[0, 0].set_xticks(list(x))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].legend(loc="upper left")

    axes[0, 1].plot(labels, [all_summary[str(hop)]["mean"] for hop in hops], marker="o", linewidth=2.0, label="all nodes", color="#2f6f9f")
    axes[0, 1].plot(labels, [activator_summary[str(hop)]["mean"] for hop in hops], marker="o", linewidth=2.0, label="activators", color="#b45f06")
    axes[0, 1].plot(labels, [inhibitor_summary[str(hop)]["mean"] for hop in hops], marker="o", linewidth=2.0, label="inhibitors", color="#38761d")
    axes[0, 1].set_title("Mean Recovery By Role")
    axes[0, 1].set_ylabel("Circuit latents recovered (%)")
    axes[0, 1].legend(loc="upper left")

    selected_hops = sorted({1, min(3, max_hops), max_hops})
    colors = ["#2f6f9f", "#b45f06", "#38761d"]
    for hop, color in zip(selected_hops, colors):
        axes[1, 0].hist(
            [float(row[f"all_hop{hop}_pct"]) for row in rows],
            bins=50,
            alpha=0.65,
            label=f"{hop} hop",
            color=color,
        )
    axes[1, 0].set_title("Per-Circuit Recovery Distribution")
    axes[1, 0].set_xlabel("Circuit latents recovered (%)")
    axes[1, 0].set_ylabel("Circuit count")
    axes[1, 0].legend(loc="upper right")

    faithfulness = [float(row["counterfactual_faithfulness"]) for row in rows]
    final_hop = [float(row[f"all_hop{max_hops}_pct"]) for row in rows]
    sizes = [max(8.0, min(float(row["circuit_latent_count"]) / 4.0, 80.0)) for row in rows]
    scatter = axes[1, 1].scatter(
        final_hop,
        faithfulness,
        s=sizes,
        c=[float(row[f"inhibitor_hop{max_hops}_pct"]) for row in rows],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    axes[1, 1].set_title(f"{max_hops}-Hop Circuit Recovery vs Faithfulness")
    axes[1, 1].set_xlabel(f"Circuit latents recovered within {max_hops} hops (%)")
    axes[1, 1].set_ylabel("Counterfactual faithfulness")
    fig.colorbar(scatter, ax=axes[1, 1], label=f"Inhibitor recovery within {max_hops} hops (%)")

    fig.suptitle("Exact Circuit Node Recovery From Coact Hops", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    fieldnames = [
        "uuid",
        "name",
        "seed_global_id",
        "seed_comp",
        "seed_latent",
        "circuit_latent_count",
        "activator_count",
        "inhibitor_count",
        "counterfactual_faithfulness",
    ]
    for prefix in ("all", "activator", "inhibitor"):
        for hop in range(1, int(stats["max_hops"]) + 1):
            fieldnames.extend([f"{prefix}_hop{hop}_count", f"{prefix}_hop{hop}_pct"])
    write_csv(path, stats["rows"], fieldnames)


def _build_summary(store_path: Path, artifact: TopCoactivationArtifact, stats: dict[str, object]) -> dict[str, object]:
    return {
        "circuit_store_path": str(store_path),
        "artifact_path": str(artifact.path),
        "mode": artifact.mode,
        "shape": list(artifact.shape),
        "threshold": stats["threshold"],
        "top_out_degree": stats["top_out_degree"],
        "max_frontier": stats["max_frontier"],
        "max_hops": stats["max_hops"],
        "hub_quantile": stats["hub_quantile"],
        "hub_cutoff_in_degree": stats["hub_cutoff_in_degree"],
        "circuit_count": stats["circuit_count"],
        "all_pct_summary": stats["all_pct_summary"],
        "activator_pct_summary": stats["activator_pct_summary"],
        "inhibitor_pct_summary": stats["inhibitor_pct_summary"],
        "top_recovered": stats["top_recovered"][:20],
    }


def _metadata_float(metadata: Mapping[str, Any], path: tuple[str, ...]) -> float:
    current: Any = metadata
    for key in path:
        if not isinstance(current, Mapping):
            return 0.0
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return 0.0

