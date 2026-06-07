"""Circuit latent commonality distributions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import configure_matplotlib
from circuit.types.feature_id import FeatureID
from store.circuits import Circuit

from .coact_overlap import SUITE_NAME
from .node_hop_overlap import DEFAULT_KINDS, load_circuit_store, resolve_circuit_store_path


FeatureKey = tuple[int, str, int]


@dataclass(frozen=True)
class CircuitLatentCommonalityResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_circuit_latent_commonality(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    rare_max_pct: float = 5.0,
    common_min_pct: float = 15.0,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> CircuitLatentCommonalityResult:
    """Generate commonality plots for exact latent reuse across circuits."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    stats = compute_circuit_latent_commonality(
        circuits,
        rare_max_pct=rare_max_pct,
        common_min_pct=common_min_pct,
        kinds=kinds,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "circuit-latent-commonality.png"
    table_path = output_dirs["tables"] / "circuit-latent-commonality.csv"
    summary_path = output_dirs["summaries"] / "circuit-latent-commonality.json"

    _write_plot(figure_path, stats)
    _write_table(table_path, stats)
    summary = _build_summary(store_path, stats)
    write_json(summary_path, summary)
    return CircuitLatentCommonalityResult(figure_path, summary_path, table_path, summary)


def compute_circuit_latent_commonality(
    circuits: Mapping[str, Circuit],
    *,
    rare_max_pct: float = 5.0,
    common_min_pct: float = 15.0,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Count how often exact latent identities recur across discovered circuits."""

    if rare_max_pct <= 0.0:
        raise ValueError("rare_max_pct must be positive")
    if common_min_pct <= rare_max_pct:
        raise ValueError("common_min_pct must be greater than rare_max_pct")
    if common_min_pct > 100.0:
        raise ValueError("common_min_pct must be at most 100")

    n_kinds = len(kinds)
    circuit_latents: dict[str, set[FeatureKey]] = {}
    circuit_lookup: dict[str, Circuit] = {}
    role_counts: dict[FeatureKey, Counter[str]] = defaultdict(Counter)

    for circuit in circuits.values():
        latents = _circuit_latents(circuit, kinds=kinds, n_kinds=n_kinds)
        if not latents:
            continue
        circuit_latents[circuit.uuid] = {key for key, _role in latents}
        circuit_lookup[circuit.uuid] = circuit
        for key, role in latents:
            role_counts[key][role] += 1

    latent_counts: Counter[FeatureKey] = Counter()
    for latents in circuit_latents.values():
        latent_counts.update(latents)

    circuit_count = len(circuit_latents)
    latent_rows = [_latent_row(key, count, role_counts[key], circuit_count=circuit_count) for key, count in latent_counts.items()]
    latent_rows.sort(key=lambda row: (-int(row["circuit_count"]), int(row["layer"]), str(row["kind"]), int(row["latent_idx"])))

    circuit_rows: list[dict[str, Any]] = []
    for uuid, latents in circuit_latents.items():
        circuit = circuit_lookup[uuid]
        counts = [int(latent_counts[key]) for key in latents]
        bucket_counts = _bucket_counts(
            counts,
            circuit_count=circuit_count,
            rare_max_pct=rare_max_pct,
            common_min_pct=common_min_pct,
        )
        total = len(counts)
        circuit_rows.append(
            {
                "uuid": circuit.uuid,
                "name": circuit.name,
                "seed_comp": _metadata_int(circuit.metadata, "seed_comp"),
                "seed_latent": _metadata_int(circuit.metadata, "seed_latent"),
                "circuit_latent_count": total,
                "singleton_latent_count": bucket_counts["singleton"],
                "rare_latent_count": bucket_counts["rare"],
                "shared_latent_count": bucket_counts["shared"],
                "common_latent_count": bucket_counts["common"],
                "singleton_latent_pct": _pct(bucket_counts["singleton"], total),
                "rare_latent_pct": _pct(bucket_counts["rare"], total),
                "shared_latent_pct": _pct(bucket_counts["shared"], total),
                "common_latent_pct": _pct(bucket_counts["common"], total),
                "mean_commonality": float(mean(counts)) if counts else 0.0,
                "median_commonality": float(median(counts)) if counts else 0.0,
                "max_commonality": max(counts) if counts else 0,
                "mean_commonality_pct": float(mean(_commonality_pct(count, circuit_count) for count in counts)) if counts else 0.0,
                "median_commonality_pct": float(median(_commonality_pct(count, circuit_count) for count in counts))
                if counts
                else 0.0,
                "max_commonality_pct": max(_commonality_pct(count, circuit_count) for count in counts) if counts else 0.0,
                "counterfactual_faithfulness": _metadata_float(circuit.metadata, ("evals", "counterfactual_faithfulness")),
            }
        )
    circuit_rows.sort(key=lambda row: str(row["uuid"]))

    commonality_counts = [int(row["circuit_count"]) for row in latent_rows]
    commonality_pcts = [_commonality_pct(count, circuit_count) for count in commonality_counts]
    per_circuit_means = [float(row["mean_commonality"]) for row in circuit_rows]
    per_circuit_mean_pcts = [float(row["mean_commonality_pct"]) for row in circuit_rows]
    per_circuit_common_pct = [float(row["common_latent_pct"]) for row in circuit_rows]

    return {
        "rare_max_pct": float(rare_max_pct),
        "common_min_pct": float(common_min_pct),
        "circuit_count": len(circuit_rows),
        "unique_latent_count": len(latent_rows),
        "latent_commonality_counts": commonality_counts,
        "latent_commonality_pcts": commonality_pcts,
        "latent_commonality_summary": _summary(commonality_counts),
        "latent_commonality_pct_summary": _summary(commonality_pcts),
        "per_circuit_mean_commonality_summary": _summary(per_circuit_means),
        "per_circuit_mean_commonality_pct_summary": _summary(per_circuit_mean_pcts),
        "per_circuit_common_pct_summary": _summary(per_circuit_common_pct),
        "bucket_latent_counts": _global_bucket_counts(
            commonality_counts,
            circuit_count=circuit_count,
            rare_max_pct=rare_max_pct,
            common_min_pct=common_min_pct,
        ),
        "latent_rows": latent_rows,
        "circuit_rows": circuit_rows,
    }


def _circuit_latents(circuit: Circuit, *, kinds: Sequence[str], n_kinds: int) -> set[tuple[FeatureKey, str]]:
    seed_comp = circuit.metadata.get("seed_comp")
    seed_latent = circuit.metadata.get("seed_latent")
    latents: set[tuple[FeatureKey, str]] = set()
    for node in circuit.nodes.values():
        fid = node.feature_id
        if not _eligible_feature(fid, kinds):
            continue
        assert fid is not None
        role = str(node.metadata.get("role", ""))
        if role == "seed" or _is_seed_feature(fid, seed_comp, seed_latent, n_kinds, kinds):
            continue
        latents.add((fid.key, role or "unknown"))
    return latents


def _eligible_feature(fid: FeatureID | None, kinds: Sequence[str]) -> bool:
    if fid is None:
        return False
    if fid.kind in ("logit", "token"):
        return False
    if fid.kind.endswith("_err"):
        return False
    return fid.kind in kinds


def _is_seed_feature(
    fid: FeatureID,
    seed_comp: object,
    seed_latent: object,
    n_kinds: int,
    kinds: Sequence[str],
) -> bool:
    if seed_comp is None or seed_latent is None:
        return False
    try:
        comp_idx, latent_idx = fid.to_component_id(n_kinds, kinds)
        return comp_idx == int(seed_comp) and latent_idx == int(seed_latent)
    except (TypeError, ValueError):
        return False


def _latent_row(key: FeatureKey, count: int, roles: Counter[str], *, circuit_count: int) -> dict[str, Any]:
    layer, kind, latent_idx = key
    role_total = sum(roles.values())
    primary_role = "unknown"
    if role_total:
        primary_role = max(roles.items(), key=lambda item: (item[1], item[0]))[0]
    return {
        "layer": int(layer),
        "kind": str(kind),
        "latent_idx": int(latent_idx),
        "latent": f"L{layer}.{kind}.f{latent_idx}",
        "circuit_count": int(count),
        "circuit_pct": _commonality_pct(int(count), circuit_count),
        "primary_role": primary_role,
        "activator_appearances": int(roles.get("counterfactual_activator", 0) + roles.get("ablation_support", 0)),
        "inhibitor_appearances": int(roles.get("counterfactual_inhibitor", 0)),
        "other_appearances": int(role_total - roles.get("counterfactual_activator", 0) - roles.get("ablation_support", 0) - roles.get("counterfactual_inhibitor", 0)),
    }


def _bucket_counts(
    counts: list[int],
    *,
    circuit_count: int,
    rare_max_pct: float,
    common_min_pct: float,
) -> dict[str, int]:
    buckets = {"singleton": 0, "rare": 0, "shared": 0, "common": 0}
    for count in counts:
        commonality_pct = _commonality_pct(count, circuit_count)
        if count <= 1:
            buckets["singleton"] += 1
        elif commonality_pct <= rare_max_pct:
            buckets["rare"] += 1
        elif commonality_pct < common_min_pct:
            buckets["shared"] += 1
        else:
            buckets["common"] += 1
    return buckets


def _global_bucket_counts(
    counts: list[int],
    *,
    circuit_count: int,
    rare_max_pct: float,
    common_min_pct: float,
) -> dict[str, int]:
    return _bucket_counts(
        counts,
        circuit_count=circuit_count,
        rare_max_pct=rare_max_pct,
        common_min_pct=common_min_pct,
    )


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    commonality = [int(value) for value in stats["latent_commonality_counts"]]
    circuit_rows = stats["circuit_rows"]
    assert isinstance(circuit_rows, list)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if commonality:
        bins = range(1, max(commonality) + 2)
        axes[0, 0].hist(commonality, bins=bins, color="#2f6f9f", alpha=0.85)
        axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Latent Commonality Across Circuits")
    axes[0, 0].set_xlabel("Circuits containing latent")
    axes[0, 0].set_ylabel("Unique latent count (log)")

    _plot_cdf(axes[0, 1], commonality)
    axes[0, 1].set_title("Cumulative Commonality Distribution")
    axes[0, 1].set_xlabel("Circuits containing latent")
    axes[0, 1].set_ylabel("Fraction of unique latents")

    bucket_labels = ["singleton", "rare", "shared", "common"]
    bucket_colors = ["#7f7f7f", "#2f6f9f", "#b45f06", "#38761d"]
    positions = range(len(circuit_rows))
    bottoms = [0.0 for _ in circuit_rows]
    for label, color in zip(bucket_labels, bucket_colors):
        values = [float(row[f"{label}_latent_pct"]) for row in circuit_rows]
        axes[1, 0].bar(positions, values, bottom=bottoms, width=1.0, color=color, alpha=0.85, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axes[1, 0].set_title("Per-Circuit Composition By Latent Rarity")
    axes[1, 0].set_xlabel("Circuit, sorted by UUID")
    axes[1, 0].set_ylabel("Circuit latents (%)")
    axes[1, 0].set_xticks([])
    axes[1, 0].legend(loc="upper right")

    sizes = [max(10.0, min(float(row["circuit_latent_count"]) / 4.0, 90.0)) for row in circuit_rows]
    scatter = axes[1, 1].scatter(
        [float(row["mean_commonality"]) for row in circuit_rows],
        [float(row["counterfactual_faithfulness"]) for row in circuit_rows],
        s=sizes,
        c=[float(row["common_latent_pct"]) for row in circuit_rows],
        cmap="viridis",
        alpha=0.75,
        edgecolors="none",
    )
    axes[1, 1].set_title("Common-Latent Composition vs Faithfulness")
    axes[1, 1].set_xlabel("Mean latent commonality in circuit")
    axes[1, 1].set_ylabel("Counterfactual faithfulness")
    fig.colorbar(scatter, ax=axes[1, 1], label="Common latents in circuit (%)")

    fig.suptitle("Latent Commonality Between Discovered Circuits", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_cdf(ax: Any, values: list[int]) -> None:
    if not values:
        return
    ordered = sorted(values)
    y = [(idx + 1) / len(ordered) for idx in range(len(ordered))]
    ax.step(ordered, y, where="post", color="#b45f06", linewidth=2.0)
    ax.set_ylim(0.0, 1.02)


def _write_table(path: Path, stats: dict[str, object]) -> None:
    rows = stats["circuit_rows"]
    assert isinstance(rows, list)
    write_csv(
        path,
        rows,
        [
            "uuid",
            "name",
            "seed_comp",
            "seed_latent",
            "circuit_latent_count",
            "singleton_latent_count",
            "rare_latent_count",
            "shared_latent_count",
            "common_latent_count",
            "singleton_latent_pct",
            "rare_latent_pct",
            "shared_latent_pct",
            "common_latent_pct",
            "mean_commonality",
            "median_commonality",
            "max_commonality",
            "mean_commonality_pct",
            "median_commonality_pct",
            "max_commonality_pct",
            "counterfactual_faithfulness",
        ],
    )


def _build_summary(store_path: Path, stats: dict[str, object]) -> dict[str, object]:
    latent_rows = stats["latent_rows"]
    assert isinstance(latent_rows, list)
    return {
        "circuit_store_path": str(store_path),
        "rare_max_pct": stats["rare_max_pct"],
        "common_min_pct": stats["common_min_pct"],
        "circuit_count": stats["circuit_count"],
        "unique_latent_count": stats["unique_latent_count"],
        "latent_commonality_summary": stats["latent_commonality_summary"],
        "latent_commonality_pct_summary": stats["latent_commonality_pct_summary"],
        "per_circuit_mean_commonality_summary": stats["per_circuit_mean_commonality_summary"],
        "per_circuit_mean_commonality_pct_summary": stats["per_circuit_mean_commonality_pct_summary"],
        "per_circuit_common_pct_summary": stats["per_circuit_common_pct_summary"],
        "bucket_latent_counts": stats["bucket_latent_counts"],
        "top_common_latents": latent_rows[:50],
        "note": (
            "Seed latents are excluded so commonality reflects non-seed circuit composition. "
            "Singleton means exactly one circuit; rare/shared/common use percent of analyzed circuits."
        ),
    }


def _summary(values: list[int] | list[float]) -> dict[str, float | int]:
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


def _pct(count: int, total: int) -> float:
    return (float(count) / float(total) * 100.0) if total else 0.0


def _commonality_pct(count: int, circuit_count: int) -> float:
    return _pct(int(count), int(circuit_count))


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    return int(value) if isinstance(value, (int, float)) else -1


def _metadata_float(metadata: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = metadata
    for key in path:
        if not isinstance(value, Mapping):
            return 0.0
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0
