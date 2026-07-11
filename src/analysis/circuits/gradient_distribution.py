"""Compare seed-gradient distributions for coact, circuit, and random latents."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch

from analysis.coactivation.data import load_top_coactivation
from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import (
    BLUE,
    INK_MUTED,
    SERIES2,
    configure_matplotlib,
    panel_figsize,
    save_figure,
    style_suptitle,
    styled_boxplot,
    styled_legend,
)
from circuit.instrument.attribution import compute_feature_gradient
from circuit.instrument.sae_graph import SAEGraphInstrument
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.types.feature_id import FeatureID
from config import config
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit
from store.context import mid_ctx, neg_ctx, top_ctx

from .coact_overlap import SUITE_NAME
from .node_hop_overlap import DEFAULT_KINDS, load_circuit_store, resolve_circuit_store_path


FIELDS = [
    "uuid",
    "name",
    "sample_index",
    "seed_global_id",
    "seed_comp",
    "seed_latent",
    "seed_layer",
    "seed_kind",
    "group",
    "match_group",
    "latent_global_id",
    "latent_comp",
    "latent_layer",
    "latent_kind",
    "latent_idx",
    "role",
    "coact_rank",
    "coact_score",
    "gradient",
    "abs_gradient",
    "log10_abs_gradient",
    "sign",
    "present_in_gradient_result",
]


@dataclass(frozen=True)
class GradientDistributionResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_gradient_distribution(
    run_root: str | Path,
    *,
    circuit_store_path: str | Path | None = None,
    output_root: str | Path | None = None,
    sample_size: int = 128,
    threshold: float = 2.0,
    max_coact_nodes: int = 64,
    random_per_observed: int = 1,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> GradientDistributionResult:
    """Generate gradient-distribution plots for coact, circuit, and random latents."""

    root = resolve_run_root(run_root)
    store_path = resolve_circuit_store_path(root, circuit_store_path)
    circuits = load_circuit_store(store_path)
    artifact = load_top_coactivation(root)
    if artifact.mode != "pmi":
        raise ValueError(f"gradient distribution requires mode='pmi', got {artifact.mode!r}")

    print("Loading discovery artifacts...")
    load_discovery_artifacts(root)
    print("Initializing model/SAE resources...")
    devices = detect_devices()
    device = devices[0]
    fast = is_fast_memory()
    compile_model = should_compile()
    loader = DataLoader(device=device, pin_memory=fast)
    inference = Inference(device=device, compile=compile_model)
    bank = SAEBank(devices=devices, load_decoders=fast, compile=compile_model)
    probe_builder = ProbeDatasetBuilder(inference, bank, loader)

    stats = compute_gradient_distribution(
        circuits,
        artifact.top_values,
        artifact.top_indices,
        inference=inference,
        bank=bank,
        probe_builder=probe_builder,
        d_sae=artifact.d_sae,
        sample_size=sample_size,
        threshold=threshold,
        max_coact_nodes=max_coact_nodes,
        random_per_observed=random_per_observed,
        kinds=kinds,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "gradient-distribution.png"
    table_path = output_dirs["tables"] / "gradient-distribution.csv"
    summary_path = output_dirs["summaries"] / "gradient-distribution.json"

    _write_plot(figure_path, stats)
    write_csv(table_path, stats["rows"], FIELDS)
    summary = _build_summary(store_path, artifact.path, stats)
    write_json(summary_path, summary)
    return GradientDistributionResult(figure_path, summary_path, table_path, summary)


def compute_gradient_distribution(
    circuits: Mapping[str, Circuit],
    top_values: torch.Tensor,
    top_indices: torch.Tensor,
    *,
    inference: Inference,
    bank: SAEBank,
    probe_builder: ProbeDatasetBuilder,
    d_sae: int,
    sample_size: int = 128,
    threshold: float = 2.0,
    max_coact_nodes: int = 64,
    random_per_observed: int = 1,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Compute raw seed gradients for coact, circuit, and matched random latents."""

    values = top_values.detach().cpu().to(torch.float32).reshape(-1, top_values.shape[-1])
    indices = top_indices.detach().cpu().to(torch.int64).reshape(-1, top_indices.shape[-1])
    sampled = _deterministic_circuit_sample(
        [
            circuit
            for circuit in circuits.values()
            if circuit.metadata.get("seed_comp") is not None and circuit.metadata.get("seed_latent") is not None
        ],
        sample_size,
    )
    n_kinds = len(kinds)
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sample_index, circuit in enumerate(sampled):
        seed_comp = int(circuit.metadata["seed_comp"])
        seed_latent = int(circuit.metadata["seed_latent"])
        seed_layer, seed_kind_idx = split_component_idx(seed_comp, n_kinds)
        if seed_kind_idx >= len(kinds):
            skipped.append({"uuid": circuit.uuid, "reason": "seed kind index out of range"})
            continue
        seed_kind = kinds[seed_kind_idx]
        seed_gid = seed_comp * int(d_sae) + seed_latent
        if seed_gid < 0 or seed_gid >= values.shape[0]:
            skipped.append({"uuid": circuit.uuid, "reason": "seed global id out of range"})
            continue

        try:
            probe = probe_builder.build_for_latent(seed_comp, seed_latent, top_ctx, mid_ctx, neg_ctx)
        except Exception as exc:
            skipped.append({"uuid": circuit.uuid, "reason": f"probe build failed: {exc}"})
            continue
        n_pos = min(int(config.discovery.probe_batch_size or 16), int(probe.pos_tokens.shape[0]))
        if n_pos <= 0:
            skipped.append({"uuid": circuit.uuid, "reason": "empty positive probe"})
            continue
        pos_tokens = probe.pos_tokens[:n_pos].to(bank.device)
        pos_argmax = probe.pos_argmax[:n_pos].to(bank.device)

        coact_meta = _coact_candidates_for_seed(
            values,
            indices,
            seed_gid,
            threshold=threshold,
            max_coact_nodes=max_coact_nodes,
        )
        circuit_meta = _circuit_candidates_for_seed(
            circuit,
            seed_gid=seed_gid,
            seed_layer=seed_layer,
            n_kinds=n_kinds,
            d_sae=d_sae,
            kinds=kinds,
        )
        coact_meta = {
            gid: meta
            for gid, meta in coact_meta.items()
            if _is_gradient_addressable(gid, seed_layer=seed_layer, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds)
        }
        circuit_meta = {
            gid: meta
            for gid, meta in circuit_meta.items()
            if _is_gradient_addressable(gid, seed_layer=seed_layer, n_kinds=n_kinds, d_sae=d_sae, kinds=kinds)
        }

        excluded = {seed_gid, *coact_meta.keys(), *circuit_meta.keys()}
        random_meta = _matched_random_candidates(
            seed_gid,
            coact_meta,
            circuit_meta,
            excluded,
            n_kinds=n_kinds,
            d_sae=d_sae,
            kinds=kinds,
            random_per_observed=random_per_observed,
        )
        candidate_meta: dict[int, dict[str, Any]] = {}
        for gid, meta in coact_meta.items():
            candidate_meta[gid] = dict(meta, group="coact", match_group="")
        for gid, meta in circuit_meta.items():
            existing = candidate_meta.get(gid)
            if existing is None:
                candidate_meta[gid] = dict(meta, group="circuit", match_group="")
            else:
                candidate_meta[gid] = dict(existing, group="coact+circuit", role=meta.get("role", ""))
        for gid, meta in random_meta.items():
            candidate_meta[gid] = dict(meta, group="random")

        candidate_fids = [
            FeatureID.from_global_id(gid, n_kinds, d_sae, kinds)
            for gid in sorted(candidate_meta)
        ]
        gradients: dict[FeatureID, float] = {}
        error = ""
        if candidate_fids:
            instrument = SAEGraphInstrument(bank)
            was_compiled = inference._compiled
            inference.disable_compile()
            try:
                inference.forward(pos_tokens, patcher=instrument, grad_enabled=True, return_activations=False)
                gradients = compute_feature_gradient(
                    instrument.graph,
                    target_layer=seed_layer,
                    target_kind=seed_kind,
                    target_latent_idx=seed_latent,
                    pos_argmax=pos_argmax,
                    candidate_nodes=candidate_fids,
                )
            except Exception as exc:
                error = str(exc)
            finally:
                if was_compiled:
                    inference.enable_compile()
            del instrument
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        seed_start = len(rows)
        for gid in sorted(candidate_meta):
            fid = FeatureID.from_global_id(gid, n_kinds, d_sae, kinds)
            grad = float(gradients.get(fid, 0.0))
            meta = candidate_meta[gid]
            rows.append(
                _row(
                    circuit,
                    sample_index=sample_index,
                    seed_gid=seed_gid,
                    seed_comp=seed_comp,
                    seed_latent=seed_latent,
                    seed_layer=seed_layer,
                    seed_kind=seed_kind,
                    gid=gid,
                    fid=fid,
                    d_sae=d_sae,
                    meta=meta,
                    gradient=grad,
                    present=fid in gradients,
                )
            )
        seed_rows.append(
            _seed_row(
                circuit,
                sample_index,
                seed_gid,
                rows[seed_start:],
                error=error,
                n_pos=n_pos,
                n_coact=len(coact_meta),
                n_circuit=len(circuit_meta),
                n_random=len(random_meta),
            )
        )
        print(
            f"[gradient-distribution] {sample_index + 1}/{len(sampled)} "
            f"{circuit.name}: coact={len(coact_meta)} circuit={len(circuit_meta)} "
            f"random={len(random_meta)} error={bool(error)}",
            flush=True,
        )

    return {
        "sample_size": int(sample_size),
        "actual_seed_count": len(seed_rows),
        "threshold": float(threshold),
        "max_coact_nodes": int(max_coact_nodes),
        "random_per_observed": int(random_per_observed),
        "rows": rows,
        "seed_rows": seed_rows,
        "skipped": skipped,
        "group_summary": _group_summaries(rows),
        "per_seed_summary": _per_seed_summary(seed_rows),
        "pairwise_abs_gradient": _pairwise_stats(rows),
    }


def _coact_candidates_for_seed(
    values: torch.Tensor,
    indices: torch.Tensor,
    seed_gid: int,
    *,
    threshold: float,
    max_coact_nodes: int,
) -> dict[int, dict[str, Any]]:
    if seed_gid < 0 or seed_gid >= values.shape[0]:
        return {}
    candidate_scores: list[tuple[float, int, int]] = []
    for rank, (score, gid) in enumerate(zip(values[seed_gid].tolist(), indices[seed_gid].tolist()), start=1):
        gid_int = int(gid)
        score_float = float(score)
        if gid_int == int(seed_gid) or gid_int < 0 or score_float <= float(threshold):
            continue
        candidate_scores.append((score_float, rank, gid_int))
    candidate_scores.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {
        gid: {"role": "", "coact_rank": rank, "coact_score": score}
        for score, rank, gid in candidate_scores[: int(max_coact_nodes)]
    }


def _circuit_candidates_for_seed(
    circuit: Circuit,
    *,
    seed_gid: int,
    seed_layer: int,
    n_kinds: int,
    d_sae: int,
    kinds: Sequence[str],
) -> dict[int, dict[str, Any]]:
    candidates: dict[int, dict[str, Any]] = {}
    for node in circuit.nodes.values():
        fid = node.feature_id
        if fid is None or fid.kind not in kinds or fid.kind.endswith("_err") or fid.kind in ("logit", "token"):
            continue
        if int(fid.layer) > int(seed_layer):
            continue
        gid = fid.to_global_id(n_kinds, d_sae, kinds)
        if gid == int(seed_gid):
            continue
        candidates[gid] = {
            "role": str(node.metadata.get("role", "")),
            "coact_rank": "",
            "coact_score": "",
        }
    return candidates


def _matched_random_candidates(
    seed_gid: int,
    coact_meta: Mapping[int, Mapping[str, Any]],
    circuit_meta: Mapping[int, Mapping[str, Any]],
    excluded: set[int],
    *,
    n_kinds: int,
    d_sae: int,
    kinds: Sequence[str],
    random_per_observed: int,
) -> dict[int, dict[str, Any]]:
    random_meta: dict[int, dict[str, Any]] = {}
    used = set(excluded)
    source_items = [(gid, "coact") for gid in sorted(coact_meta)] + [(gid, "circuit") for gid in sorted(circuit_meta)]
    for source_gid, match_group in source_items:
        source_comp = int(source_gid) // int(d_sae)
        for repeat in range(max(1, int(random_per_observed))):
            gid = _deterministic_random_gid(seed_gid, source_gid, repeat, source_comp, d_sae, used)
            if gid is None:
                continue
            fid = FeatureID.from_global_id(gid, n_kinds, d_sae, kinds)
            used.add(gid)
            random_meta[gid] = {
                "role": "",
                "coact_rank": "",
                "coact_score": "",
                "match_group": match_group,
                "source_global_id": source_gid,
                "source_layer": fid.layer,
                "source_kind": fid.kind,
            }
    return random_meta


def _deterministic_random_gid(
    seed_gid: int,
    source_gid: int,
    repeat: int,
    source_comp: int,
    d_sae: int,
    used: set[int],
) -> int | None:
    base = (
        int(seed_gid) * 1_000_003
        + int(source_gid) * 97_409
        + int(repeat) * 8_191
        + int(source_comp) * 131
    ) % int(d_sae)
    for offset in range(min(int(d_sae), 257)):
        latent = (base + offset * 9_973) % int(d_sae)
        gid = int(source_comp) * int(d_sae) + int(latent)
        if gid not in used:
            return gid
    return None


def _is_gradient_addressable(
    gid: int,
    *,
    seed_layer: int,
    n_kinds: int,
    d_sae: int,
    kinds: Sequence[str],
) -> bool:
    fid = FeatureID.from_global_id(int(gid), n_kinds, d_sae, kinds)
    return fid.kind in kinds and int(fid.layer) <= int(seed_layer)


def _row(
    circuit: Circuit,
    *,
    sample_index: int,
    seed_gid: int,
    seed_comp: int,
    seed_latent: int,
    seed_layer: int,
    seed_kind: str,
    gid: int,
    fid: FeatureID,
    d_sae: int,
    meta: Mapping[str, Any],
    gradient: float,
    present: bool,
) -> dict[str, Any]:
    abs_grad = abs(float(gradient))
    return {
        "uuid": circuit.uuid,
        "name": circuit.name,
        "sample_index": int(sample_index),
        "seed_global_id": int(seed_gid),
        "seed_comp": int(seed_comp),
        "seed_latent": int(seed_latent),
        "seed_layer": int(seed_layer),
        "seed_kind": seed_kind,
        "group": str(meta.get("group", "")),
        "match_group": str(meta.get("match_group", "")),
        "latent_global_id": int(gid),
        "latent_comp": int(gid) // int(d_sae),
        "latent_layer": int(fid.layer),
        "latent_kind": fid.kind,
        "latent_idx": int(fid.index),
        "role": str(meta.get("role", "")),
        "coact_rank": meta.get("coact_rank", ""),
        "coact_score": meta.get("coact_score", ""),
        "gradient": float(gradient),
        "abs_gradient": abs_grad,
        "log10_abs_gradient": _log10_abs(abs_grad),
        "sign": 1 if gradient > 0 else (-1 if gradient < 0 else 0),
        "present_in_gradient_result": bool(present),
    }


def _seed_row(
    circuit: Circuit,
    sample_index: int,
    seed_gid: int,
    rows: list[dict[str, Any]],
    *,
    error: str,
    n_pos: int,
    n_coact: int,
    n_circuit: int,
    n_random: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uuid": circuit.uuid,
        "name": circuit.name,
        "sample_index": int(sample_index),
        "seed_global_id": int(seed_gid),
        "n_pos": int(n_pos),
        "n_coact": int(n_coact),
        "n_circuit": int(n_circuit),
        "n_random": int(n_random),
        "error": error,
    }
    for group in ("coact", "circuit", "coact+circuit", "random"):
        values = [float(r["abs_gradient"]) for r in rows if r["group"] == group]
        row[f"{group.replace('+', '_')}_mean_abs_gradient"] = float(mean(values)) if values else 0.0
        row[f"{group.replace('+', '_')}_median_abs_gradient"] = float(median(values)) if values else 0.0
        row[f"{group.replace('+', '_')}_count"] = len(values)
    row["coact_to_random_mean_abs_ratio"] = _ratio(row["coact_mean_abs_gradient"], row["random_mean_abs_gradient"])
    row["circuit_to_random_mean_abs_ratio"] = _ratio(row["circuit_mean_abs_gradient"], row["random_mean_abs_gradient"])
    row["circuit_to_coact_mean_abs_ratio"] = _ratio(row["circuit_mean_abs_gradient"], row["coact_mean_abs_gradient"])
    return row


def _write_plot(path: Path, stats: Mapping[str, object]) -> None:
    plt = configure_matplotlib()
    rows = [row for row in stats["rows"] if row["group"] in ("coact", "circuit", "random")]
    groups = ["random", "coact", "circuit"]
    colors = {"random": INK_MUTED, "coact": SERIES2[1], "circuit": SERIES2[0]}

    fig, axes = plt.subplots(2, 2, figsize=panel_figsize(2, 2))
    for group in groups:
        vals = [float(row["log10_abs_gradient"]) for row in rows if row["group"] == group]
        if vals:
            axes[0, 0].hist(vals, bins=50, alpha=0.6, label=group, color=colors[group])
    axes[0, 0].set_title("Gradient magnitude distribution")
    axes[0, 0].set_xlabel("log10(abs(seed gradient) + 1e-12)")
    axes[0, 0].set_ylabel("Latent count")
    styled_legend(axes[0, 0], loc="best")

    for group in groups:
        vals = sorted(float(row["abs_gradient"]) for row in rows if row["group"] == group)
        if vals:
            ys = [(index + 1) / len(vals) for index in range(len(vals))]
            axes[0, 1].plot(vals, ys, label=group, color=colors[group], linewidth=2.0)
    axes[0, 1].set_xscale("symlog", linthresh=1e-10)
    axes[0, 1].set_title("ECDF of absolute gradient")
    axes[0, 1].set_xlabel("abs(seed gradient)")
    axes[0, 1].set_ylabel("Cumulative fraction")
    styled_legend(axes[0, 1], loc="best")

    seed_rows = stats["seed_rows"]
    assert isinstance(seed_rows, list)
    box_data = []
    box_labels = []
    for key, label in [
        ("coact_to_random_mean_abs_ratio", "coact/random"),
        ("circuit_to_random_mean_abs_ratio", "circuit/random"),
        ("circuit_to_coact_mean_abs_ratio", "circuit/coact"),
    ]:
        vals = [float(row[key]) for row in seed_rows if float(row[key]) > 0.0]
        if vals:
            box_data.append(vals)
            box_labels.append(label)
    if box_data:
        styled_boxplot(axes[1, 0], box_data, box_labels, [BLUE] * len(box_data))
    axes[1, 0].axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1.0)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Per-seed mean abs-gradient ratios")
    axes[1, 0].set_ylabel("Ratio")

    coact_rows = [row for row in rows if row["group"] == "coact" and row["coact_score"] != ""]
    if coact_rows:
        axes[1, 1].scatter(
            [float(row["coact_score"]) for row in coact_rows],
            [float(row["abs_gradient"]) for row in coact_rows],
            s=14,
            alpha=0.5,
            color=colors["coact"],
            edgecolors="none",
        )
    axes[1, 1].set_yscale("symlog", linthresh=1e-10)
    axes[1, 1].set_title("Coact PMI vs seed-gradient magnitude")
    axes[1, 1].set_xlabel("PMI/top-coact score")
    axes[1, 1].set_ylabel("abs(seed gradient)")

    style_suptitle(fig, "Seed Gradient Distributions: Random vs Coact vs Circuit Nodes")
    save_figure(fig, path)


def _build_summary(store_path: Path, coact_path: Path, stats: Mapping[str, object]) -> dict[str, object]:
    return {
        "circuit_store_path": str(store_path),
        "top_coactivation_path": str(coact_path),
        "sample_size": stats["sample_size"],
        "actual_seed_count": stats["actual_seed_count"],
        "threshold": stats["threshold"],
        "max_coact_nodes": stats["max_coact_nodes"],
        "random_per_observed": stats["random_per_observed"],
        "skipped_count": len(stats["skipped"]),
        "skipped": stats["skipped"],
        "group_summary": stats["group_summary"],
        "per_seed_summary": stats["per_seed_summary"],
        "pairwise_abs_gradient": stats["pairwise_abs_gradient"],
    }


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        group: _summary([float(row["abs_gradient"]) for row in rows if row["group"] == group])
        for group in ("random", "coact", "circuit", "coact+circuit")
    }


def _per_seed_summary(seed_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = (
        "coact_to_random_mean_abs_ratio",
        "circuit_to_random_mean_abs_ratio",
        "circuit_to_coact_mean_abs_ratio",
    )
    return {key: _summary([float(row[key]) for row in seed_rows if float(row[key]) > 0.0]) for key in keys}


def _pairwise_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_group = {
        group: [float(row["abs_gradient"]) for row in rows if row["group"] == group]
        for group in ("random", "coact", "circuit")
    }
    pairs = (("coact", "random"), ("circuit", "random"), ("circuit", "coact"))
    return {
        f"{left}_vs_{right}": {
            "ks_statistic": _ks_statistic(by_group[left], by_group[right]),
            "auc_left_greater": _auc(by_group[left], by_group[right]),
            "mean_ratio": _ratio(_mean(by_group[left]), _mean(by_group[right])),
            "median_ratio": _ratio(_median(by_group[left]), _median(by_group[right])),
        }
        for left, right in pairs
    }


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "min": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(ordered[0]),
        "p90": _quantile(ordered, 0.9),
        "p99": _quantile(ordered, 0.99),
        "max": float(ordered[-1]),
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    pos = float(q) * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _ks_statistic(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    xs = sorted(set(left + right))
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    i = 0
    j = 0
    best = 0.0
    for x in xs:
        while i < len(left_sorted) and left_sorted[i] <= x:
            i += 1
        while j < len(right_sorted) and right_sorted[j] <= x:
            j += 1
        best = max(best, abs(i / len(left_sorted) - j / len(right_sorted)))
    return float(best)


def _auc(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    combined = [(value, 0) for value in left] + [(value, 1) for value in right]
    combined.sort(key=lambda item: item[0])
    rank_sum_left = 0.0
    rank = 1
    i = 0
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        left_count = sum(1 for _, group in combined[i:j] if group == 0)
        rank_sum_left += left_count * avg_rank
        rank += j - i
        i = j
    u_left = rank_sum_left - (len(left) * (len(left) + 1) / 2.0)
    return float(u_left / (len(left) * len(right)))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _log10_abs(abs_value: float) -> float:
    return float(math.log10(float(abs_value) + 1e-12))


def _deterministic_circuit_sample(circuits: list[Circuit], sample_size: int) -> list[Circuit]:
    circuits = sorted(circuits, key=lambda circuit: circuit.uuid)
    if len(circuits) <= int(sample_size):
        return circuits
    positions = torch.linspace(0, len(circuits) - 1, steps=int(sample_size), dtype=torch.float64).round().to(torch.int64).unique()
    return [circuits[int(position)] for position in positions.tolist()]

