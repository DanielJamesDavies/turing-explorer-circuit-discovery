"""Run intervention evals for full and hop-pruned circuit variants."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, cast

import torch

from analysis.io import analysis_output_dirs, resolve_run_root
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.counterfactual_faithfulness import evaluate_counterfactual_faithfulness
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from store.circuits import Circuit
from store.context import mid_ctx, neg_ctx, top_ctx
from .coact_overlap import SUITE_NAME

RESULT_FIELDS = [
    "uuid",
    "name",
    "variant",
    "hop",
    "seed_comp",
    "seed_latent",
    "retained_nodes",
    "retained_edges",
    "counterfactual_faithfulness",
    "posctx_suppression_score",
    "full_counterfactual_faithfulness",
    "full_posctx_suppression_score",
    "error",
]


def run_pruned_hop_evals(
    run_root: str | Path,
    *,
    spec_path: str | Path | None = None,
    output_root: str | Path | None = None,
    limit: int | None = None,
    retry_errors: bool = False,
) -> Path:
    """Run/resume pruned-hop evals and return the results CSV path."""

    root = resolve_run_root(run_root)
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    spec_file = Path(spec_path).expanduser().resolve() if spec_path is not None else output_dirs["root"] / "pruned-hop-eval-spec.pt"
    if not spec_file.exists():
        raise FileNotFoundError(f"pruned-hop eval spec not found: {spec_file}")
    results_path = output_dirs["tables"] / "pruned-hop-eval-results.csv"

    variants = torch.load(spec_file, map_location="cpu", weights_only=False)
    if not isinstance(variants, Mapping):
        raise TypeError(f"pruned-hop eval spec must contain a mapping, got {type(variants).__name__}")
    done = _completed_keys(results_path, retry_errors=retry_errors)

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
    avg_acts = torch.zeros((bank.n_layer * len(bank.kinds), bank.d_sae), dtype=torch.float32, device=bank.device)

    _ensure_header(results_path)
    rows_written = 0
    for circuit_uuid, group in variants.items():
        if not isinstance(group, Mapping):
            continue
        full = group.get("full")
        if not isinstance(full, Circuit):
            continue
        try:
            probe = _build_probe(probe_builder, full)
        except Exception as exc:
            _write_group_error(results_path, str(circuit_uuid), group, str(exc), done)
            continue
        full_cf = _metadata_float(full.metadata, ("evals", "counterfactual_faithfulness"))
        full_sup = _metadata_float(full.metadata, ("evals", "posctx_suppression_score"))
        for variant, circuit in group.items():
            if not isinstance(circuit, Circuit):
                continue
            key = (str(circuit_uuid), str(variant))
            if key in done:
                continue
            row = _eval_variant(
                inference,
                bank,
                avg_acts,
                probe,
                circuit,
                variant=str(variant),
                full_cf=full_cf,
                full_sup=full_sup,
            )
            _append_row(results_path, row)
            done.add(key)
            rows_written += 1
            print(
                f"[pruned-hop-eval] {len(done)} done | {circuit.name} {variant}: "
                f"cf={row['counterfactual_faithfulness']} sup={row['posctx_suppression_score']}"
            )
            if limit is not None and rows_written >= int(limit):
                return results_path
    return results_path


def _build_probe(probe_builder: ProbeDatasetBuilder, circuit: Circuit) -> dict[str, torch.Tensor]:
    seed_comp = int(circuit.metadata["seed_comp"])
    seed_latent = int(circuit.metadata["seed_latent"])
    probe_data = probe_builder.build_for_latent(seed_comp, seed_latent, top_ctx, mid_ctx, neg_ctx)
    n_pos = min(int(config.discovery.probe_batch_size or 16), int(probe_data.pos_tokens.shape[0]))
    pos_tokens = probe_data.pos_tokens[:n_pos].to(probe_builder.bank.device)
    pos_argmax = probe_data.pos_argmax[:n_pos].to(probe_builder.bank.device)
    neg_tokens = _neg_tokens_for_eval(probe_builder, circuit, probe_data.neg_tokens, pos_tokens)
    return {"pos_tokens": pos_tokens, "neg_tokens": neg_tokens, "pos_argmax": pos_argmax}


def _neg_tokens_for_eval(
    probe_builder: ProbeDatasetBuilder,
    circuit: Circuit,
    stored_neg_tokens: torch.Tensor,
    pos_tokens: torch.Tensor,
) -> torch.Tensor:
    max_neg = int(config.discovery.counterfactual_gradient.max_neg_sequences or 4)
    neg_mode = str(circuit.metadata.get("neg_mode", config.discovery.counterfactual_gradient.neg_mode))
    if neg_mode == "random":
        seed = int(circuit.metadata.get("seed_comp", 0)) * 1_000_003 + int(circuit.metadata.get("seed_latent", 0))
        generator = torch.Generator(device=probe_builder.bank.device)
        generator.manual_seed(seed)
        vocab_size = int(probe_builder.inference.model.config.vocab_size)
        return torch.randint(
            0,
            vocab_size,
            (max_neg, int(pos_tokens.shape[1])),
            device=probe_builder.bank.device,
            generator=generator,
        )
    return stored_neg_tokens[:max_neg].to(probe_builder.bank.device)


def _eval_variant(
    inference: Inference,
    bank: SAEBank,
    avg_acts: torch.Tensor,
    probe: dict[str, torch.Tensor],
    circuit: Circuit,
    *,
    variant: str,
    full_cf: float,
    full_sup: float,
) -> dict[str, Any]:
    seed_comp = int(circuit.metadata["seed_comp"])
    seed_latent = int(circuit.metadata["seed_latent"])
    seed_layer, seed_kind_idx = split_component_idx(seed_comp, len(bank.kinds))
    seed_kind = bank.kinds[seed_kind_idx]
    circuit_layers = {
        int(node.feature_id.layer)
        for node in circuit.nodes.values()
        if node.feature_id is not None
    }
    try:
        cf, sup = evaluate_counterfactual_faithfulness(
            inference,
            bank,
            avg_acts,
            circuit,
            neg_tokens=probe["neg_tokens"],
            pos_tokens=probe["pos_tokens"],
            seed_layer=seed_layer,
            seed_kind=seed_kind,
            seed_latent_idx=seed_latent,
            pos_argmax=probe["pos_argmax"],
            circuit_layers=circuit_layers,
        )
        error = ""
        if not math.isfinite(float(cf)) or not math.isfinite(float(sup)):
            error = "non-finite eval score"
    except Exception as exc:
        cf = 0.0
        sup = 0.0
        error = str(exc)
    return {
        "uuid": circuit.uuid,
        "name": circuit.name,
        "variant": variant,
        "hop": 0 if variant == "full" else str(variant).replace("hop", ""),
        "seed_comp": seed_comp,
        "seed_latent": seed_latent,
        "retained_nodes": len(circuit.nodes),
        "retained_edges": len(circuit.edges),
        "counterfactual_faithfulness": cf,
        "posctx_suppression_score": sup,
        "full_counterfactual_faithfulness": full_cf,
        "full_posctx_suppression_score": full_sup,
        "error": error,
    }


def _ensure_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RESULT_FIELDS).writeheader()


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writerow(row)


def _completed_keys(path: Path, *, retry_errors: bool = False) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not retry_errors:
        return {(row["uuid"], row["variant"]) for row in rows}
    return {
        (row["uuid"], row["variant"])
        for row in rows
        if not row.get("error")
        and math.isfinite(_float_for_completion(row.get("counterfactual_faithfulness")))
        and math.isfinite(_float_for_completion(row.get("posctx_suppression_score")))
    }


def _write_group_error(
    path: Path,
    circuit_uuid: str,
    group: Mapping[str, object],
    error: str,
    done: set[tuple[str, str]],
) -> None:
    for variant, circuit in group.items():
        if not isinstance(circuit, Circuit):
            continue
        key = (circuit_uuid, str(variant))
        if key in done:
            continue
        _append_row(
            path,
            {
                "uuid": circuit_uuid,
                "name": circuit.name,
                "variant": variant,
                "hop": 0 if variant == "full" else str(variant).replace("hop", ""),
                "seed_comp": circuit.metadata.get("seed_comp", -1),
                "seed_latent": circuit.metadata.get("seed_latent", -1),
                "retained_nodes": len(circuit.nodes),
                "retained_edges": len(circuit.edges),
                "counterfactual_faithfulness": 0.0,
                "posctx_suppression_score": 0.0,
                "full_counterfactual_faithfulness": _metadata_float(circuit.metadata, ("evals", "counterfactual_faithfulness")),
                "full_posctx_suppression_score": _metadata_float(circuit.metadata, ("evals", "posctx_suppression_score")),
                "error": error,
            },
        )
        done.add(key)


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


def _float_for_completion(value: object) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return float("nan")

