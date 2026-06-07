"""Measure logit-distribution effects of top-context latent ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from analysis.io import analysis_output_dirs, resolve_run_root, write_csv, write_json
from analysis.style import configure_matplotlib
from circuit.probe_dataset import ProbeDatasetBuilder
from circuit.types.feature_id import FeatureID
from data.loader import DataLoader
from hardware import detect_devices, is_fast_memory
from model.hooks import multi_patch
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense, target_latent_activations
from store.context import top_ctx

from .sorted_pmi_decay import SUITE_NAME


DEFAULT_KINDS = ("attn", "mlp", "resid")
FIELDS = [
    "sample_index",
    "global_id",
    "component_idx",
    "layer",
    "kind",
    "latent_idx",
    "candidate_sequence_count",
    "live_active_sequence_count",
    "live_active_sequence_pct",
    "sequence_count",
    "mean_top_ctx_activation",
    "baseline_target_activation_at_probe",
    "ablated_target_activation_at_probe",
    "target_active_pct",
    "activation_removed_pct",
    "live_max_activation_across_sequence",
    "live_active_any_pct",
    "mean_live_argmax_position",
    "probe_matches_live_argmax_pct",
    "probe_position_was_fallback_zero_pct",
    "patched_stream_delta_norm",
    "patched_stream_delta_pct",
    "kl_baseline_to_ablated",
    "js_divergence",
    "top1_changed_pct",
    "baseline_top_prob_delta",
    "entropy_delta",
    "ground_truth_logprob_delta",
    "max_abs_logit_delta",
    "mean_abs_logit_delta",
    "logit_l2_delta",
    "skipped_reason",
]


@dataclass(frozen=True)
class TopCtxLogitEffectResult:
    figure_path: Path
    summary_path: Path
    table_path: Path
    summary: dict[str, object]


def plot_top_ctx_logit_effect(
    run_root: str | Path,
    *,
    output_root: str | Path | None = None,
    sample_size: int = 128,
    top_ctx_batch_size: int = 16,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> TopCtxLogitEffectResult:
    """Generate a pilot graph for single-latent top-context logit effects."""

    root = resolve_run_root(run_root)
    print("Loading discovery artifacts...")
    load_discovery_artifacts(root)
    print("Initializing model/SAE resources...")
    devices = detect_devices()
    device = devices[0]
    fast = is_fast_memory()
    # This analysis relies on inner attn/mlp module hooks for causal patching.
    # Compiled block forwards can bypass those hooks, producing false zero effects.
    compile_model = False
    loader = DataLoader(device=device, pin_memory=fast)
    inference = Inference(device=device, compile=compile_model)
    bank = SAEBank(devices=devices, load_decoders=fast, compile=compile_model)
    probe_builder = ProbeDatasetBuilder(inference, bank, loader)

    stats = compute_top_ctx_logit_effect(
        top_ctx.ctx_seq_idx,
        top_ctx.ctx_seq_val,
        inference=inference,
        bank=bank,
        probe_builder=probe_builder,
        sample_size=sample_size,
        top_ctx_batch_size=top_ctx_batch_size,
        kinds=kinds,
    )
    output_dirs = analysis_output_dirs(root, SUITE_NAME, output_root=output_root)
    figure_path = output_dirs["figures"] / "top-ctx-logit-effect.png"
    table_path = output_dirs["tables"] / "top-ctx-logit-effect.csv"
    summary_path = output_dirs["summaries"] / "top-ctx-logit-effect.json"

    _write_plot(figure_path, stats)
    write_csv(table_path, stats["rows"], FIELDS)
    summary = _build_summary(root, stats)
    write_json(summary_path, summary)
    return TopCtxLogitEffectResult(figure_path, summary_path, table_path, summary)


def compute_top_ctx_logit_effect(
    top_ctx_indices: torch.Tensor,
    top_ctx_values: torch.Tensor,
    *,
    inference: Inference,
    bank: SAEBank,
    probe_builder: ProbeDatasetBuilder,
    sample_size: int = 128,
    top_ctx_batch_size: int = 16,
    kinds: Sequence[str] = DEFAULT_KINDS,
) -> dict[str, object]:
    """Run baseline and single-latent ablation forwards for sampled top-context latents."""

    if top_ctx_batch_size <= 0:
        raise ValueError("top_ctx_batch_size must be positive")
    inference.disable_compile()
    sample_gids = sample_top_ctx_latents(top_ctx_indices, sample_size=sample_size, d_sae=bank.d_sae)
    rows: list[dict[str, Any]] = []

    for sample_index, gid in enumerate(sample_gids.tolist()):
        comp_idx = int(gid) // int(bank.d_sae)
        latent_idx = int(gid) % int(bank.d_sae)
        layer, kind_idx = split_component_idx(comp_idx, len(kinds))
        if kind_idx >= len(kinds):
            rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, "unknown", latent_idx, "kind index out of range"))
            continue
        kind = kinds[kind_idx]
        if kind not in bank.kinds:
            rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, kind, latent_idx, "kind not available in SAE bank"))
            continue

        sequence_ids = _top_ctx_sequence_ids(top_ctx_indices, comp_idx, latent_idx, top_ctx_batch_size)
        if not sequence_ids:
            rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, kind, latent_idx, "empty top-context batch"))
            continue
        candidate_count = len(sequence_ids)
        all_tokens = probe_builder._load_all_ids(sequence_ids, max_length=65)
        if all_tokens.shape[0] == 0:
            rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, kind, latent_idx, "failed to load top-context tokens"))
            continue

        tokens = all_tokens[:, :64].to(bank.device)
        target_tokens = all_tokens[:, 1:65].to(bank.device)
        if target_tokens.shape[1] < tokens.shape[1]:
            pad_len = tokens.shape[1] - target_tokens.shape[1]
            padding = torch.zeros((target_tokens.shape[0], pad_len), dtype=torch.long, device=target_tokens.device)
            target_tokens = torch.cat([target_tokens, padding], dim=1)
        fid = FeatureID(layer=layer, kind=kind, index=latent_idx)
        with torch.no_grad():
            try:
                baseline_logits, live_max, live_argmax = _baseline_logits_and_live_positions(
                    inference,
                    bank,
                    tokens,
                    fid,
                )
            except Exception as exc:
                rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, kind, latent_idx, f"baseline probe failed: {exc}"))
                continue
            live_mask = live_max > 0
            live_count = int(live_mask.sum().item())
            activation_values = _top_ctx_values_for_latent(top_ctx_values, comp_idx, latent_idx, candidate_count)
            if live_count <= 0:
                rows.append(
                    {
                        "sample_index": sample_index,
                        "global_id": int(gid),
                        "component_idx": comp_idx,
                        "layer": int(layer),
                        "kind": kind,
                        "latent_idx": latent_idx,
                        "candidate_sequence_count": candidate_count,
                        "live_active_sequence_count": 0,
                        "live_active_sequence_pct": 0.0,
                        "sequence_count": 0,
                        "mean_top_ctx_activation": float(activation_values.mean().item()) if activation_values.numel() else 0.0,
                        "baseline_target_activation_at_probe": 0.0,
                        "ablated_target_activation_at_probe": 0.0,
                        "target_active_pct": 0.0,
                        "activation_removed_pct": 0.0,
                        "live_max_activation_across_sequence": 0.0,
                        "live_active_any_pct": 0.0,
                        "mean_live_argmax_position": 0.0,
                        "probe_matches_live_argmax_pct": 0.0,
                        "probe_position_was_fallback_zero_pct": 100.0,
                        "patched_stream_delta_norm": 0.0,
                        "patched_stream_delta_pct": 0.0,
                        "kl_baseline_to_ablated": 0.0,
                        "js_divergence": 0.0,
                        "top1_changed_pct": 0.0,
                        "baseline_top_prob_delta": 0.0,
                        "entropy_delta": 0.0,
                        "ground_truth_logprob_delta": 0.0,
                        "max_abs_logit_delta": 0.0,
                        "mean_abs_logit_delta": 0.0,
                        "logit_l2_delta": 0.0,
                        "skipped_reason": "no live target activation",
                    }
                )
                continue

            tokens = tokens[live_mask]
            target_tokens = target_tokens[live_mask]
            pos_argmax = live_argmax[live_mask].to(bank.device)
            baseline_logits = baseline_logits[live_mask]
            patcher = ObservedSingleLatentAblationPatcher(
                bank,
                fid,
                pos_argmax,
            )
            _, ablated_logits, _ = inference.forward(
                tokens,
                num_gen=1,
                tokenize_final=False,
                patcher=patcher,
                return_activations=False,
                all_logits=True,
            )

        if baseline_logits is None or ablated_logits is None:
            rows.append(_skipped_row(sample_index, int(gid), comp_idx, layer, kind, latent_idx, "missing logits"))
            continue

        n_pos = int(tokens.shape[0])
        baseline_at_pos = _gather_positions(baseline_logits, pos_argmax)
        ablated_at_pos = _gather_positions(ablated_logits, pos_argmax)
        targets_at_pos = target_tokens[torch.arange(n_pos, device=target_tokens.device), pos_argmax]
        metrics = compute_logit_effect_metrics(baseline_at_pos, ablated_at_pos, targets_at_pos)
        activation_values = _top_ctx_values_for_latent(top_ctx_values, comp_idx, latent_idx, candidate_count)
        observability = patcher.summary()
        rows.append(
            {
                "sample_index": sample_index,
                "global_id": int(gid),
                "component_idx": comp_idx,
                "layer": int(layer),
                "kind": kind,
                "latent_idx": latent_idx,
                "candidate_sequence_count": candidate_count,
                "live_active_sequence_count": live_count,
                "live_active_sequence_pct": (live_count / candidate_count * 100.0) if candidate_count else 0.0,
                "sequence_count": n_pos,
                "mean_top_ctx_activation": float(activation_values.mean().item()) if activation_values.numel() else 0.0,
                **observability,
                **metrics,
                "skipped_reason": "",
            }
        )

    valid_rows = [row for row in rows if not row.get("skipped_reason")]
    return {
        "sample_size": int(sample_size),
        "actual_sample_size": int(len(sample_gids)),
        "top_ctx_batch_size": int(top_ctx_batch_size),
        "valid_count": len(valid_rows),
        "skipped_count": len(rows) - len(valid_rows),
        "rows": rows,
        "summaries": {
            "kl_baseline_to_ablated": _summary([float(row["kl_baseline_to_ablated"]) for row in valid_rows]),
            "js_divergence": _summary([float(row["js_divergence"]) for row in valid_rows]),
            "top1_changed_pct": _summary([float(row["top1_changed_pct"]) for row in valid_rows]),
            "ground_truth_logprob_delta": _summary([float(row["ground_truth_logprob_delta"]) for row in valid_rows]),
            "entropy_delta": _summary([float(row["entropy_delta"]) for row in valid_rows]),
            "max_abs_logit_delta": _summary([float(row["max_abs_logit_delta"]) for row in valid_rows]),
            "target_active_pct": _summary([float(row["target_active_pct"]) for row in valid_rows]),
            "activation_removed_pct": _summary([float(row["activation_removed_pct"]) for row in valid_rows]),
            "live_active_sequence_pct": _summary([float(row["live_active_sequence_pct"]) for row in rows]),
            "live_active_any_pct": _summary([float(row["live_active_any_pct"]) for row in valid_rows]),
            "probe_matches_live_argmax_pct": _summary([float(row["probe_matches_live_argmax_pct"]) for row in valid_rows]),
            "probe_position_was_fallback_zero_pct": _summary(
                [float(row["probe_position_was_fallback_zero_pct"]) for row in valid_rows]
            ),
            "patched_stream_delta_norm": _summary([float(row["patched_stream_delta_norm"]) for row in valid_rows]),
        },
        "by_kind": _kind_summaries(valid_rows),
        "sampled_by_kind": _kind_summaries(rows),
        "by_layer": _layer_summaries(valid_rows),
    }


def sample_top_ctx_latents(top_ctx_indices: torch.Tensor, *, sample_size: int, d_sae: int) -> torch.Tensor:
    """Deterministically sample eligible latent global IDs from nonempty top-context rows."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if top_ctx_indices.ndim != 3:
        raise ValueError("top_ctx_indices must have shape [components, d_sae, top_k]")
    nonempty = (top_ctx_indices.detach().cpu().to(torch.int64) > 0).any(dim=-1)
    samples: list[int] = []
    components = int(nonempty.shape[0])
    quota = max(1, (int(sample_size) + components - 1) // max(components, 1))
    for comp_idx in range(components):
        latent_indices = nonempty[comp_idx].nonzero(as_tuple=False).squeeze(1)
        if latent_indices.numel() == 0:
            continue
        selected = _evenly_spaced(latent_indices, quota)
        samples.extend((selected.to(torch.int64) + int(comp_idx) * int(d_sae)).tolist())
    if len(samples) > int(sample_size):
        positions = _evenly_spaced(torch.arange(len(samples), dtype=torch.int64), int(sample_size))
        samples = [samples[int(pos)] for pos in positions.tolist()]
    return torch.tensor(samples, dtype=torch.int64)


def compute_logit_effect_metrics(
    baseline_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    target_tokens: torch.Tensor,
) -> dict[str, float]:
    """Compare baseline and ablated next-token distributions."""

    if baseline_logits.shape != ablated_logits.shape:
        raise ValueError("baseline_logits and ablated_logits must have matching shapes")
    if baseline_logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if target_tokens.ndim != 1 or int(target_tokens.shape[0]) != int(baseline_logits.shape[0]):
        raise ValueError("target_tokens must have shape [batch]")

    base_log_probs = F.log_softmax(baseline_logits.float(), dim=-1)
    ablated_log_probs = F.log_softmax(ablated_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    ablated_probs = ablated_log_probs.exp()
    kl = (base_probs * (base_log_probs - ablated_log_probs)).sum(dim=-1)
    mixture = (base_probs + ablated_probs).clamp_min(1e-12) * 0.5
    js = 0.5 * (
        (base_probs * (base_log_probs - mixture.log())).sum(dim=-1)
        + (ablated_probs * (ablated_log_probs - mixture.log())).sum(dim=-1)
    )
    base_top = base_probs.argmax(dim=-1)
    ablated_top = ablated_probs.argmax(dim=-1)
    base_top_prob = base_probs.gather(1, base_top.unsqueeze(1)).squeeze(1)
    ablated_base_top_prob = ablated_probs.gather(1, base_top.unsqueeze(1)).squeeze(1)
    base_entropy = -(base_probs * base_log_probs).sum(dim=-1)
    ablated_entropy = -(ablated_probs * ablated_log_probs).sum(dim=-1)
    targets = target_tokens.to(torch.long).unsqueeze(1)
    base_target_lp = base_log_probs.gather(1, targets).squeeze(1)
    ablated_target_lp = ablated_log_probs.gather(1, targets).squeeze(1)
    logit_delta = baseline_logits.float() - ablated_logits.float()
    return {
        "kl_baseline_to_ablated": float(kl.clamp_min(0.0).mean().item()),
        "js_divergence": float(js.clamp_min(0.0).mean().item()),
        "top1_changed_pct": float((base_top != ablated_top).float().mean().item() * 100.0),
        "baseline_top_prob_delta": float((base_top_prob - ablated_base_top_prob).mean().item()),
        "entropy_delta": float((ablated_entropy - base_entropy).mean().item()),
        "ground_truth_logprob_delta": float((base_target_lp - ablated_target_lp).mean().item()),
        "max_abs_logit_delta": float(logit_delta.abs().max().item()),
        "mean_abs_logit_delta": float(logit_delta.abs().mean().item()),
        "logit_l2_delta": float(torch.linalg.vector_norm(logit_delta, dim=-1).mean().item()),
    }


def _baseline_logits_and_live_positions(
    inference: Inference,
    bank: SAEBank,
    tokens: torch.Tensor,
    fid: FeatureID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    live_target_dense: list[torch.Tensor] = []

    def capture_hook(layer_idx: int, activations: tuple[torch.Tensor, ...]) -> None:
        if layer_idx != fid.layer:
            return
        kind_idx = bank.kinds.index(fid.kind)
        top_acts, top_indices = bank.encode(activations[kind_idx], fid.kind, fid.layer)
        target_dense = target_latent_activations(top_acts, top_indices, fid.index)
        live_target_dense.append(target_dense.detach())

    _, logits, _ = inference.forward(
        tokens,
        num_gen=1,
        tokenize_final=False,
        activations_callback=capture_hook,
        return_activations=False,
        all_logits=True,
    )
    if logits is None:
        raise RuntimeError("baseline forward did not return logits")
    if not live_target_dense:
        raise RuntimeError(f"target layer was not observed for {fid}")
    target_dense = live_target_dense[0]
    live_max, live_argmax = target_dense.max(dim=-1)
    return logits, live_max.to(logits.device), live_argmax.to(logits.device)


class ObservedSingleLatentAblationPatcher:
    """Zero one latent at probe positions and record whether the patch did work."""

    def __init__(self, bank: SAEBank, fid: FeatureID, pos_argmax: torch.Tensor):
        self.bank = bank
        self.fid = fid
        self.pos_argmax = pos_argmax.detach().cpu()
        self.baseline_probe_activation: torch.Tensor | None = None
        self.ablated_probe_activation: torch.Tensor | None = None
        self.live_max_activation: torch.Tensor | None = None
        self.live_argmax_position: torch.Tensor | None = None
        self.probe_position: torch.Tensor | None = None
        self.stream_delta_norm: torch.Tensor | None = None
        self.stream_delta_pct: torch.Tensor | None = None

    def __call__(self, model: Any):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx: int, kind: str, x: torch.Tensor) -> torch.Tensor:
        if layer_idx != self.fid.layer or kind != self.fid.kind:
            return x

        batch, seq_len, _ = x.shape
        target_dtype = x.dtype
        top_acts, top_indices = self.bank.encode(x, kind, layer_idx)
        all_latents = sparse_topk_to_dense(top_acts, top_indices, self.bank.d_sae, dtype=target_dtype)
        full_recon = self.bank.decode(all_latents, kind, layer_idx)
        error = x - full_recon

        probe_pos = self.pos_argmax.to(x.device).to(torch.long).clamp(0, seq_len - 1)
        batch_idx = torch.arange(batch, device=x.device)
        target_dense = all_latents[:, :, self.fid.index].detach().float()
        live_max, live_argmax = target_dense.max(dim=-1)
        baseline_probe = target_dense[batch_idx, probe_pos].detach().float().cpu()

        patched_latents = all_latents.clone()
        patched_latents[batch_idx, probe_pos, self.fid.index] = 0.0
        patched = self.bank.decode(patched_latents, kind, layer_idx) + error

        is_probe = torch.zeros(batch, seq_len, 1, dtype=torch.bool, device=x.device)
        is_probe[batch_idx, probe_pos] = True
        output = torch.where(is_probe, patched, x)

        delta_at_probe = (output - x)[batch_idx, probe_pos].detach().float()
        x_at_probe = x[batch_idx, probe_pos].detach().float()
        self.baseline_probe_activation = baseline_probe
        self.ablated_probe_activation = patched_latents[batch_idx, probe_pos, self.fid.index].detach().float().cpu()
        self.live_max_activation = live_max.cpu()
        self.live_argmax_position = live_argmax.cpu()
        self.probe_position = probe_pos.detach().cpu()
        self.stream_delta_norm = torch.linalg.vector_norm(delta_at_probe, dim=-1).cpu()
        self.stream_delta_pct = (self.stream_delta_norm / torch.linalg.vector_norm(x_at_probe, dim=-1).clamp_min(1e-8).cpu()) * 100.0
        return output

    def summary(self) -> dict[str, float]:
        baseline = self.baseline_probe_activation if self.baseline_probe_activation is not None else torch.zeros(0)
        ablated = self.ablated_probe_activation if self.ablated_probe_activation is not None else torch.zeros(0)
        live_max = self.live_max_activation if self.live_max_activation is not None else torch.zeros(0)
        live_argmax = self.live_argmax_position if self.live_argmax_position is not None else torch.zeros(0, dtype=torch.long)
        probe_pos = self.probe_position if self.probe_position is not None else torch.zeros(0, dtype=torch.long)
        delta_norm = self.stream_delta_norm if self.stream_delta_norm is not None else torch.zeros(0)
        delta_pct = self.stream_delta_pct if self.stream_delta_pct is not None else torch.zeros(0)
        active = baseline > 0
        removed = active & (ablated.abs() <= 1e-8)
        live_active = live_max > 0
        matches_live_argmax = live_argmax == probe_pos
        fallback_zero = (~live_active) & (probe_pos == 0)
        return {
            "baseline_target_activation_at_probe": float(baseline.mean().item()) if baseline.numel() else 0.0,
            "ablated_target_activation_at_probe": float(ablated.mean().item()) if ablated.numel() else 0.0,
            "target_active_pct": float(active.float().mean().item() * 100.0) if active.numel() else 0.0,
            "activation_removed_pct": float(removed.float().mean().item() * 100.0) if removed.numel() else 0.0,
            "live_max_activation_across_sequence": float(live_max.mean().item()) if live_max.numel() else 0.0,
            "live_active_any_pct": float(live_active.float().mean().item() * 100.0) if live_active.numel() else 0.0,
            "mean_live_argmax_position": float(live_argmax.float().mean().item()) if live_argmax.numel() else 0.0,
            "probe_matches_live_argmax_pct": float(matches_live_argmax.float().mean().item() * 100.0)
            if matches_live_argmax.numel()
            else 0.0,
            "probe_position_was_fallback_zero_pct": float(fallback_zero.float().mean().item() * 100.0)
            if fallback_zero.numel()
            else 0.0,
            "patched_stream_delta_norm": float(delta_norm.mean().item()) if delta_norm.numel() else 0.0,
            "patched_stream_delta_pct": float(delta_pct.mean().item()) if delta_pct.numel() else 0.0,
        }


def _gather_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch = int(logits.shape[0])
    pos = positions.to(logits.device).to(torch.long).clamp(0, int(logits.shape[1]) - 1)
    return logits[torch.arange(batch, device=logits.device), pos]


def _top_ctx_sequence_ids(top_ctx_indices: torch.Tensor, comp_idx: int, latent_idx: int, limit: int) -> list[int]:
    if comp_idx < 0 or comp_idx >= int(top_ctx_indices.shape[0]):
        return []
    if latent_idx < 0 or latent_idx >= int(top_ctx_indices.shape[1]):
        return []
    ids: list[int] = []
    seen: set[int] = set()
    for sid in top_ctx_indices[comp_idx, latent_idx].detach().cpu().to(torch.int64).tolist():
        sid_int = int(sid)
        if sid_int > 0 and sid_int not in seen:
            ids.append(sid_int)
            seen.add(sid_int)
        if len(ids) >= int(limit):
            break
    return ids


def _top_ctx_values_for_latent(top_ctx_values: torch.Tensor, comp_idx: int, latent_idx: int, n_pos: int) -> torch.Tensor:
    if comp_idx < 0 or comp_idx >= int(top_ctx_values.shape[0]):
        return torch.zeros(0)
    if latent_idx < 0 or latent_idx >= int(top_ctx_values.shape[1]):
        return torch.zeros(0)
    values = top_ctx_values[comp_idx, latent_idx].detach().cpu().float()
    values = values[values > 0]
    return values[:n_pos]


def _evenly_spaced(values: torch.Tensor, count: int) -> torch.Tensor:
    if values.numel() <= int(count):
        return values.to(torch.int64)
    positions = torch.linspace(0, values.numel() - 1, steps=int(count), dtype=torch.float64).round().to(torch.int64).unique()
    return values[positions].to(torch.int64)


def _skipped_row(
    sample_index: int,
    gid: int,
    comp_idx: int,
    layer: int,
    kind: str,
    latent_idx: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "sample_index": int(sample_index),
        "global_id": int(gid),
        "component_idx": int(comp_idx),
        "layer": int(layer),
        "kind": kind,
        "latent_idx": int(latent_idx),
        "candidate_sequence_count": 0,
        "live_active_sequence_count": 0,
        "live_active_sequence_pct": 0.0,
        "sequence_count": 0,
        "mean_top_ctx_activation": 0.0,
        "baseline_target_activation_at_probe": 0.0,
        "ablated_target_activation_at_probe": 0.0,
        "target_active_pct": 0.0,
        "activation_removed_pct": 0.0,
        "live_max_activation_across_sequence": 0.0,
        "live_active_any_pct": 0.0,
        "mean_live_argmax_position": 0.0,
        "probe_matches_live_argmax_pct": 0.0,
        "probe_position_was_fallback_zero_pct": 0.0,
        "patched_stream_delta_norm": 0.0,
        "patched_stream_delta_pct": 0.0,
        "kl_baseline_to_ablated": 0.0,
        "js_divergence": 0.0,
        "top1_changed_pct": 0.0,
        "baseline_top_prob_delta": 0.0,
        "entropy_delta": 0.0,
        "ground_truth_logprob_delta": 0.0,
        "max_abs_logit_delta": 0.0,
        "mean_abs_logit_delta": 0.0,
        "logit_l2_delta": 0.0,
        "skipped_reason": reason,
    }


def _write_plot(path: Path, stats: dict[str, object]) -> None:
    plt = configure_matplotlib()
    rows = [row for row in stats["rows"] if not row.get("skipped_reason")]
    assert isinstance(rows, list)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    kl_values = [float(row["kl_baseline_to_ablated"]) for row in rows]
    axes[0, 0].hist(kl_values, bins=40, color="#2f6f9f", alpha=0.85)
    axes[0, 0].set_title("Top-Context Logit Effect")
    axes[0, 0].set_xlabel("Mean KL(P baseline || P ablated)")
    axes[0, 0].set_ylabel("Latent count")

    axes[0, 1].scatter(
        [float(row["baseline_target_activation_at_probe"]) for row in rows],
        kl_values,
        c=[int(row["layer"]) for row in rows],
        cmap="viridis",
        s=20,
        alpha=0.75,
        edgecolors="none",
    )
    axes[0, 1].set_title("Live Activation vs Logit Effect")
    axes[0, 1].set_xlabel("Baseline target activation at probe")
    axes[0, 1].set_ylabel("Mean KL")

    by_kind = {kind: [float(row["kl_baseline_to_ablated"]) for row in rows if row["kind"] == kind] for kind in DEFAULT_KINDS}
    labels = [kind for kind, values in by_kind.items() if values]
    values = [by_kind[kind] for kind in labels]
    if values:
        axes[1, 0].boxplot(values, labels=labels, showfliers=False)
    axes[1, 0].set_title("Logit Effect By SAE Kind")
    axes[1, 0].set_ylabel("Mean KL")

    layer_summary = stats["by_layer"]
    assert isinstance(layer_summary, dict)
    layers = sorted(int(layer) for layer in layer_summary)
    axes[1, 1].bar(
        [str(layer) for layer in layers],
        [float(layer_summary[layer]["kl_mean"]) for layer in layers],
        color="#b45f06",
        alpha=0.75,
        label="mean KL",
    )
    ax2 = axes[1, 1].twinx()
    ax2.plot(
        [str(layer) for layer in layers],
        [float(layer_summary[layer]["top1_changed_pct_mean"]) for layer in layers],
        color="#38761d",
        marker="o",
        linewidth=1.8,
        label="top-1 changed %",
    )
    axes[1, 1].set_title("Layer Summary")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Mean KL")
    ax2.set_ylabel("Top-1 changed (%)")

    fig.suptitle("Latent Effect On Top-Context Logit Distributions", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _build_summary(root: Path, stats: dict[str, object]) -> dict[str, object]:
    return {
        "run_root": str(root),
        "sample_size": stats["sample_size"],
        "actual_sample_size": stats["actual_sample_size"],
        "top_ctx_batch_size": stats["top_ctx_batch_size"],
        "valid_count": stats["valid_count"],
        "skipped_count": stats["skipped_count"],
        "summaries": stats["summaries"],
        "by_kind": stats["by_kind"],
        "sampled_by_kind": stats["sampled_by_kind"],
        "by_layer": stats["by_layer"],
        "metric_note": "Top-context sequences are candidates; ablation is measured only on sequences where the target latent is live, at its live baseline argmax position.",
    }


def _kind_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {
        kind: {
            "count": len(kind_rows),
            "kl_mean": _mean([float(row["kl_baseline_to_ablated"]) for row in kind_rows]),
            "top1_changed_pct_mean": _mean([float(row["top1_changed_pct"]) for row in kind_rows]),
            "target_active_pct_mean": _mean([float(row["target_active_pct"]) for row in kind_rows]),
            "activation_removed_pct_mean": _mean([float(row["activation_removed_pct"]) for row in kind_rows]),
            "live_active_sequence_pct_mean": _mean([float(row["live_active_sequence_pct"]) for row in kind_rows]),
            "live_active_any_pct_mean": _mean([float(row["live_active_any_pct"]) for row in kind_rows]),
            "probe_matches_live_argmax_pct_mean": _mean([float(row["probe_matches_live_argmax_pct"]) for row in kind_rows]),
            "probe_position_was_fallback_zero_pct_mean": _mean(
                [float(row["probe_position_was_fallback_zero_pct"]) for row in kind_rows]
            ),
            "patched_stream_delta_norm_mean": _mean([float(row["patched_stream_delta_norm"]) for row in kind_rows]),
            "max_abs_logit_delta_mean": _mean([float(row["max_abs_logit_delta"]) for row in kind_rows]),
        }
        for kind in DEFAULT_KINDS
        for kind_rows in [[row for row in rows if row["kind"] == kind]]
        if kind_rows
    }


def _layer_summaries(rows: list[dict[str, Any]]) -> dict[int, dict[str, float | int]]:
    layers = sorted({int(row["layer"]) for row in rows})
    return {
        layer: {
            "count": len(layer_rows),
            "kl_mean": _mean([float(row["kl_baseline_to_ablated"]) for row in layer_rows]),
            "top1_changed_pct_mean": _mean([float(row["top1_changed_pct"]) for row in layer_rows]),
            "target_active_pct_mean": _mean([float(row["target_active_pct"]) for row in layer_rows]),
            "activation_removed_pct_mean": _mean([float(row["activation_removed_pct"]) for row in layer_rows]),
            "live_active_sequence_pct_mean": _mean([float(row["live_active_sequence_pct"]) for row in layer_rows]),
            "live_active_any_pct_mean": _mean([float(row["live_active_any_pct"]) for row in layer_rows]),
            "probe_matches_live_argmax_pct_mean": _mean([float(row["probe_matches_live_argmax_pct"]) for row in layer_rows]),
            "probe_position_was_fallback_zero_pct_mean": _mean(
                [float(row["probe_position_was_fallback_zero_pct"]) for row in layer_rows]
            ),
            "patched_stream_delta_norm_mean": _mean([float(row["patched_stream_delta_norm"]) for row in layer_rows]),
            "max_abs_logit_delta_mean": _mean([float(row["max_abs_logit_delta"]) for row in layer_rows]),
        }
        for layer in layers
        for layer_rows in [[row for row in rows if int(row["layer"]) == layer]]
    }


def _summary(values: list[float]) -> dict[str, float | int]:
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


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0
