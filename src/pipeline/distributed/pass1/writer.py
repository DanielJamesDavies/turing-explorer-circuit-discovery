"""Writer orchestration for distributed pass-1 merge outputs."""

from __future__ import annotations

import json
from pathlib import Path
import time
import tracemalloc
from typing import Dict

import torch

from config import config

from ..interfaces import build_output_paths
from ..layout import build_run_layout
from ..manifest import DistributedRunManifest, ManifestStatus, save_manifest
from ..pass2_replay import assign_pass2_replay_sequences
from ..seq_repr_mapping import shard_table_fingerprint
from ..shard_table import sequence_ids_for_shards
from .context_merge import (
    load_and_merge_mid_ctx_candidate_partials,
    load_and_merge_mid_ctx_reservoir_partials,
    load_and_merge_top_ctx_partials,
)
from .contracts import PASS1_PARTIAL_FILENAMES
from .latent_stats_merge import load_and_merge_latent_stats_partials
from .logit_ctx_merge import load_and_merge_logit_ctx_partials
from .reports import build_pass1_sanity_report
from .seq_latent_index_merge import merge_seq_latent_index_shards
from .seq_repr_merge import load_and_merge_seq_repr_partials


def merge_pass1_worker_outputs(
    manifest: DistributedRunManifest,
    *,
    seq_latent_index_enabled: bool = True,
    vocab_size: int | None = None,
    mid_ctx_num_ctx_sequences: int | None = None,
    mid_ctx_band_low_sigma: float = 0.5,
    mid_ctx_band_high_sigma: float = 1.5,
    mid_ctx_on_truncation: str = "replay_fallback",
    mid_ctx_merge_mode: str | None = None,
    mid_ctx_sampling_seed: int | None = None,
) -> Dict[str, object]:
    """Merge all worker pass-1 outputs and write canonical global artifacts."""

    start_time = time.perf_counter()
    tracemalloc.start()
    layout = build_run_layout(manifest)
    output_paths = build_output_paths(layout.run_root)
    output_paths.run_root.mkdir(parents=True, exist_ok=True)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    partial_paths = _pass1_partial_paths(manifest)

    stage_start = time.perf_counter()
    print("[pass1_merge] loading and merging latent_stats partials", flush=True)
    latent_stats_payload = load_and_merge_latent_stats_partials(
        partial_paths["latent_stats"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    print(
        f"[pass1_merge] latent_stats merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )
    stage_start = time.perf_counter()
    print("[pass1_merge] loading and merging top_ctx partials", flush=True)
    top_ctx_payload = load_and_merge_top_ctx_partials(
        partial_paths["top_ctx"],
        expected_config_hash=manifest.normalized_config_hash,
    )
    print(
        f"[pass1_merge] top_ctx merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )
    resolved_mid_ctx_merge_mode = mid_ctx_merge_mode or str(config.distributed.mid_ctx_merge.mode)
    resolved_sampling_seed = int(
        mid_ctx_sampling_seed
        if mid_ctx_sampling_seed is not None
        else (
            config.distributed.mid_ctx_merge.sampling_seed
            if config.distributed.mid_ctx_merge.sampling_seed is not None
            else manifest.sampling_seed
        )
    )
    stage_start = time.perf_counter()
    print(f"[pass1_merge] loading and merging mid_ctx partials mode={resolved_mid_ctx_merge_mode}", flush=True)
    if resolved_mid_ctx_merge_mode == "weighted_reservoir":
        mid_ctx_payload = load_and_merge_mid_ctx_reservoir_partials(
            partial_paths["mid_ctx_candidates"],
            expected_config_hash=manifest.normalized_config_hash,
            num_ctx_sequences=mid_ctx_num_ctx_sequences,
            band_low_sigma=mid_ctx_band_low_sigma,
            band_high_sigma=mid_ctx_band_high_sigma,
            sampling_seed=resolved_sampling_seed,
            dataset_fingerprint=shard_table_fingerprint(manifest.shard_table),
        )
    elif resolved_mid_ctx_merge_mode == "candidate_pool":
        mid_ctx_payload = load_and_merge_mid_ctx_candidate_partials(
            partial_paths["mid_ctx_candidates"],
            latent_stats_payload=latent_stats_payload,
            expected_config_hash=manifest.normalized_config_hash,
            num_ctx_sequences=mid_ctx_num_ctx_sequences,
            band_low_sigma=mid_ctx_band_low_sigma,
            band_high_sigma=mid_ctx_band_high_sigma,
            on_truncation=mid_ctx_on_truncation,
        )
    else:
        raise ValueError("unsupported mid_ctx merge mode")
    print(
        f"[pass1_merge] mid_ctx merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )
    stage_start = time.perf_counter()
    print("[pass1_merge] loading and merging seq_repr partials", flush=True)
    seq_repr_payload = load_and_merge_seq_repr_partials(
        partial_paths["seq_repr"],
        expected_config_hash=manifest.normalized_config_hash,
        sequence_ids_by_worker={
            int(worker_id): sequence_ids_for_shards(manifest.shard_table, shard_ids)
            for worker_id, shard_ids in manifest.work_assignments.pass1_shards.items()
        },
    )
    print(
        f"[pass1_merge] seq_repr merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )
    stage_start = time.perf_counter()
    print("[pass1_merge] loading and merging logit_ctx partials", flush=True)
    logit_ctx_payload = load_and_merge_logit_ctx_partials(
        partial_paths["logit_ctx"],
        expected_config_hash=manifest.normalized_config_hash,
        vocab_size=vocab_size,
    )
    print(
        f"[pass1_merge] logit_ctx merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )

    artifacts = {
        "latent_stats": (output_paths.latent_stats, _with_canonical_metadata(latent_stats_payload, manifest, "latent_stats")),
        "top_ctx": (output_paths.top_ctx, _with_canonical_metadata(top_ctx_payload, manifest, "top_ctx")),
        "mid_ctx": (output_paths.mid_ctx, _with_canonical_metadata(mid_ctx_payload, manifest, "mid_ctx")),
        "seq_repr": (output_paths.seq_repr, _with_canonical_metadata(seq_repr_payload, manifest, "seq_repr")),
        "logit_ctx": (output_paths.logit_ctx, _with_canonical_metadata(logit_ctx_payload, manifest, "logit_ctx")),
    }
    stage_start = time.perf_counter()
    for name, (path, payload) in artifacts.items():
        artifact_start = time.perf_counter()
        print(f"[pass1_merge] writing {name} -> {path}", flush=True)
        _atomic_torch_save(payload, path)
        print(
            f"[pass1_merge] wrote {name} elapsed={time.perf_counter() - artifact_start:.1f}s",
            flush=True,
        )
    print(
        f"[pass1_merge] artifact writes complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )

    stage_start = time.perf_counter()
    print("[pass1_merge] merging seq_latent_index shards", flush=True)
    seq_latent_index_report = merge_seq_latent_index_shards(
        [worker.pass1_dir / "seq_latent_index" for worker in layout.workers.values()],
        output_paths.seq_latent_index_dir,
        expected_shard_ids=[record.shard_index for record in manifest.shard_table],
        enabled=seq_latent_index_enabled,
        shard_id_ranges={
            record.shard_index: (record.global_start_id, record.global_end_id)
            for record in manifest.shard_table
        },
    )
    print(
        f"[pass1_merge] seq_latent_index merge complete elapsed={time.perf_counter() - stage_start:.1f}s",
        flush=True,
    )

    _validate_written_artifacts(artifacts)
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report = build_pass1_sanity_report(
        manifest,
        {
            "latent_stats": latent_stats_payload,
            "top_ctx": top_ctx_payload,
            "mid_ctx": mid_ctx_payload,
            "seq_repr": seq_repr_payload,
            "logit_ctx": logit_ctx_payload,
        },
        artifact_paths={name: str(path) for name, (path, _payload) in artifacts.items()},
        seq_latent_index_report=seq_latent_index_report,
        elapsed_s=time.perf_counter() - start_time,
        peak_cpu_memory_bytes=max(int(current_memory), int(peak_memory)),
    )
    report_path = layout.reports_dir / "pass1_sanity_report.json"
    _atomic_write_json(report_path, report)
    print(f"[pass1_merge] wrote sanity report -> {report_path}", flush=True)

    completed_manifest = assign_pass2_replay_sequences(manifest, top_ctx_payload).model_copy(
        update={"status": ManifestStatus.COMPLETED}
    )
    save_manifest(completed_manifest, manifest.manifest_path)
    print(
        "[pass1_merge] complete "
        f"elapsed={report['timing']['elapsed_s']:.1f}s "
        f"peak_cpu_memory_bytes={report['timing']['peak_cpu_memory_bytes']}",
        flush=True,
    )

    return {
        "artifacts": {name: str(path) for name, (path, _payload) in artifacts.items()},
        "seq_latent_index": seq_latent_index_report,
        "sanity_report": str(report_path),
        "manifest_path": manifest.manifest_path,
        "status": completed_manifest.status.value,
        "elapsed_s": report["timing"]["elapsed_s"],
        "peak_cpu_memory_bytes": report["timing"]["peak_cpu_memory_bytes"],
    }


def _with_canonical_metadata(
    payload: Dict[str, object],
    manifest: DistributedRunManifest,
    artifact_name: str,
) -> Dict[str, object]:
    enriched = dict(payload)
    enriched["metadata"] = {
        "schema_version": 1,
        "artifact_name": artifact_name,
        "run_id": manifest.run_id,
        "config_hash": manifest.normalized_config_hash,
        "manifest_path": manifest.manifest_path,
        "source": "distributed_pass1_merge",
    }
    enriched["config_hash"] = manifest.normalized_config_hash
    return enriched


def _pass1_partial_paths(manifest: DistributedRunManifest) -> Dict[str, list[Path]]:
    layout = build_run_layout(manifest)
    partial_paths: dict[str, list[Path]] = {
        name: [] for name in PASS1_PARTIAL_FILENAMES
    }
    for worker_id in range(manifest.worker_count):
        worker_dir = layout.workers[worker_id].pass1_dir
        for artifact_name, filename in PASS1_PARTIAL_FILENAMES.items():
            path = worker_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"missing pass1 partial: {path}")
            partial_paths[artifact_name].append(path)
    return partial_paths


def _atomic_torch_save(payload: Dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)


def _atomic_write_json(path: str | Path, payload: Dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def _validate_written_artifacts(
    artifacts: Dict[str, tuple[Path, Dict[str, object]]],
) -> None:
    for artifact_name, (path, expected_payload) in artifacts.items():
        if not path.exists():
            raise FileNotFoundError(f"merged artifact was not written: {path}")
        loaded = torch.load(path, map_location="cpu")
        if not isinstance(loaded, dict):
            raise ValueError(f"merged {artifact_name} artifact must contain a dict")
        for key, expected_value in expected_payload.items():
            if isinstance(expected_value, torch.Tensor):
                loaded_value = loaded.get(key)
                if not isinstance(loaded_value, torch.Tensor) or not torch.equal(
                    loaded_value.cpu(), expected_value.cpu()
                ):
                    raise ValueError(f"merged {artifact_name}.{key} failed validation")


__all__ = ["merge_pass1_worker_outputs"]
