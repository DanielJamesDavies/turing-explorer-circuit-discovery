import os
import sys
import time
import torch
from pathlib import Path
from typing import Optional, cast, Dict, Tuple, List
from config import config
from model.turingllm import TuringLLMConfig
from sae.topk_sae import SAEConfig
from store.utils import _AutoAllocTensor


class TopCoactivation:

    top_indices  = _AutoAllocTensor()
    top_values   = _AutoAllocTensor()
    freq_factors = _AutoAllocTensor()
    """
    Computes top co-activating latents: for each target latent, finds which other
    latents fire most strongly (by magnitude) when the target is active.

    Two-phase design:
      1. Dump phase (update_batch): during the second pass, compute per-sequence
         frequency-adjusted candidate profiles and store them in pre-allocated CPU tensors.
      2. Reduce phase (reduce): after the second pass, call a C++ extension that
         aggregates candidates across sequences with sum dedup and produces the
         final top-K per target latent.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_config = TuringLLMConfig()
        self.sae_config = SAEConfig()

        self.num_components = self.llm_config.n_layer * 3
        self.d_sae = self.sae_config.d_sae
        self.n_latents_per_latent = cast(int, config.latents.top_coactivation.n_latents_per_latent or 64)
        self.n_candidates_per_component = cast(int, config.latents.top_coactivation.n_candidates_per_component or 16)

        self.M = min(
            self.n_latents_per_latent * 4,
            self.num_components * self.n_candidates_per_component,
        )

        self._allocated = False
        self._mode: Optional[str] = None
        self.total_tokens_processed = 0

        # Dump buffers (allocated by prepare_dump)
        self.candidate_ids: Optional[torch.Tensor] = None
        self.candidate_vals: Optional[torch.Tensor] = None
        self.seq_id_to_row: Dict[int, int] = {}
        self.sid_to_row_tensor: Optional[torch.Tensor] = None
        self.dump_device_type = "cpu"
        self.dump_timing: Dict[str, float] = {}
        self.dump_batches = 0
        self.dump_components = 0

    @property
    def mode(self) -> str:
        if self._mode is None:
            self._mode = cast(str, config.latents.top_coactivation.mode or "freq_weighted")
        return self._mode

    def allocate(self, device: Optional[torch.device] = None) -> None:
        """Explicitly allocate the large GPU tensors. Safe to call multiple times."""
        if self._allocated:
            if device is not None and device != self.device:
                self.set_device(device)
            return

        if device is not None:
            self.device = device

        self.top_indices = torch.zeros(
            (self.num_components, self.d_sae, self.n_latents_per_latent),
            dtype=torch.int32, device=self.device,
        )
        self.top_values = torch.zeros(
            (self.num_components, self.d_sae, self.n_latents_per_latent),
            dtype=torch.float32, device=self.device,
        )
        self.freq_factors = torch.ones(
            self.num_components * self.d_sae,
            dtype=torch.float32, device=self.device,
        )
        self._allocated = True

    # ------------------------------------------------------------------
    # Frequency factors
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_frequency_factors(self, active_counts: torch.Tensor, alpha: Optional[float] = None, epsilon: float = 1e-6) -> None:
        self.allocate(active_counts.device if active_counts.is_cuda else None)
        if alpha is None:
            alpha = cast(float, config.latents.top_coactivation.freq_alpha or 2.0)
        counts = active_counts.flatten().float()
        self.freq_factors = 1.0 / (torch.log(counts + 1.0 + epsilon)) ** alpha
        self.freq_factors[torch.isinf(self.freq_factors) | torch.isnan(self.freq_factors)] = 1.0

    # ------------------------------------------------------------------
    # Phase 1 — Dump
    # ------------------------------------------------------------------

    def prepare_dump(self, sequence_ids: List[int]) -> None:
        """Pre-allocate candidate tensors and build the sequence-ID-to-row mapping."""
        t0 = time.perf_counter()
        S = len(sequence_ids)
        requested_device = cast(str, getattr(config.latents.top_coactivation, "dump_device", "cpu") or "cpu")
        if requested_device == "gpu" and self.device.type != "cuda":
            print("  [top_coactivation] dump_device='gpu' requested but current device is not CUDA; using CPU dump.")
            requested_device = "cpu"
        dump_device = self.device if requested_device == "gpu" else torch.device("cpu")
        self.dump_device_type = requested_device
        self.candidate_ids = torch.zeros(S, self.M, dtype=torch.int32, device=dump_device)
        self.candidate_vals = torch.zeros(S, self.M, dtype=torch.float32, device=dump_device)
        self.seq_id_to_row = {int(sid): row for row, sid in enumerate(sequence_ids)}
        if sequence_ids:
            max_sid = max(int(sid) for sid in sequence_ids)
            self.sid_to_row_tensor = torch.full((max_sid + 1,), -1, dtype=torch.int64, device=dump_device)
            sids = torch.tensor([int(sid) for sid in sequence_ids], dtype=torch.int64, device=dump_device)
            self.sid_to_row_tensor[sids] = torch.arange(S, dtype=torch.int64, device=dump_device)
        else:
            self.sid_to_row_tensor = torch.empty(0, dtype=torch.int64, device=dump_device)
        self.dump_timing = {
            "prepare": time.perf_counter() - t0,
            "allocate": 0.0,
            "dense_zero": 0.0,
            "scatter": 0.0,
            "score": 0.0,
            "component_topk": 0.0,
            "concat_global_topk": 0.0,
            "cpu_transfer": 0.0,
            "row_lookup": 0.0,
            "cpu_write": 0.0,
            "final_cpu_transfer": 0.0,
            "total_update": 0.0,
        }
        self.dump_batches = 0
        self.dump_components = 0
        print(f"  Candidate dump allocated on {dump_device}: {S} sequences x {self.M} candidates "
              f"({S * self.M * 8 / 1e6:.1f} MB)")

    def _add_dump_time(self, key: str, seconds: float) -> None:
        self.dump_timing[key] = self.dump_timing.get(key, 0.0) + seconds

    def dump_timing_summary(self) -> str:
        if not self.dump_timing:
            return "  [timing] top_coactivation update_batch profile unavailable"
        total_update = self.dump_timing.get("total_update", 0.0)
        lines = [
            "  [timing] top_coactivation update_batch profile:",
            f"    batches={self.dump_batches:,}  components={self.dump_components:,}",
            f"    prepare:             {self.dump_timing.get('prepare', 0.0) * 1000:.1f} ms",
            f"    allocate/check:      {self.dump_timing.get('allocate', 0.0):.2f} s",
            f"    dense zero:          {self.dump_timing.get('dense_zero', 0.0):.2f} s",
            f"    scatter/mean:        {self.dump_timing.get('scatter', 0.0):.2f} s",
            f"    scoring:             {self.dump_timing.get('score', 0.0):.2f} s",
            f"    component topk:      {self.dump_timing.get('component_topk', 0.0):.2f} s",
            f"    concat/global topk:  {self.dump_timing.get('concat_global_topk', 0.0):.2f} s",
            f"    transfer to CPU:     {self.dump_timing.get('cpu_transfer', 0.0):.2f} s",
            f"    row lookup:          {self.dump_timing.get('row_lookup', 0.0):.2f} s",
            f"    CPU write:           {self.dump_timing.get('cpu_write', 0.0):.2f} s",
            f"    final CPU transfer:  {self.dump_timing.get('final_cpu_transfer', 0.0):.2f} s",
            f"    measured total:      {total_update:.2f} s",
        ]
        if self.dump_batches:
            lines.append(f"    avg update/batch:    {total_update / self.dump_batches * 1000:.1f} ms")
        return "\n".join(lines)

    def _score_freq_weighted(self, dense: torch.Tensor, comp_idx: int) -> torch.Tensor:
        """Apply the frequency adjustment factor to the mean activations."""
        ff_start = comp_idx * self.d_sae
        ff_end = ff_start + self.d_sae
        dense *= self.freq_factors[ff_start:ff_end].unsqueeze(0)
        return dense

    def _score_raw(self, dense: torch.Tensor) -> torch.Tensor:
        """Return the mean activations without any frequency adjustment."""
        return dense

    def _target_shard_ranges(self, n_targets: int, n_shards: int) -> List[Tuple[int, int]]:
        n_shards = max(1, min(n_shards, n_targets))
        base = n_targets // n_shards
        rem = n_targets % n_shards
        ranges: List[Tuple[int, int]] = []
        start = 0
        for shard_idx in range(n_shards):
            size = base + (1 if shard_idx < rem else 0)
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

    def _save_reduce_shard(
        self,
        shard_dir: Path,
        shard_idx: int,
        n_shards: int,
        target_start: int,
        target_end: int,
        shard_ids: torch.Tensor,
        shard_vals: torch.Tensor,
    ) -> Path:
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"shard_{shard_idx:05d}.pt"
        tmp_path = shard_dir / f".shard_{shard_idx:05d}.pt.tmp"
        payload = {
            "schema": "top_coactivation_reduce_shard_v1",
            "shard_idx": shard_idx,
            "n_shards": n_shards,
            "target_start": target_start,
            "target_end": target_end,
            "num_components": self.num_components,
            "d_sae": self.d_sae,
            "n_latents_per_latent": self.n_latents_per_latent,
            "mode": self.mode,
            "top_indices": shard_ids.reshape(target_end - target_start, self.n_latents_per_latent).cpu().to(torch.int32),
            "top_values": shard_vals.reshape(target_end - target_start, self.n_latents_per_latent).cpu().to(torch.float32),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, shard_path)
        return shard_path

    def _cleanup_reduce_shard_files(self, shard_paths: List[Path]) -> None:
        for path in shard_paths:
            for candidate in (path, path.with_name(f".{path.name}.tmp")):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    print(f"  [top_coactivation] WARNING: could not remove partial shard file {candidate}: {exc}")

    def _merge_reduce_shards(
        self,
        shard_paths: List[Path],
        ranges: List[Tuple[int, int]],
        n_targets: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_ids = torch.empty((n_targets, self.n_latents_per_latent), dtype=torch.int32)
        flat_vals = torch.empty((n_targets, self.n_latents_per_latent), dtype=torch.float32)
        for shard_idx, (path, (target_start, target_end)) in enumerate(zip(shard_paths, ranges)):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") != "top_coactivation_reduce_shard_v1":
                raise ValueError(f"Unexpected top-coactivation shard schema in {path}")
            if int(payload.get("shard_idx", -1)) != shard_idx:
                raise ValueError(f"Unexpected shard index in {path}")
            if int(payload.get("target_start", -1)) != target_start or int(payload.get("target_end", -1)) != target_end:
                raise ValueError(f"Unexpected target range in {path}")
            if int(payload.get("num_components", -1)) != self.num_components or int(payload.get("d_sae", -1)) != self.d_sae:
                raise ValueError(f"Shard {path} was produced for a different latent layout")
            if int(payload.get("n_latents_per_latent", -1)) != self.n_latents_per_latent:
                raise ValueError(f"Shard {path} was produced with a different top-K")

            expected_shape = (target_end - target_start, self.n_latents_per_latent)
            shard_ids = payload["top_indices"].to(torch.int32)
            shard_vals = payload["top_values"].to(torch.float32)
            if tuple(shard_ids.shape) != expected_shape or tuple(shard_vals.shape) != expected_shape:
                raise ValueError(f"Shard {path} has unexpected tensor shape")
            flat_ids[target_start:target_end] = shard_ids
            flat_vals[target_start:target_end] = shard_vals
        return flat_ids, flat_vals

    @torch.no_grad()
    def update_batch(
        self,
        batch_ids: torch.Tensor,
        component_latents: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        dump_row_start: Optional[int] = None,
    ) -> None:
        """
        Compute per-sequence candidate profiles and write them to the dump tensors.

        For each component, scatter_add the sparse SAE activations into a dense
        mean-activation vector, apply frequency adjustment (if configured), 
        take the top-N, then keep the global top-M across all components.
        """
        total_t0 = time.perf_counter()
        t0 = total_t0
        self.allocate(batch_ids.device if batch_ids.is_cuda else None)
        self._add_dump_time("allocate", time.perf_counter() - t0)
        cand_ids_buf = self.candidate_ids
        cand_vals_buf = self.candidate_vals
        assert cand_ids_buf is not None and cand_vals_buf is not None, \
            "Call prepare_dump() before update_batch()"
        B = batch_ids.shape[0]
        device = self.device
        N = self.n_candidates_per_component
        mode = self.mode

        all_vals: list[torch.Tensor] = []
        all_ids: list[torch.Tensor] = []
        last_T: Optional[int] = None

        for comp_idx in range(self.num_components):
            if comp_idx not in component_latents:
                continue
            top_acts, top_indices = component_latents[comp_idx]
            T = top_acts.shape[1]
            last_T = T

            t0 = time.perf_counter()
            dense = torch.zeros(B, self.d_sae, device=device, dtype=torch.float32)
            self._add_dump_time("dense_zero", time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            if mode == "pmi":
                # Binary presence count (how many tokens in the sequence did this fire at?)
                dense.scatter_add_(
                    1,
                    top_indices.reshape(B, -1).long(),
                    (top_acts.reshape(B, -1) > 0).float(),
                )
                # No division by T - we want the absolute count of tokens for PMI
            else:
                # Magnitude sum
                dense.scatter_add_(
                    1,
                    top_indices.reshape(B, -1).long(),
                    top_acts.reshape(B, -1).float(),
                )
                dense /= T
            self._add_dump_time("scatter", time.perf_counter() - t0)

            # Route to scoring method
            t0 = time.perf_counter()
            if mode == "freq_weighted":
                dense = self._score_freq_weighted(dense, comp_idx)
            elif mode == "raw":
                dense = self._score_raw(dense)
            elif mode == "pmi":
                # No frequency adjustment in pmi mode dump
                pass
            self._add_dump_time("score", time.perf_counter() - t0)

            n_cand = min(N, dense.shape[1])
            t0 = time.perf_counter()
            vals, ids = dense.topk(n_cand, dim=1)
            self._add_dump_time("component_topk", time.perf_counter() - t0)
            all_vals.append(vals)
            all_ids.append(ids + comp_idx * self.d_sae)
            self.dump_components += 1

        if mode == "pmi" and last_T is not None:
            # Increment total tokens processed across all batches
            # (used for the global rate in PMI post-processing)
            # Assuming all components are present in the batch, T is consistent
            # We pick T from the last component processed
            self.total_tokens_processed += B * last_T

        if not all_vals:
            self.dump_batches += 1
            self._add_dump_time("total_update", time.perf_counter() - total_t0)
            return

        t0 = time.perf_counter()
        cand_vals = torch.cat(all_vals, dim=1)
        cand_ids = torch.cat(all_ids, dim=1)

        M_actual = min(self.M, cand_vals.shape[1])
        top_vals, top_pos = cand_vals.topk(M_actual, dim=1)
        top_ids = cand_ids.gather(1, top_pos)
        self._add_dump_time("concat_global_topk", time.perf_counter() - t0)

        if dump_row_start is not None:
            t0 = time.perf_counter()
            rows = torch.arange(
                dump_row_start,
                dump_row_start + B,
                dtype=torch.int64,
                device=cand_ids_buf.device,
            )
            self._add_dump_time("row_lookup", time.perf_counter() - t0)
            t0 = time.perf_counter()
            actual_m = top_ids.shape[1]
            if self.dump_device_type == "gpu":
                cand_ids_buf[rows, :actual_m] = top_ids.to(torch.int32)
                cand_vals_buf[rows, :actual_m] = top_vals
            else:
                top_ids_cpu = top_ids.cpu().to(torch.int32)
                top_vals_cpu = top_vals.cpu()
                self._add_dump_time("cpu_transfer", time.perf_counter() - t0)
                t0 = time.perf_counter()
                cand_ids_buf[rows, :actual_m] = top_ids_cpu
                cand_vals_buf[rows, :actual_m] = top_vals_cpu
            self._add_dump_time("cpu_write", time.perf_counter() - t0)
        elif self.dump_device_type == "gpu":
            assert self.sid_to_row_tensor is not None
            t0 = time.perf_counter()
            batch_ids_d = batch_ids.to(self.sid_to_row_tensor.device, dtype=torch.int64)
            in_range = (batch_ids_d >= 0) & (batch_ids_d < self.sid_to_row_tensor.shape[0])
            rows = torch.full_like(batch_ids_d, -1)
            rows[in_range] = self.sid_to_row_tensor[batch_ids_d[in_range]]
            valid_mask = rows >= 0
            self._add_dump_time("row_lookup", time.perf_counter() - t0)
            if valid_mask.any():
                t0 = time.perf_counter()
                actual_m = top_ids.shape[1]
                cand_ids_buf[rows[valid_mask], :actual_m] = top_ids[valid_mask].to(torch.int32)
                cand_vals_buf[rows[valid_mask], :actual_m] = top_vals[valid_mask]
                self._add_dump_time("cpu_write", time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            top_ids_cpu = top_ids.cpu().to(torch.int32)
            top_vals_cpu = top_vals.cpu()
            self._add_dump_time("cpu_transfer", time.perf_counter() - t0)

            batch_ids_list = batch_ids.cpu().tolist()
            t0 = time.perf_counter()
            rows = torch.tensor([self.seq_id_to_row.get(int(sid), -1) for sid in batch_ids_list], dtype=torch.int64)
            valid_mask = rows >= 0
            self._add_dump_time("row_lookup", time.perf_counter() - t0)
            if valid_mask.any():
                t0 = time.perf_counter()
                valid_rows = rows[valid_mask]
                actual_m = top_ids_cpu.shape[1]
                cand_ids_buf[valid_rows, :actual_m] = top_ids_cpu[valid_mask]
                cand_vals_buf[valid_rows, :actual_m] = top_vals_cpu[valid_mask]
                self._add_dump_time("cpu_write", time.perf_counter() - t0)
        self.dump_batches += 1
        self._add_dump_time("total_update", time.perf_counter() - total_t0)

    # ------------------------------------------------------------------
    # Phase 2 — Reduce (C++ extension)
    # ------------------------------------------------------------------

    def reduce(
        self,
        seq_offsets: torch.Tensor,
        seq_targets_global: torch.Tensor,
        seq_len: int = 64,
        active_count: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Run the C++ post-processing reduction.
        Aggregates candidate dumps across sequences per target latent using
        sum-dedup, then keeps the top-K co-activating latents.
        """
        native_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "native"))
        if native_dir not in sys.path:
            sys.path.insert(0, native_dir)
        try:
            import top_coactivation_reduce
        except ImportError:
            raise ImportError(
                f"Could not load top_coactivation_reduce from {native_dir}. "
                f"Build it with: cd src/native && python setup.py build_ext --inplace"
            )

        assert self.candidate_ids is not None and self.candidate_vals is not None

        max_sid = int(seq_offsets.shape[0])
        
        # OLD METHOD (Commented out for easy revert)
        # sid_to_row = torch.full((max_sid,), -1, dtype=torch.int64)
        # for sid, row in self.seq_id_to_row.items():
        #     if 0 < sid < max_sid:
        #         sid_to_row[sid] = row

        # NEW VECTORIZED METHOD (VRAM-safe on CPU)
        sid_to_row = torch.full((max_sid,), -1, dtype=torch.int64)
        if self.seq_id_to_row:
            sids = torch.tensor(list(self.seq_id_to_row.keys()), dtype=torch.int64)
            rows = torch.tensor(list(self.seq_id_to_row.values()), dtype=torch.int64)
            mask = (sids > 0) & (sids < max_sid)
            sid_to_row[sids[mask]] = rows[mask]

        reduce_omp_threads = getattr(config.latents.top_coactivation, "reduce_omp_threads", None)
        reduce_schedule_chunk = int(getattr(config.latents.top_coactivation, "reduce_schedule_chunk", 256) or 256)
        reduce_backend = cast(str, getattr(config.latents.top_coactivation, "reduce_backend", "single_process") or "single_process")
        reduce_shards = int(getattr(config.latents.top_coactivation, "reduce_shards", 1) or 1)
        reduce_shard_output_dir = getattr(config.latents.top_coactivation, "reduce_shard_output_dir", None)
        print(
            "  [top_coactivation] reducer controls: "
            f"backend={reduce_backend} shards={reduce_shards} "
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')} "
            f"override={reduce_omp_threads if reduce_omp_threads is not None else '<none>'} "
            f"schedule=dynamic,{reduce_schedule_chunk}"
        )

        reduce_kwargs = {
            "omp_threads": int(reduce_omp_threads) if reduce_omp_threads is not None else 0,
            "schedule_chunk": reduce_schedule_chunk,
            "print_timings": True,
        }

        t_copy = time.perf_counter()
        candidate_ids_cpu = self.candidate_ids.contiguous().cpu()
        candidate_vals_cpu = self.candidate_vals.contiguous().cpu()
        self._add_dump_time("final_cpu_transfer", time.perf_counter() - t_copy)

        reduce_args = (
            candidate_ids_cpu,
            candidate_vals_cpu,
            seq_offsets.contiguous().cpu(),
            seq_targets_global.contiguous().cpu(),
            sid_to_row,
            self.num_components,
            self.d_sae,
            self.n_latents_per_latent,
        )

        n_targets = self.num_components * self.d_sae

        def call_reduce_range(start: int, end: int, print_timings: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            kwargs = dict(reduce_kwargs)
            kwargs["print_timings"] = print_timings
            kwargs["target_start"] = start
            kwargs["target_end"] = end
            try:
                return top_coactivation_reduce.reduce_topk(*reduce_args, **kwargs)
            except TypeError as exc:
                legacy_bits = ("omp_threads", "schedule_chunk", "print_timings", "target_start", "target_end")
                if not any(bit in str(exc) for bit in legacy_bits):
                    raise
                if start != 0 or end != n_targets:
                    raise RuntimeError(
                        "target_sharded reduction requires a rebuilt Phase 7 top_coactivation_reduce extension."
                    ) from exc
                print("  [top_coactivation] native reducer was built before Phase 7 controls; using legacy call signature.")
                legacy_ids, legacy_vals = top_coactivation_reduce.reduce_topk(*reduce_args)
                return legacy_ids.reshape(n_targets, self.n_latents_per_latent), legacy_vals.reshape(n_targets, self.n_latents_per_latent)

        if reduce_backend == "target_sharded":
            ranges = self._target_shard_ranges(n_targets, reduce_shards)
            print(f"  [top_coactivation] target_sharded ranges: {len(ranges)}")
            if reduce_shard_output_dir:
                shard_dir = Path(str(reduce_shard_output_dir))
                print(f"  [top_coactivation] writing reducer shards to {shard_dir}")
                shard_paths: List[Path] = []
                try:
                    for shard_idx, (start, end) in enumerate(ranges):
                        print(f"  [top_coactivation] reducing shard {shard_idx + 1}/{len(ranges)} targets [{start}, {end})")
                        shard_path = shard_dir / f"shard_{shard_idx:05d}.pt"
                        shard_paths.append(shard_path)
                        shard_ids, shard_vals = call_reduce_range(start, end, print_timings=(shard_idx == 0))
                        self._save_reduce_shard(
                            shard_dir,
                            shard_idx,
                            len(ranges),
                            start,
                            end,
                            shard_ids,
                            shard_vals,
                        )
                except Exception:
                    self._cleanup_reduce_shard_files(shard_paths)
                    raise
                print(f"  [top_coactivation] merging {len(shard_paths)} reducer shard files")
                flat_ids, flat_vals = self._merge_reduce_shards(shard_paths, ranges, n_targets)
            else:
                flat_ids = torch.empty((n_targets, self.n_latents_per_latent), dtype=torch.int32)
                flat_vals = torch.empty((n_targets, self.n_latents_per_latent), dtype=torch.float32)
                for shard_idx, (start, end) in enumerate(ranges):
                    print(f"  [top_coactivation] reducing shard {shard_idx + 1}/{len(ranges)} targets [{start}, {end})")
                    shard_ids, shard_vals = call_reduce_range(start, end, print_timings=(shard_idx == 0))
                    flat_ids[start:end] = shard_ids.reshape(end - start, self.n_latents_per_latent)
                    flat_vals[start:end] = shard_vals.reshape(end - start, self.n_latents_per_latent)
            top_ids = flat_ids.reshape(self.num_components, self.d_sae, self.n_latents_per_latent)
            top_vals = flat_vals.reshape(self.num_components, self.d_sae, self.n_latents_per_latent)
        else:
            flat_ids, flat_vals = call_reduce_range(0, n_targets, print_timings=True)
            top_ids = flat_ids.reshape(self.num_components, self.d_sae, self.n_latents_per_latent)
            top_vals = flat_vals.reshape(self.num_components, self.d_sae, self.n_latents_per_latent)

        self.top_indices = top_ids
        self.top_values = top_vals
        self._allocated = True

        if self.mode == "pmi":
            assert active_count is not None, "active_count must be provided for pmi mode reduction"
            self._apply_pmi_postprocess(active_count, seq_offsets, seq_targets_global, seq_len)

        # Free dump buffers
        self.candidate_ids = None
        self.candidate_vals = None
        self.seq_id_to_row = {}
        self.sid_to_row_tensor = None

    def _apply_pmi_postprocess(
        self,
        active_count: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_targets_global: torch.Tensor,
        seq_len: int,
    ) -> None:
        """
        Post-process top_values (binary firing counts) into PMI log-scores.

        All tensors are kept on CPU throughout — self.top_values is on CPU after
        reduce_topk returns, and there is no reason to move to GPU for this arithmetic.
        """
        C, d_sae, K = self.top_values.shape
        pmi_clamp_min = cast(float, config.latents.top_coactivation.pmi_clamp_min or -5.0)
        pmi_clamp_max = cast(float, config.latents.top_coactivation.pmi_clamp_max or 10.0)

        # Derive total_tokens_globally from active_count and SAE sparsity K.
        # active_count[c].sum() == total_tokens_globally * k_sae (each token fires k_sae latents
        # per component), so total_tokens = active_count[0].sum() / k_sae.
        # This is the correct all-data global count from Pass 1, not the smaller top_ctx count.
        k_sae = self.sae_config.k
        total_tokens_globally = max(1, int(active_count[0].sum().item()) // k_sae)

        # global_rate[j]: how often latent j fires per token, globally across all data.
        # active_count is [C, d_sae] on CPU — keep everything on CPU.
        global_rate = active_count.flatten().float() / total_tokens_globally

        # per_target_tokens[g]: total token positions across all sequences in target g's top_ctx.
        per_target_tokens = self._compute_total_tokens_per_target(seq_offsets, seq_targets_global, seq_len)

        # context_rate[T, j]: how often latent j fires per token in target T's context sequences.
        context_count = self.top_values  # [C, d_sae, K] — summed binary counts from reduce, on CPU
        context_rate = context_count / per_target_tokens.view(C, d_sae, 1).clamp(min=1)

        # j_rate[T, j]: global firing rate for each candidate latent j.
        j_global_ids = self.top_indices.long()  # [C, d_sae, K]
        j_rate = global_rate[j_global_ids]

        # PMI = log( P(j fires | T's context) / P(j fires globally) )
        pmi = (context_rate / j_rate.clamp(min=1e-10)).log().clamp(pmi_clamp_min, pmi_clamp_max)

        self.top_values.copy_(pmi)

    def _compute_total_tokens_per_target(
        self,
        seq_offsets: torch.Tensor,
        seq_targets_global: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        For each target latent (global ID), count the total token positions across all sequences
        in its top_ctx set. Returns a float tensor of shape [C * d_sae].

        seq_offsets and seq_targets_global form a CSR structure indexed by SEQUENCE ID:
          seq_targets_global[seq_offsets[sid] : seq_offsets[sid+1]] = target IDs for sequence sid.
        We need the inverse: for each target g, how many sequences have g in their top_ctx?
        """
        num_targets = self.num_components * self.d_sae
        # scatter_add to count how many sequences each target global ID appears in
        valid_mask = (seq_targets_global >= 0) & (seq_targets_global < num_targets)
        valid_targets = seq_targets_global[valid_mask].long()
        target_seq_counts = torch.zeros(num_targets, dtype=torch.float32)
        target_seq_counts.scatter_add_(
            0, valid_targets, torch.ones(valid_targets.shape[0], dtype=torch.float32)
        )
        return target_seq_counts * seq_len

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, path: str) -> None:
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.allocate()
            if "top_indices" in checkpoint:
                self.top_indices.copy_(checkpoint["top_indices"])
            if "top_values" in checkpoint:
                self.top_values.copy_(checkpoint["top_values"])
            if "freq_factors" in checkpoint:
                self.freq_factors.copy_(checkpoint["freq_factors"])
            if "total_tokens_processed" in checkpoint:
                self.total_tokens_processed = checkpoint["total_tokens_processed"]
            
            # Verify mode compatibility
            stored_mode = checkpoint.get("mode")
            if stored_mode is not None and stored_mode != self.mode:
                print(f"[top_coactivation] WARNING: Stored mode '{stored_mode}' "
                      f"differs from current config mode '{self.mode}'. "
                      f"Co-activation values may be inconsistent.")
        except Exception as e:
            print(f"TopCoactivation load failed (likely no file yet): {e}")

    def save(self, path: str) -> None:
        if not self._allocated:
            return
        torch.save({
            "top_indices": self.top_indices,
            "top_values": self.top_values,
            "freq_factors": self.freq_factors,
            "total_tokens_processed": self.total_tokens_processed,
            "mode": self.mode,
        }, path)

    def set_device(self, device: torch.device) -> None:
        self.device = device
        if self._allocated:
            self.top_indices = self.top_indices.to(device)
            self.top_values = self.top_values.to(device)
            self.freq_factors = self.freq_factors.to(device)


top_coactivation = TopCoactivation(device=torch.device("cpu"))
