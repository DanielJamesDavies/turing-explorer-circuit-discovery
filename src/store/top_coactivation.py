import os
import sys
import torch
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
        S = len(sequence_ids)
        self.candidate_ids = torch.zeros(S, self.M, dtype=torch.int32)
        self.candidate_vals = torch.zeros(S, self.M, dtype=torch.float32)
        self.seq_id_to_row = {int(sid): row for row, sid in enumerate(sequence_ids)}
        print(f"  Candidate dump allocated: {S} sequences x {self.M} candidates "
              f"({S * self.M * 8 / 1e6:.1f} MB)")

    def _score_freq_weighted(self, dense: torch.Tensor, comp_idx: int) -> torch.Tensor:
        """Apply the frequency adjustment factor to the mean activations."""
        ff_start = comp_idx * self.d_sae
        ff_end = ff_start + self.d_sae
        dense *= self.freq_factors[ff_start:ff_end].unsqueeze(0)
        return dense

    def _score_raw(self, dense: torch.Tensor) -> torch.Tensor:
        """Return the mean activations without any frequency adjustment."""
        return dense

    @torch.no_grad()
    def update_batch(
        self,
        batch_ids: torch.Tensor,
        component_latents: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Compute per-sequence candidate profiles and write them to the dump tensors.

        For each component, scatter_add the sparse SAE activations into a dense
        mean-activation vector, apply frequency adjustment (if configured), 
        take the top-N, then keep the global top-M across all components.
        """
        self.allocate(batch_ids.device if batch_ids.is_cuda else None)
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

        for comp_idx in range(self.num_components):
            if comp_idx not in component_latents:
                continue
            top_acts, top_indices = component_latents[comp_idx]
            T = top_acts.shape[1]

            dense = torch.zeros(B, self.d_sae, device=device, dtype=torch.float32)
            
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

            # Route to scoring method
            if mode == "freq_weighted":
                dense = self._score_freq_weighted(dense, comp_idx)
            elif mode == "raw":
                dense = self._score_raw(dense)
            elif mode == "pmi":
                # No frequency adjustment in pmi mode dump
                pass

            n_cand = min(N, dense.shape[1])
            vals, ids = dense.topk(n_cand, dim=1)
            all_vals.append(vals)
            all_ids.append(ids + comp_idx * self.d_sae)

        if mode == "pmi":
            # Increment total tokens processed across all batches
            # (used for the global rate in PMI post-processing)
            # Assuming all components are present in the batch, T is consistent
            # We pick T from the last component processed
            self.total_tokens_processed += B * T

        if not all_vals:
            return

        cand_vals = torch.cat(all_vals, dim=1)
        cand_ids = torch.cat(all_ids, dim=1)

        M_actual = min(self.M, cand_vals.shape[1])
        top_vals, top_pos = cand_vals.topk(M_actual, dim=1)
        top_ids = cand_ids.gather(1, top_pos)

        top_ids_cpu = top_ids.cpu().to(torch.int32)
        top_vals_cpu = top_vals.cpu()

        batch_ids_list = batch_ids.cpu().tolist()
        
        # OLD METHOD (Commented out for easy revert)
        # for b_idx, sid in enumerate(batch_ids_list):
        #     row = self.seq_id_to_row.get(int(sid), -1)
        #     if row < 0:
        #         continue
        #     actual_m = top_ids_cpu.shape[1]
        #     cand_ids_buf[row, :actual_m] = top_ids_cpu[b_idx]
        #     cand_vals_buf[row, :actual_m] = top_vals_cpu[b_idx]

        # NEW VECTORIZED METHOD (VRAM-safe on CPU)
        rows = torch.tensor([self.seq_id_to_row.get(int(sid), -1) for sid in batch_ids_list], dtype=torch.int64)
        valid_mask = rows >= 0
        if valid_mask.any():
            valid_rows = rows[valid_mask]
            actual_m = top_ids_cpu.shape[1]
            cand_ids_buf[valid_rows, :actual_m] = top_ids_cpu[valid_mask]
            cand_vals_buf[valid_rows, :actual_m] = top_vals_cpu[valid_mask]

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

        top_ids, top_vals = top_coactivation_reduce.reduce_topk(
            self.candidate_ids.contiguous(),
            self.candidate_vals.contiguous(),
            seq_offsets.contiguous().cpu(),
            seq_targets_global.contiguous().cpu(),
            sid_to_row,
            self.num_components,
            self.d_sae,
            self.n_latents_per_latent,
        )

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
