"""
Negative context builder (ANN retrieval).

After Pass 1 has collected seq_repr, top_ctx, and mid_ctx, this module:
  1. Builds a TorchANNIndex (exact cosine via chunked matmul + topk) over all
     sequence representations.
  2. For each SAE component, vectorises the entire pipeline end-to-end on the
     compute device — no Python loops over individual latents:
       a. Active latent detection via tensor nonzero.
       b. Query matrix built with a single gather + index_add_ scatter-mean.
       c. Batched search (F.normalize @ index.T + topk).
       d. Membership filter via torch.searchsorted on encoded (qi, seq_id) keys.
       e. Bulk write to neg_ctx via index_copy_.
  3. Stores the top N_neg hardest negatives per latent (cosine similarity order).

Compute device: config.hardware.ann_device  ("auto" | "gpu" | "cpu")
  "auto" — GPU if CUDA available, else CPU.

No external ANN library is required — the entire pipeline uses PyTorch primitives
(cuBLAS on GPU, MKL/OpenBLAS on CPU).

All timing is printed via tqdm for performance analysis and optimisation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, cast

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config
from model.turingllm import TuringLLMConfig

if TYPE_CHECKING:
    from store.context import Context
    from store.seq_repr import SeqRepr


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class NegCtxStats:
    n_latents_attempted:       int = 0
    n_latents_skipped_low_pos: int = 0
    n_latents_populated:       int = 0
    n_latents_zero_negatives:  int = 0
    fill_counts: List[int] = field(default_factory=list)

    t_index_build: float = 0.0
    t_pos_collect: float = 0.0   # active detection + pair gather
    t_qmat_build:  float = 0.0   # scatter-mean query matrix
    t_query:       float = 0.0   # matmul + topk search
    t_filter:      float = 0.0   # searchsorted filter
    t_write:       float = 0.0   # index_copy_ write to neg_ctx
    t_total:       float = 0.0

    @property
    def fill_rate_mean(self) -> float:
        return float(np.mean(self.fill_counts)) if self.fill_counts else 0.0

    @property
    def fill_rate_p10(self) -> float:
        return float(np.percentile(self.fill_counts, 10)) if self.fill_counts else 0.0

    @property
    def fill_rate_p50(self) -> float:
        return float(np.percentile(self.fill_counts, 50)) if self.fill_counts else 0.0

    @property
    def fill_rate_p90(self) -> float:
        return float(np.percentile(self.fill_counts, 90)) if self.fill_counts else 0.0

    def print_summary(self, n_sequences: int) -> None:
        print(f"  [neg_ctx] Latents attempted:       {self.n_latents_attempted:,}")
        print(f"  [neg_ctx] Skipped (low PosCtx):    {self.n_latents_skipped_low_pos:,}")
        print(f"  [neg_ctx] Populated:               {self.n_latents_populated:,}")
        print(f"  [neg_ctx] Zero negatives found:    {self.n_latents_zero_negatives:,}")
        if self.fill_counts:
            print(f"  [neg_ctx] Fill count (/{n_sequences}) "
                  f"mean={self.fill_rate_mean:.1f}  "
                  f"p10={self.fill_rate_p10:.1f}  "
                  f"p50={self.fill_rate_p50:.1f}  "
                  f"p90={self.fill_rate_p90:.1f}  "
                  f"min={min(self.fill_counts)}  "
                  f"max={max(self.fill_counts)}")
        print(f"  [neg_ctx] Timing breakdown:")
        print(f"    Index build:      {self.t_index_build * 1000:8.1f} ms")
        print(f"    PosCtx collect:   {self.t_pos_collect * 1000:8.1f} ms")
        print(f"    Qmat scatter:     {self.t_qmat_build  * 1000:8.1f} ms")
        print(f"    Matmul + topk:    {self.t_query       * 1000:8.1f} ms")
        print(f"    Filter:           {self.t_filter      * 1000:8.1f} ms")
        print(f"    Write:            {self.t_write       * 1000:8.1f} ms")
        print(f"    Total:            {self.t_total       * 1000:8.1f} ms  ({self.t_total:.1f} s)")

    def save(self, path: str) -> None:
        data = {
            "n_latents_attempted":       self.n_latents_attempted,
            "n_latents_skipped_low_pos": self.n_latents_skipped_low_pos,
            "n_latents_populated":       self.n_latents_populated,
            "n_latents_zero_negatives":  self.n_latents_zero_negatives,
            "fill_rate_mean": round(self.fill_rate_mean, 2),
            "fill_rate_p10":  round(self.fill_rate_p10,  2),
            "fill_rate_p50":  round(self.fill_rate_p50,  2),
            "fill_rate_p90":  round(self.fill_rate_p90,  2),
            "fill_count_min": int(min(self.fill_counts)) if self.fill_counts else 0,
            "fill_count_max": int(max(self.fill_counts)) if self.fill_counts else 0,
            "t_index_build_ms": round(self.t_index_build * 1000, 1),
            "t_pos_collect_ms": round(self.t_pos_collect * 1000, 1),
            "t_qmat_build_ms":  round(self.t_qmat_build  * 1000, 1),
            "t_query_ms":       round(self.t_query       * 1000, 1),
            "t_filter_ms":      round(self.t_filter      * 1000, 1),
            "t_write_ms":       round(self.t_write       * 1000, 1),
            "t_total_ms":       round(self.t_total       * 1000, 1),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# TorchANNIndex — pure PyTorch, no external dependency
# ---------------------------------------------------------------------------

class TorchANNIndex:
    """
    Exact cosine similarity index.

    Build: L2-normalise vecs → store on device.
    Search: F.normalize(queries) @ index.T → topk  (cuBLAS / MKL).

    Results are returned on the same device as the index (no CPU round-trip),
    so downstream GPU filtering can start immediately.
    """

    def __init__(self, vecs: torch.Tensor, device: torch.device):
        t0 = time.perf_counter()
        self.device = device
        self.n      = vecs.shape[0]
        self.dim    = vecs.shape[1]

        # chunk_size: larger = fewer kernel launches = better GPU utilisation.
        # Peak intermediate tensor = chunk_size × N × 4 B.
        # GPU:  4096 × 65536 × 4 = 1 GB — safe for 16 GB+ VRAM.
        # CPU:  512  × 65536 × 4 = 128 MB — conservative.
        self.chunk_size = 4096 if device.type == "cuda" else 512

        self.index = F.normalize(vecs.float(), dim=1).to(device)

        build_ms = (time.perf_counter() - t0) * 1000
        print(f"  [neg_ctx] ANN index on {device} "
              f"— {self.n:,} vecs × {self.dim} dims "
              f"— built in {build_ms:.1f} ms "
              f"— chunk_size={self.chunk_size}")

    @torch.no_grad()
    def search(
        self,
        queries: torch.Tensor,   # [Q, D], any device
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (similarities [Q, k], indices [Q, k]) on self.device.
        Similarities are in descending order (highest first = hardest negatives first).
        """
        K      = min(k, self.n)
        q_norm = F.normalize(queries.float(), dim=1)
        if q_norm.device != self.device:
            q_norm = q_norm.to(self.device)

        Q        = q_norm.shape[0]
        all_sims = torch.empty(Q, K, dtype=torch.float32, device=self.device)
        all_idxs = torch.empty(Q, K, dtype=torch.int64,   device=self.device)

        for start in range(0, Q, self.chunk_size):
            end   = min(start + self.chunk_size, Q)
            sims  = q_norm[start:end] @ self.index.T                   # [C, N]
            ts, ti = sims.topk(K, dim=1, sorted=True)
            all_sims[start:end] = ts
            all_idxs[start:end] = ti

        return all_sims, all_idxs   # on self.device


# ---------------------------------------------------------------------------
# Device selector
# ---------------------------------------------------------------------------

def _ann_device() -> torch.device:
    cfg = cast(str, config.hardware.ann_device or "auto")
    if cfg == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("hardware.ann_device = 'gpu' but CUDA is not available.")
        return torch.device("cuda")
    if cfg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Per-component processing — fully vectorised, no Python loop over latents
# ---------------------------------------------------------------------------

# Max pairs processed in one scatter-mean chunk (bounds peak GPU tensor size).
# 65536 pairs × 1024 × 4 B = 256 MB — safe even on 8 GB VRAM.
_PAIR_CHUNK = 65536


@torch.no_grad()
def _process_component(
    comp_idx:      int,
    top_ctx:       "Context",
    mid_ctx:       "Context",
    neg_ctx:       "Context",
    index:         TorchANNIndex,
    K:             int,
    n_neg:         int,
    min_pos_ctx:   int,
    stats:         NegCtxStats,
    total_n_seqs:  int,
    slot_to_id_d:  "torch.Tensor | None",
    id_to_slot_d:  "torch.Tensor | None",
) -> dict:
    """
    Process one SAE component end-to-end without any Python loop over latents.

    slot_to_id_d / id_to_slot_d are None when seq_repr is uncapped (slot == seq_id).

    Returns a dict of per-step timing (seconds) for the tqdm postfix.
    """
    device = index.device
    d_sae  = top_ctx.ctx_seq_idx.shape[1]
    timing: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 1. Move component slices to compute device                          #
    # ------------------------------------------------------------------ #

    top_ids_d  = top_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_top]
    top_mask_d = top_ctx.ctx_seq_val[comp_idx].to(device).float() > 0          # [d_sae, N_top]
    mid_ids_d  = mid_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_mid]
    mid_mask_d = mid_ctx.ctx_seq_val[comp_idx].to(device).float() > 0          # [d_sae, N_mid]

    # ------------------------------------------------------------------ #
    # 2. Active latent detection + flatten (qi, seq_id) pairs            #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    pos_counts = top_mask_d.sum(dim=1) + mid_mask_d.sum(dim=1)   # [d_sae]
    active_js  = (pos_counts >= min_pos_ctx).nonzero(as_tuple=True)[0]  # [Q]
    Q          = active_js.shape[0]

    stats.n_latents_attempted       += d_sae
    stats.n_latents_skipped_low_pos += d_sae - Q

    if Q == 0:
        timing["pos"] = time.perf_counter() - t0
        return timing

    # Gather (qi_within_Q, seq_id) pairs for all active latents in one pass.
    at_ids  = top_ids_d[active_js]    # [Q, N_top]
    at_mask = top_mask_d[active_js]   # [Q, N_top]
    am_ids  = mid_ids_d[active_js]    # [Q, N_mid]
    am_mask = mid_mask_d[active_js]   # [Q, N_mid]

    t_qi, t_ki = at_mask.nonzero(as_tuple=True)   # [M_top]
    m_qi, m_ki = am_mask.nonzero(as_tuple=True)   # [M_mid]

    all_qi     = torch.cat([t_qi, m_qi])
    all_seqids = torch.cat([at_ids[t_qi, t_ki], am_ids[m_qi, m_ki]])   # 1-indexed

    # Remove sentinel 0 only — range check is handled by the slot filter below.
    valid      = (all_seqids > 0) & (all_seqids <= total_n_seqs)
    all_qi     = all_qi[valid]
    all_seqids = all_seqids[valid]
    M          = all_qi.shape[0]

    timing["pos"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 3. Build query matrix — scatter-mean of PosCtx reps                #
    #    Chunked over pairs to bound peak intermediate tensor size.       #
    #    When capped, filter to pairs whose seq is actually in the index, #
    #    then convert seq_ids → 0-indexed slot rows.                      #
    # ------------------------------------------------------------------ #

    t0   = time.perf_counter()
    D    = index.dim
    sums = torch.zeros(Q, D, dtype=torch.float32, device=device)

    if id_to_slot_d is not None:
        # Capped: only sequences stored in the ANN index contribute to the centroid.
        slots    = id_to_slot_d[all_seqids]      # [M] slot (0 = not stored)
        in_index = slots > 0
        qi_qmat  = all_qi[in_index]
        rows_all = (slots[in_index] - 1)         # 0-indexed slot row
    else:
        qi_qmat  = all_qi
        rows_all = all_seqids - 1                # 0-indexed seq_id row (slot == seq_id)

    Mq  = qi_qmat.shape[0]
    cnt = torch.bincount(qi_qmat, minlength=Q).float().unsqueeze(1)    # [Q, 1]

    for pair_start in range(0, Mq, _PAIR_CHUNK):
        pair_end    = min(pair_start + _PAIR_CHUNK, Mq)
        chunk_rows  = rows_all[pair_start:pair_end]
        chunk_qi    = qi_qmat[pair_start:pair_end]
        chunk_reps  = index.index[chunk_rows]            # [C, D] float32
        sums.index_add_(0, chunk_qi, chunk_reps)

    qmat = F.normalize(sums / cnt.clamp(min=1e-8), dim=1)   # [Q, D]
    timing["qmat"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 4. Batched search — matmul + topk on device                        #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()
    nn_sims, nn_idx = index.search(qmat, K)   # [Q, K] on device
    timing["query"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 5. Membership filter — GPU searchsorted on encoded (qi, seq_id)    #
    #    Encodes every pair as  qi × stride + seq_id  (int64)            #
    #    to allow a single sorted-set lookup across all Q×K candidates.  #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    # stride must be > max possible seq_id to avoid encoding collisions.
    stride = total_n_seqs + 2
    # Positive set: encode all (qi, seq_id) pairs from PosCtx.
    encoded_pos, _ = (all_qi * stride + all_seqids).sort()   # [M] sorted

    # Candidate set: convert ANN slot indices → actual seq_ids, then encode.
    qi_range   = torch.arange(Q, dtype=torch.int64, device=device)
    if slot_to_id_d is not None:
        nn_seq_ids = slot_to_id_d[(nn_idx + 1).long()]       # [Q, K] slot → seq_id
    else:
        nn_seq_ids = nn_idx + 1                              # [Q, K] slot == seq_id
    encoded_cands  = (qi_range[:, None] * stride + nn_seq_ids).reshape(-1)  # [Q*K]

    # Binary search: is each candidate in the positive set?
    idxs   = torch.searchsorted(encoded_pos, encoded_cands)
    idxs   = idxs.clamp(0, M - 1)
    is_neg = (encoded_pos[idxs] != encoded_cands).reshape(Q, K)          # [Q, K]

    # Select the first n_neg negatives per row (already sorted by desc similarity).
    cum_neg  = is_neg.long().cumsum(dim=1)              # [Q, K]
    selected = is_neg & (cum_neg <= n_neg)              # [Q, K]

    n_found = selected.sum(dim=1)                       # [Q] int64 on device
    timing["filter"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # 6. Bulk write to neg_ctx                                            #
    # ------------------------------------------------------------------ #

    t0 = time.perf_counter()

    n_found_cpu = n_found.cpu()
    stats.fill_counts.extend(n_found_cpu.tolist())
    stats.n_latents_zero_negatives += int((n_found_cpu == 0).sum().item())
    stats.n_latents_populated      += int((n_found_cpu >  0).sum().item())

    # Build full [d_sae, n_neg] output tensors on the compute device, then
    # transfer as a single contiguous block to CPU — avoids scattered CPU writes.
    full_ids  = torch.zeros(d_sae, n_neg, dtype=torch.int32,   device=device)
    full_sims = torch.zeros(d_sae, n_neg, dtype=torch.float32, device=device)

    # Fast path: all rows filled to exactly n_neg (virtually always true when K >> n_neg).
    if bool((n_found_cpu == n_neg).all().item()):
        q_ids  = nn_seq_ids[selected].reshape(Q, n_neg).to(torch.int32)   # [Q, n_neg]
        q_sims = nn_sims[selected].reshape(Q, n_neg)                       # [Q, n_neg]
    else:
        # Variable-fill path: some rows have fewer than n_neg negatives.
        q_ids  = torch.zeros(Q, n_neg, dtype=torch.int32,   device=device)
        q_sims = torch.zeros(Q, n_neg, dtype=torch.float32, device=device)
        for qi in range(Q):
            nf = int(n_found_cpu[qi].item())
            if nf == 0:
                continue
            sel_pos = selected[qi].nonzero(as_tuple=True)[0][:n_neg]
            q_ids[qi,  :nf] = nn_seq_ids[qi, sel_pos].to(torch.int32)
            q_sims[qi, :nf] = nn_sims[qi, sel_pos]

    # Scatter Q active-latent rows into the full d_sae output (GPU-side).
    full_ids[active_js]  = q_ids
    full_sims[active_js] = q_sims

    # Single contiguous GPU→CPU transfer per component (~10 MB each).
    neg_ctx.ctx_seq_idx[comp_idx] = full_ids.cpu()
    neg_ctx.ctx_seq_val[comp_idx] = full_sims.cpu()

    timing["write"] = time.perf_counter() - t0
    return timing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_neg_ctx(
    seq_repr: "SeqRepr",
    top_ctx:  "Context",
    mid_ctx:  "Context",
    neg_ctx:  "Context",
) -> NegCtxStats:
    """
    Populate neg_ctx for all latents with sufficient PosCtx data.

    Each of the 36 SAE components is processed fully vectorised:
    no Python loops over the ~40K active latents per component.

    Returns a NegCtxStats instance with fill-rate distribution and timing.
    """
    llm_cfg = TuringLLMConfig()
    n_comp      = llm_cfg.n_layer * 3
    n_neg       = neg_ctx.num_ctx_sequences
    n_neighbors = cast(int, config.latents.neg_ctx.n_neighbors or 500)
    min_pos_ctx = cast(int, config.latents.neg_ctx.min_pos_ctx  or 8)

    stats   = NegCtxStats()
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Build ANN index                                                      #
    # ------------------------------------------------------------------ #

    t0           = time.perf_counter()
    total_n_seqs = seq_repr.n_seqs     # full dataset count (for stride + valid filter)
    n_stored     = seq_repr.n_stored   # actual ANN index size (≤ total_n_seqs)
    device       = _ann_device()

    raw_vecs = seq_repr.repr_buf[1:n_stored + 1].float()   # [n_stored, D] float32, CPU
    index    = TorchANNIndex(raw_vecs, device=device)
    stats.t_index_build = time.perf_counter() - t0

    K = min(n_neighbors, n_stored)

    # Move slot-mapping tensors to compute device once (reused across components).
    # None when uncapped (slot == seq_id, no mapping needed).
    if seq_repr.is_capped and seq_repr.slot_to_id is not None and seq_repr.id_to_slot is not None:
        slot_to_id_d: torch.Tensor | None = seq_repr.slot_to_id.to(device)
        id_to_slot_d:  torch.Tensor | None = seq_repr.id_to_slot.to(device)
    else:
        slot_to_id_d = None
        id_to_slot_d = None

    # ------------------------------------------------------------------ #
    # Per-component loop — tqdm for progress; timing breakdown in postfix #
    # ------------------------------------------------------------------ #

    pbar = tqdm(range(n_comp), desc="  [neg_ctx]", unit="comp", leave=True)
    for comp_idx in pbar:
        tc0 = time.perf_counter()

        timing = _process_component(
            comp_idx, top_ctx, mid_ctx, neg_ctx,
            index, K, n_neg, min_pos_ctx, stats,
            total_n_seqs, slot_to_id_d, id_to_slot_d,
        )

        comp_s = time.perf_counter() - tc0
        stats.t_pos_collect += timing.get("pos",    0.0)
        stats.t_qmat_build  += timing.get("qmat",   0.0)
        stats.t_query       += timing.get("query",  0.0)
        stats.t_filter      += timing.get("filter", 0.0)
        stats.t_write       += timing.get("write",  0.0)

        active = stats.n_latents_attempted - stats.n_latents_skipped_low_pos
        pbar.set_postfix({
            "active": f"{active // (comp_idx + 1):,}",
            "ms":     f"{comp_s * 1000:.0f}",
            "pos_s":  f"{stats.t_pos_collect:.1f}",
            "qmat_s": f"{stats.t_qmat_build:.1f}",
            "srch_s": f"{stats.t_query:.1f}",
            "flt_s":  f"{stats.t_filter:.1f}",
        })

    pbar.close()
    stats.t_total = time.perf_counter() - t_start
    return stats
