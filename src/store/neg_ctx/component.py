"""Per-component negative-context processing."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .ann import TorchANNIndex
from .sharded_ann import ShardedANNIndex
from .stats import NegCtxStats

if TYPE_CHECKING:
    from store.context import Context
    from store.seq_repr import SeqRepr


# Max pairs processed in one scatter-mean chunk (bounds peak GPU tensor size).
# 65536 pairs x 1024 x 4 B = 256 MB - safe even on 8 GB VRAM.
_PAIR_CHUNK = 65536


@torch.no_grad()
def _process_component(
    comp_idx: int,
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
    index: TorchANNIndex,
    K: int,
    n_neg: int,
    min_pos_ctx: int,
    stats: NegCtxStats,
    total_n_seqs: int,
    slot_to_id_d: "torch.Tensor | None",
    id_to_slot_d: "torch.Tensor | None",
) -> dict:
    """
    Process one SAE component end-to-end without any Python loop over latents.

    slot_to_id_d / id_to_slot_d are None when seq_repr is uncapped (slot == seq_id).

    Returns a dict of per-step timing (seconds) for the tqdm postfix.
    """
    device = index.device
    d_sae = top_ctx.ctx_seq_idx.shape[1]
    timing: dict[str, float] = {}

    top_ids_d = top_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_top]
    top_mask_d = top_ctx.ctx_seq_val[comp_idx].to(device).float() > 0        # [d_sae, N_top]
    mid_ids_d = mid_ctx.ctx_seq_idx[comp_idx].to(device, dtype=torch.int64)  # [d_sae, N_mid]
    mid_mask_d = mid_ctx.ctx_seq_val[comp_idx].to(device).float() > 0        # [d_sae, N_mid]

    t0 = time.perf_counter()

    pos_counts = top_mask_d.sum(dim=1) + mid_mask_d.sum(dim=1)   # [d_sae]
    active_js = (pos_counts >= min_pos_ctx).nonzero(as_tuple=True)[0]  # [Q]
    Q = active_js.shape[0]

    stats.n_latents_attempted += d_sae
    stats.n_latents_skipped_low_pos += d_sae - Q

    if Q == 0:
        timing["pos"] = time.perf_counter() - t0
        return timing

    # Gather (qi_within_Q, seq_id) pairs for all active latents in one pass.
    at_ids = top_ids_d[active_js]    # [Q, N_top]
    at_mask = top_mask_d[active_js]  # [Q, N_top]
    am_ids = mid_ids_d[active_js]    # [Q, N_mid]
    am_mask = mid_mask_d[active_js]  # [Q, N_mid]

    t_qi, t_ki = at_mask.nonzero(as_tuple=True)  # [M_top]
    m_qi, m_ki = am_mask.nonzero(as_tuple=True)  # [M_mid]

    all_qi = torch.cat([t_qi, m_qi])
    all_seqids = torch.cat([at_ids[t_qi, t_ki], am_ids[m_qi, m_ki]])  # 1-indexed

    # Remove sentinel 0 only; range check is handled by the slot filter below.
    valid = (all_seqids > 0) & (all_seqids <= total_n_seqs)
    all_qi = all_qi[valid]
    all_seqids = all_seqids[valid]
    M = all_qi.shape[0]

    timing["pos"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = index.dim
    sums = torch.zeros(Q, D, dtype=torch.float32, device=device)

    if id_to_slot_d is not None:
        # Capped: only sequences stored in the ANN index contribute to the centroid.
        slots = id_to_slot_d[all_seqids]      # [M] slot (0 = not stored)
        in_index = slots > 0
        qi_qmat = all_qi[in_index]
        rows_all = slots[in_index] - 1        # 0-indexed slot row
    else:
        qi_qmat = all_qi
        rows_all = all_seqids - 1             # 0-indexed seq_id row (slot == seq_id)

    Mq = qi_qmat.shape[0]
    cnt = torch.bincount(qi_qmat, minlength=Q).float().unsqueeze(1)  # [Q, 1]

    for pair_start in range(0, Mq, _PAIR_CHUNK):
        pair_end = min(pair_start + _PAIR_CHUNK, Mq)
        chunk_rows = rows_all[pair_start:pair_end]
        chunk_qi = qi_qmat[pair_start:pair_end]
        chunk_reps = index.index[chunk_rows]  # [C, D] float32
        sums.index_add_(0, chunk_qi, chunk_reps)

    qmat = F.normalize(sums / cnt.clamp(min=1e-8), dim=1)  # [Q, D]
    timing["qmat"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_sims, nn_idx = index.search(qmat, K)  # [Q, K] on device
    timing["query"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    # stride must be > max possible seq_id to avoid encoding collisions.
    stride = total_n_seqs + 2
    # Positive set: encode all (qi, seq_id) pairs from PosCtx.
    encoded_pos, _ = (all_qi * stride + all_seqids).sort()  # [M] sorted

    # Candidate set: convert ANN slot indices to actual seq_ids, then encode.
    qi_range = torch.arange(Q, dtype=torch.int64, device=device)
    if slot_to_id_d is not None:
        nn_seq_ids = slot_to_id_d[(nn_idx + 1).long()]  # [Q, K] slot -> seq_id
    else:
        nn_seq_ids = nn_idx + 1                         # [Q, K] slot == seq_id
    encoded_cands = (qi_range[:, None] * stride + nn_seq_ids).reshape(-1)  # [Q*K]

    # Binary search: is each candidate in the positive set?
    idxs = torch.searchsorted(encoded_pos, encoded_cands)
    idxs = idxs.clamp(0, M - 1)
    is_neg = (encoded_pos[idxs] != encoded_cands).reshape(Q, K)  # [Q, K]

    # Select the first n_neg negatives per row (already sorted by desc similarity).
    cum_neg = is_neg.long().cumsum(dim=1)  # [Q, K]
    selected = is_neg & (cum_neg <= n_neg)  # [Q, K]

    n_found = selected.sum(dim=1)  # [Q] int64 on device
    timing["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()

    n_found_cpu = n_found.cpu()
    stats.fill_counts.extend(n_found_cpu.tolist())
    stats.n_latents_zero_negatives += int((n_found_cpu == 0).sum().item())
    stats.n_latents_populated += int((n_found_cpu > 0).sum().item())

    # Build selected active rows on the compute device, then transfer only
    # those rows back to the CPU store.
    # Fast path: all rows filled to exactly n_neg (virtually always true when K >> n_neg).
    if bool((n_found_cpu == n_neg).all().item()):
        q_ids = nn_seq_ids[selected].reshape(Q, n_neg).to(torch.int32)  # [Q, n_neg]
        q_sims = nn_sims[selected].reshape(Q, n_neg)                    # [Q, n_neg]
    else:
        # Variable-fill path: some rows have fewer than n_neg negatives.
        q_ids = torch.zeros(Q, n_neg, dtype=torch.int32, device=device)
        q_sims = torch.zeros(Q, n_neg, dtype=torch.float32, device=device)
        for qi in range(Q):
            nf = int(n_found_cpu[qi].item())
            if nf == 0:
                continue
            sel_pos = selected[qi].nonzero(as_tuple=True)[0][:n_neg]
            q_ids[qi, :nf] = nn_seq_ids[qi, sel_pos].to(torch.int32)
            q_sims[qi, :nf] = nn_sims[qi, sel_pos]

    comp_ids = neg_ctx.ctx_seq_idx[comp_idx]
    comp_vals = neg_ctx.ctx_seq_val[comp_idx]
    comp_ids.zero_()
    comp_vals.zero_()

    active_js_cpu = active_js.cpu()
    comp_ids[active_js_cpu] = q_ids.cpu()
    comp_vals[active_js_cpu] = q_sims.cpu()

    timing["write"] = time.perf_counter() - t0
    return timing


@torch.no_grad()
def _process_component_sharded(
    comp_idx: int,
    top_ctx: "Context",
    mid_ctx: "Context",
    neg_ctx: "Context",
    seq_repr: "SeqRepr",
    index: ShardedANNIndex,
    K: int,
    n_neg: int,
    min_pos_ctx: int,
    stats: NegCtxStats,
    total_n_seqs: int,
    query_device: torch.device,
    slot_to_id_d: "torch.Tensor | None",
    id_to_slot_d: "torch.Tensor | None",
) -> dict:
    """Process one component using globally merged results from sharded indexes."""

    d_sae = top_ctx.ctx_seq_idx.shape[1]
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    top_ids_d = top_ctx.ctx_seq_idx[comp_idx].to(query_device, dtype=torch.int64)
    top_mask_d = top_ctx.ctx_seq_val[comp_idx].to(query_device).float() > 0
    mid_ids_d = mid_ctx.ctx_seq_idx[comp_idx].to(query_device, dtype=torch.int64)
    mid_mask_d = mid_ctx.ctx_seq_val[comp_idx].to(query_device).float() > 0

    pos_counts = top_mask_d.sum(dim=1) + mid_mask_d.sum(dim=1)
    active_js = (pos_counts >= min_pos_ctx).nonzero(as_tuple=True)[0]
    Q = active_js.shape[0]

    stats.n_latents_attempted += d_sae
    stats.n_latents_skipped_low_pos += d_sae - Q

    if Q == 0:
        timing["pos"] = time.perf_counter() - t0
        return timing

    at_ids = top_ids_d[active_js]
    at_mask = top_mask_d[active_js]
    am_ids = mid_ids_d[active_js]
    am_mask = mid_mask_d[active_js]

    t_qi, t_ki = at_mask.nonzero(as_tuple=True)
    m_qi, m_ki = am_mask.nonzero(as_tuple=True)

    all_qi = torch.cat([t_qi, m_qi])
    all_seqids = torch.cat([at_ids[t_qi, t_ki], am_ids[m_qi, m_ki]])
    valid = (all_seqids > 0) & (all_seqids <= total_n_seqs)
    all_qi = all_qi[valid]
    all_seqids = all_seqids[valid]
    M = all_qi.shape[0]
    timing["pos"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    D = index.dim
    sums = torch.zeros(Q, D, dtype=torch.float32, device=query_device)

    if id_to_slot_d is not None:
        slots = id_to_slot_d[all_seqids]
        in_index = slots > 0
        qi_qmat = all_qi[in_index]
        rows_all = slots[in_index] - 1
    else:
        qi_qmat = all_qi
        rows_all = all_seqids - 1

    Mq = qi_qmat.shape[0]
    cnt = torch.bincount(qi_qmat, minlength=Q).float().unsqueeze(1).to(query_device)

    for pair_start in range(0, Mq, _PAIR_CHUNK):
        pair_end = min(pair_start + _PAIR_CHUNK, Mq)
        chunk_rows = rows_all[pair_start:pair_end].cpu() + 1
        chunk_qi = qi_qmat[pair_start:pair_end]
        chunk_reps = seq_repr.repr_buf[chunk_rows].float().to(query_device)
        chunk_reps = F.normalize(chunk_reps, dim=1)
        sums.index_add_(0, chunk_qi, chunk_reps)

    qmat = F.normalize(sums / cnt.clamp(min=1e-8), dim=1)
    timing["qmat"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    nn_sims, nn_idx = index.search(qmat, K, merge_device=query_device)
    timing["query"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stride = total_n_seqs + 2
    encoded_pos, _ = (all_qi * stride + all_seqids).sort()
    qi_range = torch.arange(Q, dtype=torch.int64, device=query_device)
    if slot_to_id_d is not None:
        nn_seq_ids = slot_to_id_d[(nn_idx + 1).long()]
    else:
        nn_seq_ids = nn_idx + 1
    encoded_cands = (qi_range[:, None] * stride + nn_seq_ids).reshape(-1)
    idxs = torch.searchsorted(encoded_pos, encoded_cands)
    idxs = idxs.clamp(0, M - 1)
    is_neg = (encoded_pos[idxs] != encoded_cands).reshape(Q, nn_seq_ids.shape[1])
    cum_neg = is_neg.long().cumsum(dim=1)
    selected = is_neg & (cum_neg <= n_neg)
    n_found = selected.sum(dim=1)
    timing["filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_found_cpu = n_found.cpu()
    stats.fill_counts.extend(n_found_cpu.tolist())
    stats.n_latents_zero_negatives += int((n_found_cpu == 0).sum().item())
    stats.n_latents_populated += int((n_found_cpu > 0).sum().item())

    if bool((n_found_cpu == n_neg).all().item()):
        q_ids = nn_seq_ids[selected].reshape(Q, n_neg).to(torch.int32)
        q_sims = nn_sims[selected].reshape(Q, n_neg)
    else:
        q_ids = torch.zeros(Q, n_neg, dtype=torch.int32, device=query_device)
        q_sims = torch.zeros(Q, n_neg, dtype=torch.float32, device=query_device)
        for qi in range(Q):
            nf = int(n_found_cpu[qi].item())
            if nf == 0:
                continue
            sel_pos = selected[qi].nonzero(as_tuple=True)[0][:n_neg]
            q_ids[qi, :nf] = nn_seq_ids[qi, sel_pos].to(torch.int32)
            q_sims[qi, :nf] = nn_sims[qi, sel_pos]

    comp_ids = neg_ctx.ctx_seq_idx[comp_idx]
    comp_vals = neg_ctx.ctx_seq_val[comp_idx]
    comp_ids.zero_()
    comp_vals.zero_()
    active_js_cpu = active_js.cpu()
    comp_ids[active_js_cpu] = q_ids.cpu()
    comp_vals[active_js_cpu] = q_sims.cpu()

    timing["write"] = time.perf_counter() - t0
    return timing


__all__ = ["_PAIR_CHUNK", "_process_component", "_process_component_sharded"]
