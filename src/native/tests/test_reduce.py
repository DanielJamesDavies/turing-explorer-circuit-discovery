"""
Synthetic test for top_coactivation_reduce.reduce_topk

Tests:
  1. Basic correctness: known inputs → expected outputs
  2. Self-filtering: a target's own global ID must not appear in its results
  3. Sum aggregation: duplicate global IDs across sequences are summed
  4. Scaling test: realistic dimensions (smaller scale) with timing
"""

import os
import time
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import top_coactivation_reduce


def test_basic_correctness():
    """
    Tiny example:
      2 components, d_sae=4, K=2, M=3
      3 sequences (rows 0, 1, 2), with sequence IDs 1, 2, 3

      Target global_id=0 (comp=0, lat=0) appears in sequences 1 and 2
      Target global_id=5 (comp=1, lat=1) appears in sequence 3
    """
    num_components = 2
    d_sae = 4
    K = 2
    M = 3

    candidate_ids = torch.tensor([
        [1, 2, 3],   # seq row 0 (sid=1)
        [1, 3, 5],   # seq row 1 (sid=2)
        [0, 2, 7],   # seq row 2 (sid=3)
    ], dtype=torch.int32)

    candidate_vals = torch.tensor([
        [1.0, 2.0, 0.5],  # row 0
        [3.0, 1.5, 0.1],  # row 1
        [4.0, 1.0, 2.0],  # row 2
    ], dtype=torch.float32)

    # CSR: seq_offsets built from bincount of sequence IDs
    # sid=1 → target 0, sid=2 → target 0, sid=3 → target 5
    # bincount([1,2,3]) = [0, 1, 1, 1]
    # cumsum = [0, 1, 2, 3]
    seq_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    seq_targets = torch.tensor([0, 0, 5], dtype=torch.int64)

    sid_to_row = torch.tensor([-1, 0, 1, 2], dtype=torch.int64)  # sid 0 unused

    top_ids, top_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids, candidate_vals, seq_offsets, seq_targets,
        sid_to_row, num_components, d_sae, K
    )

    assert top_ids.shape == (num_components, d_sae, K)
    assert top_vals.shape == (num_components, d_sae, K)

    # Target 0 (comp=0, lat=0): sees rows 0 and 1
    # Pairs (after filtering self_id=0):
    #   row 0: (1, 1.0), (2, 2.0), (3, 0.5)
    #   row 1: (1, 3.0), (3, 1.5), (5, 0.1)
    # After dedup+sum: id=1 → 4.0, id=2 → 2.0, id=3 → 2.0, id=5 → 0.1
    # Top-2 by value: (1, 4.0), (2, 2.0) or (3, 2.0) — tie broken by nth_element
    t0_ids  = top_ids[0, 0].tolist()
    t0_vals = top_vals[0, 0].tolist()
    assert t0_ids[0] == 1 and abs(t0_vals[0] - 4.0) < 1e-5, f"Expected (1, 4.0), got ({t0_ids[0]}, {t0_vals[0]})"
    assert t0_vals[1] == 2.0, f"Expected second value 2.0, got {t0_vals[1]}"
    assert t0_ids[1] in (2, 3), f"Expected second id to be 2 or 3, got {t0_ids[1]}"

    # Target 5 (comp=1, lat=1): sees row 2
    # Pairs (after filtering self_id=5): (0, 4.0), (2, 1.0), (7, 2.0)
    # Top-2: (0, 4.0), (7, 2.0)
    t5_ids  = top_ids[1, 1].tolist()
    t5_vals = top_vals[1, 1].tolist()
    assert t5_ids[0] == 0 and abs(t5_vals[0] - 4.0) < 1e-5, f"Expected (0, 4.0), got ({t5_ids[0]}, {t5_vals[0]})"
    assert t5_ids[1] == 7 and abs(t5_vals[1] - 2.0) < 1e-5, f"Expected (7, 2.0), got ({t5_ids[1]}, {t5_vals[1]})"

    # Targets with no sequences should be all zeros
    assert top_ids[0, 1].sum() == 0
    assert top_vals[0, 1].sum() == 0.0

    print("PASS: test_basic_correctness")


def test_self_filtering():
    """Ensure a target's own global ID is excluded from results."""
    num_components = 1
    d_sae = 4
    K = 2
    M = 4

    # Target 2 (comp=0, lat=2) — self_id = 2
    # Sequence has candidate 2 with the highest value — it must be filtered
    candidate_ids = torch.tensor([[2, 0, 1, 3]], dtype=torch.int32)
    candidate_vals = torch.tensor([[99.0, 1.0, 2.0, 3.0]], dtype=torch.float32)

    seq_offsets = torch.tensor([0, 1], dtype=torch.int64)
    seq_targets = torch.tensor([2], dtype=torch.int64)
    sid_to_row = torch.tensor([-1, 0], dtype=torch.int64)

    top_ids, top_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids, candidate_vals, seq_offsets, seq_targets,
        sid_to_row, num_components, d_sae, K
    )

    t2_ids = top_ids[0, 2].tolist()
    assert 2 not in t2_ids, f"Self-ID 2 should be filtered, got {t2_ids}"
    assert t2_ids[0] == 3 and abs(top_vals[0, 2, 0].item() - 3.0) < 1e-5
    assert t2_ids[1] == 1 and abs(top_vals[0, 2, 1].item() - 2.0) < 1e-5

    print("PASS: test_self_filtering")


def test_sum_aggregation():
    """Verify that duplicate IDs across sequences are summed, not maxed."""
    num_components = 1
    d_sae = 4
    K = 1
    M = 2

    # Target 0: appears in sequences 1 and 2
    # Both sequences have candidate id=1 with values 3.0 and 4.0
    # Sum should be 7.0
    candidate_ids = torch.tensor([
        [1, 2],
        [1, 3],
    ], dtype=torch.int32)
    candidate_vals = torch.tensor([
        [3.0, 1.0],
        [4.0, 2.0],
    ], dtype=torch.float32)

    seq_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    seq_targets = torch.tensor([0, 0], dtype=torch.int64)
    sid_to_row = torch.tensor([-1, 0, 1], dtype=torch.int64)

    top_ids, top_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids, candidate_vals, seq_offsets, seq_targets,
        sid_to_row, num_components, d_sae, K
    )

    assert top_ids[0, 0, 0].item() == 1, f"Expected id=1, got {top_ids[0, 0, 0].item()}"
    assert abs(top_vals[0, 0, 0].item() - 7.0) < 1e-5, f"Expected summed value 7.0, got {top_vals[0, 0, 0].item()}"

    print("PASS: test_sum_aggregation")


def test_scaling():
    """
    Realistic-ish dimensions (1/10th scale) with timing.
    4 components, d_sae=4096, K=32, M=128, S=5000 sequences
    ~64 sequences per target → ~16K targets with data
    """
    num_components = 4
    d_sae = 4096
    K = 32
    M = 128
    S = 5000
    n_targets = num_components * d_sae  # 16384
    seqs_per_target = 8

    torch.manual_seed(42)
    candidate_ids = torch.randint(0, n_targets, (S, M), dtype=torch.int32)
    candidate_vals = torch.rand(S, M, dtype=torch.float32)

    # Build a synthetic CSR: each target gets ~seqs_per_target random sequences
    # Total entries ≈ n_targets * seqs_per_target = 131072
    target_list = []
    sid_list = []
    for g in range(n_targets):
        sids = torch.randint(1, S + 1, (seqs_per_target,))
        target_list.extend([g] * seqs_per_target)
        sid_list.extend(sids.tolist())

    sid_tensor = torch.tensor(sid_list, dtype=torch.int64)
    target_tensor = torch.tensor(target_list, dtype=torch.int64)

    # Sort by sid to build CSR
    order = sid_tensor.argsort()
    sid_sorted = sid_tensor[order]
    target_sorted = target_tensor[order]

    max_sid = S
    counts = torch.bincount(sid_sorted.int(), minlength=max_sid + 1).long()
    seq_offsets = torch.cumsum(counts, dim=0)

    sid_to_row = torch.arange(0, max_sid + 1, dtype=torch.int64)
    sid_to_row[0] = -1  # sid=0 unused

    print(f"\nScaling test: {num_components} comps, d_sae={d_sae}, S={S}, M={M}, K={K}")
    print(f"  Total targets: {n_targets}")
    print(f"  Total CSR entries: {len(target_sorted)}")

    t0 = time.perf_counter()
    top_ids, top_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids, candidate_vals, seq_offsets, target_sorted,
        sid_to_row, num_components, d_sae, K
    )
    elapsed = time.perf_counter() - t0

    assert top_ids.shape == (num_components, d_sae, K)
    assert top_vals.shape == (num_components, d_sae, K)

    n_nonempty = (top_vals.sum(dim=-1) > 0).sum().item()
    print(f"  Non-empty targets: {n_nonempty}/{n_targets}")
    print(f"  Time: {elapsed:.3f}s")

    # Verify outputs are sorted descending per target
    for _ in range(100):
        c = torch.randint(0, num_components, (1,)).item()
        l = torch.randint(0, d_sae, (1,)).item()
        vals = top_vals[c, l].tolist()
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1], f"Not sorted at ({c},{l}): {vals}"

    print("PASS: test_scaling")


def test_full_scale():
    """
    Full-scale dimensions: 36 components, d_sae=40960, K=32, M=128, S=50000
    With ~64 sequences per target → 94.4M CSR entries.
    This tests real-world memory and timing.
    """
    num_components = 36
    d_sae = 40960
    K = 32
    M = 128
    S = 50000
    n_targets = num_components * d_sae  # 1,474,560
    seqs_per_target = 64

    print(f"\nFull-scale test: {num_components} comps, d_sae={d_sae}, S={S}, M={M}, K={K}")
    print(f"  Total targets: {n_targets:,}")
    total_csr = n_targets * seqs_per_target
    print(f"  Total CSR entries: {total_csr:,}")
    print(f"  Candidate data: {S * M * 8 / 1e6:.1f} MB")
    print(f"  CSR data: {total_csr * 8 / 1e6:.1f} MB")
    print(f"  Output data: {n_targets * K * 8 / 1e6:.1f} MB")

    torch.manual_seed(123)
    print("  Allocating candidate data...")
    candidate_ids = torch.randint(0, n_targets, (S, M), dtype=torch.int32)
    candidate_vals = torch.rand(S, M, dtype=torch.float32)

    print("  Building synthetic CSR...")
    # Build CSR more efficiently: random sids per target, then sort
    sids_all = torch.randint(1, S + 1, (n_targets, seqs_per_target), dtype=torch.int64)
    targets_all = torch.arange(n_targets, dtype=torch.int64).unsqueeze(1).expand(-1, seqs_per_target)

    sid_flat = sids_all.reshape(-1)
    target_flat = targets_all.reshape(-1)

    order = sid_flat.argsort()
    sid_sorted = sid_flat[order]
    target_sorted = target_flat[order]

    max_sid = S
    counts = torch.bincount(sid_sorted.int(), minlength=max_sid + 1).long()
    seq_offsets = torch.cumsum(counts, dim=0)

    sid_to_row = torch.arange(0, max_sid + 1, dtype=torch.int64)
    sid_to_row[0] = -1

    print("  Running reduce_topk...")
    t0 = time.perf_counter()
    top_ids, top_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids, candidate_vals, seq_offsets, target_sorted,
        sid_to_row, num_components, d_sae, K
    )
    elapsed = time.perf_counter() - t0

    n_nonempty = (top_vals.sum(dim=-1) > 0).sum().item()
    print(f"  Non-empty targets: {n_nonempty:,}/{n_targets:,}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Throughput: {n_targets / elapsed:,.0f} targets/sec")

    print("PASS: test_full_scale")


if __name__ == "__main__":
    test_basic_correctness()
    test_self_filtering()
    test_sum_aggregation()
    test_scaling()
    test_full_scale()
    print("\nAll tests passed!")
