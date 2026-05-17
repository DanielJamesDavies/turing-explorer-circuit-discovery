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


def reshape_reduce(top_ids, top_vals, num_components, d_sae, K):
    return top_ids.reshape(num_components, d_sae, K), top_vals.reshape(num_components, d_sae, K)


def python_reduce_topk(candidate_ids, candidate_vals, seq_offsets, seq_targets, sid_to_row, num_components, d_sae, K):
    """Small deterministic reference implementation for reducer unit tests."""
    n_targets = num_components * d_sae
    top_ids = torch.zeros((num_components, d_sae, K), dtype=torch.int32)
    top_vals = torch.zeros((num_components, d_sae, K), dtype=torch.float32)

    for g in range(n_targets):
        scores = {}
        for sid in range(1, seq_offsets.numel()):
            start = int(seq_offsets[sid - 1])
            end = int(seq_offsets[sid])
            if not (seq_targets[start:end] == g).any():
                continue
            row = int(sid_to_row[sid])
            if row < 0:
                continue
            for cand_id, cand_val in zip(candidate_ids[row].tolist(), candidate_vals[row].tolist()):
                if cand_id == g or cand_val <= 0.0:
                    continue
                scores[cand_id] = scores.get(cand_id, 0.0) + float(cand_val)

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:K]
        c = g // d_sae
        l = g % d_sae
        for i, (cand_id, cand_val) in enumerate(ordered):
            top_ids[c, l, i] = cand_id
            top_vals[c, l, i] = cand_val

    return top_ids, top_vals


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
    top_ids, top_vals = reshape_reduce(top_ids, top_vals, num_components, d_sae, K)

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
    top_ids, top_vals = reshape_reduce(top_ids, top_vals, num_components, d_sae, K)

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
    top_ids, top_vals = reshape_reduce(top_ids, top_vals, num_components, d_sae, K)

    assert top_ids[0, 0, 0].item() == 1, f"Expected id=1, got {top_ids[0, 0, 0].item()}"
    assert abs(top_vals[0, 0, 0].item() - 7.0) < 1e-5, f"Expected summed value 7.0, got {top_vals[0, 0, 0].item()}"

    print("PASS: test_sum_aggregation")


def test_reference_equivalence_small():
    """Compare native reducer against a Python reference on a small deterministic case."""
    num_components = 2
    d_sae = 5
    K = 3
    M = 4
    S = 6
    n_targets = num_components * d_sae

    candidate_ids = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [0, 2, 5, 6],
        [1, 7, 8, 9],
        [3, 4, 8, 9],
        [0, 1, 5, 7],
    ], dtype=torch.int32)
    candidate_vals = torch.tensor([
        [1.0, 2.0, 0.0, 4.0],
        [2.0, 1.0, 3.0, 4.0],
        [5.0, 1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5, 4.5],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 1.5, 2.5, 3.5],
    ], dtype=torch.float32)

    # sid 1 -> targets 0, 1; sid 2 -> targets 0, 5; etc.
    per_sid_targets = {
        1: [0, 1],
        2: [0, 5],
        3: [2, 5, 8],
        4: [3],
        5: [4, 8],
        6: [9],
    }
    seq_targets_list = []
    offsets = [0]
    for sid in range(1, S + 1):
        seq_targets_list.extend(per_sid_targets[sid])
        offsets.append(len(seq_targets_list))

    seq_offsets = torch.tensor(offsets, dtype=torch.int64)
    seq_targets = torch.tensor(seq_targets_list, dtype=torch.int64)
    sid_to_row = torch.tensor([-1, 0, 1, 2, 3, 4, 5], dtype=torch.int64)

    native_ids, native_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids,
        candidate_vals,
        seq_offsets,
        seq_targets,
        sid_to_row,
        num_components,
        d_sae,
        K,
        print_timings=False,
    )
    native_ids, native_vals = reshape_reduce(native_ids, native_vals, num_components, d_sae, K)
    ref_ids, ref_vals = python_reduce_topk(
        candidate_ids,
        candidate_vals,
        seq_offsets,
        seq_targets,
        sid_to_row,
        num_components,
        d_sae,
        K,
    )

    assert native_vals.shape == ref_vals.shape
    for g in range(n_targets):
        c = g // d_sae
        l = g % d_sae
        native_pairs = [(int(i), round(float(v), 5)) for i, v in zip(native_ids[c, l], native_vals[c, l]) if v > 0]
        ref_pairs = [(int(i), round(float(v), 5)) for i, v in zip(ref_ids[c, l], ref_vals[c, l]) if v > 0]
        ref_scores = {}
        for sid in range(1, seq_offsets.numel()):
            start = int(seq_offsets[sid - 1])
            end = int(seq_offsets[sid])
            if not (seq_targets[start:end] == g).any():
                continue
            row = int(sid_to_row[sid])
            if row < 0:
                continue
            for cand_id, cand_val in zip(candidate_ids[row].tolist(), candidate_vals[row].tolist()):
                if cand_id == g or cand_val <= 0.0:
                    continue
                ref_scores[cand_id] = round(ref_scores.get(cand_id, 0.0) + float(cand_val), 5)

        assert [v for _, v in native_pairs] == [v for _, v in ref_pairs], (
            f"target {g}: native values={native_pairs}, ref values={ref_pairs}"
        )
        for cand_id, cand_val in native_pairs:
            assert ref_scores.get(cand_id) == cand_val, (
                f"target {g}: native candidate {(cand_id, cand_val)} not in reference scores {ref_scores}"
            )

    print("PASS: test_reference_equivalence_small")


def test_target_range_equivalence_small():
    """Reducing target ranges and stitching them must equal a full reduction."""
    num_components = 2
    d_sae = 5
    K = 3
    candidate_ids = torch.tensor([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [0, 2, 5, 6],
        [1, 7, 8, 9],
        [3, 4, 8, 9],
        [0, 1, 5, 7],
    ], dtype=torch.int32)
    candidate_vals = torch.tensor([
        [1.0, 2.0, 0.0, 4.0],
        [2.0, 1.0, 3.0, 4.0],
        [5.0, 1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5, 4.5],
        [4.0, 3.0, 2.0, 1.0],
        [0.5, 1.5, 2.5, 3.5],
    ], dtype=torch.float32)
    seq_offsets = torch.tensor([0, 2, 4, 7, 8, 10, 11], dtype=torch.int64)
    seq_targets = torch.tensor([0, 1, 0, 5, 2, 5, 8, 3, 4, 8, 9], dtype=torch.int64)
    sid_to_row = torch.tensor([-1, 0, 1, 2, 3, 4, 5], dtype=torch.int64)
    n_targets = num_components * d_sae

    full_ids, full_vals = top_coactivation_reduce.reduce_topk(
        candidate_ids,
        candidate_vals,
        seq_offsets,
        seq_targets,
        sid_to_row,
        num_components,
        d_sae,
        K,
        print_timings=False,
    )

    stitched_ids = torch.empty_like(full_ids)
    stitched_vals = torch.empty_like(full_vals)
    ranges = [(0, 3), (3, 7), (7, n_targets)]
    for start, end in ranges:
        part_ids, part_vals = top_coactivation_reduce.reduce_topk(
            candidate_ids,
            candidate_vals,
            seq_offsets,
            seq_targets,
            sid_to_row,
            num_components,
            d_sae,
            K,
            print_timings=False,
            target_start=start,
            target_end=end,
        )
        assert part_ids.shape == (end - start, K)
        stitched_ids[start:end] = part_ids
        stitched_vals[start:end] = part_vals

    assert torch.equal(stitched_ids, full_ids)
    assert torch.allclose(stitched_vals, full_vals)
    print("PASS: test_target_range_equivalence_small")


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
    top_ids, top_vals = reshape_reduce(top_ids, top_vals, num_components, d_sae, K)
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


def benchmark_full_scale():
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
    top_ids, top_vals = reshape_reduce(top_ids, top_vals, num_components, d_sae, K)
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
    test_reference_equivalence_small()
    test_target_range_equivalence_small()
    test_scaling()
    if os.environ.get("RUN_FULL_SCALE_REDUCE_BENCHMARK") == "1":
        benchmark_full_scale()
    print("\nAll tests passed!")
