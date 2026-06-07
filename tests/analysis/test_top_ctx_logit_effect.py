import torch

from analysis.coactivation.top_ctx_logit_effect import (
    _top_ctx_sequence_ids,
    compute_logit_effect_metrics,
    sample_top_ctx_latents,
)
from sae.dense import sparse_topk_to_dense


def test_compute_logit_effect_metrics_is_zero_for_identical_logits():
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 3.0, -2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    metrics = compute_logit_effect_metrics(logits, logits.clone(), targets)

    assert abs(metrics["kl_baseline_to_ablated"]) < 1e-8
    assert abs(metrics["js_divergence"]) < 1e-8
    assert metrics["top1_changed_pct"] == 0.0
    assert abs(metrics["baseline_top_prob_delta"]) < 1e-8
    assert abs(metrics["entropy_delta"]) < 1e-8
    assert abs(metrics["ground_truth_logprob_delta"]) < 1e-8
    assert metrics["max_abs_logit_delta"] == 0.0
    assert metrics["mean_abs_logit_delta"] == 0.0
    assert metrics["logit_l2_delta"] == 0.0


def test_compute_logit_effect_metrics_detects_distribution_change():
    baseline = torch.tensor([[4.0, 0.0, -1.0], [0.0, 4.0, -1.0]], dtype=torch.float32)
    ablated = torch.tensor([[0.0, 4.0, -1.0], [0.0, 4.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    metrics = compute_logit_effect_metrics(baseline, ablated, targets)

    assert metrics["kl_baseline_to_ablated"] > 0.0
    assert metrics["js_divergence"] > 0.0
    assert metrics["top1_changed_pct"] == 50.0
    assert metrics["baseline_top_prob_delta"] > 0.0
    assert metrics["ground_truth_logprob_delta"] > 0.0
    assert metrics["max_abs_logit_delta"] == 4.0
    assert metrics["mean_abs_logit_delta"] > 0.0
    assert metrics["logit_l2_delta"] > 0.0


def test_sample_top_ctx_latents_samples_nonempty_rows_deterministically():
    top_ctx_indices = torch.zeros((2, 5, 3), dtype=torch.int64)
    top_ctx_indices[0, 1, 0] = 10
    top_ctx_indices[0, 3, 0] = 11
    top_ctx_indices[1, 0, 0] = 12
    top_ctx_indices[1, 4, 0] = 13

    samples = sample_top_ctx_latents(top_ctx_indices, sample_size=4, d_sae=5)

    assert samples.tolist() == [1, 3, 5, 9]


def test_sample_top_ctx_latents_trims_to_requested_size():
    top_ctx_indices = torch.ones((3, 4, 2), dtype=torch.int64)

    samples = sample_top_ctx_latents(top_ctx_indices, sample_size=5, d_sae=4)

    assert samples.numel() == 5
    assert samples.tolist() == sample_top_ctx_latents(top_ctx_indices, sample_size=5, d_sae=4).tolist()
    assert all(0 <= int(sample) < 12 for sample in samples.tolist())


def test_top_ctx_sequence_ids_uses_positive_unique_candidates_in_order():
    top_ctx_indices = torch.zeros((2, 3, 6), dtype=torch.int64)
    top_ctx_indices[1, 2] = torch.tensor([0, 7, 7, 3, 9, 11], dtype=torch.int64)

    ids = _top_ctx_sequence_ids(top_ctx_indices, comp_idx=1, latent_idx=2, limit=3)

    assert ids == [7, 3, 9]


def test_topk_to_dense_latents_keeps_active_value_for_duplicate_indices():
    top_acts = torch.tensor([[[0.5, 0.0, 0.0]]], dtype=torch.float32)
    top_indices = torch.tensor([[[0, 0, 0]]], dtype=torch.int64)

    dense = sparse_topk_to_dense(top_acts, top_indices, d_sae=4)

    assert dense[0, 0, 0].item() == 0.5
