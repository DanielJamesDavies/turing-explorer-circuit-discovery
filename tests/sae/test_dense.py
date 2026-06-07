import torch

from sae.dense import sparse_topk_to_dense, target_latent_activations


def test_sparse_topk_to_dense_preserves_latent_zero_with_duplicate_padding():
    top_acts = torch.tensor([[[0.75, 0.0, 0.0]]], dtype=torch.float32)
    top_indices = torch.tensor([[[0, 0, 0]]], dtype=torch.int64)

    dense = sparse_topk_to_dense(top_acts, top_indices, d_sae=4)

    assert dense.tolist() == [[[0.75, 0.0, 0.0, 0.0]]]


def test_target_latent_activations_uses_max_over_duplicate_indices():
    top_acts = torch.tensor([[[0.25, 0.9, 0.0], [0.0, 0.4, 0.0]]], dtype=torch.float32)
    top_indices = torch.tensor([[[2, 2, 0], [0, 2, 2]]], dtype=torch.int64)

    target = target_latent_activations(top_acts, top_indices, latent_idx=2)

    assert torch.allclose(target, torch.tensor([[0.9, 0.4]], dtype=torch.float32))
