import torch

from analysis.coactivation.coact_degrees import compute_coact_degrees
from analysis.coactivation.component_pair_heatmap import compute_component_pair_heatmap
from analysis.coactivation.component_bipartite_graph import compute_component_bipartite_graph
from analysis.coactivation.component_signature_similarity import compute_component_signature_similarity
from analysis.coactivation.hub_corrected_coacts import compute_hub_corrected_coacts
from analysis.coactivation.jaccard_overlap import compute_jaccard_overlap
from analysis.coactivation.latent_profile_clustering import compute_latent_profile_clustering
from analysis.coactivation.latent_profile_pca import compute_latent_profile_pca
from analysis.coactivation.mutual_coact_graph import compute_mutual_coact_graph
from analysis.coactivation.mutual_neighbor_similarity import compute_mutual_neighbor_similarity
from analysis.coactivation.profile_similarity_distribution import compute_profile_similarity_distribution
from analysis.coactivation.profile_utils import build_hashed_coact_profiles
from analysis.coactivation.same_cross_distribution import compute_same_cross_distribution
from analysis.coactivation.sorted_pmi_decay import compute_sorted_pmi_decay
from analysis.coactivation.threshold_counts import compute_threshold_counts
from analysis.coactivation.top_ctx_sequence_overlap import compute_top_ctx_sequence_overlap
from analysis.coactivation.top_coact_hubs import compute_top_coact_hubs


def test_compute_sorted_pmi_decay_sorts_each_target_before_quantiles():
    values = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0],
                [5.0, -1.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    stats = compute_sorted_pmi_decay(values, quantiles=(0.5,))

    assert stats["ranks"] == [1, 2, 3]
    assert stats["quantiles"]["p50"] == [4.0, 1.5, -0.5]
    assert stats["mean"] == [4.0, 1.5, -0.5]
    assert stats["num_targets"] == 2
    assert stats["top_k"] == 3


def test_compute_threshold_counts_counts_strong_coacts_per_target():
    values = torch.tensor(
        [
            [
                [0.0, 3.0, 1.0],
                [5.0, -1.0, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    stats = compute_threshold_counts(values, thresholds=(1.0, 2.0))

    assert stats["summaries"]["gt_1"]["p50"] == 1.5
    assert stats["summaries"]["gt_2"]["mean"] == 1.0
    rows_gt2 = [row for row in stats["rows"] if row["threshold_key"] == "gt_2"]
    assert rows_gt2[1]["target_count"] == 2


def test_component_pair_heatmap_counts_high_pmi_rates():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    stats = compute_component_pair_heatmap(values, indices, d_sae=2, threshold=2.0)

    assert stats["pair_counts"] == [[2, 2], [2, 2]]
    assert stats["high_counts"] == [[1, 1], [0, 2]]
    assert stats["high_rate"][1][1] == 1.0


def test_same_cross_distribution_splits_by_component():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    stats = compute_same_cross_distribution(values, indices, d_sae=2, bins=5, value_range=(-1.0, 4.0))

    assert stats["same_summary"]["count"] == 4
    assert stats["cross_summary"]["count"] == 4
    assert stats["same_summary"]["pmi_gt2_count"] == 3
    assert stats["cross_summary"]["pmi_gt2_count"] == 1


def test_hashed_coact_profiles_use_positive_pmi_weights():
    values = torch.tensor([[[3.0, -2.0], [0.0, 4.0]]], dtype=torch.float32)
    indices = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.int64)

    profiles = build_hashed_coact_profiles(
        values,
        indices,
        sample_indices=torch.tensor([0, 1]),
        hash_bins=4,
    )

    assert torch.allclose(profiles[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(profiles[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_component_signature_similarity_returns_target_component_cosines():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    stats = compute_component_signature_similarity(values, indices, d_sae=2, threshold=2.0)

    assert stats["similarity"][0][0] > 0.999
    assert stats["similarity"][1][1] > 0.999
    assert stats["top_similar_component_pairs"]


def test_latent_profile_pca_and_similarity_distribution_smoke():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    pca_stats = compute_latent_profile_pca(values, indices, d_sae=2, max_samples=4, hash_bins=4)
    similarity_stats = compute_profile_similarity_distribution(
        values,
        indices,
        d_sae=2,
        max_samples=4,
        hash_bins=4,
        bins=5,
    )

    assert pca_stats["sample_count"] == 4
    assert len(pca_stats["pc1"]) == 4
    assert similarity_stats["same_pair_count"] == 2
    assert similarity_stats["cross_pair_count"] > 0


def test_top_coact_hubs_counts_high_pmi_latent_ids():
    values = torch.tensor([[[3.0, 0.0], [5.0, 4.0]]], dtype=torch.float32)
    indices = torch.tensor([[[2, 1], [2, 3]]], dtype=torch.int64)

    stats = compute_top_coact_hubs(
        values,
        indices,
        num_components=2,
        d_sae=2,
        threshold=2.0,
        top_n=2,
    )

    assert stats["total_high_pmi_edges"] == 3
    assert stats["unique_high_pmi_coact_latents"] == 2
    assert stats["top_hubs"][0]["global_latent_id"] == 2
    assert stats["top_hubs"][0]["high_pmi_count"] == 2


def test_component_bipartite_graph_selects_strong_edges():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    stats = compute_component_bipartite_graph(values, indices, d_sae=2, threshold=2.0, max_edges=2)

    assert len(stats["edges"]) == 2
    assert stats["edges"][0]["high_count"] >= stats["edges"][1]["high_count"]


def test_latent_clustering_and_jaccard_overlap_smoke():
    values = torch.tensor(
        [
            [[3.0, 0.0], [1.0, 4.0]],
            [[5.0, -1.0], [2.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2], [1, 3]],
            [[2, 0], [3, 1]],
        ],
        dtype=torch.int64,
    )

    cluster_stats = compute_latent_profile_clustering(
        values,
        indices,
        d_sae=2,
        max_samples=4,
        hash_bins=4,
        cluster_count=2,
        iterations=2,
    )
    jaccard_stats = compute_jaccard_overlap(
        values,
        indices,
        d_sae=2,
        max_samples=4,
        top_k=1,
        bins=5,
    )

    assert cluster_stats["sample_count"] == 4
    assert len(cluster_stats["cluster_summaries"]) <= 2
    assert jaccard_stats["same_pair_count"] == 2
    assert jaccard_stats["cross_pair_count"] > 0


def test_mutual_coact_graph_finds_reciprocal_pairs():
    values = torch.tensor([[[3.0, 1.0], [4.0, 0.0], [5.0, 1.0], [3.5, 0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[1, 2], [0, 2], [3, 0], [2, 1]]], dtype=torch.int64)

    stats = compute_mutual_coact_graph(values, indices, d_sae=4, threshold=2.0, top_n=5)

    assert stats["mutual_pair_count"] == 2
    assert stats["top_pairs"][0]["mutual_strength"] >= stats["top_pairs"][1]["mutual_strength"]


def test_hub_corrected_coacts_penalizes_frequent_destinations():
    values = torch.tensor([[[5.0, 4.0], [4.5, 4.0], [4.0, 3.5]]], dtype=torch.float32)
    indices = torch.tensor([[[2, 1], [2, 0], [2, 1]]], dtype=torch.int64)

    stats = compute_hub_corrected_coacts(values, indices, d_sae=3, threshold=2.0, top_n=3)

    assert stats["edge_count"] == 6
    assert stats["max_in_degree"] == 3
    assert stats["top_corrected_edges"]


def test_mutual_neighbor_similarity_scores_shared_neighbors():
    values = torch.tensor([[[3.0, 2.5], [4.0, 2.8], [3.5, 0.0], [3.2, 0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[2, 3], [2, 3], [0, 1], [0, 1]]], dtype=torch.int64)

    stats = compute_mutual_neighbor_similarity(values, indices, d_sae=4, threshold=2.0, max_samples=4, bins=5)

    assert stats["sample_count"] == 4
    assert stats["same_pair_count"] > 0
    assert stats["same_summary"]["common_count_mean"] > 0


def test_top_ctx_sequence_overlap_uses_exact_sequence_ids():
    values = torch.tensor([[[3.0], [4.0], [0.0]]], dtype=torch.float32)
    indices = torch.tensor([[[1], [2], [0]]], dtype=torch.int64)
    ctx_seq_idx = torch.tensor([[[10, 11, 12], [11, 12, 13], [20, 21, 22]]], dtype=torch.int64)

    stats = compute_top_ctx_sequence_overlap(
        values,
        indices,
        ctx_seq_idx,
        d_sae=3,
        threshold=2.0,
        max_edge_samples=2,
        bins=4,
    )

    assert stats["sample_count"] == 2
    assert stats["coact_summary"]["overlap_mean"] == 1.0
    assert stats["top_pairs"][0]["shared_top_ctx_sequence_count"] == 2


def test_coact_degrees_expands_multi_hop_neighborhoods():
    values = torch.tensor(
        [
            [
                [3.0, 0.0],
                [3.0, 0.0],
                [3.0, 0.0],
                [0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 1],
            ]
        ],
        dtype=torch.int64,
    )

    stats = compute_coact_degrees(
        values,
        indices,
        d_sae=4,
        threshold=2.0,
        max_samples=4,
        top_out_degree=1,
        max_frontier=4,
        hub_quantile=1.0,
    )

    assert stats["sample_count"] == 4
    assert stats["reach_summary"]["1"]["max"] >= 1
    assert stats["reach_summary"]["2"]["max"] >= 2
    assert stats["new_summary"]["3"]["count"] == 4

