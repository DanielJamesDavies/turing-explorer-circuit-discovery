#include <torch/extension.h>
#include <algorithm>
#include <vector>
#include <random>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Streaming reservoir update (Algorithm R) for mid-context latent stores.
 *
 * Receives a batch of in-band (latent_idx, seq_id, score) triples, sorted
 * ascending by latent_idx, and updates the per-latent reservoirs in-place.
 *
 * Each latent's reservoir is updated sequentially within its group so that
 * Algorithm R remains unbiased when multiple samples arrive in the same batch.
 * Different latent groups are processed in parallel via OpenMP.
 * Per-thread mt19937 RNGs avoid lock contention.
 */

void reservoir_update(
    torch::Tensor lat_idxs,         // int32  [N_pairs] sorted ascending by latent index
    torch::Tensor seq_ids,          // int32  [N_pairs]
    torch::Tensor scores,           // float32 [N_pairs]
    torch::Tensor reservoir_ids,    // int32  [d_sae, N_mid]  modified in-place
    torch::Tensor reservoir_scores, // float32 [d_sae, N_mid] modified in-place
    torch::Tensor reservoir_fill,   // int32  [d_sae]         modified in-place
    torch::Tensor reservoir_n,      // int64  [d_sae]         modified in-place
    int64_t N_mid
) {
    TORCH_CHECK(lat_idxs.dtype()         == torch::kInt32,   "lat_idxs must be int32");
    TORCH_CHECK(seq_ids.dtype()          == torch::kInt32,   "seq_ids must be int32");
    TORCH_CHECK(scores.dtype()           == torch::kFloat32, "scores must be float32");
    TORCH_CHECK(reservoir_ids.dtype()    == torch::kInt32,   "reservoir_ids must be int32");
    TORCH_CHECK(reservoir_scores.dtype() == torch::kFloat32, "reservoir_scores must be float32");
    TORCH_CHECK(reservoir_fill.dtype()   == torch::kInt32,   "reservoir_fill must be int32");
    TORCH_CHECK(reservoir_n.dtype()      == torch::kInt64,   "reservoir_n must be int64");
    TORCH_CHECK(!lat_idxs.is_cuda(),     "all tensors must be on CPU");

    int64_t N = lat_idxs.size(0);
    if (N == 0) return;

    const int32_t* lat_ptr  = lat_idxs.data_ptr<int32_t>();
    const int32_t* seq_ptr  = seq_ids.data_ptr<int32_t>();
    const float*   scr_ptr  = scores.data_ptr<float>();
    int32_t*       res_ids  = reservoir_ids.data_ptr<int32_t>();
    float*         res_scr  = reservoir_scores.data_ptr<float>();
    int32_t*       res_fill = reservoir_fill.data_ptr<int32_t>();
    int64_t*       res_n    = reservoir_n.data_ptr<int64_t>();

    // Build group boundaries with a single scan over sorted lat_idxs.
    struct Group { int32_t lat; int64_t start; int64_t end; };
    std::vector<Group> groups;
    groups.reserve(N);
    for (int64_t i = 0; i < N; ) {
        int32_t lat   = lat_ptr[i];
        int64_t start = i;
        while (i < N && lat_ptr[i] == lat) ++i;
        groups.push_back({lat, start, i});
    }
    int n_groups = static_cast<int>(groups.size());

    // One mt19937 per thread, seeded with hardware entropy XOR thread id.
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::random_device rd;
    std::vector<std::mt19937> rngs(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        rngs[t].seed(rd() ^ (static_cast<uint32_t>(t) * 2654435761u));
    }

    // Parallel over latent groups; dynamic scheduling handles variable group sizes.
#pragma omp parallel for schedule(dynamic, 64)
    for (int g = 0; g < n_groups; ++g) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& rng = rngs[tid];

        int32_t lat    = groups[g].lat;
        int64_t gstart = groups[g].start;
        int64_t gend   = groups[g].end;

        int32_t fill = res_fill[lat];
        int64_t n    = res_n[lat];
        int32_t* ids_row = res_ids + static_cast<int64_t>(lat) * N_mid;
        float*   scr_row = res_scr + static_cast<int64_t>(lat) * N_mid;

        // Process each sample for this latent sequentially (Algorithm R).
        for (int64_t j = gstart; j < gend; ++j) {
            ++n;
            int32_t sid = seq_ptr[j];
            float   scr = scr_ptr[j];

            if (fill < static_cast<int32_t>(N_mid)) {
                ids_row[fill] = sid;
                scr_row[fill] = scr;
                ++fill;
            } else {
                // Replace slot r with probability N_mid / n.
                std::uniform_int_distribution<int64_t> dist(0, n - 1);
                int64_t r = dist(rng);
                if (r < N_mid) {
                    ids_row[r] = sid;
                    scr_row[r] = scr;
                }
            }
        }

        res_fill[lat] = fill;
        res_n[lat]    = n;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "reservoir_update", &reservoir_update,
        "Update mid-context reservoirs for a sorted batch of in-band (latent, seq_id, score) triples.\n\n"
        "Args:\n"
        "  lat_idxs:         [N_pairs] int32   — latent indices within component, sorted ascending\n"
        "  seq_ids:          [N_pairs] int32   — global sequence IDs\n"
        "  scores:           [N_pairs] float32 — per-sequence mean activation scores\n"
        "  reservoir_ids:    [d_sae, N_mid] int32   — reservoir seq IDs, modified in-place\n"
        "  reservoir_scores: [d_sae, N_mid] float32 — reservoir scores, modified in-place\n"
        "  reservoir_fill:   [d_sae] int32   — fill count per latent (0..N_mid), modified in-place\n"
        "  reservoir_n:      [d_sae] int64   — total in-band hit count per latent, modified in-place\n"
        "  N_mid:            int — reservoir capacity\n",
        py::arg("lat_idxs"),
        py::arg("seq_ids"),
        py::arg("scores"),
        py::arg("reservoir_ids"),
        py::arg("reservoir_scores"),
        py::arg("reservoir_fill"),
        py::arg("reservoir_n"),
        py::arg("N_mid")
    );
}
