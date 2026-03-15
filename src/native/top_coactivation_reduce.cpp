#include <torch/extension.h>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Pair {
    int32_t id;
    float val;
};

/*
 * Open-addressing hash table for O(n) dedup+sum.
 * Power-of-2 size with linear probing.
 * Uses a generation counter to avoid full clears between targets.
 */
struct HashTable {
    static constexpr int CAPACITY = 16384; // must be power of 2, > max possible unique IDs per target
    static constexpr int MASK = CAPACITY - 1;

    struct Slot {
        int32_t key;
        float   val;
        int32_t gen;  // generation counter — slot is valid only if gen == current_gen
    };

    Slot slots[CAPACITY];
    int32_t current_gen;
    int32_t used_indices[CAPACITY]; // tracks which slots were written (for fast scan)
    int used_count;

    HashTable() : current_gen(0), used_count(0) {
        std::memset(slots, 0, sizeof(slots));
    }

    void reset() {
        current_gen++;
        used_count = 0;
        // If gen wraps around (every ~2B calls), do a full clear
        if (current_gen <= 0) {
            current_gen = 1;
            std::memset(slots, 0, sizeof(slots));
        }
    }

    void insert_add(int32_t key, float val) {
        int h = (key * 2654435761u) & MASK; // Knuth multiplicative hash
        while (true) {
            if (slots[h].gen != current_gen) {
                // Empty slot — insert new
                slots[h].key = key;
                slots[h].val = val;
                slots[h].gen = current_gen;
                used_indices[used_count++] = h;
                return;
            }
            if (slots[h].key == key) {
                // Existing key — add value
                slots[h].val += val;
                return;
            }
            h = (h + 1) & MASK;
        }
    }

    // Collect all entries into buf, returns count
    int collect(Pair* buf) const {
        int count = 0;
        for (int i = 0; i < used_count; i++) {
            int h = used_indices[i];
            buf[count++] = {slots[h].key, slots[h].val};
        }
        return count;
    }
};

static void process_target(
    HashTable&     ht,
    Pair*          buf,
    const int32_t* cand_ids,
    const float*   cand_vals,
    const int64_t* seq_rows,
    int            n_seqs,
    int            M,
    int32_t        self_id,
    int32_t*       out_ids,
    float*         out_vals,
    int            K
) {
    ht.reset();

    for (int s = 0; s < n_seqs; s++) {
        int64_t row = seq_rows[s];
        if (row < 0) continue;
        const int32_t* row_ids  = cand_ids  + row * M;
        const float*   row_vals = cand_vals + row * M;
        for (int m = 0; m < M; m++) {
            if (row_ids[m] != self_id && row_vals[m] > 0.0f) {
                ht.insert_add(row_ids[m], row_vals[m]);
            }
        }
    }

    int unique_count = ht.collect(buf);

    if (unique_count == 0) {
        std::memset(out_ids, 0, K * sizeof(int32_t));
        std::memset(out_vals, 0, K * sizeof(float));
        return;
    }

    int actual_k = std::min(K, unique_count);

    std::nth_element(buf, buf + actual_k, buf + unique_count,
                     [](const Pair& a, const Pair& b) { return a.val > b.val; });

    std::sort(buf, buf + actual_k,
              [](const Pair& a, const Pair& b) { return a.val > b.val; });

    for (int i = 0; i < actual_k; i++) {
        out_ids[i]  = buf[i].id;
        out_vals[i] = buf[i].val;
    }
    if (actual_k < K) {
        std::memset(out_ids + actual_k, 0, (K - actual_k) * sizeof(int32_t));
        std::memset(out_vals + actual_k, 0, (K - actual_k) * sizeof(float));
    }
}

/*
 * Build the inverted index from the CSR (sequence -> targets) into
 * per-target sequence row lists, then run the reduce in parallel.
 */
std::tuple<torch::Tensor, torch::Tensor> reduce_topk(
    torch::Tensor candidate_ids,
    torch::Tensor candidate_vals,
    torch::Tensor seq_offsets,
    torch::Tensor seq_targets,
    torch::Tensor sid_to_row,
    int64_t num_components,
    int64_t d_sae,
    int64_t K
) {
    TORCH_CHECK(candidate_ids.dtype() == torch::kInt32, "candidate_ids must be int32");
    TORCH_CHECK(candidate_vals.dtype() == torch::kFloat32, "candidate_vals must be float32");
    TORCH_CHECK(seq_offsets.dtype() == torch::kInt64, "seq_offsets must be int64");
    TORCH_CHECK(seq_targets.dtype() == torch::kInt64, "seq_targets must be int64");
    TORCH_CHECK(sid_to_row.dtype() == torch::kInt64, "sid_to_row must be int64");
    TORCH_CHECK(candidate_ids.is_contiguous(), "candidate_ids must be contiguous");
    TORCH_CHECK(candidate_vals.is_contiguous(), "candidate_vals must be contiguous");

    const int M = static_cast<int>(candidate_ids.size(1));
    const int64_t max_sid = seq_offsets.size(0) - 1;
    const int64_t n_targets = num_components * d_sae;
    const int64_t total_entries = seq_targets.size(0);

    const auto* cand_ids_ptr  = candidate_ids.data_ptr<int32_t>();
    const auto* cand_vals_ptr = candidate_vals.data_ptr<float>();
    const auto* offsets_ptr   = seq_offsets.data_ptr<int64_t>();
    const auto* targets_ptr   = seq_targets.data_ptr<int64_t>();
    const auto* sid_row_ptr   = sid_to_row.data_ptr<int64_t>();

    auto t0 = std::chrono::high_resolution_clock::now();

    // --- Build inverted index: target_global_id -> list of sequence rows ---

    std::vector<int> target_counts(n_targets, 0);
    for (int64_t i = 0; i < total_entries; i++) {
        int64_t g = targets_ptr[i];
        if (g >= 0 && g < n_targets) {
            target_counts[g]++;
        }
    }

    std::vector<int64_t> inv_offsets(n_targets + 1, 0);
    for (int64_t g = 0; g < n_targets; g++) {
        inv_offsets[g + 1] = inv_offsets[g] + target_counts[g];
    }

    int64_t inv_total = inv_offsets[n_targets];
    std::vector<int64_t> inv_seq_rows(inv_total);

    std::vector<int> write_pos(n_targets, 0);
    for (int64_t sid = 1; sid <= max_sid; sid++) {
        int64_t start = offsets_ptr[sid - 1];
        int64_t end   = offsets_ptr[sid];
        int64_t row   = sid_row_ptr[sid];
        for (int64_t j = start; j < end; j++) {
            int64_t g = targets_ptr[j];
            if (g >= 0 && g < n_targets) {
                int64_t pos = inv_offsets[g] + write_pos[g]++;
                inv_seq_rows[pos] = row;
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Find max sequences any single target has (for buffer sizing)
    int max_seqs = 0;
    for (int64_t g = 0; g < n_targets; g++) {
        int c = target_counts[g];
        if (c > max_seqs) max_seqs = c;
    }

    // --- Allocate output ---
    auto top_ids  = torch::zeros({num_components, d_sae, K}, torch::kInt32);
    auto top_vals = torch::zeros({num_components, d_sae, K}, torch::kFloat32);
    auto* out_ids_ptr  = top_ids.data_ptr<int32_t>();
    auto* out_vals_ptr = top_vals.data_ptr<float>();

    auto t2 = std::chrono::high_resolution_clock::now();

    // --- Parallel reduce with thread-local hash tables and buffers ---
    int num_threads = 1;
    #ifdef _OPENMP
    num_threads = omp_get_max_threads();
    #endif

    const int buf_capacity = HashTable::CAPACITY;
    std::vector<HashTable> thread_hts(num_threads);
    std::vector<std::vector<Pair>> thread_bufs(num_threads);
    for (int t = 0; t < num_threads; t++) {
        thread_bufs[t].resize(buf_capacity);
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (int64_t g = 0; g < n_targets; g++) {
        int64_t inv_start = inv_offsets[g];
        int64_t inv_end   = inv_offsets[g + 1];
        int n_seqs = static_cast<int>(inv_end - inv_start);

        if (n_seqs == 0) {
            std::memset(out_ids_ptr + g * K, 0, K * sizeof(int32_t));
            std::memset(out_vals_ptr + g * K, 0, K * sizeof(float));
            continue;
        }

        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif

        process_target(
            thread_hts[tid],
            thread_bufs[tid].data(),
            cand_ids_ptr,
            cand_vals_ptr,
            inv_seq_rows.data() + inv_start,
            n_seqs,
            M,
            static_cast<int32_t>(g),
            out_ids_ptr  + g * K,
            out_vals_ptr + g * K,
            static_cast<int>(K)
        );
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("  [C++] Inverted index build: %.1f ms\n", ms(t0, t1));
    std::printf("  [C++] Output alloc:         %.1f ms\n", ms(t1, t2));
    std::printf("  [C++] Parallel reduce:      %.1f ms  (%d threads)\n", ms(t2, t3), num_threads);
    std::printf("  [C++] Total:                %.1f ms\n", ms(t0, t3));

    return {top_ids, top_vals};
}

int get_omp_max_threads() {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

bool has_openmp() {
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_omp_max_threads", &get_omp_max_threads, "Get the max number of OpenMP threads");
    m.def("has_openmp", &has_openmp, "Check if compiled with OpenMP support");
    m.def("reduce_topk", &reduce_topk,
          "Reduce per-sequence co-activation candidates to top-K per target latent.\n\n"
          "Args:\n"
          "  candidate_ids:  [S, M] int32 - candidate global latent IDs per sequence\n"
          "  candidate_vals: [S, M] float32 - candidate freq-adjusted magnitudes\n"
          "  seq_offsets:    [max_sid+1] int64 - CSR cumulative offsets\n"
          "  seq_targets:    [total] int64 - CSR target global IDs\n"
          "  sid_to_row:     [max_sid+1] int64 - sequence ID to candidate row mapping\n"
          "  num_components: int - number of SAE components (36)\n"
          "  d_sae:          int - SAE dictionary size (40960)\n"
          "  K:              int - number of top co-occurring latents to keep (32)\n"
          "\n"
          "Returns:\n"
          "  (top_ids, top_vals): each [num_components, d_sae, K]\n",
          py::arg("candidate_ids"),
          py::arg("candidate_vals"),
          py::arg("seq_offsets"),
          py::arg("seq_targets"),
          py::arg("sid_to_row"),
          py::arg("num_components"),
          py::arg("d_sae"),
          py::arg("K")
    );
}
