# Turing Explorer Circuit Discovery

> Unsupervised, multi-pass pipeline for discovering minimal, faithful sub-networks inside TuringLLM using Sparse Autoencoders.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

---

## Overview

This project implements a **circuit discovery** pipeline for transformer interpretability. Given a trained TuringLLM model and a bank of 36 Sparse Autoencoders (SAEs), it:

1. Decomposes activations into sparse, interpretable latent features
2. Mines co-activation statistics across the dataset
3. Constructs candidate circuits using 17 distinct discovery methods
4. Evaluates each circuit for **faithfulness**, **sufficiency**, **completeness**, and **minimality**

A *circuit* is a minimal sub-network of SAE latents whose activations alone faithfully reproduce the model's original behaviour on a given concept or task.

---

## How It Works

```
                   Dataset shards (.npy)
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Pass 1 — latent statistics + context collection        │
  │  Welford stats · top/mid/neg context reservoirs         │
  │  logit predictions · pooled residual representations    │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  ANN Step — negative context construction               │
  │  Pure-PyTorch approximate nearest-neighbour search      │
  │  finds semantically similar but latent-inactive seqs    │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Pass 2 — co-activation graph                           │
  │  Co-magnitude scoring over top-context sequences        │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Candidate selection — seed latent picking              │
  │  Activity-filtered, frequency-aware seed set            │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Circuit discovery — 17 methods per seed                │
  │  Gradient · statistical · sparse-expansion families     │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Evaluation — faithfulness / sufficiency /              │
  │  completeness / minimality pruning                      │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                         outputs/
```

---

## Project Structure

```
.
├── config.yaml                   — master configuration
├── requirements.txt
├── tests/                        — unit + integration tests
└── src/
    ├── main.py                   — entry point (full pipeline)
    ├── discover_circuits.py      — standalone discovery CLI
    ├── display_latents.py        — interactive latent inspector
    ├── search_latents.py         — keyword search over latents
    ├── ablation_sensitivity.py   — ablation sensitivity analysis utility
    ├── hardware.py               — hardware detection and device helpers
    ├── config.py                 — config loader and typed access
    ├── model/                    — TuringLLM transformer, inference, hooks, tokenizer
    ├── sae/                      — SAE bank (36 SAEs), Triton top-K, cublasLt encoder, fused linear-ReLU
    ├── pipeline/                 — pass 1, ANN step, pass 2, candidate selection, discovery orchestration, persist
    ├── store/                    — latent stats, context stores, co-activation graph, circuit store, search cache
    ├── circuit/
    │   ├── types/                — FeatureID, SparseAct, fused ops
    │   ├── instrument/           — SAE graph hooks, attribution, patcher, neg-ctx baseline
    │   ├── discovery/            — all 17 discovery method implementations + METHODS.md
    │   ├── probe_dataset.py      — positive/negative probe dataset builder
    │   ├── discovery_window.py   — per-seed discovery orchestrator
    │   └── feature_selection.py  — candidate latent filtering
    ├── eval/                     — faithfulness, upstream faithfulness, sufficiency, completeness, minimality
    ├── display/                  — Rich terminal heatmap, circuit listing, top-candidate viewer
    ├── observability/            — timing, progress tracking, circuit logging, console helpers
    ├── debug/                    — standalone debug and profiling scripts
    └── native/                   — C++/CUDA extensions (Welford, reservoir, co-magnitude, top-K)
```

---

## Model & SAE Details


| TuringLLM       |        | SAE Bank        |                    |
| --------------- | ------ | --------------- | ------------------ |
| Layers          | 12     | Total SAEs      | 36                 |
| Embedding dim   | 1,024  | Kinds           | attn · mlp · resid |
| Attention heads | 16     | Dictionary size | 40,960             |
| MLP hidden size | 4,096  | Top-K sparsity  | 128                |
| Vocabulary      | 50,304 | Input dim       | 1,024              |
| Context length  | 1,024  |                 |                    |


The SAE bank supports multi-GPU layer splitting, `torch.compile`, a cublasLt fused encoder, a Triton radix-select top-K kernel, and per-kind CUDA stream parallelism.

---

## Installation

**Requirements:** Python 3.12, CUDA 13.0, Linux

> **Windows users:** the native CUDA extensions and PyTorch CUDA builds require a Linux environment. Use [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) with a CUDA-capable GPU and install the CUDA toolkit inside WSL before proceeding.

### 1 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Build native C++/CUDA extensions

```bash
cd src/native
python setup.py build_ext --inplace
cd ../..
```

The extensions provide:

- CUDA Welford statistics kernel (`latent_stats_cuda.cu`)
- cublasLt fused Linear+ReLU (`linear_relu.cu`)
- OpenMP co-magnitude reducer (`comagnitude_reduce.cpp`)
- OpenMP Algorithm-R reservoir sampler (`mid_reservoir.cpp`)

### 4 — Configure paths

Edit `config.yaml` and set:

```yaml
weights:
  model_path: "models/TuringLLM/model_<checkpoint>.pt"
  sae_path: "models/TuringLLM/SAE"
data:
  dataset_path: "data"
```

---

## Usage

> **Windows users:** run all commands inside WSL 2 with the virtual environment activated (`source .venv/bin/activate`).

### Run the full pipeline

```bash
./scripts/run.sh
```

Or directly:

```bash
python src/main.py
```

Runs both collection passes, ANN negative context construction, co-activation graph building, seed selection, and circuit discovery. All outputs are written to `outputs/`.

> The script sets `PYTHONPATH` automatically. If running `src/main.py` directly, ensure `src/` is on your `PYTHONPATH` or run from the project root.

### Inspect a latent interactively

```bash
./scripts/display_latents.sh
```

Or directly:

```bash
python src/display_latents.py --layer 3 --kind mlp --latent 512
```

Renders a Rich terminal heatmap of top/mid/neg context sequences for any latent, alongside its top predicted tokens.

### Keyword search over latents

```bash
./scripts/search_latents.sh
```

Or with automatic patch-clamp evaluation:

```bash
./scripts/search_latents_and_run_patch.sh
```

Or directly:

```bash
python src/search_latents.py --query "Paris,France"
python src/search_latents.py --query "capital" --run_patch_clamp
```

Queries the Parquet search cache for latents whose top contexts match the given keywords.

### Standalone circuit discovery

```bash
./scripts/discover.sh
```

Or directly:

```bash
python src/discover_circuits.py
```

Runs only the discovery phase (requires pre-built `outputs/` from a prior pipeline run).

### List discovered circuits

```bash
./scripts/list_circuits.sh
```

Prints a sorted summary of all accepted circuits from `outputs/circuits/summary.json`.

### View top candidate latents

```bash
./scripts/get_top_candidates.sh
```

Displays the highest-scoring seed latent candidates selected for discovery.

### Run ablation matrix

```bash
./scripts/ablation_matrix.sh
```

Sweeps discovery hyperparameters and evaluates circuit quality across the matrix.

---

## Configuration Reference

All settings live in `config.yaml`. The most commonly adjusted keys:


| Key                                                   | Description                                                                                 | Default        |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------- |
| `weights.model_path`                                  | Path to TuringLLM checkpoint                                                                | —              |
| `weights.sae_path`                                    | Directory of SAE weights                                                                    | —              |
| `data.dataset_path`                                   | Directory of tokenised `.npy` shards                                                        | —              |
| `data.n_shards`                                       | Number of shards to process                                                                 | `256`          |
| `data.batch_size`                                     | Sequences per inference batch                                                               | `512`          |
| `hardware.multi_gpu`                                  | Split layers across all GPUs                                                                | `false`        |
| `hardware.memory`                                     | `efficient` (CPU offload) or `fast` (pin memory)                                            | `efficient`    |
| `hardware.compile`                                    | `torch.compile` for model + SAEs                                                            | `true`         |
| `hardware.parallel_kinds`                             | Dispatch attn/mlp/resid SAE encodes on separate CUDA streams                                | `false`        |
| `hardware.ann_device`                                 | Device for ANN neg-ctx search: `auto`, `gpu`, or `cpu`                                      | `auto`         |
| `latents.top_ctx.n_sequences`                         | Top sequences stored per latent                                                             | `64`           |
| `latents.mid_ctx.n_sequences`                         | Reservoir size per latent (mid-band activations)                                            | `64`           |
| `latents.mid_ctx.band_low_sigma`                      | Lower edge of mid band (mean + N×std)                                                       | `0.5`          |
| `latents.mid_ctx.band_high_sigma`                     | Upper edge of mid band (mean + N×std)                                                       | `1.5`          |
| `latents.mid_ctx.warmup_batches`                      | Batches per component before mid-ctx updates begin                                          | `100`          |
| `latents.neg_ctx.n_sequences`                         | Negative sequences stored per latent                                                        | `64`           |
| `latents.neg_ctx.n_neighbors`                         | ANN query K per latent (oversampled before pos-ctx filter)                                  | `512`          |
| `latents.neg_ctx.min_pos_ctx`                         | Minimum pos-ctx sequences required to attempt neg retrieval                                  | `8`            |
| `latents.neg_ctx.repr_mode`                           | Residual representation mode: `mean_pool` or `last_token`                                   | `mean_pool`    |
| `latents.neg_ctx.max_repr_seqs`                       | Cap on sequences stored in seq_repr (`null` = all)                                          | `200000`       |
| `latents.logit_ctx.n_tokens_per_latent`               | Token prediction entries tracked per latent                                                 | `32`           |
| `latents.logit_ctx.topk_output_tokens`                | Top predicted tokens stored per latent                                                      | `32`           |
| `latents.top_coactivation.n_latents_per_latent`       | Co-activation graph degree                                                                  | `64`           |
| `latents.top_coactivation.n_candidates_per_component` | Candidates extracted per component per sequence during dump                                  | `16`           |
| `latents.top_coactivation.freq_alpha`                 | Exponent for frequency adjustment of co-activation scores                                   | `2.0`          |
| `latents.top_coactivation.mode`                       | Scoring mode: `pmi`, `freq_weighted`, or `raw`                                              | `pmi`          |
| `latents.top_coactivation.pmi_clamp_min`              | Lower bound for PMI values                                                                  | `-5.0`         |
| `latents.top_coactivation.pmi_clamp_max`              | Upper bound for PMI values                                                                  | `10.0`         |
| `discovery.n_seeds`                                   | Seed latents to discover from                                                               | `128`          |
| `discovery.probe_batch_size`                          | Sequences per instrumented forward pass                                                     | `4`            |
| `discovery.neg_ctx_eval_max`                          | Neg sequences used to build the per-seed ablation baseline for eval (0 = zero ablation)     | `16`           |
| `discovery.min_faithfulness`                          | Minimum faithfulness to accept a circuit                                                    | `0.2`          |
| `discovery.min_active_count`                          | Minimum lifetime firing count for a latent to be a candidate                                | `1`            |
| `discovery.max_neighbors`                             | Default max co-activation neighbors per node (fallback for all methods)                     | `32`           |
| `discovery.methods`                                   | List of discovery methods to run                                                            | see below      |
| `persist.save_workers`                                | Max parallel `torch.save` threads                                                           | `1`            |
| `persist.search_cache_enabled`                        | Whether to build the keyword search cache                                                   | `true`         |
| `persist.search_cache_n_sequences`                    | Sequences stored per latent in the search cache                                             | `8`            |
| `persist.search_cache_component_chunk`                | Components processed at a time during cache build                                           | `4`            |


---

## Discovery Methods


| Method                                  | Algorithm                                                                                           |
| --------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `attn_top_coact_sparse_expansion`       | Variable-depth BFS over attn co-activation graph; full MLP/resid passthrough                        |
| `mlp_top_coact_sparse_expansion`        | Same, MLP-targeted; full attn/resid passthrough                                                     |
| `resid_top_coact_sparse_expansion`      | Same, resid-targeted; full attn/MLP passthrough                                                     |
| `attn_mlp_top_coact_sparse_expansion`   | BFS over attn+MLP; full resid passthrough                                                           |
| `attn_resid_top_coact_sparse_expansion` | BFS over attn+resid; full MLP passthrough                                                           |
| `mlp_resid_top_coact_sparse_expansion`  | BFS over MLP+resid; full attn passthrough                                                           |
| `all_top_coact_sparse_expansion`        | BFS over all kinds; no passthrough                                                                  |
| `hard_negative_coact_sparse_expansion`  | BFS activators + gradient-validated inhibitors discovered from hard-negative contexts               |
| `coactivation_statistical`              | Threshold-based co-activation edge pruning                                                          |
| `logit_attribution`                     | Two-pass gradient: `activation × gradient` node/edge scoring                                        |
| `sfc_attribution_patching`              | SFC-style integrated-gradient node scores + `delta × gradient` edges (Marks et al. 2024)           |
| `neighborhood_expansion`                | Two-hop statistical neighbourhood; no gradients                                                     |
| `top_coact_attr`                        | Legacy multi-hop upstream/downstream attribution patching                                           |
| `differential_activation`               | Pos/neg contrast scan ranks activators/inhibitors; gradient validates causal edges                  |
| `gradient_upstream`                     | Multi-hop backward attribution: selects top-K upstream latents per hop via `activation × gradient` |
| `layerwise_gradient_upstream`           | Layer-by-layer backward sweep; attributes each node against all upstream layers                     |
| `counterfactual_gradient`               | Gradient attribution on neg-ctx to find absent activators and present inhibitors                    |


The expansion depth for sparse methods is configured per-method with `coact_depth`, e.g. `[32, 16]` = depth-2 BFS with 32 neighbors at hop 1 and 16 at hop 2.

---

## Evaluation

Each discovered circuit is scored with multiple forward passes:

```
Faithfulness = 1 − MSE(circuit_logits, original_logits)
                   ─────────────────────────────────────
                   MSE(baseline_logits, original_logits)
```

The baseline is either zero-ablation or a mean over neg-ctx sequences, controlled by `discovery.neg_ctx_eval_max`.


| Metric                    | Measures                                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Faithfulness**          | How well the circuit alone reproduces the model's output distribution                                         |
| **Upstream faithfulness** | Variant used by gradient-based methods: measures how well the circuit recovers the seed latent's activation   |
| **Kind-local faithfulness** | Faithfulness computed while patching only the target kinds; used as the acceptance gate by the sparse-expansion family and `differential_activation` |
| **Sufficiency**           | Whether the circuit is sufficient to produce the correct prediction                                           |
| **Completeness**          | Whether removing the circuit degrades the model                                                               |
| **Minimality**            | Leave-one-out pruning of redundant nodes                                                                      |


A circuit is accepted if its faithfulness exceeds `discovery.min_faithfulness` (default `0.2`). Gradient-based methods (`gradient_upstream`, `layerwise_gradient_upstream`, `counterfactual_gradient`) use upstream faithfulness as their acceptance gate; sparse-expansion methods gate on kind-local faithfulness.

---

## Outputs

All outputs are written to `outputs/` after a full pipeline run:


| File                              | Contents                                          |
| --------------------------------- | ------------------------------------------------- |
| `latent_stats.pt`                 | Per-latent Welford statistics (mean, std, counts) |
| `top_ctx.pt`                      | Top-activating sequences per latent               |
| `mid_ctx.pt`                      | Mid-band reservoir sequences per latent           |
| `neg_ctx.pt`                      | ANN-retrieved negative context sequences          |
| `seq_repr.pt`                     | Pooled residual stream representations            |
| `logit_ctx.pt`                    | Top predicted tokens per latent                   |
| `top_coactivation.pt`             | Co-activation graph (values + indices)            |
| `search_cache.parquet`            | Keyword-searchable token index                    |
| `candidates.pt`                   | Selected seed latents                             |
| `circuits/discovered_circuits.pt` | All discovered circuits with scores               |
| `circuits/summary.json`           | Per-seed summary of accepted circuits             |


---

## Tests

```bash
pytest tests/
```

Native extension tests:

```bash
python src/native/tests/test_topk.py
python src/native/tests/test_reduce.py
```

---

## Tech Stack


| Category    | Details                                                      |
| ----------- | ------------------------------------------------------------ |
| **Core**    | PyTorch 2.10 · CUDA 13.0 · Python 3.12                       |
| **Kernels** | Triton (top-K) · cublasLt (fused Linear+ReLU) · OpenMP (C++) |
| **ML**      | Transformers 5.1 (Phi-3 tokeniser) · torchao 0.16            |
| **Data**    | pandas 3.0 · pyarrow 23.0 · NumPy                            |
| **Display** | Rich 14.3                                                    |
| **Build**   | Ninja 1.13                                                   |


