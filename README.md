# Turing Explorer Circuit Discovery

> Unsupervised, multi-pass pipeline for discovering minimal, faithful sub-networks inside TuringLLM using Sparse Autoencoders.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%20%7C%202.8%20%7C%202.10-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.6%20%7C%2012.8%20%7C%2013.0-green)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

---

## Overview

This project implements a **circuit discovery** pipeline for transformer interpretability. Given a trained TuringLLM model and a bank of 36 Sparse Autoencoders (SAEs), it:

1. Decomposes activations into sparse, interpretable latent features
2. Mines co-activation statistics across the dataset
3. Constructs candidate circuits using 21 seed-based discovery methods (plus one seed-free method)
4. Evaluates each circuit causally — faithfulness, suppression, node presence, sufficiency, completeness, and minimality

A *circuit* is a minimal sub-network of SAE latents whose activations alone faithfully reproduce the model's original behaviour — where the "behaviour" is typically a **latent-as-endpoint**: the activation of a chosen seed SAE latent, rather than an output-level task. See [RESEARCH.md](RESEARCH.md) for the research framing, role taxonomy, and findings; the production recipe is the **learned-mask (TRI-AMP) family** described below.

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
  │  Neg-ctx Step — negative context construction           │
  │  Exact (brute-force) cosine k-NN over pooled residual   │
  │  reprs finds semantically similar, latent-inactive seqs │
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
  │  Criteria-scored, activity-filtered seed set            │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Circuit discovery — 21 methods per seed                │
  │  Gradient · statistical · sparse-expansion ·            │
  │  learned-mask families                                  │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Evaluation + post-circuit analysis                     │
  │  faithfulness / suppression / presence / pruning        │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                         outputs/
```

> The negative-context search is **exact** k-nearest-neighbour (full cosine matmul + `torch.topk`), not approximate — the `ann` naming in the code and the `hardware.ann_device` config key are legacy labels.

---

## Repository Map

| Path | What it is |
| ---- | ---------- |
| `src/` | The pipeline itself (see Project Structure below) |
| `scripts/` | Shell/Python entry-point wrappers (run, discover, search, distributed driver, benchmarks) |
| `tests/` | Unit + integration tests (`pytest tests/`) |
| `config.yaml` | Master configuration |
| `config_examples/` | Ready-made profiles: `h100-8x-distributed-simple-exact.yaml`, `local-distributed-smoke.yaml`, `rtx-5070-ti.yaml` |
| `docs/` | Operator guides — the [H100 8× full-run guide](docs/h100-8x-full-run-guide.md), native fused runbook, gradient-differential method spec |
| `RESEARCH.md` | One-page statement of the research programme — read this first |
| `experiments/` | 45+ numbered experiment campaigns; the evidence behind every number in the paper (`% data:` comments in `paper/main.tex` point here). See `experiments/README.md` and `experiments/INDEX.md` |
| `paper/` | LaTeX draft (`main.tex`), figures, references; build with `paper/build.ps1` |
| `agent-planning/` | Design/implementation plans (notably `multi-device-improvements/` for the distributed pipeline) |
| `external/circuit-tracer/` | Git submodule — reference Attribution-Graphs implementation used as an external baseline, pinned to the commit all circuit-tracer numbers were produced with |
| `description.md` / `concise_description.md` | Long- and short-form module-by-module repo walkthroughs |
| `dev-notes/` | Local research notes and evidence data (untracked; cited from source docstrings) |
| `models/`, `data/` | Gitignored, user-supplied: TuringLLM checkpoint + SAE bank, and tokenised `.npy` shards |
| `outputs/` | All pipeline artifacts (see Outputs below) |

---

## Project Structure

```
.
├── config.yaml                   — master configuration
├── requirements.txt
├── requirements-cu128.txt         — hosted CUDA 12.8 / PyTorch 2.8 profile
├── requirements-cu126.txt         — hosted CUDA 12.6 / PyTorch 2.7 fallback
├── tests/                        — unit + integration tests
└── src/
    ├── main.py                   — entry point (full pipeline)
    ├── discover_circuits.py      — standalone discovery CLI
    ├── build_search_cache.py     — standalone keyword search-cache builder
    ├── display_latents.py        — interactive latent inspector
    ├── search_latents.py         — keyword search over latents
    ├── ablation_sensitivity.py   — ablation sensitivity analysis utility
    ├── hardware.py               — hardware detection and device helpers
    ├── config.py                 — config loader and typed access
    ├── model/                    — TuringLLM transformer, inference, hooks, tokenizer
    ├── sae/                      — SAE bank (36 SAEs), Triton top-K, cublasLt encoder, fused exact top-k backend
    ├── data/                     — dataset shard loader
    ├── pipeline/                 — pass 1, neg-ctx step, pass 2, candidate selection, discovery orchestration, persist
    │   ├── distributed/          — manifest-driven multi-worker pipeline (controller, workers, merge, equivalence)
    │   └── negative_context_stage/
    ├── store/                    — latent stats, context stores, co-activation graph, circuit store, search cache
    │   └── neg_ctx/              — exact k-NN backends (single-GPU / multi-GPU / index-sharded)
    ├── circuit/
    │   ├── types/                — FeatureID, SparseAct, fused ops
    │   ├── instrument/           — SAE graph hooks, attribution, patcher, learned_mask engine, restoration, position-aware sets
    │   ├── discovery/            — discovery method implementations (+ top_coact_expansion/ sub-package)
    │   ├── analysis/             — post-circuit analysis plugins (config `analysis.methods`)
    │   ├── probe_dataset.py      — positive/negative probe dataset builder
    │   ├── discovery_window.py   — per-seed discovery orchestrator + METHOD_REGISTRY
    │   └── feature_selection.py  — candidate latent filtering
    ├── eval/                     — faithfulness families, floors, suppression, node presence, prune families
    │   └── sae/                  — SAE-quality evaluation (density, reconstruction, CE-recovered, report)
    ├── analysis/                 — offline analysis/plotting CLI
    ├── utils/                    — negative-context selector
    ├── display/                  — Rich terminal heatmap, circuit listing, top-candidate viewer
    ├── observability/            — timing, progress tracking, circuit logging, console helpers
    ├── debug/                    — standalone debug and profiling scripts
    └── native/                   — C++/CUDA extensions (Welford, reservoir, coactivation reduce, fused top-k)
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


The SAE bank supports multi-GPU layer splitting, `torch.compile`, a cublasLt fused encoder, a Triton radix-select top-K kernel (`sae.topk_backend`), an experimental fused exact top-k encode backend (`sae.encode_backend: fused_exact_topk`, with an optional native CUDA kernel), streaming vs deferred first-pass encoding (`first_pass.sae_encode_mode`), and per-kind CUDA stream parallelism.

---

## Installation

**Requirements:** Python 3.12, Linux, and a CUDA/PyTorch profile matching the
target machine.

> **Windows users:** the native CUDA extensions and PyTorch CUDA builds require a Linux environment. Use [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install) with a CUDA-capable GPU and install the CUDA toolkit inside WSL before proceeding.

### 0 — Clone (with submodules, optional)

The `external/circuit-tracer` baseline is a git submodule; a plain clone silently lacks it. It is only needed for the circuit-tracer baseline comparisons in `experiments/`:

```bash
git submodule update --init external/circuit-tracer
```

(It also requires its own virtual environment with an older `transformers` — see `experiments/README.md`.)

### 1 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install Python dependencies

Pick the requirements file that matches the CUDA runtime in the target
environment:

| File | Target | Torch stack |
| ---- | ------ | ----------- |
| `requirements.txt` | Local CUDA 13.0 profile | `torch 2.10.0+cu130`, `torchvision 0.25.0+cu130`, `torchao 0.16.0` |
| `requirements-cu128.txt` | Preferred hosted H100 profile, e.g. RunPod PyTorch 2.8 / CUDA 12.8 | `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `torchao 0.13.0` |
| `requirements-cu126.txt` | Hosted CUDA 12.6 fallback | `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `torchao 0.12.0` |

```bash
pip install -r requirements.txt
```

For a RunPod-style CUDA 12.8 image, use:

```bash
pip install -r requirements-cu128.txt
```

For a CUDA 12.6 provider image, use:

```bash
pip install -r requirements-cu126.txt
```

Before building native extensions, confirm PyTorch and the CUDA toolkit agree:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.device_count())"
nvcc --version
```

### 3 — Build native C++/CUDA extensions

If the machine has more than one CUDA toolkit or an older `/usr/bin/nvcc`, make
sure `CUDA_HOME` and `PATH` point at the toolkit matching `torch.version.cuda`.
For example, with the CUDA 13 local profile:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
```

On managed hosted images, `nvcc` may already be on `PATH`; only set
`CUDA_HOME` when the default toolkit does not match the installed PyTorch wheel.
You can save local exports in an ignored env file such as `.env.local`.

```bash
cd src/native
python setup.py build_ext --inplace
cd ../..
```

If this build fails, first check for a PyTorch/toolkit mismatch rather than
assuming CUDA 13 is required. The CUDA version reported by PyTorch and the
toolkit selected by `nvcc --version` should be from the same profile.

After building on a target host, run:

```bash
python -m pytest src/native/tests/ -q
python -m pytest tests/store/test_top_coactivation_modes.py -q
```

The extensions provide:

- CUDA Welford statistics kernel (`latent_stats_cuda.cu`)
- cublasLt fused Linear+ReLU (`linear_relu.cu`)
- CUDA fused exact top-k encoder (`linear_relu_topk_exact.cu`)
- OpenMP top-coactivation reducer (`top_coactivation_reduce.cpp`)
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

Ready-made hardware profiles live in `config_examples/`.

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

Runs both collection passes, exact k-NN negative context construction, co-activation graph building, seed selection, and circuit discovery. All outputs are written to `outputs/`.

> The script sets `PYTHONPATH` automatically. If running `src/main.py` directly, ensure `src/` is on your `PYTHONPATH` or run from the project root.

### Inspect a latent interactively

```bash
./scripts/display_latents.sh
```

Or directly:

```bash
python src/display_latents.py --layer 3 --kind mlp --latent 512
```

Renders a Rich terminal heatmap of top/mid/neg context sequences for any latent, alongside its top predicted tokens. (`--sequences` controls how many sequences are shown; omitting all flags drops into interactive mode.)

### Build the keyword search cache

```bash
./scripts/build_search_cache.sh
```

The search cache is **no longer built during the pipeline** by default (`persist.search_cache_enabled: false`, `persist.build_search_cache_after_pipeline: false`); run this after a pipeline run to produce `outputs/search_cache.parquet`.

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

Queries the Parquet search cache for latents whose top contexts match the given keywords (requires the search cache to have been built). Tuning flags: `--n_latents`, `--n_sequences`, `--n_patch`, `--n_gen`.

### Standalone circuit discovery

```bash
./scripts/discover.sh
```

Or directly:

```bash
python src/discover_circuits.py
```

Runs only the discovery phase (requires pre-built `outputs/` from a prior pipeline run). Note that the wrapper script **always re-runs candidate selection** (it passes `--reselect`); call the Python entry point directly to reuse saved candidates. Other flags: `--candidates <path>`, `--n-seeds <n>`.

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
./scripts/ablation_matrix.sh --layer 3 --kind mlp --latent 512
```

Ablation sensitivity analysis for a chosen latent (wraps `src/ablation_sensitivity.py`; flags, not positional args).

---

## Distributed & Multi-GPU

Beyond `hardware.multi_gpu` (layer splitting), the repo has two independent scaling mechanisms:

### 1 — Manifest-driven distributed pipeline

`src/pipeline/distributed/` runs the collection passes as a controller + multiple workers with deterministic merge and equivalence checks. Run modes (`distributed.mode`): `single_process` (default), `distributed_simple_exact`, `distributed_mapreduce_exact`, and `distributed_experimental_fast` (requires an explicit experimental acknowledgement in config).

```bash
python scripts/run_distributed_full_pipeline.py --manifest distributed/manifest.json
```

drives an existing manifest through every stage. The full operator walkthrough — manifest creation, pass-1 workers, merge, neg-ctx, pass-2 reduce, candidate selection, discovery, artifact checks — is [docs/h100-8x-full-run-guide.md](docs/h100-8x-full-run-guide.md), with matching configs in `config_examples/`. Design rationale: `agent-planning/multi-device-improvements/`.

### 2 — Per-GPU discovery sharding (`SEED_SHARD`)

Discovery shards by seed across independent processes. A process with shard `i/k` takes candidates where `index % k == i` and writes `outputs/circuits/discovered_circuits.shard<i>.pt` (resumable per shard). Set `discovery.seed_shard` in config, or override per process with the `SEED_SHARD` env var so all GPUs share one `config.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0 SEED_SHARD=0/8 python src/discover_circuits.py
```

See `experiments/045-h100-production/h100_launch.sh` for the production 8-GPU launcher (per-shard retries + logs).

Related knobs: `latents.neg_ctx.backend` (`single_gpu_exact` | `multi_gpu_exact` | `multi_gpu_index_sharded_exact`) and `latents.top_coactivation.reduce_backend` (`single_process` | `target_sharded`).

---

## Configuration Reference

All settings live in `config.yaml` (top-level sections: `weights`, `data`, `hardware`, `sae`, `first_pass`, `latents`, `discovery`, `persist`, `analysis`, plus schema-only `distributed`). The most commonly adjusted keys:


| Key                                                   | Description                                                                                 | Default        |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------- |
| `weights.model_path`                                  | Path to TuringLLM checkpoint                                                                | —              |
| `weights.sae_path`                                    | Directory of SAE weights                                                                    | —              |
| `data.dataset_path`                                   | Directory of tokenised `.npy` shards                                                        | —              |
| `data.n_shards`                                       | Number of shards to process                                                                 | `3030`         |
| `data.batch_size`                                     | Sequences per inference batch                                                               | `512`          |
| `hardware.multi_gpu`                                  | Split layers across all GPUs                                                                | `false`        |
| `hardware.memory`                                     | `efficient` (CPU offload) or `fast` (pin memory)                                            | `efficient`    |
| `hardware.compile`                                    | `torch.compile` for model + SAEs                                                            | `true`         |
| `hardware.parallel_kinds`                             | Dispatch attn/mlp/resid SAE encodes on separate CUDA streams                                | `false`        |
| `hardware.ann_device`                                 | Device for the exact k-NN neg-ctx search: `auto`, `gpu`, `cpu`, or `cuda[:N]`               | `auto`         |
| `hardware.keep_model_loaded_for_neg_ctx`              | High-VRAM mode: skip model/SAE offload+reload around neg-ctx                                | `false`        |
| `sae.encode_backend`                                  | `standard` or experimental `fused_exact_topk`                                               | `standard`     |
| `sae.topk_backend`                                    | `pytorch` (`torch.topk`) or `triton` radix-select kernel                                    | `pytorch`      |
| `first_pass.sae_encode_mode`                          | `streaming` (encode in hooks) or `deferred` (buffer, encode after forward)                  | `streaming`    |
| `latents.top_ctx.n_sequences`                         | Top sequences stored per latent                                                             | `64`           |
| `latents.mid_ctx.n_sequences`                         | Reservoir size per latent (mid-band activations)                                            | `64`           |
| `latents.mid_ctx.band_low_sigma`                      | Lower edge of mid band (mean + N×std)                                                       | `0.5`          |
| `latents.mid_ctx.band_high_sigma`                     | Upper edge of mid band (mean + N×std)                                                       | `1.5`          |
| `latents.mid_ctx.warmup_batches`                      | Batches per component before mid-ctx updates begin                                          | `100`          |
| `latents.neg_ctx.n_sequences`                         | Negative sequences stored per latent                                                        | `64`           |
| `latents.neg_ctx.n_neighbors`                         | k-NN query K per latent (oversampled before pos-ctx filter)                                 | `512`          |
| `latents.neg_ctx.min_pos_ctx`                         | Minimum pos-ctx sequences required to attempt neg retrieval                                  | `8`            |
| `latents.neg_ctx.backend`                             | `single_gpu_exact`, `multi_gpu_exact`, or `multi_gpu_index_sharded_exact`                   | `single_gpu_exact` |
| `latents.neg_ctx.repr_mode`                           | Residual representation mode: `mean_pool` or `last_token`                                   | `mean_pool`    |
| `latents.neg_ctx.max_repr_seqs`                       | Cap on sequences stored in seq_repr (`null` = all)                                          | `200000`       |
| `latents.logit_ctx.n_tokens_per_latent`               | Token prediction entries tracked per latent                                                 | `32`           |
| `latents.logit_ctx.topk_output_tokens`                | Top predicted tokens stored per latent                                                      | `32`           |
| `latents.top_coactivation.n_latents_per_latent`       | Co-activation graph degree                                                                  | `64`           |
| `latents.top_coactivation.n_candidates_per_component` | Candidates extracted per component per sequence during dump                                  | `16`           |
| `latents.top_coactivation.freq_alpha`                 | Exponent for frequency adjustment of co-activation scores                                   | `2.0`          |
| `latents.top_coactivation.mode`                       | Scoring mode: `pmi`, `freq_weighted`, or `raw`                                              | `pmi`          |
| `latents.top_coactivation.reduce_backend`             | `single_process` or `target_sharded` pass-2 reduction                                       | `single_process` |
| `latents.seq_latent_index.enabled`                    | Write per-shard (seq_id, latent_id) index files to `outputs/seq_latent_index/`              | `true`         |
| `discovery.n_seeds`                                   | Seed latents to discover from                                                               | `16`           |
| `discovery.seed_criteria`                             | Criteria used to score/shortlist seed latents (see the long list in `config.yaml`)          | `stratified_random` |
| `discovery.seed_filter.{layers,kinds}`                | Allowlists restricting seed layers/kinds (empty = all)                                      | `[]`           |
| `discovery.seed_shard`                                | Discovery shard `i/k` (env var `SEED_SHARD` overrides)                                      | `0/1`          |
| `discovery.probe_sequence_count`                      | Pos sequences used by discovery attribution                                                 | `64`           |
| `discovery.probe_batch_size`                          | Sequences per grad-enabled instrumented forward pass                                        | `4`            |
| `discovery.eval_sequence_count` / `eval_batch_size`   | Pos sequences / batch size for faithfulness evals                                           | `64` / `16`    |
| `discovery.position_aware`                            | Position-aware allowed sets (union over the seed's causal prefix)                           | `false`        |
| `discovery.floor_source`                              | Mean-ablation floor convention: `posctx`, `negctx`, `global`, or `diverse`                  | `posctx`       |
| `discovery.neg_ctx_eval_max`                          | Neg sequences used to build the per-seed ablation baseline for eval (0 = zero ablation)     | `16`           |
| `discovery.min_faithfulness`                          | Minimum faithfulness to accept a circuit (per-method overrides exist)                       | `0.2`          |
| `discovery.min_active_count`                          | Minimum lifetime firing count for a latent to be a candidate                                | `1`            |
| `discovery.max_neighbors`                             | Default max co-activation neighbors per node (fallback for all methods)                     | `32`           |
| `discovery.methods`                                   | List of discovery methods to run                                                            | see below      |
| `discovery.learned_mask.*`                            | Learned-mask engine (steps, lr, l1_lambda, floors, amplitudes — see below)                  | house recipe   |
| `distributed.mode`                                    | Pipeline run mode (see Distributed & Multi-GPU)                                             | `single_process` |
| `persist.save_workers`                                | Max parallel `torch.save` threads                                                           | `1`            |
| `persist.search_cache_enabled`                        | Build the keyword search cache during the pipeline                                          | `false`        |
| `persist.build_search_cache_after_pipeline`           | Build the cache at end of run (`false` = defer to `scripts/build_search_cache.sh`)          | `false`        |
| `persist.atomic_saves`                                | Write via `.tmp` + atomic rename                                                            | `true`         |
| `analysis.methods`                                    | Post-circuit analyses merged into circuit metadata / `summary.json`                         | 7 methods      |

Each discovery method also has its own config block under `discovery.<method_name>` — see the comments in `config.yaml` for the per-method knobs.

---

## Discovery Methods

21 methods are registered in `METHOD_REGISTRY` (`src/circuit/discovery_window.py`), plus one seed-free method:


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
| `ablation_gradient`                     | Positive-context necessity: active upstream latents whose ablation suppresses the seed              |
| `activation_gradient`                   | `grad(seed peak) × activation` on pos-ctx, no counterfactual baseline                               |
| `hybrid_gradient`                       | Runs counterfactual + ablation gradient, fuses the circuits by FeatureID, re-evaluates              |
| `circuit_tracer_baseline`               | SAE-adapted Attribution Graphs baseline: direct-effects adjacency + influence power iteration       |
| `cluster_contrast` *(seed-free)*        | k-means over neg-ctx embeddings + KL-divergence gradient attribution per cluster                    |


The expansion depth for sparse methods is configured per-method with `coact_depth`, e.g. `[32, 16]` = depth-2 BFS with 32 neighbors at hop 1 and 16 at hop 2.

The gradient methods additionally take an `attribution_mode`: `counterfactual_gradient` supports `local`, `ig_mean`, `restoration`, `ig_restoration`, `ig_negctx`, and the mask modes `mask_contrast` / `mask_negctx` / `mask_inject`; `ablation_gradient` supports `local`, `ig_mean`, `restoration`, `ig_restoration`, and `mask` (abl-mask). The mask modes invoke the learned-mask engine below.

---

## Learned-Mask Circuits (TRI-AMP)

The production discovery recipe is the **learned continuous mask** (`src/circuit/instrument/learned_mask.py`, configured under `discovery.learned_mask`). Instead of reading membership off any single gradient — gradient signs are state-dependent to near-independence — it optimises a soft membership vector `m ∈ (0,1)^N` directly against a reconstruction loss on the seed's pre-activation, with L1 sparsity pressure, and lets the loss decide what stays. Optimisation uses a train slice of the probe set; the final loss is also reported on a held-out slice.

Six objectives share one engine: `pos` (abl-mask — sparsest set reproducing natural firing), `contrast` (also keeps the seed silent on neg-ctx), `negctx` (minimal gate-opening edit that fires the seed on neg-ctx), `raise` (smallest effective brake set), `pin` (pinned-driver twin of `pos`), and `inject` (learned gate + additive delta).

**TRI-AMP MASK** — the recommended configuration — is `objective="pos"` + `mask_floor_source="triple"` + `free_amplitude=true`: reproduce the seed under zero, neg-ctx, and pos-ctx ablation floors *at once*, with a learned per-latent amplitude on top of the gate. It produces **weighted circuits** (a membership set *plus* a coefficient vector — the two must be reported together), null-validated against random same-size sets. Run it via the pipeline by setting the `attribution_mode` mask modes above, or via the production driver in `experiments/045-h100-production/`.

The `discovery.learned_mask` defaults are a frozen house recipe (steps 400, lr 0.05, `l1_lambda` 1e-4 as a per-seed probe anchor); the config-comment block in `src/config.py` documents the calibration law and evidence trail.

---

## Evaluation

Circuits are scored causally with multiple forward passes; the metric depends on the method family.

**Classic MSE faithfulness** (statistical/expansion methods, completeness, minimality):

```
Faithfulness = 1 − MSE(circuit_logits, original_logits)
                   ─────────────────────────────────────
                   MSE(baseline_logits, original_logits)
```

The baseline is either zero-ablation or a mean over neg-ctx sequences, controlled by `discovery.neg_ctx_eval_max`. The mean-ablation floor convention is selected by `discovery.floor_source` (`src/eval/floors.py`).


| Metric                        | Measures                                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Faithfulness**              | How well the circuit alone reproduces the model's output distribution (MSE ratio above)                       |
| **Upstream faithfulness**     | How well the circuit recovers the seed latent's activation; gate for the older gradient methods               |
| **Kind-local faithfulness**   | Faithfulness computed while patching only the target kinds; gate for the sparse-expansion family and `differential_activation` |
| **Counterfactual faithfulness** | Seed activation recovered on neg-ctx under the circuit intervention; gate for `counterfactual_gradient` (paired with `posctx_suppression_score`) |
| **Suppression score**         | Seed suppression when the circuit is ablated on pos-ctx; gate for `ablation_gradient` (`min_suppression_score`) |
| **Ablation faithfulness**     | SFC-style mean-ablation faithfulness at the seed endpoint (`src/eval/ablation_faithfulness.py`); free vs pinned modes, optional error-node sites, and per-latent `keep_scales` for scoring weighted (TRI-AMP) circuits |
| **Node presence**             | Per-node activator presence / inhibitor absence rates on pos-ctx, plus a circuit-sufficiency ratio            |
| **Sufficiency**               | Whether the circuit is sufficient to produce the correct prediction                                           |
| **Completeness**              | Whether removing the circuit degrades the model                                                               |
| **Minimality / pruning**      | Leave-one-out pruning, plus scalable variants: magnitude prune (bisection), recurrence prune, effect-threshold prune |


Acceptance gates are **per-method**: the default threshold is `0.2` (`discovery.min_faithfulness` and per-method overrides), applied to each method's own gate metric — e.g. `counterfactual_gradient` gates on counterfactual faithfulness, `ablation_gradient` on suppression score, sparse-expansion methods on kind-local faithfulness. After acceptance, the post-circuit analyses in `analysis.methods` (layer distribution, edge-weight Gini, node rarity, etc.) are merged into each circuit's metadata and `summary.json`.

---

## Outputs

Artifacts written to `outputs/` by a full pipeline run:


| File                                    | Contents                                          |
| --------------------------------------- | ------------------------------------------------- |
| `latent_stats.pt`                       | Per-latent Welford statistics (mean, std, counts) |
| `top_ctx.pt`                            | Top-activating sequences per latent               |
| `mid_ctx.pt`                            | Mid-band reservoir sequences per latent           |
| `neg_ctx.pt`                            | Exact-k-NN-retrieved negative context sequences   |
| `seq_repr.pt`                           | Pooled residual stream representations            |
| `logit_ctx.pt`                          | Top predicted tokens per latent                   |
| `top_coactivation.pt`                   | Co-activation graph (values + indices)            |
| `seq_latent_index/`                     | Per-shard (seq_id, latent_id) index files (if `latents.seq_latent_index.enabled`) |
| `search_cache.parquet`                  | Keyword-searchable token index (only when cache building is enabled, or via `scripts/build_search_cache.sh`) |
| `candidates.pt`                         | Selected seed latents                             |
| `circuits/discovered_circuits.pt`       | All discovered circuits with scores (`discovered_circuits.shard<i>.pt` under seed sharding) |
| `circuits/summary.json`                 | Per-seed summary of accepted circuits (evals, post-analysis, amplitude stats) |
| `circuits/summary.xlsx`                 | Spreadsheet view of the summary + correlations    |
| `discovery_logs/`                       | Per-seed pass/fail discovery logs                 |
| `cluster_circuits/`                     | Cluster-contrast circuits (if `cluster_contrast` is enabled) |

Distributed runs additionally write per-worker artifacts under `outputs/<run_id>/distributed/`.

---

## Tests

```bash
pytest tests/
```

Test suites cover the pipeline (including the distributed controller/worker/merge path), discovery methods, the learned-mask engine, the eval families, stores, and analyses.

Native extension tests:

```bash
python -m pytest src/native/tests/ -q
```

(covers the top-k kernel, the reducer, and the fused exact top-k encoder)

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
