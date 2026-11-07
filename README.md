# Turing Explorer Circuit Discovery

Code for **weighted circuit masking (WCM)**: for one target sparse-autoencoder (SAE) latent, find a small set of
upstream SAE latents, each with a learned coefficient, that by itself reproduces the target's activation on held-out
contexts. The paper is *What Makes a Latent Fire? Discovering Causal Dependencies Between Sparse Autoencoder Latents*
(Davies and Liu, 2026; draft in `paper/main.tex`).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7%20%7C%202.8%20%7C%202.10-orange)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)

---

## Status

- The method is frozen as **`WCM_VERSION=v4`** (from 2026-11-04). That is the version every production fit runs.
- The final paper results come from the 8×H100 pod run of **2026-11-07**: the three model tracks of
  `experiments/082-public-runs/chain.sh`, plus the baseline arenas A and D of `experiments/107-true-baselines/`.
  Numbers in the current paper draft that predate that run are marked `\pending` or come from earlier protocol
  versions.
- Where an older document (an experiment README, a comment, the paper's hyperparameter table) disagrees with the code,
  the code is current.

---

## The method in brief

The full, code-verified specification (every value with a `file:line` cite) is
`docs/wcm-specification.md`. The method and evaluation protocol is `paper/protocol.md`
(§0 = the current production protocol, WCM v4).

**Endpoint.** The target is an internal SAE latent (layer *L*, site kind *k*), not the model's output. The fit asks
the target's pre-activation at its anchor position (the argmax of its activation in each context) to stay at its
natural value.

**Parameters.** Every latent at every SAE site causally upstream of the target gets:
- a gate logit θ, with gate m = σ(θ/T), and the temperature T annealed from 1 to 0.05. The gate is deterministic, not
  hard-concrete;
- an amplitude ψ, with coefficient α = softplus(ψ). These coefficients α are what "weighted" refers to.

The circuit is the set {θ > 0} together with each member's α. Members and coefficients are always reported together.
The unweighted control arm (`U_`) fixes α = 1.

**Loss (v4).** Each training step runs the model with every non-member ablated in four ways. In every case the circuit
must reproduce the target:
- **Z**: non-members set to zero;
- **A**: non-members at the activating-context mean, under a sparsity-preserving Top-K fill;
- **C**: non-members at the contrast-context mean, under the same fill. Contrast contexts are close contexts where the
  target is verified silent;
- **R** (resample): non-members take a real contrast context's code at the same position.

The terms are weighted Z 1, A 0.25, C 0.25 and R 1.0. On top of them the loss has:
- a rank-keep hinge (weight 1e-1), so the target is not pushed out of the Top-K;
- a censored post-Top-K read (temperature 0.05);
- a flat L1 price λ on σ(θ) for every kept gate, plus a guard so α cannot grow behind a closed gate.

**Optimisation.** AdamW (weight decay 0.05) for 400 steps on 48 training contexts (32 strongest + 16 mid-band), 4
contexts per step. The learning rate warms up over the first 10% of steps, then follows a geometric decay with a
peak of 0.1575. λ = 1e-3 on TuringLLM. On the public models, λ is chosen by a pre-stated size rule on a pilot
(`experiments/082-public-runs/pilot_lambda.py`).

**Evaluation (held-out).** All scores are on contexts the fit never saw, using the target's post-Top-K activation
(`src/eval/wcm_eval.py`):

| Score | Field | Meaning |
|---|---|---|
| Z | `free0_tk` | circuit alone, every non-member at zero |
| A | `freeM_topk_tk` | circuit alone, non-members at the activating-context mean (sparse fill) |
| C | `freeN_topk_tk` | circuit alone, non-members at the contrast-context mean (sparse fill) |
| Necessity | `phi_sup_blind_tk` | how much the target drops when the members are removed from the live model |
| Sufficiency to induce | `phi_cf_alpha_blind_tk` | members injected into silent contrast contexts. Recorded on every row, **not** part of the pass rule |

Each ratio is (x − e)/(a_pos − e). Here a_pos is the natural target and e is the empty-circuit baseline under the same
ablation.

**Pass rule** (`wcm_eval.passes`): Z, A and C all in [0.75, 1.5], **and** necessity ≥ 0.9, on the held-out strongest
contexts. Rows with a vacuous denominator are dropped from the pass rate and counted separately.

**Donor-background test** (`src/eval/donor_test.py`). This test re-scores each circuit with non-members taken from
real donor contexts instead of means. Donors come from three pools: activating, contrast and random corpus windows.
The **resample pass** = the production pass **and** donor recovery for A and C in [0.75, 1.5].

---

## Models and dictionaries

| | TuringLLM | GPT-2 small | Pythia-70m |
|---|---|---|---|
| Model | own 12-layer model, d 1024, 16 heads, vocab 50,304 (`src/model/turingllm.py`) | HF `gpt2` | HF `EleutherAI/pythia-70m` |
| Dictionaries | own Top-K SAEs: 36 (attn, mlp, resid × 12 layers), 40,960 latents, k 128 | OpenAI v5 32k Top-K SAEs, k 32, 4 site kinds (attn, resid_mid, mlp, resid) | EleutherAI sparsify 32k, k 16, 3 site kinds |
| Weights | user-supplied under `models/` (gitignored) | Hugging Face Hub | Hugging Face Hub |
| Corpus | TuringLLM corpus shards under `data/` (gitignored) | OpenWebText (apollo, gpt2-tokenised), 4M 64-token windows | The Pile (uncopyrighted), 4M windows |

The families are defined in `src/model/families.py` (adapters in `src/model/hf_adapters.py`, public dictionaries in
`src/sae/public_bank.py`). Select one with `model.family` in the config, or with `FAMILY=` for the driver.

The baseline arenas in experiment 107 add GPT-2 transcoders with attention SAEs (arena A) and Gemma-2-2B (arena D).

---

## Repository layout

| Path | What it is |
|---|---|
| `src/` | The library: model, SAEs, data pass, WCM engine, evaluation (below) |
| `experiments/` | Numbered experiment folders, each with its own README. The current ones are listed below; older ones are historical. `experiments/README.md` and `experiments/INDEX.md` index them |
| `paper/` | LaTeX draft (`main.tex`), `protocol.md` (method and evaluation protocol; §0 = v4), `tables/` (table builders), `figures/`. Build with `paper/build.ps1` (see `paper/README.md`) |
| `tests/` | Unit and integration tests (pytest) |
| `scripts/` | Pipeline wrappers (`run.sh`, `discover.sh`, …) and pod bootstraps (`pod_bootstrap_public.sh`, `pod_bootstrap_global.sh`; `pod_bootstrap.sh` is the older tri-amp run's) |
| `config.yaml` | Configuration read by the data pass and the pipeline (schema and defaults: `src/config.py`) |
| `config_examples/` | Hardware profiles for the data-pass pipeline |
| `docs/` | Older operator guides and repo walkthroughs (`description.md`, `h100-8x-full-run-guide.md`); they describe the pre-WCM pipeline |
| `RESEARCH.md` | Research framing: the question, the claims, the evidence and the limits |
| `external/circuit-tracer/` | Git submodule, used by the older circuit-tracer comparisons (`git submodule update --init external/circuit-tracer`) |
| `third_party/` | Gitignored. Other papers' official code, cloned at pinned commits for experiment 107 (list in `experiments/107-true-baselines/README.md`) |
| `models/`, `data/`, `outputs/` | Gitignored. TuringLLM weights and SAEs, corpus shards, data-pass stores |
| `data_public/`, `outputs_public/` | Gitignored. Public-model corpora and data-pass stores (built by the chain, stage S1/S3) |
| `dev-notes/`, `agent-planning/` | Gitignored. Local research notes and agent plans |

### `src/` subpackages

| Package | Contents |
|---|---|
| `model/` | TuringLLM, Hugging Face adapters for GPT-2 / Pythia, model families, hooks, tokenizer, inference engine |
| `sae/` | SAE bank (Top-K, dense / JumpReLU, public banks), Triton and fused Top-K encoders |
| `data/` | Corpus shard loader |
| `pipeline/` | The data pass: pass 1 (statistics, contexts), exact k-NN negative contexts, pass 2 (co-activation), candidate selection, persistence; `distributed/` for multi-worker runs |
| `store/` | On-disk stores read by the fits: latent stats, top / mid / negative contexts, sequence representations, logit context, co-activation graph, circuits, search cache |
| `circuit/` | `instrument/learned_mask.py` (the WCM engine), patchers and attribution code; `discovery/` (discovery methods; WCM enters through `ablation_gradient.py`); `protocol_contexts.py`, `context_split.py`, `heldout_rule.py` (training and held-out contexts) |
| `eval/` | `wcm_eval.py` (Z / A / C / necessity / induce and the pass rule), `donor_test.py`, `floors.py`, `ablation_faithfulness.py`, `wcm_specificity.py`, older faithfulness and pruning families; `sae/` (SAE quality) |
| `analysis/` | Offline analysis and plotting (`style.py` = the figure theme) |
| `export/explorer/` | Builds the data bundle for the Turing Explorer web viewer |
| `display/`, `observability/`, `utils/`, `debug/` | Terminal viewers, logging and timing, negative-context selector, profiling scripts |
| `native/` | C++ / CUDA extensions (Welford statistics, reservoir sampler, co-activation reducer, fused Linear+ReLU and Top-K) |

Entry points at the top of `src/`: `main.py` (full data-pass pipeline), `discover_circuits.py`, `display_latents.py`,
`search_latents.py`, `build_search_cache.py`, `ablation_sensitivity.py`, `config.py`, `hardware.py`.

### Current experiments

| Folder | Role |
|---|---|
| `062-h100-protocol-v1/` | **The driver** (`driver.py`): fits and scores circuits, one GPU per process (`launch.sh` shards across GPUs). Also companions: `rescore_alpha1.py`, `stable_core.py`, `merge.py` |
| `082-public-runs/` | **The pod chain** (`chain.sh`) for all three tracks, target frames, λ pilot rule, progress tools |
| `094-tl-v2-records/` | The TuringLLM 8,000-target list and its fresh, deduplicated, group-split context records |
| `081-task-circuits/` | WCM with the model's output as the endpoint, on GPT-2 behavioural tasks (the engine arena D reuses) |
| `083-v1-comparison/` | Our own baseline rows (IG / AtP rankings, fitted amplitudes) on rebuilt held-out sets. Its adapted versions of other papers' methods are withdrawn |
| `084-harder-nulls/` | Harder random-circuit nulls (members drawn near the anchor, coefficients fitted as WCM fits them) |
| `103-donor-stage/`, `104-v3-cross-model/` | The donor test as a native scorer and pod stage; the v3 port, bit-identity check and cross-model check |
| `107-true-baselines/` | Other papers' methods as published, on their own substrates, with WCM run there too. Arena A: Ge et al. and Dunefsky et al. on GPT-2 transcoders + attention SAEs. Arena D: attribution graphs (circuit-tracer) and Caples et al. on Gemma-2-2B. Arenas B and C: CircuitLasso and SFC. `latent_pilot/` holds the evidence for v4. `specs/` documents every method |
| `108-batch-size/` | Does a larger effective batch (gradient accumulation) give better or more stable circuits? Production is unchanged |

Other folders feed these: `049` (the frozen TuringLLM scorer `amp_eval_pass_v2.py`), `059` (context-pool harness),
`069` / `070` (public data pass and adapters), `073` (strict / resample evaluation), `085` / `086` (held-out leakage
audit and rebuild). Folders before about 058 are earlier method versions and stay as a record.

---

## Installation

Requirements: Linux (on Windows use WSL 2), Python 3.12, and a CUDA / PyTorch stack matching the machine. The
project's own environment is `.venv` at the repo root, created inside WSL.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt            # pick the file for your CUDA runtime, see below
```

| File | Torch stack |
|---|---|
| `requirements.txt` | `torch 2.10.0+cu130`, `torchvision 0.25.0+cu130`, `torchao 0.16.0` (local CUDA 13 profile) |
| `requirements-cu128.txt` | `torch 2.8.0+cu128`, `torchvision 0.23.0+cu128`, `torchao 0.13.0` (hosted H100 images) |
| `requirements-cu126.txt` | `torch 2.7.1+cu126`, `torchvision 0.22.1+cu126`, `torchao 0.12.0` (fallback) |

All three pin `transformers 5.1.0`. On a pod, `scripts/pod_bootstrap_public.sh` builds the venv over the image's
torch, with its own pins.

**Native extensions** (used by the data pass; the chain checks them at stage S2):

```bash
cd src/native && python setup.py build_ext --inplace && cd ../..
python -m pytest src/native/tests/ -q
```

`nvcc --version` should match `torch.version.cuda`. If it does not, set `CUDA_HOME` and `PATH` to the matching
toolkit.

**Paths.** For TuringLLM, set `weights.model_path`, `weights.sae_path` and `data.dataset_path` in `config.yaml`. The
public families need no local weights: models and dictionaries come from the Hugging Face Hub.

**Baseline environments (experiment 107 only).** Each baseline runs its official code in its own venv, built by the
arena's `setup_env.sh` against the clones in `third_party/`:

| venv | Used for | Built by |
|---|---|---|
| `ge107` | Ge et al. (OpenMOSS Language-Model-SAEs v0.1.0), Python 3.10 | `107-true-baselines/arena_a/setup_env.sh` |
| `tc107` | Dunefsky et al. (`transcoder_circuits`), Python 3.10 | `107-true-baselines/arena_a/setup_env.sh` |
| `ct107` | circuit-tracer (attribution graphs) | `107-true-baselines/arena_d/setup_env.sh` |
| `caples107` | Caples et al. (`ScalableSAECircuits`) | `107-true-baselines/arena_d/setup_env.sh` |

These venvs default to `~/venvs/`. Their exact pins are in each script's header and in the arena READMEs. Arena D
needs a Hugging Face token for `google/gemma-2-2b`. Export it in the shell, and never write it to a file.

---

## Running

All commands run from the repo root with `PYTHONPATH=src` (the wrapper scripts set it).

### Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q           # tests/ (pytest.ini: testpaths = tests, pythonpath = src)
CUDA_VISIBLE_DEVICES="" python -m pytest --collect-only -q   # collection check: 1,937 tests on 2026-11-06
```

Some experiment folders keep their own CPU tests next to the code (e.g. `experiments/082-public-runs/test_082_*.py`,
`experiments/094-tl-v2-records/test_094_*.py`). Run them by path. GPU-marked tests skip on CPU.

### Fit and score circuits (the driver)

`experiments/062-h100-protocol-v1/driver.py` takes its settings from environment variables. Production needs
`WCM_VERSION=v4`. The driver's own default is `v1`, so a run without that variable is not production.

```bash
CUDA_VISIBLE_DEVICES=0 SHARD=0/1 MODE=main WCM_VERSION=v4 FAMILY=gpt2-small \
  TARGETS=<targets.txt> OUT=<out_dir> LIMIT=1 STEPS=20 \
  PYTHONPATH=src python experiments/062-h100-protocol-v1/driver.py
```

- `MODE=main` fits each target, then scores it on the held-out strongest and mid-band sets.
- `MODE=sweep` runs the λ grid (`W_<λ>` = WCM, `U_<λ>` = unweighted), selected with `SWEEP_ARMS`.
- `LIMIT` and `STEPS` are for smoke tests only.
- A public family needs its data-pass stores (`DATA`, `STORES`, default `data_public/<family>`,
  `outputs_public/<family>`).
- Outputs are resumable per target. They are the context records, the circuits, the scored jsonl rows, the training
  curves (`train.shard<i>.jsonl`), and a `WCM_VERSION` marker. The driver refuses to mix versions in one `OUT`.

The full list of environment variables is in the driver's docstring.

### Local smoke of the pod chain

```bash
DRY_RUN=1 TL_TRACK=1 bash experiments/082-public-runs/chain.sh     # print every command; runs nothing (CPU)
LIMIT=2 TL_TRACK=1 bash experiments/082-public-runs/chain.sh       # GPU smoke on the 069 smoke stores, 20 steps
```

Point the smoke at a fresh `WORK=` directory. An `OUT` directory marked with an older version is refused.

### The data pass (stores)

The fits read the stores written by the data pass:
- **TuringLLM:** `python src/main.py` (or `scripts/run.sh`). It runs pass 1 (latent statistics, top / mid contexts),
  the exact k-NN negative-context step, pass 2 (co-activation), and candidate selection, into `outputs/`.
- **Public models:** `experiments/069-public-data-pass/run_data_pass.py`, driven by the chain's stage S3 (pass 1 and
  negative contexts only).

The older inspection tools still work against `outputs/`: `display_latents.py`, `search_latents.py`,
`discover_circuits.py` and `scripts/list_circuits.sh`. `scripts/run_distributed_full_pipeline.py` with
`docs/h100-8x-full-run-guide.md` covers the distributed data pass.

---

## The pod run

`experiments/082-public-runs/chain.sh` runs everything on one 8-GPU pod. Its header documents every stage and
environment variable.
1. Bootstrap the public tree with `scripts/pod_bootstrap_public.sh`.
2. Bootstrap a second checkout for TuringLLM with `ROOT=/root/turing-tl TL_TRACK=1 bash scripts/pod_bootstrap_global.sh`.
3. Launch the chain with `TL_TRACK=1`, plus `ARENA_A=1` / `ARENA_D=1` for the arenas.

The step-by-step guide is `docs/reproducing-the-runs.md`.

| Track | Stages |
|---|---|
| Pythia-70m | S1 corpus → S2 native check → S3 data pass → S4 records + target frame (512) → S5 λ pilot → S6 main + α = 1 + nulls + stable core + unweighted arm + resample + donor test |
| GPT-2 small | S1 → S2 → S3 → S7 records + frame (512) → S8 λ pilot → S9 main + arms (no unweighted arm) |
| both | S10 pack, md5-verify on the volume |
| TuringLLM | T0 artefact check → T1 corpus scan → T2 fresh records (094; an early gate on 64 targets, then the rest, with quarantine) → T3 main fits on 8,000 targets + α = 1 → T4 λ sweep (190 targets, 8 WCM + 5 unweighted arms) → T5 stable core → T6 harder nulls → T7 own baselines (083) → T8 case-study and gallery pack → (T9 off) → T10 without-R arm (`v4nr`) → T11 donor test of the main circuits → T12 donor test of the sweep |
| Arena A (after the main chain) | `107-true-baselines/arena_a/run_pod.sh`, P0–P7: Ge + Dunefsky vs WCM v4 (necessity term on, weight 0.3) on 48 GPT-2 targets, 5 λ. It has a hard time limit |
| Arena D (after arena A) | `107-true-baselines/arena_d/pod_stage_d.sh`, P0–P5: attribution graphs and Caples masks vs behavioural WCM on 4 Gemma-2-2B agreement tasks. It runs WCM `v2` with arena C's settings, not v4 (spec §6.1) |

**Checkpoints:**
- C2: every record check must pass.
- C3: the main public fit waits until a λ is adopted (`echo <λ> > <STATE>/lam_<model>`).
- Every stage writes a tar + md5 and verifies it on the results volume before it is marked done. Reruns skip finished
  stages and resume per target.

---

## Configuration

`config.yaml` configures the data pass and the older pipeline. `src/config.py` holds the schema and defaults. Unknown
keys are errors (`extra='forbid'`). **The production WCM settings are not read from `config.yaml`.** They come from
the driver (`WCM_VERSIONS`, `WCM_LR_MULT` and `WCM_RANK` in `driver.py`) on top of `LearnedMaskConfig` in
`src/config.py`. Spec §2–§3 lists the values that result.

Commonly changed keys (checked 2026-11-06):

| Key | `config.py` default | repo `config.yaml` |
|---|---|---|
| `weights.model_path`, `weights.sae_path` | required | `models/TuringLLM/...` |
| `data.dataset_path` | required | `data` |
| `data.n_shards` | 256 | 3030 |
| `data.batch_size` | 512 | 512 |
| `model.family` | `turingllm` (`pythia-70m`, `gpt2-small`) | not set |
| `model.dtype` | None (= float32; public families only) | not set |
| `hardware.multi_gpu` / `memory` / `compile` | false / `efficient` / true | same |
| `hardware.ann_device` | `auto` (device for the exact k-NN search; "ann" is a legacy name) | `auto` |
| `sae.encode_backend` | `standard` | `standard` |
| `sae.topk_backend` | `triton` | `pytorch` |
| `first_pass.sae_encode_mode` | `streaming` | `streaming` |
| `latents.{top_ctx,mid_ctx,neg_ctx}.n_sequences` | 64 | 64 |
| `latents.neg_ctx.n_neighbors` / `backend` / `max_repr_seqs` | 512 / `single_gpu_exact` / 200000 | same |
| `latents.top_coactivation.mode` | `freq_weighted` | `pmi` |
| `latents.seq_latent_index.enabled` | false | true |
| `discovery.n_seeds` | 128 | 16 |
| `discovery.probe_sequence_count` / `probe_batch_size` | 16 / 16 | 64 / 4 |
| `discovery.eval_sequence_count` / `eval_batch_size` | 16 / 16 | 64 / 16 |
| `discovery.context_protocol` | — | `legacy` (`v1` = the protocol's contexts) |
| `discovery.seed_shard` | `0/1` (env `SEED_SHARD` overrides) | not set |
| `discovery.neg_ctx_eval_max` / `min_faithfulness` | 16 / 0.2 | 16 / 0.2 |
| `persist.search_cache_enabled` / `build_search_cache_after_pipeline` | true / true (the second is forced false in distributed modes) | false / false |
| `persist.atomic_saves` | true | true |

For every other key, see `src/config.py`. Its comment blocks also record why each learned-mask default was chosen.
Some of those comments predate v4; spec §8 lists them.

---

## Licence and citation

Apache License 2.0 (`LICENSE.txt`). Code in `third_party/` and `external/` keeps its own licence. One clone,
`ScalableSAECircuits`, has no licence and is not redistributed.

If you use this code, please cite the paper (preprint in preparation; the details will be updated on release):

```
Daniel Davies and Weibo Liu. What Makes a Latent Fire? Discovering Causal Dependencies Between
Sparse Autoencoder Latents. 2026.
```

---

## Not yet verified

- GPU bit-reproducibility of v4 fits and invariance under MPS co-tenancy are not guaranteed in code (spec §3).
- The full test suite was only collected, not run, for this README (2026-11-06).
- The final public-model λ is decided on the pod (checkpoint C3), so it is not given here.
