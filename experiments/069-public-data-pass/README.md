# 069 — public-model data pass (DAN-37)

The data-collection passes that give Pythia-70m and GPT-2 small the artefact set TuringLLM has, so protocol-v1 contexts
(`src/circuit/protocol_contexts.py: build_record`) can be built for every target of the planned H100 runs. Contexts are
mined by our own pipeline (`src/pipeline/`), not downloaded. The real corpora are not on disk; everything here was
proven end to end on CPU on small wikitext-103 corpora built from the local HF cache.

## Files

| File | What it does |
|---|---|
| `prepare_corpus.py` | Tokenises a text source with the family's tokenizer into the loader's shard format; dedupes exact duplicate windows; writes `corpus_manifest.json` |
| `common.py` | `configure()`: points the global config at a family + shard dir (must run before `store.*` is imported); store-grid check; path guard (refuses `data/`, `outputs/`, `../Runs/`) |
| `run_data_pass.py` | pass 1 → neg_ctx → pass 2 → candidates, the functions `run_pipeline.py` calls (no discovery), into a per-family root; writes `data_pass_manifest.json` |
| `check_records.py` | `build_record` on a few targets, then independent checks of every pool (see "End-to-end result") |
| `dup_report.py` | Duplicate-rate report by content (corpus, top store, mid store, mid-vs-strong pool); optional TuringLLM reference |
| `estimate_corpus.py` | Corpus size needed for full 64/64 stores, from measured firing rates; calibrated on TuringLLM's own stores |
| `freq_scan.py` | Per-latent sequence firing rates at a few sites over a larger corpus (tail of the estimate) |
| `neuronpedia_check.py` | GPT-2: Neuronpedia's cached contexts re-encoded through our adapter (offline; cached records only) |
| `gpu_native_check.py` | For the H100: latent_stats_cuda vs PyTorch fallback, GPU vs CPU top store, on a public family |

Roots: shards in `data_public/<family>/`, artefacts in `outputs_public/<family>/` (both git-ignored). Nothing in this
experiment reads or writes TuringLLM's `data/`, `outputs/` or `../Runs/` except the read-only reference measurements
(`--reference*` flags).

## Shard format (`prepare_corpus.py`)

`shard_<N>.npy`, int64, flat `[lead, t1..t64, -1, lead, t1..t64, -1, ...]`: exactly what `src/data/loader.py` reads with
`skip_first_token=True` (the default every consumer uses), so a loaded sequence is `t1..t64`. Both SAE families were
trained with `prepend_bos=False` and context 64 (`model/families.py`; 033 and 053 used 64-token windows, no BOS), so the
sequence carries no BOS. `lead` = the tokenizer's BOS/EOS id (a reader that keeps the first token sees the prepend-BOS
form). TuringLLM's shards hold `[lead, BOS, 63 tokens]`; its sequences start with BOS because TuringLLM was trained that
way. Sequence ids are 1-based and positional over the sorted shard list, as for TuringLLM; 8,192 sequences per shard by
default. Documents are concatenated with EOS between them (standard packing; `--no-doc-sep` gives 033/070's plain
concatenation) and cut into consecutive, non-overlapping 64-token windows. Exact duplicate windows are
dropped as they are cut (`--no-dedupe` keeps them), so contexts are deduplicated by sequence once, at the source.
`provenance/shard_<N>.docs.npy` records each sequence's source document. The loader treats every top-level `*.npy` as a
shard, so nothing else may be written at the top level.

Sources: local `*.parquet`, `*.jsonl`, `*.jsonl.zst` (needs `pip install zstandard`), `*.txt`, or an HF dataset repo
(`--hf-dataset REPO --hf-pattern GLOB`, via `snapshot_download`; this is where a real run would download). `--doc-mode
row` (each row is a document) for OpenWebText / Pile, `wikitext` for the smoke corpus. The manifest records the corpus
name, licence (a placeholder until confirmed), source files and sizes, tokenizer, format, counts, the duplicate rate,
per-shard sequence counts and bytes, the command line and the git commit.

## What each pass produces

| Pass | Artefact | Shape (Pythia / GPT-2) | Used by |
|---|---|---|---|
| 1 | `latent_stats.pt` | per-token and per-sequence Welford stats, `[C, 32768]`, C = 18 / 48 | mid band, PMI, candidates |
| 1 | `top_ctx.pt` | 64 strongest sequences per latent (mean activation over 64 tokens), `[C, 32768, 64]` | strongest pool, neg_ctx, pass 2 |
| 1 | `mid_ctx.pt` | reservoir (Algorithm R, `mid_reservoir` kernel) of sequences scoring in μ + [0.5, 1.5] σ, after `warmup_batches` | mid-band pool |
| 1 | `seq_repr.pt` | mean-pooled last-layer resid per sequence, fp16, `[n+1, 512 / 768]`, capped at 200k sequences | close contrast ranking |
| 1 | `logit_ctx.pt` | top next-token predictions per latent | candidates (logit criteria) |
| 1 | `seq_latent_index/` | per-shard (sequence, latent) index, top 12 per component | analysis |
| ANN | `neg_ctx.pt`, `neg_ctx_stats.json`, `global_negctx_ids.pt` | 64 nearest non-activating sequences per latent with ≥ 8 positives; their union is the close selector's candidate pool | contrast set |
| 2 | `top_coactivation.pt` | top-64 PMI co-activating latents per latent (replays the top-store sequences; `top_coactivation_reduce` kernel) | candidates |
| — | `candidates.pt` | `stratified_random` seeds (`config.discovery.n_seeds`) | target frame |

GPT-2's site activations are the SAE input (layer-normalised x), so its `seq_repr` pools the normalised last-layer resid;
the selector's own forward reprs use the same space, so the close ranking is consistent.

## src changes (family-aware; TuringLLM unchanged)

| File | Change |
|---|---|
| `model/families.py` | `StoreGrid` + `public_store_grid()`: the family's (n_layer, kinds, d_sae, k, d_model); `None` on TuringLLM |
| `store/context.py`, `latent_stats.py`, `logit_context.py`, `top_coactivation.py`, `seq_repr.py` | override their grid from `public_store_grid()` only when it is not None (C = n_layer × len(kinds); d_sae; k for PMI; d_model for `seq_repr`). The TuringLLM path still constructs `TuringLLMConfig()` / `SAEConfig()` as before (tests that patch those keep working) |
| `store/neg_ctx/backends.py` | components walked = the family grid (`_num_components()`) |
| `circuit/feature_selection.py` | kinds for the seed filter, cross-kind criteria and the summary from the family (4 on GPT-2) |
| `sae/public_bank.py` + `config.model.skip_missing_saes` (default False) | a dictionary that cannot be loaded becomes a missing site (`bank.missing_sites`, `bank.has_site`) instead of an error |
| `pipeline/encoding.py` | skips missing sites (only for a bank with `skip_missing=True`); their store rows stay empty |
| `store/context.py::_update_top` | a zero-score sequence is stored as the empty slot (id 0). **Bug found here:** with fewer firing sequences than slots, the row was padded with whatever zero-score ids `torch.topk`'s tie-break kept: empty slots on CUDA (TuringLLM's store holds no id with value 0: checked), but non-firing sequence ids on CPU (the first Pythia smoke run had 23M of them in 458k rows, which would have entered strongest pools). A no-op for every store whose zero-score slots already held id 0 |

Already family-aware before this issue: `Inference` / `SAEBank` dispatch, `bank.kinds` / `n_layer` in `first_pass`,
`encoding`, `NegContextSelector` and `protocol_contexts`; `upstream_sites` (layers with no upstream site: Pythia attn 0
and mlp 0, GPT-2 attn 0) via `bank.same_layer_upstream`, which `discovery.skip_no_upstream` uses to drop them from the
frame. The data passes keep stores for those sites (harmless; they are upstream nodes for other targets).

Native kernels: `latent_stats_cuda` and `top_coactivation_reduce` take the grid as arguments; `mid_reservoir` takes
the reservoir width; none assumes TuringLLM's shapes. `TURING_NATIVE_CPU_ONLY` is a build-time flag
(`src/native/setup.py`); at run time `latent_stats_cuda` is used only for CUDA tensors (the CPU smoke ran the PyTorch
fallback), while `mid_reservoir` and `top_coactivation_reduce` are CPU extensions and ran here on Pythia and GPT-2
shapes. `gpu_native_check.py` covers the CUDA kernel.

## End-to-end result (CPU, wikitext-103 train shard 0)

Shared 16-core CPU (other jobs running), float32, batch 128. Commands as in "The real run" with
`--max-sequences 4096 --seqs-per-shard 1024` (Pythia) / `2048` (GPT-2), `--batch-size 128 --mid-warmup 8` / `4`, and
`--skip-missing-saes` for GPT-2.

| | Pythia-70m | GPT-2 small |
|---|---|---|
| corpus | 4,096 × 64 = 262,144 tokens, 4 shards, 0 duplicate windows | 2,048 × 64 = 131,072 tokens, 2 shards, 0 duplicates |
| grid | 18 components (6 × attn/mlp/resid) × 32,768, k 16, d_model 512 | 48 (12 × attn/resid_mid/mlp/resid) × 32,768, k 32, d_model 768 |
| missing SAE sites | none | layer 10 × 4 (not cached): skipped, rows empty, listed in the run manifest |
| pass 1 / neg_ctx / pass 2 | 13 m 09 s / 1 m 22 s / 10 m 32 s | 38 m 01 s / 7 m 36 s / 23 m 38 s |
| `latent_stats.pt` | 23.6 MB | 62.9 MB |
| `top_ctx.pt` / `mid_ctx.pt` | 226.5 / 309.1 MB | 604.0 / 824.2 MB |
| `seq_repr.pt` / `logit_ctx.pt` | 4.2 / 155.7 MB | 3.1 / 415.2 MB |
| `neg_ctx.pt` / `neg_ctx_stats.json` / `global_negctx_ids.pt` | 302.0 / 2.9 / 0.03 MB | 805.3 / 8.7 / 0.02 MB |
| `top_coactivation.pt` / `candidates.pt` | 304.4 / 0.002 MB | 811.6 / 0.002 MB |
| `seq_latent_index/` | 4 files, 7.1 MB | 2 files, 8.7 MB |
| neg_ctx rows filled (≥ 8 positives → 64/64 negatives) | 362,328 of 589,824 | 1,086,874 of 1,572,864 |

Store sizes scale with the grid, not the corpus (seq_repr and the id maps excepted).

**Protocol-v1 records** (`check_records.py`; `records_check.json` in each root). `build_record` ran unchanged on the
stores; every usable record passed every independent check: pool ids unique and disjoint, pool tokens equal the
loader's sequence, the target fires in every strongest and mid-band context (peaks re-encoded, equal to the builder's
to 1e-4), and every contrast context is verified silent (target outside Top-K at all 64 positions of a fresh forward;
max target activation 0.0) and is none of the target's own top / mid ids.

| target | fires in | strongest | mid-band | train (D / B) | contrast (train / held) |
|---|---|---|---|---|---|
| Pythia 2.resid.9392 | 32 % of seqs | 64 | 64 | 32 / 16 | 64 (48 / 16) |
| Pythia 1.mlp.18087 | 24 % | 64 | 64 | 32 / 16 | 64 |
| Pythia 1.resid.8622 | 28 % | 64 | 64 | 32 / 16 | 64 |
| Pythia 2.mlp.17562 | 11 % | 64 | 34 | 32 / 16 | 64 |
| Pythia 1.attn.5489 | 11 % | 64 | 20 | 32 / 15 | 64 |
| Pythia 3.attn.17283 | 12 % | 64 | 19 | 32 / 14 | 64 |
| Pythia 0.resid.24152 (random) | 2.7 % | 64 | none (thin, arm A) | 48 strongest-train | 64 |
| Pythia 1.resid.26277 (random) | 0.24 % | 10 | none (arm A) | 8 strongest-train | 64 |
| GPT-2 3.resid.26481 | 52 % | 64 | 64 | 32 / 16 | 64 |
| GPT-2 1.resid.9750 | 45 % | 64 | 62 | 32 / 16 | 64 |
| GPT-2 8.resid.17415 | 22 % | 64 | 33 | 32 / 16 | 64 |
| GPT-2 9.resid_mid.14337 | 17 % | 64 | 11 | 32 / 8 | 64 |
| GPT-2 3.attn.21014 | 16 % | 64 | 14 | 32 / 10 | 64 |
| GPT-2 8.attn.9570 | 99.8 % | 64 | 64 | 32 / 16 | **4** (3 / 1): almost nothing in 2,048 sequences is silent |
| GPT-2 1.resid.23840 (random) | 0.5 % | 11 | none (arm A) | 8 strongest-train | 64 |
| GPT-2 4.resid_mid.19721 (random) | 0.05 % | 1 | — | skipped: `too_few_contexts` | — |

Thin mid pools and the 4-context contrast set are corpus-size effects (2-4k sequences). The mid store admits a sequence
against the running band, so with a 4-8-batch warm-up some mid contexts sit outside the final μ + [0.5, 1.5] σ (one
GPT-2 row has none inside); 100 warm-up batches of 512+ on a real corpus make this negligible (TuringLLM's same rule).

## Duplicate-rate report

`dup_report.py` → `duplicate_report.json`. A repeat is a stored context whose 64 tokens equal an earlier context of the
same latent's row (ids are unique per row by construction, so a repeat is the same text under two ids).

| | corpus windows duplicated | top store: contexts repeated / rows with a repeat | mid store | mid pool text already in strongest pool |
|---|---|---|---|---|
| Pythia smoke (deduped at prep) | 0 of 4,096 | 0 / 0 (474,680 rows) | 0 (380,293 rows) | 0 |
| GPT-2 smoke (deduped at prep) | 0 of 2,048 | 0 / 0 (1,336,139 rows) | 0 (1,113,403 rows) | 0 |
| **TuringLLM (reference, read-only: `data/`, `outputs/`)** | **26.5 %** (18.24M unique of 24.82M) | **23.3 % / 96.9 %** (row median 23 %, p90 39 %) | 0.2 % / 6.4 % | 0.002 % |
| Neuronpedia GPT-2 (23 cached records) | — | mean 18.5 %, range 0-40 % of its ≤ 45 contexts | — | — |

The smoke corpora are wikitext, which has no exact duplicate 64-token windows in its first 262k tokens, so the zeros
mostly show the dedupe path and the report work; the real corpora will show the rate the dedupe removes in
`corpus_manifest.json` (`duplicate_window_rate`). The TuringLLM row is the finding: a quarter of its corpus windows are
exact duplicates, and the top store (the protocol's strongest pool) is 23 % repeated text, confirmed by exact token
comparison on 40 random rows (26.7 %). The mid store is barely affected (reservoir sampling rarely draws the same text
twice). Because the protocol splits the strongest pool into train / held-out by peak rank, identical contexts (identical
peaks, adjacent ranks) can land on both sides: 062's held-out strongest scores may include exact copies of training
contexts. Not fixed here (TuringLLM's data and stores are read-only for this issue); see "Decisions".

## Neuronpedia (independent check, offline)

`neuronpedia_check.py` → `outputs_public/gpt2-small/neuronpedia_check.json`, on the 24 records 053 cached in
`~/neuronpedia_cache` (nothing fetched; one record has no contexts). Neuronpedia's contexts are 63 display tokens. Our
adapter + bank reproduce Neuronpedia's per-context peak to a median 1.5 % (no BOS, our and the SAEs' convention; 4.4 %
with a BOS prepended), so site, normalisation and dictionary agree. Per-token values do not match exactly (max |diff|
3-4 typical): Neuronpedia's windows are excerpts of longer sequences, so the model saw a different prefix. Three
position-sensitive latents (0.resid.27810, 2.resid.19132, 5.resid.4434) agree only with a BOS prepended, i.e.
Neuronpedia ran with a BOS. Duplicates: mean 18.5 % of Neuronpedia's contexts, up to 40 % (the issue's 3-33 %). Even
the 2,048-sequence smoke store holds 64 contexts for 16 of the 23 latents (Neuronpedia: ≤ 45). For the real run,
`neuronpedia_check.py` is the independent check to repeat (more latents need cached or fetched records; fetching is a
network call, Daniel's choice).

## The real run

### Commands

```bash
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"   # or the pod's repo root
export PYTHONPATH=src
# 1. corpus (one per model; <CORPUS> = local OpenWebText / Pile files, or --hf-dataset REPO --hf-pattern GLOB)
.venv/bin/python experiments/069-public-data-pass/prepare_corpus.py --family gpt2-small \
    --out data_public/gpt2-small --input <OPENWEBTEXT_DIR_OR_FILES> --doc-mode row --max-sequences <N_GPT2> \
    --source-name "OpenWebText (<version>)" --licence "<confirmed licence>"
.venv/bin/python experiments/069-public-data-pass/prepare_corpus.py --family pythia-70m \
    --out data_public/pythia-70m --input <PILE_JSONL_ZST_FILES> --doc-mode row --max-sequences <N_PYTHIA> \
    --source-name "The Pile (<subset/version>)" --licence "<confirmed licence>"
# 2. GPU sanity of the native kernels on the public grids (minutes)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/069-public-data-pass/gpu_native_check.py \
    --family pythia-70m --data data_public/pythia-70m
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/069-public-data-pass/gpu_native_check.py \
    --family gpt2-small --data data_public/gpt2-small
# 3. data passes, one model per GPU (100 warm-up batches is config.yaml's default; --dtype: see Decisions)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/069-public-data-pass/run_data_pass.py --family pythia-70m \
    --data data_public/pythia-70m --out outputs_public/pythia-70m --batch-size 2048 --dtype bfloat16 \
    --n-seeds <FRAME_SIZE>
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/069-public-data-pass/run_data_pass.py --family gpt2-small \
    --data data_public/gpt2-small --out outputs_public/gpt2-small --batch-size 2048 --dtype bfloat16 \
    --n-seeds <FRAME_SIZE>
# 4. reports
for F in pythia-70m gpt2-small; do
  .venv/bin/python experiments/069-public-data-pass/dup_report.py --family $F --data data_public/$F --out outputs_public/$F
  .venv/bin/python experiments/069-public-data-pass/estimate_corpus.py --family $F --data data_public/$F --out outputs_public/$F \
      --real-warmup-seqs 204800
  .venv/bin/python experiments/069-public-data-pass/check_records.py --family $F --data data_public/$F --out outputs_public/$F --n-auto 16
done
.venv/bin/python experiments/069-public-data-pass/neuronpedia_check.py --data data_public/gpt2-small --out outputs_public/gpt2-small
```

Stages can be rerun on their own (`--stages negctx,pass2,candidates` loads pass 1's artefacts from `--out`). The fit
side then needs `config.model.family`, `data.dataset_path` / `n_shards` pointing at the family's shards and the stores
loaded from `outputs_public/<family>` (see `common.configure` / `common.load_stores`).

Batch size: `--batch-size 2048` (or 4096 under bf16, as TuringLLM ran): the SAE pre-activations of one batch are
`batch × 64 × 32,768` floats per site (34 GB at 4,096 in float32).

### How much corpus (1,024 targets, 64 strongest + 64 mid-band each)

A latent that fires in a fraction p of 64-token sequences, with a fraction f of its firing sequences in the mid band,
fills its top store at N ≥ 64 / p and its mid store at N ≥ W + 64 / (p f) (W = warm-up sequences). `estimate_corpus.py`
reads p and f per latent from a finished pass; on TuringLLM's own stores it predicts 90.3 % of latents with both
stores full at its 24.8M sequences, and 90.3 % is what the mid store measures, so the model is calibrated.

Measured f (median, all three models): 0.09-0.13, not the 0.24 a normal distribution would give (per-sequence scores
are heavy-tailed). Share of all latents (a uniform draw = the `stratified_random` frame) with both stores full:

| sequences N (tokens = 64 N) | TuringLLM (measured p) | Pythia 2.mlp | Pythia 4.resid | Pythia 1.attn |
|---|---|---|---|---|
| 0.5M (32M) | | 82 % | 68 % | 43 % |
| 1M (64M) | 86 % | 89 % | 77 % | 51 % |
| 2M (128M) | 88 % | 94 % | 86 % | 57 % |
| 5M (320M) | 90 % | ≥ 96.8 % | ≥ 90.9 % | ≥ 62 % |
| 24.8M (1.6B, TuringLLM's) | 90 % | | | |

The Pythia columns are `freq_scan.py` over 32,768 wikitext sequences at three sites (`freq_scan.json`): latents seen in
fewer than 5 of them count as not full, so the columns are lower bounds, valid up to N ≈ 3M. Pythia 1.attn has 29 %
latents silent in 32,768 sequences (TuringLLM's attention sites have the same dead tail: its kind-0 p10 is 0). The
whole-dictionary passes (4,096 / 2,048 sequences) resolve p only for 65 % (Pythia) / 80 % (GPT-2, layer 10 excluded) of
latents; for those, full stores need a median 115k / 84k and a p90 of 334k / 178k sequences.

Reading: the median latent needs ~0.1M sequences; the tail sets the size. Beyond ~2-5M sequences coverage grows by ~1
point per doubling on TuringLLM (its last 10 % are dead or fire in < 1 in 10^5 sequences). Pythia's k = 16 makes its
latents sparser than TuringLLM's (k = 128), so at equal N it covers fewer, mostly at attention sites. Suggested:
**10M sequences (640M tokens) per model**: ≥ 90 % on Pythia's mlp / resid sites from the lower bounds, TuringLLM-level
coverage expected elsewhere, at 40 % of TuringLLM's corpus; a uniform 1,024-target frame then has ~5-15 % thin or
skipped targets, mostly attention latents (DAN-9 decides whether the frame conditions on stored contexts, as TuringLLM's
did through pre-v1 survival). 24.8M matches TuringLLM exactly at ~2.5× the time. Disk: 528 bytes per sequence in shards
(5.3 GB at 10M) plus the corpus text.

### How long (one H100 per model; estimate, measure on the first ~200 batches)

TuringLLM's 2026-05-31 run (8 GPUs, `Runs/20260531-152059-37117a33/distributed/*.log`): pass 1 4.95 s and pass 2 4.1 s
per 4,096-sequence batch per GPU, bf16, 36 × 40,960 grid, d_model 1,024; pass 2 replays the unique top-store
sequences (16.5M of 24.8M, 67 %). Store updates scale with components × d_sae (Pythia 0.40×, GPT-2 0.98× TuringLLM's)
and the SAE GEMMs with components × d_model × d_sae (0.20× / 0.73×). Under bf16 (`--dtype bfloat16`) that gives about
2 s (Pythia) and 5 s (GPT-2) per 4,096 sequences in pass 1:

| N | Pythia pass 1 + 2 + neg_ctx | GPT-2 pass 1 + 2 + neg_ctx |
|---|---|---|
| 10M | ~1.4 h + 0.8 h + 0.25 h ≈ **2.5 h** | ~3.3 h + 1.9 h + 0.6 h ≈ **6 h** |
| 24.8M | ≈ 6 h | ≈ 14 h |

float32 (the default; what 033/053/070 validated; PyTorch does not use TF32 for matmuls by default) makes the SAE
GEMMs ~10× slower: 158 / 580 TFLOP per 4,096 sequences at ~60 TFLOP/s adds roughly 2.5 s (Pythia) and 9 s (GPT-2) per
batch in both passes, i.e. Pythia at 10M ≈ 5 h and GPT-2 ≈ 16 h. The CPU smoke rates (Pythia 5 seq/s, GPT-2 1 seq/s,
shared CPU) say nothing about the H100. Corpus preparation is single-process tokenisation at roughly 0.5-1M tokens/s
(the 2M-token scan corpus took 4 s): ~10-20 minutes for 640M tokens.

### What the H100 run needs

1. **Corpora, downloaded and licences confirmed (Daniel).** OpenWebText for GPT-2 (the OAI v5 SAEs' training data),
   the Pile for Pythia-70m (the model's training data; the training data of `EleutherAI/sae-pythia-70m-32k` is not
   recorded in the locally cached files, confirm it on the repo / sparsify config, and its training context length:
   033 used 64-token, no-BOS windows and so does this pipeline). Only N sequences are needed (10M = 640M tokens), so a
   subset of shards is enough; `prepare_corpus.py` streams files in order, so give it a shuffled or arbitrary shard
   subset rather than a sorted one if the source is ordered.
2. **GPT-2 layer-10 SAEs** (`v5_32k_layer_10` in attn-out / resid-mid / mlp-out, `v5_32k_layer_10.pt` in resid-post).
   The data pass tolerates their absence (`--skip-missing-saes`), but every GPT-2 target at layer ≥ 10 needs them for
   the fit, and their store rows are empty without them. With them cached, drop `--skip-missing-saes`.
3. **Native extensions built for the pod's CUDA** (`cd src/native && python setup.py build_ext --inplace`), then
   `gpu_native_check.py` for both families (first run of `latent_stats_cuda` on a non-TuringLLM grid).
4. **Precision decision** for the data pass (`--dtype`; see Decisions).
5. **Resources per model**: one GPU; RAM for the stores (GPT-2 ≈ 4 GB of artefacts, Pythia ≈ 1.5 GB) plus the
   pass-2 candidate dump, 2 KB per replayed sequence (256 candidates × 8 bytes; ≈ 13 GB at 10M × 67 %, held on the
   GPU under `dump_device: gpu`); disk ≈ shards + 5 GB.
6. **Consumers pointed at the family's roots**: `config.model.family`, `data.dataset_path` / `n_shards` and the store
   paths (`outputs_public/<family>`; 049/062 `setup` loads `outputs/` today). Contexts for the 1,024 targets then come
   from `protocol_contexts.build_record` exactly as `check_records.py` calls it.

Not covered: the distributed multi-GPU pipeline (`pipeline/distributed/`) is untested on the public grids; the
single-process path above is what ran. The mid reservoir is seeded from `std::random_device`, so mid stores are not
reproducible run to run (TuringLLM's are not either).

## Decisions (Daniel)

1. **Corpus and N.** OpenWebText / the Pile and their licences; N per model (suggested 10M sequences = 640M tokens;
   24.8M for strict parity with TuringLLM).
2. **Data-pass precision.** bf16 (~3× faster for GPT-2; only perturbs which near-tied sequences enter a store: the fit
   and the eval re-encode every context at their own precision, and the contrast silence check runs then) or float32
   (what 070 validated; GPT-2 at 10M ≈ 16 h on one H100).
3. **TuringLLM duplicates.** 23 % of the strongest-pool contexts in TuringLLM's store are exact text repeats (26.5 % of
   its corpus windows). Options: report it; measure the train / held-out leakage in 062's records; or dedupe
   TuringLLM's pools by content at record-build time (would change TuringLLM's contexts, so a protocol change). The
   public models are deduped at the source, so their pools are not comparable to TuringLLM's on this point unless one
   of these is done.
4. **Frame (DAN-9).** A uniform draw includes dead / ultra-rare latents (Pythia attention especially) that cannot
   fill pools at any affordable N; condition on stored contexts or accept thin / skipped targets.
5. **Sequence length 64 with a padded last target.** As on TuringLLM, a 64-token sequence leaves the 64th target
   token as padding (`protocol_contexts._load` reads 65). `prepare_corpus.py --seq-len 65` would give a real one but
   changes the stored sequence; kept at 64 for parity.
