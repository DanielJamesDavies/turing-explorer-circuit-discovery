# 082 public runs: the pod run chain for Pythia-70m, GPT-2 small and the TuringLLM refit

## Question

Does WCM, applied unchanged, give the same picture on public models with public dictionaries as on TuringLLM? This
folder holds the chain that runs the whole protocol on one 8-GPU pod: Pythia-70m and GPT-2 small from raw corpus to
scored circuits, and, as a third track, the TuringLLM refit of 8,000 targets on fresh records. It also holds the
target-frame script, the λ-pilot report, the job error gate, progress views, CPU tests, a local λ pre-pilot and
local smoke runs.

The paper's public-model results are still pending on this run (`paper/main.tex:154, 207, 503`). The design it
states comes from here:
- dictionaries and models: `main.tex:501, 844-845`;
- 512 targets per model, the frame and the 30-target pilot: `main.tex:846`;
- the corpora: `main.tex:849`;
- the λ rule: `main.tex:868`.

The method spec is `paper/protocol.md` §0 (v4); its public-model rows cite this chain (`protocol.md:17, 195-200, 309`).

## Algorithm version

Every fit of every track runs under `WCM_VERSION=v4` (production: v3 + geometric lr schedule at peak ×3.15, warm-up
0.1, and the censored-activation term; v3 = Z + A + C + R_C with R weight 1.0 and rank-keep 1e-1). `chain.sh` exports
it for every track and passes it to every launch. The driver marks each OUT with its version and refuses to mix
versions, so a v4 OUT is always fresh. `v4nr` (v4 without R) is used only by the T10 appendix arm; `v1` only by the
optional T9 re-score of the fixed 15k circuits.

## The chain (`chain.sh`)

Run from anywhere; it changes to the repo root. The two public tracks and the TuringLLM track run side by side as
background processes. All multi-GPU work goes through `062-h100-protocol-v1/launch.sh` (one shard per GPU slot, P
processes per GPU, retried up to 5 times) and shares one **fit lock**.

### Public tracks (one per model)

| Step | Pythia-70m | GPT-2 small | What it does |
|---|---|---|---|
| S1 | S1 | S1 | **Corpus**, fixed in advance: 4,000,000 windows of 64 tokens (`N_SEQ`), exact-deduplicated, revisions pinned. Pythia: `monology/pile-uncopyrighted` `train/00.jsonl.zst` @3be90335. GPT-2: `apollo-research/Skylion007-openwebtext-tokenizer-gpt2` parquet files 0–2 @f02886b5 (about 5.8M windows before dedupe). `069/prepare_corpus.py` |
| S2 | S2 | S2 | **Native check** on the data GPU (`069/gpu_native_check.py`). On failure that model's data pass falls back from bf16 to float32 |
| S3 | S3 | S3 | **Data pass**, pass 1 + contrast contexts only (`069/run_data_pass.py --stages pass1,negctx`). Pass 2 and candidates are skipped because nothing downstream reads them |
| C2 | S4 | S7 | **Records and target frame.** `069/check_records.py` (gate: every record check must pass, else the track stops), `dup_report.py`, `estimate_corpus.py`, then `make_targets_public.py`. The lists are md5-frozen in `STATE`, and every later stage re-checks the md5 |
| C3 | S5 | S8 | **λ pilot.** Driver `MODE=sweep` over the 30 pilot targets (Pythia `W_0.0003, W_0.001, W_0.003`; GPT-2 `W_0.001, W_0.003, W_0.01`), then `pilot_lambda.py`. The main fit then **waits** until a λ is written to `STATE/lam_<model>` (or `LAM_<model>` is set); the fit lock is released meanwhile |
| main | S6 | S9 | **Main fit and arms**, in order, each with its own done marker and archive: `main` (required) → `alpha1` (α = 1 re-score) → `harder_nulls` (084: 128 targets × {anchor, contrib} × 1 draw, a 2nd draw on the first 32, WIN 8; production members kept in the anchor pool and excluded from the contrib pool; no DAN-17 references) → `stable_core` (32) → `unweighted` (Pythia only: `U_<λ>` on 128 targets) → `strict` (073 resample on the nulls list; its random variant is the anchor null's draw 0) → `donor` (donor-background test of every main circuit; `src/eval/donor_test.py`) → `donor_unweighted` (Pythia only). A failed optional arm is listed in `STATE/failed.list` and does not stop the track |
| C4 | S10 | S10 | **Pack** (both models): results and stores tarballs, one md5 file, `md5sum -c` on the volume, entry counts |

### TuringLLM track (`TL_TRACK=1`)

It runs in its own checkout `TL_ROOT` (default `/root/turing-tl`), with `config.yaml` required to equal
`config-h100-triamp.yaml`, into `TL_V2_OUT` (`094/out_v2`). Every fit and score reads the fresh 094 records
(`TL_CTX_FRESH`).

| Stage | What it does |
|---|---|
| T0 | **Artefact check:** stores, models, scripts, 094 lists against their manifest md5s, the driver knows `WCM_VERSION`, `TL_V2_OUT` holds no other version |
| T1 | **Corpus scan** (`086/scan.py`, top 512 per target of the 8k on the content-deduplicated corpus), then merge |
| T2 | **Fresh records** (`094/build_records.py`): an early gate first (build + independent verify of the 64 gate targets spread over the 35 cells; any failure stops the track), then build + verify of the rest. A later failure **quarantines** that target (`094/quarantine.py`): every later stage reads the lists minus the quarantined ones (`TL_LISTS`) |
| T3 | **Main fits** of the 8k (fit → held-out strongest and mid-band scores → 056 specificity), in 8 chunks (line i → chunk i mod 8, so every chunk spans every layer), one lock holder per chunk; then α = 1 |
| T4 | **λ sweep** on the 190 sweep targets: WCM at 8 λ (2.5e-5 … 4e-3) and unweighted at 5 (1e-5 … 1e-3), one launch per arm. `W_0.001` reuses the T3 circuit |
| T5 | **Stable core** on 128 targets |
| T6 | **Harder nulls** (084) on the 126-target panel, with DAN-17 references all refitted under the running version |
| T7 | **083 baselines** (`083/run_pod.sh`, our own baselines only): stages `ref attr fitted` (`TL_DAN159_STAGES`), then `table`. WCM / unweighted references = the T4 sweep |
| T8 | **Case-study (6) and gallery (9) circuits**: checks they exist among the T3 fits and packs them with their records |
| T9 | Optional (`TL_ASRUN=1`): the leakage note. The fixed 15k v1 circuits of the as-run 1,000 subset, scored on their as-run records and on 086-rebuilt held-out sets |
| T10 | **Without-R arm** (only when the running version has R): `v4nr` at `W_0.001` on the 190 sweep targets, into its own OUT `094/out_v2_noR` |
| T11 | **Donor-background test** of every 8k main circuit, in T3's chunks |
| T12 | **Donor-background test** of every sweep arm in one launch, then of the T10 arm |

A switched-off stage (T7, T9–T12) is reported as SKIPPED and not marked done, so a later rerun with the switch on
runs it.

### After both tracks

- `ARENA_A=1`: exp 107 arena A (`107-true-baselines/arena_a/run_pod.sh`) on the whole pod, with its own time limit
  and soft / hard stop.
- `ARENA_D=1`: then exp 107 arena D (`107-true-baselines/arena_d/pod_stage_d.sh`).
- Neither starts if the main chain or the TuringLLM track failed (unless `ARENA_*_ANYWAY=1`).
- `STOP_POD=1` stops the pod only after S10's verification passed in this run and the TuringLLM track finished and
  packed. A failed arena does not keep the pod up; its logs are packed first.

### Shared machinery

- **GPU packing.** The fit pool is `FIT_GPUS` (2–7) plus the data GPU of every model whose data pass is done (Pythia
  0, GPT-2 1). One holder of the fit lock at a time. Public stages have priority: a TuringLLM sub-step takes the lock
  only while no public stage is waiting (`STATE/want.*`), and each TuringLLM sub-step (scan, build, verify, each main
  chunk, each sweep arm, each 083 stage) takes it separately.
- **Job error gate** (`job_errors.py`, via `run_job`). Every fitting launch is watched. If its first `JOB_GATE` (16)
  output rows all carry an `error` field, the launch's process group is stopped and the stage fails. Otherwise errors
  are counted and the first message is logged to `chain.events`, the launch log and `LOGS/job_errors.jsonl`.
- **Checkpoints.** Every stage boundary, and every arm inside S6 / S9, writes a tarball and md5 to `CKPT`, copies
  it to `RESULTS` (at most 4 copies at once) and runs `md5sum -c` on the volume. Only then is the stage marked done.
- **Resume.** Every stage writes `STATE/<key>.done`; a rerun skips done stages, and the fits resume per target.
  Delete a marker to redo its stage. `FROM=S4` treats S1–S3 as done.

### Main environment variables (defaults in brackets)

| Variable | Meaning |
|---|---|
| `DRY_RUN=1` | Print every command, run nothing, write no file |
| `LIMIT=N` | Local smoke: fresh `WORK`, the 069 smoke stores, lists cut to their first N, `STEPS=SMOKE_STEPS` [20], P 1, GPU 0, local `CKPT` / `RESULTS`, `FROM` [S4] |
| `MODELS` | ["pythia-70m gpt2-small"] |
| `FROM`, `STOP_AFTER`, `STOP_AFTER_<model>` | Stage window [S1 … S10] |
| `WCM_VERSION` | [v4] |
| `LAM_<model>` | The adopted λ (else `STATE/lam_<model>`; smoke 1e-3) |
| `PILOT_ARMS_<model>` | The pilot grid (above) |
| `UNWEIGHTED_<model>` | [Pythia 1, GPT-2 0] |
| `SEED`, `FRAME` | [20261024], [fired] |
| `NULLS_SUB_N`, `NULLS_SUB_DRAWS`, `NULLS_WIN`, `NULLS_EXCLUDE`, `NULLS_REFS`, `NULLS_DAN17_REFIT` | 084 settings [32, 2, 8, contrib, wcm, 0] |
| `P`, `STAGGER`, `FIT_GPUS`, `DATA_GPU_<model>` | [6], [5], [2..7], [0 / 1] |
| `N_SEQ`, `BATCH`, `DTYPE` | [4,000,000], [2048], [bfloat16] |
| `WORK`, `LOGS`, `STATE`, `CKPT`, `RESULTS`, `TAG` | [this folder], [WORK/logs], [LOGS/state], [/root/ckpt], [/workspace/results], [pub] |
| `DATA_ROOT`, `STORE_ROOT` | [data_public], [outputs_public] |
| `CTX_DIR`, `CTX_DIR_<model>`, `SAE_PRELOAD`, `MPS` | Records dir [OUT/ctx], [0], [1] |
| `JOB_GATE`, `JOB_WATCH_EVERY` | [16], [30 s] |
| `DONOR`, `DONOR_<model>`, `DONOR_SCRIPT`, `DONOR_P` | Public donor arms [1, src/eval/donor_test.py, P] |
| `TL_TRACK`, `TL_STOP_AFTER`, `TL_ROOT`, `TL_PY` | [0], [T12], [/root/turing-tl], [TL_ROOT/.venv] |
| `TL_V2_OUT`, `TL_WORK`, `TL_CTX_FRESH`, `TL_TARGETS_DIR`, `TL_CORPUS_DIR` | TuringLLM paths |
| `TL_P`, `TL_SCAN_P`, `TL_BUILD_P`, `TL_VERIFY_P`, `TL_083_P`, `TL_MAIN_CHUNKS` | [P, 1, 3, 6, 4, 8] |
| `TL_SWEEP_ARMS`, `TL_NULLS_SUB_N` | 8 WCM + 5 unweighted arms, [32] |
| `TL_DAN159`, `TL_DAN159_STAGES`, `TL_ASRUN` | T7 on [1, "ref attr fitted"], T9 off [0] |
| `TL_NOR`, `TL_NOR_VERSION`, `TL_NOR_ARM`, `TL_NOR_OUT` | T10 [1, v4nr, W_0.001, 094/out_v2_noR] |
| `TL_DONOR`, `TL_DONOR_MAIN`, `TL_DONOR_SWEEP`, `TL_DONOR_POOL` | T11 / T12 [1, 1, 1, TL_V2_OUT/donor/random_pool.pt] |
| `ARENA_A`, `ARENA_D`, `ARENA_*_GPUS`, `ARENA_*_ANYWAY`, `ARENA_A_MAX_H`, `ARENA_*_STOP_EPOCH` | Post-chain exp 107 stages [off] |
| `STOP_POD` | [0] |

The header of `chain.sh` lists every variable and the interfaces of the scripts it calls.

```
DRY_RUN=1 TL_TRACK=1 bash experiments/082-public-runs/chain.sh                   # print the full plan
TL_TRACK=1 STOP_POD=1 nohup bash experiments/082-public-runs/chain.sh > experiments/082-public-runs/logs/chain.out 2>&1 &
LIMIT=2 TL_TRACK=1 bash experiments/082-public-runs/chain.sh                     # local smoke (FROM=S4)
echo 1e-3 > experiments/082-public-runs/logs/state/lam_pythia-70m                # adopt a λ at C3
bash experiments/082-public-runs/progress.sh                                     # one-screen status
python experiments/082-public-runs/progress_tqdm.py [--once] [--limit]            # per-GPU progress bars
```

## The other files

| File | What |
|---|---|
| `make_targets_public.py` | **Target frame** (S4 / S7, CPU, no SAE loaded). 512 targets per model, seed 20261024: Pythia 32 per component × 16 (attn 0 and mlp 0 have no upstream SAE site); GPT-2 10–11 per site over 47 sites. `--frame fired`: a latent is eligible if it fires in ≥ 200 distinct windows (`--min-seqs`) and in ≤ 50% of the windows (hub cap). Subsets: pilot 30 (disjoint from main), nulls 128 (inside main, the 32 that get a 2nd null draw first), stable 32, unweighted 128 (Pythia only). Deterministic (PCG64 on SeedSequence([seed, stream, component])), lists in 062's order; the manifest records per-component counts, exclusions by rule and md5s. Existing lists that differ are never overwritten (`--force` overrides) |
| `pilot_lambda.py` | **λ-pilot report** (CPU). Per λ and depth band: median size (p25, p75), share in [100, 1000], pass rate (reported only). Pre-stated rule, sizes only: keep λ = 1e-3 if its pooled median size is in [100, 1000]; else the grid λ whose pooled median is nearest 494 nodes on a log scale (ties: nearer 1e-3, then the larger λ). Writes `pilot_lambda.json` with `"adopted": false` |
| `job_errors.py` | **Job error gate**: `snapshot` / `watch` / `check` (see above) |
| `progress.sh` | Read-only one-screen view: markers, logs, row counts, nvidia-smi |
| `progress_tqdm.py` | Read-only tqdm view: per-GPU bars from the shard logs, per-track bars, error jobs, the T2 quarantine, failures |
| `prelim_lambda_local.sh` | **Local λ pre-pilot** on the 069 smoke stores (wikitext, a few thousand sequences): a separate frame draw (seed 0), the pod pilot's grids, then `pilot_lambda.py`. Indicative only |
| `test_082_targets.py`, `test_082_pilot_lambda.py`, `test_082_job_errors.py`, `test_082_progress_tqdm.py` | CPU tests (`python -m pytest experiments/082-public-runs/test_082_*.py -q`, with `PYTHONPATH=src` and `CUDA_VISIBLE_DEVICES=`) |
| `golden_2026-10-24/` | A whole-tree golden check (`golden_tree.py`): the TuringLLM fit + eval + specificity + α = 1 through the 062 driver, once with the committed tree (`head_gpu/`) and once with the working tree (`work_gpu/`), on 2 targets; `memwatch.sh` pauses it when host memory runs low. `public_smoke_pythia.log` is a records check on 8 Pythia smoke records |
| `prelim/` | The pre-pilot's outputs (below) |
| `smoke_v2/`, `smoke_v3/`, `smoke_gpt2_v2/`, `smoke_final2/` (+ `.out` / `.log`) | Local `LIMIT=2` smoke runs of earlier chain versions (below) |

## Results

**No pod results are in this folder.** The pod outputs (`out_<model>/`, `out_turingllm/`, `logs/`) are gitignored
and come back as tarballs; none are present here, so this README reports no public-model or refit numbers.

**Local λ pre-pilot** (`prelim/report_pythia-70m.txt`, `prelim/report_gpt2-small.txt`). 5 pilot targets per model on
the smoke stores, 400 steps:

| model | λ | pooled median size (p25, p75) | in [100, 1000] |
|---|---|---|---|
| Pythia-70m | 3e-4 | 326 (315, 621) | 60% |
| Pythia-70m | 1e-3 | 172 (117, 257) | 80% |
| Pythia-70m | 3e-3 | 81 (66, 93) | 20% |
| GPT-2 small | 1e-3 | 333 (273, 541) | 100% |
| GPT-2 small | 3e-3 | 237 (162, 360) | 80% |
| GPT-2 small | 1e-2 | 100 (70, 130) | 60% |

The rule keeps λ = 1e-3 for both (pooled median inside [100, 1000]). Both sweeps finished with 0 error rows; the
Pythia sweep took about 7 minutes and the GPT-2 sweep about 21 minutes on the local GPU (`prelim/timing.log`). With 5
targets and thin smoke-store contexts, this shows only that the pod grids bracket the size range.

**Golden check** (`golden_2026-10-24/head_gpu/`, `work_gpu/`): on both targets (`0.resid.23442`, `3.attn.11307`) the
eval and training rows of the two trees are identical except for the timing fields (`secs`, `t_fit`).

**Smoke runs.** All at `LIMIT=2`, 20 steps, P 1, GPU 0, so their scores carry no information; they test wiring only.
They predate v4 and the donor stages (T10–T12 and the public donor arms).
- `smoke_v2/`, `smoke_v3/`: Pythia + the TuringLLM track under `WCM_VERSION=v2`. S4–S10 and T0–T8 done, T9 skipped,
  chain rc 0.
- `smoke_gpt2_v2/`: GPT-2 under v2. S7–S10 done, rc 0.
- `smoke_final2/`: both models + an older TuringLLM layout (T0–T5), from a chain version whose header line records
  no `WCM_VERSION`. One T3 failure, then rc 0 on the rerun.
- Later chain smokes, including the v3 and donor stages, are in `103-donor-stage/smoke_chain/` and
  `104-v3-cross-model/smoke_chain*/`.

## Status and caveats

- The chain is the pod run's code. The λ at C3 is adopted by hand; nothing in the chain adopts the pilot's
  recommendation.
- The licences of both public corpora are marked UNCONFIRMED in the chain (Pythia: the HF card says "other"; the
  apollo OpenWebText copy has no card licence).
- `DRY_RUN=1` prints the plan without touching files or locks; use it to check any change of environment before a
  pod run.
