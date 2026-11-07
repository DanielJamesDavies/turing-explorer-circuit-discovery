# Reproducing the runs

This guide describes how to run the production pipeline end to end on one 8-GPU machine: the three model tracks of
`experiments/082-public-runs/chain.sh` (Pythia-70m, GPT-2 small, TuringLLM) and the two baseline arenas of
`experiments/107-true-baselines/` (arena A and arena D). The method itself is specified in
`docs/wcm-specification.md`.

Every fit runs WCM `v4`. The chain exports `WCM_VERSION=v4` for every track, and its stage T0 refuses a driver that does
not support the chain's version.

Paths below assume the layout the bootstrap scripts use: the public tree at `/root/turing`, the TuringLLM tree at
`/root/turing-tl`, local scratch for archives at `/root/ckpt`, and a persistent network volume at `/workspace` (results
go to `/workspace/results`). All of these can be changed through environment variables (`ROOT`, `TL_ROOT`, `CKPT`,
`RESULTS`); see the headers of the scripts.

---

## 1. Requirements

### Hardware

| Item | Requirement | Why |
|---|---|---|
| GPUs | 8 × H100 80 GB (SXM) | the chain's GPU layout: data passes on GPUs 0 / 1, fits on 2–7, then all 8 |
| Container disk | ≥ 200 GB | corpora, stores, venvs, model weights, archives in `/root/ckpt` (arena D's saved graphs add up to ~32 GB) |
| Network volume at `/workspace` | ≥ 40 GB free; **≥ 60 GB free** when arena D saves its attribution graphs (the default) | every verified archive is copied there |
| Host RAM | **≥ ~256 GB** for arena A | each arena A deep WCM fit takes about 25 GB of host RAM, one fit per GPU, so 8 at once is about 200 GB. Nothing guards this |
| CUDA MPS | `nvidia-cuda-mps-control` available | the chain starts the MPS daemon (unless `MPS=0`) and runs 6 fit processes per GPU (`P=6`) |

On a host with less RAM, run fewer arena A fits at once, e.g. `ARENA_A_GPUS=0,1,2,3` on a 128 GB host (about 25 GB per
GPU plus headroom).

### Software

- A Linux image with a system PyTorch + CUDA install. The bootstraps build a venv on top of it
  (`--system-site-packages`) and pin `transformers 5.1.0` and its companions.
- `git`, `md5sum`, `tmux` (recommended); `pigz` optional (faster packing).
- `python3.10` for arena A's baseline venvs and `python3.11` for arena D's (installed with `apt-get` if missing; the
  arena scripts print the command).
- A Hugging Face read token for the gated `google/gemma-2-2b` (arena D only). Set `HF_TOKEN` in your environment, in
  the shell that will launch the chain. Never write it to a file.

### Inputs not in the repository

| Input | Used by | Where it must be |
|---|---|---|
| TuringLLM weights + 36 SAEs (`models/`), corpus shards (`data/`), data-pass stores (`runs_store.tgz`) | TuringLLM track | `/workspace/models`, `/workspace/data`, `/workspace/runs_store.tgz` |
| Results archive of the earlier 15k-target TuringLLM run | TuringLLM bootstrap, step 7 | `FULL_TGZ` in `scripts/pod_bootstrap_global.sh` (default under `/workspace/results/`) |
| `tl-extras.tgz` + `tl-extras.md5` (the 15,046 as-run context records and stored companion circuits of that run) | TuringLLM bootstrap, step 8 | `/root/tl-extras.tgz` (`TL_EXTRAS`) |
| Arena A's tuning choice `out/tune/CHOSEN.env` | arena A stage P5 (refuses to start without it) | `experiments/107-true-baselines/arena_a/out/tune/CHOSEN.env` |
| Arena D's local bf16 means file (optional) | arena D gate cross-check only (skipped if missing) | `experiments/107-true-baselines/arena_d/out/caples_means/rc_gap6_bf16_SMOKE.pt` |

The public tracks need none of these: their models and dictionaries come from the Hugging Face Hub, and the chain
builds their corpora (stage S1).

The TuringLLM inputs are not publicly distributed with the repository. **UNVERIFIED:** which later TuringLLM stages
read the 15k-run archive and `tl-extras.tgz` beyond the bootstrap, which requires both when `TL_TRACK=1`.

`out/` directories are gitignored, so `CHOSEN.env` must be created by hand. The production choice is:

```
AA_TUNE_CONFIG=nec0.3
NEC_W=0.3
```

---

## 2. Setup

Run these in a `tmux` session on the machine. B and D can run in parallel.

| Step | Command | Time | Check |
|---|---|---|---|
| A. Clone | `git clone --branch multi-device https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git /root/turing && git -C /root/turing log -1 --format=%h` | 1 min | the hash is the commit you intend to run |
| B. Public bootstrap | `bash /root/turing/scripts/pod_bootstrap_public.sh 2>&1 \| tee /root/boot_public.log` | ~30 min (HF downloads, ~13 GB) | adapter tests pass; GPU check passes; transformers 5.1.0 |
| C. TuringLLM inputs | copy `tl-extras.tgz` and `tl-extras.md5` to `/root/`, then `cd /root && md5sum -c tl-extras.md5` | a few min | `OK` |
| D. TuringLLM bootstrap | `ROOT=/root/turing-tl TL_TRACK=1 bash /root/turing/scripts/pod_bootstrap_global.sh 2>&1 \| tee /root/boot_tl.log` | ~20–30 min | `READY` line; `as-run records: 15046`; `corpus_hash.npy md5 ... == local` |
| E. GPU count | `nvidia-smi -L \| wc -l` | — | 8; both logs end cleanly |

What the bootstraps do:

- **`scripts/pod_bootstrap_public.sh`** (public tree): checks out the branch (`BRANCH`, default `multi-device`) and
  resets to `origin` (unless `NO_RESET=1`), builds `.venv` over the system torch with the pins, builds the native
  extensions, downloads Pythia-70m, GPT-2 and their 18 + 48 dictionaries, runs the adapter and corpus tests on CPU
  (a skipped test counts as a failure), runs a GPU-vs-CPU check, and prints a summary. It is idempotent; re-run after
  any failure.
- **`scripts/pod_bootstrap_global.sh`** with `ROOT=/root/turing-tl TL_TRACK=1` (a second checkout, so the public tree is
  never reset): copies `config-h100-triamp.yaml` to `config.yaml`, copies `models/` and `data/` from the volume, unpacks
  the stores, builds a venv and the native extensions, pre-builds the shard index, then (TuringLLM steps 7–9) unpacks
  the 15k-run archive and `tl-extras.tgz` (md5-checked) and recomputes experiment 085's corpus index, checking
  `corpus_hash.npy` against a fixed md5. If that check fails, copy a known-good
  `experiments/085-heldout-audit/out/corpus_hash.npy` into `/root/turing-tl/experiments/085-heldout-audit/out/`.

### Arena A preparation

The arena scripts never download anything. When something is missing, they print the commands and exit with code 2.

1. Read the plan: `cd /root/turing && DRY_RUN=1 bash experiments/107-true-baselines/arena_a/run_pod.sh | less`.
2. Run the commands it prints under P0 / P1. They are:
   - clone `third_party/Language-Model-SAEs` at `67bfb0b` and `third_party/transcoder_circuits` at `8721ff0`;
   - `ROOT=/root/turing PY=./.venv/bin/python bash experiments/107-true-baselines/arena_a/fetch_assets.sh pod`
     (GPT-2, Dunefsky transcoders, Kissane z-SAEs; ≈ 4.5 GB);
   - install `python3.10` if missing;
   - `ROOT=/root/turing TORCH_INDEX=cu121 bash experiments/107-true-baselines/arena_a/setup_env.sh both`
     (the `ge107` and `tc107` venvs; ≈ 3.9 GB of pip).

   `ROOT` matters: both scripts default it to a developer path, so always pass `ROOT=/root/turing`.
3. Create `out/tune/CHOSEN.env` (section 1).
4. Gate: `STOP_AT=P1 bash experiments/107-true-baselines/arena_a/run_pod.sh` must end `P1 envs: DONE`.

P0 also links a Hugging Face cache path that three arena A modules hard-code to this machine's cache, and regenerates
experiment 035's LN-folded transcoders. Both are local operations.

### Arena D preparation

1. Read the plan: `DRY_RUN=1 bash experiments/107-true-baselines/arena_d/pod_stage_d.sh | less`.
2. Run the commands it prints under P0:
   - clone `third_party/ScalableSAECircuits` at `f54a4ef` and `third_party/circuit-tracer` at `255d989`;
   - install `python3.11` if missing;
   - set `HF_TOKEN` in the shell, without echoing it: `read -rsp 'HF read token: ' HF_TOKEN && export HF_TOKEN; echo`;
   - `ROOT=/root/turing PY=python3.11 FETCH=1 bash experiments/107-true-baselines/arena_d/setup_env.sh`
     (venvs `ct107` + `caples107`, ~12 GB of pip, plus ~25 GB of weights and the Neuronpedia Dallas JSON).
3. Optionally copy the local means file (section 1) for the gate's cross-check.
4. Gate: `DW_STOP_AFTER=P2 bash experiments/107-true-baselines/arena_d/pod_stage_d.sh` must end `P2 gate: DONE` with
   both halves passing.

Arena D runs online by default (`DW_HF_OFFLINE=0`), so `HF_TOKEN` must stay exported in the shell that launches the
chain. By-name loads follow `main`; the resolved revisions are recorded in `out/pod/checks/hf_refs.txt`.

---

## 3. Launch

Print the plan first (CPU only, nothing is written):

```bash
cd /root/turing && DRY_RUN=1 TL_TRACK=1 TL_ROOT=/root/turing-tl ARENA_A=1 ARENA_D=1 \
  bash experiments/082-public-runs/chain.sh | less
```

Then launch the full run:

```bash
cd /root/turing && mkdir -p experiments/082-public-runs/logs && \
  TL_TRACK=1 TL_ROOT=/root/turing-tl ARENA_A=1 ARENA_D=1 STOP_POD=1 \
  nohup bash experiments/082-public-runs/chain.sh > experiments/082-public-runs/logs/chain.out 2>&1 &
```

| Variable | Meaning |
|---|---|
| `TL_TRACK=1` | run the TuringLLM track (T0–T12) beside the two public tracks |
| `TL_ROOT` | the TuringLLM checkout (default `/root/turing-tl`); its venv is picked up automatically |
| `ARENA_A=1` | run arena A after S10 and the TuringLLM track |
| `ARENA_D=1` | run arena D after arena A (whatever arena A's result) |
| `STOP_POD=1` | stop the machine after the final verified backup (see below). Uses `runpodctl stop pod $RUNPOD_POD_ID`, so it applies to RunPod hosts only; leave it unset elsewhere |
| `ARENA_A_MAX_H` | arena A's hard time limit in hours (default: 1.5 × its estimate, 15.75 h on 8 GPUs). Set it explicitly on a machine with fewer GPUs |
| `ARENA_D_STOP_EPOCH` | optional: no new arena D job after this time (`$(date -d '<time>' +%s)`); running jobs are killed `DW_HARD_GRACE` (1 h) later |

With `STOP_POD=1` the machine stops only after the final public archives are verified on the volume (C4) **and** the
TuringLLM track has finished and packed. The arenas do not gate the stop: each packs its own results, and a failed
arena's logs go to `/workspace/results/pub-arena-logs.tgz` (best effort) before the stop. On any other failure the
machine stays up.

Neither arena starts if the main chain failed (`ARENA_A_ANYWAY=1` / `ARENA_D_ANYWAY=1` override this).

---

## 4. Stages

### Public tracks (run side by side, one per model)

| Stage | Pythia-70m | GPT-2 small | What it does / produces |
|---|---|---|---|
| S1 | ✓ | ✓ | fixed corpus: 4,000,000 windows of 64 tokens at pinned revisions (Pythia: one Pile file; GPT-2: apollo OWT parquet files 0–2, exact dedupe) |
| S2 | ✓ | ✓ | native-extension check (a failure falls back to float32 for that model's data pass) |
| S3 | ✓ | ✓ | data pass: pass 1 + negative contexts only (data GPU 0 for Pythia, 1 for GPT-2) |
| S4 / S7 | S4 | S7 | records + target frame: 512 targets (≥ 200 firing windows, hub cap 50%, seed 20261024), subsets pilot 30, nulls 128, stable 32, unweighted 128 (Pythia only). C2 gate |
| S5 / S8 | S5 | S8 | λ pilot: 30 targets × 3 λ, then the C3 checkpoint |
| S6 / S9 | S6 | S9 | main fits → α = 1 rescore → harder nulls → stable core → unweighted arm (Pythia only) → resample (073) → donor test (+ unweighted donor arm for Pythia) |
| S10 | both | | pack results + stores per model, one md5 file, verify on the volume (C4) |

### TuringLLM track (`TL_TRACK=1`)

| Stage | What it does / produces |
|---|---|
| T0 | artefact check: models, data, stores, target lists (md5 against the manifest), driver supports the chain's `WCM_VERSION`, `config.yaml` matches |
| T1 | corpus scan (experiment 086 `scan.py`), top 512 windows per target |
| T2 | fresh records (experiment 094): an early gate on 64 targets spread over every cell (any failure stops the TuringLLM track), then the rest; later failures are quarantined, not fatal |
| T3 | main fits on 8,000 targets (8 chunks) + α = 1 rescore |
| T4 | λ sweep on 190 targets: 8 WCM + 5 unweighted arms |
| T5 | stable core (128 targets) |
| T6 | harder nulls (experiment 084) |
| T7 | own baselines (experiment 083: re-scored references, IG / AtP rankings, fitted amplitudes, then the table). `TL_DAN159=0` skips it |
| T8 | case-study and gallery circuits (taken from T3) packed with their records |
| T9 | off by default (`TL_ASRUN=1` enables a re-score of the earlier 15k run's circuits) |
| T10 | without-R arm: `v4nr` at λ 1e-3 on the 190 sweep targets, its own `OUT` |
| T11 | donor test of the 8k main circuits (8 chunks) |
| T12 | donor test of every sweep arm + the T10 arm |

The public tracks have priority on the shared fit GPUs; TuringLLM sub-steps fill the gaps, including the wait at C3.

### Arena A (`ARENA_A=1`): `experiments/107-true-baselines/arena_a/run_pod.sh`

| Stage | What it does |
|---|---|
| P0 | asset checks (clones, GPT-2, transcoders, z-SAEs, OWT parquet); prints download commands and exits 2 if anything is missing |
| P1 | baseline venvs `ge107` / `tc107`: pip check, CUDA visible, Ge's own test |
| P2 | arena A corpus (first 524,288 windows of OWT file 0, offline from the S1 cache) + data pass on GPU 0 |
| P3 | 48 targets (tuning dev targets excluded), 8 prompts each, records |
| P4 / P4h | Ge et al. and Dunefsky et al. as published, one worker per GPU; P4h repeats them on 8 held-out prompts per target |
| P5 | WCM v4 + necessity term (from `CHOSEN.env`): scope a1 (48 targets) and a2 (24 transcoder targets) × 5 λ |
| P6 / P6b | scoring with the donor test and self-test, Ge recovery; per-prompt scoring on train and held-out prompts |
| P7 | summary, results + stores archives with md5, copied to and verified on `/workspace/results`. Runs on every run, including after a time-limit stop |

Arena A has a time limit: a soft stop (no new unit) `ARENA_A_HARD_GRACE` (3600 s) before the hard stop, then every
arena A process gets SIGTERM and SIGKILL 30 s later (logged in `out/pod/logs/hard_stop.log`). P7 then packs whatever
finished and `run_pod.sh` exits 4 ("arena_a: STOPPED by its time limit" in the chain's events). `ARENA_A_STOP_EPOCH`
gives an explicit soft stop instead; if both are set, the earlier wins.

### Arena D (`ARENA_D=1`): `experiments/107-true-baselines/arena_d/pod_stage_d.sh`

| Stage | What it does |
|---|---|
| P0 | asset checks (two clones, pinned HF snapshots, Dallas JSON, `HF_TOKEN` present) |
| P1 | venvs `ct107` / `caples107`: pip check, CUDA visible |
| P2 | gate on two GPUs in parallel: the Caples half (means recomputed with their functions on the fp32 weights, identity and fidelity checks) and the attribution-graph half (prompts, identity, fidelity in fp32 strict, their own tests). A failing half turns off its production jobs |
| P3 | 18 jobs from one queue, one per free GPU: Caples masks and WCM on their substrate first, then WCM on the PLT transcoders (4 tasks × 2 means modes) and attribution graphs (4 tasks × 40 held-out prompts). Every attribution graph is saved (`AG_SAVE_GRAPHS=1`, ~0.1 GB each, ~16 GB total) |
| P4 | Caples fidelity + 9 `score_d.py all` jobs |
| P5 | summary, `pub-arena-d-results.tgz` + md5, then (after that is verified) `pub-arena-d-graphs.tgz` + md5, both verified on the volume. P5 also runs after a failure |

Arena D's WCM is the behavioural WCM run as `WCM_VERSION=v2` with arena C's settings (see the specification, §6.1).
Cut lines: `DW_D1=0` (Caples substrate only), `DW_D2_NATIVE=0` (Caples mode only), `AG_SAVE_GRAPHS=0` (no saved
graphs, if the volume is short of space).

---

## 5. Manual checkpoints

Times are approximate, measured from launch.

| When | Checkpoint | What to do |
|---|---|---|
| ~0:20 | **C1** data-pass throughput | `tail experiments/082-public-runs/logs/pythia-70m/dp_pass1.log`. Expect pass 1 at ~0.6 h for Pythia and ~1.3 h for GPT-2. Much slower means something is wrong |
| ~0:55 | T2 early gate (automatic) | `logs/chain.events` shows `T2 early gate: 64 targets, 64 ok`. On a failure the TuringLLM track stops (the public tracks go on): read `GATE FAIL` in `logs/turingllm/records_report.log`, fix, and rerun the chain |
| ~1:30–2:15 | **C2** records + frame (automatic gate) | glance at `logs/<model>/targets.log`: excluded counts per rule, 512 main targets |
| ~1:45 | T2 quarantine (automatic, never stops) | `records_report.log` shows `T2 quarantine: N of 8000 targets quarantined` and any `MUST-INCLUDE` warning. The list is `/root/turing-tl/experiments/082-public-runs/out_turingllm/lists/quarantine.tsv` |
| ~1:45–2:15 | **C3 Pythia λ** | read `experiments/082-public-runs/out_pythia-70m/pilot_lambda.json` (the pre-stated rule's recommendation). Adopt it: `echo <λ> > experiments/082-public-runs/logs/state/lam_pythia-70m` |
| ~2:30–3:00 | **C3 GPT-2 λ** | the same with `out_gpt2-small/` and `lam_gpt2-small` |
| end of the main chain | **C4** final verify (automatic) | `pub-final.md5` checked on `/workspace/results`, plus the `pub-turingllm-T*` archives |

The main public fit waits at C3 until a λ file exists (or `LAM_pythia_70m=…` / `LAM_gpt2_small=…` is set in the
environment). The GPUs do not idle meanwhile: TuringLLM work fills the wait. A late C3 delays the end by about the same
time.

---

## 6. Monitoring

**Progress bars** (redraws every 60 s; `tqdm` is installed by both bootstraps):

```bash
cd /root/turing && .venv/bin/python experiments/082-public-runs/progress_tqdm.py      # Ctrl-C to stop
```

Options: `--once` (one frame), `--interval 30`, `--tl-root DIR` (default `/root/turing-tl`), `--limit` (the local
smoke). It shows one bar per GPU, one per track (stage + main fits), the T2 quarantine, and the FAILED / GIVING UP
counts. It only reads what the chain writes.

**Detailed text view:**

```bash
cd /root/turing && TL_ROOT=/root/turing-tl watch -n 120 bash experiments/082-public-runs/progress.sh
```

It shows stage markers and recent events; per model the corpus, data pass, target counts and arm rows; the TuringLLM
track's stages with rate and ETA and its `OUT` version marker; GPUs, MPS, disk, and the verified archives.

**Job errors.** Every fitting job's error rows are counted. A job whose first 16 fits all error is stopped and marked
`JOB … FAILED` in `chain.events`. Other errors are logged as `JOB …: n rows, e errors`. The full list is
`logs/job_errors.jsonl`.

**Failures:**

```bash
grep -h "FAILED\|GIVING UP" experiments/082-public-runs/logs/chain.events experiments/082-public-runs/logs/*/*/*.log
```

**Arenas:** `tail -f experiments/107-true-baselines/arena_a/out/pod/logs/events.log` (per-stage times in
`out/pod/steps.tsv`, GPU memory in `out/pod/gpu_mem.csv`) and
`tail -f experiments/107-true-baselines/arena_d/out/pod/logs/events.log` (per-command times in `out/pod/cmds.tsv`).
The chain's own arena logs are `experiments/082-public-runs/logs/arena_a/run_pod.log` and `logs/arena_d/pod_stage_d.log`.

Before arena A's P5 starts, check host RAM with `free -g` (section 1).

---

## 7. Outputs and verification

Every stage boundary (and every arm inside S6 / S9) writes a `.tgz` and an `.md5` to `/root/ckpt`, copies both to
`/workspace/results` (at most 4 copies at once), and runs `md5sum -c` **on the volume**. A stage is marked done only
after that check passes. All archive names start with `TAG` (default `pub`).

| Archive | Contents |
|---|---|
| `pub-<model>-S*-*.tgz` | per-stage and per-arm archives of the public tracks |
| `pub-<model>-S10-results.tgz`, `pub-<model>-S10-stores.tgz` | final per-model results and data-pass stores |
| `pub-final.md5`, `pub-final.counts` | one md5 file over the S10 archives; entry counts per archive |
| `pub-turingllm-T<n>-*.tgz` | one archive per TuringLLM stage (e.g. `T3-main`, `T4-sweep`, `T11-donor-main`) |
| `pub-arena-a-results.tgz`, `pub-arena-a-stores.tgz`, `pub-arena-a.md5` | arena A |
| `pub-arena-d-results.tgz` + `pub-arena-d.md5`; `pub-arena-d-graphs.tgz` + `pub-arena-d-graphs.md5` | arena D (graphs in their own archive) |
| `pub-arena-logs.tgz` | only if an arena failed and `STOP_POD=1` |

To verify a copy after transfer:

```bash
cd <download dir> && md5sum -c pub-final.md5 && md5sum -c pub-arena-a.md5 && md5sum -c pub-arena-d.md5
```

and `md5sum -c` on each per-stage `.md5` file the same way.

---

## 8. Approximate GPU-hours

Measured per-stage costs on 8 × H100. The fit stages were measured with v3; v4 fits take about the same time (×0.99–1.01).

| Stage | GPU-h |
|---|---:|
| T3 main fits | 54.3 |
| T4 λ sweep | 15.0 |
| T7 own baselines (trimmed to `ref attr fitted`) | ~10 |
| T2 records | 10.6 |
| T11 donor test (main) | 5.0 |
| T1 corpus scan | 3.5 |
| T6 harder nulls | 3.1 |
| T5 stable core | 1.7 |
| T3 α = 1 rescore | 1.3 |
| T12 donor test (sweep) | 1.2 |
| T10 without-R arm | 0.8 |
| Pythia-70m track (S4–S9) | 10.7 |
| GPT-2 small track (S7–S9) | 26.7 |
| Public data passes | 2.3 |
| Arena A | up to ~68, plus ~2 h serial (P0–P3) |
| Arena D | **UNVERIFIED** |

The main chain (without arenas) sums to roughly 145 GPU-h, about 20 h of wall time on 8 GPUs. The public models' speed on
H100 relative to a smaller GPU was estimated, not measured, so their figures are the least certain. To shorten the
run: `DONOR_VARIANTS=fit` runs only the `fit` donor variant (cuts T11 by about two thirds); `TL_DAN159=0` skips T7.

---

## 9. Troubleshooting

- **Resume.** Fix the cause, then rerun the same launch command. Finished stages are skipped by their markers
  (`experiments/082-public-runs/logs/state/*.done`), and fits resume per target. To redo a stage, remove its marker.
  The arenas resume the same way (`out/pod/state/`).
- **Version mismatch.** The driver marks each `OUT` with its `WCM_VERSION` and refuses to mix versions. A smoke or
  rerun with a different version needs a fresh `WORK` (or `OUT`) directory.
- **T0 fails.** It names the missing artefacts; re-run the TuringLLM bootstrap (`TL_ROOT` must be bootstrapped with
  `TL_TRACK=1`).
- **Corpus hash mismatch in the TuringLLM bootstrap (step 9).** Provide a known-good `corpus_hash.npy` (section 2).
- **An arena exits 2.** It is waiting for downloads or venv builds; run the commands it printed (its log is under
  `experiments/082-public-runs/logs/arena_*/`), then rerun the chain.
- **Arena A on fewer GPUs.** The default time limit scales with the GPU count (28.5 h on 4); set `ARENA_A_MAX_H` to
  what you can afford. Its default is a hang guard, not a plan.
- **Volume short of space.** Set `AG_SAVE_GRAPHS=0` before arena D starts.
- **Host RAM.** See section 1; reduce `ARENA_A_GPUS`.
- **Stale lock flags** from a killed chain are cleared at start-up; nothing to do by hand.

### Local smoke

On a single GPU, the chain can run a short smoke on small stores:

```bash
LIMIT=2 TL_TRACK=1 MODELS=pythia-70m WORK=<fresh dir> bash experiments/082-public-runs/chain.sh
```

`LIMIT=2` cuts every target list to its first 2 targets, uses 20 steps, one process on GPU 0, no MPS, and local
archive directories under `WORK`. It runs the public tracks from S4 on experiment 069's smoke stores. **UNVERIFIED:**
those smoke stores are not in the repository, so a fresh clone must build them first.
