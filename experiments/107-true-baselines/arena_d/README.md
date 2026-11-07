# Arena D: attribution graphs (D1) and Caples masks (D2) on Gemma-2-2B, plus the WCM adapters (plan)

## Status 5, newest (agent `hardstop`, CPU only): the AG graphs are saved

- `pod_stage_d.sh`'s `ag_*` jobs now run **`ag_tasks_graphs.py`**: `ag_tasks.py` unchanged, plus every attribution
  graph saved as `out/pod/ag_tasks/<task>/held/p<id>.graph.pt`. The adjacency is float16 (~0.1 GB each, ~16 GB for
  4 × 40). The per-prompt outputs are identical to `ag_tasks.py`'s (checked on the tiny CPU model).
  `AG_SAVE_GRAPHS=0` switches back to `ag_tasks.py`; the default is 1.
- **P5:** the results tgz leaves `*.graph.pt` out and is verified on RESULTS first. The graphs then go into their own
  `pub-arena-d-graphs.tgz` + `.md5` (pigz when present), copied and md5-checked the same way. A slow or failed 16 GB
  copy therefore cannot delay or sink the results archive.
- P5's own commands are spared by the hard stop (pid files `P5_*`), so a pack that runs across the hard-stop time is
  not killed.
- **Disk.**
  - Container disk: +~16 GB of graphs in the work dir, plus their archive in `/root/ckpt` (≤ 16 GB, less after gzip),
    so ≤ ~32 GB extra on top of this README's ≥ 150 GB for arena D.
  - Global volume: +≤ 16 GB. The runbook's "≥ 40 GB free" for the main archives becomes **≥ 60 GB free**.
- Tested on CPU: P5 on a fake work dir with one graph gives both archives verified and no graph in the results tgz.

## Status 4 (plan D8 / D9 built; CPU dry runs only, nothing downloaded, nothing remote)

- **D8 `pod_stage_d.sh`** (header = the full stage list): P0 asset checks (prints the clone / `setup_env.sh FETCH=1`
  commands for approval, exit 2), P1 env gate, P2 gate on GPUs 0,1 (the halves in parallel; `DW_GATE_GPUS` default since status 5, `DW_GATE_GPUS=0` = GPU 0 alone), P3 one shared
  job queue over the 8 GPUs (flock; resumable per job via each runner's own resume), P4 caples_fidelity + `score_d.py
  all`, P5 summary (`pod_summary_d.py`) + tgz + md5 on `/workspace/results`, also after a failure. `STOP_EPOCH` = no new
  unit; a hard kill `DW_HARD_GRACE` later. `pod_checks_d.py`: google fp32 → bf16 vs unsloth tensor by tensor; pod vs
  local Caples means. Queue tested on CPU (fake jobs: 4 "GPUs", resume, failure, STOP_EPOCH).
- **Found while wiring:** (1) `chain.sh` exports `WCM_VERSION=v4`, which 081's `task_common` refuses (v1 / v2 only):
  `wcm_gemma.py` would have exited at import inside the chain. The stage pins `WCM_VERSION=v2` (= 081's default, what
  arena C's adopted runs and the GPU-night smoke use) and drops the chain's `PYTHONPATH=src`. (2) The chain's STOP_POD
  required every arena to succeed, so a failed arena A kept the pod up; fixed in D9. (3) `score_d.py` covers the
  residual family only: D1's rows are ours = `wcm_gemma.py`'s eval.jsonl, theirs = ag_tasks.py's completeness /
  replacement; our scoring of AG circuits and their metric on WCM-PLT circuits have no runner yet (open).
- **D9:** `082/chain.sh` `arena_d_stage` after arena A (`ARENA_D=1`, off by default); runbook entry.

## Status 3 (plan D5 built, D6 written, D7 script ready; CPU only, no GPU, nothing downloaded)

- **D5 built, CPU dry runs pass:** `run_caples.sh` (grids A / B / C / D / dense, one `git clone --local --no-hardlinks`
  copy per grid, HEAD + clean-tree check, the copy's committed `masks/` moved aside so every JSON there is ours; DRY_RUN
  prints every command); `caples_fidelity.py` (model LD 3.21 ± 0.02, size ± 30 % (floor 2), f ± 0.05, final T vs the
  loop arithmetic = 643.6, F5 Jaccard, F7 nesting; dry run: our grid arithmetic reproduces all 20 anchor folder names
  exactly, anchors-vs-anchors PASS); `ag_fidelity.py` (F1 provenance incl. sha256 of all 71 installed files vs RECORD:
  PASS; F3 / F4 / F5 / F6 on the GPU night; dry run on their tiny test model: F5 identity, F6 batch invariance and our
  score formula = `compute_graph_scores` all PASS); `ag_tasks.py` (`prompts` in .venv wrote the 4 SVA tasks' 100 + 40
  prompts to `out/ag_tasks/prompts/`; `run` in ct107, resumable per prompt, transcoders pinned @3412e19b; tiny-model run
  and resume checked). The real `ReplacementModel` build (unsloth bf16 + PLT) was checked on CPU offline: loads in 20 s.
- **Findings:** (1) grids B and C's anchor JSONs store **positions**, not latents (every index ≤ 6); A and D store
  latents. So F5 latent Jaccard exists for A (and D) only; for B / C caples_fidelity compares per-position histograms
  from our `.pt` masks. (2) Offline, transformers 4.57.3 (ct107) `AutoTokenizer.from_pretrained("<repo id>")` calls the
  hub API and fails; `ag_tasks.local_or_id` passes the cached snapshot folder instead. (3) Offline, the pinned
  transcoder revision 3412e19b cannot resolve (only snapshot bd577315 is cached, byte-identical layer files): the AG
  scripts fall back to it **only when HF_HUB_OFFLINE=1** and record `transcoders_used`; on the pod the pin is used.
  (4) The plan's Caples smoke "N 64" cannot run: their `do_training_run` evaluates batches [0, −1..−9] (needs ≥ 160
  prompts); `run_pilot_d.sh` uses `--n 176 --mean-batches 4`.
- **Ours, labelled:** pruned-graph replacement / completeness (spec §9.2) = their formulas with each dropped feature
  node turned into an error node (incoming edges removed, joins the error set).
- **D6:** `test_real_open_gate_identity_gpu` (marked `gpu`, both families, bitwise vs raw HF on 8 rc prompts; open gate,
  identity transform, plain call). Skips on CPU. **D7:** `run_pilot_d.sh` (steps, sampler, watchdog, summary in its
  header; `DRY_RUN=1` clean).

## Status 2 (plan items D1-D3 built; CPU only, no GPU, nothing downloaded)

- **D1 (engine changes wired):** `GemmaSAEBank.mean_skip_positions` = 1 for both families (0 for any other family);
  `JumpReLUSAE.threshold` is what `sae.dense.seed_threshold` reads. Tests: seed_threshold reads the buffer and
  `seed_activation` = the encode (both families; NullSAE slot → None); pooled means (`collect_site_means`) = the mean
  over valid positions ≥ 1, and 081's chunk-weighted `Engine.site_means` (chunks 2/2/1, ragged prompts) equals it.
- **Bug found and fixed in the adapter (Caples mode):** `Inference.forward` always registers the capture hooks, so with a
  patcher the splice ran **twice** per layer, and a plain forward was the true model, not the spliced one. Now the splice
  runs in exactly one hook (the patch hook when one is active, else the capture hook, else the site model's own on a
  plain call); nested patches raise. Test: capture-only, plain call and capture + patch all equal their splice (1e-5).
- **D2 (task runner):** `wcm_gemma.py run` = arena C's `wcm_marks.run` on 081's Engine, both families, V3_LOGIT unchanged,
  `--means pooled | per_position`, `--mode native | caples`, resumable per (family, task, data, λ, steps, means,
  substrate). Data `sfc` (100 / 40, SFC's loader with the Gemma tokenizer; D1 default) or `caples` (their filter and
  file-order batches: rc = 3000 records, 187 batches, training = batches 0..83 = 1,344 prompts as their 2B fit walks,
  mean set 640; held-out = first 400 deduplicated length-7 `rc_test` records [ours, below]; D2 default). The seed slot is the last real
  site (layer 25 on gsres is a NullSAE). Tests (tiny random Gemma-2, vocab 256k, real rc prompts, 6 steps):
  PLT sfc pooled, gsres native per_position, gsres caples per_position all write a circuit and the rerun skips it;
  the caples C fill is the file's; a means file made on other prompts is refused.
- **D3 (Caples fill):** `caples_means.py` (caples107) runs their `get_sae_means` / `get_sae_error_means` unmodified
  (bug kept) and also the exact means of the same prompts; saves `theirs`, `exact`, `mean_tokens`, and their LDs per
  batch (first 10 and [0, −1..−9]). `wcm_gemma.load_caples_means` / `apply_caples` load it (eps_bar BOS row must be 0).
  **Test 4 passes on a tiny random Gemma-2** (their side: TL 2.11 `from_pretrained` with the sizes patched in +
  sae-lens 5.3.3 JumpReLU SAEs): max |LD ours − theirs| = 3e-8 for the model, error-dropped and mean-error forwards;
  their `exact` means = ours (1e-4); their `theirs` means = the running-mean recursion (n = batch index + 1) over our
  per-prompt codes, and differ from exact (the bug is visible).
- **Daniel's decisions (applied):**
  1. *D2-WCM held-out subset (ours), deduplicated.* Rule: walk `rc_test.json` in file order under their filter (both
     prefixes 7 tokens); drop every record whose clean or patch prefix equals a clean or patch prefix of the 1,344
     training prompts or of the 640-prompt mean set (a full prompt = prefix + answer, so full-prompt repeats are
     covered); keep the first 400 that remain. **140 records dropped** before the 400th kept one (`split_meta.
     held_dropped_dup`). Their own sets are unchanged: training batches 0..83, mean set (patch prompts of batches
     0..39), eval batches [0, −1..−9] and the filter, overlaps included (tested). **None of their evaluation reads
     rc_test**: their eval batches are batches of `rc_train` (0 and 178..186; batch 0 is a training batch). Only SFC's
     protocol (our `--data sfc`, D1) uses `rc_test`.
  2. *Caples mode is per-position only.* `--mode caples --means pooled` is refused (engine and CLI; tested), and
     `--mode caples` needs `--data caples`. Audit of what Caples mode takes from theirs: the substrate (decode(ĉ) + ε̄,
     BOS stream kept = their `torch.where` + mean error), ε̄ and the C fill (their functions' output, bug kept), the
     prompts, filter, batches and mean set, and the reference (the spliced model, as their `model_logits`).
     **Unavoidably ours, labelled:** the WCM fit itself (Z / A / C / R terms, the A mean = our per-position mean over
     the training clean prompts, R donors = the training patch prompts, 400 × 4 presentations of the training pool,
     amplitudes), the held-out scoring (Z / A / C / necessity on the deduplicated subset). `--caples-source exact` (the
     true means, ours) is an optional, separately keyed diagnostic row; the default is theirs.
- **Test count:** 9 original + 2 × seed_threshold + 2 × BOS means + splice-once + 3 tiny fits + pooled refusal +
  held-out dedup + test 4 = 20 CPU tests, all pass; `test_real_caples_mode_ld_gpu` (marked `gpu`) skips unless
  `ARENA_D_CAPLES_PT` and CUDA are set.
- **For the GPU night (D6 / D7):** (1) `caples_means.py --hf-model unsloth/gemma-2-2b --dtype bfloat16 --device
  cuda` (labelled smoke; the pod runs it fp32 on `google/gemma-2-2b`), then `ARENA_D_CAPLES_PT=<that file> pytest -m gpu
  -k real_caples` (real test 4, bound 1e-2 fp32 / 0.15 bf16, stack noise); (2) D6 identity test (not yet written);
  (3) tiny real-model WCM fits on both families for timings. On the pod `caples_means.py` must run before
  `wcm_gemma.py run --mode caples`.
- **Open for Daniel:** none from D1-D3 (both decisions above applied).
- **D4 (scorer, `score_d.py` + `test_score_d.py`, CPU only):** every circuit scored both ways.
  *Ours* (`.venv`): our own per-position patcher on the deduplicated held-out subset, substrates native (error live)
  and Caples (`decode(ĉ) + ε̄`, ε̄ and the C fill = caples_means.py's file, "theirs" default, "exact" optional row);
  fields as score_task (`task_*`). *Theirs* (`caples107`, `score_d.py theirs`): their `SAEMasks` +
  `run_sae_hook_fn(circuit_mask, mean_mask=True, use_mean_error=True)` on their eval batches [0, −1..−9]; `f_model`
  (= their JSON's faithfulness), `f_spliced`, and SFC-normalised `sfc_*` with m(∅). `all` = ours, then theirs, merged
  into `scores.jsonl`. **Definitional choices (ours unless said):**
  1. Caples membership: (SAE, position, latent) is a member when θ > 0 in `sae_mask_{i}.pt` (their read-out); file i =
     the i-th SAE of their list (layers ascending); their run JSON is checked (indices, total_density) when present.
  2. Sizes: `n_latents` = distinct (site, latent) with a member at some position ≥ 1; `pairs_nonbos` = (site, pos ≥ 1,
     latent) pairs (WCM: n × (L − 1)); `pairs_all` = every θ > 0 pair incl. BOS = their `total_density` (WCM: n × L);
     `n_bos_pairs` reported (a BOS pair has no effect: BOS is never edited).
  3. Frames: Z / A / C with members at α × their live re-encoded value at their own positions, every other (site,
     position, latent) at the fill (zero / our per-position clean mean / C fill). k = d_sae, so the "_topk" fill is the
     plain mean fill (asserted). Caples masks carry α = 1.
  4. Necessity, two columns: `task_nec` (a) members → 0, error live (score_task's); `task_nec_errclean` (b) members →
     0 with the SAE error at **every** dictionary site held at its clean-run value (site output = decode(ĉ_live) +
     err_clean). In Caples mode the error is ε̄ in the clean and edited runs alike, so (b) = (a) by definition (not
     recomputed; `nec_errclean_note`). Residual family only.
  5. Theirs on WCM circuits: members broadcast over positions into their binary `SAEMasks`; **α is dropped** (their
     class has no amplitude). Their means are loaded from caples_means.py's file (their functions' output) into
     `sae.mean_ablation` / `sae.mean_error` ([1, T, d], as their functions leave them); `--recompute-means` reruns them.
  6. `their_protocol` (Caples mode, ours, diagnostic): their evaluation re-stated in our HF stack (binary, α dropped,
     C fill, their batches); it must equal theirs (cross-stack check).
  7. Controls `keep_all` and `empty` are scored with every run (`--no-controls` to skip).
  **Tests (`test_score_d.py`, 6 CPU tests, all pass, 2.6 min; tiny random Gemma-2, the same model and SAEs on both
  sides through a caples_means.py fixture of the real rc prompts; one fake Caples run folder; a 6-step WCM fit in
  Caples mode + a 32-latent subset of it):** keep-all Z / A / C raw LD = full LD **exactly** in both substrates; our
  per-position path on the collapsed WCM circuits = score_task (Engine.score) within 1.8e-7 on all 8 frames in both
  substrates, and the [T, d_sae] broadcast of a collapsed circuit scores the same; theirs (caples107 `SAEMasks`) of the
  Caples mask and both WCM circuits = our re-statement on their batches (`their_protocol`, 1e-4); their model / spliced
  refs = caples_means.py's LDs; their keep-all = their spliced model (1e-5; mean + (c − mean)·1 is not bitwise c);
  sizes = their total_density; a corrupted run JSON is refused. The tiny model's LD is ~0, so ratio fields (Z / A / C
  ratios, necessity) are mostly None there; real numbers wait for the GPU night. Existing 20 adapter tests still pass.

## Status 1 (Daniel approved the local downloads and venvs; CPU only, no GPU)

**Downloaded (only the approved list):**
- `google/gemma-scope-2b-pt-res` @ `fd571b47` `layer_0/width_16k/average_l0_105/params.npz` (302,131,416 B) into the
  WSL HF cache. All five Caples SAEs are now local.
- Neuronpedia Dallas → Austin graph JSON → `out/fidelity/gemma-fact-dallas-austin.json` (2,750,651 B; MD5 = the S3
  ETag `e3e4151b8d660302d540fc52b0eba1af`).

**Gated access (`google/gemma-2-2b`):** access: yes (checked by an authenticated metadata request with Daniel's read
token from an environment variable; the token was never printed or stored). The official weights were not downloaded;
they are for the pod.

**Environments** (WSL; `pip check` clean on both; freezes `env_freeze_ct107.txt`, `env_freeze_caples107.txt`):

| venv | built | deviation from §1 / `setup_env.sh` draft |
|---|---|---|
| `~/venvs/ct107` | circuit-tracer 0.5.3 from git, `direct_url.json` commit = `255d989` (F1 pass), torch 2.13.0+cu130, transformer-lens 3.2.1, transformers 4.57.3, nnsight 0.8.0rc1, huggingface_hub 0.36.2, numpy 2.5.4 | Python **3.12.3** (WSL has no 3.11, no uv); torch from PyPI (cu130 wheels), because the cu128 index stops at 2.11.0; this is 038's working combination on this card |
| `~/venvs/caples107` | sae-lens 5.3.3, transformer-lens 2.11.0, transformers 4.47.1, torch 2.8.0+cu128, numpy 1.26.4, datasets 2.21.0, huggingface-hub 0.27.1, safetensors 0.4.5, wandb 0.31.0 | Python **3.12.3**; added **pyarrow 18.1.0** (the resolver's pyarrow needs numpy ≥ 2, which broke `import datasets`; arena C pins the same) |

**Checks run (CPU):**
- circuit-tracer's own test suite, from a copy of its `tests/` (`out/ct_tests/`, logs `pytest_cpu.log`,
  `pytest_cpu_failed.log`), `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, `-m "not requires_disk"`: **158 passed,
  9 failed, 84 skipped** (CUDA-only). The TransformerLens-backend Gemma exactness tests (`test_small_gemma_model`,
  `test_large_gemma_model`: an injected feature's adjacency column = the change in every downstream pre-activation and
  logit) **pass**. All 9 failures are nnsight-backend tests (gemma, gemma3, llama, gpt-oss) that fetch a config /
  tokenizer from the Hub, which offline mode blocks; arena D uses the TL backend.
- caples107: their SAE-selection code (interp_bottlenecks.py:138-159) on sae-lens 5.3.3's directory picks exactly
  L0 105 / 70 / 82 / 74 / 73 at layers 0 / 6 / 12 / 18 / 24; all five load offline as `jumprelu`,
  `blocks.L.hook_resid_post`, d_sae 16,384. TL 2.11's Gemma-2 config is built in (no gated config fetch), so
  `hf_model=` with the unsloth weights works offline for the smoke.
- WCM adapter tests (`test_wcm_gemma.py`, repo `.venv`, CPU, tiny random Gemma-2 + the cached real dictionaries):
  **9 / 9 pass** (2 min): JumpReLU encode = the circuit-tracer / sae-lens formula, seed-tap bias = b_enc; PLT site
  open gate and no-edit transforms give logits **bitwise** equal to the raw HF model; the PLT read equals TL's
  `ln2.hook_normalized` (x / sqrt(mean x² + eps), fp32) and × (1 + w) gives the HF norm output; a PLT edit lands
  exactly at the `post_feedforward_layernorm` output and spares BOS; resid native open gate bitwise identity; the
  Caples-substrate mode equals a re-statement of their `build_sae_hook_fn(use_mask, mean_mask, use_mean_error=True)`
  splice, open and with a masked circuit (atol 1e-5); real PLT layer 3 and Gemma Scope layers 0 / 6 load with the
  expected shapes and encode / decode as the formula.

**Written:** `caples_smoke.py` (the labelled smoke wrapper of §2; parses; not run, since a CPU run loads the 2B model),
`wcm_gemma.py` (families `gemma-2-2b-plt`, `gemma-2-2b-gsres-gap6`, `JumpReLUSAE`, `GemmaSAEBank`, two site models,
`register()`) + `test_wcm_gemma.py`. PLT loading falls back offline to the cached snapshot `bd577315` (byte-identical
layer files). Not yet: the task runner (arena C's 081-engine path on these families, BOS-aware means, the per-position
Caples fill from their `get_sae_means`), the per-position scorer, the latent-target driver, the real-model identity
test (needs the 2B model in RAM; first GPU night or pod).

**Proposed src changes (proposals only; nothing under `src/` was touched):**
1. *Threshold-aware target activation.* The scorer reads a target latent's activation as `relu(pre)`
   (`src/eval/wcm_eval.py:317, 465`, `src/eval/donor_test.py:227`). For JumpReLU dictionaries the activation is
   `pre · (pre > θ)`, so positive sub-threshold pre-activations would count as active. Proposal: an optional
   `activation(pre)` method on the SAE (default ReLU) used at those reads. Only the latent-target rows (D1 WCM latent
   targets, AG-latent scored by our tests) are affected; the task rows (logit objective) are not.
2. *The arena A risk-2 audit* (every engine path writes x + a decoded code difference at a site) is shared with
   DAN-195; no change expected, but it must be checked once for both arenas.
3. *BOS in means.* Gemma's BOS position carries huge activations (as Pythia's, arena C Finding 1). The site models
   never edit position 0, but pooled A / C means must also exclude it; if the 081 engine's pooled means cannot skip
   position 0 without a src change, that is a proposal (arena A has the same BOS question, its §5 [decide]).

Status: **plan and inventory only. Nothing downloaded, installed or run.** Stretch goal for the rented 8×H100 pod.
Linear: DAN-201 (D1), DAN-202 (D2), DAN-203 (WCM adapter), parent DAN-188. Method details are in
`../specs/attribution-graphs.md` and `../specs/caples.md`; this file only adds what those specs leave open: what to
download, what fits where, the adapter design, the environments, and the pod stage. Where this file and a spec
disagree on a method fact, the spec is the source, except for the four corrections in §0, which were checked against
the code and the authors' saved files.

## 0. New findings that change the plan

1. **Gated weights.** No HF token is configured in the HF cache or environment (`get_token()` is None), so every
   gated call must pass Daniel's read token explicitly as `HF_TOKEN` from an environment variable. **Resolved:
   access to `google/gemma-2-2b` = yes** (status 1 above). On the pod, export `HF_TOKEN` in
   the shell, never in a file.
2. **Caples' 2B configuration keeps a mean error term.** `scripts/sae_gap.sh` passes `--use_mean_error`: every SAE site
   outputs `decode(c') + ε̄`, where ε̄ is the per-position mean SAE error over 640 corrupted prompts (running mean, same
   bug as the codes). The error is not dropped as in the 9B post. The 107 README's D2 line ("the SAE error is dropped")
   should read "replaced by its per-position corrupted mean" for the 2B row. The faithfulness denominator in this path
   is the **true model** (`helpers/utils.py:582-584, 620`).
3. **The spec's 2B anchors are not from the committed script.** `masks/sva/rc_train_saegap6_mean_error/` holds 20 runs
   from four λ grids. Each grid is reproducible with the unmodified CLI (`--start_threshold / --end_threshold`). The
   λ values match `linear_map` exactly:

   | grid | CLI | λ → (size, f vs true model) |
   |---|---|---|
   | A = `sae_gap.sh` as committed | 0.1 → 10, 5 steps | 0.1 → (31, 0.882); 0.527 → (12, 0.653); 1.731 → (3, 0.00); 3.711 → (1, 0.005); 6.467 → (0, 0.005) |
   | B (the spec's anchors) | 0.005 → 0.1 | 0.005 → (790, 0.965); 0.0102 → (370, 0.972); 0.0224 → (157, 0.963); 0.0414 → (87, 0.901); 0.0672 → (47, 0.968) |
   | C | 5e-5 → 0.005 | 5e-5 → (16,951, 0.869); 2.6e-4 → (11,188, 0.869); 8.7e-4 → (5,683, 0.913); 1.9e-3 → (3,120, 0.907); 3.2e-3 → (1,745, 0.945) |
   | D (the Colab grid) | 0.01 → 0.2 | 0.01 → (847, 0.731); 0.0205 → (451, 0.572); 0.0447 → (224, 0.39); 0.0827 → (123, 0.456); 0.1345 → (66, 0.019) |

   Grid D does not fit grids B / C: λ 0.01 gives 847 latents at f 0.73 where B's 0.005 → 790 at 0.965 and 0.0102 →
   370 at 0.972. It was probably made with another code state. **Use A, B, C as fidelity anchors; report D, don't
   match it.** Their model LD on the eval batches is logit_diff / f = 3.214 for every run (e.g. 3.102 / 0.965).
   No seed is set, so expect noise in size (spec: accept ±30%).
4. **Running their script in place would overwrite these anchors.** `interp_bottlenecks.py` writes to
   `masks/<task>_saegap6_mean_error/<...>_run/` relative to the working directory, i.e. exactly the folder above, and
   reads `data/` and `helpers/` relative to it too. It must run from a **copy** of the clone (§5.2), never from
   `third_party/`.

Smaller findings: their script needs `WANDB_MODE=disabled` (it calls `wandb.init` every λ) and an explicit
`--hf_cache` (the default is the authors' cluster path); `get_sae_means` / `do_training_run` default to `cuda:0`.
TransformerLens keeps Gemma's unembedding as a separate copy, so a TL Gemma-2-2B is ~3.2 B parameters: 12.8 GB in fp32,
6.4 GB in bf16 (consistent with 038's measured 6.1 GiB resident). On Gemma's tokenizer, `rc_train.json` has 100,731
length-7 records (N = 3000 is used), `rc_test` 25,269; `simple` is length 3, `within_rc` and `nounpp` 6 (all answers
single tokens).

## 1. Download inventory (sizes from the HF API, 2026-11-03; nothing fetched)

Local = WSL `~/.cache/huggingface/hub` (690 GB free on `/`). `models/` and `outputs_public/` hold nothing for Gemma.

| # | item | source @ revision | files | size | gated | local? | needed by |
|---|---|---|---|---|---|---|---|
| 1 | Gemma-2-2B, official | `google/gemma-2-2b` @ `c5ebcd40` | `model-0000{1,2,3}-of-00003.safetensors` (fp32), `config.json`, `tokenizer.{json,model}`, `tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`, `model.safetensors.index.json` | **10.48 GB** | **yes (manual)** | no | D2 as published (TL loads it by name, fp32); D1 fp32 / their CUDA tests; WCM fp32 |
| 2 | Gemma-2-2B, mirror | `unsloth/gemma-2-2b` @ `25319945` | `model.safetensors` (**bf16**), tokenizer files | 5.25 GB | no | **yes** (complete) | D1 bf16 (038's route via `hf_model=`); local smokes; check bf16(#1) == #2 |
| 3 | PLT transcoders (circuit-tracer `gemma` alias) | `mwhanna/gemma-scope-transcoders` @ `3412e19b` | `config.yaml`, `layer_{0..25}.safetensors` (fp32, 16,384 latents each). **Not** `features/*.bin` (3.98 GB of viewer data; `resolve_transcoder_paths` only fetches `layer_*.safetensors`) | **7.86 GB** | no | **yes**: snapshot `bd577315`; all 26 LFS sha256 equal those of `3412e19b` (the only later commit edits README metadata) | D1, WCM-PLT |
| 4 | Gemma Scope residual SAEs, Caples' picks (gap 6, 16k, L0 closest to 100 from sae-lens 5.3.3's directory) | `google/gemma-scope-2b-pt-res` @ `fd571b47` | `layer_0/width_16k/average_l0_105/params.npz`, `layer_6/…/average_l0_70`, `layer_12/…_82`, `layer_18/…_74`, `layer_24/…_73` (302 MB each) | 1.51 GB | no | layers 6 / 12 / 18 / 24 **yes**; **layer 0 `l0_105` no** (the cache has `l0_46` only) | D2, WCM-gsres |
| 5 | CLT-426k (optional, stretch) | `mntss/clt-gemma-2-2b-426k` @ `b1e9ab37` | `config.yaml`, `W_enc_{0..25}.safetensors` (75.5 MB each), `W_dec_{0..25}.safetensors` (75.5 MB × (26 − ℓ)) | **28.46 GB** (1.96 + 26.50) | no | no | D1-CLT row only |
| 6 | Neuronpedia Dallas graph (F3 reference) | `neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json` (ETag `e3e4151b…`, 2025-08-30) | one JSON | 2.75 MB | no | no | D1 fidelity F3 |
| 7 | SVA task data | `third_party/ScalableSAECircuits/data/sva/*.json` (= SFC's files) | already on disk | — | — | **yes** | D2, D1-behaviour, WCM tasks |
| 8 | Latent-target corpus (only if the latent-rooted rows run) | `Salesforce/wikitext` `wikitext-103-raw-v1/train-00000-of-00002.parquet` (038's corpus) | one parquet | 157 MB | no | **yes** | WCM-PLT latent targets + AG latent-rooted |

Not needed: `mntss/gemma-scope-transcoders` (points at Google's .npz, no lazy loading), `google/gemma-scope-2b-pt-transcoders`,
`mntss/clt-gemma-2-2b-2.5M` (171 GB), any 9B file.

**pip packages** (two new venvs; versions in `setup_env.sh`; ~5-7 GB each with the CUDA wheels):
- `ct107` (D1): `circuit-tracer` from git @ `255d989` (non-editable), Python 3.11 (their CI), `torch==2.13.0` (cu128;
  038's working version on this card), `transformer-lens==3.2.1` (the last TL release that allows `transformers<5`, so
  what their CI resolved on 2026-10-14; also 038's), `transformers==4.57.3`, `nnsight==0.8.0rc1` (the 0.8 release on
  2026-10-14; the TL backend does not use it), `huggingface_hub==0.36.2`, `pytest`.
- `caples107` (D2): Python 3.11, `sae-lens==5.3.3` (latest on 2025-01-27, the HEAD commit date; `< 6` is required by
  `SAE.from_pretrained(...)[0]` and `cfg.hook_layer`), `transformer-lens==2.11.0` (latest then), `transformers==4.47.1`,
  `torch==2.8.0` (cu128; same labelled choice as arena C, needed for the 5070 Ti), `numpy<2`, `datasets<3`,
  `wandb` (imported; disabled at run time), `pandas`, `einops`, `matplotlib`.
- WCM side: the repo's own environment (no new packages; HF `Gemma2ForCausalLM` is in transformers ≥ 4.42).

**Totals.** Pod, required: #1 + #3 + #4 + #6 = **19.9 GB**; with #2 (bf16 cross-check) 25.1 GB; with the CLT 53.6 GB.
Two venvs ~12 GB. Outputs: a full attribution graph is ~170 MB (Dallas), so save only pruned node / edge lists plus
the few graphs needed for F2-F7. **Provision ≥ 150 GB of pod disk for arena D.**
Local, to unblock the local smokes: #4 layer 0 (302 MB) + #6 (2.75 MB) + the two venvs. #1 locally only if the token
exists and an fp32 CPU check is wanted (10.5 GB).

## 2. Memory: what can run on the local card (RTX 5070 Ti 16 GB; WSL 30 GB RAM, ~20 GB free)

| method / smoke | weights on GPU | verdict on 16 GB | smallest local check that still catches bugs |
|---|---|---|---|
| D1 AG, PLT, bf16 (unsloth via `hf_model=`), lazy enc + dec, batch 64, cap 5000 | TL 6.4 GB; PLT lazy | **yes, measured** (038: 6.10 GiB resident, 64-token graphs at cap 16,384) | F3 Dallas graph, bf16 (one night slot, minutes) |
| D1 AG, fp32 (the `from_pretrained` default; their CUDA test) | 12.8 GB + PLT gathers + the 4.4 GB edge buffer on long prompts | **no headroom; don't plan on it** | their CPU tests on random tiny Gemma configs: `pytest tests/test_attributions_gemma.py -k "small or large"` (exactness of edges, no download) |
| D1 AG, CLT-426k | 6.4 GB + per-layer gathers (spec §7.3: 3 GB at batch 64 / 10k features) | maybe, batch 16-32, ≤ 32 tokens; needs 28.5 GB disk | skip locally; pod only |
| D2 Caples, unmodified (`interp_bottlenecks.py`, fp32 TL by name) | 12.8 + 5 SAEs 1.5 + their grads 1.5 + logits ≈ 15.5 GiB; host peak during TL conversion ~25-36 GB | **no**: over or at the limit on GPU and host, needs the gated weights, and CPU is blocked by the hard-coded `cuda:0` | `caples_smoke.py` [ours, smoke only, never a reported row]: imports their `helpers.utils` unmodified, loads `HookedSAETransformer.from_pretrained("google/gemma-2-2b", hf_model=<unsloth bf16>, dtype=bf16)`, passes `device=` explicitly, N = 64, 40 → 4 mean batches, one λ, a few steps; ~8 GB GPU, or CPU in bf16 (~7 GB RAM, slow but feasible) |
| WCM-PLT (HF bf16 + 26 PLT bf16 enc + dec 3.9 GB) | ~9-10 GB + activations | **yes** at ≤ 64 tokens | CPU: open-gate identity (max \|Δlogit\| = 0) and encode parity on 2-3 layers, 4 tokens, fp32 |
| WCM-gsres (HF + 5 res SAEs) | bf16 6.7 GB; fp32 12 GB at 7 tokens | **yes** (bf16), fp32 tight but SVA prompts are 7 tokens | CPU identity + Caples-substrate reference test (§4.3) |

Rule of thumb: everything bf16 fits locally; nothing fp32 that loads Gemma through TransformerLens does. The pod's
80 GB cards take every configuration in fp32, which is also each library's own default (TL, `ReplacementModel`).

## 3. Rows and evaluations

Rule 2 (107 README): D1 and D2 share the model only. Four comparisons, each within one substrate:

| row | substrate | method | root | label |
|---|---|---|---|---|
| D1-AG | Gemma-2-2B + PLT (`mwhanna`) | `attribute()` + `prune_graph` over node thresholds | last-position logits | as published |
| D1-AG-lat | same | A1 truncate at p, A2 `max_feature_nodes=None`, A3 038 `prune_rooted` | a PLT latent (L, p, t) | **our adaptation** |
| D1-WCM | same | WCM, PLT sites (§4.1), λ curve | logit diff (tasks) / PLT latent (targets) | ours |
| D2-Caples | Gemma-2-2B + Gemma Scope res, gap 6 (L0 105/70/82/74/73), mean error | `interp_bottlenecks.py` unmodified, grids A / B / C (+ a dense grid for the curve) | SVA logit diff | "Caples et al.'s code, Gemma-2-2B (an unpublished configuration of the authors')" |
| D2-WCM | same, Caples substrate (per-position mean error, BOS untouched) | WCM logit objective, λ curve | SVA logit diff | ours |
| D2-WCM-native | same SAEs, true error kept | WCM | SVA logit diff | ours; a different substrate, reported separately |

**Task choice for D1 (decision for Daniel).** AG's home turf is a logit root. Proposed primary D1 comparison: the
**SVA tasks** (rc first, then simple / within_rc / nounpp), K = 40 prompts (SFC's evaluation set), one AG graph per
prompt, against one WCM logit circuit on the PLT sites fitted on the 100 discovery prompts. The latent-rooted rows
(D1-AG-lat vs WCM latent targets) are second; they need a target panel and the wikitext data pass on the PLT family.

**Scoring (rule 5, every row twice):**
- **ours:** Z / A / C / necessity (and R when the donor pools exist) through our HF stack, error live (D1, D2-native)
  or in the Caples substrate (D2-WCM, D2-Caples). Per-prompt AG circuits are applied position-collapsed on held-out
  prompts (labelled ours). Caples' per-position masks are scored per position (new scorer path, §4.4).
- **theirs:** D1 = completeness / replacement of the graph pruned to the row's node set, pruned → error (our
  implementation of their metric, spec §9.2), plus their unpruned `compute_graph_scores`. D2 = their own
  `run_sae_hook_fn(circuit_mask=SAEMasks(...), mean_mask=True, use_mean_error=True)` and `logit_diff / model_ld` on
  their eval batches `[0, −1, …, −9]`, applied to WCM circuits through their own `SAEMasks` class (members broadcast
  over positions); plus the SFC-normalised version with m(∅).
- **Size:** distinct (site, latent) pairs; second column = native units (AG feature nodes (layer, pos, feature) and
  error nodes; Caples (SAE, position, latent) pairs; WCM × (L − 1) positions).

## 4. WCM adapter design for Gemma-2-2B (DAN-203; to be written as `wcm_gemma.py` + `test_wcm_gemma.py`)

Same pattern as arena C's `wcm_marks.py`: nothing under `src/` changes; the module registers families in
`model.families.FAMILIES`, site models in `model.hf_adapters._SITE_MODELS` and banks via the `PublicSAEBank` dispatch
patch. Task fits reuse arena C's 081-engine path (`make_engine`, `--means pooled | per_position`, behavioural v3 =
Z + A + C + R, no rank-keep, geometric lr) with a new family; latent targets reuse 062's `driver.py` unchanged.

HF Gemma-2 block (transformers `Gemma2DecoderLayer`): `h = x + post_attention_layernorm(attn(input_layernorm(x)))`;
`out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))`. RMSNorm multiplies by (1 + w), computed in
fp32. BOS is position 0 in every prompt.

### 4.1 Family `gemma-2-2b-plt` (D1): one transcoder site per layer, kind `tc`

- **Read** c = JumpReLU(W_enc x_n + b_enc; θ) with x_n = the RMS-normalised MLP input before the (1 + w) weight, i.e.
  TL's `ln2.hook_normalized`. Recompute it from the `pre_feedforward_layernorm` input in fp32 exactly as the norm does
  (dividing the module output by (1 + w) after the bf16 cast adds rounding; 038 found that folding w is "silently
  wrong").
- **Write** y ← y + (ĉ − c) W_dec at the `post_feedforward_layernorm` output (TL `hook_mlp_out`). The real MLP runs, so
  the transcoder error stays live; all gates open ⇒ the stream is unchanged bitwise.
- **Fitting a read/write-split site into the engine's one-tensor site contract without changing `src/`
  (implemented in `wcm_gemma.GemmaPLTSiteModel`):** arena A's mechanism (arena_a/README.md §5): the hook
  calls `on_site(L, "tc", x_n)` and adds `(y − x_n)` (the engine's x + decode(ĉ) − decode(c), minus x) to the
  `post_feedforward_layernorm` output. One mechanism for GPT-2 (DAN-195) and Gemma; no src change. (The earlier
  concatenation idea, z = [x_n, y], is dropped: it doubled `family.d_model`.) Arena A's risk 2 applies: the edit is
  exact only if no engine path writes anything other than x + (decoded code difference) at a site.
- **Universe for target (L, t):** `tc` sites at layers < L; `same_layer_upstream["tc"] = ()`. Attention, embeddings,
  BOS and errors are not nodes (matches AG's rows).
- **BOS:** the site model restores position 0 after every edit (circuit-tracer zeroes features and errors there; with
  mean fills an unrestored BOS would be edited). Means exclude position 0.
- **JumpReLU and k:** 16,384 latents, L0 5-95 per layer; bank `k = d_sae` as in arena C (sparse fill = plain mean fill;
  stated).

### 4.2 Family `gemma-2-2b-gsres-gap6` (D2): residual sites at layers 0 / 6 / 12 / 18 / 24, kind `resid`

- Forward hook on `Gemma2DecoderLayer` output[0] (TL `blocks.L.hook_resid_post`); the other 21 layers hold arena C's
  inert `NullSAE`.
- Gemma Scope JumpReLU as sae-lens 5.3.3 loads it: pre = x W_enc + b_enc (no b_dec subtraction), c = pre · (pre > θ),
  x̂ = c W_dec + b_dec. Parity test against sae-lens `SAE.encode / decode` on real activations.
- **Modes** (a site-model flag, applied after the engine's transform):
  - `native`: error kept, x' = x + (ĉ − c) W_dec (the usual WCM semantics).
  - `caples` (the like-for-like substrate): output = x̂(ĉ) + ε̄_pos at positions ≥ 1, x at BOS, where ε̄ is the
    per-position mean error computed **with their function** (`get_sae_error_means`, running-mean bug kept, same 640
    corrupted prompts). With every gate open this is their "model + SAEs (mean error)" forward; the reference LD of the
    objective is computed through the same site model, so it is that spliced model automatically.
- Fills: A / C per position (arena C's `per_position` path); C = the corrupted per-position code mean, which is
  Caples' ablation value (theirs with the running-mean bug; ours exact; both reported).

### 4.3 Tests (CPU unless stated)

1. Open-gate identity, both families: max |Δlogit| = 0 against the raw HF model (fp32, 4-8 tokens).
2. Encode parity: PLT vs circuit-tracer's `SingleLayerTranscoder.encode` (in `ct107`) on real activations, support
   Jaccard ≥ 0.99 per layer (spec F10); res SAEs vs sae-lens (in `caples107`).
3. Stack parity: our HF full-model LD and SAE-spliced LD equal TL's within 1e-2 on 16 SVA prompts (spec F2).
4. Caples-substrate reference: our `caples`-mode open-gate LD equals their `run_sae_hook_fn(use_mean_error=True)` LD.
5. Tiny fits (6 steps) on each family: run, write a circuit.
6. GPU (night or pod): timing pilot per family at batch 4 × 7 tokens (tasks) and 4 × 64 tokens (latent targets).

### 4.4 Scorer additions

- Per-position membership (Caples' masks are (position, latent)); read their `sae_mask_{i}.pt` (θ > 0).
- Position-collapsed AG circuits from pruned graphs (038's `seed_circuit` for the latent-rooted rows).
- Pruned-graph completeness / replacement (pruned → error), our implementation of their metric.

## 5. Entry points (to be written; names fixed now so the pod stage can call them)

### 5.1 D1 (`ct107`)
- `ag_fidelity.py`: F1 provenance (`direct_url.json` commit), F3 Dallas (cap 5000, batch 64, prune 0.7 / 0.9; logit
  probs ±0.01 vs " Austin" 0.450, feature-node Jaccard ≥ 0.9 against the JSON), F4 22 pinned nodes, F5 `prune_rooted`
  == `prune_graph`, F6 batch / lazy invariance. Run in bf16 (unsloth) locally and in bf16 + fp32 (google) on the pod.
- Their own tests: `pytest tests/test_attributions_gemma.py` (CPU dummy tests locally; `test_gemma_2_2b` on the pod,
  fp32, google weights), run from a copy of the clone's `tests/` against the installed package.
- `ag_tasks.py`: per SVA prompt, `attribute(ids, model, max_n_logits=10, desired_logit_prob=0.95, batch_size=…,
  max_feature_nodes=None, offload=None)`; `prune_graph` at node thresholds {0.1, …, 0.9, 0.95, 0.98, 0.99}, edge 0.98;
  save pruned node lists + scores, not graphs. Resumable per prompt. Transcoders pinned as
  `"mwhanna/gemma-scope-transcoders@3412e19be8f72d47f8683cb92028a46513222b2b"` (the `@rev` form `HfUri.from_str`
  parses), never the `gemma` alias (it resolves `main`).
- `ag_latent.py` (adaptation, second priority): A1-A3 on a PLT target panel, prompts ≤ 32 tokens.

### 5.2 D2 (`caples107`)
- `run_caples.sh`: `git clone --local third_party/ScalableSAECircuits $RUN/caples_src` (a copy, so `masks/` there is
  fresh and the anchors in `third_party/` stay untouched), check `git -C $RUN/caples_src rev-parse HEAD` = `f54a4ef`
  and an empty `git status`, then from inside the copy:
  `WANDB_MODE=disabled python interp_bottlenecks.py --task sva/rc_train --model_name google/gemma-2-2b --sae_gap 6
  --use_mask --use_mean_error --mean_mask --per_token_mask --run_training_thresholds --start_threshold S
  --end_threshold E --n_threshold_steps 5 --portion_of_data 0.4 --hf_cache $HF_HOME/hub --config_file /nonexistent`
  for (S, E) ∈ {(0.1, 10) = A, (0.005, 0.1) = B, (5e-5, 0.005) = C}; then a denser grid for the curve. One process
  per GPU (`CUDA_VISIBLE_DEVICES`), since `cuda:0` is hard-coded. The token comes from `HF_TOKEN` in the environment.
- `caples_fidelity.py`: model LD on their eval batches = 3.21 ± 0.02 (from the saved JSONs); logged final temperature
  ≈ 644 (S = 187 × 0.4 = 74.8, 84 steps, r_end = 83 / 74.8 × 1.1 = 1.221); sizes within ±30% and f within ±0.05 of
  grids A-C; per-SAE latent Jaccard vs the saved indices (F5).
- `caples_smoke.py` [ours, smoke only]: the bf16 / small-N wrapper of §2.

### 5.3 WCM (repo env)
- `wcm_gemma.py` (+ tests): families, banks, site models, `run` for tasks (as arena C's `wcm_marks.run`), and
  `driver_gemma.py` (062 driver on the PLT family) for latent targets.

## 6. Environment plan

`setup_env.sh` (in this folder) is a **draft, not run**. It builds `ct107` and `caples107` in WSL / on the pod, records
`pip freeze` next to this README, and smoke-imports each library. It downloads nothing beyond pip packages; the weights
are fetched by a separate, explicit step (`fetch_weights` in the script, off by default) once Daniel approves the list
in §1. HF downloads use `HF_HUB_ENABLE_HF_TRANSFER` only if `hf_transfer` is installed in that venv.

## 7. Smoke plan

Before the pod (each item needs Daniel's go for the installs / downloads; GPU items need a free night):
1. CPU, no weights: build `ct107` / `caples107`; run circuit-tracer's CPU dummy-Gemma exactness tests; import
   sae-lens 5.3.3 and resolve the five SAE ids from its directory (must give L0 105/70/82/74/73).
2. CPU, cached weights: WCM adapter tests §4.3 items 1, 2 (PLT), 5; `caples_smoke.py` on CPU in bf16 (needs the 302 MB
   layer-0 SAE).
3. GPU night (if free): F3 Dallas in bf16; WCM tiny fits on both families; one Caples smoke on GPU in bf16; timing
   pilot.

On the pod, in order (GPU 0 runs the gate; the others wait for it):
1. Environments + weights (#1, #3, #4, #6; #5 only if the CLT row is in scope). Check bf16(google) == unsloth tensor by
   tensor on CPU (tells whether local bf16 results carry over).
2. Gate (≤ 30 min): D1 F1 / F3 (bf16 and fp32) / F5 / their `test_gemma_2_2b`; D2 grid A (≈ 5-10 min) against §0.3;
   WCM identity + parity on both families.
3. Production (parallel, one job per GPU, resumable per circuit / prompt / λ).

## 8. Rough pod timing (estimates, not measurements; no AG or Gemma WCM timing exists anywhere we can read)

| stage | work | GPU-h (H100) |
|---|---|---|
| setup | 2 venvs, ~20-50 GB of weights | 0 (≈ 20-30 min wall) |
| gate | as §7 | ≈ 0.5 |
| D2-Caples | grids A-C + one dense grid (~25 λ): ~84 steps of 2 forwards + 1 backward at 16 × 7 tokens, plus ~1,300 single-prompt mean forwards and a ~2 min model load per invocation | ≈ 1 |
| D2-WCM | rc (+ 3 SVA tasks), 9 λ × 2 means × 2 modes, ~1-2 min per fit at 7 tokens | ≈ 2-4 |
| D1-AG tasks | 4 tasks × 40 prompts; ~7k active features per 7-token prompt, all expanded; seconds to a minute per graph | ≈ 1-2 |
| D1-WCM tasks | 4 tasks × 9 λ × 2 means on 26 PLT sites | ≈ 2-4 |
| D1 latent (second priority) | 30 targets × 8 prompts AG (≤ 32 tokens, ~1 min each) + 30 targets × 5 λ WCM (~5 min each) | ≈ 4 + 12 |
| CLT (stretch) | AG + WCM-CLT at L ≤ 12 | not estimated |

Tasks only: ≈ 7-12 GPU-h, i.e. **~2-3 h wall on 8 GPUs** after a ~1 h setup + gate. With the latent rows ≈ 25-30
GPU-h, ~4-5 h wall. The first timing pilot replaces these numbers.

## 9. What a pod stage would call

**Built as `arena_d/pod_stage_d.sh`** (its header is the current stage list; the chain runs it with
`ARENA_D=1` after arena A). The original sketch:

```
source arena_d/setup_env.sh            # venvs; with FETCH=1 also the weights of §1
run_gate.sh                            # GPU 0: ag_fidelity.py, pytest, run_caples.sh grid A, test_wcm_gemma.py
GPU 1-2: run_caples.sh B, C, dense     # caples107
GPU 3-4: ag_tasks.py --task rc|simple|within_rc|nounpp
GPU 5-6: wcm_gemma.py run --family gemma-2-2b-plt  --tasks … --lams …
GPU 7:   wcm_gemma.py run --family gemma-2-2b-gsres-gap6 --mode caples|native …
then:    score_d.py (both evaluations for every row) → out/curves_d.md
```

Each job writes `out/<row>/…jsonl` with one line per finished unit, so a restart skips finished work.

## 10. What can and cannot be validated before the pod

**Can** (given approval for the venvs and the 305 MB of downloads, and a GPU night): environment resolution and
imports; circuit-tracer's own CPU exactness tests; sae-lens SAE-id resolution; WCM adapter identity / parity tests on
CPU; Caples' functions end to end in our bf16 wrapper on unsloth weights; F3 Dallas in bf16; WCM tiny fits; first
timings on the 5070 Ti.

**Cannot:** anything on `google/gemma-2-2b` (no token here; access unknown); Caples' script unmodified (needs those
weights and > 16 GB in fp32); fp32 AG and their `test_gemma_2_2b`; whether google's fp32 cast to bf16 equals the
unsloth file; the CLT; H100 timings; the anchor grids A-C. The WCM adapters (DAN-203) and the transcoder site
(DAN-195) are not written yet; arena D will only be pod-ready if they are built and pass §4.3 before the pod.

## 11. Biggest risks for the pod

1. **Gated weights:** access confirmed; the remaining risk is only getting `HF_TOKEN` into the pod
   shell. Without it, D2 as published cannot run (TL loads the model by name) and D1 falls back to the unsloth bf16
   mirror (038's route, labelled).
2. **Build time:** DAN-203 (two families, the split-site trick, the Caples substrate mode, per-position scoring) and
   the D1 evaluation code are about a week of work in the 107 README's estimate; the plan had four days before the pod,
   shared with arenas C and A. Recommended scope if short: D2 only (Caples grids + D2-WCM on the rc task), then D1 tasks.
3. **Overwriting the authors' anchors** if their script is run in `third_party/` (§0.4).
4. **Anchor ambiguity and noise:** the committed grid A is mostly empty circuits below λ 0.5; the informative anchors
   (B, C) come from uncommitted CLI arguments; no seed.
5. **Precision drift:** unsloth is bf16, google fp32; circuit-tracer's default is fp32 while Neuronpedia's graph was
   made with circuit-tracer 1.0.0 (before PR #116's `find_threshold` change), so F3 counts may move slightly.
6. **Dependency drift:** circuit-tracer's open `transformer-lens>=2.16` would take TL 4.x if transformers were not
   capped; pin everything (§1). sae-lens 5.3.3 pins old packages (`pyzmq==26.0.0`, `datasets<3`, `safetensors<0.5`);
   use Python 3.11.
7. **Disk and output size:** full graphs ~170 MB each; the CLT is 28.5 GB.
8. **HF repo drift:** `mwhanna` main moved on 2026-10-28 (README only); always pin `@3412e19b…`.
