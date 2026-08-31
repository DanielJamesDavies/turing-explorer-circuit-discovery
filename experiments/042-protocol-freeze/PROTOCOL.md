# Frozen comparison protocol + 8xH100 run spec

Status: DRAFT for Daniel's sign-off. The 6-seed local pilot (2026-08-22
to -26) proved the pipeline end-to-end; this file freezes exactly what
the large run repeats, so the H100 results are the pilot's protocol at
scale and nothing else. Anything not listed here is out of scope for
the big run.

## 1. What is frozen

Three self-contained ARENAS. A comparison lives entirely inside one
arena; nothing is ever compared across arenas.

| arena | model | dictionary | node universe | seeds (pilot) |
|---|---|---|---|---|
| gemma-tc | Gemma-2-2B | GemmaScope JumpReLU transcoders (circuit-tracer's shipped "gemma" scan, zero conversion) | per-layer transcoder features + error + embed nodes | 6 @ L4/L6 |
| llama-tc | Llama-3.2-1B | EleutherAI TopK skip-transcoders (131k, k=32), converted weights (RMS fold baked, bf16) | same | 6 @ L4/L6 |
| turingllm | TuringLLM | home SAE bank (k=128 / 40,960) | home sites (MLP + resid bands) | 22-seed panel |

Fixed everywhere: probe_sequence_count=64, 8 windows/seed,
acceptance band [0.8, 1.25] on BOTH zero-fill and mean-fill,
production triple floor for our fits, exam family unchanged from the
panel runs.

### Arms (exact tags in the rows files)

* OURS: `triamp400` (tri-amp), `gate400` (tri-mask);
  lambda sweep via `LAMTAG` -> `triamp400_lam<v>` / `gate400_lam<v>`
  (4 values, pilot-chosen; sweeps are frontier data, production arms
  stay untagged).
* EXT (external methods AS SHIPPED):
  - `ct_published_*`: circuit-tracer attribution + their
    `prune_graph(node_threshold=0.8, edge_threshold=0.98)` verbatim,
    logit-rooted. Human curation excluded (impossible to automate);
    everything else is their pipeline. Variants `_matched`, `_med`,
    `_union` (size policy, section 3).
  - `ct_seed_pinned_*`: identical pipeline + the single-change blocker
    (seed node force-kept, sink exemption). Reported as == published
    when the seed survives, and as EMPTY when it doesn't; never
    silently substituted.
  - `ct_seed_rooted_*`: their pruning body, root vector = seed rows
    (labelled adaptation, not "circuit-tracer").
  - `theirs`, `theirs_x2/x4(/x16/x64)`, `theirs_full`: direct
    adjacency-ranking cuts (appendix-only; NOT labelled
    circuit-tracer).
  - `sfc`, `sfc_x*`, `sfc_full`: SFC-style attribution patching
    (activation x gradient of seed pre-act, exact model, seed-rooted).
  - home matrix arms `abl_ig` / `cf_ig` / `resto` (+`@n`, `@x4/16/64`)
    and `ge-hier` (turingllm arena only).
* HYB (decomposition arms, dashed in figures, never "shipped methods"):
  `<prefix>_amp` = external selection + our support-restricted
  amplitude fit (lambda 0); `ct_seed_rooted_matched_amp`.
* LAD: `null*` (random, fitted alpha), `nullsup*` (support-matched
  null, drawn from anchor-firing latents, firing-count matched),
  `coact_raw` / `coact_amp`.

### Gates (must pass BEFORE any scoring counts)

1. Provenance: circuit-tracer == decoderesearch/circuit-tracer @
   `8f1e2438` (content-based check, `git diff -w --ignore-cr-at-eol`;
   SRC resolved via `circuit_tracer.__file__`). Record in
   `CIRCUIT_TRACER_VERSION`.
2. Identity: prune_rooted(logit root) == their prune_graph exactly.
3. Harness identity on the model (convention probe; exact 0 on home,
   stated drift budget on Gemma TL-vs-HF ~4e-4).
4. Selfcheck: our members re-scored through the external-scoring path
   reproduce the production rows.
5. Published-graph reproduction (Dallas->Austin 22/22 pinned features).
6. Batch/lazy-eager invariance by node-identity Jaccard + edge corr
   (positional diffs are meaningless).

The full harness is `038-transcoder-compare-gemma/
ct_faithfulness.py`; it runs ONCE per environment (so: once on the
H100 image) and its output ships with the results.

### Seed-survival policy (frozen decision)

As-published pruning does not protect the seed (their research prunes
to the logit; survival is incidental). We report all three arms and
let each speak: published (survival rate is itself a result),
pinned (Daniel's one-change blocker), rooted (their body re-rooted).
No post-hoc rescue of dead seeds in the published arm.

### Metrics + statistics (frozen with the figure)

Per point: n (nodes), f0 (zero-fill faithfulness), fm (mean-fill),
sup (necessity), cf. Figure: 2x3 frontier (f0 & sup vs log n), band
shaded, per-seed dots + log-clustered median lines, line break across
>40x size gaps, dotted = HYB. Stats (`040-comparison-figure/
stats.py`): matched-size comparisons only within 3x of our size; wins
by |f0-1| (overshoot is failure); nodes-to-band = smallest measured n
with BOTH f0 and fm in band, right-censored "> largest measured";
bootstrap 95% CI over seeds. Gemma sup quoted as sup_rel when e0 != 0.

## 2. 8xH100 run spec

Goal: the same protocol at ~8x the seeds, to turn pilot readings into
quotable distributions.

### Proposed scale (Daniel to confirm)

| arena | seeds | layers | rationale |
|---|---|---|---|
| gemma-tc | 48 | L4/L6 + a deeper band (e.g. L10/L13) | flagship arena; depth stratification answers "is this an early-layer artifact" |
| llama-tc | 48 | L4/L6 + deeper | error-budget mechanism at scale |
| turingllm | keep the 22-seed panel; add the multiples + sweeps only if any are missing | panel is already the paper's spine |

### Sharding

One arena leg per GPU pair is overkill; the natural unit is the SEED.
Add `SEED_SHARD=<i>/<k>` to the three drivers (theirs_*, sfc_*,
ours_*): each process takes seeds where `index % k == i`. Rows files
become `rows.shard<i>.jsonl`, concatenated after the run (every row
already carries layer/latent, so concatenation is safe; the `done`
guard reads the shard's own file only). 8 GPUs -> k=8, one shard per
GPU, embarrassingly parallel; no cross-process locks needed.

### Budget (from pilot wall-clock, RTX ~16 GB)

* Attribution+prune: Gemma ~4.5 h / 6 seeds; Llama ~2.5 h / 6 seeds.
* Scoring (ours_* incl. CT arms, hybrids, sweeps): dominated by fits
  at 41-66 s each when not memory-boxed.
* An H100 (80 GB, ~3-6x this GPU on these kernels) should clear a
  48-seed arena leg in roughly the pilot's wall-clock per shard;
  the whole matrix comfortably inside a working day on 8 GPUs.
  (Estimate, not a promise — first measure one shard, then extrapolate;
  NEVER extrapolate from a timing taken while memory is spilling.)

### Environment

* venv: pin from the pilot (`pip freeze` of venv-ct + main .venv ->
  `requirements-ct.txt` / `requirements.txt` in this dir) — generate at
  freeze time, before the image is built.
* circuit-tracer: the submodule at `external/circuit-tracer`
  @ 8f1e2438. Run gates (section 1) once on the image.
* Weights on LOCAL NVMe, never a network mount (the /mnt/x lesson:
  216 MB/s uncached vs 28.5 GB/s local; ~6x on attribution).
* Working knobs: `LAZY_ENC=1 BATCH=256 MAX_FEATURE_NODES=16384`;
  `SFC_BATCH=2` on 131k-wide leaves (revisit upward on 80 GB);
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* VRAM cap policy: `MEM_FRAC` such that cap = total - display - margin.
  On headless H100s display=0; the Windows/WDDM shared-memory
  oversubscription problem does not exist on Linux — the allocator
  OOMs honestly. Keep the cap anyway as a fairness guard between
  co-tenant shards if two legs share a GPU.

### Harness traps checklist (every one of these bit the pilot)

1. NEW ARM => register its tag in the completeness `needed` set AND
   check every seed-level short-circuit guard. Three separate
   incidents: hybrid arms, theirs_x mults, matrix fill's `resto@n`
   seed guard. Grep for `in done` in the driver before launching.
2. No blanket per-seed try/except: use the 3-consecutive-failure
   abort. An empty rows file can print "ALL DONE".
3. Smoke test must be big enough to trip its own guards (>= 1 full
   seed, verify rows WRITTEN not just exit 0).
4. After any "ALL DONE", count the new rows before believing it
   (`grep -c <tag> rows.jsonl`).
5. Timings while memory spills are corrupt; discard them.
6. `pkill` patterns must be bracket-escaped (`[o]urs_`) or they match
   the launcher itself.
7. Long chains: `set -e` kills the chain on the first transient OOM —
   retry-per-leg without it.
8. Machine sleep kills WSL chains (local only; moot on the cluster).

## 3. Size policy for external circuits (unchanged from pilot)

Their pruning yields a NATURAL size (report as `_med` at the per-seed
median window size and `_union` across windows) and a MATCHED size
(`_matched`, cut to our n_ref) — both reported, neither alone. The
direct-ranking multiples (x2..x64) exist to draw the frontier between
those points and are appendix material.
