# 071 — Contrast-context sweep (DAN-64)

Redesigned from an earlier 8-arm version; renamed from `065-contrast-sweep`, which clashed with
`065-circuit-describer`. Does the contrast source WCM trains its C term on matter? Four training sources, all
post-Top-K silent: **close** (protocol v1 production), **random**, **distant**, and **fused** (an explicit mix of
the three, added later). close, random and distant are also the three eval sets. Everything else is protocol v1
as the 15k run executed it (`paper/protocol.md`): activating contexts, split, γ, rank-keep, λ policy, steps,
scorer, pass rule.

This is a **test of contrast sources, not a production decision**. There is no decision rule and no winner: the
report gives the 4 × 3 train-source × eval-source table and its descriptive statistics (section 4).

**Status:**
- **Three-arm design:** 19 unit tests, a CPU dry run (`out_dry2/`, 1.attn.17708, 155 s: all three sets
  64 + 32 reserve, fire rate 0, overlaps 0, similarity close 0.886 > random 0.680 > distant 0.228, close re-selection
  Jaccard 0.969 on CPU), and the local RTX smoke (`out_smoke2/smoke.md`, 8 targets, all checks ok). On those 8
  targets, circuits trained on close generalise poorly to the random and distant eval sets; those trained on random
  do well on all three.
- **fused arm:** code written and CPU-checked. 28 unit tests pass. Two CPU dry runs on 1.attn.17708,
  `MODE=sets`:
  - `out_dry3/` (144 s, all four sets built; fused there is the superseded `071-fused-v1` merged-then-split
    version). All four sets are 64 (48 + 16) + 32 reserve, fire rate 0, with no activating overlap. All cross-set
    overlaps and held-out-in-training counts are 0. close, random and distant are identical to `out_dry2`'s (ids,
    tokens, reserve, version).
  - `out_dry4/` (113 s): `out_dry3`'s three eval-set files copied in and resumed unchanged; only fused (v2) built.
    - fused is 22 / 21 / 21 with **train 16 / 16 / 16 and held-out 6 / 5 / 5**, and a 12 / 10 / 10 reserve.
    - Fire rate 0, no activating overlap, every overlap and held-out-in-training count 0. Each sub-mode dropped
      exactly its own source's 96.
    - The parts' median similarity is close 0.871 > random 0.701 > distant 0.258; fused overall 0.704. The 64 contexts
      and the reserve are the same as v1's; only their train / held-out assignment changed.
- Nothing has run on a GPU with the fused arm. The old 8-arm design's smoke (`out_smoke/smoke.md`) and dry run
  (`out_dry/`) are kept as they were.

## 1. Files

| File | What |
|---|---|
| `sweep.py` | Per target: four disjoint contrast sets → fits (4 arms × 3 λ) → 3-set eval battery → random baselines. `SHARD=i/k`, resumable |
| `logic.py` | Pure logic: sources and build order, exclusion filter, fused quotas, per-part split and round-robin merge, set order, set versions, held-out-in-training counts, pass rule, matched size |
| `report.py` | The 4 × 3 table at matched size by band, the fused − random line, sizes, contamination (fused per sub-mode), set integrity, random baselines, timings. `SMOKE=1`: smoke checks |
| `make_targets.py` | Writes `targets.txt` (280), `targets_random.txt` (45), `targets_smoke.txt` (8), `members.txt` (seed 20261009) |
| `smoke.sh` | Local RTX smoke test (section 7) |
| `launch_pod.sh` | 4×H100 launcher (MPS, 16 processes) |
| `test_logic.py` | Unit tests (CPU): `CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/071-contrast-sweep/test_logic.py -q` |
| `out_dry4/` | The CPU dry run of the current fused (v2; one target, sets only; the eval sets resumed from `out_dry3`) |
| `out_dry3/` | The first four-set CPU dry run (its fused file is the superseded v1, refused by version) |
| `out_dry2/`, `out_smoke2/` | The three-arm design's CPU dry run and local smoke (`out_smoke2` resumes into the four-arm smoke, section 7) |
| `out_smoke/`, `out_dry/`, `smoke.log` | The old 8-arm design's smoke and dry run (kept; not readable by this design) |

`sweep.py` imports 062's `driver.py` (`Runner.contexts / fit / score`), 062's `random_null.py` (the support-restricted
engine), 049's scorer and 059's harness, and edits none of them.

## 2. Design

### Targets

Targets are a stratified sample of the 15k run (`make_targets.py`; unchanged from the old design).
- **Frame:** fitted (status ok), not vacuous on held-out strongest, with the context record and production circuit
  on disk. That leaves 14,994 targets.
- **Sample:** 8 per (layer, kind) cell. Layer-0 attention has no upstream SAE site and is not in the run, so there
  are **35 cells, giving 280 targets**.
- **Random-baseline subsample:** 5 per (depth band, kind) cell of the 280, so 45 targets.

Every arm uses the same targets and the same activating contexts: the cached v1 record, copied read-only into
`OUT/ctx`.

### Contrast sources (the training arms and the eval sets)

| Arm | Eval set? | How it is built |
|---|---|---|
| `close_silent` | yes | **v1 production:** `rec["neg"]` from the cached record, byte-identical to what the 15k run trained on, in its stratified order. It is re-selected once (`select(..., "close")`) to check the selector reproduces it (`v1_match`), and that selection continued past 64 is its reserve |
| `random_silent` | yes | `select(..., "random")`: a deterministic per-target shuffle of the candidate pool, kept if post-Top-K silent |
| `distant_silent` | yes | `select(..., "distant")`: the least similar candidates first (`seq_repr` cosine to the top-context store), kept if silent |
| `fused_silent` | **no** (training only) | Sub-mode thirds of close, distant and random, with src's fused quotas (`_select_fused`): close n//3 + remainder, distant and random n//3 each. That is 22 / 21 / 21 of the 64 and 12 / 10 / 10 of the reserve. Each part is the sub-mode's selection, kept if silent |

All sources use the selector arguments of the 15k run's close set (`protocol_contexts._SELECT_KW`, `selection_seed`
0). Each set has 64 contexts plus a reserve of 32.

**Why fused.** On the 8 smoke targets, random-trained circuits did well on all three eval sets and close-trained ones
did not. Daniel's hypothesis: random wins because it spans close, mid-range and distant contexts. fused is an explicit
mix of all three, so "variety helps" predicts **fused ≥ random**. The report gives that comparison as one paired line
(section 4). It is still descriptive: no decision rule.

**Disjoint sources.** Each arm is scored on exactly the same three held-out sets, so the training source is the only
independent variable. The sets are built in this order (`logic.BUILD_ORDER`):

1. **close** — the production set as is (64), then its reserve: the close re-selection past the production
   contexts.
2. **random** — excluding every close context: train, held-out and reserve.
3. **distant** — excluding every close and every random context (train, held-out and reserve).
4. **fused** — excluding every close, random and distant context (train, held-out and reserve). Its sub-modes are
   selected in src's order, close, distant, random. Each excludes, on top of that, the contexts fused's earlier
   sub-modes took (set and reserve parts). Each sub-mode's ranking starts with its own source's set, so the selection
   asks past it (64 + 32 + quota + 16). In effect fused-close is the next most similar silent contexts after close's
   96, fused-distant the next least similar after distant's, and fused-random the same random permutation continued
   past random's.
   - **Split per part, then merge (`logic.fused_split_merge`).** Each part is split separately by v1's stratified
     rule, in its selector order. The 16 held-out slots are apportioned to the parts by largest remainder
     (`fused_held_counts`: 22 × 16/64 = 5.5 and 21 × 16/64 = 5.25 give 6 / 5 / 5), so the totals are exactly 48 / 16.
     The training slices are merged round robin (close, distant, random, ...) first and the held-out slices last.
     So the first 48 are training and the last 16 held out, as every consumer slices the set.
   - **Of 64: train 16 / 16 / 16 (exact thirds), held-out 6 / 5 / 5** (close / distant / random). Only the training
     mix matters: the fit's C mean uses the 48, and fused is not an eval set, so its held-out 16 are never scored.
   - The reserve parts are merged round robin. A short sub-mode is not topped up by the others (as in src).
   - Per-sub-mode counts are in the set file (`parts`, `part_counts`), the metadata (`fused_quotas`,
     `reserve_quotas`, per-sub-mode `submodes` with their own `dropped`) and `sets.jsonl` (`part_*`).
   - (A first version merged the 64 round robin and then applied the stratified split, which gave train
     18 / 15 / 15. Its set files, tagged `071-fused-v1`, are refused.)

- **Exclusion (`logic.disjoint_keep`).** A context is dropped by corpus id **or** by token content (two ids with
  identical tokens are the same context). Every source also drops the target's activating contexts (strongest and
  mid-band, by content; the selector already drops the top- and mid-store ids) and repeats of its own contexts, so a
  source's train and held-out never share a window.
- **How.** The selector has no exclusion argument, and src is not edited. `sweep.py` asks for 64 + 32 + 16 contexts
  and filters them in the selector's order. If fewer than 96 survive and the selector is not exhausted, it asks for
  twice as many (at most 4 doublings). The selector's order does not depend on the count asked for (close and
  distant: the similarity ranking; random: one fixed permutation of the candidates, truncated), so this only extends
  the selection. The drop counts are in each set's metadata (`dropped`).
- **Order and split.** The first 64 survivors, in the selector's order, are stratified with v1's
  `stratified_order(-arange(n))`: the stratified 48 train first, 16 held out last. The next 32 are the reserve.
  close keeps the production order and split.
- **Checks (`Sets.build_all`).** After the build, sweep.py asserts, per target: zero pairwise overlap between the
  four sets (content and id, including reserves); no held-out context of any set is a training context of any arm
  (so fused's training contexts are never a held-out context of an eval set); no random, distant or fused context is
  an activating context. A failure is a target error.
  - **One tolerated exception:** the production close set's own exact duplicates. 5 of the 280 production sets
    (4.attn.4255, 5.mlp.7129, 7.attn.29799, 10.attn.35910, 11.attn.7313) have 1 held-out context token-identical to
    one of their training contexts (protocol-v1 §3.7: boilerplate). close_silent uses the production set unchanged,
    so this is recorded (`ho_in_train_close_silent` on the close row, `n_ho_trained` on close × close eval rows) and
    listed by the report, not asserted. None of the 8 smoke targets is among them.
- **Fewer than 4 usable contexts** makes a source unusable for that target (v1's `no_contrast` rule). Its rows are
  `skip`, and the report uses only targets complete for every arm, λ and eval set.

The reserve is no longer used for scoring (there is no held-out swapping). It is kept, and excluded from the later
sources, so the sets stay disjoint beyond their 64 if the reserve is ever used.

### Contamination and integrity log (per target × source, `sets.shard*.jsonl`)

- `frac_fire` (overall, train, held): the share of contexts where the target's max post-Top-K activation over the 64
  positions is above 0.
- `pre_rel_*`: the context's max relu(w·x + b) divided by the activating reference. The reference is the median over
  the 48 strongest-train contexts of the same quantity (`NegContextSelector.posctx_reference`).
- `sim_*`: the max cosine to the top-context store, which is the close ranking's own score.
- `ovl_content_<other>`, `ovl_id_<other>`: overlap with each other set (train + held-out + reserve). Must be 0.
- `ho_in_train_<arm>`: held-out contexts of this set that are training contexts of that arm. Must be 0, except
  close/close (above).
- fused only, per sub-mode `<p>` (close, distant, random): `part_n_<p>`, `part_train_<p>`, `part_held_<p>`,
  `part_reserve_<p>` (counts), `part_fire_<p>`, `part_pre_rel_med_<p>`, `part_sim_med_<p>`.
- **Re-logging.** If any of a target's set rows is missing, all four are (re)written. A set added to a resumed `OUT`
  (fused) changes the other rows' overlap and held-out-in-training columns. The report keeps the last row per
  (target, set).
- `n_activating_overlap`, `n_dup_internal`, `v1_match` (close), the selector metadata with `dropped`, `t_select`.

### Fit

`driver.Runner.fit` is called with `dict(rec, neg=<arm's set>)`. It uses the production config (γ_C = γ_A = 0.25,
rank-keep 3e-3, 400 steps, lr 0.05, train-only means), and the C mean is over the set's first 48. The λ grid is
**{3e-4, 1e-3, 3e-3}** for every arm, fused included. The config is re-applied before every fit, because the engine reads λ at fit
time. `close_silent` at λ 1e-3 is the production circuit, copied from `SRC_OUT/main/circuits` (`REUSE_MAIN=1`).

## 3. Common evaluation battery

- **Every circuit, every eval set.** Each circuit (4 arms × 3 λ, and each random baseline) is scored by the protocol
  scorer (`score_circuit(c, {}, skip_roles=True)`, `EVAL_SPLIT=explicit`) on held-out strongest, once per eval set:
  **close_silent, random_silent, distant_silent**. That gives the 4 × 3 train-source × eval-source table. fused is
  not an eval set.
- **The eval set is the set itself:** its 48 training contexts give the C mean (dense) and its 16 held-out contexts
  are the sufficiency-to-induce set. It is the same tensor for every arm. There is no per-pair swapping.
- **What is common to all cells:** the activating contexts (48 strongest-train reference + 16 held-out strongest),
  hence Z, A and necessity (identical up to GPU numerics).
- **What changes per eval set:** the C mean, eN_*, freeN_*, the contrast anchors and the φ_ind / φ_cf induce scores.
- **Recorded per row:** `n_ho_trained` (held-out contrast contexts of the eval set that the arm trained on; 0, and
  asserted except close × close) and `n_ctr_train_overlap` (48 on the diagonal, 0 off it).

## 4. What the report gives (descriptive; no decision rule)

`report.py` uses targets complete for every arm × λ × eval set, with vacuous targets (`vacuous_tk`, Z denominator)
excluded.

- **Matched size.** n\* = the median size of close_silent at λ 1e-3, the production operating point. Each arm's value
  at its three λ is interpolated linearly in log10(median size) to n\*. Outside an arm's size range the endpoint is
  used and the arm is marked `*`. Each band uses its own n\*.
- **The 4 × 3 table at matched size, overall and per depth band (L0-4, L5-7, L8-11):**
  - the pass rate (protocol v1: Z = `free0_tk`, A = `freeM_topk_tk`, C = `freeN_topk_tk` in [0.8, 1.5] and
    `phi_sup_blind_tk` ≥ 0.9, activation read; C is the eval set's own C; a missing score fails), with the row mean;
  - the medians of Z, A, C, necessity and sufficiency to induce (`phi_cf_alpha_blind_tk`).
- **Circuit size** (median members) per arm and λ, overall and per band; the raw pass rate per λ.
- **Arm differences** in matched pass rate (random − close, distant − close, fused − close, random − distant,
  fused − random), per eval set and averaged, with 95% paired-bootstrap intervals (targets resampled, 2,000
  resamples, seed 0, n\* and sizes recomputed per resample), overall and per band. Descriptive intervals only.
- **Fused vs random (the variety hypothesis), one line:** fused − random in matched pass rate, averaged over the
  three eval sets, paired over the complete targets, with its bootstrap interval (and per eval set). "Variety helps"
  predicts ≥ 0; the line states the estimate, not a verdict.
- **Contamination per source:** fire rate (train / held), pre-activation relative to the activating reference,
  similarity; overall and by band. For fused also per sub-mode: counts (set / train / held-out / reserve), fire rate,
  pre-activation and similarity.
- **Random baselines** (45 targets, per arm × eval set, fused included): pass rate and median C / necessity /
  induce, beside the fitted λ 1e-3 circuits on the same targets.
- **Set integrity:** cross-set overlaps, the held-out-in-training matrix, the production-duplicate targets, errors.
- **Timings.**

## 5. Size and cost

| Item | Count |
|---|---|
| WCM circuits | 4 arms × 3 λ × 280 = 3,360, of which 280 reused (close_silent λ 1e-3) → **3,080 fresh fits** (the three-arm 2,240 + fused's 840) |
| Random baselines (fitted coefficients, 1 draw) | 45 targets × 4 arms = **180 fits** (135 + fused's 45) |
| Scorer runs | (3,360 + 180) × 3 eval sets = **10,620** (7,965 + fused's 2,655) |
| Contrast selections | 280 × 6 (close re-selection, random, distant, fused's three sub-modes) = **1,680**, plus rare doublings |
| Contamination reads | 280 × 4 sets × 64 contexts |

Pod time on 4×H100 with 16 processes under MPS, at ~50 s per fit per process (DAN-64's figure; 062 measured 71 s at
6 processes per GPU):

| Part | Process-seconds | Wall time (÷ 16) |
|---|---|---|
| Fits | 3,260 × 50 = 163,000 | 2.8 h |
| Battery | 10,620 × 1.7 = 18,050 (production H100 figure; at the RTX's 3.1 s: 32,900) | 0.3 h (0.55 h) |
| Sets + contamination reads | 280 × ~15 = 4,200 | 0.07 h |
| Start-up (stagger 16 × 20 s, store loading) | | 0.1 h |
| Shard imbalance (deep targets dominate: L11 fits ~6× L0; ~17-18 targets per shard) | | ~0.35 h |
| **Run** | | **~3.65 h (3.9 h), ~15 GPU-h on 4 GPUs** |

With ~30 min of bootstrap and pod smoke, the session is **~4.2–4.4 h, ~$59–62**. The three-arm design was
~3.2–3.4 h, ~$45–48, so fused adds ~1 h, ~$14. (The old 8-arm design was ~7.5 h, ~$105.) If needed,
`PER_CELL=4 RANDOM_PER_CELL=3 SUFFIX=_4 make_targets.py` and `TARGETS=… RANDOM_TARGETS=…` halve it.

## 6. Commands

**Local (done):**

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/071-contrast-sweep/test_logic.py -q
```

**Local smoke** (RTX, after the GPU queue is free; section 7). Resumes `out_smoke2/` (the three-arm smoke): only
fused is built, fitted, scored and given its random baseline (~10–12 min):

```bash
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/071-contrast-sweep/smoke.sh'
```

**Pod.** Proposed commands, for Daniel's approval before anything runs.

1. **Commit and push** `experiments/071-contrast-sweep` (code, target lists, `members.txt`; `out*` stays local). The
   pod pulls `multi-device`.
2. **Bootstrap** as for 062. Use the global volume `inland_copper_crab` at `/workspace`.

   ```bash
   git clone --branch multi-device https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git /root/turing && bash /root/turing/scripts/pod_bootstrap_global.sh
   ```

3. **Extract** the 280 targets' context records and production circuits (560 members, ~75 MB) from the 15k run's
   tarball on the volume:

   ```bash
   cd /root/turing && D=experiments/071-contrast-sweep && mkdir -p $D/src_out && tar xzf /workspace/results/062-full-20260927-1924.tgz -C $D/src_out -T $D/members.txt && ls $D/src_out/out/ctx | wc -l && ls $D/src_out/out/main/circuits | wc -l
   tar xzf /workspace/results/062-full-20260927-1924.tgz -C $D/src_out --wildcards 'out/main/eval.shard*.jsonl' && ls $D/src_out/out/main/eval.shard*.jsonl | wc -l
   ```

   Expect 280 and 280, then 48. Each command decompresses the whole 1.9 GB tarball once. The second extracts the
   15k run's eval rows. The pod smoke's production re-score check needs them, and without them it is skipped
   silently (on the first pod it then matched bitwise, 0.0000 on every score).
4. **Pod smoke** (1 GPU, ~12 min): 2 targets, all 4 arms, λ 1e-3, the full 3-set battery and the random baseline
   (all 4 arms) on 2.mlp.10238. This is where the production re-score check is binding (H100), and it gives the
   H100 timings for section 5.

   ```bash
   cd /root/turing && D=experiments/071-contrast-sweep && ORDER=list LIMIT=2 TARGETS=$D/targets_smoke.txt OUT=$D/out_podsmoke SRC_OUT=$D/src_out/out ARMS=close_silent,random_silent,distant_silent,fused_silent LAMS=0.001 RANDOM_TARGETS=2.mlp.10238 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src ./.venv/bin/python -X utf8 -u $D/sweep.py && SMOKE=1 OUT=$D/out_podsmoke SRC_OUT=$D/src_out/out PYTHONPATH=src ./.venv/bin/python $D/report.py
   ```

   `targets_smoke.txt` starts with 1.attn.17708 and 2.mlp.10238.
5. **Launch:**

   ```bash
   cd /root/turing && mkdir -p experiments/071-contrast-sweep/logs && chmod +x experiments/071-contrast-sweep/launch_pod.sh && K=4 P=4 nohup ./experiments/071-contrast-sweep/launch_pod.sh > experiments/071-contrast-sweep/logs/launch.log 2>&1 &
   ```

6. **Watch:**

   ```bash
   tail -f /root/turing/experiments/071-contrast-sweep/logs/run.shard0.log
   ```

   For a partial report at any time:

   ```bash
   cd /root/turing && PYTHONPATH=src ./.venv/bin/python experiments/071-contrast-sweep/report.py | head -80
   ```

7. **Copy off before stopping the pod.** Circuits are ~0.3 GB.

   ```bash
   mkdir -p /workspace/results && cd /root/turing/experiments/071-contrast-sweep && tar czf /workspace/results/071-out-$(date +%Y%m%d-%H%M).tgz out logs && md5sum /workspace/results/071-out-*.tgz
   ```

**Host RAM.** Each process holds the model, the SAE bank and the stores. The selector also preloads a negctx token
cache (`token_cache_max_gb` 10). 062 ran 48 such processes on 8×H100, but check `free -g` on a 4×H100 pod against
16 processes. `NO_TOKEN_PRELOAD=1` turns the cache off: selection gets slower, results are unchanged.

## 7. Smoke test: what it checks

`smoke.sh` runs 8 targets (layers 1–11, all kinds) × {close_silent, random_silent, distant_silent, fused_silent} ×
λ 1e-3, with the full 3-set battery and the random baseline (all 4 arms) on 2.mlp.10238. It then runs
`report.py SMOKE=1`, which prints `[ok]`/`[FAIL]` lines and writes `out_smoke2/smoke.md`.

**Contrast sets:**
- Every set is usable on every target and has a reserve.
- **Silence** of all four sets: fire rate ≤ 0.02 per target (a context can flip with the forward batch size).
- No set contains an activating context.
- **Pairwise disjointness = 0** (content and id, including reserves), and no held-out context of any set is a
  training context of any arm.
- **Similarity ordering:** median similarity close > random > distant (the per-target share is printed).
- close_silent's re-selection reproduces the production set: Jaccard mean ≥ 0.97, min ≥ 0.9.
- **fused:** disjoint from the three eval sets (its own overlap columns all present and 0); trains on no held-out
  context of an eval set (`ho_in_train_fused_silent` logged and 0 on every eval-set row); the sub-mode thirds
  (22 / 21 / 21 in the set, train 16 / 16 / 16, held-out 6 / 5 / 5, 12 / 10 / 10 in the reserve); its parts' median
  similarity close > random > distant.

**Battery wiring:**
- Every arm is scored on every eval set; fused on all three eval sets.
- For each eval set, the scorer read **48 + 16 activating and 48 + 16 contrast contexts**.
- **`n_ho_trained` = 0** on every eval and random row (and checked separately on fused's rows); cross-source rows
  share no training contexts.
- Every upstream site is ablated.
- **No errors** in status, eval or random rows.

**Reproduction (pod-smoke check, H100):** the production close_silent circuit, re-scored on close_silent, against
the 15k run's eval rows: median |diff| ≤ 0.005 and max ≤ 0.05 over Z, A, C, necessity and induce. It is binding only
when every status row's device is an H100. Locally it is printed as `not binding`: RTX vs H100 float noise cannot meet
it (the old smoke: median |diff| up to 0.018 on Z). Same circuit size as production is checked everywhere.

**Printed:** timings (fit, scorer, set build) to re-check section 5, the pass rate by arm × eval, and the random
baseline's pass rate.

## 8. Outputs (`OUT`, default `out/`)

| Path | Contents |
|---|---|
| `ctx/<key>.pt` | Copied v1 records |
| `sets/<source>/<key>.pt` | Contrast sets: ids, tokens (stratified), n_train, reserve, metadata, `version`; fused also `parts`, `reserve_parts`, `part_counts` |
| `circuits/<arm>/<λ>/<key>.pt` | WCM circuits |
| `random/circuits/<arm>/<key>.d0.pt` | Random-baseline circuits |
| `sets.shard*.jsonl` | One row per target × source: contamination, similarity, cross-set overlaps, held-out-in-training counts, selector metadata |
| `eval.shard*.jsonl` | One scorer row per target × arm × λ × eval set, plus `n_nodes`, `t_fit`, `fresh_fit`, `n_ho_trained`, `n_ctr_train_overlap`, `t_score` |
| `random.shard*.jsonl` | The same per target × arm × eval set, plus `n_ref`, `n_drawn`, `short_sites` |
| `train.shard*.jsonl` | Fit stats and training curves (062's `train_row`) |
| `status.shard*.jsonl` | Per-target timings, `device`, `ho_close_own` |
| `report.md` / `smoke.md` | The report / the smoke checks |

## 9. Notes

- **Set files are versioned per source** (`logic.set_version`). close, random and distant keep `071-disjoint-v1`,
  the tag the three-arm design wrote, because their construction is unchanged (fused is built after them and does
  not affect them). So `out_smoke2/sets` stays valid and a resumed `OUT` only builds fused. fused carries
  `071-fused-v2` (`071-fused-v1`, the superseded merged-then-split fused, is refused). A cached set from another design (e.g. `out_smoke/sets`, no tag) is refused, so a new design never
  resumes on old sets: use a fresh `OUT`.
- **Distant pool.** The selector keeps the `max(m·128, 2048)` least similar candidates for a request of m, not the
  pipeline's `distant_pool_size` 512. The order is the same for any m, so this only sets how deep the scan can go.
- **Scorer split.** `EVAL_SPLIT=explicit` governs only the activating split. The contrast split is always
  `split_n(n)`. That is exact for these sets: they hold 48 + 16, or n − round(n/4) + the rest when shorter.
- **Old references.** `experiments/062-h100-protocol-v1/queue_2026-10-09b.sh` (already run) still names
  `065-contrast-sweep`; 062 is not edited here.
