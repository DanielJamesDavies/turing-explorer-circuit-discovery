# 084 — Harder random-circuit nulls (DAN-181)

Built locally, CPU smoke only; nothing at scale has run yet. **Daniel's decisions applied**
(target panel from the 190 sweep targets, WIN = 8, production members in the pool, members' own contribution
deciles with a shortfall table, public panels from 082; see "Decisions" at the end).

The paper's random-circuit paragraph (`main.tex` "Random circuits.", L369–380) and the coefficient triad
(`fig:coef-triad`, L1044–1056) rest on the DAN-17 null: random members of the production circuit's per-site sizes,
coefficients fitted exactly as WCM fits them, 0 of 747 pass. That null is weaker than it sounds:

- **Its members are mostly inert.** They are drawn from latents active *anywhere* on the 48 strongest-train contexts
  (`random_null.live_latents`). Most of them are zero at the positions that feed the anchor, so their coefficients get
  no gradient and only decay. The median α is exactly 0.7975 = softplus(softplus⁻¹(1) · (1 − 0.05 · 0.05)⁴⁰⁰), the
  gradient-free AdamW trajectory (`nulls_core.alpha_inert`; checked in the tests).
- **It fails on its own training contexts too** (Z median 0.00 on the training reference), so it shows that
  unrelated latents cannot be fitted. It does not test whether WCM's *selection* matters.

This experiment adds two harder nulls, the α = 1 literature null for every random set, and the diagnostics that
tell "can't fit" from "fits but doesn't generalise".

## Design

Per target (its production circuit `062/out_full/out/main/circuits/<key>.pt`, its protocol-v1 context record and
training arm), `harder_nulls.py`:

1. **Profile** on the fit's training contexts (arm B: 32 strongest + 16 mid-band; thin targets arm A): every upstream
   latent's anchor lag, corpus frequency decile and contribution to the target (definitions below).
2. **Draws.** For each null and draw, one member set at the production circuit's per-site counts (written to
   `draws/<null>/<key>.d<draw>.json` before the fit, so a resume refits the identical set).
3. **Fit**, exactly as DAN-17: `random_null.install_support_wrapper` (support = the drawn members, λ = 0,
   `binarize="none"`, θ₀ = 40, so only the coefficients move), through `driver.Runner.fit` with the production method
   (`Runner.method(GAMMA, LAM, True)`: triple floor γ 0.25, rank-keep 3e-3, AdamW lr 0.05 / wd 0.05, 400 steps,
   batch 4, the same training contexts and contrast set).
4. **Score** with the production scorer (`driver.Runner.score`):
   - `held`: held-out strongest, the paper's primary set;
   - `train`: 16 of the fit's strongest *training* contexts (`nulls_core.train_subset` over `D_train`, or over
     strongest-train for arm A), with the same 48-context reference set;
   - `*_a1`: every member at α = 1 on held-out strongest (`rescore_alpha1.ones`): the size-matched, nothing-fitted
     null of Shi et al. (2410.13032) and CircuitKIT.
5. **References on the same machine:** `wcm` (the production circuit, held + train) and `wcm_a1`; `dan17` (the
   DAN-17 random circuit of the same target, held + train) and `dan17_a1`. The DAN-17 circuit is the stored one
   where it exists (`062/out_random/circuits/<key>.d0.pt`, then `062/out_random_scale`); otherwise it is drawn and
   fitted here exactly as `062/random_null.py` does (`nulls_core.dan17_draw`: `random.Random("<key>-null")`,
   members live anywhere on strongest-train; `DAN17_REFIT=1`). Each row carries `dan17_source`. All of a target's
   rows are scored on one machine, so the comparisons are not hardware-confounded (H100 vs RTX refits differ, golden
   Jaccard 0.41–0.96).

| Arm | Members | Coefficients | Rows |
|---|---|---|---|
| `wcm` / `wcm_a1` | the production circuit | fitted / 1 | held, train / held |
| `dan17` / `dan17_a1` | DAN-17: live anywhere on strongest-train, uniform (stored, or refit here) | fitted / 1 | held, train / held |
| **`anchor`** / `anchor_a1` (null 1, primary) | anchor-active pool (production members included), frequency-decile matched per member | fitted / 1 | held, train / held |
| **`contrib`** / `contrib_a1` (null 2) | as `anchor`, matched on contribution decile first, then frequency decile | fitted / 1 | held, train / held |
| `uniform` (optional, `NULLS=anchor,contrib,uniform`) | anchor-active pool drawn uniformly | fitted / 1 | diagnostic: isolates the pool from the matching |

### Matching definitions (exact)

- **Anchor lag.** On one training context with anchor *a* (the target's argmax position,
  `protocol_contexts` `arg`), an upstream latent at position *p* ≤ *a* with post-Top-K activation > 0 has lag *a − p*.
  Its lag is the minimum over the fit's training contexts. Positions after the anchor are ignored.
- **Anchor-active pool.** Lag ≤ `WIN` (**default 8**, decided by Daniel; was 3: active at the anchor or one of the 8
  positions before it, on at least one training context), at each of the production circuit's sites, **production
  members included** (decided: a plain draw from the matched pool, the standard random baseline).
  `n_overlap` / `overlap_wcm` / `overlap_frac` record how many production members each draw took (table 8).
  **Final (Daniel): the contribution-matched null EXCLUDES the production members** (a substitution
  test: its top contribution bins held almost only members, so with members in the pool ~54% of its draws were
  members). `EXCLUDE_MEMBERS` is now a list of nulls whose pool drops the members: default `contrib`; `0` = none,
  `1` = every null. The anchor-active null keeps them (`n_overlap` reported).
- **Per-site counts.** The null takes exactly the production circuit's member count at every site. A site whose pool
  is smaller takes all of it (`short_sites` records it, as DAN-17 does).
- **Frequency decile.** Per site, over the latents with a nonzero corpus count (`latent_stats.active_count`, the token
  count from the data pass): decile = ⌊10 · rank / n_alive⌋, ascending, ties by index; never-active latents are −1.
- **Contribution.** Per latent, the mean over the training contexts of |Σ_{p ≤ a} a_p · ∂pre/∂a_p|, where pre is the
  target's encoder pre-activation at the anchor and a_p the latent's natural code. Every upstream site is replaced by
  decode(its own natural code) + its clean residual, with SFC pass-through gradients (ig_baseline's instrument at
  α = 1 with a zero floor, sparse leaves). So it is activation × gradient with the total (mediated) effect.
- **Contribution bin (matched).** Per target, the production members' *own* contribution deciles: edges at their
  10th … 90th percentiles give bins 0–9 (10% of members each); pool latents below the members' minimum are bin −1,
  above their maximum bin 10. Matching on these bins reproduces the members' contribution distribution at decile
  resolution. Deciles over the pool would not: on the smoke target 194 of 195 production members sit in the pool's
  top contribution decile, so pool deciles would only say "top 10%".
- **Contribution decile (reported).** Per target, rank deciles over the union of the anchor-active pool (all sites)
  and the production members; table 5 shows every set's histogram on this common scale.
- **Matching.** For each site, the members in random order; each takes the pool latent (without replacement) whose
  bin is nearest its own: `anchor` bin = (frequency decile); `contrib` bin = (contribution bin, frequency decile),
  compared lexicographically, so contribution is matched first. Ties between bins and within a bin are uniform.
  `n_exact` (members matched to their exact bin) and `dist0` (summed |first-bin difference|) are recorded per draw,
  and `by_bin` (per member first bin: members, drawn, exact, summed distance), pooled into the **match shortfall
  table** (report table 9).
- **Seeds.** Each (target, null, draw) has its own RNG (crc32 of the three), stable across shards and resumes.

### Targets

Decided: about 128 targets per model, drawn from the panel the other comparisons use.

- **TuringLLM: 126 = 14 per band × kind cell** from the 190 sweep targets with a production circuit
  (`062/targets_sweep.txt`, the DAN-159 / 083 panel; `7.attn.17087` has no circuit). `make_panel.py` writes the frozen
  `targets_turingllm_nulls.txt` + `.json` (md5s, counts). The first 63 (7 per cell) are the **63 targets of the DAN-17
  sweep run** (`062/out_random`, itself a seeded 7-per-cell draw from the same list); the other 7 per cell are a
  seeded draw (`nulls_core.PANEL_SEED` 20261024) from the rest of each cell. The union of two uniform draws, so each
  cell is still a uniform sample. The first `SUB_N` = 63 get `SUB_DRAWS` = 4 draws of each null; the rest get 1.
- **DAN-17 pairing.** 65 of the 126 have a stored DAN-17 circuit: all 63 subsample targets (`out_random`, 2 draws
  each) plus 2 in `out_random_scale`. Only 8 of the 190 sweep targets are in the 747-target DAN-17 run, so the old
  "first 252 of DAN-17's order" pairing does not carry over. **Chosen:** the other 61 get a DAN-17-style draw fitted
  here with DAN-17's own sampler and RNG seed (`DAN17_REFIT=1`), so every target has a `dan17` row on the same
  machine. Report table 7 counts stored versus refit.
- **Public models: 082's `targets_<model>_nulls.txt`** (`make_targets_public.py`, subset `nulls`, a stratified subset of
  `random`, so of main): Pythia-70m 128 = 8 per component (16 components), GPT-2 small 128 over its 47 sites (2–3 each).
  The list is written subsample first. Its first 64 are a stratified half (Pythia 4 per component; GPT-2 1–2 per site)
  and get 4 draws (`SUB_N=64`; manifest `nulls_sub_n`). The DAN-17 reference is always refit there (no stored run).
- `TARGETS=<list>.txt` takes any list in file order; `STRATIFY=1` or a `.csv` re-draws stratified (family-aware
  bands: 12 layers L0-4 / 5-7 / 8-11, otherwise thirds, so Pythia-70m L0-1 / 2-3 / 4-5).

## Diagnostics and report

Every row carries the scorer's full output plus: `pass` (DAN-8), the coefficient distribution (`alpha_median`,
quantiles, `alpha_moved` = share with |α − α_inert| > 0.01, `alpha_up` / `alpha_down`, the modal share), the member
set's profile (`set`: frequency- and contribution-decile histograms, median log10 corpus count, median contribution,
anchor-active and prefix-active shares), the draw's match diagnostics, the overlap with the production members, and
timings. `profile*.jsonl` holds one row per target: pool size per lag window, the production and DAN-17 members'
coverage by window, and both sets' profiles.

`REPORT=1` (or `report.py`) writes `out/report.md` and `summary.json`:

1. per arm: pass rate on draw 0 (one per target) with a 95% Wilson interval, the all-draw rate, Z / A / C / necessity
   median [IQR], induce, the share with any single score in band;
2. pass rate by band × kind per arm;
3. training split vs held-out per fitted arm (paired): pass on each, "fits but doesn't generalise", Z ≥ 0.8 on train;
4. coefficients: α median, moved / up / down shares, the share of circuits with every coefficient inert;
5. member sets (production, DAN-17, each null): decile histograms, corpus frequency, contribution, anchor activity;
6. per-target quantile of the WCM score in its null draws (Z, A, C, necessity, min(Z, A, C)): median quantile, share
   where WCM beats every draw, share where WCM passes and every draw fails, share of targets with any passing draw;
7. pool: anchor-active pool size and the production members' lag coverage (w0 … w63, prefix); DAN-17 stored / refit
   counts; whether members were in the pool;
8. match quality per null: exact-bin share, mean bin distance, drawn / reference size, sites with a short pool, and the
   overlap with the production members (count and share of the drawn set);
9. **match shortfall** by the member's own first bin (anchor: frequency decile; contrib: the members' contribution
   bin −1..10), every draw pooled: members, % drawn, % matched exactly, mean bin distance.

## Commands

```bash
# the frozen TuringLLM panel (CPU; refuses to overwrite a differing list)
PYTHONPATH=src .venv/bin/python experiments/084-harder-nulls/make_panel.py
# local GPU queue (TuringLLM, the 126 panel; resumable; waits for the other experiments' GPU jobs)
wsl -e bash -lc 'cd "<repo>" && bash experiments/084-harder-nulls/run_local.sh > experiments/084-harder-nulls/run_local.log 2>&1'
# one process by hand / multi-GPU through 062's launcher (SHARD=i/k, per-shard row files)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python -X utf8 experiments/084-harder-nulls/harder_nulls.py
GPUS=0,1,2,3 P=6 MERGE=0 SCRIPT=experiments/084-harder-nulls/harder_nulls.py bash experiments/062-h100-protocol-v1/launch.sh
# report (no model)
PYTHONPATH=src .venv/bin/python experiments/084-harder-nulls/report.py
# tests (CPU)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/084-harder-nulls/test_084_nulls.py -q
```

Env: `TARGETS (the panel) N_TARGETS (0 = all) N_DRAWS (1) SUB_N (63) SUB_DRAWS (4) NULLS (anchor,contrib) WIN (8)
A1 (1) SPLITS (held,train) REFS (wcm,dan17) RN_DIR (comma list; TuringLLM default 062/out_random,062/out_random_scale)
DAN17_REFIT (1) EXCLUDE_MEMBERS (contrib) MOVE_TOL (0.01) ATTR_BS (4) OUT84 LIMIT PROFILE_ONLY STRATIFY REPORT`, and the
driver's `FAMILY DATA STORES LAM STEPS SHARD SAE_PRELOAD CTX_DIR`.

### Public models and the pod (082 chain)

The script imports 062's `driver` / `random_null` / `rescore_alpha1`, so `FAMILY` / `DATA` / `STORES` / `LAM` carry
over, and it reads `K`, kinds, layers and `latent_stats` from the family's bank and stores. `chain.sh` runs it as the
`harder_nulls` arm of S6 / S9 (it **replaces** the DAN-17 `random` arm, D3), on `targets_<model>_nulls.txt` with
`SUB_N=32 SUB_DRAWS=2 N_DRAWS=1 WIN=8 EXCLUDE_MEMBERS=contrib REFS=wcm RN_DIR= DAN17_REFIT=0` on the public
models (final: no DAN-17 reference refits there; TuringLLM keeps `REFS=wcm,dan17` with the 65 stored + 61
refits on its 126 panel, `SUB_N=32`) (`chain.sh nulls_env`), and the 073 resample
stage reads the anchor null's draw-0 circuits as its random variant. The TuringLLM run is the chain's track stage T4
(after the held-out rebuild, with `CTX_DIR` = the rebuilt records).

## Smoke (GPU, the decided design)

RTX 5070 Ti, `STEPS=8` (plumbing, not results); outputs in the session scratchpad (not kept in the repo).
- **TuringLLM**, panel targets `3.attn.11307` (position 0: stored DAN-17 from `out_random`) and `3.attn.37473`
  (position 63: DAN-17 refit in-process), `SUB_N=1 SUB_DRAWS=2`:
  - 30 rows, 0 errors, 83 s in total;
  - WIN 8 pool ~82k latents; the production members are 99.7–100% anchor-active at WIN 3 and 100% at WIN 8 on
    these two targets;
  - the refit's α is judged against the 8-step inert value (0.993); the stored one's against 400 steps (0.7975);
  - report tables 7–9 render (stored 1 / refit 1; members included; shortfall by bin).
- **Pythia-70m** (069 smoke stores, `out_smoke_pythia/run`; `SAE_PRELOAD=0`, `RN_DIR=` so DAN-17 is refit): 2
  targets, 30 rows, 0 errors. Overlap 1.0 everywhere, because the 8-step smoke circuits keep every latent (see the
  10-23 note below).
- Tests: `test_084_nulls.py` 28 passed. New tests cover:
  - the panel (prior first, 14 per cell, seeded, interleaved, short cells) and the frozen list;
  - DAN-17's sampler reproduced;
  - members included by default, with overlap recorded;
  - the shortfall counts (`by_bin`) and report tables 8 / 9;
  - the decided defaults in a fresh interpreter.

## Smoke (CPU, original design)

All CPU, `CUDA_VISIBLE_DEVICES=` (the GPU was busy). STEPS = 8, so null fits are barely trained; the smoke checks
plumbing, not results.

- **TuringLLM** `3.attn.22257` (first target of the DAN-17 order), 2 draws of each null, every arm and split:
  18 rows, 0 errors (`out_smoke1/`, `report.md`). Profile: pool 55,620 anchor-active latents over 9 sites; the 195
  production members are 65% anchor-active at WIN = 3 and 100% within 8 positions (DAN-17's set: 28%). Production
  members: 90% in their site's top frequency decile, 99% in the pool's top contribution decile; DAN-17's set: spread
  over frequency, 57% in the bottom contribution decile. Draws: `anchor` matched every member to its exact frequency
  decile; `contrib` matched 51% exactly (mean 1.4 bins off), skewed low, because the non-member pool has few latents
  as strong as the production members' upper half (table 8). At 8 steps: WCM rescored here Z 1.00 / A 1.44 / C 0.78
  (production row on the H100: 1.08 / 1.61 / 0.80); DAN-17 0.00 everywhere (α moved 41%); anchor 0.00 / 0.01 / 0.00;
  contrib 0.00 / 0.56 / 0.00, necessity 0.23. Reference α: WCM 99% moved from the gradient-free value 0.7975.
- **Pythia-70m** (069 smoke stores; `out_smoke_pythia/`): driver main (STEPS 8) → `random_null.py` → harder nulls →
  report on 2 targets, 30 rows, 0 errors. At 8 steps the production circuits keep every latent (98k / 262k), so
  this run used `EXCLUDE_MEMBERS=0` (with exclusion the pool is empty; such a draw is now recorded as
  `skip: empty_support` instead of a rejected fit).
- **Profiles of the first 9 targets** (`out_profile9/`, `PROFILE_ONLY=1`, one per band × kind): pools of 56k–218k
  anchor-active latents; production members anchor-active at WIN = 3: 99–100% on 6 targets, 66–84% on 3; 65–90% of
  production members in their site's top frequency decile and 96–100% in the pool's top contribution decile. The
  DAN-17 sets: 16–25% top frequency decile, 3–6% top contribution decile, 18–29% anchor-active. So the DAN-17 null
  differs from the production circuits on all three axes the new nulls match.
- **Contribution instrument**: the sparse activation × gradient equals ig_baseline's dense
  `InterpolatedCodeInstrument` (α = 1, zero floor) to 9e-8 relative on `3.attn.22257` (8 contexts; totals 13.3195 =
  13.3195).
- Tests: `test_084_nulls.py` 22 passed (samplers: per-site counts, anchor activity, member exclusion, exact and
  nearest-bin matching for both nulls, determinism under seed; lags, deciles, member bins; α inert value = DAN-17's
  0.7975; the α = 1 scoring path and the training-split record; report logic on synthetic rows). Full suite
  `pytest tests`: 1664 passed, 3 skipped.

## Timings and runtime estimates

Measured on CPU (16 cores, WSL, 30 GB): TuringLLM start-up 260 s and **18.4 GB RSS** (two TuringLLM CPU jobs do not fit
in WSL's 30 GB: they get OOM-killed); profile 50–115 s per target warm (lags 9–42 s + contribution 30–93 s, 9–35
sites; 166 s on the first, cold target); one score
200–270 s; an 8-step null fit 150–190 s. Pythia: one score 50–75 s; an 8-step fit ~60 s.

GPU estimates (not measured here: the GPU was busy). Per-unit costs from the runs that did measure them: the DAN-17
null ran on the RTX 5070 Ti at 49.5 s per target (fit + held score + live pass; `062/out_random_scale/run.log`), the
062 eval at 3–6 s per score, so fit ≈ 42 s and score ≈ 5 s locally; the profile is assumed ≈ 6 s (CPU: 0.7 of a
score). Per target: 6 reference scores, and per null draw 1 fit + 3 scores.

Revised for the decided design. Per target: 3 WCM reference scores and 3 DAN-17 scores (the DAN-17 fit is
added when it is refit); per null draw, 1 fit + 3 scores.

| Run | Fits | Scores | RTX 5070 Ti | H100 (TuringLLM r = 2.0–3.4 per GPU at P = 6, 062; public r = 1.5–2.7, pod plan) |
|---|---|---|---|---|
| **TuringLLM panel**: 126 targets, 2 nulls × (63 × 4 + 63 × 1), 61 DAN-17 refits | 691 | 2,646 | **~11.9 h** | ~3.5–6 GPU-h; ~30–45 min on 8×H100 |
| **Pythia-70m nulls list**: 128, 2 nulls × (64 × 4 + 64 × 1), 128 refits (fit 35 s, score 4 s) | 768 | 2,688 | ~10.6 local-h | ~4–7 GPU-h |
| **GPT-2 small nulls list**: 128, same (fit 195 s, score ~15 s) | 768 | 2,688 | ~53 local-h | ~20–36 GPU-h |
| GPT-2 with `SUB_DRAWS=2` (pod-plan cut) | 512 | 1,920 | ~36 local-h | ~13–24 GPU-h |
| GPT-2 with `SUB_DRAWS=2`, `REFS=wcm` (no DAN-17 refit) | 384 | 1,536 | ~27 local-h | ~10–18 GPU-h |

GPU smoke (RTX 5070 Ti, 8 fit steps, so not representative of the 400-step costs): TuringLLM 30–53 s per
target for 15 rows; Pythia 37–124 s per target.

## Decisions (Daniel) and what they imply

- **Panel:** ~128 targets per model drawn from the comparison panel (above): TuringLLM 126 from the 190 sweep targets;
  Pythia 128 = 8 per component; GPT-2 128 over its 47 sites. Extra draws (4) for 7 per cell on TuringLLM (63), and
  for the stratified half (64) on the public models.
- **WIN = 8.**
- **Production members included** in the pool (`EXCLUDE_MEMBERS=0`), overlap reported (table 8).
- **Contribution bins** stay the members' own deciles; the shortfall table (9) is reported.
- **Public panels** come from `082/make_targets_public.py` (`nulls`).

**What inclusion does to the contribution-matched null (GPU smoke, TuringLLM `3.attn.11307` /
`3.attn.37473`, 8 fit steps).**
- With the members in the pool, every member found an exact bin: shortfall 0%, against 51% exact under exclusion on
  10-23.
- But the `contrib` draws took **~54% production members** (overlap 147–153 of ~280). The anchor-null draws took ~1%.
- So in the top contribution bins the contrib null mostly re-draws the production circuit itself. Its numbers then
  partly measure WCM, not a null (on the smoke, necessity 1.00 and C 0.14 versus 0.00 for anchor).
- **Daniel should confirm** whether `contrib` keeps members (the decision as stated) or excludes them while `anchor`
  keeps them. Either way the report shows the overlap per arm (table 8), and table 9 shows where the non-member pool
  runs short (by bin).

The original options are kept for the record:

1. **WIN** (the anchor-active window, default 3). The issue says "at, or just before". On the first 9 targets (one
   per band × kind, `out_profile9/`), WIN = 3 covers 99–100% of the production members for every MLP / resid target
   and the layer-6 attention target, but 66% (3.attn), 84% (9.attn) and 84% (11.mlp); WIN = 8 covers ≥ 99% everywhere.
   Options: keep 3 (strict reading; recommended, with the coverage reported), or 8, or add a lag-matched variant. The
   profile rows report coverage at 0/1/2/3/4/8/16/63 either way; `PROFILE_ONLY=1` over the 252 is cheap (~1–2 min per
   target on CPU) and settles it before any fit.
2. **Member exclusion** (default on: the issue's "anchor-active non-members"). Off gives a pure random draw from the
   pool that may contain production members.
3. **Contribution bins** = the production members' own deciles (not pool deciles; see the definitions). On the smoke
   target only half the members found a same-site non-member of their own bin: the contribution-matched null is
   infeasible at the top, which is itself evidence that selection picks the strongest contributors. Report table 8
   with it.
4. **Draws**: 1 per target + 4 on the first 63 (7 per cell). `SUB_DRAWS=2` saves ~4 h locally but makes the quantile
   test 3-valued.
5. **Public models**: the stage runs on the chain's `random` list; the cut `N_TARGETS=128 SUB_N=32 SUB_DRAWS=2` is
   recommended for GPT-2 (cost table).

## What it resolves in the paper

- **Random-circuit paragraph (L369–380).** Report the DAN-17 null *with its caveat* (its coefficients are inert:
  median α at the gradient-free value, `alpha_moved` ≈ 0; it fails on its training contexts too), then the
  anchor-active and contribution-matched nulls as the tests of selection: pass rate with interval, median Z / A / C,
  training vs held-out, and the per-target quantile of the WCM score. Add the framing from DAN-46: literature nulls
  (Shi et al. 2410.13032, CircuitKIT) are size-matched and fit nothing (our `*_a1` rows give that number directly);
  once circuits carry fitted coefficients the right control fits coefficients on the random members too.
- **`fig:coef-triad`.** Panel (a) gains the anchor-active and contribution-matched bars next to DAN-17 (by band ×
  kind, table 2); panel (b) their Z / A / C medians and IQRs (table 1). The α = 1 bar can add the random sets at
  α = 1. The caption states the inert-coefficient caveat for the DAN-17 bar.
- **Frequency objection (DAN-46 tier 3).** Table 5 shows each null's frequency-decile distribution next to the
  production members'.
