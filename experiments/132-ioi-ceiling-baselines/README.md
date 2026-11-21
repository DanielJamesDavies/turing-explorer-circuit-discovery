# Exp 132: IOI ceiling and published baselines on the 081 substrate

Overnight 2026-11-19 (agent ioi-base; shares the GPU with exp 133, ioi-wcm). No src or existing-experiment file was
changed. Every number below is from the 081 scorer, unchanged.

## Question

The task is IOI in GPT-2 small with the OpenAI v5 32k Top-K SAEs. This is exp 081's substrate: 48 sites (attn,
resid_mid, mlp and resid, × 12 layers). Scoring is the 081 scorer: `eval.wcm_eval.score_task` through 081's engine, on
081's held-out IOI items, with 081's A / C means. The pass rule is Z, A, C ∈ [0.75, 1.5] and nec ≥ 0.9.

1. Can ANY circuit on these dictionaries pass the rule (the ceiling)?
2. Do other published circuit-discovery algorithms succeed where WCM failed? In 081 v4 every IOI arm failed: W Z 0.01,
   U Z 0.14-0.20, W_CF Z 0.02.

## Answer

1. **The substrate imposes no ceiling under this scorer. The task split does.**
   - **The substrate passes.** The scorer keeps every SAE error live, so the full dictionary is exactly the full model
     (Z = A = C = 1.000). A non-trivial latent set also passes: the latents active on the held-out prompts, 271k, gives
     Z 1.000.
   - **The split does not.** 081's IOI held-out items use names, places and objects from the other half of Wang et
     al.'s lists. There are 0 shared answer tokens (49 vs 50), and 58 of the 96 held-out prompt token types never
     occur in training. Any circuit selected on training prompts can therefore contain only training-active latents.
   - **The cap that follows** comes from the union of every latent active on any training prompt (242,770 latents, all
     48 sites in scope). It reaches only **Z 0.231 / A 0.229 / C 0.228, accuracy 0.68**. No training-selected circuit
     can pass on 081's held-out set with all 48 sites in scope. 081's U at 132k (Z 0.20) is already near this cap.
   - **The fix is the scope.** With the ablation scope set to the 24 attn + mlp sites (resid and resid_mid run live),
     the same training-active pool restricted to attn + mlp (164k) **passes: Z 0.850, A 0.850, C 0.850, nec 1.12, acc
     0.955**. Name identity then reaches the output through the live residual stream.
2. **The published baselines do not succeed.** None of SFC (re-implementation), attribution patching or activation
   difference passes anywhere on its curve, from 10 to ~250k latents, in either scope.
   - **Scope 48:** every point has |Z| ≤ 0.03.
   - **Attn + mlp scope:** Z ≤ 0.01, falling to −0.19 at large sizes. A circuit that passes exists in this scope (the
     pool above), so this failure is the methods', not the substrate's.
   - **The reason, a diagnostic (ours):** a contrastive attribution scores g · (a_cf − a_clean). It gives exactly 0 to
     every latent whose code is identical in the clean and the ABC prompt, for example the shared template and
     first-token features. The Z / A / C frames zero or fill exactly those latents. Adding the 11.5k such training-active
     attn + mlp latents to AtP's top-k makes AtP pass at **k = 100k (111k latents): Z 0.831 / A 0.805 / C 0.806, nec
     1.11**. On all 5,000 items: 0.820 / 0.802 / 0.802. Those 11.5k latents alone give Z 0.02.
   - **Not a name-split effect.** On exp 133's same-names held-out items the baselines still fail (AtP ≤ 0.022, SFC
     ≤ 0.024). On the same items the training-active pool passes at scope 48 (Z 1.002), and so does 081's U at 75k /
     132k (Z 0.985 / 0.993).
   - **Compared with WCM,** at 500 held-out items: 081's v4 W is at Z 0.01-0.03 and U at Z 0.14-0.19. Both lie above
     every baseline point at 48-site scope, and all of them sit under the 0.23 name-coverage cap.

## How the scorer treats non-members and errors (from the code)

- **Errors.** `score_task` ablates through `CircuitOnlyPatcher` with `keep_error_sites=None`, so every in-scope site
  keeps its SAE error live in every frame (Z, A, C and nec).
  - The error is recomputed from the patched stream at that site, `x − decode(encode(x))`.
  - No frame of this scorer ablates errors, so exp 129's "zero the resid errors" collapse does not apply to 081 scores.
  - Keeping every latent at a site is the identity there.
- **Non-members.** Z sets them to 0. A and C use the sparsity-preserving fill: members keep their values, and the top
  (k − #members active) non-members by mean are set to their A or C mean.
- **Sites outside the scope** (the `sites` argument; 081 passes all 48) run live and are not re-encoded.
- **Why every site kind matters at scope 48:** a resid site is re-encoded and replaced. A circuit without a latent at a
  resid site therefore deletes everything that latent carried, at every downstream use.

## Stage 1: ceiling (all 5,000 held-out items; `out/ceiling.md`, `out/ceiling.jsonl`)

Full-model metric 3.669; alpha = 1 throughout.

| variant | members | scope | n latents | e0 | Z | A | C | nec | acc | pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| full_all | every latent, 48 sites | 48 | 1,572,864 | −0.044 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | yes (trivial) |
| scope_attnmlp | every latent at attn + mlp | attn + mlp (resid, resid_mid LIVE) | 786,432 | −0.144 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | yes (trivial) |
| scope_resid | every latent at resid | resid (other 36 LIVE) | 393,216 | −0.007 | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | yes (trivial) |
| full_attnmlp_in48 | every latent at attn + mlp, none at resid / resid_mid | 48 | 786,432 | −0.044 | 0.010 | 0.007 | 0.003 | 1.027 | 0.500 | no |
| full_resid2_in48 | every latent at resid + resid_mid, none at attn / mlp | 48 | 786,432 | −0.044 | −0.027 | −0.030 | −0.032 | 0.991 | 0.471 | no |
| full_resid_in48 | every latent at resid only | 48 | 393,216 | −0.044 | −0.006 | −0.004 | 0.006 | 0.990 | 0.485 | no |
| **active_all** | every latent active on a TRAINING prompt (081's RND pool) | 48 | 242,770 | −0.044 | **0.231** | **0.229** | **0.228** | 1.025 | 0.677 | no |
| **active_scope_attnmlp** | the training-active pool's attn + mlp latents | attn + mlp (resid LIVE) | 164,355 | −0.144 | **0.850** | **0.850** | **0.850** | 1.118 | 0.955 | **yes** |
| active_scope_resid | the training-active pool's resid latents | resid (others LIVE) | 39,688 | −0.007 | 0.257 | 0.256 | 0.260 | 1.130 | 0.687 | no |
| active_scope_resid2 | the training-active pool's resid + resid_mid latents | resid + resid_mid (attn, mlp LIVE) | 78,415 | −0.010 | 0.099 | 0.099 | 0.101 | 1.052 | 0.581 | no |
| oracle_heldout | ORACLE: latents active on the held-out prompts themselves | 48 | 271,339 | −0.044 | 1.000 | 1.006 | 1.009 | 1.005 | 0.997 | yes (oracle) |
| oracle_union | ORACLE: training-active ∪ held-out-active | 48 | 326,053 | −0.044 | 1.000 | 1.000 | 1.000 | 1.006 | 0.997 | yes (oracle) |

**Reading the table.**
- **The substrate is fine.** The oracle row shows a latent set of 271k on these dictionaries that reproduces IOI
  exactly at scope 48.
- **Selection on training cannot reach it.** Training-selected sets lack the held-out names' latents, and at any
  resid-scope site those latents get zeroed. With resid live (attn + mlp scope), names flow through the residual
  stream untouched, and the training pool passes.
- **Scorer frames.** The scorer has no error-ablated frame, so the "errors live" frame is the only one; all rows above
  are in it. Under "full dictionary restricted to site subsets", dropping a site from the scope means it runs live
  (scope_* rows). Keeping no members there instead means it is replaced by error + fill (the *_in48 rows).

## Stage 2: baselines on IOI, same substrate, same scorer

### What was run

- **Attributions** (`attrib.py`, training items only): clean IOI prompt vs its own ABC counterfactual, metric
  m = logit(IO) − logit(S) at the anchor (081's IOI task metric). From one pass:
  - **ig = SFC's node effect** (Marks et al. 2025; re-implementation of the official `_pe_ig`, `feature-circuits@7fbd82b`,
    per `107/specs/sfc.md` §3.3):
    - per site, 10-step IG with only that site moved on the straight line clean → cf (α = k/10, k = 0..9, as the
      code); other sites run the unmodified model;
    - IE = ḡ_f · Δf per latent, and Σ ḡ_ε · Δε for the error node;
    - moving x_s = decode(f) + ε jointly is SFC's joint (f, ε) interpolation, because decode is affine;
    - completeness check, first batch: the IG sum is within 1-6% of the exact single-site patch effect at 4 sites
      (`out/logs/stage2a.log`).
  - **atp:** attribution patching, the k = 0 gradient × Δ (SFC's `method='attrib'`).
  - **adiff:** |f_cf − f_clean|.
  - **Aggregation:** SFC's rule for non-templatic data (`--aggregation sum`). IOI mixes templates and lengths 14-20
    tokens, so the effects are summed over positions, then averaged over items. Positions are collapsed natively, so
    the native count = (site, latent) pairs + error nodes (rule 6: both columns are in `out/curves.md`).
  - **Items:** 100 training items (SFC's `--num_examples`) and all 1,600.
  - **Precision:** float32. 583 s for 1,600 items.
- **Circuits** (`sweep.py`):
  - **SFC:** {(site, latent) : |mean IE| > T_N}. T_N runs over SFC's own `t.logspace(-4, 0, 15)`, plus 1e-6 .. 5e-5
    (ours: the native grid stops at sizes below the 081 circuits).
  - **AtP / act-diff:** top-k on a 1-2-5 grid from 10 up to every non-zero.
  - Alpha = 1 throughout: these methods have no coefficients.
  - Two scopes: **48** (081's) and **@am**. @am is our adaptation: candidates restricted to attn + mlp, scope = attn +
    mlp, resid / resid_mid live.
- **Scoring:** 081's scorer, on the first 500 held-out items (081's seeded `HELDOUT_N` subset; 081 used 5,000).
  - 081's own v4 circuits were rescored on the same 500 items (`c081`). Their numbers match 081's 5,000-item values
    within 0.03 (W 1e-4: Z 0.010 vs 0.011; U 1e-6: 0.185 vs 0.198).
  - The selected points are rescored on all 5,000 items (stage 4, below).
- **Labels.**
  - **SFC = "our re-implementation"** of the published algorithm, run off its native substrate (Pythia-70m + Marks
    SAEs). It is on GPT-2 + OAI Top-K SAEs here, with 081's IOI data and metric.
  - **SFC error nodes are not ablatable here.** The 081 scorer keeps every error live. SFC's own evaluation ablates
    non-circuit errors, so this choice favours SFC on Z, if anything.
  - SFC's own evaluation (position-specific dense mean ablation over patch prompts) was not run. Rule 5's "paper's own
    evaluation" column is therefore missing for this arena.
  - sfc1600, atpabs (top-k by mean |IE per item|), the @am scope and +inv are ours.

### Curves (500 held-out items; full tables per point in `out/curves.md`)

Z / A / C range over each curve; "best" = the point with the largest min(Z, A, C).

| method | label | sizes | Z range | best point (n: Z / A / C, nec, acc) | passes |
|---|---|---|---|---|---|
| 081 W (WCM v4) | ours, 081 | 21k-75k | 0.010-0.026 | 75k: 0.026 / 0.029 / 0.014, 1.03, 0.52 | no |
| 081 U (unweighted) | ours, 081 | 75k-132k | 0.137-0.185 | 132k: 0.185 / 0.183 / 0.179, 1.03, 0.68 | no |
| SFC (IG, 100 items), scope 48 | re-implementation | 25-149k | −0.046-0.020 | 590: 0.011 / 0.001 / 0.001, 1.40, 0.50 | no |
| SFC (IG, 1,600 items), scope 48 | ours (more data) | 26-213k | −0.061-0.013 | 1,929: 0.004 / 0.018 / 0.007, 1.31, 0.49 | no |
| SFC (100 items), attn + mlp scope | ours (scope) | 3-96k | −0.194-0.011 | 7: 0.007 / 0.004 / 0.001, 0.57, 0.48 | no |
| AtP top-k, scope 48 | Syed et al. / SFC `attrib` | 10-247k | −0.061-0.012 | 2k: 0.004 / 0.019 / 0.006, 1.17, 0.50 | no |
| AtP top-k, attn + mlp scope | ours (scope) | 10-171k | −0.164-0.001 | 20: 0.001 / 0.002 / 0.000, 0.58, 0.48 | no |
| AtP |IE| top-k, scope 48 | ours | 10-10k (partial) | −0.014-0.009 | 50: 0.008 / −0.002 / −0.001 | no |
| act-diff top-k, scope 48 | reference | 10-265k | −0.092-0.026 | 10k: 0.010 / 0.010 / 0.004, 1.07, 0.49 | no |
| **diag: AtP@am + invariant latents** | ours, diagnostic | 11.5k-182k | −0.137-0.875 | **182k: 0.875 / 0.876 / 0.876, 1.10, 0.95; smallest pass 111k: 0.831 / 0.805 / 0.806, 1.11, 0.93** | **yes, ≥ 111k** |
| diag: AtP + invariant latents, scope 48 | ours, diagnostic | 19.7k-267k | −0.043-0.234 | 267k: 0.234 / 0.230 / 0.229 (= the name cap) | no |

"Invariant latents" = the training-active latents whose attribution is exactly 0 on every training item, i.e. whose code
is identical in clean and counterfactual. That is 11,483 latents at attn + mlp and 19,726 over all 48 sites, from the
1,600-item AtP. Alone they give Z 0.02.

### Same-names view (exp 133's `held_sn` items; names from the training halves, no training item)

This view asks whether the baselines' failure is only the name split. The items are exp 133's 998 fresh items, drawn
from the training halves of the lists (`133/tasks/ioi_sn.json`, read only). Training rows, the C pool and the A / C
means are 081's. Ceiling rows use all 998 items (`out_sn/ceiling.md`); curve points use the first 500
(`out/curves_sn500.jsonl`).

| circuit | scope | n | Z | A | C | nec | acc | pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| training-active pool (ceiling) | 48 | 242,770 | 1.002 | 1.002 | 1.002 | 0.996 | 0.992 | **yes** |
| training-active pool, attn + mlp (ceiling) | attn + mlp | 164,355 | 1.002 | 1.002 | 1.002 | 1.019 | 0.991 | **yes** |
| 081 U λ 1e-6 | 48 | 131,984 | 0.993 | 0.989 | 0.990 | 1.029 | 0.990 | **yes** |
| 081 U λ 1e-5 | 48 | 75,301 | 0.985 | 0.969 | 0.959 | 1.028 | 0.994 | **yes** |
| 081 W λ 1e-4 / 3e-5 | 48 | 21k / 75k | 0.014 / 0.014 | | | | 0.53 / 0.50 | no (v4 W's own collapse; exp 133: lr) |
| AtP top-k, k = 10 .. 247k | 48 | | −0.284 .. 0.022 | | | | ≤ 0.51 | no |
| AtP top-k @am, k = 10 .. 171k | attn + mlp | | −0.425 .. 0.000 | | | | ≤ 0.48 | no |
| SFC (100 items), T_N = 1 .. 7.2e-4 (25 .. 24k; budget cut the rest) | 48 | | −0.051 .. 0.024 | | | | ≤ 0.52 | no |
| diag: AtP@am + invariant latents, k = 100k | attn + mlp | 111,483 | 1.045 | 1.009 | 1.009 | 1.039 | 0.998 | **yes** (also k 171k: 1.002) |

So once names are shared, the training pool passes even at scope 48, and so do 081's U circuits. The attribution
baselines still fail at every size. Their failure is NOT the name split: they lack the invariant latents. Adding those
latents makes AtP pass.

### Full 5,000-item rescoring of selected points

All 5,000 of 081's held-out items (`out/curves_hall.jsonl`); the 500-item values in brackets.

| circuit | n | Z | A | C | nec | acc | pass |
|---|---:|---:|---:|---:|---:|---:|---|
| diag: AtP@am + invariant latents, k = 100k (attn + mlp scope) | 111,483 | 0.820 (0.831) | 0.802 (0.805) | 0.802 (0.806) | 1.150 | 0.944 | **yes** |
| diag: AtP@am + invariant latents, k = 171k (attn + mlp scope) | 182,374 | 0.870 (0.875) | 0.870 (0.876) | 0.870 (0.876) | 1.120 | 0.959 | **yes** |
| SFC (100 items), T_N = 0.0373, scope 48 (its best 500-item point) | 590 | 0.006 (0.011) | 0.001 (0.001) | −0.002 (0.001) | 1.407 | 0.490 | no |
| AtP top-k, every non-zero (k = 247k), scope 48 | 247,451 | −0.063 (−0.061) | −0.002 (−0.013) | −0.005 (−0.014) | 1.014 | 0.424 | no |

The 500-item subset tracks the full set within 0.012 on these points.

Commands for stages 3 and 4:

```
bash $E/gpujob.sh stage3_sn bash $E/stage3.sh
FIRST=2 LAST=2 HELDOUT_N=all BUDGET_MIN=30 METHODS=atp@am+inv,sfc100,atp \
  ONLY_KNOBS=atp@am+inv:k100000,atp@am+inv:k170891,sfc100:T0.0373,atp:k247451 bash $E/stage2b.sh
```

(sweep_hall_1 failed at start on a HELDOUT_N=all bug, since fixed in stage2b.sh; sweep_hall_2 is the run.)

## Verdict

- **Can any circuit pass on this substrate?** Yes; the substrate does not impose a ceiling under the 081 scorer. The
  errors stay live, the full dictionary is exact, and an oracle latent set of 271k passes at scope 48.
- **Can a circuit selected on training items pass on 081's held-out set?** Not at scope 48: the cap is Z ≈ 0.23,
  because held-out names are disjoint from training names. Yes in attn + mlp scope: the training-active pool passes at
  Z 0.85.
- **Do the published baselines succeed where WCM failed?** No. SFC (re-implementation), AtP and activation difference
  all sit at |Z| ≤ 0.03 at scope 48, below 081's U (0.19). They fail in attn + mlp scope too, and on same-name items,
  where 081's U passes. They fail for a structural
  reason: contrastive attribution gives exactly zero to the latents shared by the clean and ABC prompts, and a Z / A / C
  ablation needs those latents.

## Caveats

- **Curve points use 500 held-out items, not 5,000.** The rescored 081 circuits differ from their 5,000-item values
  by ≤ 0.03.
- **Site-space edits.** IG / AtP move the site activation in the SAE's input space, which is layer-normalised per
  token for GPT-2. The adapter applies an edit as x + Δx_s · scale_clean, the same convention the scorer uses. A
  model-space patch would differ slightly at resid sites.
- **SFC is run with node scores only,** without its edges. SFC evaluates node sets, so this is its own evaluation
  object (spec §1).
- **"Invariant latents" and the oracle rows are diagnostics, not methods.** The oracle reads the held-out prompts.
- **Not run:** SFC's own evaluator, other baselines (CircuitLasso, Ge, Caples), and any IOI per-position (templatic)
  aggregation.

## Files and commands

All commands run from the repo root in WSL. GPU jobs go through `gpujob.sh`, which takes the shared `.gpu-lock` and
runs a watchdog (kill above 15.3 GB GPU or below 3 GB RAM).

```
E=experiments/132-ioi-ceiling-baselines
ENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
# stage 1 ceiling (5,000 held-out items)
bash $E/gpujob.sh ceil_A env $ENV VARIANTS=full_all,scope_attnmlp,full_attnmlp_in48,full_resid2_in48 ./.venv/bin/python -X utf8 -u $E/ceiling.py
bash $E/gpujob.sh ceil_B env $ENV VARIANTS=active_all,active_scope_attnmlp,full_resid_in48,scope_resid ./.venv/bin/python -X utf8 -u $E/ceiling.py
# stage 2a: attributions on the 1,600 training items + extra ceiling rows (active_scope_resid[2], oracle_*)
bash $E/gpujob.sh stage2a bash $E/stage2a.sh
# stage 2b: curves on 500 held-out items (resumable; one job per call)
FIRST=1 LAST=1 HELDOUT_N=500 bash $E/stage2b.sh
FIRST=2 LAST=2 HELDOUT_N=500 METHODS=atp@am+inv,atp+inv,adiff,sfc1600,atpabs,sfc100@am+inv bash $E/stage2b.sh
# stage 3: same-names view
bash $E/gpujob.sh stage3_sn bash $E/stage3.sh
# tables
./.venv/bin/python -I -B $E/report.py
```

| file | content |
|---|---|
| `common132.py` | engine / task / means setup (081's), `score` = engine.score with an explicit scope, `HELD_SET=sn` |
| `ceiling.py` | stage 1 variants → `out/ceiling.{jsonl,md}` |
| `attrib.py` | IG / AtP / act-diff on the training items → `out/attr/scores_n{100,1600}.pt` |
| `sweep.py` | circuits from the scores, scored → `out/curves_<tag>.jsonl` |
| `report.py` | `out/curves.md` |
| `gpujob.sh`, `stage2a.sh`, `stage2b.sh`, `stage3.sh` | GPU job wrappers |
| `out/logs/`, `out/gpujob.log`, `out/watch.csv` | logs |
