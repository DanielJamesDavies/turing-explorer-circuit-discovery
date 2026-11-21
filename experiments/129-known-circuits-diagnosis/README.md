# 129 — Why 117's known-circuit check did not recover the published heads (diagnosis, CPU only)

**Question.** 117 / 120 fitted WCM v4 circuits (062 production driver, λ 1e-3, 400 steps) for 15 GPT-2 small OAI v5
latents on the published IOI and greater-than pathways. The circuits kept published *positions* but not published
*heads* (1 / 66 head-class checks), and 120 concluded the published mechanism "runs largely through the SAE error
terms". Is that a WCM limitation or an artefact of the test? Everything here re-reads 117 / 120 outputs (read-only) or
runs CPU forward / backward passes of GPT-2 small + the OAI SAEs on 117's held-out task prompts. No GPU was used.

**Answer, in order of weight.**

1. **The circuits do not compute the task variable, so there is no head structure to find in them** (WCM limitation:
   objective + pass rule). On minimal pairs the full model separates conditions; every circuit gives the same value
   for both (`out/e_func.md`, 64 held-out prompts):
   - IOI name-mover latents (target name n as IO vs the same prompt with n as the subject S): full model 18.06 → 1.89,
     18.12 → 0.36, 17.66 → 1.19 (drops of 90-98%). Corpus circuits 13.86 → 13.74, 14.08 → 13.81, 7.51 → 7.27; task
     circuits 18.38 → 18.02, 19.23 → 18.80, 18.85 → 18.41 (drops of 1-3%). The circuits are "name n is in the
     prompt" detectors (all go negative when n is absent: −1.5 to −5.0, except the failing 9.attn.27039 corpus circuit,
     5.77). They have no use for S-inhibition, so they do not contain it.
   - IOI-SI and IOI-IND (IOI prompt vs its ABC counterfactual): full model 2.74 vs 0.78 (7.attn.29964), 2.50 vs 1.46
     (8.attn.18439), 5.72 vs −0.33 (5.attn.25375), 5.55 vs 0.29 (5.attn.28128). All 12 circuits of the 6 SI / IND
     targets: IOI and ABC values within 0.4 of each other (e.g. 6.80 vs 6.84, 11.11 vs 11.15).
   - Greater-than MLP latents: |Spearman ρ(pre, YY)| full model 0.59-0.76; corpus circuits −0.41 to 0.37 (median
     |ρ| 0.12), task circuits −0.14 to 0.38. Only 10.mlp.4386's corpus circuit keeps the sign (−0.41 vs −0.64), and it
     fails faithfulness.
   - Why the pass rule allows this: Z / A / C (`src/eval/wcm_eval.py`: free0 / freeM_topk / freeN_topk) all score the
     circuit's reproduction of the activation on contexts **where the target fires**; the contrast set only supplies
     the mean values non-members are set to for C. Nothing in the scored criteria requires the circuit to be silent
     when the target is silent, or to track the variable that switches the target. A circuit that reproduces the
     *level* from token-identity and template features passes; the routing heads (which decide *when* to fire) are
     not needed for that. (Whether the training loss has a term on silent rows I did not check; speculation that it
     does not bind here.)
   - Why the training data allow it (test artefact, partly): corpus windows and IO-substituted task prompts never
     contain the name-as-subject case. The fit never sees a context where "name present" and "target fires" come
     apart, so the minimal circuit omits the S-inhibition signal.

2. **The error-term explanation of 117 / 120 is wrong for the heads that matter** (test artefact / misreading).
   - Full model, exact per-head first-order attribution (head output oc_h through every route; `c_heads.py`,
     `out/d_report.md`): for the 3 name-mover targets the S-inhibition heads at END contribute 2.99, 3.08, 2.21 to
     the target's pre-activation per prompt, of which 2.83, 2.88, 1.87 (85-95%) goes through attention-out
     **dictionary slots**, not the SAE error. Induction heads: about half through the dictionary.
   - 120's collapse comes from the **residual-stream** SAE errors, not the attention / MLP errors. Forward-only, full
     model, 15 targets, target pre-activation as a ratio to the unedited model (median, range): errors zeroed at attn
     sites 0.98 (0.87-1.28), at mlp sites 0.97 (0.44-1.39), attn + mlp 0.94 (0.30-1.49; only 9.mlp.3600 below 0.46),
     resid_mid + resid −0.23 (−0.78 to 0.89), all (120's regime) −0.26 (−0.90 to 0.27). For the NM targets: attn + mlp
     0.91-0.94, resid −0.16 to −0.23. Replacing the whole residual stream by its
     reconstruction at every layer is what removes the computation; the published heads' outputs are largely in the
     attention dictionary.
   - In the circuits' own Z frames the S-inhibition heads' total effect (members + live error) is |≤ 0.19| per prompt
     (≤ 8% of the full model's 2.2-3.1) — they are functionally absent, consistent with point 1, not hidden in the error.

3. **The pre-stated head rule could not be passed by the full model itself** (test artefact).
   - Scored as if it were a circuit, against each circuit's own 20 nulls, the full model passes **2 / 66** head-class
     checks (`out/a_mass.md`). The binding clause is "share ≥ 2× uniform base": the full model meets it in 2 / 66
     (median full/base: S-inhibition 1.88, induction 1.43, Hanna heads 1.80, duplicate-token 0.71, previous-token
     0.30). Without that clause it passes 54 / 66 and the circuits 24 / 66.
   - So "1 / 66" is not interpretable as a recovery rate; the full model's "published head ranked first 15 / 15" is a
     ranking statement the rule never tested. The real gap is the rank: best published head at its role ranks median
     2nd in the full model vs 168th in the circuits (117 attribution). With exact head attribution (NM targets),
     S-inhibition / induction heads rank 5-6 / 9-12 in the full model, and 3-92 / 234-416 in the circuits.
   - The rule also tests the wrong upstream classes in places: for the L5 induction-output targets the expected
     classes are duplicate-token heads at S2 (not an input to induction heads in Wang et al.) and previous-token heads
     at S1+1, which act through keys / attention pattern and get little first-order mass even in the full model
     (previous-token 0.30× base). IOI-IND "NOT RECOVERED" is fixed by the rule, before any circuit is looked at.

4. **Target choice** (test artefact, 2 of 9 IOI targets).
   - 8.attn.16224 (IOI-SI) fires *more* on the ABC counterfactual than on IOI in the full model (7.78 vs 5.33), and
     5.attn.27767 (IOI-IND) likewise (5.85 vs 4.91). They are not S-inhibition / induction outputs in the IOI sense
     (117 already had to fill their contrast with corpus windows because the counterfactuals never went silent).
   - The NM targets are clean in the full model (point 1).

5. **Residual-site members** (79% of members; `out/b_members.md`). Correction to the brief's "shortcut" framing: in
   WCM's circuit frame every non-member at every site is ablated and resid members are re-encoded from the edited
   stream, so a resid member cannot deliver information whose writers were ablated. What the 79% does show
   (`f_embed.py`, `out/f_embed.md`, exact writer split of each resid / resid_mid member's natural pre-activation):
   - at the IO position of the 3 NM targets (the 0.89-0.95 IO attribution share), the resid members' value comes
     from the embedding block: embed 0.02-0.06 + attn L0 0.19-0.26 + MLP L0 0.43-0.51 (= 0.73-0.83), MLP L1+
     0.18-0.28, attention L1+ −0.02 to 0.03. The name identity at IO needs no head above layer 0;
   - so the circuits describe the target as "token n (from the embedding / L0 block) read at IO" — the value side of
     the name movers — and nothing on the query side. That matches point 1 (b): the training contexts do not require
     the "which name not to copy" signal.
   - For greater-than, YY-position members get 0.63-2.50 (signed share) from attention L1+, i.e. the circuits do use
     mid-layer attention writes there, but not in a YY-tracking way (ρ ≈ 0, point 1).

6. **Faithfulness and λ** (not decisive). Head recovery does not track passing: 117's task circuits pass 42 / 60 task
   folds yet none reproduces a minimal-pair contrast. The S-inhibition effect in the full model (2-3 units of an ~18
   pre-activation, ~15%) is not small, so λ 1e-3 pruning "small contributions" is not the main reason; a λ sweep was
   not run (GPU).

**Reading.** The 117 result should not be read as "WCM cannot find known circuits because the mechanism is in the
SAE error". It should be read as: *a WCM circuit, as fitted and scored, reproduces the target's level on firing
contexts and is free to drop whatever decides when the target fires.* On these GPT-2 latents that drops the published
routing heads (S-inhibition, induction) and keeps the content they move (the IO name from the embedding block, late
MLPs). This is a real limitation of the objective and pass rule (no requirement on silent / counterfactual contexts),
compounded by test artefacts (unpassable head rule, two non-IOI targets, training contexts without the
discriminating case). The error terms matter only in the residual stream, which is why 120's error-free refit could
not be a known-answer test.

**What the paper can claim.** Circuits identify which input features and positions a latent reads (IO name for
name-mover outputs; late MLPs at END), and they are faithful to the activation level on the contexts they were fitted
on; they are not guaranteed to reproduce the latent's selectivity across minimal pairs, and on IOI / greater-than they
do not (0 / 15 targets), so they do not recover the routing heads. The "runs through the SAE errors" sentence proposed
in 120 should not go in: zeroing the attention / MLP errors leaves 91-94% of the NM targets' activation.

## Next tests (none run; GPU ones described only)

1. **Minimal-pair faithfulness in the pass rule** (CPU, ~1 h with this repo's code): add, per task, the swap / ABC
   contrast as a *scored* condition (circuit must reproduce full(a) − full(b)). Re-score the 30 existing circuits
   (`e_func.py` already computes the numbers; the answer is 0 / 15 targets).
2. **Refit with minimal pairs in the training rows** (GPU, ~2-3 h on the local 5070 Ti for 15 targets × 1 record):
   task records whose training rows include name-as-S (NM) and ABC (SI / IND) prompts, with a loss term on the
   counterfactual rows' activation (silent or the full model's value). Then rerun `structure117.py` and `c_heads.py`
   frames. Prediction if point 1 is right: S-inhibition / induction head effect in the circuit frame rises toward the
   full model's 2-3 units per prompt.
3. **λ sweep** on the 3 NM targets (GPU, λ ∈ {1e-4, 3e-4, 1e-3, 3e-3}, ~1 h): does the S-inhibition head effect
   appear at low λ even without minimal pairs.
4. **Planted circuits (Tracr / InterpBench)** would answer a different question: whether WCM recovers a known
   computation when it is guaranteed to be in the dictionary. It is fair only if (i) the sites are head-level
   (per-head z or per-head output SAEs / identity dictionaries) or attention-only, with no residual sites in scope, so
   a head cannot be described away as a resid re-encoding; (ii) the training and scored contexts include the
   counterfactuals that the planted algorithm distinguishes (otherwise point 1 recurs by construction); (iii) the
   planted model has no SAE error (Tracr's compiled residual basis is exact). InterpBench's SIIT models are the better
   choice (trained, realistic superposition-free heads, published ground-truth node sets). Cost: CPU-scale models;
   the engine needs a head-site adapter (new code, ~1-2 days).
5. **Resid-error-only regime** (GPU, as 120 but with errors zeroed only at attn + mlp sites): the full model keeps
   0.91-0.94 of the NM targets' activation there, so this is a usable known-answer regime on GPT-2, unlike 120's.

## Files

| path | what |
|---|---|
| `a_mass.py` → `out/a_mass.md/json` | re-read of 117 / 120 structure rows: full model scored by the pre-stated rule; looser readings; mass by site kind |
| `a_roles.py` → `out/a_roles.md` | attribution share by token role per frame |
| `b_members.py` → `out/b_members.md/json` | circuit members by site kind / layer (117 and 120) |
| `c_heads.py` → `out/c_heads.jsonl` | exact per-head (total / dictionary / error route) and per-site error attribution, full model + 117 circuit Z frames; forward-only partial error zeroing. 64 held-out prompts |
| `d_report.py` → `out/d_report.md/json` | tables from `c_heads.jsonl` |
| `e_func.py` → `out/e_func.md/json` | minimal-pair / YY controls, full model vs circuits |
| `f_embed.py` → `out/f_embed.md/json` | writer split of resid members' natural pre-activation (check max diff 2.5e-5) |
| `peek_tasks.py` | prints the task-file layout |
| `out/c_heads_smoke16.jsonl` | 16-prompt smoke of `c_heads.py` (one target) |

Run (WSL, repo root): `CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 ./.venv/bin/python -I experiments/129-known-circuits-diagnosis/<script>.py`.
`c_heads.py` takes ~5 min per target on 16 CPU threads alone (much longer when sharing the CPU); it resumes from
`out/c_heads.jsonl`.

## Caveats

- First-order attribution (gradient × head output at gain 1); the softmax path of query-side heads is linearised.
- `c_heads.py` finished all 15 targets x 3 frames (45 rows, 18:41). Across all 33 head-class cells per frame, the
  published heads' signed effect in the circuits is a median 0.00x the full model's (corpus and task circuits), and
  their best rank by exact |effect| is median 13 (full) vs 195 (corpus) / 157 (task) (`out/d_report.md`).
- The exact head share still puts the full model at only ~1.1x base for S-inhibition (rank 5-6): even exact
  attribution makes share-of-mass a weak statistic; ranks and signed effects are the better readout.
- e_func's name-as-S prompts put n in both S slots of held-out prompts (IO unchanged); 64 prompts per condition.
