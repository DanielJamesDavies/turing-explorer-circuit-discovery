# 135: would growing IOI circuits from members' upstream parents help? (cheap diagnostic)

Agent ioiparent, night of 2026-11-19/20. **Not for the paper** (IOI was dropped from it). The result decides whether
"grow circuits from members' upstream parents" becomes a Linear backlog issue.

## Question (Daniel, 2026-11-19)

If a latent is selected in the residual stream, something upstream must have caused it. Would growing circuits from
members' upstream parents help IOI? Before building that, measure two things on the IOI circuits that already exist:

1. **Closure.** Does each residual-stream member have a strong in-circuit parent? What share of its natural
   pre-activation comes from in-circuit parents, out-of-circuit latents, SAE error, the embedding and the skip path?
2. **Candidates.** At the IOI positions (S1, IO, S2, END), which high-attribution latents were never in the fit's
   candidate set, and which were offered and dropped?

Decision rule:
- **Missing:** needed latents are missing from the candidates, or in-circuit parents explain little, so parent-growing
  is worth trying.
- **Offered and dropped:** they were offered and dropped, and closure is already decent. Then the 0.23 ceiling is the
  ablation setup or the objective, and closure won't fix it.

## Design (written before the GPU run)

### Circuits (all from 133, `133/out/circuits/`, read only)

The closure question needs residual-stream members, so it applies to the **48-site** fits. 133's best 48-site fits
(peak lr 0.05, paired R, the ones that pass on same names; dis Z under the 0.23 cap except the largest):

| circuit | n | resid_mid / resid members | dis Z | sn Z |
|---|---|---|---|---|
| `W_1e-3_lr05_pair` | 16,721 | 6,530 / 6,770 | 0.15 | 0.89 |
| `W_3e-4_lr05_pair` | 31,946 | 11,453 / 11,809 | 0.25 | 0.93 |
| `W_1e-4_lr05_pair` | 58,747 | 18,047 / 18,736 | 0.37 | 0.97 |

Two attn + mlp-scope circuits are read for measurement 2 and for functional retention only. They have no residual
members, because resid is live in that scope:
- `AM_W_3e-3_lr05_pair`: 1,361 latents, the sparse same-name pass;
- `AM_W_1e-6_lr05_pair`: 116,833 latents, the only disjoint-name pass.

### What "offered" means in this engine (read from the code)

- 133's fits call `run_learned_mask` with `support=None` and no `theta_init_mode`. So `learned_mask` initialises
  **every latent of every in-scope site** at `theta_init` 4 (gate sigmoid(4) = 0.98, above the keep threshold 0.5).
  - Code: `_run_learned_mask_impl`, the uniform branch, `thetas = full(d_sae, theta_init)`.
  - `config.py`: `theta_init` 4.0, `keep_threshold` 0.5.
  - 081's `engine_kwargs` passes `support` only for RND.
- So all 1,572,864 latents (48-site) or 786,432 (attn + mlp) are offered, and **nothing is "never in the candidate
  set" in the engine's sense.**
- What I count instead:
  - **dropped, train-active:** offered, active on at least one of the 1,600 training prompts (also reported at that
    role), not in the circuit;
  - **dropped, never train-active:** offered but never active on any training prompt. The fit had no clean-code signal
    for these: under Z they contribute 0 whatever the gate, and under A / C only their fill value. In practice they
    are *unavailable to any training-based selection*, parent-growing included. This is the 132 cap mechanism.

### Measurement 1: closure (exact, natural forward)

- **What is decomposed.** For every active member at a resid_mid or resid site (TopK, so act = pre-activation),
  pre_i(t) = Σ_c w̃_i · comp_c(t) / s_tgt(t) + b_eff_i, which is exact:
  - w̃_i = the centred encoder row (the SAE input is per-token LN, so centring is exact);
  - s_tgt = the target's LN scale at t.
- **Components** (GPT-2: resid_mid_L = resid_{L-1} + attn_L; resid_L = resid_mid_L + mlp_L; resid_{-1} = the
  embedding). Each parent component is split by its own SAE, in model space: comp = s_src (Σ f_j d_j + b_dec + ε) +
  mean.
- **View A (the brief's definition).** Parents = members at resid_{L-1} (the same position: the skip path), attn_L and
  mlp_L (same layer, same position).
  - Categories: in-circuit parents, out-of-circuit latents, SAE error, embedding (layer 0), bias (b_eff + b_dec), and
    the skip subtotal.
  - Also: the single strongest in-circuit parent's share.
- **View B (resid only).** resid_L = resid_mid_L + mlp_L, i.e. the nearest residual parent.
- **View C (positions explicit).** The attn_L part replaced by its exact source-position decomposition:
  - attn_L(t) = Σ_h Σ_s A_h[t, s] (LN1(x_in[s]) W_V^h + b_V^h) W_O^h + b_O, with LN1's scale frozen per source
    position;
  - x_in[s] is split by the resid_{L-1} SAE at s (in / out / err / b_dec), or is the embedding at layer 0;
  - reported by source role (io, s1, s1_next, s2, end, other).
- **Exactness check.** Each view's parts must sum to the activation; the maximum relative residual is reported.
- **Functional closure.** The scorer's Z frame: `CircuitOnlyPatcher`, members at α, non-members 0, errors live, the
  circuit's own scope. Reported: each member's re-encoded code at its natural occurrences, as retention =
  Σ circuit / Σ natural (codes before α), and the share of occurrences still active.
- **Item sets:** the first 400 training items, the first 500 dis (081's held-out, disjoint names), and the first 500 sn
  (133's same names).

### Measurement 2: attribution at the IOI positions

- **Per (site, latent, position role),** with the metric m = logit(IO) − logit(S) at the anchor:
  - z = g_f · f, the first-order effect of zeroing the latent: the Z frame, where 0.23 is measured;
  - atp = g_f · (f_ABC − f), 132's AtP.
  - g_f = W_dec g_x (132 `attrib.py`'s additive-leaf gradient), summed over the role's positions and averaged over
    items.
- **Item sets:** all 1,600 training items (these also give the train-active masks), 500 dis, 500 sn.
- **Per circuit, role (io, s1, s1_next, s2, end), N ∈ {20, 100, 500} top latents by |mean attribution|:**
  - the share in the circuit;
  - the share dropped while train-active (and train-active at that role);
  - the share never train-active;
  - kinds;
  - the share of total |attribution| mass at the role carried by members.

### Reading the decision rule (fixed before the run)

- **"Missing"** holds if either:
  - the top-attribution latents at the name positions are mostly *never train-active*, which is unreachable by
    selection; or
  - view A / C in-circuit parents explain under about half of members' pre-activation, with out-of-circuit latents
    carrying most of the rest.
- **"Offered and dropped"** holds if the top latents are mostly train-active and dropped (or kept), *and* in-circuit
  parents explain most of the pre-activation, *and* the Z frame retains the members.
- **Separate readings for dis and sn.** On dis, never-train-active latents are expected (132: 0.23 cap). The question
  for parent-growing is whether the *training-side* structure is closed.

## Results (GPU run 2026-11-20 02:27-02:35, RTX 16 GB, peak 11.6 GB; full tables in `out/report.md`)

### Verdict: "offered and dropped" (closure is not the problem); no parent-growing backlog issue for IOI

- **Closure is already decent on the data the fits see.** In-circuit parents give 83-90% of resid members'
  input-dependent pre-activation, on train and on same-name items.
- **Every high-attribution latent at S1 / IO / S2 / END on training items was offered (all are) and is in the
  circuit.**
- **On the disjoint-name held-out set the gap is latents that never fire on any training prompt.** No training-side
  growth, parent-based or otherwise, can add them. A small remainder fires in training, but rarely and with
  near-zero training attribution, and the fits dropped it.
- So the 0.23 ceiling is the split together with the 48-site zero-ablation setup (132's mechanism), not missing
  parents.

### Measurement 1: closure (view A, the brief's parents; pooled over all active member occurrences)

"Drive" is the input-dependent part of the pre-activation, i.e. act minus the SAE biases. The bias is large and
negative here, about −0.6 to −0.75 of act, which is why the raw in-circuit share exceeds 1.

| circuit (48-site) | set | kind | drive share: in-circuit / out-of-circuit / SAE err / emb | single parent ≥ 25% of act | in-circuit ≥ 50% of act |
|---|---|---|---|---|---|
| W 1e-3, 16.7k | train | resid_mid | 0.83 / 0.10 / 0.04 / 0.03 | 0.88 | 0.88 |
| W 1e-3, 16.7k | train | resid | 0.84 / 0.12 / 0.04 / 0 | 0.88 | 0.90 |
| W 1e-3, 16.7k | dis | resid | 0.78 / 0.19 / 0.03 / 0 | 0.86 | 0.88 |
| W 3e-4, 31.9k | train | resid | 0.87 / 0.08 / 0.04 / 0 | 0.91 | 0.95 |
| W 1e-4, 58.7k | train | resid | 0.90 / 0.05 / 0.05 / 0 | 0.93 | 0.97 |
| W 1e-4, 58.7k | dis | resid | 0.85 / 0.11 / 0.04 / 0 | 0.91 | 0.95 |

- **sn ≈ train in every row.**
- **The skip path dominates.** Most in-circuit drive comes from the same-position resid_{L-1} member: 1.06-1.37 of act.
  Attention parents give 0.10-0.18 and MLP parents 0.16-0.22.
- **View B** (resid_mid_L + mlp_L) gives the same picture: in-circuit 1.18-1.55 of act.
- **View C** (attention resolved by source position): the attention part is small, about 0.25-0.30 of act at resid_mid
  and 0.13-0.18 at resid. Most of it comes from "other" positions; the S1 / IO / S2 source positions contribute ≤ 0.03 of
  act each.
- **Where the dis shortfall sits.** At the name positions on dis, the out-of-circuit share of act rises to 0.29-0.49,
  against 0.01-0.13 on train and sn. Members that fire on unseen names are driven partly by latents outside the
  circuit, which is where the dis shortfall sits.
- **Exactness:** the decomposition is exact to ≤ 4.5e-3 relative (float32).
- **Functional closure (the scorer's Z frame).**
  - Members keep only 0.53-0.64 of their natural code: pooled, with α applied 0.50-0.64. The share of member
    occurrences still active is 0.48-0.58.
  - The task metric is still largely kept on train and sn: D 3.26-3.58 against 3.59-3.68 for the full model.
  - On dis, retention is barely lower (0.48-0.56) while D collapses (0.53-1.37 against 3.69).
  - So retention is not what separates passing from failing.

### Measurement 2: high-attribution latents at the IOI positions (top-100 by |mean attribution|)

| circuit | set | role | z (zero-ablation): in / dropped & train-active / never train-active | atp (vs ABC): in / dropped & train-active / never train-active | atp mass in circuit (top-20 row) |
|---|---|---|---|---|---|
| W 1e-3 | train | io / s1 / s2 / end | 0.99-1.00 / ≤ 0.01 / 0 | 0.99-1.00 / ≤ 0.01 / 0 | 0.75-0.87 |
| W 1e-3 | dis | io | 0.94 / 0.02 / 0.04 | **0.28 / 0.37 / 0.35** (top-20: 0.25 / 0.25 / 0.50) | 0.32 |
| W 1e-3 | dis | s1 | 0.99 / 0.01 / 0 | 0.58 / 0.23 / 0.19 | 0.40 |
| W 1e-3 | dis | s2 / end | 1.00 / 0 / 0 | 0.95-1.00 / ≤ 0.05 / 0 | 0.68-0.70 |
| W 1e-4 | train | all four | 1.00 / 0 / 0 | 1.00 / 0 / 0 | 0.92-0.97 |
| W 1e-4 | dis | io | 0.96 / 0 / 0.04 | **0.47 / 0.18 / 0.35** (top-20: 0.40 / 0.10 / 0.50) | 0.46 |
| W 1e-4 | dis | s1 | 1.00 / 0 / 0 | 0.69 / 0.13 / 0.18 | 0.53 |

- **What "dropped, train-active" means on dis-IO.** These are the top-100 AtP latents that were offered, are
  train-active at IO, and are not in the circuit: 30 for W 1e-3, 12 for W 1e-4.
  - They fire at IO on a median of only 11 / 3 of the 1,600 training items.
  - Their median rank by training IO attribution is ~11k / ~17k.
  - The fits had essentially no training signal to keep them, and a parent-growing rule run on training prompts would
    not see them either.
- **The attn + mlp-scope circuits** (resid live) show the same pattern:
  - AM_W_1e-6 (117k) holds ≥ 0.97 of every top-100 on every set;
  - the sparse AM_W_3e-3 (1.4k) holds 0.55-0.75 of top-100 z, and what it drops is train-active (never-active
    ≤ 0.03).
- **Candidate set, from the code:** every latent of every in-scope site starts at gate 0.98, so "never in the
  candidate set" is empty by construction. The meaningful split is train-active versus never train-active (see
  Design).

### Caveats

- **The attribution is first order** (gradient × activation), on 500 held-out and 1,600 training items. z ranks
  residual latents highly almost by construction: zeroing a resid latent removes everything it carries.
  - The contrastive atp column is the one that isolates name content, and it is where dis differs.
- **Closure shares are signed sums.** Out-of-circuit latents can cancel inside their category. The "single parent"
  and "≥ 50%" occurrence columns are the robust reads.
- **Views A / B / C freeze the LN scales** (the SAE input LN at each site, and LN1 for view C). Given those, they are
  exact; they say nothing about what happens off the natural run. The functional Z-frame read covers that.
- **The parent definition is limited.** It includes only same-position parents for resid / attn / mlp; view C is the
  only cross-position read, and it is coarse (by category, not per latent).
- **One fit per configuration (133 has no replicates).** The 500-item reads have small standard errors, but fit
  noise is unmeasured.
- **What was assumed:**
  - that 133's 48-site lr-0.05 paired-R fits are "the best existing IOI WCM circuits" for a closure question. The
    attn + mlp fits are 133's best on its scorer, but have no residual members.
  - "never train-active" is my operationalisation of "never in the candidate set". The engine's own candidate set is
    everything.

### Files

- `closure135.py`: the GPU stages `attr` and `closure` → `out/attr_<set>.pt`, `out/closure_<circuit>_<set>.pt`.
- `analyse135.py`: CPU tables → `out/report.md`.
- `gpujob135.sh`: lock-free watchdog wrapper (the lock is taken / released by the agent with its file tools).
- `out/`: `report.md` (all tables), `run.log`, `gpujob.log`, `watch.csv`, the `.pt` tensors (680 MB).
- The 4-item CPU smoke test (`out_cputest/`) passed and was deleted afterwards (563 MB, in .agent-trash for 8 h).

Reused unchanged: 081 `engine.Engine` / `batch` / `task_common`, src `PublicSAEBank`, `layer_norm_in`,
`CircuitOnlyPatcher`, `multi_patch`, and 132's additive-leaf gradient method (re-typed, not imported). Written here:
the decomposition, the retention capture, and the role aggregation.
