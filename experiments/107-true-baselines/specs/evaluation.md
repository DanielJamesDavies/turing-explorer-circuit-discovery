# 107 spec: how circuit-discovery methods are compared (evaluation protocol)

Written by `spec-eval`. This is the comparison and evaluation protocol for experiment 107. The per-method
algorithms (SFC, Ge et al., CircuitLasso) are in their own spec files. Here: metric definitions, ablations, curves, area
statistics, train/held-out practice, efficiency reporting, a recommended protocol for arenas A/B/C, and pitfalls.

**Tags.** `[paper §x]` is a claim read from the paper; `[code file:line @SHA]` is read from the code at that commit;
`[inferred]` is my reading where the source is silent; `[ours]` is a recommendation of ours, not established practice.
Line numbers for code were read from the raw file at the stated SHA.

**Sources** (all read through WebFetch):
- MIB: Mueller et al. 2025, arXiv 2504.13151; circuit-track code github.com/hannamw/MIB-circuit-track, main =
  `b759df34433c9e31043ba9e02908ce0bf20e894f` (2025-06-30), the same commit pinned as the submodule of
  github.com/aaronmueller/mib.
- Kharlapenko et al. 2025, arXiv 2504.13756 (SFC scaled to Gemma).
- Syed, Rager, Conmy 2023, arXiv 2310.10348 (attribution patching vs ACDC).
- Marks et al. 2025, arXiv 2403.19647 (SFC), and its code github.com/saprmarks/feature-circuits, main =
  `7fbd82b895ae16294f4e6fc7bfc675d1d680d659` (2025-07-14).
- Conmy et al. 2023, arXiv 2304.14997 (ACDC).
- Hanna, Pezzelle, Belinkov 2024, arXiv 2403.17806 ("Have faith in faithfulness").
- Miller, Chughtai, Saunders 2024, arXiv 2407.08734 (faithfulness metrics not robust); AutoCircuit
  github.com/UFO-101/auto-circuit.
- Zhang & Nanda 2024, arXiv 2309.16042 (activation-patching best practice).
- Extra, because they bear directly on ablation choice, nulls and runtime reporting: Li & Janson 2024, arXiv
  2409.09951 (optimal ablation); Bhaskar et al. 2024, arXiv 2406.16778 (Edge Pruning); Shi et al. 2024, arXiv
  2410.13032 (hypothesis tests for circuits).

---

## 1. Metric definitions

### 1.1 Notation

- `M` is the full model, `C` the circuit (a set of nodes, or of edges), `∅` the empty circuit, and `M∖C` the complement.
- `m(·)` is the task metric averaged over an evaluation set `D`, computed with every component *outside* the named set
  replaced by its ablation value (§2).
- So `m(M)` is the clean model. `m(∅)` is the model with every in-scope component ablated. `m(C)` keeps `C` live and
  ablates the rest.

### 1.2 Faithfulness (normalised): the common form

    f(C) = [ m(C) − m(∅) ] / [ m(M) − m(∅) ]

- **Where it is used:**
  - SFC [Marks §3.2];
  - MIB Eq. 1 [MIB §3.1];
  - Kharlapenko Eq. 4 [Kharlapenko §4.1.3];
  - Hanna et al., written as `(m − b′)/(b − b′)`, with `b′` the model on the corrupted input [Hanna §4.2].
- **Meaning.** `f = 0` means no better than the empty circuit; `f = 1` means it matches the full model. The
  normalisation is what gives 0 and 1 their meaning [MIB §3.1].
- **Out-of-range values.** `f` is not clipped in any source.
  - A value above 1 means the circuit beats the full model. A negative value means it is worse than empty, e.g. it
    favours the corrupt answer [Miller §4].
  - Miller et al. treat out-of-range per-prompt values as a warning sign, and note that averages can hide extremes on
    both sides [Miller §4.2, Fig. 3 caption].
- **The empty circuit:**
  - Under counterfactual patching, `m(∅)` = the model with every component patched from the counterfactual input, so
    `m(∅) ≈ m(M on x′)` [Hanna §2; MIB §3.1].
  - Under mean or zero ablation, `m(∅)` = every in-scope node mean- or zero-ablated.
  - In SFC's code, the empty circuit is all-False node masks, so every feature and every error term takes the ablation
    value; with `--handle_errors keep`, error terms stay live [code feature-circuits/ablation.py:205-225 @7fbd82b].
- **Which nodes are "in scope".** SFC evaluates only the latter 2/3 of the circuit. The first third is ignored, because
  early components process specific tokens that differ between train and test [Marks §3.2].
  - The code exposes this as `--start_layer`: layers below it are not patched at all, i.e. "given to the model for
    free". The default is `-1`, all layers [code ablation.py:104-109, 130-132 @7fbd82b].
  - This choice changes both `m(∅)` and the size count (pitfall P9).

### 1.3 Aggregation over prompts: ratio of means

- **Both reference codebases use a ratio of means.** Average `m` over the evaluation set first, then normalise once:
  - MIB [code MIB_circuit_track/evaluation.py:17-20, 48-55 @b759df3];
  - SFC [code ablation.py:181-185, 228 @7fbd82b].
- **The alternative gives higher scores.** Miller et al. show that averaging per-prompt ratios gives "substantially
  higher" scores than the ratio of averages [Miller §4.1, Fig. 4].
- **Rule [ours].** Within one circuit's evaluation, use the ratio of means. Across circuits (targets, tasks), report the
  median and IQR plus a pass rate, and a worst-case statistic (the 10th percentile), following Miller's advice to report
  worst-case as well as average [Miller §4.2].

### 1.4 Completeness (SFC)

    completeness(C) = f(M∖C) = [ m(M∖C) − m(∅) ] / [ m(M) − m(∅) ]      ideal 0

- This is the faithfulness of the complement [Marks §3.2, Fig. 3 caption].
- It is computed by inverting the node mask (`complement=True`) [code ablation.py:45-46 @7fbd82b].
- **Kharlapenko** report both curves, `f(C)` (ideal 1) and `f(M∖C)` (ideal 0), against size [Kharlapenko §4.1.3,
  Fig. 7].
- **Relation to our necessity** [inferred]:
  - our `phi_sup_blind` is `(a_pos − a_{M∖C}) / (a_pos − e0)` [code src/eval/wcm_eval.py:106-109];
  - this equals `1 − completeness` *when the same ablation defines both the complement and the empty circuit*;
  - ours zeroes the members in the live model and normalises by the zero-fill empty circuit, so it is the Z-ablation
    instance of `1 − f(M∖C)`.
- **MIB.** MIB does not report completeness: it is intractable for most models, and MIB uses it only via InterpBench's
  AUROC [MIB App. B].

### 1.5 Integrated metrics (MIB): CPR and CMD

    CPR = ∫ f(C_k) dk                (higher is better)
    CMD = ∫ |1 − f(C_k)| dk          (lower is better, 0 best)

- `k` is the fraction of the full graph in `C_k` [MIB §3.1].
- **What each rewards.**
  - CPR rewards components with positive effect.
  - CMD rewards any strong effect, positive or negative, that brings the circuit close to the model.
  - MIB builds CPR circuits from high *signed* scores and CMD circuits from high *absolute* scores [MIB App. D.1].
- **The grid:**
  - `k ∈ {.001, .002, .005, .01, .02, .05, .1, .2, .5, 1}` [MIB §3.1; code evaluation.py:31 @b759df3];
  - hard-coded, not a parameter;
  - `C_k` must satisfy `|C_k|/|N| ≤ k`. The grid is an **upper bound** on size [MIB §3.1].
- **How each circuit is built.** For each `k`, the code keeps `int(k · n_scored_items)` items:
  - by top-n score, `apply_topn`;
  - or by greedy growth from the logits, `apply_greedy`, which is edge-level only and used only for
    information-flow routes [code evaluation.py:35-43 @b759df3; code run_evaluation.py:33 @b759df3].
- **Area computation:**
  - trapezoid rule over the 9 adjacent pairs of grid points;
  - no segment from 0 to 0.001;
  - the x-axis is `k` itself, linear by default, or `ln k` when `log_scale=True` (default `False`);
  - not the realised (weighted) edge count, which is recorded but not integrated [code evaluation.py:12-16, 45-46,
    58-74 @b759df3].
- **Consequence of the linear default** [inferred from the code]:
  - the segment 0.5→1 carries 50% of the x-width and 0.2→1 carries 80%;
  - linear-k CPR is dominated by large circuits, where every method approaches `f = 1`;
  - the log option weights each decade equally.
- **Size count.** MIB counts size in edges, weighted by the fraction of a node's neurons in the circuit [MIB §3.1,
  Eq. 2]. Including a node costs all its outgoing edges, which is why node-level methods "do poorly" on the edge axis
  [MIB §3.3].
- **No SAE nodes.** MIB's circuit track does not use SAE or other featurised nodes. SAE-feature circuits are named as
  future work [MIB App. B].
- **Random baseline.** CPR ≈ 0.25–0.32 and CMD ≈ 0.68–0.75 across tasks [MIB App. D.3]. Area scores therefore have a
  non-zero floor that must be shown.
- **Averaging across tasks.** The leaderboard averages a sigmoid of each CPR/CMD to put tasks on one scale [MIB App. C].
- **No uncertainty.** No seeds or confidence intervals are reported [MIB, whole paper; inferred from absence].

### 1.6 Other figures of merit in the sources

- **ROC against a reference circuit.** Edge (or node) classification against a human circuit:
  - TPR/FPR over the knob sweep, Pareto frontier, "pessimistic segments" between frontier points (Fawcett 2006), and
    area under the curve [Conmy §4.1, Fig. 3; Syed §4.1, Fig. 2; Miller §5];
  - node-level ROC is used where methods are node-level [Conmy App. E.3].
  - **Not usable for us:** there is no reference circuit for latent targets. Hanna shows that overlap with a reference
    is a poor proxy for faithfulness [Hanna §5, App. G].
- **KL to the full model vs number of edges** (Pareto plot) on a held-out set [Conmy §4.2, Fig. 4].
- **Attribution vs activation patching agreement.** R² of attribution against exact patching is 0.27, and the slope of
  0.531 means attribution overstates effects about 2× [Syed §5.1, Fig. 4a]. Kharlapenko use correlation between
  approximated and actual indirect effects, by layer [Kharlapenko App. E.2, Figs. 20-21].
- **Ablation loss gap Δ** (optimal ablation) [Li & Janson §2.1, Eq. 2].
- **Hypothesis tests against size-matched random circuits:**
  - sufficiency: is the circuit more faithful than random circuits of at least `k` edges?
  - partial necessity: does knocking out the circuit hurt more than knocking out random circuits?
  - 100 random circuits per test; one-sided binomial test; `q* = 0.9`, `α = 0.05` [Shi §3.2, §4.1, App. B.2].
- **Ge et al.'s leaf-sum recovery** (Theorem 3.1): covered by the Ge spec. It is an exact identity in the linearised
  graph, not an intervention.

---

## 2. Ablation choices and their documented effects

| ablation | definition | who designs around it | documented effects |
|---|---|---|---|
| **zero** | the component is set to 0 | Kharlapenko (SAE latents) [Kharlapenko §4.1.3]; Conmy's tracr runs [Conmy §4.1]; Shi's induction/tracr runs [Shi App. D.1] | **For SAEs:** Kharlapenko argue it "better aligns" with sparse SAE features [Kharlapenko §4.1.3]. **ACDC:** gives lower-loss subgraphs on the main tasks; recovers tracr perfectly at any τ>0 but does badly there with corrupted activations [Conmy App. E.2, Tables 2-3]; IOI edge AUC 0.539 (zero) vs 0.869 (corrupted) [Conmy Tables 2-3]. **SP/HISP:** with zeros they can optimise a "reset" network, i.e. find circuits that don't exist [Conmy App. L]. **Hanna:** cites Chan et al. against zero ablation [Hanna §2]. **Miller:** use it when you want to include mechanisms such as output-name selection [Miller §6]. |
| **mean** | the component is set to its mean over a reference dataset; SFC uses position-specific means [Marks §3.2 fn. 2] | SFC [Marks §3.2; code ablation.py:19 @7fbd82b, mean over the batch dim]; IOI (Wang et al.) | **Scores vs resample:** higher than resample on IOI, significantly so for node ablation (p = 1e-5) [Miller §4.1, Fig. 3]. **Reference set size:** faithfulness rises monotonically with the size of the mean-ablation set [Miller §4.1, Fig. 4]. **Ablated model can sit above chance:** mean-ablating the whole Sports Players model gives ~33%, near chance, vs 0% under resample [Miller App. D.2]. **Optimal-ablation comparison:** Δ_opt is 33% of Δ_mean (median, IOI) [Li & Janson §2.4, Table 1]. |
| **resample** | the component's value comes from a random other prompt in the dataset | causal scrubbing; Miller's comparisons | **Scores:** lowest faithfulness of the standard choices on IOI [Miller §4.1]. **Optimal-ablation comparison:** Δ_opt is 17.7% of Δ_resample [Li & Janson Table 1]; at 1,000 IOI edges the OA gap is 88% below resample [Li & Janson §3.2]. |
| **counterfactual / patching** (STR) | the component's value comes from a minimally different prompt `x′` with the answer swapped | MIB default [MIB §3.2; code evaluation.py:14 `intervention='patching'`]; Hanna [Hanna §2]; Syed [Syed §3.2]; Bhaskar ("interchange") [Bhaskar §4.1]; SFC's attribution uses clean/patch pairs | **In MIB:** CF circuits beat mean and OA circuits, which score about the same [MIB §3.3]. **Zhang & Nanda:** STR is preferred over Gaussian noise, which is out of distribution and disrupts mechanisms [Zhang & Nanda §3.2, §6]; results depend on *which* tokens are corrupted [Zhang & Nanda App. F]. **Li & Janson:** CF is a partial ablation and can leak input information [Li & Janson §2.2]. |
| **optimal (OA)** | a constant per component that minimises expected loss on the task, learned by SGD from the mean [Li & Janson §2.1, §2.3]; MIB learns it by minimising cross-entropy, ≤1000 steps, early stop at 150 [MIB App. D.2] | Li & Janson's UGS | **Gap:** the smallest ablation gap of any total ablation (Prop. 2.3). Median Δ_opt is 11.1% of Δ_zero. Highest rank correlation with CF (0.907) [Li & Janson §2.4]. **Cost:** intractable at scale [MIB §3.2 fn. 10]. **Risk:** can surface mechanisms the model does not use [Li & Janson App. A]. |

**The cross-cutting finding.** The optimal circuit "cannot be defined" without fixing the ablation and the metric.
- With matched ablations, ACDC, SP and HISP each recover their own ground truth [Miller §5, Fig. 5].
- Tracr X-Proportion: one edge is needed under zero ablation and not under resample [Miller §5, App. D.3].
- **Recommendations:**
  - evaluate edge-defined circuits with edge ablation and position-defined circuits with position-specific ablation;
  - be cautious when scoring a method against a reference found under a different ablation [Miller §6].

**Error terms (SAE substrates):**
- SFC keeps error nodes in the graph and can keep or remove them at evaluation [Marks §3.2].
- Removing residual-stream error nodes severely disrupts the model; removing MLP/attention error nodes disrupts it
  less [Marks §3.2].
- In code, `--handle_errors remove` replaces error terms with the *ablation* value, not zero, despite a help text that
  says "zero ablate all" [code ablation.py:22, 50-53, 58-60 @7fbd82b].
- Kharlapenko do not say how error terms are treated in their faithfulness [Kharlapenko §4.1.3; inferred from
  absence].

---

## 3. Curves and summary statistics

### 3.1 What the literature does with different knobs

- **MIB.** Every method must produce a *score for every component*. Circuits at each `k` are then cut from the scores
  by top-n (or greedy), so all methods land exactly on the same grid.
  - MIB explicitly avoids per-method thresholds λ by looking at Pareto optimality between faithfulness and size
    [MIB §3.2, App. D.1].
  - Submissions may also be up to nine Boolean circuits at given sizes [MIB App. C; MIB-circuit-track README].
  - Where a method's knob is not a ranking (UGS, a sparsity λ), MIB picks λ on validation data (UGS λ = 1e-3 from
    {1e-2…1e-7}) [MIB App. D.2].
- **ACDC / SP / HISP.** Each sweeps its own knob: ACDC τ, SP λ (30 values from 0.01 to 250), HISP the head count k.
  - Methods are compared on the Pareto frontier of the resulting points, with pessimistic segments between them
    [Conmy §4.1, App. D.1].
  - Edge counts for node-level methods (SP, HISP) are the edges between unmasked nodes [Conmy App. D.1-D.2].
- **Syed.** EAP is reported "for all k values", i.e. as a full top-k curve [Syed §3.3].
- **Hanna.** One fixed edge grid for all scoring methods: n = 30, 40, …, 100, 200, …, 1000, extended to 2000 if
  nothing is faithful at 1000.
  - Greedy growth from the logits, then pruning of nodes without parents or children [Hanna §4.2, App. E].
  - The target is ~1-2% of edges at ≥85% faithfulness [Hanna §4.2].
- **Bhaskar (Edge Pruning).** The sparsity target is the knob, hit exactly by binary-searching the mask threshold.
  ACDC is placed by τ, EAP by top-k. Results are compared at edge sparsity (%) [Bhaskar §3, §4.2, App. A].
- **SFC.** The node threshold `T_N` is swept and faithfulness is reported against nodes [Marks §3.2, Fig. 3].
  - The headline small circuit is "the smallest with faithfulness > 0.2": 86 nodes for Pythia, 223 for Gemma
    [Marks §3.3].
- **Kharlapenko.** Faithfulness and complement faithfulness are plotted against node count (inferred: the caption
  gives no unit).
  - The reading reported is the size needed for a faithfulness level: "500 nodes … average faithfulness of 0.6"
    [Kharlapenko §4.1.3, Fig. 7].
  - They do *not* compare methods at matched size [Kharlapenko; inferred from absence].

### 3.2 The size axis

- **Node vs edge.**
  - Edge-level grids penalise node-level methods, because one node costs many edges [MIB §3.3].
  - Conmy add node-level ROCs to be fair to node-level methods [Conmy App. E.3].
  - Miller find edge-level ablation scores IOI "substantially higher" than node-level [Miller §4.1].
  - Rule: the axis unit must be the unit every method natively selects, or one that all of them can be converted to
    without favouring one [inferred from the three].
- **Absolute vs fraction.**
  - MIB uses the fraction of the full graph [MIB §3.1].
  - Hanna and SFC/Kharlapenko use absolute counts [Hanna §4.2; Marks §3.3; Kharlapenko §4.1.3].
  - Bhaskar uses both [Bhaskar §4.2, §5].
  - A fraction is needed only when graphs of different sizes are averaged; *within one substrate* absolute and fraction
    differ by a constant [inferred].
- **Position.**
  - Distinguishing token positions lowers faithfulness consistently [Miller §4.1].
  - SFC nodes are (feature, position) on templatic data, with effects summed over positions on non-templatic data
    [Marks §3.1, App. A.2].
  - Kharlapenko nodes are (feature, token type), aggregated over the tokens of that type [Kharlapenko §4.1.1].
  - So a "node count" means different things across methods unless the position convention is fixed [inferred].

### 3.3 Reading a curve at a grid size (the "≤ k" rule)

MIB defines the circuit at grid point `k` as one with size ≤ k [MIB §3.1]. Conmy's pessimistic segments make the same
conservative choice between frontier points [Conmy §4.1]. For a method that gives a finite set of sweep points
`{(s_i, f_i)}` rather than a ranking:

    reading(g) = f_j  with  j = argmax { s_i : s_i ≤ g }           (largest sweep circuit within budget g)
    if no s_i ≤ g:  reading(g) = f(∅) = 0 for CPR, |1 − 0| = 1 for CMD

Each point of this rule has a source or follows from one:
- **Within budget, no interpolation.**
  - This is MIB's upper-bound definition applied to a finite family [MIB §3.1; ours to apply it to sweeps].
  - No source interpolates faithfulness between two different circuits. A value between sweep points is not a circuit
    anyone has [inferred].
- **Below the smallest size the method reaches.**
  - The only circuit within budget is the empty circuit, so `f = 0` by definition of the normalisation [MIB §3.1
    Eq. 1; ours].
  - This is MIB's behaviour when `int(k·n) = 0`: an empty graph is evaluated at that point [code evaluation.py:37
    @b759df3].
- **Above the largest size it reaches.**
  - The method's largest circuit is within every larger budget, so it is carried forward as a step [ours, consistent
    with "≤ k"].
  - Faithfulness is never extrapolated upward.
- **Pareto variant.** Use `max { f_i : s_i ≤ g }` instead of the largest `s_i ≤ g`.
  - This is the Pareto frontier [Conmy §4.1; MIB §3.2 "Pareto optimality"].
  - It is fair only if the max is taken on a **selection split, not the reported split** (P16) [ours].
  - When curves are non-monotone the two variants differ. Report which one is used.
- **CMD side.** For CMD, use the circuit built from |score| if the method distinguishes the two [MIB App. D.1]. Knob
  methods give one family, so the same circuits serve both.

### 3.4 Area on a fixed grid

    grid G = (g_1 < … < g_n);   x_i = log10 g_i   (log axis)  or  x_i = g_i  (MIB default, linear)
    AUC_f   = Σ_{i=1}^{n−1} (x_{i+1} − x_i) · (r_i + r_{i+1}) / 2,         r_i = reading(g_i)
    AUC_cmd = same with r_i = |1 − reading(g_i)|
    normalised:  nAUC = AUC / (x_n − x_1)        (a mean faithfulness over the grid, on the [0,1] scale)

- **Trapezoid and both axes:** [code evaluation.py:58-74 @b759df3]. MIB uses `ln`; the base cancels in `nAUC`.
- **The normalisation by width** is ours. It makes areas on different grids comparable and readable as "average
  faithfulness per decade" [ours].
- **Random floor.** Report the same area for a size-matched random circuit at every grid point (MIB's random baseline
  [MIB App. D.3]; Shi's size-matched random references [Shi §3.2]).
- **Restricted area.** Also report the area restricted to the sizes *every* method reaches, so a method is not
  credited or punished for range alone [ours].

### 3.5 Matched-size readings and inverse readings

- **Matched size.** Report `reading(g)` at a few named grid sizes, as a table. This is Hanna's fixed-n comparison
  [Hanna §4.2] and Bhaskar's matched sparsity [Bhaskar §4.2]. Always print the realised size `s_j` next to the reading
  [ours].
- **Inverse reading.** Report the smallest realised size whose held-out `f ≥ φ`:
  - SFC's "smallest circuit with f > 0.2" [Marks §3.3];
  - Kharlapenko's "nodes for f = 0.6" [Kharlapenko §4.1.3];
  - Hanna's ≥85% target [Hanna §4.2];
  - the deep-target ratio in our λ sweep (061) is the same kind of number [inferred];
  - report "not reached" rather than extrapolating.

---

## 4. Train vs held-out; per-prompt vs per-task

- **Held-out is the norm in the scaled SAE work:**
  - SFC scores on a test split of contrastive pairs not used for discovery [Marks §3.2, Fig. 3 caption];
  - Kharlapenko estimate effects on one batch and ablate on a held-out batch [Kharlapenko §4.1.3];
  - ACDC's KL-vs-edges plots use a held-out test set [Conmy §4.2];
  - MIB scores on public/private test splits, and the leaderboard uses the private one [MIB App. C, App. D.3];
  - Bhaskar split train/val/test and note that prior work evaluated on the discovery examples [Bhaskar §4.1].
- **Discovery-set scoring is the exception.** CircuitLasso scores on its training prompts (107 README; to confirm in
  the CircuitLasso spec).
- **Hyperparameters on validation.** MIB tunes UGS λ on validation data [MIB App. D.2]. Edge Pruning tunes its node
  target from pilot runs [Bhaskar App. A].
- **Per-task circuits:**
  - SFC, MIB, Kharlapenko, ACDC and Edge Pruning fit one circuit per task over a dataset, and evaluate it as a ratio of
    dataset means;
  - in SFC, effects are averaged over examples (templatic) or summed over positions then averaged (non-templatic)
    [Marks §3.1].
- **Per-prompt circuits.** Ge's circuits are per prompt (Ge spec). No source we read evaluates a per-prompt circuit on
  other prompts, so that reading is **ours** and is labelled as such [ours; README rule 4].
- **Cross-task faithfulness** is a separate, informative test: a circuit scored on a related task [Hanna §5;
  Kharlapenko §4.1.3, Fig. 8, cross-task specificity].

---

## 5. Efficiency reporting

- **Pass counts:**
  - EAP is 2 forward + 1 backward, a constant number of passes [Syed abstract, §3.2];
  - EAP-IG with `m` steps costs ≈ `m` forward-backward passes (m = 5 used; m = 2 unfaithful on Greater-Than)
    [Hanna §3, App. C];
  - activation patching was dropped on GPT-2 IOI for cost [Syed §4.1].
- **Wall-clock on one stated GPU:**
  - Edge Pruning, ACDC and EAP were all run on one H100 "for a fair runtime comparison", at several dataset sizes
    (200 / 400 / 100k examples) [Bhaskar Table 1];
  - ACDC on IOI took 8 minutes on an A100; runs keeping >10% of edges took hours [Conmy App. F.1].
- **Memory and caps:**
  - MIB marks intractable cells "-" (runtime: EActP, OA; memory: UGS at batch size 1 on an 80 GB GPU) and excluded
    methods taking more than one week [MIB §3.2 fn. 10, App. D.3];
  - Edge Pruning notes higher memory than EAP and reports its 13B run as 32 H100s and ~5,000 GPU-hours [Bhaskar §5].
- **Recommended fields** [ours, following Bhaskar's same-GPU rule]:
  - `secs_setup` (dictionary load, caches);
  - `secs_scores` (one attribution or score pass, if any);
  - `secs_per_circuit`;
  - `secs_per_curve` (the full knob sweep);
  - `peak_mem_mb` (`torch.cuda.max_memory_allocated`, reset per stage);
  - `n_fwd`, `n_bwd` (counted with hooks);
  - batch size and dtype;
  - GPU model.
- **Per curve, not just per circuit** [inferred from §3.1]:
  - SFC, Ge and EAP get the whole curve from one score pass, because the threshold is free once scores exist;
  - WCM and CircuitLasso pay one optimisation or solve per knob value;
  - per-circuit and per-curve costs can therefore rank methods in opposite orders.

---

## 6. Recommended protocol for 107's three arenas

### 6.1 Size axis and grid

1. **Unit = distinct latent nodes `(site, latent)`, position-collapsed** [ours].
   - **Why:** WCM selects latents. CircuitLasso is position-free (token-averaged codes).
   - **Converting the others:**
     - SFC's (feature, position) nodes and Ge's per-prompt nodes are collapsed to distinct `(site, latent)` for the
       axis;
     - the native count (with positions, with error nodes, edges where defined) is reported in a second column.
   - **Why it matters:** MIB shows the unit changes the ranking [MIB §3.3]; Miller shows positions change scores
     [Miller §4.1].
   - **Error nodes:** SFC error nodes are counted in a separate column and **not** on the latent axis, since only SFC
     has them [ours]. Whether they stay live at evaluation follows the method's own default for its own score and
     "kept" for ours (P10).
2. **Grid = 1-2-5 per decade, absolute counts:**
   - `G = {10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5}` (13 points);
   - this is MIB's own 1-2-5 spacing [MIB §3.1], transposed from fractions of edges to latent counts [ours];
   - per arena, print `g / |node universe|`, so the grid can also be read as MIB-style fractions.
3. **Area on a `log10` axis, normalised by width (§3.4)** [MIB code supports it: evaluation.py:65-67 @b759df3; the
   choice of log is ours].
   - A linear axis over 10¹–10⁵ would give 90% of the weight to 10⁴–10⁵, where SFC and Kharlapenko circuits never
     reach (86–500 nodes) [Marks §3.3; Kharlapenko §4.1.3].
   - Report the linear-axis area too, labelled MIB-default, for compatibility.

### 6.2 Readings and statistics (every arena)

4. **Curve per method from its own knob, read by the ≤ k rule (§3.3).**
   - Below a method's smallest circuit the reading is `f(∅) = 0`; above its largest, the largest circuit carries
     forward [MIB §3.1; ours].
   - Report each method's reached range `[s_min, s_max]`.
5. **Summary statistics per (arena, method, evaluation):**
   - (a) `nAUC_f` and `nAUC_cmd` over the full grid;
   - (b) the same over the common reached range;
   - (c) matched-size readings at `g ∈ {100, 1e3, 1e4}` with the realised size;
   - (d) inverse readings at `φ ∈ {0.6, 0.75}`, i.e. Kharlapenko's level and our band floor
     [Kharlapenko §4.1.3; src/eval/wcm_eval.py:87];
   - (e) completeness / necessity at the same grid points [Marks §3.2; Kharlapenko Fig. 7b];
   - (f) the size-matched random-circuit row [MIB App. D.3; Shi §3.2].
6. **Aggregation:**
   - within a target or task, the ratio of means [MIB and SFC code];
   - across targets, the median and IQR, the 10th percentile, and the pass rate [Miller §4.2; ours];
   - ΔAUC between two methods on the same targets as a paired bootstrap over targets, 95% CI [ours; no source gives
     CIs];
   - vacuous denominators (`|m(M) − m(∅)|` small) are excluded and counted, never divided [src/eval/wcm_eval.py:81;
     MIB code has no guard, evaluation.py:55].
7. **Selection.** Where a knob value or a Pareto frontier is chosen, choose it on the train or selection split and
   report on held-out [MIB App. D.2; Bhaskar §4.1; ours].

### 6.3 Ablation per arena

Each row's *own* score uses the ablation the method was designed around. *Our* score uses our fixed set. Both are
shown [Miller §5-6: the circuit is only defined relative to the ablation].

| arena | method's own evaluation (its ablation) | our evaluation (same for both methods) | why |
|---|---|---|---|
| A: Ge vs WCM, GPT-2 transcoders + attn SAEs | **Ge's leaf-sum recovery in the linearised graph** (no intervention; Ge spec) | Z (zero fill), A (sparsity-preserving mean fill), C (contrast fill), necessity, R (resample), on the real model | Zero is natural for sparse ReLU features [Kharlapenko §4.1.3], but zero can over-credit [Conmy App. L], so mean and resample are shown with it [Miller §4.1]. The weak random control on ReLU dictionaries means the random row (5f) is mandatory here (README arena A caveat). |
| B: CircuitLasso vs WCM, GPT-2 Top-K SAEs | **CircuitLasso's Marks-style faithfulness/completeness**, with the ablation their paper uses (CircuitLasso spec to fill in), on their training prompts **and** on held-out | Z, A, C, necessity, R | Held-out is added because discovery-set scoring is the documented exception [Bhaskar §4.1]. |
| C: SFC vs WCM, Pythia-70m Marks SAEs | **SFC's `ablation.py`**: mean ablation (default), test split, `--examples` 100, `--handle_errors default`, `--start_layer` as in the paper **and** `-1` (all layers) [code ablation.py:74-109 @7fbd82b; Marks §3.2] | Z, A, C, necessity, R, on the same test split | Mean is SFC's design ablation [Marks §3.2]. Counterfactual patching is what MIB finds strongest [MIB §3.3]. Since SFC's tasks come with counterfactual pairs, also report SFC's `ablation.py` with `--ablation resample` and MIB-style patching as extra columns [Miller §6; ours]. |

### 6.4 A method with no knob for some sizes

8. **The method is never given an operating point by us** (README rule 3).
   - Its missing range is reported, not filled: the ≤ k rule gives `f(∅)` below and a carried step above (§3.3).
   - Its reached range is printed in the table.
   - The common-range area (5b) is the fair comparison of shape [ours].
9. **Densify the sweep, don't invent a rule.** If a method's knob produces sparse sizes (WCM λ, CircuitLasso λ),
   add knob values until every grid point inside the reached range has a sweep point within a factor of 2 below it.
   The density used is recorded [ours; cf. SP's 30-value λ sweep, Conmy App. D.1].
10. **Per-prompt methods (Ge):**
    - each prompt's curve is read on the grid by the ≤ k rule, then readings are averaged across prompts at each grid
      point [ours];
    - the per-prompt size distribution is shown;
    - held-out readings of per-prompt circuits are labelled "ours" (§4).

### 6.5 "Scored by its own evaluation and by ours" side by side

11. **One table per arena, rows = method × fit split, column groups:**
    - `own eval (train)`, `own eval (held-out)`, `our eval (held-out)`, `efficiency`;
    - each group carries `nAUC_f`, `nAUC_cmd`, the matched readings and the inverse readings;
    - a two-panel figure per arena: the left panel is the method's own metric vs latent count, the right panel is our
      Z/A/C/necessity vs latent count;
    - both methods appear in both panels on the same x-axis, with the random row dashed.
12. **The home-turf effect is the diagonal.** For method X, (own eval − our eval) on X's own row versus on WCM's row
    is the visible home-turf gap (README rule 5) [Miller §6: be cautious scoring against a criterion built under a
    different ablation].
13. **Cross-arena.** No averaging of numbers across arenas (README rule 2). If a single summary is ever needed, use
    MIB's per-task sigmoid normalisation and label it [MIB App. C].

### 6.6 Efficiency (every arena)

14. Run on the same GPU and dtype, and report the §5 fields per circuit and per curve [Bhaskar Table 1]. A method that
    exceeds a stated cap (e.g. 24 h per curve, or OOM on the 16 GB card) is shown as "-" with the reason, as MIB does
    [MIB §3.2 fn. 10, App. D.3].

---

## 7. Pitfalls documented in the literature

- **P1. Discovery ablation ≠ evaluation ablation.** A method looks better when scored with the ablation it optimised
  [Miller §5-6; Conmy App. L; Li & Janson §3.2]. Countered by the two-evaluation table (§6.5).
- **P2. Evaluating on discovery prompts.**
  - Inflates scores; prior work commonly did it [Bhaskar §4.1].
  - SFC ignores the early third precisely because tokens differ between train and test [Marks §3.2].
- **P3. Node vs edge axis.** Edge grids penalise node methods [MIB §3.3]. Edge ablation scores higher than node
  ablation [Miller §4.1].
- **P4. Token positions.** Position-specific circuits score differently, and edge-level IOI with positions has a median
  above 100% [Miller §4.1].
- **P5. Mean of ratios vs ratio of means.** Per-prompt averaging inflates [Miller §4.1]. Both reference codes use the
  ratio of means.
- **P6. Metric choice.**
  - KL's gradient is zero at the clean input, so attribution-patching scores are zero with KL as the metric [Syed §3.3].
  - Probability misses negative components [Zhang & Nanda §4.2, §6].
  - Task-metric matching was unstable for ACDC (IOI logit diff 1.5–5.0 vs model 3.55) [Conmy App. C, E.1].
  - Logit difference is the standard for task circuits [MIB §3.1 fn. 6; Zhang & Nanda §6].
  - (Our arenas A/B use a latent read, not a logit; the analogous choice is which read of the target is used, and it
    must be one read throughout [src/eval/wcm_eval.py:22-24].)
- **P7. Overlap as a proxy.** The smallest IOI EAP circuit containing ≥90% of the manual circuit's nodes is 0% faithful
  [Hanna App. G]. Never compare circuits by overlap alone.
- **P8. Linear-k area.** MIB's default linear trapezoid is dominated by the largest sizes [code evaluation.py:58-74
  @b759df3; inferred]. Use the log axis on a 4-decade grid (§6.1).
- **P9. Free layers.** Excluding early layers (`start_layer`, SFC's "ignore the first 1/3") raises `f` and shrinks the
  counted size [Marks §3.2; code ablation.py:130-132 @7fbd82b]. Report it and apply the same scope to both methods.
- **P10. Error terms.**
  - Keeping residual error live carries much of the computation [Marks §3.2].
  - SFC's `remove` sets error terms to the ablation value, not zero [code ablation.py:50-60 @7fbd82b].
  - Use the same error handling for both methods within one evaluation.
- **P11. Attribution approximations in early layers.**
  - Attribution-patching IE correlates poorly with true effects before layer 6, and non-residual SAEs are adequate only
    in the last third [Kharlapenko App. E.2].
  - R² = 0.27 against activation patching [Syed §5.1].
  - Methods that rank by attribution will look worst on deep-to-shallow circuits; report curves by target depth band
    [ours].
- **P12. Ablation reference set size.** Mean-ablation faithfulness grows with the size of the reference set
  [Miller §4.1, Fig. 4]. Fix and report it, and use the same set for both methods.
- **P13. Unequal tuning effort.**
  - ACDC is sensitive to the corruption distribution, hyperparameters and metric [Conmy §6].
  - MIB tunes λ on validation [MIB App. D.2].
  - Give every method the same sweep budget and record it [ours].
- **P14. Cancellation.** Averages can hide per-prompt values far above 100% or below 0 [Miller §4.2]. Report
  percentiles, not only means.
- **P15. Vacuous denominators.** `m(M) ≈ m(∅)` gives meaningless or infinite `f`. MIB's code divides by zero
  [code evaluation.py:55 @b759df3]. Exclude and count these cases.
- **P16. Selecting on the test set.** A Pareto frontier or best knob chosen on the reported split is optimistic.
  Select on train or validation [MIB App. D.2; Bhaskar §4.1].
- **P17. No random floor.** Random circuits already score CPR ≈ 0.25–0.32 [MIB App. D.3]. Areas without the random row
  are uninterpretable. Use size-matched random references [Shi §3.2].
- **P18. Out-of-distribution ablations.**
  - Gaussian-noise corruption disrupts mechanisms [Zhang & Nanda §3.2].
  - Total ablations "spoof" input information [Li & Janson §2.1].
  - Mean-ablating everything can leave the model above chance [Miller App. D.2].
  - The empty circuit's `m(∅)` must be reported per ablation, since it sets the zero of every score.
- **P19. Per-curve vs per-circuit cost.** Threshold methods amortise one score pass over the whole curve; optimisation
  methods do not. Report both (§5).

---

## 8. Where 107's README departs from established practice

1. **The MIB area (rule 3) is not MIB's CPR.**
   - MIB's grid is *fractions of edges of the whole computation graph*, from 0.001 to 1, on a linear trapezoid, with
     every method providing a full ranking. It has no SAE nodes [MIB §3.1, App. B; code evaluation.py:31, 58-74].
   - Our grid is *absolute latent counts* 10¹–10⁵ on a log axis, with knob sweeps instead of rankings.
   - It should be called "MIB-style area (log latent-count grid)", not CPR/CMD.
2. **The citation for matched-size readings (rule 3) is weak.**
   - Kharlapenko et al. read *one method's* curve against node count ("500 nodes for 0.6") and do not compare methods
     at matched sizes.
   - The established precedents for matched-size comparison across methods are Hanna et al. (fixed edge grid) and
     Bhaskar et al. (matched sparsity). Cite those, and keep Kharlapenko for the node-count axis and the inverse
     reading.
3. **"No operating point" (rule 3) still needs a reading rule.**
   - MIB's ≤ k definition *is* a rule, and so is a Pareto frontier.
   - The README should name the ≤ k rule, the empty/step behaviour outside the reached range, and selection on train
     (§3.3, §6.4).
4. **Held-out evaluation of per-prompt circuits (rule 4) has no precedent.** Ge, ACDC and EAP circuits are scored on
   their own prompts or tasks. It is a legitimate test, but it is our design and must be labelled as such.
5. **Ablation and metric details in the README's literature list.**
   - Syed's KL point is about KL *as the attribution target* (zero gradient), not KL as an evaluation metric. ACDC uses
     held-out KL as its evaluation [Conmy §4.2].
   - Kharlapenko's "unreliable before layer 6" is about the IE approximation's correlation with real effects
     [Kharlapenko App. E.2]. Their circuit search itself started at layer 10 [Kharlapenko §4.1.3].
6. **Rule 5 (scored twice) agrees with Miller et al.'s recommendation** to keep the evaluation ablation in view.
   - One gap: the README's "our tests" are latent-read tests, so in arena C, SFC's own score is a logit-difference
     faithfulness and ours a latent/logit read under WCM's fills.
   - Our side must be run with the logit objective's read so the two are about the same endpoint (README arena C
     already says logit objective; keep it explicit in the table).
