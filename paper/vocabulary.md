# Vocabulary sheet

Settled 2026-09-22 (Linear DAN-10; the issue comment holds the field survey and
sources). The paper uses only the terms in the **Paper** column. Code, READMEs
and data files keep their old names; this sheet is the mapping, not a rename.

**Title:** *What Makes a Latent Fire?*
**The activity:** *latent-targeted circuit discovery*. **The method:** *weighted
circuit masking (WCM)*. **The object:** *a weighted circuit for a target latent*. In prose the circuits are *dataset-level*
(reusable across the target's contexts), which places them between per-prompt
attribution graphs and weight-based global circuits. "Latents explain latents"
is used as a slogan in the introduction only.

## Terms

| Concept | Paper | Old names (prose, code, data) |
|---|---|---|
| SAE unit vs the concept it may encode | **latent** (unit); **feature** (concept) | latent, feature (used interchangeably) |
| What is being explained | **target latent** | seed, seed latent, endpoint |
| Elements of a circuit | **nodes**; "circuit latents" in prose | members, membership |
| Signed effect of a node | **not used**: nodes carry no role labels (2026-09-22, DAN-72). Attribution baselines speak of the *sign of a score* only | activator / inhibitor / support (`ablation_support`), excitatory / inhibitory, drivers, brakes |
| The target's value before vs after the sparsity nonlinearity | **pre-activation** (before Top-K / JumpReLU) vs **activation** (after it). Every score states which one it reads, and one ratio never mixes the two (DAN-66) | preact, `_pre`, relu-pre; post-TopK, `_tk`, code |
| Per-node fitted multiplier | **scaling coefficient** α | amplitude, alpha, gain, `amp_*`, `keep_scales` |
| Membership + coefficients | **weighted circuit** | tri-amp circuit |
| The method (family) | **weighted circuit masking (WCM)**, with the ablation methods it was trained under as a spec: WCM[Z], WCM[C], WCM[A], WCM[Z+C], **WCM[Z+C+A]** (the paper's default, plain "WCM"). Z = zero ablation, C = mean ablation over contrast contexts, A = mean ablation over activating contexts. Write the spec with "+", never run together (ZCA is ZCA whitening). Loss weights live in the protocol, not the name. | tri-amp (= WCM[Z+C+A]), dual-floor mask (+ amplitudes = WCM[Z+C]), single-floor arms, `mask_floor_source`, `free_amplitude` |
| Gate-only special case (α fixed at 1) | **unweighted circuit masking**, never abbreviated. The qualifier also keeps it clear of "circuit masking", a hardware side-channel term | gate-only mask, gate400, abl-mask |
| Signed-coefficient variant | **signed WCM** | neg-amp signed, `signed_amplitude` |
| Value given to non-circuit latents | **ablation value** / **ablation method** | floor, fill |
| Zero / mean replacement | **zero ablation**; **mean ablation over activating / contrast contexts** | zero floor, posctx-mean floor, negctx-mean floor |
| Training under all three at once | **trained jointly under zero ablation and two mean ablations** | triple floor |
| Mean ablation that keeps the code k-sparse | **sparsity-preserving mean ablation** | respect_topk, Top-K re-imposed fill, "on-manifold" |
| Circuit alone reproduces the target where it fires | **faithfulness** (SFC normalised form) | free0, freeM, `F0`, `FMd`, `freeM_topk`, closure |
| Ablating the circuit removes the target | **necessity** = 1 − SFC completeness = (a(M) − a(M∖C)) / (a(M) − a(∅)) | φ_sup, `sup`, suppression |
| Setting the circuit's values where the target is silent switches it on | **sufficiency to induce**; symbol **φ_ind** | φ_cf, `cf_amp`, `phi_cf_*`, drive, counterfactual faithfulness |
| Circuit nodes held at clean-run values vs recomputed from the edited stream | **clamped** vs **recomputed** | pinned vs free, φ_pin |
| Contexts where the target fires | **activating contexts** | posctx, positive contexts, top contexts |
| Retrieved contrasting contexts | **contrast contexts**; **random contrast contexts** when drawn at random | negctx, negative contexts |
| Fitting vs scoring contexts | **train / held-out** | train / held-out |
| Random control circuits | **random-circuit baseline** (size- and site-matched, coefficients fitted) | null, fitted-amplitude null, permuted null |
| SAE reconstruction residual | **SAE error (term)**; **error node** when it is a graph node | SAE error, error node |
| Sites | **attention-output, MLP-output, residual-stream** (post); hook names only in the appendix | attn / mlp / resid, `att`, `hook_*` |
| Different reruns give different nodes | **non-identifiability**; the set of valid circuits is an **equivalence class** ("non-unique" is fine in prose) | non-uniqueness, underdetermined, families |
| Loss term keeping the target at its natural Top-K rank | **rank-keep** (term), weight γ_R | rank_mode "keep", `rank_weight`, rank term |
| Whether a circuit restores the target without lifting its neighbourhood | **specificity** (reported, never claimed for the method) | off-target inflation, concept amplifier (prose only) |
| Latents a circuit lifts above the unmodified model's Top-K cut | **latents lifted above the cut** (vs the empty circuit) | switch-ons, switched_on |
| Latents that co-fire with the target at its site | **siblings** | sibling latents |
| Targets that barely enter their site's Top-K | **near-threshold targets**, reported separately | near-threshold stratum |

## Definitions to give on first use

- **Latent / feature:** "We call the SAE's learned directions *latents*, reserving *feature* for the concept a latent may encode" (following Gemma Scope, 2408.05147).
- **Scaling coefficient α:** a fitted multiplier on each node's live activation, a non-binary extension of a learned membership mask (cf. Edge Pruning's relaxed mask); unlike that mask it may exceed 1.
- **Sparsity-preserving mean ablation:** non-circuit latents take their mean values only up to the dictionary's k active latents per token, so the code stays k-sparse (motivated as keeping codes in-distribution; do not claim "on-manifold").
- **Faithfulness:** (a(C) − a(∅)) / (a(M) − a(∅)), with the ablation method always stated, and a read the same way (pre-activation or activation) in every term.
- **Rank-keep:** "the target must hold its natural rank among its site's latents in each ablation run"; zero once it does, so it never pulls the circuit towards a state the unmodified model lacks.
- **Contrast contexts (protocol v1, 2026-09-24):** "corpus sequences ranked by highest cosine similarity to any activating context, activating contexts excluded, kept only if the target stays out of its Top-K". Silence means post-Top-K activation 0, not a zero pre-activation.
- **Necessity:** (a(M) − a(M∖C)) / (a(M) − a(∅)), i.e. one minus SFC completeness; the same formula on every model.
- **Sufficiency to induce:** "a denoising-style test (Heimersheim & Nanda, 2024) in which the circuit's latents are set to their activating-context values in contrast contexts". The silence rule, if any, is stated once in the protocol; if a silence check is adopted (DAN-64), add "(verified silent: …)" at that one place.
- **Contrast contexts (general):** they are not minimal pairs; "hard negatives" is the ML analogue. **Random contrast contexts** are drawn uniformly from the corpus (and silence-checked the same way). The protocol-v1 definition of the default ("close") source is the entry above. The pre-v1 retrieval *store* (kNN to the centroid, unchecked) is called "the retrieval store" and serves only the attribution baselines.
- **Clamped vs recomputed:** a node is clamped when fixed to its clean-run value, recomputed when re-encoded from the edited stream; cite circuit tracing's constrained patching as the analogue.

## Do not use

- **suppression** as a metric name (collides with "copy suppression" and suppress-as-inhibit)
- **counterfactual faithfulness**, or "counterfactual" for our contexts (means paired-input values in MIB and Li & Janson)
- **denoising sufficiency** (implies paired runs and restoration)
- **non-activating contexts** for retrieved contexts, unless silence is verified (DAN-7)
- **latent circuits** (Langdon & Engel's method) or **local circuits** (input-specific in Ge et al. and circuit tracing)
- **completeness** for the circuit-alone test; **minimality** unless tested
- **knockout** for zero ablation (a knockout is a mean ablation in IOI)
- **on-manifold** as a claim; **steering coefficient** for α; **replacement model** for our SAE-spliced model; **resample ablation** for our contrast-context mean; **global circuit** for a dataset-level circuit
- **excitatory**, **inhibitory**, **inhibitor**, **role** (of a node), **release**
- **closure**, **drivers**, **brakes**, **seed**, **endpoint**, **floor**, **fill**, **amplitude**, **gain** in formal text

## Decided since the first version (2026-09-22)

- **Silence wording:** sufficiency to induce is defined "in contrast contexts", which is true whether or not the protocol checks silence; the silence rule lives in the protocol only.
- **Added after the implementation audit (DAN-65):** pre-activation vs activation; necessity's formula; random contrast contexts.

- **Method:** weighted circuit masking (WCM). Collision check: no use of the phrase, and no interpretability method called WCM (two unrelated robotics papers, arXiv 2607.29613 and 2607.22999). Cite Caples et al. (2025), Edge Pruning and CircuitLasso as the closest methods.
- **Gate-only baseline:** unweighted circuit masking, not abbreviated. Plain "circuit masking" is avoided because it is an established hardware-security term (arXiv 2106.12714; NIST "Masked Circuits").
- **Roles (revised 2026-09-22):** no role labels at all. Leave-one-out effects are strongly non-additive and no signed subset (gradient sign, fitted sign, leave-one-out, contrast-context detectors) passes a set-level test (experiments/054-inhibitor-roles). Necessity and sufficiency to induce intervene on the circuit as one set; inhibitors are not a contribution and are not mentioned.
- **Symbols:** φ_cf renamed **φ_ind**; φ_sup and φ_pin appear only after their prose names (necessity, clamped faithfulness).
