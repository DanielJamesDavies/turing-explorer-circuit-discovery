# 107: the method comparison with other papers' true algorithms (plan, 2026-11-01)

Status: **plan settled, code cloned, nothing run.** Asked by Daniel on 2026-11-01; decisions recorded below
(2026-11-01 evening). This supersedes the 083 / 098 comparison rows (see "What leaves the comparison"). Linear: DAN-188
and its sub-issues DAN-189 to DAN-204.

The exact algorithm, pseudocode, setup, fidelity checks and open ambiguities of every method are in `specs/`:

| spec | method |
|---|---|
| `specs/sfc.md` | Marks et al. 2025, Sparse Feature Circuits (incl. §12: branches, bug fixes, the 2024 memory blow-up) |
| `specs/ge.md` | Ge et al. 2024, hierarchical attribution |
| `specs/dunefsky.md` | Dunefsky et al. 2024, transcoder circuits |
| `specs/circuitlasso.md` | Yin et al. 2026, CircuitLasso |
| `specs/attribution-graphs.md` | Ameisen et al. 2025, attribution graphs (circuit-tracer) |
| `specs/caples.md` | Caples et al. 2025, learned SAE-latent masks |
| `specs/evaluation.md` | the shared evaluation: curves, size grid, area, statistics |

Where this README and a spec disagree, the spec is the more detailed source; fix the README.

## The rule

1. **Each baseline is the paper's own algorithm, as published.** Where official code exists we run it unmodified at a
   pinned commit. Where it does not, we re-implement the published algorithm and label the row "re-implementation".
   Anything we change makes the row ours, and it is labelled "our adaptation" or left out.
2. **WCM runs on every dictionary a baseline runs on.** Comparisons are only made within one substrate (model plus
   dictionaries); comparing across dictionaries is invalid.
3. **No operating point is chosen for a baseline. No size matching.** Each method gets a faithfulness-vs-size curve
   from its own knob (WCM λ, Ge τ, SFC node threshold, Dunefsky branches, CircuitLasso λ, AG pruning, Caples λ).
   Curves are compared by (`specs/evaluation.md`):
   - each method's natural operating point (its paper's default);
   - matched-size readings of the curves (precedents: Hanna et al., Bhaskar et al.);
   - a **MIB-style area**: the area under the curve on a log latent-count grid, 1-2-5 steps from 10 to 10⁵ (13
     points). This is not MIB's CPR / CMD, and is named "MIB-style" for that reason.
   Curves are read at "size ≤ k" with no interpolation; below a method's range f = 0, above it the last value carries.
   Any selection is made on training prompts and reported on held-out ones.
4. **Per-prompt methods are not asked to be something else.** Ge, Dunefsky and attribution graphs are per prompt. Both
   methods are fitted on the same prompt subset, then compared on those prompts (home turf), on held-out prompts
   (generalisation; scoring a per-prompt circuit on held-out prompts is our design and is labelled), and on efficiency
   (wall-clock and peak memory per circuit, same GPU).
5. **Every row is scored twice:** by our tests (Z, A, C, necessity, R, on the real model) and by the paper's own
   evaluation, so the home-turf effect is visible.
6. **Size** counts distinct (site, latent) pairs, positions collapsed. Each method's native count (e.g. SFC's
   (site, position, latent), Caples' per-position masks) goes in a second column (decision 7).
7. **Statistics:** medians with p10, paired bootstrap CIs over targets or tasks.

## Arenas (decided 2026-11-01)

| arena | model + dictionaries | baseline | code | WCM side |
|---|---|---|---|---|
| **C** | Pythia-70m-deduped + Marks SAEs (+ error nodes) | Sparse Feature Circuits | official, `feature-circuits@7fbd82b` | new adapter (DAN-194) |
| **A1** | GPT-2 small + Dunefsky transcoders (MLPs) + Kissane attention SAEs (`hook_z`) | Ge et al. 2024 | authors' library code, OpenMOSS `Language-Model-SAEs@v0.1.0` | new transcoder + attention-SAE adapter (DAN-195) |
| **A2** | GPT-2 small + Dunefsky transcoders only (attention through raw heads, OV with frozen pattern) | Dunefsky et al. 2024 | official, `transcoder_circuits@8721ff0` | the DAN-195 adapter with transcoder sites only |
| **B** | GPT-2 small + OpenAI Top-K residual SAEs (32k, k = 32) | CircuitLasso (Yin et al. 2026, EMNLP) | none: re-implementation + our labelled read-out rule | the pod's GPT-2 run (070 adapter) |
| **D1** | Gemma-2-2B + circuit-tracer per-layer transcoders (PLT) | attribution graphs | official, `circuit-tracer@255d989` | Gemma transcoder adapter (DAN-203) |
| **D2** | Gemma-2-2B + Gemma Scope 16k residual SAEs | Caples et al. 2025 | authors' code `ScalableSAECircuits@f54a4ef`, **2B configuration** | the 050 / 048 Gemma Scope path (DAN-203) |

Attribution patching and integrated gradients are not standalone rows. They are SFC's node-effect estimators and appear
inside arena C as SFC uses them. D1 and D2 share the model only, not the dictionaries, so they are two separate
comparisons (rule 2).

### Arena C: Sparse Feature Circuits (`specs/sfc.md`)

- **Code:** main HEAD `7fbd82b` with `dictionary_learning@61ac634`; nnsight 0.3.7, torch < 2.6, numpy < 2, in its
  own environment. Run nodes only: `circuit.py --aggregation none --nodes_only --num_examples 100 --batch_size 10`
  (~4 GB peak). SFC's own evaluation uses node ablation only, never edges (issue #14, confirmed by the owner).
- **The 2024 memory blow-up** was dense per-unit vectors at `64d1cb3` (`attribution.py:342,377-380`), fixed in
  `2b3ffae` ("more efficiently storing sparse tensors"), which HEAD contains. The `jvp-fix` and `iclr-updates`
  branches are contained in main; no branch is needed.
- **Three labelled evaluator edits** (their evaluation script does not run at HEAD as shipped): the `load_examples`
  import (`loading_utils` → `data_loading_utils`), `model_name` set to Pythia, and the issue #15 TypeError fix at
  `ablation.py:47-49`. These touch evaluation only, not circuit discovery.
- **Their settings:** IG with 10 steps; 15 node thresholds from 1e-4 to 1 for the curve; mean ablation over the batch;
  evaluation on the first 40 test examples of the shortest length, per-position means.
- **The 2×2:** {SFC, WCM} × {behaviour (their SVA tasks, logit difference), latent targets}. SFC rooted at a latent
  needs a `metric_fn` plus code changes for full circuits, so the latent-rooted SFC cells are **our adaptation**.
  WCM on the behaviour tasks uses the logit objective.
- **Held-out:** their test split leaks their training split (081). Report on their test split and on a deduplicated
  split, both labelled (decision 8).
- **Fidelity:** reproduce one published number first: the across-RC circuit at threshold 0.1 (86 nodes, faithfulness
  0.21). Checks F1–F11 in the spec.

### Arena A1: Ge et al. (`specs/ge.md`)

- **Code:** the authors' own library implementation, `TransformerHierachicalAttributor` / `HierachicalAttributor` in
  OpenMOSS Language-Model-SAEs (`3ff7275`, `f9cc2d6`, tag v0.1.0 `67bfb0b`, with
  `tests/intergration/test_attributor.py`; removed by v1.1.0). Pinned at v0.1.0. The README's earlier "no code
  released" was wrong.
- **The gate:** `torch.where(v * grad > threshold, grad, 0)`, signed, during the backward pass; default 0.1.
- **Nodes:** transcoder latents and attention-SAE latents. Residual SAEs are **not** nodes. There is no GPT-2
  word-embedding SAE.
- **Dictionaries:** Ge's own are not public. Public dictionaries of the same kind: Dunefsky transcoders (validated in
  035) and Kissane attention SAEs (`ckkissane/attn-saes-gpt2-small-all-layers`, on `hook_z`; L5 and L7 are 49,152
  wide). The row is labelled "Ge et al.'s code on public GPT-2 dictionaries".
- **LayerNorm:** their code does not freeze it; their paper's App. B says to. Recorded as an ambiguity; both are
  reported if they differ.
- **Fidelity checks:** (1) with gating **off**, the leaf sum reproduces the root in the linear graph (the old "τ = 0"
  check was wrong because the gate is signed); (2) gating during the backward, not after, on a toy graph; (3)
  hierarchical beats post-hoc pruning on IOI-style prompts (their Fig. 2b).
- **WCM side:** transcoder intervention semantics from 035 (`mlp_out ← mlp_out + (ĉ − c) W_dec`, transcoder error
  passes through). Report the anchor-support rate per target; on ReLU dictionaries the random null has been weak (035,
  052), and is reported as uninformative if it is.

### Arena A2: Dunefsky et al. (`specs/dunefsky.md`)

- **Code:** `transcoder_circuits@8721ff0`, weights `pchlenski/gpt2-transcoders@4f3ca345`; TransformerLens 1.11.0,
  torch 2.2.0, `sae_training`, in its own environment.
- **Calls:** `make_sae_feature_vector` → `greedy_get_top_paths(num_iters, num_branches, do_raw_attribution=True)` →
  `paths_to_graph`. The defaults (`do_raw_attribution=False` with capping) hit an operator-precedence bug in the cap,
  so the raw path is used and the reason recorded.
- **Nodes:** transcoder latents; attention goes through raw heads (OV with a frozen pattern), with no attention SAEs.
  So the node universe differs from A1, which is why A2 is its own arena; WCM runs on transcoder sites only.
- **Size knob:** `num_branches` (and `num_iters`). Positive contributions only; b_O not attributed; BOS source excluded.
- **Fidelity:** the walkthrough's mlp8tc[89] at 3×15: root 7.5700, top path 0.8468, 24 nodes / 33 edges / 11
  latents; plus the greater-than checks. Their own evaluation is qualitative, so our tests carry the comparison.

### Arena B: CircuitLasso (`specs/circuitlasso.md`)

- **No code released;** EMNLP 2026 main (v2, 2026-10-09). Re-implementation, reusing 098's `lasso_core.py` (FISTA),
  which needs a no-intercept mode and a logistic mode; its λ = paper λ / n.
- **Data:** GPT-2 with the OpenAI 32k Top-K residual SAEs after each block (their layer ℓ = our block ℓ − 1). CoLA's
  8,551 training sentences with `<|endoftext|>` prepended, codes averaged over tokens (including that token); one row
  per sentence. Not our corpus windows.
- **Lasso:** consecutive layers only, unit-length columns, no centring, no intercept.
- **Gaps:** no layer-to-layer λ is given (the label regression uses 1e-5, CE), and no rule turns coefficients into a
  latent-rooted circuit. Our read-out rules, both labelled ours: R1 = the non-zero connected set, swept over λ; R2 =
  top-k parents per node.
- **Kept, at the lowest priority** (decision 5). Runs after the pod (needs its GPT-2 WCM circuits).

### Arena D1: attribution graphs (`specs/attribution-graphs.md`)

- **Code:** `decoderesearch/circuit-tracer@255d989` (the safety-research URL redirects there).
- **Substrate:** Gemma-2-2B with the per-layer transcoders (PLT), bf16 lazy loading, 6.1 GB (measured in 038). The
  2.5M-feature CLT does not fit 16 GB; the 426k CLT is an estimate only.
- **Root:** `attribute()` roots at the last-position logits only. That logit-rooted graph is the true row. A
  latent-rooted graph is **our adaptation** (038's `prune_rooted`; options A1 cut prompt, A2 no cap, A3 in the spec).
- **Pruning defaults disagree** across their README, code and Neuronpedia (0.8 / 0.98, 0.7 / 0.9, 0.8 / 0.95); each
  is recorded, Neuronpedia's is used for the fidelity check.
- **Fidelity:** Neuronpedia's Dallas → Austin graph (PLT, cap 5000, pruning 0.7 / 0.9): 697 feature nodes, 137 error
  nodes, 34,801 edges, p(Austin) = 0.450.
- **WCM side** needs a Gemma transcoder site (DAN-203). CLT targets are limited to about layer ≤ 12.

### Arena D2: Caples et al., on Gemma-2-2B (`specs/caples.md`; decision 4)

- **What is published:** a MATS report (LessWrong / AF, 2025-01-10) on Gemma-2-9B with four Gemma Scope 16k residual
  SAEs, fp32, ~43 GB. It does not fit 16 GB.
- **What we run (Daniel, 2026-11-01): the same method on Gemma-2-2B.** The authors' own code has a 2B path after the
  post (`interp_bottlenecks.py`, `scripts/sae_gap.sh`: `gemma-scope-2b-pt-res`, SAEs at layers 0/6/12/18/24 at L0 ≈
  100, lr 0.05). The row is labelled "Caples et al.'s method and code, Gemma-2-2B (an unpublished configuration of
  the authors')". Precision: bf16, or fp32 with two memory-only changes (SAE params frozen; reference forward under
  `no_grad`), each labelled.
- **Method:** mask σ(T·θ), T = 200^r, θ₀ = 1; loss |mean LD_SAE − mean LD_circuit| + λ Σ gates; Adam; keep θ > 0;
  per-position mean ablation **with the running-mean bug kept** as published (a fixed-mean row would be ours). The
  SAE error is dropped (WCM keeps it; stated in the write-up).
- **Root:** the task's logit difference only. A latent-rooted Caples row is our adaptation (and is our former
  "unweighted arm").
- **Tasks:** SVA on 2B (their code tasks need a model above 2B, per the post). Held-out: the SFC `rc_test.json` split,
  which they ship but do not use.
- **Fidelity:** only the authors' post-publication 2B saegap6 masks are available as anchors (47 → 0.968, 87 → 0.901,
  157 → 0.963, 370 → 0.972, 790 → 0.965 against the true model). The 9B checks F1–F7 do not apply.

## Third-party code (cloned 2026-11-01, Daniel's approval)

All under the repo-root `third_party/` folder, which is gitignored (never committed). Each in its own environment when
run.

| folder | repo | commit |
|---|---|---|
| `third_party/feature-circuits` | saprmarks/feature-circuits | `7fbd82b895ae16294f4e6fc7bfc675d1d680d659` |
| `third_party/feature-circuits/dictionary_learning` | saprmarks/dictionary_learning (declared as a submodule but not pinned at that commit, so cloned by hand) | `61ac634845bd76c839482f3b725ab3d898c8b277` |
| `third_party/Language-Model-SAEs` | OpenMOSS/Language-Model-SAEs (tag v0.1.0) | `67bfb0b9f835f772f3af2aa176745c6b6135373a` |
| `third_party/transcoder_circuits` | jacobdunefsky/transcoder_circuits | `8721ff06b64889e8cc50bb1c5e869c7f72b8303d` |
| `third_party/circuit-tracer` | decoderesearch/circuit-tracer | `255d989e93be1299d53d67b59e9fd76a0e274b14` |
| `third_party/ScalableSAECircuits` | NainaniJatinZ/ScalableSAECircuits (no licence; HEAD, which has the 2B path; the post-date commit is `00998a7`) | `f54a4efce908432b8771ea671a248ee30ec2b3af` |

Weights and data are fetched separately, with the revisions in each spec.

## Local GPU rules (Daniel, 2026-11-01)

- Runs happen **overnight only, at most 8 h a night**, on the 16 GB card. Nothing runs during the day.
- Every job is **resumable per circuit** (a finished circuit is never refitted) and is launched with a **hard stop
  time (07:30)**; a job killed at the stop loses at most the circuit in progress.
- Each arena starts with a short timing pilot before its overnight run.

## Decisions (Daniel, 2026-11-01 evening)

1. **Pod T7 off by default** (`TL_DAN159=0` in `082/chain.sh`; runbook budget ~136 GPU-h, ~$469). Done (DAN-189).
2. **Clone the official code** into a gitignored folder. Done (table above).
3. **Arena A split** into A1 (Ge) and A2 (Dunefsky).
4. **Caples on Gemma-2-2B**, locally, with the authors' 2B code path, labelled. "It doesn't have to be the same model,
   just the same method." No 9B pod session.
5. **CircuitLasso kept**, lowest priority, with our labelled read-out rule.
6. Size units: both counts (rule 6). SFC held-out: their test split plus a deduplicated split, labelled.

## Prior replications (literature check, 2026-11-01)

- **SFC:** official code. Kharlapenko et al. 2025 (arXiv 2504.13756) scaled it to Gemma-2B: nodes are (latent, token
  type); size chosen by node count; zero ablation on a held-out batch; attribution patching unreliable before layer 6.
- **Attribution patching:** Syed et al. 2023 report all top-k as a curve; KL breaks it as a metric; R² 0.27 against
  activation patching.
- **MIB** (arXiv 2504.13151, code `hannamw/MIB-circuit-track@b759df3`): area under the faithfulness curve over a
  fixed grid of graph fractions (CPR / CMD), linear-axis trapezoid. Our area is on a log latent-count grid, hence
  "MIB-style".

## What leaves the comparison

These were our adaptations presented as other papers' methods:
- **AtP / IG "natural size":** the position-union, keep-half cut is ours.
- **AtP / IG / SFC-ranking "at WCM's size":** size matched to WCM, on our Top-K substrate.
- **CircuitLasso "adapted":** per-position samples, all upstream sites jointly.
- **Ge "τ bisected to WCM's size":** on Top-K SAEs, gradients through nonlinearities, pooled over contexts.

They stay in the record (083, 098) as internal diagnostics, not as rows attributed to those papers. Consequences:
- the pod chain's T7 stage, which ran these rows, is off by default (decision 1);
- the tutor PDF's Table 1 and the paper's method-comparison appendix are rewritten from the arena results (DAN-199);
- `paper/main.tex:1560` must say that Caples drops the SAE error, and the 2026-09-28 audit §4's temperature schedule
  is wrong (T reaches 627–993, not ~340) (`specs/caples.md`).

## Order and estimates (estimates, not measurements)

| arena | build | overnight runs |
|---|---|---|
| C | ~2 days | ~1 night |
| A1 + A2 | ~3–4 days | ~2–3 nights |
| D1 + D2 | ~1 week | ~2–3 nights |
| B | ~1 day | ~2 nights, after the pod |

Order: C → A → D → B.

## Sources

- Marks et al., feature-circuits: https://github.com/saprmarks/feature-circuits
- Ge et al. 2024: https://arxiv.org/html/2405.13868v2; code https://github.com/OpenMOSS/Language-Model-SAEs (v0.1.0)
- Dunefsky et al. 2024: https://arxiv.org/abs/2406.11944; code https://github.com/jacobdunefsky/transcoder_circuits
- Yin et al. 2026: https://arxiv.org/html/2606.16939
- Ameisen et al. 2025: https://transformer-circuits.pub/2025/attribution-graphs/methods.html; code
  https://github.com/decoderesearch/circuit-tracer
- Caples et al. 2025 (LessWrong / AF); code https://github.com/NainaniJatinZ/ScalableSAECircuits
- Kharlapenko et al. 2025: https://arxiv.org/html/2504.13756
- Syed et al. 2023: https://arxiv.org/html/2310.10348
- MIB, Mueller et al. 2025: https://arxiv.org/html/2504.13151
