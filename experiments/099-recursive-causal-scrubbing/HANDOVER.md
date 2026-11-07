# WCM causal-scrubbing research handover

This document records the work performed in this chat, the interpretation agreed with the user, and what another agent needs to continue. It describes completed evidence; the next experiments below have not been run. Repository-relative paths are relative to:

```text
Windows: X:\Projects\AIs\Turing\Publication\3 Implementation\2
WSL:     /mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2
Experiment: experiments/099-recursive-causal-scrubbing/
```

## Start here

1. Read this handover and [the main evidence summary](out/summary.md).
2. Read [the panel pass/fail comparison](out/panel18/pass_groups.md) and the recursive reports for [development](out/recursive_development_diagnostics/report.md) and [check](out/recursive_check_diagnostics/report.md).
3. Inspect [the adapter](run_recursive_model.py), especially `build_graph()` and `selected()`, before changing the hypothesis.
4. Read [the original execution plan](../../agent-planning/causal-scrubbing-pilot-2026-10-28.md) for the wider intended protocol.

The central result is mixed: fresh WCM circuits often pass their production interventions but fail whole-donor background robustness. Actual recursion on two coarse residual-group hypotheses gives fitted-gain balanced firing agreement of **50.6% and 87.6%**. These evaluate different objects and must be reported separately. No recursive evaluation of every individual member in a complete WCM circuit has been completed.

All GPU jobs launched for this campaign finished. Completion is recorded in [finish_status.json](out/finish_status.json). No unfinished campaign job needs to be resumed. No manuscript changes were made by this work.

## User intent and research context

The user asked us to read Redwood's causal-scrubbing work thoroughly, assess the implications for WCM and its paper, compare related methods, and implement an initial experimental campaign. The user accepted the plan and explicitly authorized execution. Subsequent questions concerned whether production-passing circuits do better, what the approximately 88% result means, and whether the hypothesis requires all circuit members to activate in TopK.

The research question is: **which information carried by WCM-selected components remains sufficient under specified activation interchanges, and what missing dependencies explain failures?**

Keep separate:

- The original model's computation at natural feature strength.
- WCM's fitted weighted computation under its production interventions.
- An exact decomposition of the original model using fitted gains plus an explicit compensating remainder.

WCM's plausible contribution is joint sparse membership and positive-gain fitting through nonlinear model computation across specified ablations for an SAE target. Internal targets and masking alone are not a novelty claim. Existing historical headline figures, including the discussed approximately 46%, should not be treated as clean confirmation: older records had duplicate leakage. This pilot's clean 18-target evidence does not repair historical training or establish a bank-wide rate.

## Completed work

### 1. Literature and protocol review

Read and discussed the original [causal-scrubbing post](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), including footnotes/comments, and [technical appendix](https://www.lesswrong.com/posts/kcZZAsEjwrbczxN2i/causal-scrubbing-appendix).

The following is the conclusion of that earlier review, not a new literature search on the handover date:

| Work reviewed | Relevant evaluation and qualification |
|---|---|
| [Sparse Feature Circuits / SFC v3](https://arxiv.org/html/2403.19647v3) | Contrastive/zero IG discovery, held-out mean-ablation faithfulness/complement tests and downstream utility. No full recursive scrubbing was identified. Evaluation details, including exclusion of the first third of the circuit and SAE-error interventions, matter when comparing. |
| [CircuitLasso v2](https://arxiv.org/html/2606.16939v2) | Sparse observational regression, known-graph recovery, original-model SAE-node ablation, edge ablation inside a linear surrogate, and downstream utility. No full recursive scrubbing was identified. |
| [ACDC](https://arxiv.org/html/2304.14997) | Appendix N explicitly describes a restricted causal-scrubbing interpretation: important nodes admit no interchange; unimportant edges use a shared corrupted donor. |
| [Edge Pruning](https://arxiv.org/html/2406.16778) | Actual clean/corrupt edge replacement, without a detailed recursive semantic interpretation of the kind considered here. |
| [EAP-IG](https://arxiv.org/html/2403.17806) | Edge attribution and intervention-based faithfulness; distinguish this from a full recursive semantic hypothesis. |
| [Anthropic attribution graphs](https://www.transformer-circuits.pub/2025/attribution-graphs/methods.html) | Perturbation-prediction tests with acknowledged limits; not the full recursive algorithm evaluated here. |
| [MIB](https://arxiv.org/html/2504.13151) | Includes interchange-intervention causal-variable tests as well as circuit metrics. |

We corrected an overly broad suggestion that other state-of-the-art work does not address causal scrubbing. Do not claim universal superiority or that WCM is uniquely held to rigorous standards. Different tests constrain different hypotheses; there is no automatic universal hierarchy.

### 2. Clean whole-donor background benchmark

Frozen panel: 18 targets, two per depth-band/site-kind cell, nine development and nine check. Selection used existing smoke records and a seeded draw from an available scan, without using new scrubbing scores. This is an **availability-based pilot**, not an unbiased population sample. See [panel.json](panel.json) and the `targets_*.txt` lists.

The benchmark:

- Reused the clean record-building/verification machinery in experiment 094; constructed eight missing records.
- Independently checked all 18 records, including splits, duplicates, actual training count, firing and silence. All passed. The engine consumed 48 training examples.
- Fit fresh standard v2 WCM circuits for 400 steps. No historical fitted circuit was reused as clean evidence.
- Evaluated production scores, eight paired donor draws per background, natural-strength diagnostics, pinned-member diagnostics, empty controls and self donors.
- Used activating donors (`A`), held-out silent/contrast donors (`C`), and a random pool (`R`). The 512-context random pool excluded exact/near matches against panel records.
- Added a ten-integration-point adapted IG ranking from experiment 083, using an aggregate SFC-style ranking truncated to WCM's total node count, at natural strength.
- Recorded raw outputs, anchor statistics, post-TopK values, ReLU(pre-activation) diagnostics, false firing, contexts, seeds and provenance.

The existing patcher from experiment 073 replaces the unselected background while retaining live SAE errors and preserving its TopK fill budget. **This is whole-donor background resampling, not the general recursive algorithm.** Pinned members receive their clean natural activations and diagnose missing upstream dependencies; they do not establish an autonomous circuit.

Primary completed results, from [summary.json](out/summary.json):

| Method | Production passes | Donor-background passes | Median nodes | Median pooled A / C / R recovery |
|---|---:|---:|---:|---|
| WCM, fitted gains | 13/18 | 0/18 | 428.5 | 0.084 / 0.007 / 0.004 |
| Adapted IG, natural strength | 0/18 | 0/18 | 428.5 | -0.002 / 0.000 / 0.000 |

The IG comparison does not isolate selection quality: WCM receives gain fitting, IG does not. This is not an unchanged reproduction of published SFC, and no superiority over published SFC task results is established.

Recovery in the final summary pools donor outputs and their matched empty baselines **before dividing**:

```text
recovery = (mean_draws(output) - mean_draws(empty))
           / (natural_mean - mean_draws(empty))
```

Undefined/small denominators are retained as undefined, not silently counted as successful. Some earlier detailed tables use medians of per-draw ratios. These are different summaries; use `final_report.py` / `out/summary.*` for the pooled headline and label alternatives accurately.

The exact compatibility rules are in `final_report.py`:

- Production pass: each of `free0_tk`, `freeM_topk_tk`, `freeN_topk_tk` is defined and in `[0.8, 1.5]`, and `phi_sup_blind_tk >= 0.9`.
- Donor pass: production `free0_tk` in `[0.8, 1.5]`, necessity `phi_sup_blind_tk >= 0.9`, and pooled donor A and C recoveries each in `[0.8, 1.5]`.

These are continuity rules for this project, not universal causal-scrubbing acceptance thresholds.

### 3. Published-recursion reference engine and validation

Implemented [scrubbing.py](scrubbing.py), with explicit low-level graph `G`, interpretation graph `I`, and correspondence `c`:

- Validates graph structure, DAG order, inputs/outputs and correspondence/edge constraints.
- Uses exact empirical conditional classes for interpretation values.
- For important parents, samples a donor matching the relevant interpretation value of the **current local reference**, then recursively recomputes that parent.
- Uses one shared unconditional donor for all unimportant parents at a particular mapped node.
- Gives repeated occurrences along different paths independent recursive draws. Never memoizes stochastic activations globally by graph-node identity. Natural activations and interpretation classes can be cached.
- Logs paths, references, donors and support; supports explicit expanded-tree replay and exact distribution enumeration on tiny graphs.
- Preserves empirical multiplicity. Rejects NaN categories and tensor categories without an explicit categorical encoder.

Eleven engine tests and five adapter tests passed (16 total), with compile checks and real-model controls. Tests cover known correct/false hypotheses, exact probabilities, shared ancestors, shared unimportant donors, duplicate data mass, singleton supports, seeds, validation and gain decompositions.

[Toy results](out/toy_report.md): deliberately false hypotheses preserve unconditional means but have context-paired MSE `0.5`; a correct hypothesis has MSE `0`. Expanded-tree replay error was zero across the sampled checks. This is why the real-model graph includes paired firing-disagreement loss rather than treating raw mean preservation as sufficient evidence.

### 4. Actual recursion on a coarse original-model hypothesis

The adapter in [run_recursive_model.py](run_recursive_model.py) executes the full recursion **for its declared coarse graph**. It does not execute a hypothesis for every individual WCM member.

Current applicability: layer-2 residual targets only. The adapter exposes WCM-selected groups at residual cuts after transformer blocks 0 and 1. Embedding/block 0, block 1, and block 2/target readout are native composite computations.

Low-level graph, schematically:

```text
x -> raw0 -> members0 ---+-> state0 -> raw1 -> members1 ---+-> state1 -> prediction
          -> remainder0 -+                 -> remainder1 -+
x ------------------------------------------------------------------> loss
prediction ---------------------------------------------------------> loss
```

At each cut:

```text
members = phi_alpha(raw_residual)
remainder = raw_residual.float() - members
state = (members + remainder).cast_to_model_dtype()
```

`phi_alpha` selects currently active post-TopK SAE values for WCM's fixed membership, multiplies them by either 1 or frozen fitted gains, and decodes their contribution. Thus both arms are exact decompositions of the **original model**. Fitted gains are compensated in the remainder. Inserting gains while dropping this compensation would define a different model.

All other residual information, including SAE reconstruction error and gain compensation, is explicitly in the remainder and is resampled as unimportant. This error treatment differs from the background benchmark's live-error convention.

#### Exact interpretation and TopK assumption

Define `b(x) = 1` iff the clean target SAE feature fires after TopK at a fixed read position, otherwise `0`.

At each important intermediate interpretation node, the high-level value is this same clean teacher bit `b(x)` (passed through the interpretation graph). Remainder nodes are omitted from `I`. The hypothesis is:

> The selected residual groups at cuts 0 and 1 transmit the target-firing state, while within-class differences and unselected remainder information can be interchanged without changing that state.

Important-path donors therefore match the teacher firing class. Remainder donors are unconditional and can come from either class. Each recursive call uses its own local reference, following the sampler's semantics.

**We do not assume all WCM members activate, or enter TopK simultaneously.** A selected feature contributes only where it is in the current SAE's TopK; otherwise it contributes zero. Different examples can use different subsets and magnitudes. Gains are applied after encoding and do not activate an inactive feature. No donor condition matches the entire member pattern or all member amplitudes. The mixed decoded residual is not separately projected through another TopK operation at the cut; TopK is used when encoding contributions and reading the target.

This is teacher-label-conditioned representational invariance. The matching bit is obtained from the clean target's future output; this is not a blind predictor, a human semantic annotation, a per-member algorithmic explanation, or an AND condition over members. A high score supports only this specified invariance on these donors.

#### Data, controls and results

The fixed position is selected from training examples by maximum target-firing count, breaking ties at the lowest position. Every held-out context is read at that common position; no donor-specific anchor realignment is used.

| Setting | Development | Check |
|---|---|---|
| Target | `2.resid.7349` | `2.resid.29259` |
| Position | 28 | 6 |
| Selected group sizes, cuts 0 / 1 | 50 / 39 | 37 / 25 |
| Full WCM circuit nodes | 265 | 160 |
| Held-out references and donor dataset | 32 | 32 |
| Firing / silent contexts | 3 / 29 | 4 / 28 |
| Recursive draws per reference per arm | 32 | 32 |

Every one of the 32 references was evaluated in the final runs. Donors are sampled with replacement from the finite empirical dataset; self donors are allowed. Non-input conditional support and self-draw fractions are reported. Input-ID matching is necessarily singleton and should not be confused with informative intermediate matching.

| Arm | Development balanced firing agreement | Check balanced firing agreement |
|---|---:|---:|
| Full residual state | 100.0% | 100.0% |
| Empty groups | 48.8% | 49.2% |
| WCM membership, natural strength (`wcm_a1`) | 54.3% | 74.4% |
| WCM membership, fitted-gain decomposition (`wcm_fit`) | 50.6% | 87.6% |
| Active random groups, natural strength | 48.8% | 49.2% |
| First cut full, second WCM at natural strength | 62.0% | 83.7% |
| Second cut full, first WCM at natural strength | 65.8% | 83.6% |
| Full SAE reconstruction at both cuts | 95.3% | 95.4% |

Active random controls match per-cut group sizes and are drawn from features active on training examples, excluding the WCM members. They have no fitted gains. Full-SAE reconstruction retains encoded state, while its reconstruction-error remainder remains resampled.

For each arm, unintervened graph predictions exactly matched an independent full inference path on all 32 contexts: max absolute error `0`, with no changed firing labels. Full-state controls were 100%. Expanded-tree replay matched the first draw for every reference and arm.

Balanced firing agreement gives each class equal weight:

```text
0.5 * (P(scrubbed fires | clean fires) + P(scrubbed silent | clean silent))
```

The user's approximately **88%** refers to the check target's fitted arm: firing-reference agreement `97/128 = 75.8%`, silent-reference agreement `890/896 = 99.3%`, whose equal-weight average is `87.6%`. It is neither the fraction of circuits passing nor the percentage of a mechanism explained.

Intervals in the reports bootstrap references within firing classes, not individual donor trials. There are only three/four positive contexts. These intervals describe a tiny fixed selected dataset and do not quantify generalization to arbitrary targets or a new donor population. The cut-restoration/full-SAE additions were exploratory diagnostics chosen after inspecting initial development outcomes. Both recursive targets have now been inspected and are no longer untouched checks for future tuning.

### 5. Diagnostics and the production pass/fail question

[Read diagnostics](out/panel18/read_diagnostics.md) retain ReLU(pre-activation) separately from actual post-TopK activation. Much of the contrast-donor failure includes loss of encoder signal; it cannot all be attributed to TopK competition.

[Numerical round-trip controls](out/identity_roundtrip.json) gave all-kept relative anchor RMSE of about `0.284%`, `0.547%`, and `0.275%` on `2.resid.7349`, `6.mlp.209`, and `10.attn.39376`. Arbitrary-position maxima can be larger, including TopK discontinuities. Raw deviations are preserved. This drift is much smaller than the observed donor-background loss; do not claim every background self donor is exactly identical.

The post-hoc [passing/failing comparison](out/panel18/pass_groups.md), with pooled recoveries and medians across targets:

| Measurement | Production passing, n=13 | Production failing, n=5 |
|---|---:|---:|
| Live recovery, A | 0.0815 | 0.0863 |
| Live recovery, C | 0.0079 | 0.0067 |
| Live recovery, R | 0.0027 | 0.0051 |
| Pinned recovery, A | 0.5315 | -0.0119 |
| Pinned recovery, C | 0.3115 | 0.0091 |
| Pinned recovery, R | 0.3587 | 0.0224 |
| Donor passes | 0/13 | 0/5 |
| Median nodes | 451 | 350 |

Production passes show no clear live-background advantage here, but show higher pinned recovery. This is consistent with selected information being useful while dependencies needed to regenerate it are missing. It does not prove that diagnosis. Groups are small and confounded by size/site/target; all six residual targets are production passes. Both recursive targets also pass production, so there is no recursive production-failing comparator. No new fitting/GPU experiment was needed for this analysis.

## Code and artifact map

All paths in this table are inside this experiment unless otherwise indicated.

| Path | Purpose |
|---|---|
| `README.md` | Experiment overview; contains historical initial-run sections as well as later results. Use the final summary for current headline numbers. |
| `scrubbing.py`, `test_scrubbing.py`, `test_adapter.py` | Generic engine and 16 CPU tests. |
| `toy_report.py`, `out/toy_report.*` | Exact toy examples and replay evidence. |
| `prepare_panel.py`, `panel.json`, `targets_*.txt` | Frozen panel selection and split. Do not regenerate to silently change the evaluated panel. |
| `run_baseline.py` | Audit, fresh WCM fits, background interventions, optional IG/separately fit unweighted support. |
| `run_panel.py` | Build missing 094 records and run the panel with one resident model. |
| `run_recursive_model.py` | Two-cut graph adapter and actual recursive sampling. |
| `finish_diagnostics.py` | Completed numerical/recursive follow-up launcher; hardcodes original output directories. |
| `wait_and_finish.py` | Completed one-off dependency-wait helper, not an automation or work to restart. |
| `report.py`, `plot_results.py`, `diagnose_reads.py` | Panel reporting, PNG/PDF and pre/post diagnostics. |
| `report_recursive.py` | Recursive metrics, reference bootstrap and coverage reporting. |
| `final_report.py`, `compare_pass_groups.py` | Pooled final summary and post-hoc production groups. |
| `fingerprint_weights.py`, `out/weight_fingerprints.json` | SHA-256 of the model checkpoint and 36 SAE checkpoints: 37 files, about 15.1 GB total. |
| `out/summary.md`, `out/summary.json` | Completed campaign headline. |
| `out/panel18/targets/*.json` | Raw production/background metrics and donor outcomes per target. |
| `out/panel18/circuits/*.wcm.pt` | Fresh fitted circuits consumed by recursive runs. |
| `out/panel18/audit.json`, `manifest.json`, `source_check.json` | Record checks, source/config/context fingerprints and during-run integrity. |
| `out/panel18/train.jsonl` | Fit records. |
| `out/panel18/background_recovery.png`, `.pdf` | Standalone figures, visually inspected. |
| `out/records/ctx_v2/*.pt`, `out/records/build/` | Clean records and build log. |
| `out/recursive_development_diagnostics/`, `out/recursive_check_diagnostics/` | Final 32-reference/32-draw runs: setup, manifest, results, reports, source snapshots. |
| `out/first3/`, `out/recursive_two_cut/` | Initial harness runs; retain as historical evidence. |

The initial recursive run in `out/recursive_two_cut` used seven references and 16 draws; its natural/fitted agreements were 53.1%/47.1%. These are superseded for headline purposes by the full development follow-up's 54.3%/50.6%. Do not accidentally mix them.

## Provenance and workspace cautions

- Exact executed sources are retained in run `source_snapshot/` directories. The panel has 313 source files. Manifests also fingerprint contexts/configuration; checkpoint content hashes are in `out/weight_fingerprints.json`.
- The panel's during-run source check passed. After execution, `run_panel.py` was changed to accept `--out` and `--records` so reproductions can use fresh paths. This is documented in `out/panel18/post_run_edits.json`; the executed launcher is preserved in its snapshot. Do not overwrite the original provenance to match the current source.
- New/changed experimental settings should use a fresh output directory. Manifest mismatch guards deliberately reject incompatible cache reuse.
- `run_recursive_model.py` still defaults to the **initial** first-three run, old smoke records, 16 draws and four references per class. Pass the explicit arguments below for a full-panel follow-up.
- `run_panel.py --records` changes the record destination, but its build log directory remains `out/records/build`. For this panel, reuse already audited records. If fully isolating new record construction, inspect/fix that launcher path first and record the change.
- `finish_diagnostics.py`, `final_report.py`, `compare_pass_groups.py`, `toy_report.py` and some helpers write fixed original output locations. Do not rerun casually if preserving original artifacts; parameterize/copy intentionally for new work.
- Many unrelated files were already modified/untracked, including production code and manuscript files. This chat's campaign work is in experiment 099 and `agent-planning/causal-scrubbing-pilot-2026-10-28.md`. The separate `agent-planning/causal-scrubbing-research-plan-2026-10-28.md` was present/read, not authored as part of our execution. Do not reset or overwrite others' work.
- Use `git status --short --ignore-submodules=all`; ordinary status can hit a permissions issue in `external/circuit-tracer`.
- User instruction: **never recursive forced delete**. No such deletion was used.

## Environment and reproduction

No dependency installation was needed. CPU checks used Windows Python 3.11 / torch 2.9.1 CPU. GPU campaign manifest: WSL Python 3.12.3, torch `2.8.0+cu128`, CUDA 12.8, NVIDIA RTX 5070 Ti (16 GB), repository `.venv`. Model: TuringLLM, with checkpoint `models/TuringLLM/model_1722550239_03986.pt` and the 36 per-layer/type SAE checkpoints under `models/TuringLLM/SAE/`. Inspect saved config/manifests before assuming the current production config still matches.

CPU tests, from repository root:

```powershell
python experiments/099-recursive-causal-scrubbing/test_scrubbing.py
python experiments/099-recursive-causal-scrubbing/test_adapter.py
```

GPU examples below are **bash commands inside WSL**, from the WSL repository root. They use fresh output names. These commands are documented here, not newly executed for this handover. WSL launch required escalation in this desktop session; an existing `wsl --exec bash` approval was used. Follow the actual tool permission requirements when continuing.

Reproduce the clean panel using already audited records:

```bash
cd '/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2'
EXP=experiments/099-recursive-causal-scrubbing
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$EXP/run_panel.py" --out "$EXP/out/panel18-reproduction" --records "$EXP/out/records/ctx_v2"
```

Reproduce the full development recursive settings using the original fresh fitted panel:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u "$EXP/run_recursive_model.py" --baseline "$EXP/out/panel18" --contexts "$EXP/out/records/ctx_v2" --target 2.resid.7349 --out "$EXP/out/recursive-development-reproduction" --draws 32 --max-base-per-class 32 --diagnostics
.venv/bin/python "$EXP/report_recursive.py" --run "$EXP/out/recursive-development-reproduction"
```

For the existing check reproduction, change the target to `2.resid.29259` and use another fresh output directory. Reproducing it does not make it an untouched methodological check again. To evaluate newly fitted circuits, change `--baseline` to their run and keep context hashes consistent.

Reporting from a fresh panel, using an environment with NumPy/matplotlib:

```bash
.venv/bin/python "$EXP/report.py" --run "$EXP/out/panel18-reproduction"
.venv/bin/python "$EXP/plot_results.py" --run "$EXP/out/panel18-reproduction"
.venv/bin/python "$EXP/diagnose_reads.py" --run "$EXP/out/panel18-reproduction" --contexts "$EXP/out/records/ctx_v2"
```

The historical source snapshot is the authority for an exact old-code reproduction. Current-source reruns can differ if shared production code/configuration has subsequently changed; compare manifests rather than assuming equality. Do not copy snapshots over the user's working tree.

## Relevant existing experiments and known limitations

| Existing code | How it relates |
|---|---|
| `experiments/073-strict-eval/strict_eval.py`, `diag.py` | Background patcher and live/pinned/self/pre/post controls reused here; not the generic recursion. |
| `experiments/094-tl-v2-records/` | Clean records and independent verification. Initial inspection found local smoke/dry runs, not a completed local 8k production confirmation. The clean scope is its exact/anchor-prefix/near rules, not a source-document split. |
| `experiments/086-heldout-rebuild/out/scan_smoke40/scan_top512.pt` | Available scan frame used for panel/record work. |
| `experiments/083-v1-comparison/` | Adapted IG machinery; support-restricted fitting also exists but was not the IG arm's gain-matching protocol here. |
| `experiments/098-circuitlasso-baseline/` | Adapted Lasso, with departures from published CircuitLasso documented there. Not run in this pilot. |
| `experiments/095-v2-resample-appendix/` | Historical v2/v2CR panel with known duplicate leakage. Development only, not clean confirmation. Earlier reviewed rates were standard 50/72 production and 0/72 resample; adding resample training 45/72 and 17/72. Do not substitute these into clean headlines. |
| `experiments/077-antiquity-scrub/antiquity_scrub.py` | Existing semantic/group swaps for antiquity; not executed in this campaign and not full recursion. Candidate case `9.resid.20419` is preselected, not population evidence. |
| `src/circuit/instrument/sae_graph.py` | Attribution tensors/gradients, not an independently executable causal-scrubbing `G/I/c` graph. |
| `experiments/023-recursive-map/` | Recursive driver selection, not recursive activation scrubbing. |

Not completed: a separately discovered unweighted baseline (the CLI supports it), gain-capacity-matched IG, adapted Lasso in this campaign, a fresh clean resample-trained WCM comparison, the antiquity semantic hypothesis, all-individual-node WCM recursion, or a new large population evaluation. Resetting WCM gains to one is not discovering an unweighted circuit.

## Recommended continuation

The evidence points first to a **support-size versus recursive robustness experiment**. Full encoded state performs much better than the small groups, and pinning helps more than live recomputation. These observations motivate testing missing dependencies; they do not yet prove that more members alone will solve the problem.

1. **Freeze a new development/check protocol before new method tuning.** Both existing recursive targets and the panel results have been inspected. Continue using them for development, but select and lock fresh check targets/contexts for subsequent confirmation. Preserve separate training, interpretation-development and evaluation data. Increase positive donor coverage; three/four firing contexts is too little for broad conclusions.
2. **Measure a small size frontier on development targets.** Keep the exact original-model graph, read-position rule, donor classes and error treatment fixed. Compare WCM support sizes obtained through training-only rankings or weaker-sparsity fits. Never choose additions using held-out recursive outcomes. Include natural/fitted arms, equal-size active random controls, full-state and full-SAE controls. Record total WCM size and per-cut size separately.
3. **Distinguish support loss from amplitude/encoder loss.** Retain paired firing and continuous activation errors, pre/post reads, per-class outcomes, cut restorations and donor support. Do not rely on a single mean recovery ratio or balanced accuracy alone. Compare random/IG supports fairly when attributing gains to membership selection or coefficient fitting.
4. **Then test a focused training modification at matched budget.** Candidate: intermediate-state reconstruction or recursive/resampled activation training, with gain regularization/bounds if justified. Keep standard WCM as the unchanged comparator. This is a proposed experiment, not implemented work. Recursive training could be expensive; preserve path semantics if optimizing execution.
5. **Confirm once on the new frozen check set.** Keep failures/exclusions visible. Report a size/robustness trade-off if compact groups fail but larger groups succeed. If training or interpretation changes after a check, another untouched check is needed.
6. **Expand the hypothesis only with an explicit graph.** A later semantic antiquity case needs meaningful variables, dependencies, correspondence and token alignment; exact teacher firing labels do not provide this automatically. All-node recursion needs a faithful executable decomposition of bypass/error paths and potentially large path-expansion cost. Do not silently promote the coarse result into a full-circuit claim.

A successful next report should say exactly which graph, interpretation, donor distribution, gain/remainder convention and size were tested, and distinguish development evidence from confirmation. The current defensible conclusion is that WCM's production faithfulness does not ensure the tested interchange invariances; compact selected groups can carry substantial state information in one case, while another case remains close to the empty control.
