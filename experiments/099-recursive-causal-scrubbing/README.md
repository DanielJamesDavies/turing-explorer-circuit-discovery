# 099: clean support robustness and a recursive-scrubbing pilot

Follows `agent-planning/causal-scrubbing-pilot-2026-10-28.md`.
Production discovery code and manuscript are not modified by this experiment.

There are two different experiments here:

1. **Whole-donor background robustness**, using the production WCM intervention and scorer on fresh records.
2. **Actual recursive scrubbing of a coarse two-cut hypothesis**, using an independently tested finite-data engine.

Do not merge their scores or describe the background benchmark as full recursive scrubbing.

## Files and outputs

| File | Purpose |
|---|---|
| `scrubbing.py` | Explicit G/I/c graphs, conditional recursive sampling, shared unimportant donors per node, independent path occurrences, expanded-tree replay, exact tiny-distribution enumeration |
| `test_scrubbing.py` | Graph, correspondence, sampler, grouping, shared-ancestor, replay and false-hypothesis tests |
| `test_adapter.py` | Exact gain/remainder decomposition, full-state/empty controls, and metric edge cases |
| `toy_report.py` | Exact finite examples exposing why unchanged raw means can conceal incorrect hypotheses |
| `run_baseline.py` | Audits records, fresh 400-step fits, production scores, paired donor backgrounds and pinning diagnostics |
| `prepare_panel.py`, `panel.json` | Frozen availability-based balanced 18-target panel; no new scores used for selection |
| `run_panel.py` | Builds missing records through 094 and runs WCM plus adapted IG, reusing one model load |
| `run_recursive_model.py` | Original-model two-cut residual adapter, both natural-strength and fitted-gain exact decompositions |
| `report.py`, `report_recursive.py`, `plot_results.py` | Reports and standalone figures from saved artifacts |
| `fingerprint_weights.py` | SHA-256 content fingerprints of the checkpoint and all SAE checkpoint files |
| `out/first3/` | Completed first-three-target WCM diagnostic, raw outputs, audit, manifest and matching source snapshot |
| `out/records/` | Clean copied/newly constructed records for the 18-target panel |
| `out/panel18/` | Frozen-panel comparison artifacts; see report for actual completion count |
| `out/recursive_two_cut/` | Completed initial recursive development case, source snapshot, cases and donor coverage |

## Initial three-target result

The full campaign is now complete. The evidence summary is `out/summary.md` / `out/summary.json`.
All 18 clean records passed the audit. WCM passed the production rule on 13/18, but 0/18 passed the
pooled donor-background rule; matched-size adapted IG passed neither rule on any target. Median WCM size was
428.5 nodes. This comparison includes fitted gains only for WCM and does not isolate selection quality.

Actual recursive follow-ups evaluated every one of 32 held-out references, with 32 draws per reference:

| Target | WCM natural strength | WCM fitted-gain decomposition | Active random | Full SAE reconstruction | Full residual state |
|---|---:|---:|---:|---:|---:|
| Development: 2.resid.7349 | 54.3% | 50.6% | 48.8% | 95.3% | 100% |
| Check: 2.resid.29259 | 74.4% | 87.6% | 49.2% | 95.4% | 100% |

Entries are balanced firing agreement, not binary pass rates. The same coarse two-cut interpretation is tested
on both targets; these are not full-circuit semantic explanations. Keeping substantially more encoded state
improves this test considerably, while fitted gains help one selected group and hurt the other.
Cut-restoration diagnostics and uncertainty are in `out/recursive_{development,check}_diagnostics/report.md`.
Ordinary all-kept SAE round trips have 0.27-0.55% relative anchor RMSE on the three numerical-control targets;
these are much smaller than the observed donor-background losses.

All three fresh records independently passed structural, duplicate, actual-training-count, firing and silence checks.

| Target | Nodes | Production pass | Median recovery under activating / contrast / random donors |
|---|---:|---|---|
| 2.resid.7349 | 265 | yes | .043 / .024 / .000 |
| 6.mlp.209 | 495 | yes | .000 / .000 / .000 |
| 10.attn.39376 | 1,069 | no (activating-mean denominator undefined) | .060 / .000 / .000 |

The fits use 400 steps and eight draws per donor background. Donor recovery is a ratio of held-out anchor means,
with the empty circuit computed under the same donor draw. Pinned-member results are higher in many cases but
remain incomplete. These are selected pilot targets, not a population pass-rate estimate.

The background patcher retains live SAE errors, as does the production scorer. Self-donor tests have small
anchor errors but can have larger maximum errors at arbitrary positions because of bf16 and Top-K discontinuities.
Raw deviations are retained rather than claiming exact numerical identity. The separately constructed recursive
adapter below uses a cancellation-preserving decomposition and passed its original-model equality checks exactly.

## What the real recursive experiment means

The first target is `2.resid.7349`. At layer-0 and layer-1 residual boundaries, the selected WCM groups contain
50 and 39 latents. The remaining native transformer computations are explicit composite nodes.

The hypothesis is that each selected residual group transmits the binary target-firing state. That state is
defined by clean-model firing, not a human semantic annotation. The common read position, 28 in the initial run,
was selected using training examples only. All donors are the 32 held-out strongest/contrast examples at that same
position, removing anchor misalignment. Three donors fire and 29 are silent. Seven references (three firing, four
silent) have 16 recursive draws each; draws are not independent input examples.

For a residual r, selected decoded contribution phi_alpha(r) and remainder r-phi_alpha(r) sum back to the original
residual. Both coefficient conventions are therefore exact graph rewrites of the original model. In the fitted
case, the remainder explicitly includes the compensating gain term. Every unintervened graph agreed exactly with
the independent full inference path and preserved all firing labels.

The published recursive sampler is run on this graph, including independent path draws and recursive recomputation.
Unimportant remainder subtrees are resampled, including SAE reconstruction errors. This is a different error-path
hypothesis from the live-error background benchmark.

| Arm | Balanced firing agreement |
|---|---:|
| Full residual state | 100.0% |
| WCM residual groups, natural strength | 53.1% |
| WCM residual groups, fitted gains with compensating remainder | 47.1% |
| Active random groups at natural strength | 46.1% |
| Empty groups | 46.1% |

This rejects a strong coarse transmission hypothesis in a development case. It is not a score for the complete
265-node WCM circuit, nor evidence that all hypotheses about this target must fail. See `out/recursive_two_cut/report.md`
for context-level uncertainty and per-class errors. Full residual states are the positive control; their success
shows that the test can preserve the state with an adequate representation.

## Reproduction

From repository root, CPU checks (Windows Python or WSL):

```text
python experiments/099-recursive-causal-scrubbing/test_scrubbing.py
python experiments/099-recursive-causal-scrubbing/test_adapter.py
python experiments/099-recursive-causal-scrubbing/toy_report.py
```

GPU commands use the existing WSL environment, TuringLLM weights and artifact stores:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/099-recursive-causal-scrubbing/run_baseline.py --out experiments/099-recursive-causal-scrubbing/out/first3-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/099-recursive-causal-scrubbing/run_panel.py --out experiments/099-recursive-causal-scrubbing/out/panel-reproduction
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/099-recursive-causal-scrubbing/run_recursive_model.py --out experiments/099-recursive-causal-scrubbing/out/recursive-reproduction
```

Current code has additional diagnostics and guards relative to the original three-target run. Its exact source
files are retained in `out/first3/source_snapshot/` and `out/panel18/source_snapshot/`; recursive source files are retained in
their run's `source_snapshot/`. A changed source/context/config manifest refuses cache reuse. Use a new
output directory for altered experiments, rather than overwriting evidence.

The IG arm is the documented latent-target adaptation from 083, using ten integration points and the aggregate
SFC-style ranking, truncated to the WCM node count. It is at natural strength and does not receive WCM's gain-fitting
capacity; this comparison does not isolate node selection from coefficient fitting. Published SFC is not rerun
unchanged, and no superiority over its published task-level results is claimed.

The panel is balanced by depth/site and constrained by available scan artifacts, with nine development targets
and nine checks. It is not an unbiased sample of the entire SAE bank. No new production/scrubbing outcome was used
to pick it. Any later modification chosen using its check results requires a fresh confirmation set.

## Recommended next experiment

On development targets, measure a small support-size frontier at the residual cuts, using only training-derived
rankings or newly fitted weaker-sparsity WCM supports. Preserve natural-strength and fitted-strength results
separately. Then test an intermediate-state reconstruction/resampling objective at matched size and budget.
The full-SAE controls show that there is recoverable signal beyond the small groups; whether it can be captured
compactly remains open. Both existing recursive targets have now been inspected, so further method tuning needs
a new check set. The semantic antiquity case and recursion over every individual WCM node remain later work.
