# 107 arena C: SFC (official code) vs WCM on Pythia-70m-deduped + Marks SAEs

Linear: DAN-191 (SFC setup), DAN-194 (WCM adapter), DAN-196 (runs). Spec: `../specs/sfc.md`.

## SFC setup

- **Code:** `third_party/feature-circuits@7fbd82b` (main HEAD) with `dictionary_learning@61ac634`. Both are untouched
  clones; every change we make lives in this folder and is labelled.
- **Dictionaries:** the Marks et al. Pythia-70m-deduped set (`10_32768`, 19 sites) from exp 030, linked into the clone
  at `dictionaries/pythia-70m-deduped`.
- **Environment:** `setup_sfc_env.sh` builds `~/venvs/sfc107` in WSL. Exact versions are in `env_freeze.txt`.
  - nnsight 0.3.7, numpy 1.26.4, transformers 4.47.1 (Dec 2024, the nnsight 0.3.7 era), dictionary_learning
    @61ac634.
  - **torch 2.8.0+cu128 instead of < 2.6 (OURS-5).** The RTX 5070 Ti (Blackwell) needs torch ≥ 2.7. The pin only
    guarded `t.load`'s `weights_only` default; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` restores it.
  - Without explicit pins, pip resolves 2026 releases: numpy 2.5, transformers 5.22, scipy 1.18. Those don't import
    with nnsight 0.3.7, so the HF and scientific stacks are pinned to late 2024.
- **Runner:** `run_sfc.sh`.
  - Nodes-only discovery for each task: `circuit.py --num_examples 100 --batch_size 10 --aggregation none
    --nodes_only`, features and neurons.
  - Then their evaluation, `sfc_faithfulness.py`: the faithfulness notebook transcribed, with the labelled edits
    OURS-1..5 in its docstring.
  - Resumable: circuit files and evaluation results are cached. Hard stop at `STOP_AT` (07:30).
  - circuit.py always plots after saving, and its per-position plotter fails on a nodes-only circuit. The runner
    accepts exactly that failure when the circuit file exists.
- **Issue #15 is real here:** their `run_with_ablations` raises the TypeError under nnsight 0.3.7. Our copy (OURS-2)
  changes one shape read. Every summary records `fix15_used: true`.

## Fidelity checks (`specs/sfc.md` §10)

| check | result | verdict |
|---|---|---|
| F2 splice exactness (every node and error kept) | faithfulness 1.0000 and completeness 0.0000 on all four tasks | pass |
| F3 across-RC at T_N = 0.1 | faithfulness **0.177** (paper 0.21 ± 0.05) | faithfulness passes |
| F3 node count | 41 at layers ≥ 2 (what the evaluator counts), 52 over all 19 sites; 38 distinct latents + 14 error nodes; paper Fig. 7 has 86 | count off; open (see below) |
| F4 Fig. 3, features | the mean curve reaches 0.5 at **92 nodes** (paper about 100); 0.53 at 100, 0.69 at 200 | pass |
| F4 Fig. 3, neurons | 0.5 at **1,899 nodes** (paper about 1,500; window 750–3,000) | pass |
| F4 completeness, features | about 0.000 at 300 nodes | pass |
| F4 without error nodes | mean faithfulness ≤ 0.08 up to 200 nodes; never reaches 0.5 (matches the paper's finding that errors matter) | qualitative pass |
| F6 determinism (two batch-10 runs) | max \|ΔIE\| = 0.0 | pass |
| F8 evaluation/discovery overlap | see below | measured |
| F9 memory | discovery ≤ 1.7 GB host RSS, about 9 s (features) and 13 s (neurons) per task; evaluation peak 3.15 GB GPU, about 1 s per threshold | pass |
| F10 batch 10 vs batch 1 | max \|ΔIE\| **0.031** (an error node at resid_2: 0.061 vs 0.030); 8,928 of 3.7M entries differ by more than 1e-4, 638 by more than 1e-3; at T_N = 0.1 the node set changes by **1 node** (52 vs 51) | fails the 1e-4 tolerance; small and local, recorded as a property of their code at this SHA |
| F1 dictionary stats, F5 IG vs exact, F7 error-node edges, F11 paper-era code | not run yet | to do |

**The F3 count.** Faithfulness matches, but neither count reaches the paper's 86 (±25% would be 65–108). Fig. 7 is
drawn from the full circuit with edges, at node / edge thresholds 0.1 / 0.01. What it counts is not stated. The spec's
F11 (nodes-only `rc_train` at the paper-era commit `64d1cb3`) is the next check. Until then, the 86 is not reproduced.

**The full circuit with edges** (`get_circuit.sh` settings: node 0.1, edge 0.01, aggregation none, rc_train) ran in
2 min 22 s with 2.0 GB host RAM, so HEAD's sparse-storage fix holds. Daniel's 2024 out-of-memory does not recur. The
figure (`out/sfc/circuits_edges/figures/`) has the same structure as the paper's Fig. 7:
- plural-noun / plural-subject features on the subject;
- relative-pronoun features on "that";
- verb-form features ("promotes plural / singular verb forms") at the final position feeding y.
It has the same ~52 nodes as the nodes-only count, not 86.

**F8, held-out overlap:** of the 40 evaluation examples (first 40 shortest of `<task>_test.json`), how many are
discovery examples, and how many appear anywhere in `<task>_train.json`, by `clean_prefix`:

| task | in the 100 discovery examples | in the train file |
|---|---:|---:|
| rc | 0 | 40 |
| nounpp | 1 | 18 |
| simple | **40** | 40 |
| within_rc | 5 | 40 |

So SFC's "test" evaluation of `simple` reuses its discovery prompts exactly, and for rc and within_rc every test prompt
is in the train file. Every arena C row (SFC and WCM) is therefore reported on their test split **and** on a
deduplicated held-out split (decision 8 in the 107 README).

## Outputs

`out/sfc/` (gitignored):
- `circuits/*_nodeall.pt`, the per-position node effects, features and neurons, four tasks;
- `faithfulness/summary_*.json`, the curves;
- `*_F4.json`, the Fig. 3 readings;
- `logs/`.

Checks: `sfc_checks.py` (count, diff, overlap) and `sfc_curves.py` (F4).

## First comparison (DAN-196)

**Rows.** Data mode `sfc` throughout: SFC's 100 discovery / 40 evaluation examples (the leak above applies to both
methods).
- **WCM:** `out/wcm_tasks_zacr`, behavioural WCM = Z + A + C + R (no rank-keep). 18 sites (embed out), 400 steps,
  λ ∈ {3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3}, A / C means pooled and per_position. 48 circuits.
- **SFC:** their nodes-only circuits at their 15 thresholds `logspace(-4, 0, 15)`.

**Scoring, each method under both evaluations:**
- `score_sfc_ours.py`: SFC under ours → `out/cross/sfc_ours.jsonl`;
- `score_wcm_sfc.py`, SFC env: WCM under theirs → `out/cross/wcm_sfc.jsonl`;
- `donor_tasks.py`: donor recovery;
- `curves.py`: figures `out/figs/curves_<task>.{png,pdf}`, areas `out/cross/summary.json`.

**Labelled choices (ours):**
1. **SFC → our tests:** members = the position-collapsed union of feature nodes (|IE| > T_N at any position) at the
   18 sites, α = 1. Embed nodes are dropped. Error nodes are unused, because our scorer always keeps errors. SFC's
   native counts are recorded beside ours.
2. **WCM → SFC's evaluator:** "WCM membership at α = 1 under SFC's evaluator":
   - act = True at every position for every member;
   - errors **kept** (resc all True, WCM's semantics; SFC counts them as nodes, 12·T) or **removed**
     (`features_wo_errs`);
   - everything else is SFC's (start_layer 2, mean over the 40 eval patch prompts, 40 examples).
3. **Size unit:** distinct latents over the 18 sites. SFC's native unit (layers ≥ 2, features + errors) is the second
   row of each figure.
4. **"Pass" under SFC's evaluator:** faithfulness in [0.75, 1.5] (SFC defines no pass rule).
5. **Curve value under our evaluation:** min(Z, A, C). Our pass = `passes_task` and not vacuous. Resample pass adds
   donor A and C recovery in [0.75, 1.5].
6. **Area:** MIB-style nAUC on the 1-2-5 log grid 10..1e5 (13 points), reading by the ≤ k rule, 0 below a method's
   smallest circuit (specs/evaluation.md §3.3-3.4).

**Tests:**
- `test_cross.py`: SFC's evaluator on the WCM "everything" circuit gives faithfulness 1.000004 / 0.999915 / 0.999953 /
  0.999974; its F(M) and F(∅) equal their own get_fcs run.
- **SFC at T_N = 0 under our Z is 0.709, not 1.** SFC's IE is exactly 0 wherever the clean and patch prompts agree
  (position 0, "The": the IG path has Δ = 0), so T_N → 0 never includes the 91 latents that are active only there.
  Adding them gives Z = 0.999998. This is a property of the conversion, not a scoring bug, and it penalises SFC under
  our zero ablation in particular. SFC's own evaluator mean-ablates position 0 to its (identical) value and leaves
  layers 0-1 unablated, so there it costs nothing.

**Smallest passing size (distinct latents; "-" = never):**

| task | WCM, ours (pooled / per-pos) | WCM resample pass (pooled / per-pos) | SFC, ours (pooled / per-pos) | SFC, SFC's eval (latents; SFC nodes) | WCM, SFC's eval, errors kept (pooled / per-pos) |
|---|---|---|---|---|---|
| simple | 1,199 / 895 | 1,199 / 895 | - / - | 233; 162 | 576 / 895 |
| within_rc | 1,540 / 1,365 | 1,540 / 1,365 | - / - | 224; 173 | 1,540 / 737 |
| rc | 8,658 / 12,441 | 8,658 / 12,441 | - / 2,878 | 785; 600 | 2,528 / 5,948 |
| nounpp | - / 16,299 | - / 16,299 | - / - | 572; 419 | 1,551 / 773 |

With errors removed, no circuit of either method reaches 0.75 under SFC's evaluator. The best values are SFC 0.31 /
0.45 / 0.41 / 0.02 and WCM 0.30 / 0.35 / 0.50 / 0.01 (simple / within_rc / rc / nounpp).

**Area (nAUC):**

| task | WCM ours (pooled / per-pos) | SFC ours (pooled / per-pos) | SFC, SFC's eval (errors as nodes / removed) | WCM, SFC's eval kept (pooled / per-pos) | WCM, SFC's eval removed (pooled / per-pos) |
|---|---|---|---|---|---|
| simple | 0.479 / 0.508 | 0.214 / 0.292 | 0.729 / 0.162 | 0.527 / 0.536 | 0.113 / 0.143 |
| within_rc | 0.354 / 0.486 | 0.017 / 0.305 | 0.733 / 0.254 | 0.472 / 0.520 | 0.162 / 0.172 |
| rc | 0.275 / 0.260 | −0.018 / 0.181 | 0.611 / 0.212 | 0.353 / 0.308 | 0.159 / 0.147 |
| nounpp | 0.333 / 0.341 | 0.068 / 0.232 | 0.647 / −0.039 | 0.432 / 0.496 | −0.006 / −0.009 |

**Reading:**
- **The home-turf effect is large in both directions.** Each method wins on its own evaluation. Under ours, WCM's area
  is above SFC's on every task, and SFC passes once (rc, per_position, 2,878). Under SFC's, SFC's curve (errors as
  nodes) is above WCM's on every task, and SFC reaches 0.75 at 224-785 latents.
- **Range caveat:** WCM's grid starts at 319-1,551 latents, so the area gives it 0 below that. The λ grid needs points
  above 1e-3 to cover SFC's 10²-10³ range (specs/evaluation.md §6.4 item 9).
- **Errors drive SFC's own numbers:** with errors removed, SFC's own area drops to 0.16-0.25 (nounpp −0.04), about
  WCM's errors-removed level. WCM keeps every error by construction. Under SFC's evaluator with errors kept, it counts
  12·T extra nodes and gains the resid-error advantage SFC's Fig. 3 shows.
- **Donor recovery** (WCM): A / C in band at the passing sizes. R-pool recovery (corpus windows, ours) is low or
  negative everywhere (−0.98 to 0.74). Templates of 2-6 tokens do not survive random-text backgrounds.
- **Run note:** the WCM run's driver script was edited while it ran. Bash re-read the file, and the log ends with a
  syntax error after the last fit. All 48 circuits and eval rows were written.
