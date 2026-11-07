# 073 (+ 074): stricter evaluation of the 15k-run circuits, resample ablation and reads at every position

No refitting. This scores the production circuits of the 15k protocol-v1 run (`062-h100-protocol-v1/out_full`) with
two tests that are stricter than the production pass rule.

## Questions

- **073, resample ablation (the causal-scrubbing critique).** The production test replaces every latent outside the
  circuit with zero or a mean. Do the circuits still reproduce the target when everything outside them takes its
  value on another real context instead?
- **074, all-position simulation (the Gao critique).** The pass rule reads the target only at the anchor. Does the
  circuit-only run track the target at every position, including on contrast contexts where it should stay silent?

## Design (`strict_eval.py`)

**Targets:** the 747 targets that have a random-null circuit (`062/out_random_scale/circuits/<key>.d0.pt`), so every
target has a paired random baseline. They are processed in a band x kind interleaved order (`order_targets`), so a
partial run stays stratified.

**Variants per target:**
- `wcm`: the production circuit (`062/out_full/out/main/circuits`) at its fitted α;
- `a1`: the same members at α = 1;
- `rand`: the target's random-null circuit with its fitted α;
- `empty`: no members.

**073, resample fill (`ResamplePatcher`).** At every upstream site, members keep α x their live value. Every other
latent takes the donor context's natural sparse code at the same position: the donor's active non-member latents,
largest first, fill the Top-K budget left after the active members (the resample analogue of the `*_topk` fill).
The SAE error of the current stream is preserved, as in the engine's `CircuitOnlyPatcher`.
- Donor sets: **A** = the target's 48 strongest training contexts, **C** = its 48 training contrast contexts,
  **R** = random corpus windows from one fixed pool (`out/random_pool.pt`: 512 windows of at least 60 tokens, seed
  20261016), with a per-target draw.
- Score: (x − e) / (a_pos − e) at the held-out anchors, with x the circuit and e the empty circuit under the same
  donors, averaged over `N_DRAWS` = 2 donor draws of 16.
- Resample pass (`report.py`): Z, A and C (or R) in [0.8, 1.5], and production necessity ≥ 0.9.

**074, all-position reads.** The circuit-only target read at every position of the 16 held-out activating and 16
held-out contrast contexts, against the natural read, under fill Z (zero) and fill C (contrast mean,
sparsity-preserving, i.e. `freeN_topk`'s fill). Metrics (`sim_metrics`): Pearson r, r on activating contexts only,
fraction of variance explained, firing precision / recall, the share of contrast-context positions where the circuit
fires, and false-positive mass.

**Self-test (`SELFTEST=1`, 3 targets):** the base context as its own donor must give the natural read for the empty
circuit and for α = 1; an all-zero donor must give the engine's zero fill. `out/selftest.jsonl` shows the anchor means
agree to within bf16 round trip (e.g. 4.mlp.2484: self-donor empty 4.861 vs natural 4.904; zero donor 4.729 vs the
engine's Z 4.729), with the same max |diff| as the engine's own keep-all path.

**Diagnostic (`diag.py`):** why circuits fail under resample. 45 targets (5 per band x kind cell), donors C and R,
2 draws. Per target:
- the read both after Top-K (the pass-rule read) and as relu(w·x + b) (uncensored);
- members **live** (re-encoded from the mixed stream, as scored) vs **pinned** to their own natural values in the
  base context (α applied). Pinned much higher than live means the node set carries the signal but loses its inputs
  from outside the circuit (closure);
- a dose curve: a random fraction f ∈ {0.25, 0.5, 1} of the donor's non-member entries kept, the rest zero (f = 0 is
  the Z fill, f = 1 the full resample);
- the self-donor identity.

## Files and commands

| File | What |
|---|---|
| `strict_eval.py` | 073 + 074 scorer, resumable (rows already in `out/rows.jsonl` are skipped); `SHARD=i/k`, `FAMILY`, `TARGETS`, `RAND_DIR` |
| `report.py` | CPU: joins `out/rows.jsonl` with the 15k run's production eval rows, writes `out/report.md` |
| `diag.py` | the 45-target diagnostic, writes `out/diag.jsonl` |
| `run_073.sh`, `run_diag.sh` | WSL launchers |

```
nohup setsid bash experiments/073-strict-eval/run_073.sh > experiments/073-strict-eval/run.log 2>&1 &
SELFTEST=1 PYTHONPATH=src .venv/bin/python experiments/073-strict-eval/strict_eval.py
PYTHONPATH=src .venv/bin/python experiments/073-strict-eval/report.py          # -> out/report.md
bash experiments/073-strict-eval/run_diag.sh                                   # -> out/diag.jsonl
```

The full run took about 77 minutes on one GPU (`run.log`; median 5.1 s per target).

`strict_eval.ResamplePatcher` and `score_target` are reused elsewhere: 079 and 095 score with
`score_target(c_held=True)` (C donors = the 16 held-out contrast contexts), 108 and the v4 chain reuse
`out/random_pool.pt`, and `src/eval/donor_test.py` (the v4 donor-background test, `paper/protocol.md` §0) uses the
same fill.

## Results

`out/` is gitignored, so the outputs are not tracked. The numbers below come from `out/report.md` (747 targets, 0
errors) and `out/diag.jsonl` (45 targets) on the author's machine. `out/diag_v1.jsonl` is an earlier diagnostic pass
on the same 45 targets without the `mpre` columns; its shared columns give the same medians.

**073, resample ablation** (medians; production pass rate on these 747 targets 43.1%):

| band | n | prod pass | resample pass (A,C) | A wcm | C wcm | R wcm | rand pass (A,C) |
|---|---|---|---|---|---|---|---|
| L0-4 | 249 | 61.4% | 0.4% | 0.29 | 0.14 | 0.08 | 0.0% |
| L5-7 | 249 | 45.0% | 0.0% | 0.16 | 0.03 | 0.01 | 0.0% |
| L8-11 | 249 | 22.9% | 0.0% | 0.09 | 0.02 | 0.00 | 0.0% |
| all | 747 | 43.1% | 0.1% | 0.18 | 0.04 | 0.01 | 0.0% |

- α = 1 scores about 0 under every donor set. The resample pass with R in place of C is 0.0%.
- Of the production passes, 0.3% also pass resample (A,C).
- The A donors leak the target into the empty circuit (median e / a_pos 0.12); C and R do not.
- Two-draw stability: median |draw1 − draw2| = 0.021 (C, wcm).

**Diagnostic** (`out/diag.jsonl`, C donors, post-Top-K read, medians by kind, 15 targets each):

| kind | live | pinned | dose f = 0.25 | f = 0.5 | self-donor wcm / natural |
|---|---|---|---|---|---|
| resid | 0.13 | 0.92 | 0.53 | 0.35 | 2.16 |
| mlp | 0.03 | 0.39 | 0.29 | 0.09 | 1.28 |
| attn | 0.01 | 0.11 | 0.03 | 0.02 | 1.05 |

The uncensored read is as low as the post-Top-K one (resid 0.14, mlp 0.05, attn 0.04), so the target is not merely
pushed out of its Top-K: its pre-activation is not reproduced. Pinning the members restores residual-stream targets,
so the failure there is closure (members lose their inputs from outside the circuit), not membership. With the base
context as its own donor, the fitted α overshoot (resid 2.16): coefficients fitted to stand in for an ablated
complement over-drive the target when the complement is present.

**074, all-position reads** (`out/report.md`, medians, `wcm` variant):

| fill | band | r | fve | contrast positions firing |
|---|---|---|---|---|
| Z | L0-4 | 0.75 | 0.03 | 0.01 |
| Z | L5-7 | 0.58 | −1.06 | 0.31 |
| Z | L8-11 | 0.50 | −2.14 | 0.74 |
| Z | all | 0.59 | −0.85 | 0.25 |
| C | all | 0.66 | −0.06 | 0.03 |

Under fill C, passing vs failing production circuits: r 0.88 vs 0.35, fve 0.46 vs −0.25. Random and empty circuits
give r about 0 under both fills.

**In the paper:** `paper/main.tex` §`app:resample` (lines 1405-1442) quotes the 073 rates and medians (0.1% vs 43.1%;
0.18 / 0.04 / 0.01), the diagnostic's pinned medians (0.92 / 0.39 / 0.11) and self-donor overshoot (2.16), and the 074
correlations, variance explained and contrast firing rates above. Line 1576 cites 073 for not describing the
three-ablation objective as resample ablation.

## Status and caveats

- Scored on the 15k run's **as-run held-out sets**, which carry the duplicate-context leakage found later (085 / 086,
  DAN-157). The paper notes that a re-score on the rebuilt sets is pending (`main.tex` lines 1433-1436). Leakage can
  only flatter the activating-context reads; the contrast-context reads are unaffected.
- The circuits are protocol v1 (dense mean fill in training). 095 repeats the resample comparison under protocol v2.
- C donors here are the 48 **training** contrast contexts. 079 and 095 later scored with held-out contrast donors
  (`c_held=True`) so that circuits trained against training donors are not scored on them.
- `paper/protocol.md` §0 notes that the v4 chain still runs `strict_eval.py`, on the harder nulls' circuit pool.
