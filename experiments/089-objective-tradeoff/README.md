# 089: is the A / C failure a coefficient trade-off against Z, or are the members insufficient?

## Question

Many failing protocol-v1 circuits pass Z (zero ablation) and necessity but fail A and C (the mean ablations). Two
readings:
- **trade-off:** the coefficients are pulled towards Z at the cost of A and C, so reweighting the loss would fix it;
- **insufficient members:** the node set cannot carry A and C, whatever the coefficients.

This refits 8 such circuits under different loss weights, with the production members fixed or free.

## Design

**Targets** (`select_targets.py`, seed 20261025, CPU): failing 062 production circuits (held-out strongest, not
vacuous) with Z in [0.8, 1.5], necessity ≥ 0.9, and A and C both < 0.8. Six MLP / residual targets at L5–11, and two
attention targets with clean rank < 64, which excludes near-threshold Top-K failures. All 8 picks are MLP or
attention (`targets.txt`):

`6.mlp.23423 5.mlp.29712 9.mlp.2360 11.mlp.24503 11.mlp.16843 8.mlp.7480 8.attn.20556 11.attn.13967`

**Arms** (`run.py`): 400 steps, otherwise protocol v1 (the 15k run's contexts and train arm, ablation values from the
training contexts, AdamW lr 0.05 / wd 0.05, batch 4, rank-keep γ_R 3e-3).

| arm | what moves | γ_Z | γ_A = γ_C | λ |
|---|---|---|---|---|
| `prod` | nothing: the 062 production circuit, re-scored | – | – | – |
| `prod_refit` | gates + coefficients (production WCM) | 1 | 0.25 | 1e-3 |
| `AC_only` | gates + coefficients | 0 | 1 | 1e-3 |
| `alpha_prod` | coefficients only, on the production members (control) | 1 | 0.25 | 0 |
| `alpha_AC` | coefficients only | 0 | 1 | 0 |
| `alpha_AC_Zhalf` | coefficients only | 0.25 | 1 | 0 |

- "Coefficients only" uses 062 `random_null`'s support wrapper: support restricted to the production members, gates
  fully open (θ_init 40, no binarisation), λ 0.
- γ_Z: the engine fixes the Z term's weight at 1.0. `src/` is not edited. `run.py` reads
  `src/circuit/instrument/learned_mask.py`, applies one exact text edit in memory (the Z term's 1.0 becomes `ZW["w"]`,
  and the term is dropped at 0), and execs the result as the engine module.
- **Scoring:** `driver.Runner.score` (the production scorer) on the held-out strongest 16, and the same scorer on a
  training probe (48 strongest-train reference rows, then the fit's 32 strongest training rows). A and C are scored
  under the scorer's sparsity-preserving fill (`freeM_topk`, `freeN_topk`). The dense fill the v1 terms were trained
  under (`freeM_dense`, `freeN_dense`) is reported beside it.

## Files and commands

| File | What |
|---|---|
| `select_targets.py` | picks the 8 targets from the 062 eval / spec rows, writes `targets.txt` |
| `run.py` | patched engine, fits, scoring; resumable per (target, arm); env `ARMS`, `OUT89`, `LIMIT`, `DEADLINE` |
| `report.py` | `out/tables.md` from `out/rows.jsonl` |
| `rows.jsonl` | a tracked copy of `out/rows.jsonl` (identical) |
| `run.log` | the run's log |

```
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/089-objective-tradeoff/run.py
python experiments/089-objective-tradeoff/report.py
```

The full run took 2,694 s on one GPU (`run.log`).

## Results

`rows.jsonl` is tracked. `out/` (circuits, `train.jsonl`, `tables.md`) is gitignored; the tables below are from
`out/tables.md` on the author's machine and can be rebuilt from `rows.jsonl` with `report.py`.

Medians over the 8 targets, held-out strongest (A / C = the scorer's sparsity-preserving fill):

| arm | median n | Z | A | C | rule pass | A dense | C dense | dense-fill pass |
|---|---|---|---|---|---|---|---|---|
| prod | 714 | 0.92 | 0.42 | 0.49 | 0/8 | 0.91 | 0.80 | 3/8 |
| prod_refit | 652 | 0.92 | 0.58 | 0.70 | 0/8 | 0.96 | 0.72 | 3/8 |
| alpha_prod | 714 | 1.00 | 0.49 | 0.50 | 1/8 | 0.99 | 0.85 | 5/8 |
| alpha_AC | 714 | 0.00 | 0.00 | 0.00 | 0/8 | 0.97 | 0.91 | 0/8 |
| alpha_AC_Zhalf | 714 | 0.92 | 0.37 | 0.38 | 2/8 | 1.00 | 0.93 | 6/8 |
| AC_only | 469 | 0.00 | 0.00 | 0.00 | 0/8 | 1.00 | 0.92 | 0/8 |

The training split gives the same pass counts (`out/tables.md`), so this is not a held-out generalisation gap.

Reading:
- **Without the Z term**, the fit reaches A and C of about 1 under the *dense* fill but Z collapses to 0 (`alpha_AC`,
  `AC_only`). Under the dense mean the circuit only has to supply the difference from the mean, never a set that
  suffices on its own.
- **Reweighting reaches the dense A / C, not the scored one.** With Z kept at 0.25, the same members pass the
  dense-fill rule on 6/8 targets (`alpha_AC_Zhalf`), but only 2/8 under the scorer's sparsity-preserving fill (both
  passes are MLP: 6.mlp.23423, 5.mlp.29712).
- The gap is between the fill the A / C terms are trained under (dense) and the fill they are scored under
  (sparsity-preserving), not between Z and A / C. That points to training A and C under the scorer's fill, which 090
  tests and protocol v2 adopts.
- Two targets fail Z in every arm (9.mlp.2360 A and C stay ≈ 0; 8.mlp.7480 Z ≤ 0.64). The attention targets also do
  not pass under any reweighting.

`paper/protocol.md` §12.1 cites this as "exp 089: reweighting the dense terms cannot reach the rule's A/C" in the
evidence for protocol v2. `paper/main.tex` does not cite 089 directly.

## Status and caveats

- **Superseded** by protocol v2 (090, 093; `paper/protocol.md` §12).
- **The engine patch no longer applies.** The anchor text in `run.py`'s `EDITS` (the v1 `terms = [...]` block in
  `learned_mask.py`) occurs 0 times in the current file (checked by reading, not by running), so `install_engine()`
  raises "anchor found 0 times". Re-running needs the `src/` of the time.
- 8 targets, chosen to fail. The numbers describe these failures, not the population.
- Scored on the 15k run's as-run held-out sets, which carry the duplicate-context leakage (DAN-157). The comparison is
  paired, so it is affected only through the shared records.
