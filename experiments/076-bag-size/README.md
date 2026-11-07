# 076 bag size: how many members does a circuit need?

## Question

A common critique of large circuits is that they are an "overwhelmingly large bag of heuristics". How concentrated
are WCM circuits? If members are ranked by their causal contribution to the target, how many of them carry most of
it, and does a top-n truncation (no refit) still pass?

## What the code does

`bag_size.py`, per target, on TuringLLM through the 062 driver (imported via `073-strict-eval/strict_eval.py`, so the
circuits and records are the 15k run's, `062/out_full/out/main/circuits`):

1. **Attribution.** `A_j = α_j · ∂(mean target pre-activation at the anchors) / ∂α_j`, in the circuit-only run under
   zero fill (Z), on the 48 strongest *training* contexts. This is each member's first-order contribution through
   every path inside the circuit. No held-out text is used to rank.
2. **Concentration.** `n50`, `n80`, `n90` = the number of members holding 50 / 80 / 90% of Σ|A_j|. Also the share of
   |attribution| that is negative (members pushing the target down).
3. **Truncations.** The top n members by |A_j|, n in {1, 2, 5, 10, 20, 50, 100, 200, 500} and the full circuit,
   coefficients unchanged and no refit. Each is scored on the 16 held-out strongest contexts under Z, A (activating
   mean) and C (contrast mean), both sparsity-preserving, each against its own empty circuit, with the protocol-v1
   ratio (x − e) / (a_pos − e) on the post-Top-K read. `n_min_band` = the smallest grid n with all three in
   [0.8, 1.5].

`report.py` (CPU) writes `out/report.md`:
- the attribution concentration, by depth band;
- the truncation pass shares, among circuits that pass in production. "Production pass" is recomputed from the 062
  held-out strongest eval rows: Z, A, C in [0.8, 1.5] and necessity (`phi_sup_blind_tk`) ≥ 0.9, vacuous rows dropped;
- for coverage of the whole run, a magnitude-share proxy for all 15,004 circuits from the explorer bundle:
  share = |α| × anchor × decoder norm over the circuit total. This is a write-size proxy, not causal.

**Targets:** the first 270 of the 747 random-null targets in 073's stratified, interleaved order (90 per depth band).
Nothing in `out_full` is written.

## Files and commands

| File | What |
|---|---|
| `bag_size.py` | Attribution, concentration and truncations (GPU) |
| `report.py` | The report (CPU; reads `out/rows.jsonl`, the 062 eval rows and the explorer bundle) |
| `out/rows.jsonl` | One row per target (270 rows) |
| `out/report.md` | The report the tables below come from |
| `out_smoke/rows.jsonl`, `smoke.log` | Smoke run on 2 targets |
| `run.log` | Console log of the full run |

```
LIMIT=270 PYTHONPATH=src python experiments/076-bag-size/bag_size.py     # env: LIMIT  OUT76  ATTR_BATCH (4)
PYTHONPATH=src python experiments/076-bag-size/report.py                 # -> out/report.md
```

## Results

All from `out/report.md` (270 targets, 0 error rows; 120 of them pass in production).

**Concentration of causal attribution** (medians):

| band | n | circuit size | n50 | n80 | n90 | n80 / size | negative mass |
|---|---|---|---|---|---|---|---|
| L0-4 | 90 | 270 | 28 | 87 | 130 | 0.32 | 0.30 |
| L5-7 | 90 | 552 | 52 | 169 | 254 | 0.31 | 0.35 |
| L8-11 | 90 | 726 | 57 | 196 | 310 | 0.28 | 0.35 |
| all | 270 | 497 | 46 | 153 | 228 | 0.30 | 0.32 |

**Top-n truncations, production-passing circuits:** share with Z, A and C all in [0.8, 1.5].

| band | passing n | top 50 | top 100 | top 200 | top 500 | full | median smallest n | median size |
|---|---|---|---|---|---|---|---|---|
| L0-4 | 55 | 15% | 24% | 53% | 98% | 100% | 200 | 257 |
| L5-7 | 41 | 0% | 0% | 2% | 51% | 95% | 500 | 572 |
| L8-11 | 24 | 0% | 0% | 0% | 4% | 83% | 837 | 810 |
| all | 120 | 7% | 11% | 25% | 63% | 95% | 438 | 464 |

Truncations to 20 or fewer members pass on at most 7% of targets in any band (`out/report.md`). Median faithfulness of
the truncations over all bands stays near 0 up to n = 100 (Z 0.00, A 0.11, C 0.02 at n = 100) and reaches Z 0.91,
A 1.03, C 0.98 at n = 500.

**All 15,004 circuits, magnitude-share proxy** (not causal): median n80 = 120 of 494 members (0.25), with the top
25 members holding a median 38% of the share. By band, n80 is 72 / 140 / 153 at L0-4 / L5-7 / L8-11.

Reading: about 30% of the members carry 80% of the causal attribution, and about a third of the attribution mass is
negative. Truncating to the top members by attribution, without refitting, does not keep the circuit faithful until
n is close to the full circuit size, and deeper circuits need nearly all of their members.

## Status and caveats

- Not cited in the paper.
- The circuits and records are the 15k run's (protocol v1), not the v4 refit.
- The truncation scores use the circuit's fitted coefficients unchanged. A truncated circuit refitted at its own size
  might pass at smaller n; that is not tested here.
- The attribution is first-order (gradient × α under Z), so it can misrank members whose effect is non-linear or
  appears only under the A or C fills.
- The 120 production-passing circuits in the truncation table (55 / 41 / 24 by band) are not depth-balanced.
