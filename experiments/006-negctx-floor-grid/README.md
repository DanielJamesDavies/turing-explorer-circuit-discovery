# negctx-floor-grid (2026-07-23)

*README generated from the scripts' docstrings; the scripts are the record.*

## `runner.py`

Grid: does the negctx floor help, and does negative hardness matter?

  4 seeds (L2/L8/L9/L10)  x  2 methods  x  4 discovery floors  = 32 discoveries

METHODS are the two DISTINCT floor consumers, deliberately not the two
top-scoring arms: ig_mean reaches the floor through
gradient_base._integrated_baseline_attribution, restoration through
instrument/restoration.run_restoration_selection. Together they exercise both
wired paths. cf-ig_mean is excluded because it is the same method as
abl-ig_mean (they agree to 0.006 on every metric across all four seeds — same
posctx floor, same IG path); sfc is excluded because it never calls
resolve_site_floors at all.

FLOORS: posctx (the in-run control) plus negctx under each negative-hardness
mode -- close, random, distant.

DESIGN. The discovery floor varies; the EVALUATION is held fixed. Every arm is
scored against identical anchors -- including a FIXED negctx eval floor taken
from the neg_ctx store regardless of which negatives the discovery used -- so
every column below is comparable across floors. Change both and the comparison
means nothing.

FULL EVAL MATRIX (no single metric decides this):
  free0        zero floor, live re-encode      -- floor-independent
  freeM_dense  posctx mean fill                -- the legacy/SFC-comparable one
  freeM_topk   posctx fill, k-sparse respected -- on-manifold variant
  freeN        negctx mean fill                -- shares free0's denominator
                                                  (a_eN==0) but a different
                                                  numerator: on-manifold fill
  pinMC_dense  posctx fill + pinned drivers    -- known unbounded, reached 2.05
  pinNC        negctx fill + pinned drivers    -- the same measure, cold floor
  cf / sup     counterfactual faithfulness on negctx + support
  faith_dense  logit-metric faithfulness over ALL sites
  up_nodes, pct_dict_up, secs_discover
  a_eN         the DISCOVERY floor's own a_empty (the leak measurement)

Anchors are logged per row (a_pos/a_e0/a_eM/a_eMT/a_eNfix + ac_*), so every
ratio is reconstructible -- see experiments/005-floor-diagnostic.

Per-seed process isolation (one OOM poisons the allocator); resume-safe.
Launch via launch.sh -- do NOT inline the loop through `wsl bash -lc`, where
$i is eaten by the outer shell and every seed silently runs as SEED_IDX="".

  SEED_IDX=0..3 PYTHONPATH=src python experiments/006-negctx-floor-grid/runner.py

## Result files

`rows_s0.jsonl`, `rows_s1.jsonl`, `rows_s2.jsonl`, `rows_s3.jsonl`
