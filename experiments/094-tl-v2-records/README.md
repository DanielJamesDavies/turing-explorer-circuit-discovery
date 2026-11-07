# 094 TuringLLM v2 records (protocol v2 refit)

This folder holds the inputs of the TuringLLM refit under `WCM_VERSION=v2`: 8,000 targets on **fresh** context
records. The held-out leakage the 085 audit found is fixed at the source here: the pools are deduplicated, and the
split is group-aware. The pod stages are `082/chain.sh` T0–T9. The plan is in
`dev-notes/2026-10-23-pod-plan-public-models.md` §0v2.

| File | What |
|---|---|
| `make_targets.py` | The 8k list. Seed 20261026, a water-filled equal quota per layer × kind cell over the 062 15k (1.attn: all 134; the other cells 231 / 232). It holds every sweep, nulls, case-study, gallery and as-run-1,000 target. It writes `targets_tl8k.txt`, `targets_tl8k_<subset>.txt` and `targets_tl8k.manifest.json` (md5 of every list) |
| `build_records.py` | The fresh records, in `build_record`'s schema (`CTX_DIR` reads them). MODE=build / verify / report; the rule is in the module docstring |
| `test_094_records.py` | CPU tests: the frozen list, quotas, the symmetric duplicate test, the pool walk, the split |
| `targets_tl8k_gate64.txt` | T2's early gate: 64 targets over all 35 cells, from their own seeded generator. The 8k list is sorted, so its first 64 would all be 0.mlp. Built and verified before the rest; any failure stops the track |
| `quarantine.py` | T2's gate check (`gate`) and quarantine (`apply`). Past the gate, a target that fails verify (or has no record) goes to `lists/quarantine.tsv` with its reason. Every list minus the quarantined goes to `lists/`, which every later stage reads. Quarantined must-include targets are flagged loudly |
| `test_094_quarantine.py` | CPU tests on fake verify shards: the gate stops; a late failure quarantines and continues; the DRY_RUN reads `lists/` after T2 |
| `targets_smoke10.txt` | The 10-target local smoke, a subset of 086's smoke40, so the partial 086 smoke scan covers it |

## Local smoke

Run on 10 targets with `SCAN=086/out/scan_smoke40/scan_top512.pt` (a quarter of the corpus). Outputs are in
`out/smoke10/`.

- **Pools:** 10 / 10 records have:
  - 64 strongest rows (48 train / 16 held; D_train 32);
  - 64 mid-band rows (48 / 16; B_mid 16);
  - 64 contrast rows.
- **Strength:** held / train peak 1.00 (median of per-target medians); mean score 0.997.
- **Scan rows:** 60% of the strongest rows come from beyond the store's 64 slots. The deepest candidate rank used was
  120.
- **Rejections in the walk:**
  - strongest pool: 94 near and 69 anchor-prefix. There are no exact ones, because twins are merged by content hash
    before the walk;
  - mid pool: none;
  - contrast: 3.
- **Verify:** 10 / 10 ok (independent numpy predicates, firing and silence recomputed on the GPU).
- **Time:** 6.1 s per target to build, after a ~2.5 min one-off warm-up (the close selector). Verify takes ~13 s per
  target.

## Decisions taken here (for Daniel)

- **Mid-band held-out = 16 whenever the deduplicated mid pool has ≥ 32 rows** (`MID_HELD=fixed16`; accepted by
  Daniel). Below 32 rows it
  is v1's 3:1. The mid store has 64 slots and no deeper source, so deduplication can only shrink it.
  `MID_HELD=ratio` gives v1's 3:1 everywhere.
- **Duplicates are tested symmetrically inside a pool:** anchor-prefix under either row's anchor, and near with
  either row's 8-grams. This is `dup_groups`'s relation, so the group-aware split sees singletons only.
- **Contrast rows:** exact + near only, no anchor-prefix rule (as decided for 086).
