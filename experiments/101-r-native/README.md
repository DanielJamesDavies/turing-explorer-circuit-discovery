# 101: the resample term R made native (WCM_VERSION=v2cr)

Agent r-port. Goal: the pod run's protocol "v2 + R_C" (approved by Daniel) must not depend on
in-memory source patches. R (079's donor-resample term, 092 / 095's sparsity-preserving donor fill) is now a native,
default-off option of the production engine and a driver version.

## What was ported

| piece | before (patch) | now (native) |
|---|---|---|
| donor term, its weight gamma_R / dual_norm, appended after the A term | 079 `rs_engine` EDITS 2-3 (`patcher_rs`, `RESAMPLE`) | `run_learned_mask(resample_weight=...)`, donor patcher `patcher_donor` |
| donor codes of the pool, densified per forward for the drawn rows | 073 `capture_codes` + 079 `DonorFloors` | `capture_donor_codes` (same call, batch 16) + `DonorPool` |
| donors redrawn at every forward, one per training row, seeded CPU generator | 079 `on_forward` in `_forward_preact` | `_forward_preact` draws when `patcher.donors` is set; `resample_seed` (default 20261017) |
| donor floor never cached | 079 `DonorPatcher._const` | `LearnedMaskPatcher._const` (donors set) |
| sparsity-preserving donor fill under v2 | 092 `sp_keep_mask_donor` + 095 `_sp_delta_donor` | `donor_keep_mask` (verbatim) inside `_sparse_delta`; applies whenever `train_fill="sparse"` |
| anneal temperature, term name "resample" | 079 EDITS 4-5 | same, native |
| k donor draws per row per step | 100 `E4["k"]` | `resample_k` (term entered k times at gamma_R / k; names resample, resample2, ...) |

Files changed:
- `src/circuit/instrument/learned_mask.py`: `RESAMPLE_SEED`, `donor_keep_mask`, `capture_donor_codes`, `DonorPool`;
  `LearnedMaskPatcher(donors=...)`; engine kwargs `resample_weight` (0 = off), `resample_tokens`, `resample_k`,
  `resample_seed`; provenance `resample_weight / _k / _seed / _n_donors / _draws / _fill / _fill_calls`
  (`sparse_fill_calls` still counts C and A only, as 095 recorded it). R needs objective "pos" and a multi-term floor
  (dual / triple / pn); it carries no rank-keep / off-target term (as 079).
- `src/config.py` (`discovery.learned_mask`): `resample_weight` 0.0, `resample_pool` "contrast" | "activating",
  `resample_k` 1, `resample_seed` 20261017.
- `src/circuit/discovery/ablation_gradient.py`: with `resample_weight > 0` passes `method._resample_donors` (the caller
  must set it; refuses otherwise); off: the engine call is unchanged. Provenance gets `resample_pool`.
- `experiments/062-h100-protocol-v1/driver.py`: `WCM_VERSION=v2cr` = v2 + resample_weight 0.25, pool contrast, k 1.
  Env overrides (v2cr only): `R_WEIGHT`, `R_POOL`, `R_K`. `Runner.fit` sets `M._resample_donors` from
  `protocol_contexts.eval_reference`: contrast = the 48 contrast-train rows (= 091 `donor_tokens`, = the C term's
  rows), activating = the 48 strongest-train rows. OUT is marked `WCM_VERSION` = v2cr plus `OUT/WCM_R` (the R
  settings); another version or other R settings refuse that OUT. Train rows carry the resample stats (v2cr only).
  v1 / v2 set resample_weight 0 explicitly (= the config default): their behaviour is unchanged (checked below).

Historical scripts are untouched and still reproduce: every 079 / 092-t3 / 095 / 100-E4 anchor still matches exactly
once in the new src, the native code they inherit is dormant (`patcher_donor` is None), and `test_101_equiv_cpu.py`
runs their patched engines next to the native one.

## Design decision: unweighted arms under v2 (coordinator-approved)

The 095 patch built its donor patcher with `train_fill="dense"` and hooked the donor fill only on the free-amplitude
branch, so an UNWEIGHTED (gate-only) fit with R got the dense donor fill even under v2. Native gives the unweighted
donor term the same sparsity-preserving donor fill at alpha = 1 (the existing rule: U differs from WCM only in the
coefficients). No historical circuit used the old behaviour (100's random controls are free-amplitude), but the
difference is real and tested (`test_unweighted_sparse_differs_by_design`). Under `train_fill="dense"` both agree.

## Equivalence

**CPU, fake model, bit for bit** (`test_101_equiv_cpu.py`, 6/6 pass): native vs the 095 rs2 patch (v2CR: sparse fill,
donor fill on), vs 079's raw R (dense), vs 100 E4 k = 4: identical scores, amplitudes, per-step loss and every term's
curve. Unweighted dense identical; unweighted sparse differs by design.

**GPU, production path** (`run_smoke.sh` + `compare.py`, `out/compare.json`): the 062 driver, `MODE=main`,
`WCM_VERSION=v2cr`, on 099's audited records (as 100), 400 steps, compared with 100's Run A/C `v2cr_1e-3` circuits
(095 patch, fitted on the same src engine). The same targets under `WCM_VERSION=v2` against 099's production
circuits.

| target | v2cr vs 100 v2CR | v2 vs 099 |
|---|---|---|
| 2.attn.34021 | identical (297 members, max abs alpha diff 0, loss curve identical, 129 scores diff 0) | identical (255) |
| 6.mlp.209 | identical (451, curve identical, 133 scores diff 0) | identical (495) |
| 2.resid.29259 | identical (169, curve identical, 123 scores diff 0) | identical (160) |

Only the scorer's wall time / peak memory fields differ. 400 donor draws per fit, 48 donors, resample fill calls 2400 /
7600 / 3200 (sites x forwards, = 100's `donor_sp_calls` 2400 on 2.attn.34021). (095's own 72 v2CR circuits were fitted
before a later src change, so 100's circuits are the reference on the current engine; 100's E4 pre-flight had
already shown its rs2 refit bit-identical to them.)

## Tests

- `tests/circuit/instrument/test_learned_mask_resample.py` (new, 19): off == historical engine (weighted / unweighted,
  dense / sparse, whatever resample_tokens says; no capture, no donor patcher); on adds the "resample" term last and
  changes the loss; weight scales as gamma_R / dual_norm; donors redrawn per forward from the seeded sequence (same seed
  same fit, other seed other fit); k draws per row per step at gamma_R / k; unweighted + R (dense and sparse); the donor
  sparse fill code by hand; dense donor code and no floor caching; capture / pool; the hop passes donors only when on.
- `tests/test_wcm_version_driver.py` (extended): v2cr table, env overrides and their refusals, the OUT/WCM_R marker.
- Existing suites for touched areas: 427 passed (learned_mask, sparse fill, resample, driver, discovery modes,
  protocol_contexts, wcm_eval, public adapters, sweep runner).

## Smoke and fit time

Driver smoke: 3 targets x {v2cr, v2}, every row complete (status ok 3/3, eval strong + mid, 9 spec rows, train rows
with resample stats), scorer runs, rows carry `wcm_version` v2cr. Production post-Top-K reads, held-out strongest (Z /
A / C / necessity): v2cr 2.attn.34021 0.00 / 0.49 / 0.40 / 1.00, 6.mlp.209 0.99 / 1.05 / 0.98 / 1.00, 2.resid.29259
1.00 / 1.06 / 1.04 / 1.00.

Fit-time ratio v2cr / v2: **~1.40x** (18 targets of 100's panel, uncontended historical fits on this GPU: median 1.40,
range 1.25-1.47, total 1.40). The smoke's own times (1.23x total, 1.44x median) are not clean: another agent's GPU job
(arms-102) ran at the same time (GPU 9 GB / 78-97% busy right after), so absolute times were up to 2x 100's. A clean
re-time needs an idle GPU. An unweighted-arm GPU smoke was skipped for the same reason (CPU tests cover it).

## Open

- Pod launch: `WCM_VERSION=v2cr` for the 062-driver tracks (chain.sh exports `WCM_VERSION` to every driver job; no
  chain change needed). 081's `task_common.py` refuses any version but v1 / v2, but 082's chain does not run 081.
- `R_POOL=activating` uses the 48 strongest-train rows, which are also the 073 / 099 donor test's A pool; a pod arm
  with it would need disjoint training donors (100 E4 used mid-band rows). Not used by v2cr.
- 100's `run.py` reuse check hashes src against 099's manifest: after this change it no longer reuses 099's circuits
  (it refits them, bit-identically). 095's run.py does no such check.
