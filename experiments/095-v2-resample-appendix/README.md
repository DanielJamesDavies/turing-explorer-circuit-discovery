# 095: the resample appendix (`app:resample`) under protocol v2 (DAN-184)

The appendix quotes v1 numbers (079's 99-target confirmation: resample pass 0% -> 30%, production pass 44% -> 57%,
size x1.22). Production is now protocol v2 (`paper/protocol-v1.md` §12), so this re-runs that comparison under v2, with
two ways of adding R: in place of C (v2R) and on top of C (v2CR, 079's shape; added 2026-10-26, Daniel).

## Design

| | |
|---|---|
| Targets | exp 093's 72 (8 per depth band x site kind, seed 20261026), processed in a cell-interleaved order (`run.balanced_order`) so any prefix stays balanced |
| Arm `v2` | 093's v2 circuits, **reused, not refitted** (`093-v2-validation/out/circuits/v2`; fit time / peak memory from 093's rows, same GPU) |
| R term (both R arms) | 079's donor-resample term, donors = the target's 48 **training** contrast contexts (091 `setup_terms`: one donor per training row per forward, redrawn every forward, DonorFloors seed 20261017), weight 0.25, under the sparsity-preserving donor fill (092 t3) |
| Arm `v2R` | Z + A + R_C: R in place of C (exp 092's test 3, `t3_ZAR`) |
| Arm `v2CR` | Z + A + C + R_C: R added, C kept at v2's sparse fill and the scorer's C mean, gamma_C 0.25 (079's winning shape) |
| Both R arms | fitted here, 400 steps, `WCM_VERSION=v2` (A / C under v2's fill against the scorer's means), gamma 0.25, lambda 1e-3, rank-keep 3e-3; **no silent-position term** |
| Contexts | the 15k run's as-run records (`CTX_DIR` = `062/out_full/out/ctx`, explicit eval split), as 093. The held-out sets carry the DAN-157 duplicate leakage: the comparison is paired (all arms on the same records); absolute rates are on leaky held-out sets |
| Production scorer | `driver.Runner.score`, held-out strongest, all arms re-scored in this process (identical code; v2 checked against 093's rows). Rule pass (post-Top-K, headline) and the pre-activation pass of §12 (Z and necessity on the 0 baseline) |
| Resample scorer | 073 `strict_eval.score_target(c_held=True)`, exactly as 079's confirmation: A donors = 48 strongest-train, C donors = the 16 **held-out** contrast contexts (never a training donor), R donors = 073's evaluation pool, 2 draws. Resample pass (079 `report.py`) = production Z, rs A, rs C in [0.8, 1.5] and production necessity >= 0.9; `vacuous_tk` excluded |
| Statistics | paired, by all / band / kind: McNemar exact + paired bootstrap 95% CI (10,000 target resamples, seed 20261026) for v2 vs v2R, v2 vs v2CR, v2R vs v2CR |

**v2CR vs 079's arm (RS0.25_S0.25), what differs.** Same shape (Z + A + C + R, R weight 0.25, C kept), but:
- no silent-position term (079 had it at 0.25; 091 showed it adds nothing and Daniel dropped it);
- donors are the 48 training contrast contexts only (079: contrast + random corpus contexts);
- R is trained under the sparsity-preserving donor fill (079: raw donor code, no Top-K re-imposed);
- the protocol under it is v2 (sparse A / C fill, scorer's means), where 079's was v1 (dense fill, the fit's own means).

### Deviation from the brief: 092's engine patch could not run as-is

092's `sx_engine` is an in-memory patch on 090's `sp_engine`, whose first edit anchors the pre-v2 floor branch of
`LearnedMaskPatcher.transform`. The v2 port into src (2026-10-25) moved that branch under an `else:`, so the anchor no
longer exists and `sx_engine.source_v2r()` refuses to build on today's src. Running it on the old src would also not
be "v2 with R": 092's v2 was 090's `sp_ste` with the engine's own A mean, whereas production v2 (and 093's v2 arm)
trains against the scorer's means (`train_means_source = "eval_ref"`).

`rs2_engine.py` therefore ports 092's test 3 onto today's src (src is not edited): 079's `rs_engine.EDITS` + footer
and 092's `drop_floor` edit are reused verbatim, 092's `sp_keep_mask_donor` is cut from `sx_engine.FOOTER` and
exec'd as is, and one edit (the only one touching the v2 code) hooks the donor fill into the native v2 branch, calling
the native `sparse_code(code, H, "ste")` (identical to 092's `sp_code(..., "ste")`) with K = the dictionary's k.
Switches: `drop_floor` (v2R on, v2CR off), `sp_donor` (both on), `RESAMPLE` weight 0.25 (both).
Tests (`test_rs2_engine.py`, CPU, all pass): every anchor applies once; with every switch off the Z and A patchers
are bit-identical to the unpatched src (forward and gradients); the donor patcher without the fill is 079's raw R;
with the fill at binary gates it equals 073's `ResamplePatcher` fill exactly, with straight-through gradients.
GPU checks (`out/check.jsonl`, `check.log`): a 400-step refit of 093's v2 circuit for 2.attn.22342 through rs2_engine
is **bit-identical** to 093's (same members, same coefficients) both with every switch off (`CHECK=1`) and with v2CR's
switches at R weight 0, i.e. keep C + donor fill on + no R term (`CHECK=2`).

## Files

| File | What |
|---|---|
| `rs2_engine.py` | the ported 092 t3 engine patch (in memory) |
| `test_rs2_engine.py` | CPU tests |
| `run.py` | fits v2R / v2CR, scores all arms with both scorers; resumable per (target, arm); `ARMS`, `CHECK=1/2`, `DEADLINE` |
| `report.py` | `out/tables.md` (three arms, all pairs; also recomputes 079's v1 confirmation with the same definitions) |
| `out/rows.jsonl` | one row per (target, arm): full production-scorer row (`prod`), resample row (`resample`), pass flags |
| `out/fits.jsonl`, `out/train.jsonl`, `out/circuits/{v2R,v2CR}/` | fit stats, training curves, circuits |

## Commands (WSL, repo root)

    PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/095-v2-resample-appendix/test_rs2_engine.py
    # smoke: 1 target, 20 steps, both scorers end to end
    LIMIT=1 STEPS=20 OUT95=experiments/095-v2-resample-appendix/out_smoke PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/095-v2-resample-appendix/run.py
    # inertness checks (400 steps, first target): CHECK=1 all switches off; CHECK=2 keep C + donor fill, R weight 0
    CHECK=1 PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/095-v2-resample-appendix/run.py
    # full run (resumable; DEADLINE stops before a target that would end after it)
    DEADLINE=22:30 PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/095-v2-resample-appendix/run.py >> experiments/095-v2-resample-appendix/run.log 2>&1
    .venv/bin/python experiments/095-v2-resample-appendix/report.py

## Results (2026-10-26; full tables in `out/tables.md`, rows in `out/rows.jsonl`)

Runs on the RTX: v2R 15:16-17:08, v2CR 18:18-20:18, 0 errors (logs `smoke.log`, `check.log`, `run.log`). All 72
targets complete in all three arms, none rejected, none vacuous; no deadline cut was needed.

**Checks.** The v2 circuits re-scored here equal 093's rows exactly (max |diff| 0 over Z / A / C / necessity, both
reads; rule pass agrees 72 / 72). Both GPU inertness checks are bit-identical (above). Every v2R fit trained exactly the
terms zero, pos, resample; every v2CR fit zero, floor (C), pos, resample; donor fill active, eval_ref means, 48 donors.

**Three-arm table (n = 72; tables.md §1, §2, §3).**

| | v2 | v2R (R for C) | v2CR (R + C) | 079 v1: W0 -> RS (n = 99) |
|---|---|---|---|---|
| resample pass | 0% | 25% | 24% | 0% -> 30% |
| ... L0-4 / L5-7 / L8-11 | 0 / 0 / 0% | 42 / 29 / 4% | 33 / 33 / 4% | 0 -> 39 / 36 / 15% |
| rule pass, post-Top-K | 69% | 51% | 62% | 44% -> 57% |
| ... L0-4 / L5-7 / L8-11 | 79 / 83 / 46% | 83 / 50 / 21% | 71 / 79 / 38% | 70 / 42 / 21% -> 67 / 58 / 45% |
| pre-activation pass | 72% | 57% | 69% | - |
| median rs A / rs C / rs R | 0.08 / 0.01 / 0.00 | 0.88 / 0.74 / 0.58 | 0.85 / 0.74 / 0.55 | 0.20 / 0.05 / 0.01 -> 0.86 / 0.74 / 0.62 |
| median C (L8-11) | 0.83 | 0.48 | 0.83 | - |
| size vs baseline (median per-target ratio) | 1 | 1.11x | 1.18x | 1.22x |
| fit time vs baseline (median per target; of sums) | 1 | 1.13x; 1.20x | 1.45x; 1.47x | 1.64x; 1.95x |

**Paired differences (McNemar exact; paired bootstrap 95% CI; tables.md §1b).**

| comparison | resample pass | rule pass, post-Top-K | pre-activation pass |
|---|---|---|---|
| v2 -> v2CR | +24 pp [+14, +33], 0 / 17 discordant, p 1.5e-05 | -7 pp [-17, +3], 9 / 4, p 0.27 | -3 pp [-11, +4], 5 / 3, p 0.73 |
| v2 -> v2R | +25 pp [+15, +35], 0 / 18, p 7.6e-06 | -18 pp [-29, -7], 17 / 4, p 0.0072 | -15 pp [-25, -6], 13 / 2, p 0.0074 |
| v2R -> v2CR | -1 pp [-10, +7], 5 / 4, p 1 | +11 pp [+0, +22], 4 / 12, p 0.077 | +12 pp [+3, +22], 3 / 12, p 0.035 |
| 079 v1: W0 -> RS | +30 pp [+21, +39], 0 / 30, p 1.9e-09 | +12 pp [+5, +20], 2 / 14, p 0.0042 | - |

**Reading.**
- Both R arms buy the resample test equally (+24 / +25 pp, about 079's +30 pp), and neither helps deep targets much
  (4% at L8-11 in both, against 15% under v1).
- Keeping C (v2CR) keeps the production test: -7 pp with a CI that includes 0, and the pre-activation pass is
  unchanged. v2CR's deep C stays at v2's 0.83; v2R's falls to 0.48. The cost is 1.18x size and 1.45x fit time.
- Dropping C (v2R) costs the production test (-18 pp), concentrated at L5-7 / L8-11 and in attention / residual
  targets. v2CR beats v2R there (L5-7 +29 pp, attention +25 pp) at the same resample pass.
- Unlike v1, adding R does not raise the production pass rate under v2. v2's baseline is already 69% (v1's W0 44%),
  having fixed the dense-fill problem that v1's R term partly compensated for. 079's arm also differs in its silent
  term, mixed donors and raw R.
- The all-position correlation under the C fill falls for both R arms (0.65 -> 0.53 / 0.52; L8-11 0.40 -> 0.14 / 0.30).
  This is the side effect 079 traced to firing at silent positions; v2CR has no silent term to counter it.

**Caveats.** The held-out sets are the 15k run's as-run sets with the DAN-157 duplicate leakage: the comparison is paired
(all arms, same records), but absolute rates are on leaky held-out sets and the rebuilt records (094) may move them.
v2's fit times are 093's (same GPU, a day earlier, alternating with v1 fits in one process), so fit-time ratios carry
some machine-state noise. 092 showed that refits move single-target passes (bf16 kernel noise); read single-band
differences of 1-2 targets as noise.

**079 v1 reference** (tables.md §5, recomputed from `079/out_confirm/rows.jsonl` with the same definitions; it
reproduces the appendix's numbers).

## Draft replacement text for `app:resample` (NOT applied; `paper/main.tex` untouched)

Every number traces to `out/tables.md` (sections in the comment lines).

Replace the paragraph "Nothing in training asks a circuit to survive a real background, ..." through "Deep targets gain
least on the resample test but most on the production test ($21\%$ to $45\%$)." (main.tex lines 1332-1337) with:

```latex
% 2026-10-26: experiments/095-v2-resample-appendix (out/tables.md). Protocol v2, 72 targets = exp 093's (8 per band x
% kind), paired: v2 (093's circuits) vs v2CR (R added, C kept) vs v2R (R in place of C), both refitted; no silent-
% position term. Held-out sets: the 15k as-run sets (DAN-157 leakage; paired, absolute rates on leaky sets).
% Rates s1, paired CIs s1b, medians s2, size / time s3; v1 numbers: 079 out_confirm, recomputed in s5.
Nothing in training asks a circuit to survive a real background, so we tried adding one: a resample ablation whose replacement values are the sparse codes of the target's training contrast contexts, one donor per context, drawn afresh at every step, with the donor's sparsity kept as in the test; the test's donors are held-out contrast contexts and random corpus text, never seen in training.
On $72$ targets ($8$ per depth band and site kind), paired against the production configuration, adding it as a fourth ablation method raises the resample pass rate from $0\%$ to $24\%$ (95\% CI of the difference $+14$ to $+33$ points; $33\%$, $33\%$, and $4\%$ by depth band), while the production pass rate moves from $69\%$ to $62\%$ (CI $-17$ to $+3$ points), at $1.18\times$ the circuit size and $1.45\times$ the fitting time.
Using it in place of the contrast-mean ablation gains as much on the resample test ($25\%$) but costs the production test ($51\%$; CI $-29$ to $-7$ points), most at layers 8--11 ($46\%$ to $21\%$), where the median contrast-mean faithfulness falls from $0.83$ to $0.48$.
Under our earlier training protocol, whose dense mean fill left more circuits failing the production test ($44\%$), the same addition raised both rates on $99$ targets, the resample pass rate to $30\%$ and the production pass rate to $57\%$; with the sparsity-preserving training fill (\cref{sec:ablmask}) only the resample gain remains.
Deep targets gain least on the resample test under either protocol ($4\%$ here, $15\%$ before).
```

(`sec:ablmask`, main.tex line 271, is where the paper states the v2 training fill. The v1 sentence leaves out 079's
other differences: a silent-position penalty, random-text training donors and no sparsity-preserving donor fill; add
"with a penalty on firing at silent positions" after "the same addition" if the appendix should name the main one.)

The main-text summary (main.tex line 587) quotes the v1 result; a matching v2 sentence:

```latex
When everything outside the circuit takes its value on another real context, as in causal scrubbing, almost no circuits pass; training against such backgrounds as well raises this to $24\%$ on $72$ targets without a significant loss on the production test ($69\%$ to $62\%$; \cref{app:resample}).
```

**Not covered here, still v1 in the appendix:** the 747-target 073 scoring ("$0.1\%$ ... against $43.1\%$", medians
0.18 / 0.04 / 0.01, "random circuits pass on none"), the 45-target closure diagnostic (0.92 / 0.39 / 0.11, the 2.16
overshoot) and the 074 all-position paragraph. 095's v2 arm gives a small-sample v2 counterpart of the first: 0 of 72
v2 circuits pass the resample test against 69% under the production test, with median resample faithfulness 0.08 (A),
0.01 (C), 0.00 (R) (tables.md §1-2). It is not a like-for-like replacement: C donors here are the 16 held-out contrast
contexts (073's 747-target run used the 48 training ones), and there is no random-circuit baseline.
