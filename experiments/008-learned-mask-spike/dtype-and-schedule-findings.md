# Learned-mask: code_dtype verification + lr-schedule setup

Status at 2026-07-28 (paused overnight). Nothing here is committed.

## 1. bf16 vs fp32 — RESOLVED 2026-07-29: solution is non-unique, dtype is fine

`determinism_check.py` at L8, each dtype run twice:

```
fp32.a  n=53,955  free0=1.0581  loss=8.007881
fp32.b  n=53,955  free0=1.0581  loss=8.007881
strm.a  n=55,084  free0=1.0604  loss=8.318472
strm.b  n=55,084  free0=1.0604  loss=8.318472

fp32.a fp32.b  within  jaccard 1.0000   0.00% flip  max|dm| 0.00000
strm.a strm.b  within  jaccard 1.0000   0.00% flip  max|dm| 0.00000
fp32.* strm.*  ACROSS  jaccard 0.8350  16.50% flip  max|dm| 0.34016
```

**The pipeline is bit-deterministic within a dtype** — repeats are identical
to six decimals with zero flipped members. There is no run-to-run noise
floor, so differences in the wd and lr sweeps are real signal, not variance.
(The overnight worry that those sweeps might be measuring noise is closed.)

**Do not read bf16's higher loss as a worse optimum.** The loss is measured
in the working dtype, so 8.318 and 8.008 are not on a common scale. free0
*is* on a common scale (evaluated identically for both) and is a wash,
fractionally favouring stream: 1.0604 vs 1.0581.

**Verdict: the mask solution is non-unique.** 16.5% of members swap out with
no functional consequence; a perturbation as small as bf16 rounding finds a
different, equally good member set. This is evidence for latent redundancy,
and cleaner than the coactivation analyses because the two circuits are
matched on everything except numerical precision.

**Default stays `stream`**: free0 is a wash, it is 5x faster at L10 (§2) and
~10% faster elsewhere. Paper caveat: membership is dtype-dependent at the
~16% level even though measured circuit quality is not — a property of the
object, not a defect in the method.

Both hypotheses posed before the control were wrong; the original framing is
kept below for the record.

### Original (pre-control) framing

`code_dtype="stream"` keeps cached SAE codes in native bf16 instead of
promoting to fp32 (~1 GB peak saving). It was made the default *before* being
verified; this was the verification.

`dtype_check.py`, objective=pos, 400 steps, lr 0.05, λ 1e-4, AdamW wd 0.05,
64 probes:

| | L8 | L10 |
|---|---|---|
| n (fp32 → stream) | 53,955 → 55,084 | 106,609 → 108,068 |
| free0 (fp32 → stream) | 1.0581 → 1.0604 | 1.0703 → 1.045 |
| Jaccard | 0.835 | 0.803 |
| flipped % of union | 16.5% | 19.7% |
| max abs(Δm) on *shared* members | 0.340 | 0.391 |

**Aggregates are stable, membership is not.** free0 agrees to 0.2% (L8) /
2.4% (L10) and n to ~2%, but 16-20% of the union flips and shared members'
m values move by up to 0.39. That is far too large to be rounding — it is
trajectory divergence amplified over 400 optimisation steps.

**The conclusion does NOT follow yet.** The control was never run: fp32 was
never repeated against itself. GPU reductions are non-deterministic
regardless of dtype, so fp32-vs-fp32 may churn just as much. Two opposite
readings remain open:

- within-fp32 ≈ within-stream ≈ across → churn is run-to-run, the mask
  optimum is non-unique, and dtype is exonerated. This would be a
  *substantive* finding, not a nuisance: it would mean membership at the
  0.5 threshold is not identifiable from 64 probes, which bears directly on
  the latent-redundancy thread (many near-equivalent solutions) and on how
  any per-latent membership claim can be phrased in the paper.
- within-dtype high, across low → bf16 genuinely changes selection, and
  every learned-mask number produced since the switch needs redoing.

`determinism_check.py` runs fp32 twice and stream twice at L8 and prints the
pairwise Jaccard matrix. **Run this first tomorrow** — it gates the schedule
experiment, because a schedule difference smaller than the run-to-run noise
floor is not a result.

Related: it also gives the noise floor for *every* learned-mask comparison
we have made (the wd and lr sweeps included). Differences we have been
reading as signal may be inside it.

## 2. Unambiguous side result: fp32 falls off the VRAM cliff at L10

L10 fp32 peaked at 14.74 GB and took **633 s**; stream peaked 13.80 GB and
took **128 s** — 5x faster for ~1 GB. This is the WDDM spill cliff from the
earlier VRAM postmortem: the ~1 GB is what keeps L10 on the right side of
it. Independent of how the selection question resolves, stream earns its
place at L10 on throughput.

## 3. lr schedule — RESOLVED 2026-07-29: decay is the wrong direction, keep constant

All arms at sum(lr) = 20.00 exactly (peak lr doubled for non-constant), so
schedule SHAPE is the only variable. AdamW, wd 0.05, lambda 1e-4, 400 steps.
n is shown relative to that seed's constant arm; f0_h is held-out free0.

| seed | constant | cosine | linear | cosine_up | linear_up |
|---|---|---|---|---|---|
| L2 n | 1,689 | +24% | +11% | +37% | +26% |
| L2 f0_h | **0.9803** | 0.9291 | 0.9409 | 0.9449 | 0.9764 |
| L2 m_kept | 0.770 | 0.748 | 0.749 | 0.831 | 0.809 |
| L8 n | 55,084 | +19% | +13% | -4.6% | **-6.2%** |
| L8 f0_h | 1.0142 | 1.0284 | 1.0332 | 1.0095 | **1.0379** |
| L8 m_kept | 0.711 | 0.708 | 0.704 | 0.769 | 0.743 |
| L10 n | 108,068 | +25% | +16% | +7.1% | +2.3% |
| L10 f0_h | 0.9388 | 0.9456 | 0.9116 | 0.9728 | **1.0748** |
| L10 m_kept | 0.717 | 0.715 | 0.710 | 0.765 | 0.760 |

**Decay loses.** Bigger circuits at 3/3 seeds (+11..+25%) with no consistent
quality gain (L2 favours constant, L8 favours decay, L10 a wash). It moves
along the size/fidelity trade-off, not off it.

**Keep `lr_schedule="constant"`** — smallest at 2/3 seeds, never badly beaten.
The schedule machinery stays in: tested, and free when unused.

### Two mechanisms proposed and BOTH refuted — do not resurrect them

1. *"Matched sum(lr) makes shape the only variable, so budgets carry over."*
   FALSE. sum(lr) was matched to 20.00 exactly and circuit size still moved
   25%. The budget abstraction (sparsity = sum(lr)*lambda, decay =
   sum(lr)*wd) that the wd/lr calibration rests on does NOT determine the
   outcome by itself. Treat it as a normalisation convention, not a model.
2. *"AdamW's decoupled decay is the leak"* — it is multiplicative,
   theta_final = theta_0 * prod(1 - lr_t*wd), and by AM-GM that depends on
   the shape of {lr_t}. FALSE: under plain Adam with wd = 0 the size gap
   WIDENED (constant 55,829 -> cosine 71,360, +28%, vs +19% under AdamW).
   Decoupled decay was damping the effect, not causing it. (It is, however,
   what pulls m_kept down: 0.87 at wd=0 vs 0.71 at wd=0.05.)
3. *"Late-training lr governs size, so warmup should prune harder"* — the
   prediction the `_up` variants were built to test. FALSE: warmup shrinks
   at L8 only (-6.2%) and grows at L2 (+26%) and L10 (+2.3%).

**The one consistent effect: m_kept is monotone in schedule DIRECTION**
(decay < constant < warmup at 3/3 seeds). Shape reliably controls how
confident the surviving members are, but not how many survive. What governs
size remains unexplained — three stories have now failed, so the next claim
about it should come with a measurement attached.

## 3b. RESOLVED 2026-07-29: lr is a BUDGET DIAL. Shape does not matter.

Warmup-then-cosine, peak 0.30, floor 0.01, 10% warmup (40 steps) — the
conventional recipe the earlier arms never tested. sum(lr) = 62.15 = **3.11x**
the reference budget of 20. Run three ways to separate shape from budget:

| seed | arm | n | f0_all | f0_train | f0_hold | m_kept |
|---|---|---|---|---|---|---|
| L8 | constant (flat, 1.0x budget) | 55,084 | 1.0604 | 1.0758 | 1.0142 | 0.711 |
| L8 | warmcos_raw (hot shape, 3.11x) | **14,488** | 1.0687 | 1.0616 | 1.0900 | 0.685 |
| L8 | warmcos_match (hot shape, 1.0x) | 82,024 | 1.0296 | 1.0348 | 1.0142 | 0.838 |
| L8 | hotflat (FLAT, 3.11x) | **16,682** | 1.0438 | 1.0427 | 1.0474 | 0.737 |
| L2 | constant | 1,689 | 1.0010 | 1.0077 | 0.9803 | 0.770 |
| L2 | warmcos_raw | **1,134** | 1.0405 | 1.0536 | 1.0000 | 0.746 |
| L2 | warmcos_match | 3,630 | 0.9557 | 0.9668 | 0.9213 | 0.876 |
| L2 | hotflat | **1,111** | 1.0443 | 1.0510 | 1.0236 | 0.801 |
| L10 | constant | 108,068 | 1.0450 | 1.0833 | 0.9388 | 0.717 |
| L10 | warmcos_raw | **27,162** | 1.2054 | 1.2157 | 1.1769 | 0.651 |
| L10 | warmcos_match | 173,065 | 1.0108 | 1.0539 | 0.8912 | 0.841 |
| L10 | hotflat | **26,450** | 1.2072 | 1.2083 | 1.2041 | 0.702 |

**hotflat reproduces warmcos_raw** (within 2-3% at L2/L10, 15% at L8) while
warmcos_match — same shape, reference budget — is WORSE than constant on both
axes at 3/3. Across decay, warmup, and warmup-then-decay, no shape effect
survived matching the budget. **lr is a budget dial; tune lambda and wd
directly instead.**

Corrects the stronger claim in §3: the budget abstraction is the DOMINANT
term (4x size), with shape a second-order correction (2-28%). Not "merely a
normalisation convention".

### The 3.11x budget buys 4x compression and costs OVERSHOOT

|free0_hold - 1|, lower is better (free0 > 1 = the circuit OVER-drives the
seed; this project treats overshoot as a defect, cf's 1.23 was flagged as one):

| seed | constant | hotflat |
|---|---|---|
| L2 | 0.0197 | 0.0236 |
| L8 | **0.0142** | 0.0474 |
| L10 | **0.0612** | 0.2041 |

But the two arms are wrong in DIFFERENT WAYS, and constant's is arguably worse:

| seed | constant train->hold | gap | hotflat train->hold | gap |
|---|---|---|---|---|
| L2 | 1.0077 -> 0.9803 | 0.027 | 1.0510 -> 1.0236 | 0.027 |
| L8 | 1.0758 -> 1.0142 | 0.062 | 1.0427 -> 1.0474 | **-0.005** |
| L10 | 1.0833 -> 0.9388 | **0.145** | 1.2083 -> 1.2041 | **0.004** |

Constant OVERFITS at depth (0.145 gap at L10); the hot budget closes it to
0.004 and is simply biased high. Overfitting vs systematic bias — constant's
better holdout number at L10 rests on a large generalisation gap.

**Leading suspect for the bias: the soft/hard gap** (members train scaled by
m, evaluate binary at full value, so a circuit averaging m~0.70 is driven
harder at eval than training assumed). Within the hot arms it is monotone
3/3: m_kept 0.801/0.737/0.702 -> overshoot 0.024/0.047/0.204. NOT a complete
story — constant at L10 has similar m_kept (0.717) and UNDERshoots — so treat
as the leading candidate, not the explanation. Three mechanisms already died
today (§3).

**Action this motivates: straight-through binarisation** (already queued as
the principled soft/hard fix) is now the highest-value next build, not a
nice-to-have. If the overshoot is the soft/hard gap, STB should keep the 4x
compression AND land on 1.0.

## 3c. THE COMPRESSION RESULT (2026-07-29): lambda sweep at hotflat lr

Independent variable is **lambda**, not lr. lr scales the data gradient, the
L1 term AND the decay together, so it cannot say which compresses; the L1
term is the only sparsifier (decoupled decay pulls theta->0 i.e. m->0.5,
regularising CONFIDENCE — which is why wd moves m_kept and lambda moves n).
Held at hotflat lr = 0.155363 so sum(lr) = 62.145 and the decay budget stays
pinned at 3.11 across every row; only the sparsity budget sum(lr)*lambda
moves (6.2e-3 -> 0.199, a 32x span).

| lambda | L10 n | L10 f0_h | L8 n | L8 f0_h | L2 n | L2 f0_h |
|---|---|---|---|---|---|---|
| (constant ref) | 108,068 | 0.9388 | 55,084 | 1.0142 | 1,689 | 0.9803 |
| 1e-4 | 26,450 | 1.2041 | 16,682 | 1.0474 | 1,111 | 1.0236 |
| 2e-4 | 15,267 | 1.1565 | 15,919 | 1.1232 | **623** | **1.0236** |
| **4e-4** | **9,124** | **1.0000** | **5,237** | **1.0284** | 444 | 0.9291 |
| 8e-4 | 4,487 | 0.7857 | 2,992 | 0.8057 | 293 | 0.8307 |
| 1.6e-3 | 2,928 | 0.5578 | 1,975 | 0.4953 | 280 | 0.8071 |
| 3.2e-3 | 1,907 | 0.3129 | 1,522 | 0.9716 (!) | 242 | 0.8346 |

**Knee at lambda = 4e-4 for the deep seeds**: L10 gives 9,124 members at
held-out free0 = **1.0000 exactly** — 11.8x smaller than the default AND more
faithful (0.9388 -> 1.0000). L8 gives 5,237 at 1.0284, 10.5x smaller for a
fidelity change of 0.014 -> 0.028 deviation. L2's knee is earlier (2e-4):
623 members at 1.0236, 2.7x smaller at comparable fidelity.

**The overshoot cures itself.** free0 falls monotonically with lambda (L10
1.204 -> 1.000 -> 0.786). The low-lambda overshoot was simply an
over-inclusive circuit over-driving the seed. This WEAKENS the soft/hard-gap
explanation in §3b — downgrade straight-through binarisation from "highest
value next build" back to a normal queue item.

**Design falsification test PASSED**: m_kept stayed 0.70-0.84 across a 32x
lambda span while n moved 14x. lambda removes members, wd sets confidence —
the separation holds. (Slight upward drift 0.702 -> 0.780 is consistent with
the weakest members being the ones removed.)

### The L8 lambda=1.6e-3 "anomaly" — EXPLAINED, and it generalises

The apparent anomaly was lambda=3.2e-3 giving n=1,522 at free0 0.9716, better
than lambda=1.6e-3's 1,975 at 0.4953. holdout_data_loss (the DATA term, so
comparable across lambda, unlike loss_final) inverts the reading:

| lambda | 1e-4 | 2e-4 | 4e-4 | 8e-4 | **1.6e-3** | 3.2e-3 |
|---|---|---|---|---|---|---|
| holdout data loss | 2.063 | 2.156 | 1.883 | 2.344 | **5.063** | 2.078 |

**3.2e-3 is not the outlier — 1.6e-3 is.** That run optimised genuinely
worse; its neighbour returns to trend. So this is NOT a lower minimum found
at high lambda; it is a WORSE one found at 1.6e-3.

Nesting test (L8): if sparsification were a smooth path, each higher-lambda
circuit would be ~a subset of the one below. It is not, anywhere:

| pair | n_lo | n_hi | containment | new at high lambda |
|---|---|---|---|---|
| 1e-4 -> 2e-4 | 16,682 | 15,919 | 0.4924 | **8,081 (50.8%)** |
| 2e-4 -> 4e-4 | 15,919 | 5,237 | 0.6943 | 1,601 (30.6%) |
| 4e-4 -> 8e-4 | 5,237 | 2,992 | 0.8021 | 592 (19.8%) |
| 8e-4 -> 1.6e-3 | 2,992 | 1,975 | 0.6906 | 611 (30.9%) |
| 1.6e-3 -> 3.2e-3 | 1,975 | 1,522 | 0.6682 | 505 (33.2%) |

Containment never exceeds 0.80 and 20-51% of each circuit is absent from the
one below. **Raising lambda swaps members, it does not just remove them.**
1.6e-3 is therefore not a special failure — EVERY step lands in a different
basin and that one landed badly. The cleanest cell: 1e-4 -> 2e-4 changes size
4.6% and data loss 4.5% while replacing 50.8% of the membership.

**Second independent probe of the same thing as the bf16 result** (§1: pure
rounding, 16.5% churn at equal free0). Organising claim: **aggregates are
smooth and reproducible, membership is not.** Size falls monotonically with
lambda and runs are bit-deterministic, so the compression stands; WHICH
latents make the cut is underdetermined. Consequence: the lambda=4e-4 knee
(L10 free0 exactly 1.0000) is partly basin luck — the size trend it sits on
is solid, the exact landing point needs multi-seed confirmation.
Data: lambda_nesting_L8.json.

**NOT YET A DEFAULT**: one seed per layer, and free0 only — no freeM_dense,
freeM_topk or cf. Needs the full eval matrix + multi-seed before adoption.

### Per-site structure of the compression (L10, hotflat lambda=1e-4)

32 upstream sites, d_sae 40,960. constant mean 3,377/site (8.2% of width),
hotflat mean 827 (2.0%). Pruning is NOT uniform — kept-fraction is U-shaped in
depth: 0.32-0.51 at L0, bottoming at 0.129-0.18 across L1-L5 mlp/resid, then
climbing to 0.607 at L9-attn and 0.724 at L10-attn (nearest the seed).
Attention survives 2-3x better than mlp/resid at the same depth (L1: 0.504 vs
0.166/0.179). hotflat is also far flatter than constant (mean 827 ~ median
838, most sites 600-1,100) EXCEPT L0, which keeps ~4x a typical site
(L0-mlp 3,333, L0-resid 2,106). Data: per_site_L10.json.

### Superseded lead: linear_up helps at depth

L8 -6.2% size and +2.3% holdout; L10 +2.3% size and holdout 0.9388 -> 1.0748.
Deep seeds are exactly where free0 historically collapses, so a flat-size
14% holdout gain at L10 is worth chasing. But it is one seed per layer, sits
in the small-denominator deep band, and reverses at L2. Needs a multi-seed
run before it changes any default.

### Original (pre-run) setup

Motivation (Daniel): circuits shrink as lr rises, suggesting a run-duration
/ evidence problem rather than a sparsity one; decay lets the mask settle
instead of letting the final step decide threshold crossings.

Engine now takes `lr_schedule` ("constant" | "cosine" | "linear") and
`lr_min_frac`. Both budgets scale with the lr **integral** — sparsity is
Σlr·λ, decay is Σlr·wd — so a decay-to-zero schedule halves Σlr at the same
peak. Peak lr is therefore doubled to hold both budgets fixed, making
schedule *shape* the only variable:

| arm | peak lr | Σlr | λ budget | wd budget |
|---|---|---|---|---|
| constant | 0.05 | 20 | 2e-3 | 1.0 |
| cosine → 0 | 0.10 | 20 | 2e-3 | 1.0 |
| linear → 0 | 0.10 | 20 | 2e-3 | 1.0 |

`lr_min_frac=0` is used. An earlier justification for a non-zero floor
("otherwise the tail is pure weight decay with no data gradient") was
**wrong** and has been corrected in the engine comment: the AdamW update is
θ ← θ − lr·grad − lr·wd·θ, so both terms vanish with lr and tail steps are
simply no-ops. A zero floor causes no drift and makes the budget match exact.

Runner: `schedule_sweep.py` (+ `launch_schedule.sh`), L8 → L2 → L10,
appending to `schedule_rows.jsonl`. Constant is **re-run** rather than
reused from earlier sweeps, since the code changed underneath it (bf16
codes, empty_cache).

Caveat to state when reading the result: decay redistributes the same
evidence, it does not add any. If the limit is really 400 steps × 64
probes, reshaping the schedule cannot fix it and the probe-count sweep
(128/256) is the actual lever. A null result should be read that way rather
than as "schedules don't matter".

## Tests

Full suite green at the time of pausing: 1,374 (563 `tests/circuit` + 811
rest), including 7 new `TestLrSchedule` cases (constant Σ = steps×lr;
cosine/linear halve it; budgets track Σlr; doubling peak restores the
constant budget).
