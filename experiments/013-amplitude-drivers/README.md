# D2.3 — Amplitude as a driver knob (2026-08-01)

No rediscovery: AR uses D2.2's archived restoration rankings, AC16 uses
D1's saved direct weights; alpha fitted with D1's exact AMPC recipe
(train store negatives, bisection to a_pos, 8.0 ceiling), cf_alpha on the
frozen held-out exam. 55 new rows in `rows.jsonl` + D1's 44 AMPC rows
pooled. Tables in `summary_tables.md`. Whole run ~7 min of GPU.

## The three pre-registered questions, answered

1. **Does amplification rescue small sets at depth? YES — depth is not
   the barrier, kind is.** 10/11 seeds reach cf_alpha ~ 0.91-1.09 at
   K=64; 8/11 already at K=16, including L11 (16 latents @ gain 1.16 ->
   0.97) and L8-resid (16 @ 3.70 -> 1.06). The failures are kind-shaped:
   deep MLP needs K=64 (L8-mlp: 16 @ 8.0 ceiling -> 0.30, but 64 @ 5.5
   -> 0.98), and L9-attn stays an identity failure at every K for every
   set (gain pegged 8.0, cf_alpha <= 0.29) — amplitude cannot buy what
   the ranking does not contain.
2. **The alpha*(K, depth) law is really alpha*(K, KIND).** Resid seeds
   decay 1.2-3.7 -> the 1.04 floor by K~256 regardless of depth (L11
   needs almost nothing at ANY K); mlp needs 2.4-8.0 and stays elevated;
   attn needs ~5 at K=16 where reachable at all. And the frontier's far
   end: at L0 K=1024, alpha* = 0.92 — big sets need DAMPING below
   natural, the overshoot regime D2.2 saw from the other side.
3. **The K-vs-alpha frontier is now a reportable table** (T1): "N
   latents at gain g" traded explicitly — e.g. seed 8/20333 is
   (16, 1.89) or (1024, 1.04); L8-mlp is unreachable at (16, *) but
   cheap at (64, 5.5). The literature reports neither axis.

## AR vs AMPC: a tie with one asymmetry

Medians are indistinguishable at every K (T2: 1.007-1.043 vs
1.000-1.039). Per-seed (T3) they agree within noise EXCEPT deep mlp at
K=16, where direct-mass wins outright (L8-mlp 0.96 vs 0.30) — D0.1's
"direct-mass wins small-K imposition at depth" reproduced under
amplification. Since AMPC is also ~free to compute (one fwd+bwd vs a
full restoration run), **AMPC stays the recommended driver-circuit
constructor; restoration's ranking adds nothing at driver budgets that
its greater cost justifies.**

## Caveats

- alpha ceiling 8.0: pegged cells (deep mlp K=16, L9-attn everywhere)
  are censored — "alpha* = 8.0" means "not reachable by 8x", not a fit.
- AR took entries in archive order incl. inhibitors; injectable count
  (n_injectable) drops below K where head members have no positive
  posctx pin (e.g. 593/1024 at L0) — the frontier's K axis is nominal
  budget, not injected count.
- Single 48/16 split per seed; no resampling error bars (D4.2's job).

## Tiny-K addendum (rows_tinyk.jsonl, runner_tinyk.py — K in {1,2,4,8})

SINGLE-LATENT drivers exist: 5/11 seeds impose at cf_alpha >= 0.8 with
K=1 (L2 1.08@7.2, L5 1.00@7.8, L6 1.01@7.0, L9-resid 0.90@8.0, L11
0.97@5.0). Minimal amplified driver by kind: resid 1-4 at EVERY depth;
mlp 16 (L4) / 64 (L8); attn bimodal (L3 at 2; L9 never). Feeds D4.1
(014-d4-analyses/d41-driver-size-vs-depth.md). Caveat: K=1 cells
at the 8.0 ceiling are censored — higher gain might rescue more seeds.
