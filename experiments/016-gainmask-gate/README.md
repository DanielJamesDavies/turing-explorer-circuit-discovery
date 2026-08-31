# D3.7 gate test — per-latent ceilings vs uniform alpha (2026-08-01)

Fixed AMPC sets (D1 direct-mass, identical pin>0 inclusion), K in
{8,16,64}, 11 seeds. Per-latent ceiling = the member's MAX activation
over the seed's 48 train posctx probes (in-distribution, in-MECHANISM
anchor). Arms: BETA (one scalar interpolating pin -> ceiling, fitted on
train) and MAXINJ (everyone at ceiling), vs AMPC's uniform-alpha rows.
66 rows in rows.jsonl.

## VERDICT: the in-distribution cap is REFUTED at driver budgets

- At K=8/16, BETA is censored at beta=1.0 on 8/11 seeds and LOSES to
  uniform alpha, often badly (L0 K=8: 0.54 vs 1.04; L2 K=8: 0.30 vs
  1.07; L4-mlp K=16: 0.27 vs 0.84). The reason is in the headroom
  column: the ceiling sits only 1.2-2.9x above the posctx mean, but the
  gains that make tiny drivers work are 2.5-8x. **The amplification
  that drives seeds at small K is genuinely SUPER-NATURAL — beyond
  anything the member latents do in the seed's own firing contexts.**
- At K=64 (alpha* <= ~1.6) BETA ties AMPC — the cap only binds where
  amplification matters.
- MAXINJ at K=64 overshoots to 1.4-2.5 on strong seeds: per-latent
  ceilings with no fit are not a free lunch either.
- L9-attn: 0.0 everywhere despite 8-13x headroom — identity failure
  yet again, now also robust to per-latent amplitude shaping.

## What this kills and what survives of D3.7

KILLED: the m in [0,2] gain-mask CAPPED at an in-mechanism ceiling —
at driver budgets it is strictly weaker than AMPC's one scalar, because
its whole amplification range sits below what is needed.
SURVIVES (untested): (a) Daniel's ORIGINAL anchor was the latent's
GLOBAL topctx max (its peak over its OWN top contexts corpus-wide),
which is looser than the in-mechanism ceiling tested here and may have
the required headroom — at the cost of weaker naturality semantics;
(b) unbounded per-latent gains (a learned AMPC), whose target would be
the two uniform-alpha failures: L4-mlp (0.72-0.96 at pegged alpha=8)
and deep-mlp small-K. Neither is worth building unless those specific
cells matter to the paper.

## The positive finding (paper-relevant)

Driver amplification is not "restore members to values they take
somewhere in the mechanism" — the working gains exceed the members'
observed in-mechanism range by 2-5x. AMPC's alpha is doing something
genuinely counterfactual, not just compensating for probe averaging.
This sharpens the "engineered state" caveat on the L11/K=1 headline and
belongs next to it in the paper.

## SECOND GATE (2026-08-02): the GLOBAL topctx-max ceiling — also fails

The first gate used an IN-MECHANISM ceiling (max on the SEED's own
contexts). Daniel's original proposal named a looser anchor: each
latent's GLOBAL peak over the contexts where IT fires hardest. Measured
directly (load each member's own top contexts, take its max there):

| seed | global ceiling / posctx pin | K=8 | K=16 | K=64 | AMPC alpha |
|---|---:|---:|---:|---:|---:|
| L4-mlp | 2.1x | 0.207 | 0.288 | 0.592 | 3.46 |
| L8-mlp | 5.1x | 0.181 | 0.348 | 0.711 | 2.85 |
| L8-resid | 2.1x | 0.523 | 0.584 | **1.013** | 1.64 |
| L11-resid | 1.6x | 0.972 | **0.993** | **0.993** | 1.04 |

(cf at the fitted beta; beta = 1.0 means the ceiling was not enough to
reach the target, i.e. CENSORED.)

The global ceiling IS looser than the in-mechanism one (median 1.6-5.1x
vs 1.2-2.9x) but still **fails on the cells that matter**: at K<=16 it
is censored at beta=1.0 on 3 of 4 seeds and reaches only cf 0.18-0.58
where uniform alpha reaches ~1.0. It succeeds only where amplification
was never needed (L11, alpha 1.04) or at K=64 on the easier resid seed.

**D3.7 IS NOW FULLY CLOSED.** Both proposed ceilings are refuted for
small-K drivers, for the same reason: the required gains are
SUPER-NATURAL — larger than the member latents ever reach anywhere in
the corpus, not merely larger than they reach on this seed's contexts.
The third variant Daniel named — UNBOUNDED per-latent gains — is
already built and evaluated under another name: it is exactly the
cf-mask's delta half (delta = softplus(psi), unbounded, per-latent; see
020-cfmask and 020-cfmask). No part of the
gain-mask proposal remains untested.
