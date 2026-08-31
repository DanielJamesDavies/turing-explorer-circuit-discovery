# direct-drivers (2026-07-24)

*README generated from the scripts' docstrings; the scripts are the record.*

## `direct_parents.py`

How many DIRECT PARENTS does a deep seed actually have?

The residual passthrough gives every upstream latent a nonzero unmediated path
into the seed, so "direct parent" is a continuous weight, not a discrete edge.
This measures whether the direct-effect mass has a natural cutoff:

  1. concentration — members needed for 50/90/99% of total |direct| mass
  2. layer profile — where direct mass lives vs where MEMBERSHIP lives
     (the chain intuition: direct parents hug the seed, ancestors sit deep)
  3. direct-vs-attribution — rank agreement; a member with high attribution
     but ~zero direct weight is a MEDIATED ancestor
  4. sufficiency at the natural sets — keep the 50%/90%/99%-direct-mass sets,
     pinned and free, with size-matched random controls

Also SAVES the per-member direct weights (direct_weights_{tag}.pt) so this
never needs the GPU again.

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/direct_parents.py

## `directedge_runner.py`

Direct-edge circuits: keep only the latents that DIRECTLY drive the seed.

In a residual architecture every upstream latent reaches the seed via
  (a) the IDENTITY path — its decoder write persists in the residual stream
      and the seed's encoder reads it: strength = a_j * (W_dec[:,j] . w_seed)
      — pure geometry x clean activation, no forward pass needed; and
  (b) MEDIATED paths — intermediate attn/mlp react to it.

Attribution conflates the two. This experiment selects members by the DIRECT
strength alone and asks the user's question: pin those members to their clean
values, ablate everything else (zero fill), and see what the evals say.

Selectors at matched K (16..4096):
  direct — top-K by |pins_j * (W_dec_j . w_seed)|   (no discovery involved)
  attr   — top-K by |attribution| from the saved abl-ig_mean PA raw circuit
  rand   — K uniform upstream latents (null), one draw, seed=0

Evals per set: pinned0 (pin members, zero-fill rest — the DIRECT question) and
free0 (members re-encode live — do direct edges have any closure?).

Also reported per seed: the ANALYTIC direct sum — sum over ALL upstream
latents of pins_j * dot_j, vs a_pos. If the identity path were the whole
story, that sum (plus error/embedding terms we do not model) would predict
a_pos. The gap is the mediated share.

  PYTHONPATH=src python experiments/007-direct-drivers/runner.py

## `edge_dist.py`

Distribution of direct-effect edge weights onto the seed: how many members
have GENUINE direct edges, vs tail noise the top-K sweep dragged along?

Reports, per seed: quantiles of |w|, top-k mass shares, counts above relative
thresholds, and the participation ratio (sum w)^2 / sum w^2 — the standard
"effective number of contributors" (equals N for uniform weights, 1 for a
single dominant one).

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/edge_dist.py

## `runner.py`

Direct drivers: are the seed's DIRECT-EFFECT edges a sufficient sub-circuit?

For a saved rec2+mag circuit (abl-ig_mean PA), compute each member's
direct-effect edge weight onto the seed — SFC's edge construction, restricted
to the one target we care about:

    w(u -> seed) = grad_{u, stop(M)}(seed_pre_at_probe) * u_natural

computed with SAEGraphInstrument: every site's feature code enters through a
DETACHED leaf anchor, so the backward from the seed's pre-activation reaches
each anchor only along paths free of other feature nodes — the gradient IS the
unmediated direct effect (u_baseline = 0, free0-coherent). One instrumented
forward + one backward, member->member edges never built (that all-pairs loop
is what OOMed in July; the seed-directed slice was always cheap).

Then the eval the question asks for: keep only the top-K members by |direct
edge|, zero-ablate every other latent (free0 semantics, live re-encode), and
sweep K. Controls at every K:
  attr   — top-K by |attribution| (the existing driver ranking)
  rand   — random K members (size-matched null)

  SEED_TAG=L10 PYTHONPATH=src python experiments/007-direct-drivers/runner.py

## Result files

`parents_rows.jsonl`, `rows.jsonl`
