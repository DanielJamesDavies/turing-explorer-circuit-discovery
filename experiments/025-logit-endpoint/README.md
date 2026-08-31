# One method, two endpoints — latent vs behavioural circuits (2026-08-04)

Daniel: "could we apply abl-mask to logits like SFC's non-template
version, and see how those circuits do on logit faithfulness and by node
count?"

The SFC size comparison ("their circuits are 67-69 nodes, ours are
thousands") is confounded six ways: model, SAE, data, method, metric and
endpoint all differ. This holds five of them fixed and moves only the
endpoint.

## What was built

New `logit` objective in `src/circuit/instrument/learned_mask.py`:

    pos    loss = squared error of the SEED LATENT's pre-activation
                  against its natural value              (internal)
    logit  loss = squared error of log p(target token) at the same
                  anchor against the FULL MODEL's log-prob (behavioural)

Both REPRODUCE rather than maximise (the file's existing contract).
Supporting pieces: `_forward_logits` (full-depth masked forward — the
pre-act path can stop at the seed's layer, this cannot), `_target_logprob`,
`_natural_logprob`, a `target_tokens` argument, and a `tap_seed` flag on
`LearnedMaskPatcher` — with a behavioural endpoint there is no seed to
tap, so the seed's own site is masked like any other rather than exempted.
6 new tests (`TestLogitObjective`); suite 94 -> 100, all passing.

**Native scopes, deliberately different:**

    pos    upstream of the seed only. A latent endpoint can ONLY be
           driven from upstream, so that is its entire causal scope.
           L2: 8 sites / 327,680 slots. L8: 26 sites / 1,064,960.
    logit  every site, every kind, whole model (as SFC masks every
           submodule). 36 sites / 1,474,560 slots, independent of seed.
           The seed's posctx sequences supply only the data distribution.

Panels: 4 seeds each at L2 resid and L8 resid, lambda in
{1e-5, 1e-3, 1e-2, 1e-1}. A single lambda cannot compare the endpoints —
the house 1e-5 is fitted for the seed's pre-activation, which is far more
sensitive to upstream latents than the output distribution is — so the
comparison is read off size-vs-faithfulness CURVES.

## R1 — latent-endpoint compressibility is a SHALLOW-seed property

pos arm, mean over 4 seeds:

| lambda | L2 n -> free0 | L8 n -> free0 |
|---|---|---|
| 1e-5 | 2,432 -> **0.985** | 20,874 -> **0.903** |
| 1e-3 | 463 -> **0.897** | 1,304 -> 0.493 |
| 1e-2 | 123 -> 0.399 | 233 -> 0.102 |
| 1e-1 | 4 -> 0.039 | 50 -> 0.000 |

At the house lambda both depths sit at a near-identical FRACTION of
their scope (0.74% vs 0.90%) — L8's larger absolute count is mostly just
having 26 upstream sites instead of 8.

They compress differently. L2 gives up 9 points going 2,432 -> 463
nodes; L8 loses 41 points over the same 7x. Per-seed, 3/4 L2 seeds hold
>=0.90 at a few hundred nodes (2927: **121 nodes -> 0.857**), while 3/4
L8 seeds fall below 0.50 by ~1,300 nodes.

So the house lambda is over-provisioned at L2 (~5x) and roughly right at
L8. This NARROWS the l2-crossover verdict: Tests A and B there showed the
lambda=1e-5 circuit is not padded and not a probe-budget artifact **at
that operating point**, but neither tested whether a different sparsity
price finds a smaller genuine solution. At L2 it does.

Consistent with recursive-map RESULT 6b (closure at depth needed 5% of
the dictionary vs <0.5% at L2): depth genuinely requires more of the
stream, and that is not an artifact of the operating point.

## R2 — the endpoint effect, on COMMON denominators

Each arm was first scored against its own scope's empty-circuit
baseline. Those are the right natives but DIFFERENT denominators, so
`cross_scope.py` re-scores every L2 circuit under both:

    faith_up   circuit restored, only the 8 upstream sites mean-ablated
    faith_all  circuit restored, ALL 36 sites mean-ablated

| arm | lambda | mean n | faith_up | faith_all |
|---|---|---:|---:|---:|
| pos | 1e-5 | 2,432 | **0.684** | 0.060 |
| pos | 1e-3 | 463 | **0.467** | 0.070 |
| pos | 1e-2 | 123 | **0.277** | 0.102 |
| pos | 1e-1 | 4 | 0.074 | 0.009 |
| logit | 1e-5 | 141,650 | 0.969 | 0.552 |
| logit | 1e-3 | 10,498 | 0.771 | 0.059 |
| logit | 1e-2 | 2,509 | 0.569 | 0.106 |
| logit | 1e-1 | 743 | 0.239 | 0.074 |

**At matched size the LATENT-trained circuit carries the behaviour
better than the behaviour-trained one**: 2,432 nodes -> 0.684 against
2,509 -> 0.569; 123 nodes -> 0.277 against 743 -> 0.239. A circuit built
to reproduce an internal latent transfers to the output better than one
built on the output, at equal budget.

`faith_all` also explains why L2 and L8 logit numbers looked so
different: an upstream-only circuit reads 0.06-0.10 under whole-model
ablation **at any size** — it cannot carry the output when everything
above the seed is mean-ablated. That is the ablation scope, not the
endpoint. Only the 141,650-node logit circuit clears it (0.552).

## R3 — where the behavioural objective converges, its circuits SATURATE

L8 logit arm, per seed (36 sites, whole model):

| lambda | mean n | 1991 | 3005 | 4068 | 4468 |
|---|---:|---:|---:|---:|---:|
| 1e-5 | 126,702 | 0.640 | 0.508 | 0.644 | 0.216 |
| 1e-3 | 13,634 | 0.421 | 0.435 | -0.029 | -0.014 |
| 1e-2 | 2,536 | 0.349 | 0.567 | 0.042 | 0.065 |
| 1e-1 | 708 | 0.385 | 0.504 | 0.027 | 0.018 |

Seeds 1991 and 3005 are **flat across two orders of magnitude** —
0.35-0.57 from ~500 to ~80,000 nodes — while the L8 latent arm on the
same seeds needs ~10,000 nodes for 0.94 and reads 0.000 by ~300. A
~700-node whole-model circuit carrying 0.39-0.50 of the output is within
an order of magnitude of SFC's 67-69, which suggests their number is a
property of BEHAVIOURAL ENDPOINTS rather than of their method.

**But this is 2 of 4 seeds.** The split is bimodal, not continuous:
4068 and 4468 are near-zero or negative everywhere except the loosest
lambda. Negative faithfulness (worse than mean-ablating everything)
appeared on 3 of 8 seeds at intermediate lambda. That is the signature of
an optimiser failing, not of two kinds of seed — the lr/steps/anneal
schedule is the house fit for the PRE-ACTIVATION objective and has never
been fitted for this one.

## Verdict, and what it is not

The endpoint effect looks real: where the behavioural objective
converges, its circuits saturate at a few hundred latents while
latent-endpoint circuits at the same depth are at 0.000. That is the
SFC-shaped result reproduced with our method on our model, and it is the
first version of the size comparison with model/SAE/data/method held
fixed.

It is NOT yet a measurement. The logit arm fails to converge on ~1/3 of
seeds, is non-monotone in lambda, and goes negative. **Before any of this
is quoted: fit lr/steps/anneal for the logit objective on its own, then
re-run with more seeds.** Nothing here should be cited as a node-count
for behavioural circuits.

## Caveats

- L2 and L8 resid only; 4 seeds each.
- The logit arm runs the single ZERO floor: `dual_floor` is pos-only by
  construction in the engine, so the two arms differ in floor semantics
  as well as endpoint. Genuine asymmetry, not noise.
- Our behavioural endpoint is log p(next token) on natural text; SFC's is
  a logit DIFF between two answers on a contrastive task. A logit diff is
  a narrower, more concentrated quantity and probably easier to
  reconstruct — likely the largest remaining gap to their setup.
- Seed 3005's upstream denominator is 0.869 nats (near-degenerate); its
  pos-arm logit_faith should be ignored. Seed 4468's latent-arm
  logit_faith is negative throughout on a healthy 3.649-nat denominator.
- `matched_scope_rows.jsonl` holds a superseded earlier run where the
  logit arm was restricted to the seed's 8 upstream sites.

## Files

- `runner.py` — both arms, lambda sweep; `COMP_IDX` env selects the
  component (8 = L2 resid, 26 = L8 resid)
- `cross_scope.py` — re-scores every circuit under both scopes
- `rows.jsonl` / `members.jsonl.gz` (L2), `rows_c26.jsonl` /
  `members_c26.jsonl.gz` (L8), `cross_scope.jsonl`
