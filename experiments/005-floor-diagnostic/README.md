# floor-diagnostic (2026-07-23)

*README generated from the scripts' docstrings; the scripts are the record.*

## `floor_check.py`

Decisive floor diagnostic: is the mean-floor denominator the source of the
L9 sign flip and the L10 inflation?

Every mean-floor metric (freeM, pinMC) normalises by  den = a_pos - a_empty(F),
where a_empty(F) is the seed's activation with the circuit EMPTY and every
upstream latent pinned to floor F. The 4-seed validation run showed pinMC
uniformly NEGATIVE at L9 (-0.12..-0.17) and inflated at L10 (~2.0) across ALL
arms -- abl, cf, restoration, act-grad, sfc alike. Those arms share exactly one
input: the posctx floor (gradient_base.py:591 and both eval call sites pass
pos_tokens). So the hypothesis is that at L9 the posctx mean already drives the
seed HARDER than the natural run does, i.e.

    a_empty(posctx) > a_pos   ->   den < 0   ->   every positive numerator
                                                  reports as negative.

This measures den under five floors WITHOUT touching the repo: collect_site_means
already takes a `tokens` argument, so the proposed negctx floor is just passing
neg_tokens (which ProbeDataset carries for every seed) instead of pos_tokens.

Decisive outcome:
  * CONFIRMS the diagnosis if den(posctx) < 0 at L9 and small-positive at L10.
  * VALIDATES the negctx floor if den(negctx) is comfortably positive on both,
    i.e. a_empty(negctx) sits well below a_pos.
If den(posctx) is healthy on both seeds, the floor is NOT the problem and the
negctx design should be shelved.

No discovery is run -- these are anchors only, ~5 forward passes per seed.

  PYTHONPATH=src python experiments/005-floor-diagnostic/floor_check.py

## `negctx_integration_check.py`

End-to-end check that floor_source="negctx" actually works on the live path.

The unit tests in tests/eval/test_ablation_faithfulness.py exercise
resolve_site_floors directly with stubs. They CANNOT catch the real integration
risk: gradient_base sets self._floor_neg_tokens inside _discover(), so a wiring
mistake there leaves it None and every negctx discovery raises — invisible to
any test that does not go through discover(). This runs the real thing.

Three checks, cheapest seed (L2, 8 upstream sites):

  1. ROUTING   — resolve_site_floors under floor_source="negctx" returns
                 exactly collect_site_means(neg_tokens), and NOT the posctx
                 means it was handed.
  2. PLUMBING  — a real discover() completes under negctx, proving
                 self._floor_neg_tokens is set and reaches the resolver.
  3. EFFECT    — the negctx circuit DIFFERS from the posctx circuit. If the
                 floor changed but attribution did not, the knob is decorative.

  PYTHONPATH=src python experiments/005-floor-diagnostic/negctx_integration_check.py

## Result files

`floor_anchors.jsonl`
