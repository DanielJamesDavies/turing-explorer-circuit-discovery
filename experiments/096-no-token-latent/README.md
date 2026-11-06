# 096 — A latent with no token expression? (Linear DAN-151, "wanted" example)

**Question.** DAN-151 wants one verified example of a circuit target whose role is internal and which has no token expression, to back the sentence "our targets can capture variables that never show up as a token". The candidates were the three internal-looking case-study latents from 063: the negation pivot 9.resid.7640, the causal "but also" 8.resid.31095, and importance as a comparison dimension 9.resid.17596. The output-facing reference is the eponymous-law latent 10.resid.15497, which pushes "Law"/"law"/"Equ".

**Rule (from the issue).** A candidate qualifies only if (a) its direct effect is negligible AND (b) its promoted tokens show no coherent token family.

**Answer: none qualifies.** All three are strong direct writers with coherent promoted-token families. Do not use any of them as the "no token expression" example. The paper should keep the population-level evidence from 064 only, and not name an example.

## Method

`direct_effects.py` (CPU, a few seconds). It runs no model forward. It reads the static per-target direct effects that 064's `target_effects.py` already computed: the decoder direction through the final RMSNorm (linearised at `rms_typ` = 4.53) to the unembedding, centred over the real vocabulary. The same quantity is behind `circuits_cli.py target`.

- `boost@peak` = `l_max` × the target's peak activation: the nats by which the top promoted token gains over the average token, at the latent's strongest context. `boost@mean` uses the latent's mean activation instead.
- `supp@peak` = `l_min` × peak: the same for the most suppressed token.
- Percentiles are over all 15,004 targets and over targets at the same site.
- `z1` = `l_max / sd(l)` is token focus. It is compared with the 95th percentile of the same-site random null.

## Results

| latent | role | pass | peak act | boost@peak (nats) | boost@mean | supp@peak | z1 (site-null p95) | ctx_z | percentile (all / site) |
|---|---|---|---|---|---|---|---|---|---|
| 9.resid.7640 | negation pivot | yes | 34.3 | **3.19** | 0.49 | −1.46 | 6.0 (10.0) | 1.80 | 97 / 82 |
| 8.resid.31095 | causal "but also" | yes | 25.1 | **2.42** | 0.32 | −1.51 | 7.7 (8.7) | −0.82 | 95 / 80 |
| 9.resid.17596 | importance comparison | yes | 41.5 | **3.17** | 0.33 | −2.03 | 6.6 (10.0) | 0.62 | 97 / 82 |
| 10.resid.15497 | eponymous law (reference) | yes | 39.5 | **4.20** | 0.63 | −1.97 | 9.3 (9.7) | 0.72 | 99 / 90 |

**Population.**
- The median target has boost@peak 0.29 nats and boost@mean 0.06 nats.
- The median L8–10 resid target has boost@peak 1.46 nats, so deep resid latents write strongly in general.
- 064's "median ~0.04 nats" is a different quantity: the per-position DLA to log p(top-1) of the best fired circuit at inference. It is not a per-latent property, and should not be compared one-to-one with the numbers above.

**Promoted / suppressed tokens** (top 10, logit per unit activation, centred):
- **9.resid.7640:** promotes `it` +0.09, `no` +0.07, `though` +0.07, `although` +0.07, `sometimes`, `more`, `son`, `oh`, `both`, `training` (+0.06 each). It suppresses scattered subwords (`ribute`, `img`, `independ`, −0.04). This is a **concessive / negation family** (no, though, although), so the latent anticipates its own pivot tokens.
- **8.resid.31095:** promotes `by` +0.10, `from` +0.07, `durch` +0.07, `picked` +0.06, `upon`, `from`, `through`, `by`, `depends` (+0.05). Its suppressed tokens are scattered. This is a **coherent agent / cause-preposition family** (by, from, durch, through, upon, depends): it predicts the "caused by / depends on" continuation.
- **9.resid.17596:** promotes `ever` +0.08, `quantity` +0.07, `sword` +0.07, `taste` +0.06, `quality` +0.06, `weapon`, `anything`, `technical`, `communication`. It suppresses first names (`Anto`, `Edd`, `Andrew`, `David`, about −0.04). This is a partly coherent **dimension-noun family** (quantity, quality, taste, technical), mixed with noise.
- **10.resid.15497** (reference): promotes `Law` +0.11, `law` +0.09, `Equ` +0.08, `Second`, `laws`, `Princi`, `Circ`.

## Verdict per candidate

- **9.resid.7640: fails both (a) and (b).** Its direct effect of 3.2 nats at peak is the 97th percentile of targets and three-quarters of the reference's. Its promoted tokens are the negation/concession family.
- **8.resid.31095: fails both.** 2.4 nats at peak (95th percentile), with a clean preposition family.
- **9.resid.17596: fails (a)** at 3.2 nats (97th percentile). Its family is weaker but present (quantity/quality/taste).

The three are no less output-facing than the eponymous-law latent. Their z1 is below the site null's 95th percentile, so they are broad writers rather than single-token writers. This matches 064's finding that deep resid latents as a group write strongly to the unembedding.

## Implications for the paper (DAN-151)

- Do not name an example of a variable with no token expression. Phrase the core sentence as a capability ("can capture variables that need not show up as a token"). Support it with the population evidence only: 064's small per-position DLA of fired circuits, and "no latent checks whether a fact is correct".
- A better hunt for a true example would screen targets with boost@peak below the population median (< 0.29 nats). It would also need z1 below the null, and an internal role validated by the 063 tools. That is not done here.

## Caveats

- These are direct effects only, at the latent's peak activation (its strongest context; at mean activation they are about 7–10× smaller). A small direct effect would not rule out indirect effects; a large one, as here, does show the latent can be expressed as tokens.
- The final norm is linearised at a typical scale (rms_typ). Inference DLA (064 `inference_dla.py`) uses the exact Jacobian per position.
- Token labels are decoded from the Phi-3 tokenizer. Repeated strings (for example `from`, `by`) are distinct token ids with and without a leading space.

**Reproduce:** `PYTHONPATH=src python experiments/096-no-token-latent/direct_effects.py` → `results/direct_effects.{json,md}`.
