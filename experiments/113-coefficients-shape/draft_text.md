# Draft text for the paper (113). Daniel merges; nothing here edits main.tex

Every number has its source in brackets. Paths are relative to `experiments/113-coefficients-shape/out/` unless
they start with `paper/`.

## Coefficients subsection (main text; candidate figure `paper/figures/run2026_coef_alpha_hist.pdf`)

**What the coefficients look like.** Across the non-vacuous held-out-strongest production circuits (7,464 TuringLLM,
508 Pythia-70m, 511 GPT-2 small; 5.49 M, 249 k and 256 k members), the fitted α concentrate near one with a right
tail (Figure X(a)).

- Medians: 1.11, 1.01 and 1.02.
- Within the median circuit: p10 0.79, 0.71 and 0.68; p90 2.23, 1.79 and 2.21; the largest α is 4.4, 3.6 and 4.7.

[run2026_coef_alpha_summary.csv; paper/tables/run2026_coef_alpha_summary.tex]

**The fit mostly amplifies.**

- Share of members with α > 1: 67.9% [67.5, 68.2], 57.0% [56.4, 57.7] and 58.3% [57.0, 59.6].
- Over 90% of circuits in each model amplify a majority of their members.
- Near-switch-offs are rare: α < 0.4 for 2.4%, 2.2% and 2.9% of members.

[run2026_coef_alpha_summary.csv: share_gt1, circ_majority_gt1, share_lt04]

The coefficients therefore act as gains on members that keep their sign, not as a gate that turns members off. This
matches the collapse at α = 1 reported in Table Y (88.0% → 1.8% on TuringLLM) [109 run2026_alpha1.csv].

**Where α departs from one.** It depends most on how often the member fires.

- Members in their site's top corpus-frequency decile make up 75% of TuringLLM members and 63–65% on the public
  models. They stay near one: median α 1.07, 1.00 and 1.00.
- Rarer members are amplified more: 76–79% have α > 1 on TuringLLM and 70–74% on GPT-2.

[run2026_coef_alpha_by_freq.csv]

Dependence on layer distance is weaker. On TuringLLM, the share with α > 1 rises from 63–65% at distance 0–1 to 72% at
distance 3–4, then falls to 64% at 8–11 [run2026_coef_alpha_by_distance.csv; Figure X(b)].

## Appendix paragraph: band, overlap, and what is not measured

**Band symmetry.** The production band [0.75, 1.5] is not symmetric in log space. Two alternatives:

- **Wider lower edge, [1/1.5, 1.5]:** pass rates rise by 4.0 pp [3.6, 4.5] on TuringLLM (to 92.0%), 5.1 pp [3.1, 7.1]
  on Pythia and 5.3 pp [3.5, 7.2] on GPT-2.
- **Tighter upper edge, [0.75, 1/0.75]:** changes almost nothing. TuringLLM drops 0.4 pp [0.3, 0.6] (32 circuits);
  Pythia and GPT-2 are unchanged.

So circuits fail by under-recovering the target, not by overshooting it. The deep band is most sensitive to the lower
edge: TuringLLM L8–11 gains 7.8 pp and GPT-2 L8–11 gains 12.0 pp. [run2026_coef_band.csv;
paper/tables/run2026_coef_band.tex]

**Weighted and unweighted circuits share a core but are not nested.** We paired, on the 173 scored TuringLLM sweep
targets, the smallest passing WCM circuit with the smallest passing unweighted circuit; 142 targets have both.

- On average 81% [79, 83] of WCM members also appear in the unweighted circuit, which is 6× larger (median 1,793 vs
  284 members).
- Only 3.5% of WCM circuits are strict subsets of their unweighted circuit. Jaccard: 0.13 [0.12, 0.15].
- At the same λ = 10⁻³ the WCM-in-unweighted share is 0.59 [0.57, 0.62] and the Jaccard is 0.26 [0.24, 0.27].

[run2026_coef_overlap.csv; paper/tables/run2026_coef_overlap.tex]

The extra unweighted members are not near-duplicates of kept WCM members. For each one we took the largest decoder
cosine to a WCM member at the same site:

- Median: 0.072, against 0.068 for random latents of the same site.
- 1.1% exceed 0.3 (random latents: 0.1%), and none exceed 0.7.

[run2026_coef_overlap.csv: cos_uonly_median, cos_random_median, uonly_gt0.3, random_gt0.3]

Unweighted masking therefore buys faithfulness with additional, distinct latents, not with redundant copies of the WCM
members.

**Not measured from the stored artefacts.** These need a forward pass (113 README §2, §3, §5):

- whether α times a member's anchor activation leaves the latent's natural range;
- the per-context spread of Z, A and C (eval rows store ratios of means over the 16 held-out contexts);
- a contribution-weighted effective size. Circuits store α and the mask gate, neither of which is a contribution.
