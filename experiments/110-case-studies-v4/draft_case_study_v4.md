# Draft: §"A Circuit for Classical Antiquity Has a Readable Core" on the production circuit

For the coordinator to review. Nothing in `paper/main.tex` has been changed. Every number below comes from
`out/case/9.resid.20419.json` (summarised in `out/case/9.resid.20419_numbers.md`).

The figure is `paper/figures/run2026_case_antiquity.pdf`, drawn at 5.5 in. If adopted, it replaces `case-study.pdf`.

## What changed, in one paragraph (for the coordinator)

**What survives.**
- The production (v4) circuit for 9.resid.20419 passes the held-out test (Z/A/C 0.91/1.08/0.98, necessity 1.00) with
  945 latents (old 728).
- It shares 390 latents with the old circuit (Jaccard 0.30). Those shared latents carry 70% of the new circuit's
  contribution, and the top of the ranking is almost the same.
- The story groups together still break the target: 0.22/0.10/0.21 against random 0.54/0.69/0.68.

**What moved.**
- The classical-Greece group alone is now only moderately necessary: 0.55/0.60/0.59 against random 0.66/0.85/0.83 (it
  was 0.29/0.22/0.32). Its share of the contribution halved, from 13% to 7%.
- The early-civilisations group (Egypt, Sumer, Babylon) took over as the most necessary group: 0.50/0.40/0.47 against
  random 0.68/0.89/0.84. It also feeds the classical group (edge 0.25).
- The generic core now matters under A as well (0.27), not only under Z.
- **New and positive:** run alone under contrast-mean ablation, the v4 circuit reproduces the date contrast. Every BC
  year in the sweep reads 75–109% of the reference and every AD year 12–21%; the old circuit gave 6.9 vs 3.8. The
  appendix sentence "its date route is the weaker one" is therefore no longer true.
- The circuit also carries the lexical "BC" route ("The BC Lions" 125% alone).

**Paper text that must change.**
- §5.3 body: the numbers, and the sentence on the classical group.
- Figure caption.
- The table row in `tab:case-studies`.
- The "More on the antiquity circuit" paragraph.
- The amplifier sentence. Under A the circuit lifts 323 latents (run median 497), with sibling faithfulness 0.84 vs
  target 1.08. Under C it is specific: siblings 0.44, controls 0.08, 109 lifted.
- The running-example sentence in the Setting (76% → 79% for "5th century BC"; the reference changed from 29.49 to
  28.47).

## Draft replacement for the §5.3 body (from "Its weighted circuit, …" to the end of the subsection)

> Its weighted circuit in the production run holds $945$ latents and passes on the held-out strongest contexts (Z, A,
> C $0.91$, $1.08$, $0.98$; necessity $1.00$). Grouping its largest contributors by their own top contexts gives a
> readable core (\cref{fig:case-study}a), and it is the same core we found in the earlier protocol-v1 circuit of this
> target: the two fits share $390$ latents (Jaccard $0.30$), but these carry $70\%$ of the production circuit's
> contribution, and $34$ of the earlier circuit's $40$ largest contributors are still members. Latents for early
> civilisations (Egypt, Sumer, Babylon; $6$ latents, $10\%$ of the contribution) feed latents for classical Greece
> itself (Athens, Sparta, Socrates; $9$ latents, $7\%$): removing the early-civilisation group lowers the classical
> group by $25\%$, and a group for dates and eras feeds the early-civilisation group more weakly ($9\%$). Removing the
> early-civilisation group takes the target to $0.50$, $0.40$, and $0.47$ under Z, A, and C, and removing the
> classical group to $0.55$, $0.60$, and $0.59$, against $0.68$, $0.89$, $0.84$ and $0.66$, $0.85$, $0.83$ when as
> many random members of similar contribution rank are removed (\cref{fig:case-study}d). Removing all $19$ story-group
> latents, $2\%$ of the circuit, takes the target to $0.22$, $0.10$, and $0.21$ (random: $0.54$, $0.69$, $0.68$).
> Latents for philosophy and rhetoric, the subject of the target's training contexts, matter no more than random ones.
>
> The circuit also reproduces the target's behaviour outside its fitting contexts. Run alone on the template sentences
> under contrast-mean ablation, where the empty circuit gives $0$, it drives the target to $75$--$109\%$ of its typical
> activation for every BC year from 5000 to 100~BC and to $12$--$21\%$ for every AD year (\cref{fig:case-study}b). For
> places it keeps ancient Greece and Athens ($127\%$, $123\%$) above Rome ($99\%$), Egypt ($72\%$), and medieval France
> ($39\%$).
>
> The core is a reading, not the whole circuit: the story groups carry $20\%$ of the contribution and $904$ further
> latents most of the rest, and ``BC'' alone partly triggers the target, in the model (``The BC Lions'' $45\%$) and in
> the circuit run alone ($125\%$). A generic core of seven layer-0 and layer-1 latents that fire on most text is also
> necessary (removing it gives $0.00$, $0.27$, $0.61$): with the rest of the stream ablated, these latents keep it in
> the range later layers expect. Under activating-mean ablation the circuit restores co-firing sibling latents to
> $0.84$ of their activation and lifts $323$ latents above the Top-K cut (run median $497$); under contrast-mean
> ablation it is specific (siblings $0.44$, controls $0.08$; \cref{app:specificity}).

(Keep the existing first paragraph of the subsection, but replace the clean-model numbers. "5th century BC" is now 79%
of the typical activation. The year sweep gives 42–55% for 800–200 BC, 19–32% for older dates, and 0% from AD 1000 to
1900, with 6% at AD 2000. Places: Athens 94%, Rome 53%, Egypt 34%, medieval France 0%.)

## Draft caption for `fig:case-study`

> A passing circuit has a readable core: the production-run circuit of a latent for classical Greek antiquity (layer
> 9, residual stream, $945$ latents; \cref{sec:case-study}). \textbf{(a)}~Role groups among the circuit's largest
> contributors, named from their latents' own top contexts, with each group's size, layers, and share of the circuit's
> contribution at the target's anchor. Arrows give the drop in the destination when the source group is removed, under
> activating-mean ablation; the two left-hand groups also lower the target directly, by $0.07$ (dates) and $0.34$
> (early civilisations). The dashed grey group fires about as much on contrast contexts. Not drawn: a generic core of
> $7$ latents and $904$ further latents, together $70\%$ of the contribution. \textbf{(b,c)}~The target on template
> sentences as a percentage of its mean activation on its held-out strongest contexts, in the unmodified model
> (lines, bars) and with the circuit run alone under contrast-mean ablation (black squares; the empty circuit gives
> $0$). \textbf{(d)}~Faithfulness with each group removed (bars) against removing the same number of random members of
> similar contribution rank (mean $\pm$ sd over $8$ draws); shaded, the pass band.

## Draft row for `tab:case-studies`

> 9.resid.20419 (\cref{sec:case-study}) & a date or place read as classical Greek antiquity & ``In the 5th century BC,
> the philosopher \ldots'' $79\%$; ``In the 19th century, \ldots'' $0\%$ & $0.22 / 0.10 / 0.21$ \newline ($0.54 / 0.69
> / 0.68$) & ``BC'' is partly lexical (``The BC Lions'' $45\%$) \\

## Draft "More on the antiquity circuit" (appendix)

> Two further measurements qualify \cref{sec:case-study}. First, the earlier protocol-v1 circuit of this target (728
> latents) put more weight on the classical-Greece group, whose removal alone took it to $0.29$, $0.22$, $0.32$; in the
> production circuit the same latents share that role with the early-civilisation group, which now feeds them. The
> readable core is stable as a set of latents across the two fits, but how the fitted coefficients divide the work
> among them is not. Second, the generic core was needed only under zero ablation in the earlier circuit; in the
> production circuit it is also needed under activating-mean ablation (A $0.27$).

## Uncertain points

- The groups are the 063 groups, restricted to the latents still present. A regrouping on the v4 member report would
  likely add some new early-civilisation latents to that group, for example:
  - 4.resid.3937 (cuneiform / Egyptian myth);
  - 5.resid.28071 (4000 BCE religion and writing);
  - 7.resid.23646 (cuneiform / Sumerian);
  - 6.resid.19509 (prehistoric).

  This was not done, so the comparison stays like-for-like.
- The knock-outs and probes were re-scored locally (RTX 5070 Ti). The full circuit reads 0.895 / 1.059 / 0.975 there,
  against the H100 row's 0.906 / 1.077 / 0.985.
- The mid-band row for this target is vacuous: the activating-mean read on the mid contexts is undefined. This target
  therefore has no mid-band evidence.
