# Paper prep for the pod results, 2026-10-24

What changed in `paper/main.tex` on 2026-10-24 to get ready for the 8×H100 results (DAN-157, DAN-159/77/43,
DAN-181, DAN-40/41, DAN-182). Related Work and `app:related` were not touched; another agent owns them.

Build: `paper/build.ps1`. 0 undefined references. **The main body ends on page 12 with no lines to spare.** Any
growth in the body (for example real numbers replacing a short `\pending`) will push it onto page 13.

## Placeholder count

- `\pending{...}` uses: **16 before, 22 after.** That counts every `\pending{` in the file, including the one at
  L186 in Related Work, which was left as it was.
- New macro: `\pendingcell`, a red "?" for table cells, defined in the preamble. Table 2 uses it through a local
  `\pendrow` macro, on 18 rows of 8 cells. A grep for `\pending` also finds `\pendingcell`.

### Removed or replaced (13)

| old line | old text | now |
|---|---|---|
| 83 abstract | `Pythia-70m and GPT-2 small, DAN-40/41` | merged into one marker covering DAN-157, DAN-181 and DAN-40/41 |
| 126 intro | `Pythia-70m and GPT-2 small, DAN-40/41` | `and to 512 each of Pythia-70m and GPT-2 small, DAN-40/41` |
| 187 Setting | `public-model runs, DAN-40/41` | removed (the "Beyond TuringLLM" paragraph covers it) |
| 346 Setup | `Pythia-70m and GPT-2 small, DAN-40/41` | removed (as above) |
| 379 random circuits | `the same baseline on Pythia-70m and GPT-2 small` | folded into the harder-null marker (all three models) |
| 404 Beyond TuringLLM | `full-run results, DAN-40 and DAN-41` | a new, more specific marker |
| 420 sec:weighted | `the same comparison on Pythia-70m and GPT-2 small` | folded into the Beyond TuringLLM marker |
| 470 matched size | `the matched-size comparison ... DAN-159` | `verdict, DAN-159 and DAN-77` |
| 527 Recommendations | `Check held-out sets for duplicated text (DAN-157).` | replaced by the recommendation itself |
| 746 tab:protocol | `Pythia-70m and GPT-2 small: 1,024 targets each ... DAN-9` | replaced by the decided frame (512 each) |
| 852 app:method-comparison | `the protocol-v1 comparison on the 190 sweep targets, DAN-159` | removed with the pre-v1 table |
| 877 tab:matrix caption | `the aggregate-ranking rerun in the v1 comparison suite` | removed; the aggregate ranking is now a row |
| 884 three results | `the aggregate-ranking rerun` | removed; replaced by the three new markers below |

Kept unchanged: L174 (Related Work), and the two DAN-48 markers (attribution appendix, sec:sfc-boundary).

### New markers (current line numbers)

Main body:
- 88 (abstract): `rebuilt held-out re-score, DAN-157; harder random nulls, DAN-181; Pythia-70m and GPT-2 small, DAN-40/41`
- 133 (intro): `and to 512 each of Pythia-70m and GPT-2 small, DAN-40/41`
- 141 (intro): `harder nulls that test the selection itself, DAN-181`
- 371 (Setup): `re-scores on the rebuilt sets, DAN-157`
- 383 (main result): `rebuilt held-out headline and worst case with intervals, DAN-157`
- 407 (random circuits): `harder-null pass rates on all three models, fitted and at alpha = 1, DAN-181`
- 437 (Beyond TuringLLM): adopted λ, pass rate with interval, median size by band and kind, α = 1, unweighted on
  Pythia (DAN-40/41)
- 506 (attribution at matched size): `verdict, DAN-159 and DAN-77`

Appendix:
- 807 tab:protocol: `adopted lambda per public model, DAN-42`
- 847 app:heldout: below-floor counts, share and strength of scan rows, headline and worst case (DAN-157)
- 848 app:heldout: the paired as-run vs rebuilt pass rate on the 1,000-target subset (DAN-157)
- 900 Exact activation patching: exact-ranked circuits on the rebuilt sets (DAN-159)
- 941 fig:size-comparison: the DAN-77 figure (a framed placeholder; swap in `dan77-faithfulness-vs-size.pdf`)
- 983 tab:matrix caption: every cell, plus the exact-patching rows (DAN-159)
- 988 / 989 / 990: the three comparison questions (DAN-159, DAN-43)
- 1181 app:nulls: the full harder-null result list (DAN-181)
- 1192 fig:coef-triad caption: the null bars in both panels (DAN-181)

### Where the 087 interval macros slot in

These numbers are still as run (original held-out sets). Each place has a `%` comment saying so:
- the abstract (46%, 1.5% / 45%, 747);
- intro L139 (46%) and the contributions list (46%, 1.5% / 45%);
- the main-result sentence L383 (46%; also 494 / 386 / 91% and the site and depth rates in that paragraph);
- sec:weighted (44.7% / 1.5%) and the budget-curve numbers;
- the Conclusion (46%, 747);
- app:production, app:specificity, app:weighted-details and the triad paragraph.

The `\pending` at L383 is the anchor for `\passall` and its worst-case twin.

## Changes by section

1. **Held-out rule (DAN-157).**
   - Method, "Targets and contexts": one sentence giving the rule and the floor, citing `app:heldout`.
   - tab:protocol:
     - "deduplicated" corrected to "repeated corpus ids removed but not repeated text";
     - "472" kept as "as split, 472 had", with a comment that 085 counts 471;
     - a new **Held-out rule** row;
     - the contrast row gains the contrast held-out rule.
   - New `\subsection{Held-out Contexts}` (`app:heldout`) at the end of app:protocol. It covers:
     - the audit: 66.8% of rows repeat a training-side row (35.6% exact, 50.2% anchor-prefix, 55.1% near);
     - the cause: 52.9% of windows were written twice, twins sit at adjacent ranks, and the split straddles them;
     - mid-band nearly clean;
     - the rebuild: median 5 of 16 survive; a 512-row scan over 18.2M unique windows; firing checked twice; mid-band
       only filtered; contrast refilled from the selector;
     - below-floor reporting: headline over targets with ≥ 8 clean rows, worst case counts the rest as fails;
     - the paired 1,000-target subset;
     - the group-aware split on the public models.
   - Setup: one sentence saying the TuringLLM numbers are still as run.
   - Recommendations: the DAN-157 `\pending` became a fifth practice.
   - **No source-file rule** anywhere.
2. **Attribution baselines appendix.**
   - Rewritten:
     - shared intro with the pre-activation definition;
     - `\subsection{Estimators}` (keeps the labels `sec:modes`, `sec:cfgrad`, `sec:abgrad`, `sec:hybrid`): AtP,
       IG from the mean (IE_ig / node-level EAP-IG, `hanna2024faith`), Ge hierarchical, and ranked members with
       fitted coefficients (IG, activation, co-activation);
     - a new **Exact activation patching** paragraph with the numbers from 083 `exact_report.log`: ρ AtP 0.86 vs
       IG 0.71 (magnitudes 0.81 / 0.62); top-n overlap 79% vs 54%; max ranking 0.72 / 0.62; the candidate subset
       caveat. These are training contexts, so the rebuild does not affect them;
     - `\subsection{Rankings and Node Sets}` with the three rankings; the signed sum is "the aggregate ranking of
       prior work" with a citation to `marks2025sparse`.
   - Removed: the contrastive and ablation gradient objectives and the restoration mode.
   - Fixed: the "as attribution patching does" sentence (the library's `local` mode is not AtP; see the `%` comment).
3. **Method comparison appendix.**
   - Retitled "Method Comparison across Circuit Sizes". The text is curve-first.
   - New placeholder figure `fig:size-comparison` (DAN-77).
   - `tab:matrix` is now an 18-row skeleton:
     - natural operating point (WCM and U at λ 1e-3, AtP / IG position union, Ge);
     - truncations (AtP and IG × max / mean|score| / aggregate);
     - ranked members at α = 1 and fitted.
   - The three-results paragraph became three questions, each with a `\pending`. The pre-v1 verdict is mentioned
     once, labelled. Home-turf is kept.
   - Attribution-mass recovery is kept, retitled "(pre-v1)", and now refers to the retired panel.
   - Main text: "Attribution at matched size" now gives one sentence plus the exact-patching ρ, and a `\pending`.
4. **Random circuits (DAN-181).**
   - sec:eval: "Random-circuit baselines" is one sentence on the three member pools, citing `app:nulls`.
   - Results:
     - the inert-coefficient caveat (median α exactly 0.798, decay only; fails on training contexts);
     - the cut of "properties of the selected node set";
     - a `\pending` for all three models, fitted and at α = 1.
   - Intro: the "what passes is the selected node set, not the fitting" clause was cut, and a `\pending` added.
   - New `\paragraph{Harder random circuits.}` (`app:nulls`) in app:weighted-details. It gives:
     - full definitions (WIN 8, frequency decile, contribution bins from the members' own deciles; anchor keeps
       members, contrib excludes them);
     - α = 1 as the Shi et al. / CircuitKIT null;
     - the panel: 126 + 128 per public model, a second draw on 32.
   - A new tab:protocol row, "Harder random circuits".
   - fig:coef-triad caption: the caveat and a `\pending`.
5. **Public models (DAN-40/41).**
   - "Beyond TuringLLM":
     - "unchanged" dropped;
     - one sentence on the frame, the λ rule and TF32;
     - one `\pending` that absorbs the old L420 and L404 markers.
   - tab:protocol:
     - the targets row gives the frame: 512 each, seed 20261024, ≥ 200 windows, hub cap 50% (would remove 1.75%
       of TuringLLM targets), and a pilot of 30;
     - new **Corpora** and **Precision** rows;
     - the λ pilot rule is added to the sparsity-penalty row.
6. **DAN-182.**
   - sec:drivers-closure: a `%` comment marking it TO BE RERUN, with the bug.
   - Its first sentence now says "pre-v1, and so are all of its numbers ... an earlier version of the attribution
     baselines".

## For the references.bib owner

There are no bib entries for CircuitKIT, the Pile, OpenWebText or Pythia.
- CircuitKIT is named in `app:nulls` without a citation; there is a `% TODO` there.
- The Pile and OpenWebText are named in tab:protocol without a citation.

## Removed text (verbatim, for the record)

### app:attribution intro, objectives and modes (replaced)

```latex
WCM is compared against rank-and-cut attribution.
Every upstream latent is scored by a first-order estimate of its effect on the target: a \emph{contrastive} objective on contrast contexts, taken through the target's pre-activation because Top-K censors its activation to zero there, and an \emph{ablation} objective on activating contexts, where the activation is live.
The comparison of \cref{app:method-comparison} uses two linearisation modes: \emph{path attribution}, an integrated-gradients estimator in the style of sparse feature circuits~\citep{marks2025sparse} taken jointly over sites from their mean-ablation values, and \emph{restoration}, which re-linearises as the circuit is rebuilt (\cref{sec:modes}).
Scores are cut per position and the node set is the union over the target's causal prefix, since a Top-K code is sparse per position while the target reads every position.
The subsections below give the objectives, the modes, and the node-set construction.

\subsection{Attribution Objectives}
Two gradient objectives supply the signed scores, one for each side of the target's behaviour.
\paragraph{Contrastive gradient: why does the target not fire?}
[... pre-activation definition (kept) ...]
On a contrast context $x$, the objective is the squared gap $\Lcf(x) = \bigl(\preact(x, t^{*}) - \mu_f^{+}\bigr)^{2}$ to the target's mean pre-activation over activating contexts, taken at the position of maximum pre-activation, and each upstream latent $g$ present in a contrast context is scored by gradient-times-activation averaged over contrast contexts:
$\Scf = \frac{1}{|\negset|} \sum_{x} a_g(x, t_g) \, \partial \Lcf(x) / \partial a_g(x, t_g)$.
A latent absent from a contrast context has $a_g = 0$ there, so it is scored by the raw gradient $\partial \Lcf / \partial a_g$ instead: a sensitivity score for latents the circuit would have to switch on.
The baseline keeps the top-scoring latents of each sign of $\Scf$.
\paragraph{Ablation gradient: what sustains the firing?}
The mirror score on activating contexts,
$\Sab = \frac{1}{|\posset|} \sum_{(x,t)} a_g(x, t_g) \, \partial \preact(x, t) / \partial a_g(x, t_g)$,
is large when the upstream latent is strongly active and the target is sensitive to it; its top latents form the node set.

\subsection{Attribution Modes}
Both objectives are estimator-agnostic: evaluating the gradient at the clean activations, as attribution patching does~\citep{nanda2023attribution, syed2023attribution, kramar2024atp}, is one choice of \emph{linearisation point}, not the only one.
The two modes used here differ only in where gradients are linearised, holding objective, contexts, and selection fixed:
\item \textbf{Path attribution} (\texttt{ig\_mean}): integrated gradients along one straight path from every upstream site's mean-ablation value to its natural activations, all sites interpolated together, with downstream sites held at their interpolated codes and the SAE errors at their clean values. It follows the IE_ig node estimator of \citet{marks2025sparse} in form but not in detail: [...]
\item \textbf{Restoration}: iterated greedy re-linearisation, in which each round attributes at the \emph{current} partially restored stream, admits the top scorers, restores them by live re-encoding, and re-linearises, for as many rounds as the target has upstream sites.
```

From `sec:membership` (removed):

> Attribution returns ranked latents, so the baselines must also decide which node set a ranking defines. [...] The
> failure is a property of position aggregation, not of node selection: the union of per-position active sets runs to
> thousands of latents per site, concentrated in the residual stream.

> ... which is why these unions reach $10^{5}$--$10^{6}$ nodes (\cref{tab:matrix}).

### app:method-comparison (pre-v1, replaced)

Intro: "\Cref{tab:matrix} compares the discovery families on the same 22 depth-stratified targets, the same held-out
contexts, and the same ablation methods: the attribution baselines (\cref{sec:attr-baselines}) at their natural size
and truncated to the weighted circuit's size, an adaptation of the hierarchical attribution of
\citet{ge2024automatically}, and unweighted and weighted circuit masking. The panel predates protocol v1."

Pre-v1 table (data: experiments/031-matrix + 029-panel). The columns are: arm, n (med), φZ, φA, φind, φsup, pass.

| arm | n (med) | φZ | φA | φind | φsup | pass |
|---|---|---|---|---|---|---|
| path attribution (abl) | 591,120 | 0.99 | 0.99 | 1.15 | 1.00 | — |
| path attribution (ctr) | 591,152 | 0.99 | 0.99 | 1.15 | 1.00 | — |
| restoration (rounds=sites) | 560,077 | 0.99 | 0.97 | 1.14 | 1.00 | — |
| path attr. (abl), top-n | 369 | 1.08 | 0.00 | 0.61 | 1.00 | 0/21 |
| path attr. (ctr), top-n | 369 | 0.99 | 0.00 | 0.56 | 1.00 | 0/21 |
| restoration, top-n | 369 | 4.36 | 0.00 | 0.66 | 0.95 | 0/21 |
| hierarchical attr. (adapted) | 359 | 0.63 | 0.00 | 0.86 | 1.00 | 0/21 |
| unweighted circuit masking | 1,216 | 0.66 | 0.23 | 0.75 | 1.00 | 3/21 |
| weighted circuit, 100 steps | 1,061 | 0.96 | 1.00 | 1.01 | 1.00 | 12/21 |
| weighted circuit, 400 steps | 369 | 0.99 | 1.08 | 0.89 | 1.00 | 17/21 |

Caption (pre-v1):
- fitted on 48 contexts; medians over 22 targets;
- the provisional pass rule was [0.8, 1.25] on Z and A, plus φsup > 0.9;
- top-n rows used the max-over-(context, position) ranking;
- the Ge row was adapted (first-order gradients, aggregated over contexts, τ bisected to size).

Three-results paragraph (pre-v1):
1. At natural size every attribution arm reconstructs the target with its full per-position union, 10^5–10^6 nodes;
   the node sets largely coincide there.
2. Truncated to the weighted size, every attribution arm fails, Ge included; unweighted rarely passes; weighted passes
   on most targets. Compact faithfulness is reached only by optimising node set and coefficients together. The
   truncations are naive, and whether a better ranking narrows a gap from 0.00 to ~1 under mean ablation was pending.
3. The home-turf effect (kept in the new text).

### Main text, "Attribution at matched size" (replaced)

> On a 22-target panel, rank-and-cut attribution truncated to the weighted circuits' size failed on every target,
> unweighted circuit masking rarely passed, and weighted circuits passed on most (\cref{app:method-comparison}); the
> panel predates protocol v1.

### Main text, random circuits (cut clause)

> ... and none induces the target to half its level in contrast contexts, so faithfulness and sufficiency to induce
> are properties of the selected node set, not of the fitting capacity of the coefficients.

Intro: "..., so what passes is the selected node set, not the fitting."

### Main text, Recommendations (draft cut for the page budget)

> Build held-out sets that exclude anchor-prefix and near copies of the training contexts, not only exact ones: in our
> corpus two thirds of the original held-out strongest contexts repeated a training-side context, nearly half of them
> inexactly (\cref{app:heldout}).
