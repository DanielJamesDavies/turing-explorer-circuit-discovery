# Outreach notes (DAN-216, draft for Daniel)

Draft, 2026-11-15. Nothing here has been sent. Daniel chooses the recipients and the questions, and sends the messages himself.

## Template email

Subject: Circuits for single SAE latents — two-page summary, one question

Hi [name],

I'm an independent researcher working on circuit discovery for internal SAE latents rather than outputs. Your work on [their paper / method] is one of the closest to ours, so I'd value your view before we put the paper on arXiv.

The short version: for a target SAE latent, weighted circuit masking (WCM) learns a gate and a separate scaling coefficient for every upstream latent, trained so the circuit reproduces the target under zero, activating-mean and contrast-mean ablations at once (plus a resample term), with SAE errors kept live. At one sparsity penalty it finds a passing circuit for 82–88% of sampled latents on TuringLLM, Pythia-70m and GPT-2 small, with circuits of a few hundred latents, and size-matched random circuits given the same coefficient fitting never pass. The coefficients are what keep circuits small: in the deepest layers, unweighted circuits need a median 13.8× more latents. The two-page summary is attached.

One question, if you have a few minutes: [one specific question, e.g. from the list below].

No need for a long reply; even a sentence on whether the evaluation looks right to you would help a lot.

Thanks,
Daniel Davies
[affiliation line / link to repo]

(Before sending: recheck every number against the final PDF; the 82–88% figures are held-out pass rates that await the document-disjoint re-score, DAN-217.)

## Kinds of researchers to approach

Based on which works the paper cites most (by count of citations in main.tex) and which methods it positions against or compares with. Daniel decides who; the issue already lists his candidates.

1. **Sparse-feature-circuit authors** (the most-cited work in the paper, 12 citations; our necessity score is one minus its completeness score, and its SAE-features-with-error-nodes setup is our substrate). Possible question: is necessity with errors live, plus faithfulness under three ablations, a fair standard for a single-latent circuit?
2. **Learned-mask circuit discovery** (edge pruning, subnetwork probing, and the scaling work on learned masks over SAE latents, whose objective moved to one latent is our *unweighted* baseline). Possible question: does fixing α = 1 make our unweighted baseline a faithful version of your method, or is something lost in the move to a single latent?
3. **Ablation-method and evaluation researchers** (optimal ablation, the work showing faithfulness depends on the ablation method, the MIB benchmark, causal scrubbing, hypothesis-testing of circuits). Possible question: is a fitted-coefficient random circuit a strong enough null, and is the donor-background test the right "real background" check?
4. **Circuits for internal features by attribution** (tracing upstream of a chosen SAE or transcoder feature, or a probe direction; transcoder and attribution-graph work). Possible question: where would you expect thresholded attribution to match an optimised circuit for one latent, and where not? (The head-to-head on the production run is still pending, so do not claim one.)
5. **Attribution-patching researchers** (attribution patching and its refinements, e.g. faithfulness-aware variants). Relevant once the attribution baselines (T7) are in; until then a question about first-order scoring of upstream latents is the natural one.
6. **SAE / dictionary researchers** (Top-K SAE scaling, canonical-units work). Possible question: how much do you expect pass rates and sizes to depend on the dictionary, given that on dense ReLU dictionaries our fitted random baseline stops discriminating?
7. **The AISI contacts** named in the issue: a safety-oriented reader; the task-free framing ("the model's activations decide what to explain") is probably the most relevant angle.

## Practical notes

- One specific question per email, as the issue asks; the list above gives one candidate per kind.
- Attach the PDF from `paper/summary/build/two-page-summary.pdf` once Daniel has reviewed it.
- The LessWrong / Alignment Forum write-up (step 3) can reuse the summary's structure: idea, two results, caveat, figure.
