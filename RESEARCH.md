# Research overview

This repository is the code for *What Makes a Latent Fire? Discovering Causal Dependencies Between Sparse
Autoencoder Latents* (Davies and Liu; draft in `paper/main.tex`). This page gives the question, the claims and the
limits. `README.md` covers the method in brief and how to run it; `paper/protocol.md` §0 and
`docs/wcm-specification.md` give the exact specification.

## The question

Circuit discovery usually explains an output: a logit, a probability, a behavioural metric. An output is the
model's computation expressed in our vocabulary, and many internal variables never show up as a token. Sparse
autoencoder (SAE) latents are an imperfect but useful sample of a model's own variables, learned without labels.
So we ask, for one target latent: **which upstream latents make it fire?**

Earlier work on internal targets attributes upstream of a chosen latent and thresholds the scores, ablates one node
at a time, or regresses. We instead *optimise* a circuit of upstream latents for a single target latent, and test it
by ablating everything outside the circuit and measuring that latent.

## The method: weighted circuit masking (WCM)

Every latent at every SAE site upstream of the target gets a gate (in or out of the circuit) and a coefficient α
(how strongly it acts). Both are trained so that the circuit alone reproduces the target's pre-activation while every
non-member is ablated in four ways: set to zero, set to the activating-context mean, set to the contrast-context mean
(contexts where the target is verified silent), or resampled from a real contrast context. A sparsity price on open
gates keeps the circuit small. The result is a set of members, each with its α; the two are always reported together.

Each circuit is scored on held-out contexts for faithfulness under the zero and mean ablations, and for necessity
(removing the members from the live model removes the target). It passes only if all of these hold at once.
Sufficiency to induce (setting the members' values turns on a silent target) and a donor-background test with real
contexts in place of means are reported alongside. The unweighted control fixes α = 1.

## What the paper claims

1. **Circuits for arbitrary latents, at scale.** WCM needs no behavioural task and applies to any latent in an SAE
   bank. On thousands of randomly sampled TuringLLM latents, at one sparsity price, it finds circuits of 10^2 to 10^3
   upstream latents that pass the joint held-out test for a large share of targets. Random circuits of the same size,
   given the same coefficient fitting, do not pass.
2. **Coefficients keep circuits small.** With α fixed at 1 the same members almost never pass, and the smallest
   passing unweighted circuits for deep targets are about ten times larger, with fewer deep targets passing at all.

The pass rate is one point on a size trade-off, not a ceiling: a weaker price gives larger circuits that pass more
often. Residual-stream targets pass most often and attention outputs least. The final numbers, including Pythia-70m
and GPT-2 small, are in the paper.

## Other findings worth knowing

- **The contrast set matters.** Circuits trained against close contrast contexts (near the target's topic) generalise
  worst to other backgrounds. Production keeps close contexts so that the model is the only change between models;
  for new work, random or distant contrast contexts are the better default.
- **Node sets are not unique.** A refit that passes about as often shares roughly two thirds of the original members,
  so we report sizes and scores, not single memberships.
- **Comparisons use other papers' methods as published**, on their own substrates, with WCM run there too
  (`experiments/107-true-baselines/`).

## Why a controlled model

The primary model, TuringLLM, is a 12-layer, 254M-parameter transformer trained from scratch on ~2B tokens of
documented synthetic educational text, with our own bank of 36 Top-K SAEs. A model whose training data is known is a
more tractable subject than one trained on an opaque crawl. Pythia-70m and GPT-2 small with public SAEs test whether
the results carry over.

## Limits

- Models of at most 254M parameters, and Top-K dictionaries only. On dense ReLU dictionaries the method runs, but a
  random circuit with fitted coefficients stops being a discriminating control.
- TuringLLM's synthetic corpus aids control but limits generality.
- Memberships are not unique (above), so a circuit is a sufficient-and-necessary set, not *the* mechanism.
- Circuits are sets with coefficients. Edges between members (`experiments/057-wcm-edges/`) and recursive tests
  (`experiments/099-recursive-causal-scrubbing/`) are exploratory, not claims.

## Where to look

| Topic | Location |
|---|---|
| Method in brief, install, run | `README.md` |
| Exact specification (code-cited) | `docs/wcm-specification.md`, `paper/protocol.md` §0 |
| Paper draft | `paper/main.tex` |
| WCM engine and evaluation | `src/circuit/instrument/learned_mask.py`, `src/eval/wcm_eval.py`, `src/eval/donor_test.py` |
| Experiments | `experiments/INDEX.md`; the current ones are listed in `README.md` |
