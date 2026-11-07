# 066 Newton chain: does the "Newton's Third Law" relay depend on the name?

## Question

The 065 descriptions suggest a relay across tokens in TuringLLM for named laws such as "Newton's Third Law":

1. an apostrophe latent after "Newton" (`9.resid.29881`) that pushes "Second" / "Third";
2. an ordinal latent on "Second" / "Third" (`7.resid.4492`) that pushes "Law";
3. a latent on "Law" after "Third" (`9.attn.28565`).

A circuit cannot show whether the first step depends on *which* name it follows (knowledge of Newton) or on any name
(the kind of entity). This experiment tests that on minimal pairs, outside the circuit machinery. It runs on the live
model and the SAE bank only; no circuit is fitted or scored.

Related: the generic eponym latent `10.resid.15497` ("'s" after a name that carries a law) is one of the paper's
case studies (`paper/main.tex:1475`; experiment 063).

## What the code does

Both scripts use the 064 helpers (`experiments/064-prediction-circuits/common.py`: model, SAE bank, activation
capture) and run on CPU.

**`probe.py`: who fires where.** It runs 12 prompts and prints, at each token, the Top-K activation of 7 candidate
latents, with the model's top-3 next-token guesses at positions where any of them fire.
- Latents: the three relay latents above, `5.attn.4921` ("Zeroth Law"), `10.resid.15497` (eponym 's → Law, 063),
  `9.resid.2881` (apostrophe after proper names) and `9.resid.22174` ("Gutenberg 's").
- Prompts for the apostrophe step: real vs made-up names (Newton, Kepler, Einstein, the invented "Zorbel"), physics
  vs non-physics text ("Newton's cat"), and a non-law possessive (Shakespeare).
- Prompts for the ordinal step: First / Second / Third with and without "Law" following, "Third Law of
  Thermodynamics", "Zeroth Law", "the third chapter", "the Second World War".

**`causal.py`: does removing the latent change the prediction?** On 5 prompts ending in "<Name>'s" (classical
mechanics with Newton, Kepler, Einstein, Zorbel; and "In the story, Newton's"), it removes a latent's contribution
(activation × decoder direction) at the apostrophe and the "s" positions, keeping the SAE error term, and reads the
next-token probabilities after the "s":
- `ordinal` = p(" First") + p(" Second") + p(" Third"), the Newton-specific continuation;
- `law` = p(" Law") + p(" law") + p(" laws") + p(" La"), the generic "<name>'s → Law" continuation.

Ablations: clean; the Newton apostrophe latent; the eponym latent; both; and a control, the other `9.resid` latent
active at the "s" whose activation is closest to the Newton latent's.

## Files and commands

| File | What |
|---|---|
| `probe.py` | Per-token activations of the candidate latents on 12 minimal-pair prompts |
| `causal.py` | Latent-removal check on the next-token ordinal and law probabilities |

```
python experiments/066-newton-chain/probe.py     # CPU, about a minute
python experiments/066-newton-chain/causal.py    # CPU
```

Both print to the console and write no files.

## Results

No results are saved in the repo. Both scripts print to stdout only, and no log of a run is kept here, so this README
reports no numbers. Rerun the two commands above to see them.

## Status and caveats

- An exploratory follow-up to the 065 descriptions; nothing in the paper cites it.
- The latent roles above are the 065 labels, which are hypotheses. The probe and the removal check are the test of
  them, and their outputs are not recorded.
- The removal is a single-latent edit at two positions in the live model; it says nothing about the rest of any
  circuit.
