# 077 antiquity scrub: a hypothesis-structured swap test of the antiquity case study

## Question

The paper's case study (`paper/main.tex:575-601`, experiment 063) reads the TuringLLM target `9.resid.20419` as an
"is this classical Greek antiquity?" variable, carried by member groups for ancient / early times and for classical
Greece. Causal scrubbing asks whether such an interpretation survives swaps it says should not matter, and fails
swaps it says should. This is a group-swap test in the original model in that spirit; it is not a full recursive
scrub.

## What the code does

`antiquity_scrub.py` (GPU; the 062 driver's `Runner` via `073-strict-eval/strict_eval.py`):

- **Prompts.** "In <X>, the scholar lived in the city of" and "In <X>, the philosopher taught in the city of", with X
  from 6 classical phrases (the 5th century BC, 450 BC, ancient Greece, classical Athens, the age of Pericles,
  ancient Sparta) or 6 other ones (the 19th century, 1950, medieval France, modern Germany, the 1980s, Victorian
  England): 24 prompts.
- **Alignment.** Base and donor share the suffix, so suffix positions are aligned from the end. The shared tail is
  computed, not assumed: 8 tokens for the "scholar" suffix, 10 for the "philosopher" one (`run.log`).
- **Swap.** In the live model (every other latent live, SAE error kept), the named group's latents at the suffix
  positions are set to their values in the donor prompt. The target is read at the base's anchor, its peak over the
  suffix.
- **Groups** (named in 063 `validate.py`; the circuit's members and their contribution ranks come from
  `063-case-studies/results_full_deep/members/9.resid.20419.csv`):
  - "ancient / early times": 3 latents (`7.resid.10003`, `8.resid.13018`, `7.attn.24922`);
  - "classical Greek": 7 latents (`7.resid.37523`, `8.resid.4704`, `9.mlp.11623`, `9.mlp.21754`, `8.resid.13141`,
    `7.resid.25644`, `8.resid.34691`);
  - "both": the 10 together.
- **Conditions and predictions.**
  - same-class (classical base ← classical donor): the target is kept (≈ 1);
  - cross-class (classical base ← other donor): it drops toward the donor's level;
  - induce (other base ← classical donor): it rises toward the classical level.
- **Controls.** `N_CTRL` (8) random groups of other circuit members of the same size and similar contribution rank
  (± 15 ranks, 063's rule), seed 0.
- **Scores.** Retained = t_patched / t_base (same, cross); induced = t_patched / mean classical t (induce). Medians
  over base × donor pairs with the same suffix.

## Files and commands

| File | What |
|---|---|
| `antiquity_scrub.py` | The test; writes the rows and the report |
| `out/rows.jsonl` | One row per base × donor × group (612 rows), with the 8 control values |
| `out/report.md` | The summary table and the clean prompt reads |
| `run.log` | Console log |

```
PYTHONPATH=src python experiments/077-antiquity-scrub/antiquity_scrub.py      # env: N_CTRL (8)
```

## Results

From `out/report.md`. Clean target at the anchor: classical prompts mean 20.68, other prompts mean 0.96 (ratio 0.05).

| condition | group | n | pairs | story group | random members | prediction |
|---|---|---|---|---|---|---|
| same | ancient / early times | 3 | 60 | 1.00 | 1.00 | kept (≈ 1) |
| same | classical Greek | 7 | 60 | 0.97 | 1.00 | kept (≈ 1) |
| same | both | 10 | 60 | 0.98 | 1.00 | kept (≈ 1) |
| cross | ancient / early times | 3 | 72 | 0.60 | 0.93 | drops (toward 0.05) |
| cross | classical Greek | 7 | 72 | 0.38 | 0.91 | drops (toward 0.05) |
| cross | both | 10 | 72 | 0.21 | 0.86 | drops (toward 0.05) |
| induce | ancient / early times | 3 | 72 | 0.18 | 0.00 | rises |
| induce | classical Greek | 7 | 72 | 0.48 | 0.00 | rises |
| induce | both | 10 | 72 | 0.68 | 0.00 | rises |

Reading: every condition moves in the predicted direction, and the matched random member groups barely move the
target. Swapping in the 10 story latents from a non-classical prompt keeps 21% of the target (random groups: 86%), and
swapping them into a non-classical prompt raises the target to 68% of the classical level (random groups: 0%). The
drop does not reach the donor's level (0.05), so other latents carry part of the variable.

Clean reads range from 14.75 to 28.50 on the classical prompts. Among the other prompts, 9 of 12 read 0; the three
"philosopher taught" prompts with medieval France, modern Germany and Victorian England read 3.31–4.72
(`out/report.md`).

## Status and caveats

- Not cited in the paper. The paper's case study uses its own group definitions and knock-outs (063
  `case_study.py`, `paper/main.tex:589-593`); the groups here are a smaller 3 + 7 set.
- One target, 24 hand-written prompts, two templates. It tests this reading, not a population.
- The swaps act only at the shared suffix positions; the X phrase itself is never patched.
- The circuit members are the 15k run's (protocol v1) circuit for this target.
