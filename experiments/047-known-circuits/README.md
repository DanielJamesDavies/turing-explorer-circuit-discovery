# Recreating well-known circuits from their task datasets (tri-amp)

Daniel's ask (2026-09-05): "Recreate well known circuits using their
task datasets using tri amp mask, like IOI." Same machinery as
044-behaviours (logit objective, all 36 sites, zero floor, free
amplitudes, λ=3e-3, 400 steps, 75/25 train/held-out, matched
amp-permuted nulls), fed with constructed task prompts.

## 1. Which classic behaviours does TuringLLM have? (`probe_tasks.py`, unpadded)

| task | measure | result | verdict |
|---|---|---|---|
| IOI (Wang et al. 2022) | p(IO), argmax==IO | p(IO)≈1e-4, 0% — predicts "the"/"a" | **absent** |
| induction | 2nd-copy vs 1st-copy accuracy on a repeated natural sentence | 43% vs 41% | **absent** — no in-context copying at all (explains IOI) |
| greater-than (Hanna et al. 2023) | P(next digit > tens) − P(< tens) | 0.72, positive 118/120 | present |
| subject–verb agreement, PP distractor (Marks et al. 2025) | logit(correct) − logit(wrong) | 1.94, 81% correct | present |

Model-level finding: TuringLLM has no induction mechanism. Worth a
sentence in the paper's model description.

## 2. The padding trap (found here, affects every constructed dataset)

Left-padding a prompt with a run of token 0 (as `make_ioi.py` and the
older `044/make_knowledge.py` did) corrupts predictions: top tokens
become nonsense ("describe", "cover"). Fix: RIGHT-pad and read at a
per-row anchor (exact for a causal model). `behaviour_runner.py` now
honours `data["anchors"]`; `make_tasks.py` writes them.
`make_knowledge.py` still left-pads — fix before use.

## 3. Circuits (`make_tasks.py` → `behaviour_runner.py`, 256 prompts each)

| task | members | EF held-out | EF train | amp-nulls |
|---|---|---|---|---|
| greater-than | 1,577 | 0.984 | 0.990 | −0.006 / −0.009 |
| agreement | 2,726 | 1.002 | 1.007 | −0.019 / −0.020 |

EF = share of the log-prob gap (empty→full) for the target token that
the circuit alone recovers under zero-fill of all other latents at all
sites, members at their fitted amplitudes, on prompts the fit never saw.

## 4. Structure vs the published circuits (`analyse_task_members.py`)

Members classified by the token role their activation mass sits on.

Greater-than (1,577 members):
- decisive digit dominates: **y1_tens 157** members vs y1_units 54 vs
  y1_c1 38 — the comparison is on tens, and the circuit's weight follows.
  88 members put ≥60% of their mass on the tens digit, concentrated
  L0–L4 (digit-identity features where Hanna et al.'s heads read YY).
- prediction position (y2_c2): 120 members; the concentrated ones sit
  L5–L11 — the "compute > at the output" half of the two-part structure.
- 823 "other" (346 at L0) = template-word scaffolding: the price of the
  all-sites zero-fill frame (the circuit must rebuild the stream), not a
  property of the task.

Agreement, contrastive circuit (2,312 members):
- subject-role members 381; the ≥60%-concentrated ones (111) are
  early/mid — L0 47, L1–L4 ~11 each, tailing off by L7: the
  subject-number features Marks et al. describe, read on the subject.
- final-position members 396; concentrated ones (166) are BIMODAL —
  61 at L0 (token identity) then rising again L7–L11 (14/10/14/12/8): the
  verb-number decision features, late, at the prediction position.
- 1,525 "other" = scaffolding, as for greater-than.
Confound to fix for a cleaner recreation: in these templates the PP
noun IS the last token, so "distractor" and "verb/final position"
coincide. Next iteration: end prompts with an adverb ("The keys on the
cabinet probably") so distractor-number and final-position features
separate.

## 5. Task-metric scoring under circuit-only execution (`score_task_circuits.py`)

Held-out prompts, zero-fill at all sites, five frames.

Greater-than (1,577 members):

| frame | P(>tens) − P(<tens) | target EF |
|---|---|---|
| full model | 0.831 | 1.000 |
| circuit + amplitudes | **0.667** | 0.984 |
| circuit at α = 1 | 0.000 | 0.043 |
| amp-null | 0.000 | −0.004 |

The circuit reproduces the task decision (80% of the margin) and ONLY
with its amplitudes.

Agreement, single-token logit objective (2,726 members): target EF 1.00
but logit(correct) − logit(wrong) fell 2.87 → 0.14 — the circuit raised
both verbs. The objective never saw the wrong verb. => added a
CONTRASTIVE logit objective (engine `contrast_tokens`, runner
`CONTRAST=1`): reproduce log p(correct) − log p(wrong), the SFC metric.

Agreement, contrastive objective (2,312 members):

| frame | logit(correct) − logit(wrong) | target EF |
|---|---|---|
| full model | 2.872 | 1.000 |
| empty | 1.158 | 0.000 |
| circuit + amplitudes | **2.202** | 0.935 |
| circuit at α = 1 | 0.033 | 0.023 |
| amp-null | 1.472 | 0.000 |

Held-out EF on the margin 0.61 (train 0.81; small denominator, noisy);
the circuit recovers most of the decision margin AND the target's
log-prob. Amplitudes essential in both tasks.

Lesson: for contrastive behaviours (IOI, agreement, gender bias) the
objective must be the contrast; a single-token log-prob objective
produces a circuit for "predict a verb here", not "predict the RIGHT
verb".
