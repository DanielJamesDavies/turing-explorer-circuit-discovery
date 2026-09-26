# 050 — Interesting circuits on Gemma-2-2B (all three SAE kinds)

Daniel's call (2026-09-09): TuringLLM lacks the classic behaviours (no
IOI, no induction, no arithmetic — see 047), so the showcase circuits
should come from a model that has them. Gemma-2-2B + GemmaScope is the
substrate we already de-risked in 048 (IOI att+mlp: 212 members, EF 1.04
held-out, name movers at L22.H4/H5). This campaign finds circuits for
the behaviours Gemma *demonstrably* has, with **att + mlp + res** sites
so they are full circuits rather than an MLP shadow.

## 1. Which behaviours does Gemma have? (`probe_candidates.py`)

Each candidate is a contrastive next-token decision (target vs a
plausible wrong answer) — exactly what the tri-amp fitter needs. 60
prompts each, `unsloth/gemma-2-2b` bf16.

| task | n | argmax acc | logit diff | p(target) | example |
|---|---|---|---|---|---|
| **capital** | 20 | 0.95 | **+13.9** | 0.87 | "The capital of Norway is" → ` Oslo` vs ` Madrid` |
| **country_of_capital** | 20 | 1.00 | **+13.8** | 0.89 | "Paris is the capital city of" → ` France` |
| **induction** | 60 | 1.00 | **+11.9** | 0.76 | `… river forest monkey … river` → ` forest` |
| **code_bracket** | 60 | 1.00 | +10.3 | 0.68 | `return (n + 1` → `)` vs `:` |
| **month_succ** | 60 | 1.00 | +7.1 | 0.63 | "April, May, June," → ` July` |
| **day_succ** | 60 | 1.00 | +6.2 | 0.59 | "Monday, Tuesday, Wednesday," → ` Thursday` |
| **pronoun** | 60 | 1.00 | +4.7 | 0.68 | "Tom went to the shop because" → ` he` vs ` she` |
| **opposite** | 60 | 0.92 | +3.7 | 0.49 | "The opposite of rich is" → ` poor` |
| plural | 60 | 0.27 | +4.3 | 0.36 | margin positive but argmax is a continuation |
| past_tense | 60 | 0.28 | +1.5 | 0.10 | weak |
| greater_than | 60 | 0.30 | +4.1 | 0.34 | present but weak (0.68 in 048's probe) |
| acronym | 10 | 0.00 | +8.3 | 0.01 | model prefers to continue the phrase |
| number_succ / addition | — | — | — | — | rejected: Gemma tokenises digits separately, so ` 8` is not one token — fixed in `make_datasets.py` by putting the space in the prompt |

**Fittable (acc ≥ 0.70, margin > 1.0):** capital, country_of_capital,
induction, code_bracket, month_succ, day_succ, pronoun, opposite.
Contrast with TuringLLM, which has none of the in-context or knowledge
ones. Full numbers in `candidate_competence.json`.

## 2. Datasets (`make_datasets.py`)

Writes `<task>_rows.json` = `[{prompt, target(token id), contrast(token
id), meta}]`, the shape `048/fit_gemma_task.py` now consumes via
`ROWS=<path>` (new hook; the built-in ioi/gt/agree paths are unchanged).
Every row is validated: both answers must be a single next token and
must differ. Sizes: induction 200, code_bracket 200, capital 128,
pronoun 120, opposite 117, country_of_capital 96, addition 72,
month_succ 27, day_succ 12, number_succ 5 (the last three are limited by
how few distinct month/day/number runs exist — enlarge the templates
before fitting them).

## 3. Three-kind fits

VRAM arithmetic on the 16 GB card: model bf16 5.2 GB; SAEs in bf16 per
layer = att 134 MB + mlp 151 + res 151 = 436 MB. All 26 layers × 3 kinds
= 11.3 GB → 16.5 GB total, over the card. **`LAYER_STRIDE=2`** (new
option in the fitter) keeps all three kinds at every other layer — 14
layers, 42 sites, 6.1 GB — which still covers L22 where the IOI name
movers sit, and leaves ~5 GB for the backward graph. All 26 layers ×
3 kinds needs a pod.

```bash
TASK=induction ROWS=experiments/050-gemma-circuits/induction_rows.json \
  KINDS=att,mlp,res LAYER_STRIDE=2 OUT=experiments/050-gemma-circuits \
  TAG=amr_s2 ./.venv/bin/python experiments/048-gemma-tasks/fit_gemma_task.py
```

### 3a. THE FLOOR DECIDES WHETHER RESIDUAL SITES ARE USABLE

First three-kind attempt used the historical ZERO floor and produced a
circuit that recovers nothing:

| config | sites | members | held-out EF | α=1 EF | nulls |
|---|---|---|---|---|---|
| att+mlp+res, **zero** floor, stride 2 | 42 | 86 (att 11, mlp 10, res 65) | **0.008** | 0.014 | −0.03, −0.05 |
| att+mlp, zero floor, all 26 layers | 52 | **92** (att 57, mlp 35) | **0.815** | 0.566 | −0.01, 0.00 |

Why: zero-filling non-members at a RES site deletes the SAE-explained
part of the whole residual stream, so a sparse circuit would have to
rebuild the model's main pathway from its own members. At att/mlp sites
the same edit removes only that block's contribution and the stream
flows on. The training trace shows the collapse directly — the value
starts at 11.8 with the mask open and falls to −0.3 once the L1 prunes.

Fix (Daniel's, 2026-09-09): the **dual floor** — our engine's dual/triple
floor minus the negctx term, which these task datasets cannot supply.
`FLOOR=dual` scores the mask every step under BOTH zero-fill (weight 1)
and mean-fill (weight `FLOOR_W`, default 0.25), so a member must earn its
place under both semantics; `FLOOR=mean` is SFC-style mean ablation
alone. Site means are computed once over the training prompts at real
token positions (BOS and right-padding excluded) and are ALWAYS collected
so every run reports both evaluation frames.

**Induction, all four configurations** (held-out log p(answer) −
log p(distractor); full 11.81; each frame's EF uses its own empty
baseline — zero-fill empty −0.77, mean-fill empty −0.33):

| config | sites | members | zero-fill EF | mean-fill EF | α=1 (own frame) | nulls |
|---|---|---|---|---|---|---|
| att+mlp, zero floor, 26 layers | 52 | 92 (att 57, mlp 35) | **0.815** | — | 0.566 | ≈0 |
| att+mlp+res, zero floor | 42 | 86 (res 65) | 0.008 | — | 0.014 | ≈0 |
| att+mlp+res, **dual** floor (w 0.25) | 42 | 443 (att 47, mlp 39, res 357) | **0.604** | 0.014 | 0.131 | ≈0 |
| att+mlp+res, mean floor | 42 | 606 (att 48, mlp 31, res 527) | −0.006 | **0.456** | −0.008 | ≈0 |

Two readings, both worth keeping:

1. **The dual floor works as intended.** Adding the mean term at weight
   0.25 took the zero-frame EF from 0.008 to 0.604 with residual sites
   in: the mean term keeps gradient signal alive early, when the
   zero-frame value has already collapsed under the L1.
2. **A circuit scores in the frame it was trained under, and not the
   other.** Dual → zero 0.604 / mean 0.014; mean → zero −0.006 / mean
   0.456. The floor is not a scoring detail, it is part of what the
   circuit *is* — the same lesson as `floor-determines-membership` on
   TuringLLM, now visible on Gemma with a second substrate.

**Why induction resists residual ablation at all:** the answer is IN the
prompt. Any floor at a res site — zero or mean — destroys the token
content the circuit has to copy, so the circuit must spend members
re-encoding the words themselves (357 and 527 res members respectively).
For in-context tasks the att+mlp circuit is arguably the right object:
the stream flows naturally carrying the content, and the circuit says
which attention/MLP features do the copying. The discriminating test is
a task answered from the WEIGHTS rather than the context (capital
recall), where res ablation should cost far less — run below.

### 3c. Capital recall — the cleanest circuit so far

`capital_am_all_gemma_members.jsonl`: att+mlp, all 26 layers, zero floor,
128 prompts. **95 members (62 att, 33 mlp), held-out EF 0.939** (full
13.52, empty 1.89, circuit 12.81), α=1 → 0.324, nulls 0.010/0.012. Fit
44 s. The three-kind dual-floor version (383 members, 301 res) scored
0.298 zero-fill / 0.262 mean-fill, so residual sites cost accuracy here
too — the prediction that weight-sourced answers would tolerate residual
ablation was WRONG, and att+mlp is the better object for both tasks.

### 3b. Induction circuit (att+mlp, all 26 layers, zero floor)

`induction_am_all_gemma_members.jsonl`: 92 members, **57 attention** vs 35
MLP — attention-dominated, as induction should be. Held-out
log p(answer) − log p(distractor): full 11.81, empty 1.03, circuit 9.81
(EF 0.815), α=1 7.13 (EF 0.566), amplitude-permuted nulls 0.89 / 1.02
(EF ≈ 0). Fit 73 s. Amplitudes carry a third of the recovery, as on
TuringLLM and on the Gemma IOI att+mlp circuit.

## 4. Induction mechanism (`analyse_induction.py`, `induction_word_test.py`)

Roles per position: `prev_first` (first-copy occurrence of the last
prompt word — the match an induction head looks up), `ans_first` (the
first-copy occurrence of the ANSWER, match+1 — what it must read),
`first_copy`, `repeat`, `END` (prediction).

**Contribution at END** = α × activation × [DLA(answer) − DLA(contrast)],
i.e. each member's own account of how the answer gets promoted. Total
+2.04, from 51 positive members (+5.34) against 36 negative (−3.30). The
top contributors are LATE ATTENTION: `att L19/2485` (+0.77),
`att L22/15641` (+0.50), `att L22/2534` (+0.39), `att L22/9739`,
`att L22/7067`, `att L22/8685`, `att L23/3025` — six of the top ten are
att L22, the same layer as the IOI name movers in 048. Each fires at END
on only a subset of prompts (49–78 of 200).

NOTE on metric choice: the unweighted DLA ("copy score") is uninformative
here (median +0.001) because the decoder direction is unit-scale — the
activation-weighted contribution is the meaningful quantity, and the
first ranking by modal peak position returned an EMPTY list of
previous-token candidates because no member's MODAL peak is the single
`ans_first` position. Rank by activation at a role, not by peak.

### 4a. NEGATIVE RESULT: they are not per-word copiers

The test that identified the IOI name movers (purity of the answer word
across a member's firings + DLA rank of that word), applied to the 39
attention members firing at END on ≥10 prompts:

| | induction (this) | IOI name movers (048) |
|---|---|---|
| purity, median | **0.08** (chance 0.080) | up to 1.00 |
| purity, max | 0.29 | — |
| members at DLA rank 1 | 8 / 39 (21%, chance 4.3%) | 22 / 24 (92%) |
| distinct top words covered | 10 of 23 | 22 of 24 names |

So Gemma does NOT implement induction with word-specific features. The
L22 members are the best of a weak lot (purity 0.17–0.29, 5 of 9 at rank
1) — above chance, nothing like the name movers. This is coherent rather
than disappointing: an induction head copies WHATEVER token follows the
match, so the answer's identity rides on the attention pattern and the
residual stream content, not on a word-specific latent. It also explains
§3a: zero-filling res destroys exactly the content being copied (EF
0.008) while att/mlp ablation leaves the stream intact (0.815). IOI
differs because names are a closed class with dedicated features.

### 4b. POSITIVE RESULT: the members sit in induction heads (`induction_head_test.py`)

Attention SAEs are trained on the o_proj INPUT (the concatenated head
outputs), so each latent's decoder vector splits into 8 blocks of 256 and
the block holding most of the norm is the member's head. Assigning the 28
attention members that way, then measuring each head's attention FROM the
prediction position over 100 prompts:

| head | circuit members | attn → ans_first | attn → prev_first | vs "other" baseline |
|---|---|---|---|---|
| **L22.H5** | **6** | **0.252** | 0.060 | **5×** |
| **L22.H4** | **2** | **0.228** | 0.057 | **4×** |
| L12.H1 | 0 | 0.335 | 0.019 | 4× |
| L19.H1 | 0 | 0.304 | 0.061 | 6× |
| L23.H6 | 0 | 0.171 | 0.067 | 3× |
| L5.H4 | 1 | 0.019 | **0.394** | duplicate/previous-token head |
| L5.H2 | 2 | 0.033 | **0.227** | duplicate/previous-token head |
| L12.H3 | 0 | 0.055 | 0.268 | duplicate/previous-token head |

**8 of the 9 L22 members live in H4/H5, the two heads that attend to the
answer's first occurrence** — the induction-head signature, recovered
from a logit endpoint with no supervision on attention patterns. The
circuit also holds members in L5.H2/H4, previous-token heads, i.e. both
halves of Olsson's mechanism.

**The same heads do IOI.** 048 localised Gemma's IOI name movers to L22
head 4 (12 features) and head 5 (8). So L22.H4/H5 is Gemma's
"copy the right token from context" machinery, recruited by both classic
tasks — but realised differently at feature level: per-NAME features for
IOI (purity → 1.00, 22/24 at DLA rank 1), content-general features for
induction (purity at chance, §4a). Same heads, two tasks, two feature
regimes. This is the paper-worthy observation from this campaign.

Caveats: L12.H1 (0.335) and L19.H1 (0.304) are strong induction heads
with NO circuit members — the circuit is incomplete, consistent with EF
0.815 rather than 1.0. BOS takes 30–80% of attention in most heads
(Gemma's attention sink) and is excluded from the "other" baseline.
