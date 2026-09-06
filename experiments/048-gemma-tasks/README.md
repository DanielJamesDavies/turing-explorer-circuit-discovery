# Known circuits on Gemma-2-2B (tri-amp, GemmaScope SAEs)

Follow-up to 047 (TuringLLM) at Daniel's request (2026-09-07): do the
same task circuits on Gemma-2-2B, where the classic circuits were
actually found, and see whether we match up.

Substrate: `037-gemmascope` conventions — unsloth/gemma-2-2b (bf16),
GemmaScope 16k JumpReLU SAEs at TIER 2 sparsity, intervention
`x <- x + (chat - c) @ W_dec` (SAE error preserved; BOS position never
edited). Site kinds (`fit_gemma_task.py`, `KINDS=att,mlp,res`; conventions
verified by FVU/L0 in `probe_kinds.py`): **mlp** = `post_feedforward_layernorm`
output (`gemma-scope-2b-pt-mlp`); **att** = `self_attn.o_proj` INPUT, the
2048-dim concatenated head outputs (`gemma-scope-2b-pt-att`); **res** =
decoder-layer output / resid_post (`gemma-scope-2b-pt-res`). Section 2 is
MLP-only (26 sites); section 4 is att+mlp (52 sites). All three kinds
(78 sites, ~11 GB of SAEs) do not fit next to the model on the 16 GB RTX
— that run goes on the pod. Endpoint: contrastive logit (log p(target) − log p(contrast));
zero floor; free amplitudes; λ=3e-3, 400 steps; 75/25 split; matched
amp-nulls; α=1 control. Prompts right-padded, BOS prepended.

## 1. Task competence (`probe_gemma_tasks.py`) — Gemma vs TuringLLM

| task | TuringLLM (047) | Gemma-2-2B |
|---|---|---|
| IOI | absent (p(IO)≈1e-4) | **present**: logit diff IO−S 3.81, 119/120 positive, argmax==IO 62% |
| induction (repeated sentence) | absent (41%→43%) | **present**: 36% → 95% |
| greater-than | 0.72 | 0.68 |
| agreement (PP distractor) | 81% | 98% |

Gemma has every classic behaviour; TuringLLM lacks the two that need
in-context copying. A clean model-level contrast for the paper.

## 2. Results, MLP sites only (`fit_gemma_task.py`, held-out prompts, 26 MLP-SAE layers)

| task | members | value (held-out) full → circuit | EF | α=1 EF | amp-nulls |
|---|---|---|---|---|---|
| **IOI** (logp IO − logp S) | **53** | 3.65 → 3.87 (empty −0.58) | **1.05** | 0.45 | 0.00, −0.05 |
| greater-than (logp target) | 168 | −1.09 → −1.04 (empty −4.34) | 1.02 | 0.83 | −0.10, 0.03 |
| agreement (logp correct − wrong) | 34 | 2.62 → 2.26 (empty 1.06) | 0.77 | 0.67 | 0.22, 0.03 |

Fits take 35–46 s on the RTX (Gemma is 2B but the circuits are tiny).
Why so small: the substrate preserves each SAE's error term (SFC's
error nodes), so the circuit explains the FEATURE-mediated part of the
behaviour and the residual rides on the error — unlike TuringLLM's
all-sites zero-fill, which had to rebuild the whole stream (1.5k–2.7k
members). Amplitudes matter less here than on TuringLLM (α=1 keeps
45–83%) for the same reason. With 34–53 members the matched nulls have
visible variance (one agreement null at 0.22): report several.

## 3. IOI structure, MLP-only circuit (`analyse_gemma_ioi.py`, roles IO / S1 / S2 / END)

53 members over 19 layers. Concentrated (≥60% mass) members:
- **S2 (the repeated subject): 3** — L4/11136 (α 2.83), L8/3222
  (α 3.15), L12/8332 (α 2.30, fires on 196/255). Duplicate-token /
  S-inhibition candidates, in the early-mid layers where Wang et al.'s
  duplicate-token and induction heads sit.
- **END (prediction position): 2** — L4/892 (α 2.23, fires 255/255),
  L7/13327 (α 2.62). Name-mover-adjacent MLP features.
- IO: 0, S1: 0 concentrated; 42 members' mass sits on template tokens.

Reading: IOI is an ATTENTION-head circuit (name movers copy IO at END,
S-inhibition heads carry "not S" from S2); with MLP-output SAEs only we
see its MLP-side shadow — S2 features and END features — not the heads
themselves. The natural next step is the GemmaScope attention-output
SAEs (`gemma-scope-2b-pt-att`) as additional sites: then name movers
become nameable and the comparison with Wang et al. is head-for-head.

## 4. IOI with attention + MLP sites (`KINDS=att,mlp`; 52 sites)

| circuit | members | value (held-out) full → circuit (empty) | EF | α=1 EF | amp-nulls |
|---|---|---|---|---|---|
| MLP-only (sec. 2) | 53 | 3.65 → 3.87 (−0.58) | 1.05 | 0.45 | 0.00, −0.05 |
| **att + mlp** | **212** (att 135, mlp 77) | 3.65 → 3.83 (−0.68) | **1.04** | **0.08** | −0.02, 0.03 |

Fit 45 s (773 s wall incl. ~12 min of SAE downloads). Two things change
once attention features are available: the circuit takes them (135 of 212
members) and the amplitudes become essential again — the same members at
α=1 keep 8% of the behaviour vs 45% MLP-only. Nulls stay at zero.

### 4a. Structure — the mass-share view is biased against END

`analyse_gemma_ioi.py` classifies by share of activation mass, but END is
ONE position competing with ~11 "other" positions, so a feature whose
single densest position is END shows an END share of ~0.2 and lands in
"other". `analyse_ioi_density.py` fixes this (mean activation at the role
position ÷ mean over "other" positions) and adds a NAME-IDENTITY test.

**S2 (repeated subject): name-independent duplicate detection.**
52 members have S2 density ≥ 3; the strongest fire ONLY at S2 (density
> 10^5) on all 255 prompts with S-name purity 0.08 = chance — att L0/16364,
L2, L3/4041, L4, L6 (×3: 21, 608, 15614), L8 (×3), L10/4213, L12, plus mlp
L0/L2/L4. Wang et al.'s duplicate-token heads (early) and S-inhibition
heads (mid) in attention sites, exactly where expected.

**END: 24 attention features at L22, one per name = the name movers.**
Each fires at END on a small subset of prompts (16–34 of 255), and the
subset is one IO name: IO-name purity 0.3–1.0 vs chance 0.06 (4192 → Dan
1.00; 5520 → Sarah 0.65; 13975 → Amy 0.65; 9516 → Alice; 9122 → Joseph;
8737 → Paul; 8694 → Emma; …), covering 22 of the 24 names in the pool.
Direct logit attribution (`dla_att_members.py`: W_dec → o_proj → final-norm
weight → unembedding, ranked over the 24 name tokens) confirms they WRITE
that name: **22/24 features rank their own name first** (23/24 top-3;
chance 1/24). The two misses (5505, 6260) are generic END features that
fire on nearly every prompt. Head localisation of the decoder directions:
L22 head 4 (12 features) and head 5 (8), with heads 2 and 7 (2 each) —
i.e. Gemma-2-2B's name movers are **L22.H4 / L22.H5** (+H2, H7), at 85%
depth like GPT-2's L9–L10 name movers at 75–85%.

Also at END, name-INDEPENDENT attention features (att L7/2522 on 255/255,
L13 ×3, L14 ×2, L16 ×3) — task/position structure rather than content.

Reading: with attention SAEs on the o_proj input the tri-amp circuit
reproduces IOI's published anatomy head-for-head from a single learned
mask — duplicate/S-inhibition features at S2 in early-mid attention,
name-mover features at END in L22 heads 4/5, each one writing its name —
with no head ablation, path patching, or supervision on positions.
Caveats: single seed and TIER 2 SAEs; the name-per-feature basis is a
property of GemmaScope's att SAE, which the circuit selects rather than
creates; S-name purity 0.2–0.6 on some L22 features (e.g. 5226 Jim, 9470
David) hints at negative-name-mover-like members but is untested.
Residual sites (res) still to be added on the pod.
