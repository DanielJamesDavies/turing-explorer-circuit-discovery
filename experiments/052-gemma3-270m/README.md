# 052 — Gemma 3 270M as the open replacement substrate for TuringLLM

Motivation (2026-09-14/15): the TuringLLM census (049 §0e) showed its circuit
population is word assembly, frames and memorised sequences — a repertoire
limit, not a method limit. Daniel chose Gemma 3 270M + Gemma Scope 2 after a
survey of open model+SAE options. This folder holds the port.

## Stage 0 — feasibility (2026-09-16)

### 0a. Capability (`capability_probe.py`, `capability_*.json`)

One contrastive-decision battery (prompt, target, contrast) run on three
backends. A prompt is kept only when both answers are a single next token
under that model's tokenizer, so n can differ. `pos` = fraction with target
logit > contrast logit (the column comparable across tokenizers); `acc` =
argmax. Low acc with pos 1.00 means the right answer is strongly preferred
over the contrast but the top token is something else (acronym = the
merged-token issue: `NASA` is one token).

| task | Gemma 3 270M acc/pos | TuringLLM acc/pos | Gemma-2-2b acc/pos |
|---|---|---|---|
| IOI | 0.92 / 1.00 | 0.00 / 0.48 | 0.60 / 1.00 |
| induction | 0.07 / 1.00 | 0.00 / 0.59 | 1.00 / 1.00 |
| greater-than | 0.52 / 1.00 | 0.40 / 0.92 | 0.33 / 1.00 |
| agreement w/ attractor | 0.45 / 0.98 | 0.20 / 0.85 | 0.37 / 1.00 |
| two-hop (city -> language) | 0.87 / 1.00 | 0.00 / 0.67 | 1.00 / 1.00 |
| capital of country | 0.95 / 1.00 | 0.00 / 0.00 (n=1) | 0.95 / 1.00 |
| country of capital | 1.00 / 1.00 | 0.00 / 0.79 | 1.00 / 1.00 |
| month succ | 0.73 / 1.00 | 0.00 / 0.28 | 1.00 / 1.00 |
| day succ | 1.00 / 1.00 | — (split) | 1.00 / 1.00 |
| past tense | 0.75 / 0.95 | 0.00 / 0.22 | 0.28 / 0.88 |
| plural | 0.00 / 1.00 | 0.33 / 0.85 | 0.17 / 1.00 |
| opposite | 0.80 / 0.92 | 0.00 / 0.57 | 0.97 / 1.00 |
| pronoun | 1.00 / 1.00 | 0.00 / 0.47 | 1.00 / 1.00 |
| list closure | 0.85 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| acronym | 0.00 / 1.00 | 0.60 / 1.00 | 0.00 / 1.00 |
| code bracket | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Verdict: TuringLLM is at CHANCE on IOI and pronouns and cannot do capitals,
two-hop, opposites or tense. Gemma 3 270M has pos 0.92-1.00 on every task and
is broadly on par with Gemma-2-2b (10x larger) on contrastive decisions.

### 0b. Model and SAE facts (verified from config files, not spec pages)

- **Model: 18 layers x 640 hidden** (MLP 2048, 4 query heads / 1 KV head,
  head_dim 256, vocab 262,144, sliding window 512). A spec page claimed
  12 x 1024 — WRONG. ~100M transformer params, so per-token cost is a little
  under TuringLLM's ~151M, but there are **54 sites vs TuringLLM's 36**.
- `google/gemma-3-270m` is GATED (manual). Stage 0 used the ungated
  `unsloth/gemma-3-270m` mirror; byte-identity with Google's weights is
  UNVERIFIED (needs Gemma terms accepted on HF).
- **SAEs: `google/gemma-scope-2-270m-pt`**, ungated, CC-BY-4.0.
  Full-depth folders `*_all`: all 18 layers x {resid_post, attn_out,
  mlp_out} x widths **{16k, 262k} only** x L0 {small, big} (+ transcoders
  incl. affine variants, crosscoders, CLTs). Subset folders add 65k/1m at 4
  depths only.
- Architecture `jump_relu`; params.safetensors: `w_enc (d_in, W)`,
  `b_enc (W)`, `threshold (W)`, `w_dec (W, d_model)`, `b_dec (d_model)`.
  code = pre * (pre > threshold), pre = x @ w_enc + b_enc.
- **Hook points (identical to the 048 Gemma-2 conventions):**
  attn_out = `model.layers.N.self_attn.o_proj.input` (4 x 256 = 1024-dim,
  per-head blocks -> head localisation works), mlp_out =
  `model.layers.N.post_feedforward_layernorm.output`, resid_post =
  `model.layers.N.output`.
- L0 (resid_post 16k): small 10 @L0, 15 @L3, 20 @L6-L17; big 60 / 90 / 120.

### Decisions (Daniel, 2026-09-16)
1. **Width 16k** (54 x 16,384 = 885k latents, fewer than TuringLLM's 1.47M;
   262k = ~14M, ~10x heavier artifact build).
2. **L0 big** (~120 from L6 on; 60 @L0, 90 @L3). Folders:
   `{resid_post,attn_out,mlp_out}_all/layer_N_width_16k_l0_big/`.
3. Open: JumpReLU is not TopK: its threshold also censors, so the
   pre-activation objective's rationale should carry (and GemmaScope 1 worked
   in 048/050/051), but the paper's cross-SAE claim is TopK-only — needs a
   sentence. With L0 ~120 the SAEs are ~4x denser than TuringLLM's TopK, so
   circuit compactness is a thing to measure, not assume.

## Step 1 — SAE reconstruction audit (`gemmascope2.py`, `sae_recon.py`, `sae_recon_16k_big.json`)

All 54 SAEs (16k, L0 big) on wikitext-103 test (48 x 256 tokens, BOS
prepended and excluded from every statistic, never spliced). Clean loss
3.397 nats. Download + load 710 s (HF cache thereafter).

- **Encode convention: plain `x`** (no `b_dec` subtraction). The
  subtract-b_dec variant is closer to the configured L0 at 1/54 sites and
  its EV goes as low as -986. Same convention as GemmaScope 1; the fitter's
  default `SUB_BDEC=0` is correct.

| kind | EV median (range) | L0 measured vs config | unseen | dCE (nats) | recovered |
|---|---|---|---|---|---|
| att | 0.80 (0.74-0.92) | 121 vs 120 | 13.0% | **+0.013** | 0.95 |
| mlp | 0.94 (0.84-1.00) | 101 vs 120 | 13.5% | +0.044 | 0.88 |
| res | 0.97 (0.80-1.00) | 120 vs 120 | 8.0% | +0.165 | 0.99 |

- res L0 matches config at EVERY depth (60 @L0 ... 120 @L6-17) and EV 0.97:
  strong (not conclusive) evidence the unsloth mirror is the model the SAEs
  were trained on.
- mlp under-fires mid-depth on wikitext (L4 70 vs 100, L6 95 vs 120, L16 91
  vs 120) — plausibly corpus shift, not a defect.
- att EV is the lowest (0.80) yet its splice costs only +0.013 nats: the
  unexplained part of o_proj's input barely matters for the output.
  Irrelevant to correctness in our interventions anyway (x <- x + (c_hat -
  c) @ W_dec passes the SAE error through), but it bounds how much signal
  att members can carry.
- "recovered" is inflated for res (zero-ablating the residual stream gives
  L_zero 12-49 nats); quote dCE for res. Worst res sites: L1-L2 +0.31/+0.33,
  L17 +0.29.
- "unseen" = never fired in 12k tokens: an upper bound on dead latents.

## Step 2a — LOGIT-endpoint task circuits (`048/fit_gemma_task.py`, `collect_fits.py`)

NOTE: these are BEHAVIOURAL endpoints (log p(target) - log p(contrast)), not
the paper's latent endpoint. `fit_gemma_task.py` gained `SAE_SOURCE=gs2`,
`SUB_BDEC`, `FREEZE_NORM` (all default-off; 048/050/051 unchanged);
`051/make_twohop.py` gained `OUT` (token ids are tokenizer-specific).
Two-hop rows regenerated for Gemma 3: 237 (230 pass the competence filter).

- **MLP-only IOI works**: 124 members, EF 0.931 (nulls -0.09/-0.03, alpha=1
  0.12); 557 members EF 0.950. Matches Gemma-2-2b's 53-member 1.05.
- **att+mlp is UNSTABLE across settings**, not capacity-limited: IOI
  1,992 -> -0.17, 5,209 -> -0.01, 6,578 -> **0.96** (lr 0.01, 800 steps,
  lam 1e-4; nulls 0.04/-0.07), 9,143 -> -0.12. Two-hop 494 -> 0.47,
  1,257 -> 0.40, 2,625 -> 0.91, 5,911 -> 0.91 (one null 0.50), 7,018 -> 0.63.
  Best two-hop 0.91 needs ~2,600 members (Gemma-2-2b: 108 for 0.877).
- **att+mlp+res dual floor is degenerate**: the zero-fill frame reads exactly
  0.000 for empty, circuit and nulls.
- **Post-norm hypothesis REFUTED** (`norm_diag.py`): zero-filling every att
  latent leaves post_attention_layernorm output at 1.03x clean (unfrozen) vs
  0.94x (frozen), cos 0.94 — the attention output's norm is dominated by a
  constant component the SAE does not model. Freezing the norm does not
  rescue IOI (-0.11, -0.12). Freeze formula exact to 0.7% (bf16).
- Full table: `collect_fits.py`; logs `first_fits.log`, `logit_sweep.log`,
  `norm_and_mlp_diagnostics.log`.

## Step 2b — Gemma Scope 2 example stores (`download_examples.py`, `dedupe_examples.py`)

- **Only resid_post SAEs ship `examples.safetensors`** (18 files, ~733 MB
  each); attn_out / mlp_out folders have config + params only. att/mlp seed
  contexts must be built from the corpus.
- Per file: `activations`/`seq_ids`/`positions` [16384, 1000] (1,000
  activating token positions per latent, UNSORTED, several per sequence —
  dedupe by sequence), `feature_frequencies`, `logit_effects`,
  top/bottom tokens+logits [16384, 10], `tokens` [392,802, 256].
- **Verified against the model**: re-running model + SAE on stored examples
  reproduces the stored activation (median rel err 1.4%; 1/18 read 0, a
  latent sitting at its threshold).
- **One corpus shared by all 18 layers** (sha1 d2f787fa1d99); dedupe 13.2 GB
  -> 4.6 GB at `~/gemmascope2_examples` (corpus_d2f787fa1d99.npy + res_L*.npz).
  Examples index only sequences 0-130,933 of 392,802 at every layer: ~260k
  unreferenced sequences are a clean pool for negatives / att-mlp contexts.

## Step 2c — LATENT-endpoint tri-amp circuits (`fit_latent_seed.py`) — the paper's object

Production recipe (pos objective, triple floor zero 1.0 / negctx-mean 0.25 /
posctx-mean 0.10 (L<=7) or 0.05, free amplitudes + leak charge, lam 1e-3,
anneal, theta0 4, 400 steps, lr 0.05). Positives: top-64 distinct sequences
from the seed's examples (stored vs recomputed pre-act at anchor 0.7-1.2%);
negatives: random corpus windows verified silent. 48/16 held-out split.
Fitted amplitude null (R16) + permuted null; R12 vacuity guard.
Seeds: one random residual latent per depth (freq 1e-4 to 5e-3, >= 64 seqs).

**att+mlp+res upstream (the TuringLLM definition) BLOWS UP with depth**
(`latent_circuits_amr.jsonl`): empty zero-fill circuit drives the seed to
L3 18.75 (natural 31.9) -> L6 4.8e6 -> L9 1.9e12 -> L12 5.6e14 (posctx-mean
frame) -> L15 1.6e20; members 941 -> 11,956 -> 69,426 -> 384,410 ->
335,128 (the fit keeps everything to stop the explosion); every "1.000" is
vacuous (fitted nulls also ~1). Hypothesis: chained residual-site edits
compound — each res site subtracts its SAE's reconstruction of an already
off-distribution stream, and JumpReLU (unlike TopK) does not cap how many
latents fire, so that reconstruction is unbounded. Same family as the dense
ReLU "empty floors off-manifold to 1e8" in the cross-SAE work. att/mlp SAEs
read norm-bounded inputs, consistent with them staying sane. UNTESTED
mechanism (next: measure L0 at res sites under the edited stream).

**CORRECTION (2026-09-17): this is NOT a failure of residual sites, it is the
uncalibrated-lambda bulk regime already documented in `030-cross-sae` (dense
ReLU on Pythia-70m).** There: empty floors 2e3 (L2) to 1.8e8 (L4) against
a_pos 2-12 were treated as well-defined denominators; tri-amp still passed
11/12 (compact n=4, 29, 1,093); lambda 1e-2 was INERT (n grew — "off-manifold
floor losses drown the L1 term at home-scale lambdas") and **lambda 1.0
bit**; at bulk n (>= ~28k) nothing discriminates, nulls included — exactly
the 69k-384k circuits here at lambda 1e-3. Reporting rules carry over: on
non-TopK SAEs necessity + drive are load-bearing, zero-fill faithfulness is
quotable at compact n only. The clean-error "fix" is unnecessary. NEXT:
all three kinds, lambda in {1e-2, 1e-1, 1.0}, plus anchor support (TopK 0.2-0.3%,
dense 6.5-37%; JumpReLU at L0 ~120/16k expected in between).

**LAMBDA SWEEP (2026-09-17, `latent_circuits_amr_lam*.jsonl`,
`latent_lambda_sweep.log`) — lambda calibration does NOT rescue depth.**
Same 5 seeds, att+mlp+res, triple floor, 2 fitted nulls each.
Anchor support (030 definition, lambda-independent): L3 0.046, L6 0.052,
L9 0.038, L12 0.033, L15 0.035 (live latents/site 1.8k-3.6k, L0 at anchor
84-125) — ~15x TopK (0.2-0.3%), bottom of dense ReLU (6.5-37%): for the
null, Gemma Scope 2 behaves like dense, not TopK.

  seed    lam 1e-2                                 lam 1e-1        lam 1.0
  L3      300; F0 .28 FMd .90 FMn .80;             92 (over-      31 (over-
          fitted null pos/neg ~0, zero 1.9;        pruned)        pruned)
          sup .89 cf .17
  L6      9,195 vacuous                            20,496 vac.    13,517 vac.
  L9      71,404 vacuous                           138,175 vac.   164,008 vac.
  L12     295,867 vacuous                          280,069 vac.   283,759 vac.
  L15     290,781 vacuous                          491,948 vac.   258,373 vac.; nulls NaN

Only L3 has a working lambda (1e-2, and a narrow window). From L6 raising
lambda does not shrink circuits. 030 reached L4 only (empties ~1e8); here
empties reach 1e12-1e20 and the relative squared data term (~1e34) outbids
any sensible L1. **The earlier "clean-error fix is unnecessary" was WRONG:
the 030 recipe transfers to shallow seeds only; the residual chain needs a
structural fix.** Next: res_explosion_diag.py (empty circuit through 18
layers, current vs clean SAE error), then ERROR_MODE=clean in the fitter,
then validate that clean error leaves TopK results unchanged (033 Pythia).

**EXPLOSION DIAGNOSTIC (2026-09-17, `res_explosion_diag.py`, `.log`)** — empty
zero-fill circuit through all 18 layers, 8 x 128 corpus tokens, four modes.
Residual-stream norm vs clean:

  layer   amr/current      res/current   am/current   amr/CLEAN-ERROR
  L1      0.67             0.93          0.78         1.05
  L3      16               1.24          0.52         0.99
  L5      6.3e3            1.00          0.82         0.94
  L9      6.2e9            21            0.83         0.95
  L12     6.4e13           3.9e5         0.67         0.81
  L17     inf (from L13)   2.4e11        0.43         0.86

- **Mechanism confirmed**: under current error, residual-site L0 jumps from
  clean ~60-120 to 2,663 (L1), 8,522 (L3), up to 14,208; reconstruction gain
  ||c W_dec|| / ||x - b_dec|| = 4.75 -> 58 (>2), so the carried error flips
  and amplifies each layer.
- att/mlp edits accelerate it (amr explodes from L2; res-only from L9); att/mlp
  edits alone never destabilise the residual stream (0.35-1.22x) — why the
  att+mlp-only circuits were sane.
- **Holding each site's SAE error at its clean value keeps all three kinds
  bounded through all 18 layers (0.59-1.05x clean)**; residual L0 at or below
  clean. (att-site L0 still inflates mid-depth under clean error — up to ~9k at
  L3 — but the post-attention norm and the residual reset stop it compounding.)
- Semantic cost: with clean error, upstream edits reach later sites only
  through member codes, not through the SAE error (SFC's error-node
  convention); TuringLLM/033 recomputed the error. Must validate on 033 TopK.

**THE FIX, AND A FORM THAT LOOKED EQUIVALENT BUT WASN'T (2026-09-19).**
`floor_manifold_diag.py` ruled out the floor: `b_dec` IS the data mean (cos
0.975-0.999), a mean-filled site lands 0.01-0.03% from the mean, single-site
edits are benign (gain < 1) — and MEAN fill everywhere still diverges (inf by
L14). Root cause: our edit `x + (c_hat - c(x)) W_dec` equals
`c_hat W_dec + b_dec + err(x)`: members/floor plus the SAE error of the EDITED
stream. Each edit leaves a 3-15% residue, the next SAE re-encodes a
slightly-wrong stream, and with no cap on active latents the error feeds
itself. TopK bounds each mis-reconstruction to k decoder columns.

`ERROR_MODE=clean` (052/fit_latent_seed.py and 033/crosssae_topk.py) adds
`(err_clean - err(x))`, making the site `c_hat W_dec + b_dec + err_clean`:
members and floor EXACTLY as before (non-members at the floor, members
re-encoded, so closure is still measured — not pinning), only the error term
held at its clean value (SFC's error-node convention). Both correction terms
are bitwise zero at identity. Verified: anchor support reproduces current mode
to the digit, alpha=1 = 1.000, L9 empty circuit 2.08 / 0.00 / 0.00 (was 1.9e12).

**RETRACTED intermediate form**: `x + (c_hat - c_clean) W_dec` (subtract the
clean CODE). I called it algebraically identical — it is not: it equals the
above PLUS `(x - x_clean)`, so every upstream deviation leaks through every
non-member. The 033 TopK check caught it on 4 seeds: circuits larger on 4/4
(median 27 -> 38), drive overshoot on 4/4 (2.5 -> 3.5), one pass flipped.
All results made with it are archived as `*_deltaform*`
(latent_circuits_clean_lam1e-3/1e-2_deltaform.jsonl,
latent_clean_partial.log, latent_clean_deltaform_partial.log, 033
rows_clean_deltaform.jsonl, run_clean_deltaform.log) and must not be quoted.

**GROW FROM EMPTY, NOT PRUNE FROM FULL (2026-09-19).** With the clean error
fixing removal, deep seeds were still trapped: from the production gate init
(theta0 = 4, all open) the fit stays near "everything kept" at L6-L12
(207k-535k members, overshoot 3.9x to 7.9e12) and L15 collapses to 0 members
at EVERY lambda in {1e-2, 1e-1, 1.0} (`latent_circuits_clean_lam*.jsonl`,
`latent_clean_lambda_sweep_openstart.log`). Mechanism: "all open" and "empty"
are both stable, but the partly-closed states between them re-encode most
latents off-distribution through uncapped SAEs and run away (L12 alpha=1 alone
4.5e5), so gradients pin the fit near "all open". Starting EMPTY
(`THETA0=-4`, gates ~2% open) and growing avoids that region entirely.

Empty-start calibration, clean error, att+mlp+res, 2 fitted nulls each
(`latent_circuits_clean_theta-4_*.jsonl`, `latent_clean_theta-4_calibration*.log`):

  arm (lam, floors)   L3            L6            L9            L12            L15
  1e-3 default        123 PASS      304 PASS      324 .97/.76/.76  161 .81/.14/.05  223 .75/0/0
  3e-3 default        54 .79 FMd    153 .83/.60   131 .90/.73    46 F0 .26      65 F0 .47
  1e-3 heavy (w=1,1)  283 PASS      472 PASS      520 PASS       1406 .84/.62/.30 1452 .40/.29/.10
  3e-4 default        -             -             612 PASS       689 1.00/.27/.15 688 .89/.04/.01

- **L3, L6, L9 pass in all three frames with all three kinds** (L3 123 at
  1e-3; L6 304 at 1e-3, centred to .94/1.03/1.01 by heavy floors; L9 520 with
  heavy floors or 612 at 3e-4). Residual latents are the majority from L6
  (199/304, 358/472, 443/520).
- Fitted nulls never approach a circuit: worst 0.54 (zero-fill, L3 heavy);
  mean-frame nulls ~0 everywhere.
- 3e-3 under-grows at every depth; deep seeds want lam <= 1e-3.
- **L12-L15 not yet solved**: lam 3e-4 fixes zero-fill (1.00, 0.89), heavy
  floors lift mean-fill (to .62/.30, .29/.10) — they fix different frames, so
  the next test is combining them.
- **Necessity stays weak and sign-flips** (sup -0.67..0.57); on Pythia the
  same clean-error form kept sup at 1.00, so this looks like a property of the
  Gemma circuits (suppressors inside, or redundant routes), not the error term.

**BOUNDED RE-ENCODING (2026-09-19, `solution_diag.py`, `solution_diag*.log`).**
Every site att/mlp/res at all 18 layers edited at once, a random fraction of
latents kept at alpha=1 (0 = empty circuit), residual norm vs clean at L17:

  edit form                 keep 0%    10%      50%      90%
  current (TuringLLM/033)   inf        -        inf      -
  clean error               0.86       0.92     inf      inf
  cap (natural count)       2.8e7      7.1e6    17.4     0.93
  norm-matched (B)          nan        -        9.4e10   -
  cap + clean               0.86       0.84     2.5e5    4.1e9
  cap + clamp               2.62       2.10     0.69     0.93
  **cap + clamp + clean**   **0.86**   **0.84** **0.76** **1.15**  (max anywhere 1.83)

Three separate loops, one bound each: the count CAP (at most the token's clean
latent count, top by value) stops thousands of latents re-firing; the value
CLAMP (no latent above the token's largest clean latent) stops re-encoded
magnitudes growing with the stream; the CLEAN error stops removal feeding the
error term. All exact at identity (checked, 0.0). On TopK the cap is an exact
no-op (Top-K already fires k); the clamp is not, and the Pythia check is still
to run (`033 crosssae_topk.py` has `BOUND=clamp`). Fitter: `BOUND=cap+clamp`.

5 depth seeds, `ERROR_MODE=clean BOUND=cap+clamp`, lam 1e-3, 2 fitted nulls
(`latent_circuits_bound_theta{4,-4}_lam1e-3.jsonl`, `latent_bound_theta*.log`;
open-start necessity panel from `latent_circuits_bound_theta4_rescore.jsonl`):

  seed   start  members (a/m/r)       F0/FMd/FMn        null max  sup    sup_mean  cf
  L3     open   362 (40/169/153)      0.99/0.86/0.83 P  0.002     0.77   0.54      0.21
  L6     open   1706 (198/371/1137)   0.83/0.94/0.93 P  0.000     0.88   0.74      0.65
  L9     open   6935 (909/1481/4545)  0.84/0.72/0.66    0.72      -1.00  0.69      0.11
  L12    open   637 (65/127/445)      1.13/1.20/0.86 P  0.000     0.61   0.48      0.23
  L15    open   2876 (203/401/2272)   0.82/0.19/0.09    0.023     0.54   0.39      0.11
  L3     empty  134 (15/99/20)        1.10/0.76/0.75    0.000     0.15   0.34      0.02
  L6     empty  281 (39/43/199)       0.85/0.96/0.86 P  0.000     0.04   0.40      0.00
  L9     empty  307 (28/22/257)       0.91/0.77/0.70    0.18      -0.44  0.68      0.05
  L12    empty  162 (22/31/109)       0.77/0.12/0.04    0.000     0.55   0.77      0.04
  L15    empty  223 (23/18/182)       0.67/0.00/0.00    0.000     0.01   0.56      0.12

- **The production open start works again**: 3/5 all-frame passes (L3, L6,
  L12 — the first deep pass), nulls dead, necessity 0.61-0.88. Before the bound
  it was trapped at every depth (207k-535k members). Circuits are larger than
  empty-start ones (362-1706 vs 134-307). L9 is still trapped (6,935, fitted null
  0.72, sup -1.00); L15 fails the mean frames.
- **Empty start is sufficient but not necessary**: faithfulness like the
  clean-only empty start (bounding changes nothing in the stable region), but
  zero-ablation sup 0.01-0.55 with amplitude p90 up to 6.4 — a compact
  ALTERNATIVE route amplified to reproduce the seed, not the route the model
  uses. The open start (pruning the natural computation) keeps the real route.
- **Sign flips are a zero-ablation artifact**: members set to their negctx mean
  (sup_mean, SFC's convention) give 0.34-0.77 on every circuit, never negative,
  including L9 where zero-ablation gives -0.44 / -1.00.
- **Leak test (R4) negative for compact circuits**: sup under the current error
  = clean +-0.2 with no consistent sign (0.14 vs 0.15, -0.02 vs 0.04, ...). On
  the large open-start circuits current > clean (0.99 vs 0.88, 1.00 vs -1.00)
  but big ablations kill the seed under the current form whatever is removed,
  so this is not separable without a matched random ablation.
- **Cross-layer copy redundancy (R5) not supported**: adding every latent with
  decoder cosine >= 0.7 to a res member (210-18,622 extra) does no better than
  a matched random expansion (0.18 vs 0.18, 0.85 vs 0.72, 0.88 vs 0.86, 0.19 vs
  0.47, ...).

**THE MEAN FLOOR WAS DENSE — freeM_topk RESTORED (2026-09-19/20,
`floor_density.py`, `.log`).** Daniel: "have we ever tried topk or cap where
those not in the topk are zeroed? this feels more on-distribution". We have:
it is the engine's `_respect_topk_fill` (`src/eval/ablation_faithfulness.py:267`,
selected by `objective="free_mean_topk"`), and 027 R6a made freeM_topk the
PRIMARY metric on TuringLLM — dense and topk disagree 29x at depth there
(L9/1639 n=809: dense 0.051 vs topk 1.460). **This port had only the dense
fill.** Measured here (8 x 127 corpus tokens, dictionary 16,384):

  site      nat L0   floor nonzeros   top120 share   |decoded floor| / |natural|
  att L2    76       10,491           16%            0.48
  res L2    77        9,803           36%            0.98
  att L8    123      13,175           14%            0.42
  mlp L11   119      14,732           11%            0.30
  res L14   114      10,021           22%            0.94

So the mean frames filled 9k-15k latents per token against a natural 60-123.
Residual sites survive it (decoded floor 94-98% of natural norm, cos 0.997+
with b_dec — "an average residual stream"), but att/mlp sites are filled to
only 28-66% of natural magnitude, and that under-fill compounds over the
38-47 upstream sites of a deep seed.

`FLOOR_MODE=topk` ports the engine's fill (default stays `dense`); the budget
is the token's NATURAL latent count, i.e. the same envelope `BOUND=cap` uses,
so cap and fill are one convention. Both fills are now scored on every circuit
(`FMd_tk`, `FMn_tk`, `empty_tk`, and `pos_tk`/`neg_tk` on the fitted nulls).
Unit-checked against the engine contract (fill count = natural L0 - active
members; members never filled; filled = highest-mean non-members at their
means; members take the budget first).

Existing circuits RE-SCORED under the k-sparse fill (`*_tkfill.jsonl/.log`,
members unchanged, fitted against DENSE floors):

  seed  start   dense pos/neg   k-sparse pos/neg
  L3    open    0.86 / 0.83     0.99 / 1.00
  L6    open    0.92 / 0.89     0.80 / 0.81
  L9    open    0.73 / 0.66     0.86 / 0.88
  L12   open    1.18 / 0.86     1.26 / 1.16
  L15   open    0.19 / 0.09     0.54 / 0.35
  L3    empty   0.76 / 0.75     0.28 / 0.30
  L6    empty   0.96 / 0.86     0.92 / 0.88
  L9    empty   0.77 / 0.70     0.93 / 0.98
  L12   empty   0.12 / 0.04     0.51 / 0.23
  L15   empty   0.00 / 0.00     0.21 / 0.11

R6a's pattern transfers: the fills agree at L6 and diverge with depth, and
dense is the pessimistic one exactly where we were failing. It does not by
itself rescue L15, and L3-empty moves the other way (a small circuit leaning
on the dense fill). Fitting AGAINST the k-sparse floor is the real test
(`FLOOR_MODE=topk`, both starts) — 2026-09-20 run.

**FITTING AGAINST THE K-SPARSE FLOOR (2026-09-20, `FLOOR_MODE=topk`,
`latent_circuits_tkfloor_theta{4,-4}.jsonl`, `latent_tkfloor_theta*.log`)** —
clean error + cap+clamp, lam 1e-3, 2 fitted nulls, both starts:

  seed  start   n      free0   FMd     FMn     FM_tk   FN_tk   sup     cf
  L3    open    198    1.006   0.015   0.010   0.917   0.902   0.479   0.008
  L6    open    1208   0.897   0.002   0.000   0.639   0.692   0.797   0.234
  L9    open    7682   0.752   0.121   0.061   0.622   0.687  -0.279   0.040
  L12   open    980    0.861  -0.000   0.000   0.893   0.699   0.754   0.215
  L15   open    930    0.852   0.000   0.000   0.763   0.741   0.924   0.000
  L3    empty   90     0.986   0.000   0.000   0.809   0.849   0.093   0.006
  L6    empty   128    0.906   0.047   0.035   0.843   0.824   0.014   0.000
  L9    empty   137    0.871   0.022   0.006   0.522   0.647  -0.395   0.022
  L12   empty   82     0.958  -0.000   0.000   0.825   0.479   0.488   0.000
  L15   empty   137    0.736   0.000   0.000   0.258   0.152   0.194   0.128

- **Fitting on this floor destroys the dense frames** (0.00-0.12 everywhere):
  floor specialisation, exactly 026 R8's "a mask learns whatever its own floor
  rewards". Fit-dense/score-topk keeps both (e.g. L12 dense-fit reads dense
  1.18/0.86 AND topk 1.26/1.16).
- **Open start + topk floor is degenerate early**: the fill budget is
  natural L0 - members active, and at theta0 = 4 every latent is a member, so
  busy ~ natural L0, budget ~ 0, and the mean frames ARE the zero frame for
  most of the anneal. This is an artifact of fitting against it, not of the
  metric. The engine only ever used freeM_topk as an EVAL convention (fit
  floors there are the dense mean code).
- **Except at L15**, where the topk floor wins clearly: 930 members,
  0.85 / 0.76 / 0.74 with sup 0.92 and dead nulls, vs the dense fit's 2,876
  members and 0.54/0.35. Consistent with the mechanism (dense fill under-fills
  att/mlp by 30-60%, compounding over 47 sites).
- **L9 is seed-specific, not floor-specific**: trapped from the open start
  under BOTH floors (7,682 / 6,935 members, fitted null 0.70 / 0.72, sup < 0).
- **Empty start reaches 82-137 members with all three in-band at L3/L6**
  (0.99/0.81/0.85 and 0.91/0.84/0.82) but sup 0.01-0.09 and alpha median up to
  1.73 (p90 5.41) — compact AMPLIFIED substitutes. See the eval family below.
- Verdict: **keep the production dense floors for fitting, report freeM_topk /
  freeN_topk as the headline scores**, with the L15 exception recorded.

**THE EVAL FAMILY, AMPLITUDE-AWARE (2026-09-20, `EVAL_FAMILY=1`,
`family_*.jsonl/.log`)** — Daniel checked this port against the paper's Table 1
and it was missing rows. Added: role-aware phi_cf^a and phi_sup (roles by
attribution, sign of d seed / d w_i = grad x activation over train positives,
one backward per micro-batch), phi_sup^a (every member -> max(0, 1-a) x
natural: removes exactly the contribution the circuit CLAIMS), and phi_pin^a
(members clamped to a x their CLEAN values, non-members per fill). freeN is now
Top-K re-imposed as in Table 1; the dense negctx frame is kept as a secondary
column. NB Table 1 states "SAE reconstruction errors are preserved in every
row" — `ERROR_MODE=clean` + `BOUND` BREAK that, and the deviation has to be
declared for this substrate.
Trap found while building it: capturing the code and differentiating the seed
w.r.t. it gives EXACTLY zero gradient, because the identity transform
(c_hat - c) cancels; attribution has to differentiate a scaling weight instead.

**WHAT THE EVAL FAMILY FOUND (2026-09-20, `family_*.jsonl`, `valratio_*.jsonl`,
`pin_activators.jsonl`)**

1. **Role-aware phi_sup removes the sign flips.** Zeroing every member ignores
   role; the engine suppresses activators and INJECTS inhibitors at their negctx
   mean. Re-scored: L9 open -0.999 -> **0.623**, L9 empty -0.440 -> **0.411**,
   L15 empty 0.014 -> 0.491. On the tkfloor open-start circuits phi_sup role is
   **0.73-1.00 at every depth** (L9 1.000, L15 0.981). The circuits were fine;
   the necessity eval was wrong. Same for drive: phi_cf^a role L9 0.040 -> 1.262,
   L6 0.234 -> 0.796.
2. **Empty-start circuits stay unnecessary even role-aware** (0.06-0.53) with
   alpha medians to 1.73 — amplified substitutes, as suspected.
3. **phi_pin^a collapses on EVERY Gemma circuit** (0.00-0.39 against free0
   0.67-1.13), open and empty start alike. Two hypotheses tested and REJECTED:
   - inflation (members driving the seed above natural): measured free/clean
     member values over the whole window — median **0.00**, p90 0.8-1.0, only
     4-5% above 1.5x clean, 1-5% at the clamp. Members are SILENT, not inflated.
   - restored inhibition (pinning switches the brakes back on): pinning only
     the activators changes nothing (L6 0.145 -> 0.142, L15 0.011 -> 0.012).
   So a MINORITY of members that do fire drive the seed from 0.00 to 0.83, and
   giving all members their true values drops it to 0.15.
4. **CONTROL (033 README): on Pythia Top-K, free and pinned AGREE** — pin0
   median 0.789, 24/28 seeds >= 0.5, no depth decay (L1 0.97 ... L5 0.75).
   **The collapse is the uncapped JumpReLU substrate, not the method.**
   cap+clamp buys stability, not the identity of the member values.

Consequence: free scores alone OVERSTATE these Gemma circuits. Report phi_pin
beside every free score. Substrate options: keep Gemma as the honest "what
breaks on uncapped dictionaries" result, and/or train our own Top-K SAEs for
Gemma 3 270M (the backup plan from 2026-09-17) — Gemma Scope 2's transcoders and
CLTs are JumpReLU too, so they would NOT fix this.

**SPARSITY (2026-09-17, `sparsity_stats.py`, `.log`, `.json`)** — 16 x 256
clean corpus tokens, BOS excluded, dictionary 16,384:

  kind  mean L0  % dict   p5   p95   max   >=50%-on latents  ever on
  att   111      0.68%    39   204   510   0                 13,651
  mlp   102      0.62%    48   167   542   0                 14,055
  res   120      0.73%    74   169   372   0                 14,368

- NOT fixed but not unconstrained: p5 ~40-75, p95 ~170-200, individual tokens
  up to 2,761 (res L1) / 5,345 (mlp L4) on CLEAN text.
- **No always-on latents**: 0 latents fire on >=50% of tokens at any of the 54
  sites — like TopK, WHICH latents fire varies; unlike TopK, HOW MANY does too.
- Per token vs other dictionaries: TuringLLM TopK 128/40,960 = 0.31%; Pythia
  TopK 16/32,768 = 0.05%; dense ReLU (030) 6.5-37%. Gemma Scope 2 at 0.7% is a
  few x denser than our TopK dictionaries and 10-50x sparser than dense ReLU.
- **CORRECTION to the lambda-sweep note above**: "anchor support 3.3-5.2% =>
  behaves like dense" was WRONG — that used a narrower denominator (latents
  live AT THE ANCHOR across the 48 contexts) than 033's (live anywhere in the
  probe stream). Against the dictionary the rate is ~0.7-0.9%, near TopK's
  0.31%, far from dense's 6.5-37%. These SAEs ARE sparse in-distribution.
- The real difference from TopK is the missing CAP, and it only bites
  off-distribution: clean ~120 -> 2,663-14,208 under circuit edits
  (res_explosion_diag). Discovery is not what breaks; the INTERVENTION breaks,
  and the broken counterfactual then drives the fit.

No TopK SAEs exist for Gemma 3 270M (HF Hub search 2026-09-17): Gemma Scope 2
is JumpReLU; `mindchain/functiongemma-270m-sae` is JumpReLU, o_proj only,
4,096 wide, L0 399-3,658, on the FunctionGemma fine-tune;
`tim-lawson/pretrain_gemma3_c4_kl_270m_1b-pt_topk_*` are full LM checkpoints
(top-k logit distillation), not SAEs.

**att+mlp upstream only — sane and selective** (`latent_circuits_am.jsonl`):

| seed (res) | members (att/mlp) | F0 / FMd / FMn | fitted null F0/FMd/FMn | a=1 | sup | cf | empty zero |
|---|---|---|---|---|---|---|---|
| L3 #9665 | 395 (32/363) | 0.83 / 0.60 / 0.59 | 0.02 / 0.00 / 0.00 | 0.45 | 0.73 | 0.22 | 0.80 |
| L6 #10310 | 1,152 (260/892) | 0.74 / 0.78 / 0.64 | 0.00 / 0.07 / 0.01 | 0.00 | 1.00 | 0.35 | 0.00 |
| L9 #12281 | 2,657 (646/2,011) | 0.63 / 0.56 / 0.31 | 0.07 / 0.00 / 0.00 | 0.34 | 1.00 | 0.00 | 0.00 |
| L12 #2111 | 4,277 (1,452/2,825) | 1.19 / 0.94 / 0.97 | **0.69** / 0.00 / 0.02 | 1.12 | 1.00 | 0.00 | 53.1 |
| L15 #6828 | 7,121 (2,379/4,742) | 1.13 / 1.12 / 0.67 | 0.12 / 0.00 / 0.00 | 0.03 | 1.00 | 0.00 | 19.3 |

- Every seed beats its fitted null in every frame, except L12 zero-fill
  (null 0.69: only the mean frames separate real from random there).
- Size grows ~1.7x per 3 layers (395 -> 7,121): the TuringLLM depth gap again.
- ALL-PASS ([0.8, 1.25] in all three frames): **1/5** (L12) at the
  uncalibrated TuringLLM lam/weights. TuringLLM panel was 17/21 after
  calibration.
- Necessity clean (sup 1.00 at L6+); drive ~0 at L9+ (closure without drive,
  as on TuringLLM L9 and the cross-SAE attn result).
- DEFINITIONAL CHANGE vs TuringLLM: no residual-stream members.

Seed contexts (random seeds, not chosen for interest): L3 "most successful /
grand / gown ... fold"; L6 "Coined by English psychologist J."; L9 product
and software names; L12 build-config text; L15 product descriptions.

## Next: Stage 1 (adapters)
The main pipeline is TuringLLM-coupled at: `Inference` (hard-instantiates
TuringLLM), `hooks.multi_patch` (walks `model.transformer.h`), `SAEBank`
(hardcoded kinds, TopK `encode -> (top_acts, top_idx)` contract). Stage 2 =
corpus shards + discovery artifacts (latent_stats, top/mid/neg ctx,
seq_repr, logit_ctx, top_coactivation, candidates).
