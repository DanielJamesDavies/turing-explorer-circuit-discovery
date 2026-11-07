# 081 — Task circuits on GPT-2 small: WCM with an output endpoint on the original task datasets

Reworked after Daniel's review. Linear **DAN-177**. Companion to **080** (latent- vs output-targeted
circuits).

**Question.** Does weighted circuit masking (WCM), run under protocol-v1 conventions with the model's OUTPUT as the
endpoint, find faithful and small circuits for the standard behavioural tasks of the circuits literature? And does what
it finds look like the circuits the papers published? It tests the paper's two claims on behavioural endpoints:
- **claim 1:** WCM finds faithful circuits, and random circuits with fitted coefficients fail;
- **claim 2:** fitted coefficients keep circuits small.

**Why GPT-2 small (changed in the rework).** 047 found that TuringLLM has neither IOI (p(IO) ~ 1e-4) nor induction (no
in-context copying), so two of the canonical tasks could only be reported as absent. GPT-2 small is the model the
published IOI (Wang et al.) and greater-than (Hanna et al.) circuits were found in, so structure can be checked against
them head by head. `MODEL=gpt2-small` is the default. `MODEL=turingllm` still works (tasks are generated for it too),
but greater-than does not transfer: Hanna's task needs two-digit year tokens, and TuringLLM splits digits.

**How it complements 080.** 080 fits output-endpoint circuits on each latent target's own contexts (upstream scope,
the actual next token) and compares them with the latent circuits. 081 fits them on task prompts (every SAE site in
scope, a contrastive task metric) and checks them against published circuits. The two share the engine changes, the
`gap` normaliser and the fit conventions. 081 imports 080's `common.py` for the matched-size helpers.

**Status.** Code, tests, data and task generation are done (CPU). Daniel's decision: full datasets with
their natural split instead of 320-item subsamples, at protocol v1's unchanged budget of 400 steps x batch 4 (see
**Splits**). **First GPU run, under protocol v2, on IOI, greater-than and some secondary tasks; see
"v2 local run" at the end. 081 now runs under WCM_VERSION=v2 by default.**

## Model and dictionaries

`model/families.py` "gpt2-small": HF `gpt2` through `model.hf_adapters` (no TransformerLens), plus
`jbloom/GPT2-Small-OAI-v5-32k-{attn-out, resid-mid, mlp-out, resid-post}-SAEs` via `sae.public_bank.PublicSAEBank`:
- Top-K, k = 32, 32,768 latents;
- 4 kinds x 12 layers = **48 sites**;
- per-token layer-normalised input (053 transcribed and verified this).

The full bank is in WSL's HF cache; nothing is downloaded. Precision is float32 by default (053's), which means about
9.7 GB of dictionaries resident; `DTYPE=bfloat16` halves that.

**Attention SAE hook point (checked in cfg.json).** `hook_name: blocks.L.hook_attn_out`, `hook_head_index: null`. The
attn SAEs read the **summed post-W_O output** of all 12 heads, so no latent belongs to a head by construction. Heads can
still be resolved exactly. The site input is x_n = (a − mean a) / (std a + eps), with a = Σ_h o_h + b_O and o_h = z_h W_O[h]
(z = the input of `attn.c_proj`). Centring is linear and the scale is shared by all heads at a token, so a latent's
pre-activation splits exactly:

pre_i = Σ_h ((o_h − mean o_h) / s) · w_i + ((b_O − mean b_O) / s) · w_i + b_eff,i.

`engine.head_drive` computes the per-head terms, and a test checks they sum to the SAE's own pre-activation. The
structure check attributes each attention member's attribution to heads in proportion to |head drive|. This is a
decomposition of the member's pre-activation, not of its downstream effect.

## Tasks (originals; counts under GPT-2's tokenizer)

`make_tasks.py` (CPU, tokenizer only) writes `tasks/<MODEL>/<name>.json`. Each row holds:
- the prompt;
- a role label per token;
- the answers (token **sequences**, or token **sets** for greater-than);
- the dataset's own patch / counterfactual prompt, where one exists;
- the split.

Prompts are right-padded at load time with a per-row anchor, never left-padded (047's padding trap). No BOS is
prepended (the SAEs were trained without one). Downloaded code is read as data: list literals are extracted with `ast`
and nothing is imported or executed.

Counts are the latest regeneration (`make_tasks.py`, GPT-2 tokenizer): train = the whole training pool, held = the
held-out set after the 5,000 cap (uncapped size in brackets where the cap bites).

| task | source (original) | prompt -> good vs bad | C (contrast) | split | train / held | groups | len |
|---|---|---|---|---|---|---|---|
| ioi | Wang et al. 2022, Easy-Transformer `ioi_dataset.py` | 15 BABA templates + the file's ABBA swap, "mixed" -> IO vs S | **ABC** (IO, S1, S2 -> 3 fresh names; the notebook's flips) | generated, disjoint list halves | 1,600 / 5,000 | 6,579 | 14-20 |
| greater_than | Hanna et al. 2023, `gpt2-greater-than` (+ ACDC's noun list) | "The {noun} lasted from the year {XXYY} to the year {XX}" -> YY' > YY vs YY' <= YY | **XX01** (Hanna's) | generated, disjoint noun halves | 1,600 / 5,000 | 6,600 | 12 |
| sva_simple | Marks et al. 2025, `feature-circuits/data/simple_*` | "The kids" -> do vs does | patch_prefix (opposite number) | SFC's files | 150 / 85 | 40 | 2-4 |
| sva_within_rc | `within_rc_*` | "The farmers that the athlete" -> RC verb | patch_prefix | SFC's files | 3,097 / 1,728 | 800 | 5-8 |
| sva_across_rc | `rc_*` | "The lawyer that the guards hate" -> has vs have | patch_prefix | SFC's files | 34,003 / 5,000 (18,933) | 8,800 | 6-10 |
| sva_across_pp | `nounpp_*` | "The teachers near the desk" -> are vs is | patch_prefix | SFC's files | 1,727 / 469 | 1,545 | 5-7 |
| gender_pronoun | Mathwin et al. 2023, hackathon notebook | 5 templates x 20 names (the whole original set) -> he vs she | "That person ..." (their baseline) | 75 / 25 by prompt | 75 / 25 | 100 | 10 |
| factual_known | Meng et al. 2022, ROME `known_1000.json` | prompt -> true object vs another object of the same relation | **generated** | 75 / 25 by subject | 908 / 300 | 1,194 | 4-27 |
| induction | generated (Olsson et al. 2022) | R + R[:j+1] (16 random whole-word tokens) -> R[j+1] vs R'[j+1] | R' + R[:j+1] | generated, disjoint token halves | 1,600 / 5,000 | 6,600 | 19-31 |
| blimp_* (8 subsets) | BLiMP (Warstadt et al. 2020) | shared prefix -> critical word | **generated** | 75 / 25 by prefix | 690-767 / 233-251 | 473-1,000 | 1-15 |

BLiMP per subset (train / held): anaphor_gender_agreement 750 / 250, anaphor_number_agreement 750 / 250, npi_present_1
767 / 233, npi_present_2 751 / 249, principle_A_c_command 750 / 250, principle_A_case_1 749 / 251, wh_island 750 / 250,
irregular_past_participle_verbs 690 / 242.

**Drops (GPT-2).** SFC: identical items (same prefix and answer) are kept once per file: simple 600 records -> 235
items (365 duplicates), within_rc 12,000 -> 4,825 (7,175), rc 132,000 -> 52,936 (79,064), nounpp 2,400 -> 2,196 (204).
ROME: 1 fact dropped (no other object of its relation), 1,208 kept. Generated tasks: duplicate draws only (IOI 9,
greater-than 420). BLiMP: irregular_past_participle_verbs has 68 non-qualifying pairs (multi-token critical word).

Notes:
- **Single- vs multi-token answers.** Under GPT-2 every answer is a single token: IOI names (Wang's list), SFC verbs,
  he / she, all 1,209 known_1000 objects (checked) and BLiMP critical words (by the selection rule). So every GPT-2 task
  runs in token mode (greater-than in set mode). The multi-token path is implemented and tested, and it is exercised
  under `MODEL=turingllm`: factual_known there has 306 multi-token good answers and 458 multi-token bad ones out of
  1,208.
- **Single-token drops (GPT-2).** None for the GPT-2 tasks. Under TuringLLM, 275,221 IOI draws were rejected for
  multi-token names and 68,379 for a counterfactual length mismatch (the 1,600 / 5,000 items are still reached), and 20
  gender rows for length mismatch.
- **Splits.** See **Splits** below: SFC's own files, 75 / 25 by group for fixed datasets without a split, disjoint list
  halves for generated tasks, every held-out set capped at 5,000.
- **Small sources.** sva_simple (150 / 85) and gender_pronoun (75 / 25) are small because the original datasets are.
- **Greater-than details.**
  - Years: 1100-1799 (ACDC's centuries 11-17) under Hanna's validity rule (`get_valid_years`: " XXYY" must tokenise as
    [" XX", "YY"]; the first and last valid year of each century are dropped).
  - YY is balanced over 02..98, cycling as `YearDataset(balanced=True)` does.
  - Answer tokens are the stand-alone encodings of "00".."99" (all single GPT-2 tokens), as Hanna / ACDC define them.
- **Gendered pronoun.** These are the original data, found in the hackathon notebook. Their baseline was built only for
  the first template; here it is applied to all five. Their notebook used a BOS; this experiment does not.
- **Dropped from the first version:**
  - the from-memory template recreations, replaced by the originals;
  - the hand-written capitals task, replaced by ROME's `known_1000.json`;
  - the "answer-flipped prefix" BLiMP counterfactual, replaced by the generated contrast set.
- **MIB's IOI split: not added.** The authors' own templates, names and ABC distribution are used directly, so a
  further benchmark split would add a second IOI variant without adding a published circuit to compare against.

## Splits (full datasets, natural split)

**Decision (Daniel).** Use every item of each dataset with its natural split, and keep protocol v1's
training budget of 400 steps x batch 4.

**What other circuit papers do:**
- **Edge Pruning** (Bhaskar et al. 2024, arXiv 2406.16778) trains on 150-200 examples and tests on IOI 36,084,
  greater-than 12,240 and gendered pronoun 378.
- **ACDC** (Conmy et al. 2023, arXiv 2304.14997) uses 50-100 examples: IOI N = 50, greater-than 100, gendered pronoun
  100, induction 40 sequences.
- **MIB** (Mueller et al. 2025, arXiv 2504.13151) has IOI 10,000 train / 1,000 public + 1,000 private test. The private
  test set uses held-out names, which is the model for the disjoint list halves below.
- **SFC** (Marks et al., arXiv 2403.19647) releases an 80 / 20 train / test split. From its data files: simple 480 /
  120, within_rc 9,600 / 2,400, rc 105,600 / 26,400, nounpp 1,920 / 480 records (checked: the line counts match).

So training sets in this literature are small (50-200) and test sets large. The fixed budget here, 400 x 4 = **1,600
presentations per fit**, means a training pool of >= 1,600 items is never repeated in training. Smaller pools are
cycled in reshuffled epochs.

**Training pool = the whole training split** (`engine.presentations`, `task_common.presentation_order`):
- The 1,600 presentations are drawn **without replacement** until the pool is exhausted, then reshuffled. Each epoch
  is a fresh seeded permutation; the seed is the task name, so every fit of a task sees the same stream.
- The rows are handed to the mask engine in that order with `holdout_frac = 0`. The engine's micro-batch draw (`s =
  step x batch mod rows`) then walks the 1,600 rows exactly once. No `src/` change is needed for the draw.
- **A** = the mean over the **whole training pool**. **C** = the mean over the **whole contrast pool**: the patch
  prompts of every training item, or the generated contrast pool. Both are averaged over real prompt tokens, computed
  once per task (`engine.means`, chunks of 64 prompts) and passed to the engine and the scorer
  (`floor_means` / `pos_floor_means`, `means_A` / `means_C`).
- Before the rework the engine's internal 25% holdout also took a quarter of the training items out of training and out
  of the A and C means.

**Held-out sets:**
- **SFC tasks: SFC's own files** (`SFC_SPLIT=official`, the default): `*_train.json` is training and `*_test.json` is
  held out. Identical items (same prefix and answer) are kept once per file.
  - **FLAG: SFC's test files are not disjoint from its train files.** Every test prefix of simple, within_rc and rc
    occurs in its train file. Most test items are identical to a training item: simple 80 of 85, within_rc 1,667 of
    1,728, rc 18,202 of 18,933 (before the cap); nounpp 82 of 469 (248 share a prompt).
  - These overlaps are counted in each task file (`overlap`) and printed by `make_tasks.py`, but not removed: the
    official split was the decision.
  - `SFC_SPLIT=prefix` gives a prompt-disjoint alternative: both files pooled, then 75 / 25 by prefix. It is one
    `make_tasks.py` run (then re-run prepare / probe) if the leak matters for the held-out claims.
- **Fixed datasets with no official split** (each BLiMP subset's qualifying pairs, ROME known_1000's 1,208 usable
  facts, Mathwin's 100): **75 / 25 of all items**, by group (prefix; subject; prompt), seed 20261019.
- **Generated tasks (IOI, greater-than, induction): 1,600 training and 5,000 held-out items.** The source lists are
  split into disjoint halves (`task_common.split_list`, 50 / 50, because the held-out set is the larger one):

  | task | split into disjoint train / held halves | shared |
  |---|---|---|
  | ioi | NAMES 49 / 50, PLACES 4 / 4, OBJECTS 4 / 4. The ABC counterfactual names come from the item's own half | the 15 templates and their ABBA swaps (the task itself) |
  | greater_than | the 120 nouns, 60 / 60 | the 617 valid years: YY is the compared variable and must stay balanced over 02..98 on both sides, and each YY has only ~6 valid years |
  | induction | the 12,017 whole-word tokens, 6,009 / 6,008; R and R' both come from the item's half | none |

  No list was too short to split; the shortest, IOI's PLACES and OBJECTS, have 8 items each. The tests check that no
  held-out name, place, object, noun or induction token occurs in training.
- **Cap:** every held-out set is capped at **5,000** by fixed-seed sampling (`task_common.cap_held`). It bites only for
  sva_across_rc (18,933 -> 5,000).
  - The uncapped size is in each task file (`split_meta.n_held_uncapped`, `counts.held_dropped_by_cap`).
  - Capped-out rows are not written to the task file.
  - `HELDOUT_CAP=0 make_tasks.py` writes them all (for an H100 pod); any other value sets the cap.
- **Calibration** never sees a held-out item. It re-splits each task's training pool 75 / 25 by group; the validation
  side is capped like a held-out set.
- **Smoke** (`HELDOUT_N`): only the first N held-out items are used at run time. The held-out order is seeded, so they
  are a fixed random subset.

### Provenance (each folder under `data/external/` has a SOURCE.md; `/data/` is git-ignored)

| folder | source | commit / revision | licence | files (size) | sha256 |
|---|---|---|---|---|---|
| `ioi_easy_transformer` | github.com/redwoodresearch/Easy-Transformer `easy_transformer/ioi_dataset.py` | `ea15315dd24481e9e2ac5c3ef335d82907a1dc34` | MIT | ioi_dataset.py (31,437 B) | `94a72de9…a98de` |
| `greater_than_hanna` | github.com/hannamw/gpt2-greater-than `dataset.py`, `utils.py`; nouns from ArthurConmy/Automatic-Circuit-Discovery `acdc/greaterthan/utils.py` | `0e6dc8c2da87f7d2befeb2a4dd432fae404ea35a`; ACDC `bc99ace817974b5584b7ee203d596a8e2bbcd399` | hannamw: **none stated**; ACDC: MIT | 4,547 B; 21,111 B; 13,515 B | `e46a75ed…ef9d7`; `d3bede17…c677`; `3d64f5bb…d986` |
| `sfc_feature_circuits` | github.com/saprmarks/feature-circuits `data/{simple,within_rc,rc,nounpp}_{train,test}.json` | `7fbd82b895ae16294f4e6fc7bfc675d1d680d659` | MIT | 8 files, 26.9 MB total (largest rc_train 19.5 MB) | per file in SOURCE.md |
| `mathwin_gendered_pronoun` | cmathw.itch.io hackathon upload: `Automatic_Circuits_Script.ipynb` (+ `Attention_Visualisation_Script.ipynb`) | itch upload ids 7208110 / 7208109 (no versioning) | **none stated** | 13,839 B; 22,360 B | `158a328c…95071`; `d5776ac8…67fb2` |
| `rome_known_1000` | rome.baulab.info/data/dsets/known_1000.json (kmeng01/rome) | repo `0874014cd9837e4365f3e6f3c71400ef11509e04`; file Last-Modified 2022-05-21 | MIT | 343,229 B | `61daca55…f333a` |
| `blimp` | huggingface.co/datasets/nyu-mll/blimp (downloaded 2026-10-19) | `877fba0801ffb7cbd8c39c1ff314a46f053f6036` | CC BY 4.0 | 67 parquet, 3,895,588 B | `7ac5984b…fb89e` (sorted per-file list) |
| `gpt2-tokenizer` | copy of the local HF cache's `gpt2` tokenizer files (not a download) | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | MIT | 5 files, 2.85 MB | per file in SOURCE.md |

Every download was size-checked first; all are under the 50 MB cap.

## Metrics

All metrics are read at the prompt's last token, and higher means more correct (`eval.wcm_eval.task_metric`):

| kind | value | tasks |
|---|---|---|
| `logit_diff` | logit(good) − logit(bad) = log p(good) − log p(bad) | IOI, agreement, gender, facts, induction, BLiMP |
| `set_probdiff` | P(good set) − P(bad set) | greater-than: task metric (Hanna's probability difference P(> YY) − P(≤ YY)) |
| `set_logmass` | log P(good set) − log P(bad set) | greater-than: the fitted objective (engine `logit_metric="set"`) |
| `seq_diff` | log p(good seq) − log p(bad seq), log p(seq) = Σ_k log p(a_k \| prompt, a_<k) | any task with a multi-token answer |

**Greater-than.** The pass rule is applied to Hanna's probability difference. The fit reproduces its log-odds form,
which is bounded and has the same sign. Both are scored on every circuit (`task_*` and `obj_*` fields).

**Multi-token answers** are teacher-forced:
- each item becomes a (good, bad) row pair, each row = prompt + its answer[:-1];
- log-probs are read at positions anchor + k and summed;
- the pair's difference is the item's value.

In the engine (`logit_metric="seq"`) the value is written on both rows of the pair, so every per-row computation (targets,
squared errors, means) is unchanged and a mean over rows is a mean over items. To keep pairs whole, the engine requires
an even row count and an even micro-batch, and it rounds the holdout down to an even count. Valid masks cover prompt
tokens only, so the A and C means never include teacher-forced answer tokens.

### Competence rule (pre-stated; `probe.py`, `task_common.competence`)

On the held-out items, with the full model, a task is COMPETENT iff all of:
- (a) the mean task metric is > 0;
- (b) at least 70% of items have task metric > 0;
- (c) for tasks with their own counterfactuals only: mean(clean) − mean(counterfactual) > 0, scored for the clean
  item's answers.

Criterion (c) is not applicable to generated-contrast tasks, whose contrast prompts are silent by construction. Only
competent tasks are calibrated and fitted.

## Protocol mapping (protocol v1 -> tasks)

| protocol v1, latent endpoint | 081, task endpoint |
|---|---|
| reproduce the target latent's pre-activation | reproduce the task objective at the anchor (squared error against the full model) |
| maskable sites: upstream of the target | **all 48 SAE sites**: the output is the endpoint, as in SFC |
| contexts: 32 strongest + 16 mid-band train, 16 held-out | the whole training pool (75-34,003 items) and its held-out set (25-5,000), see **Splits** |
| Z: non-members at 0 | the same, at every site |
| A: mean over the activating (training) contexts | mean over the **whole training pool's clean prompts** (real prompt tokens only) |
| C: mean over close, verified-silent contrast contexts | the dataset's own **patch prompts** of every training item (IOI ABC, greater-than XX01, SFC patch_prefix, gender "That person", induction R'), or the **generated close-silent contrast pool** (below) |
| γ_A = γ_C = 0.25, free coefficients, AdamW, 400 steps, batch 4, binarisation, keep threshold | unchanged (059's `protocol_harness.configure`); the 1,600 presentations drawn from the pool without replacement within an epoch |
| rank-keep 3e-3 | off (it is about a target latent's rank) |
| λ = 1e-3 | a grid from the task calibration (`config_main_<MODEL>.json`) |
| loss scale mean(target²) | `logit_norm="gap"`, as 080 |
| lr 0.05, wd 0.05 | lr from the run config (080's adopted lr if it exists, else 0.05); wd rescaled so lr × wd = 0.05 × 0.05 |
| pass rule: Z, A_sp, C_sp in [0.8, 1.5], necessity ≥ 0.9, held-out | the same, on the task metric |
| scored with the fitted coefficients | always (the keep_scales trap); Z at α = 1 also recorded |

**Padding and valid masks.** Prompts in a batch differ in length and are right-padded. The anchor read is exact for a
causal model, but a pooled mean would average in padding, so every A / C mean (training and scoring) averages over real
prompt tokens only (`collect_site_means(valid=...)`, the engine's `pos_valid` / `neg_valid`).

## Contrast sets for tasks without patch prompts (Daniel's rule applied to outputs — FLAGGED)

ROME's facts and BLiMP's minimal pairs have no counterfactual prompt. Their C ablation uses a contrast pool generated
from the same dataset by `prepare.py`. It mirrors protocol v1's "close, verified silent" rule for latents:
- **candidates:** the other TRAINING prompts of the same dataset or subset (since the rework the whole training pool:
  690-908 prompts under GPT-2). A held-out prompt never enters C, and neither does a prompt from the item's own group.
  - The embeddings are computed once per task.
  - The cosine ranking is built in row chunks and keeps each item's 32 closest.
  - The candidate metrics read one chunk of candidates' [rows, vocab] log-probs at a time, so memory stays bounded.
- **ranking:** cosine similarity of the mean-pooled final-layer residual (the last layer's resid site activation over
  real tokens; per-token layer-normalised on GPT-2).
  - This mirrors `src/utils/neg_context_selector.py`'s close selector.
  - That selector can't be reused directly: it ranks corpus-store sequences by stored representations, not arbitrary
    prompts.
- **silent:** on the candidate prompt the model does NOT prefer the item's good answer over its bad one,
  **metric(good_i, bad_i | candidate) ≤ 0**. Multi-token answers use the summed log-probs. Silence is measured for each
  item's 32 closest candidates.
- **pool:** round robin over the training items in a seeded order. Each item adds its closest silent candidate not
  already pooled, until **64** distinct prompts (protocol v1 also uses 64 contrast contexts).
- **output:** `tasks/<MODEL>/<task>.contrast.json`, with diagnostics: the silent share, items without a silent
  candidate, and the median rank and cosine.

The threshold (≤ 0, i.e. "not preferred") is Daniel's latent rule carried over to outputs. It is a design choice and
needs his review.

**Sensitivity arm W_ZA.** For every generated-contrast task, W is also fitted on Z + A only, with the C term's weight set
to 0. It is scored identically (C still reported) and read under both the full rule and the Z + A rule. Daniel wants
both tested.

## Arms

| arm | coefficients | members | where |
|---|---|---|---|
| **W** | fitted α | the mask's choice at each λ | the W grid |
| **U** | α = 1 | the mask's choice | the U grid (claim 2) |
| **RND** | fitted α, support restricted | random, matched to W's per-site counts at each W λ, drawn from latents active on the task's training prompts | 5 draws per W λ; 062 `random_null.py`'s recipe (no penalty, gates open, no binarisation) |
| **W_ZA** | fitted α, trained on Z + A only | the mask's choice | generated-contrast tasks, at W's λ |

## BLiMP

**Provenance.** See the table above, or `data/external/blimp/SOURCE.md`.

**Single-critical-token rule** (`blimp.critical`, decided on the MODEL's tokens). A pair qualifies when:
- the two token sequences share a non-empty identical prefix;
- the first differing token on each side is a whole word (it starts a word, and the next token starts a new word, is
  punctuation, or is the end; GPT-2's "Ġ" or SentencePiece's "▁");
- BLiMP's own `one_prefix_method` flag is set (its authors' judgement that the prefix decides the critical word).

Anything after the critical word may differ. The prompt is the shared prefix, and the metric is logit(good) −
logit(bad). A subset qualifies when ≥ 80% of its pairs do.

**Qualifying under GPT-2: 13 of 67** (qualifying pairs; distinct prefixes):

| subset | phenomenon | qual. | pairs | prefixes | selected |
|---|---|---|---|---|---|
| anaphor_gender_agreement | anaphor agreement | 100% | 1,000 | 1,000 | yes |
| anaphor_number_agreement | anaphor agreement | 100% | 1,000 | 1,000 | yes |
| npi_present_1 | NPI licensing | 100% | 1,000 | 836 | yes |
| npi_present_2 | NPI licensing | 100% | 1,000 | 833 | yes |
| principle_A_c_command | binding | 100% | 1,000 | 1,000 | yes |
| principle_A_case_1 | binding | 100% | 1,000 | 940 | yes |
| principle_A_domain_1 | binding | 100% | 1,000 | 1,000 | (cap: 2 per phenomenon) |
| principle_A_domain_2 | binding | 100% | 1,000 | 1,000 | (cap) |
| wh_island | island effects | 100% | 1,000 | 998 | yes |
| irregular_past_participle_verbs | irregular forms | 93% | 932 | 473 | yes |
| determiner_noun_agreement_1 | determiner-noun agreement | 88% | 881 | 881 | (cap: 8 subsets) |
| determiner_noun_agreement_with_adjective_1 | determiner-noun agreement | 87% | 872 | 872 | (cap) |
| principle_A_case_2 | binding | 86% | 856 | 813 | (cap) |

**Subject-verb agreement still does not qualify**, because plurals and contractions split inside the word:
- regular_plural_subject_verb_agreement_1: 62%;
- distractor_agreement_relative_clause: 59%;
- distractor_agreement_relational_noun: 52%;
- irregular_plural_subject_verb_agreement_1: 47%.

SFC's own agreement data covers that phenomenon instead.

**Selection rule:**
- subsets with ≥ 160 qualifying pairs and ≥ 40 distinct prefixes;
- ranked by qualifying pairs;
- at most 2 per phenomenon;
- at most 8 in total;
- every qualifying pair of each (since the rework), split 75 / 25 by prefix.

That gives 8 subsets over 5 phenomena. **Contrast:** BLiMP pairs share their prefix, so C is the generated pool.

## Registered predictions (re-registered before any run)

These supersede the first registration. That one was written for TuringLLM and the from-memory tasks; no run
happened under it. `report.py` prints every verdict.
- **T1. Competent tasks get W circuits that pass the v1 rule at some λ.** Held-out items, full rule (Z, A_sp, C_sp,
  necessity).
  - SUPPORTED if every competent task passes; MIXED if at least half do; NOT SUPPORTED otherwise.
  - For generated-contrast tasks, the W_ZA arm and the Z + A reading are reported alongside, not substituted.
- **T2. U needs substantially more latents at matched faithfulness and passes less at matched size** (claim 2 on
  behavioural endpoints). Over competent tasks where W reaches task Z ≥ 0.8:
  - U needs ≥ 2× W's latents to reach Z ≥ 0.8 (first crossing, linear in log size; never reaching it counts as
    infinite) in ≥ 75% of tasks;
  - AND the median ratio is ≥ 2;
  - AND U, interpolated at W's smallest passing size, fails in ≥ 75% of tasks with a passing W.
  - SUPPORTED if all three hold.
- **T3. The random nulls fail.** SUPPORTED if the pass rate over all RND draws is ≤ 5%, AND, for every task, the draws'
  median task Z is below W's at the same λ in most λ.
- **T4. The structure matches the published GPT-2 circuits.** The check runs on each task's primary W circuit (the
  smallest passing one; otherwise the best worst-ablation one, flagged).
  - A head class counts as "found" when its share of the attention members' attribution, distributed over heads by
    |head drive|, within its layers, is ≥ **2×** the uniform base rate (class heads / all heads in those layers).
  - **IOI** (Wang et al.), at the stated positions:
    - name movers 9.6, 9.9, 10.0 and S-inhibition 7.3, 7.9, 8.6, 8.10 at END;
    - induction 5.5, 6.9 and duplicate-token 0.1, 3.0 at S2;
    - previous-token 2.2, 4.11 at S1+1.
  - **Greater-than** (Hanna et al.):
    - heads 5.1, 5.5, 6.1, 6.9, 7.10, 8.8, 8.11, 9.1 at END, at ≥ 2× base;
    - MLP members in layers 8-11 carry ≥ 50% of the MLP members' |attribution| at END (base rate 4/12);
    - YY-concentrated members sit earlier than END-concentrated members (bootstrap CI of the median-layer difference
      below 0).
  - SUPPORTED if every evaluable IOI and greater-than check holds; MIXED if at least half do; NOT SUPPORTED otherwise.
  - The agreement checks (subject-position members before prediction-position members; agreeing noun > distractor;
    subject share ≥ 10%) are reported as **secondary**. SFC's circuits are Pythia / Gemma circuits, not GPT-2's.
- **T5 (descriptive, no hypothesis test).**
  - Under `MODEL=turingllm`: how many of the primary W circuit's top 20 members by |anchor attribution| are 062
    production targets with their own latent circuits, and the overlap with the union of those circuits.
  - Under GPT-2, which has no 062 targets: the top members are listed with their Neuronpedia ids, for interpretation.

## How to run (in order)

All commands run from the repo root. GPU stages use WSL's `.venv` and the queue pattern of 080.

```bash
# 0. tasks (CPU; already done, deterministic): tasks/<MODEL>/*.json, summary.json, blimp_survey.json
PYTHONPATH=src python -X utf8 experiments/081-task-circuits/make_tasks.py                 # MODEL=gpt2-small

# 1. smoke (GPU; ioi + one BLiMP subset, 12 steps, 200 held-out items, every stage incl. prepare; ~15-25 min, an
#    estimate)
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/081-task-circuits/smoke.sh'

# 2. prepare (contrast sets) -> probe -> calibration (GPU, resumable), then read probe/gpt2-small/competence.json
#    and calib/gpt2-small/report.txt
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/081-task-circuits/queue_calib.sh > experiments/081-task-circuits/queue_calib.log 2>&1 < /dev/null & disown'

# 3. adopt the calibration explicitly (CPU; writes config_main_gpt2-small.json; FORCE=1 to adopt despite warnings)
ADOPT=1 PYTHONPATH=src python experiments/081-task-circuits/calibrate.py

# 4. fits (W, U, RND, W_ZA, profiles) -> report -> figure (GPU, resumable)
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && nohup setsid bash experiments/081-task-circuits/queue_local.sh > experiments/081-task-circuits/queue_local.log 2>&1 < /dev/null & disown'

# 5. report / figure any time (CPU); the paper figure at 5.5 in
PYTHONPATH=src python -X utf8 experiments/081-task-circuits/report.py
PAPER=1 PYTHONPATH=src python -X utf8 experiments/081-task-circuits/figure.py

# tests (CPU); 080 and 081 also pass in one session
PYTHONPATH=src python -m pytest experiments/081-task-circuits -q
PYTHONPATH=src python -m pytest experiments/080-latent-vs-output experiments/081-task-circuits -q
```

Useful variables:
- every script: `MODEL` (gpt2-small | turingllm).
- `make_tasks.py`: `TASKS`, `N_TRAIN` / `N_HELD` (generated tasks; 1,600 / 5,000), `HELDOUT_CAP` (5,000; 0 = no cap),
  `SFC_SPLIT` (official | prefix), `NO_BLIMP=1`, `BLIMP_QUAL_MIN`.
- run time (`engine.batch`, every GPU script): `HELDOUT_N` (only the first N held-out items; smoke.sh sets 200),
  `PROFILE_N` (held-out items per member profile; 256).
- `prepare.py`: `TASKS`, `FORCE=1`.
- `probe.py`: `TASKS`, `OUT_PROBE`, `REPORT=1`.
- `calibrate.py`: `CAL_TASKS` (ioi,greater_than,sva_across_rc), `LAMS_W`, `LAMS_U`, `LRS_W` / `LRS_U`, `STEPS`,
  `REPORT=1`, `ADOPT=1`, `DRY=1`. Calibration never sees held-out items: each task is restricted to its training items,
  re-split 75 / 25.
- `run.py`: `OUT`, `CONFIG`, `TASKS`, `FORCE_TASKS=1`, `ARMS` (W,U,RND,W_ZA), `N_DRAWS`, `NO_PROFILE=1`.
- `engine.py`: `DTYPE=bfloat16` (dictionary precision; float32 by default).

## Outputs

| path | content |
|---|---|
| `tasks/<MODEL>/` | task JSONs, `summary.json`, `blimp_survey.json`, `<task>.contrast.json` (prepare.py) |
| `probe/<MODEL>/competence.json`, `rows.jsonl.gz` | competence verdicts, per-item metric values |
| `calib/<MODEL>/` | calibration rows, report, recommendation |
| `config_main_<MODEL>.json` | the adopted grid (written only by `ADOPT=1`) |
| `out/<MODEL>/eval.jsonl` | one row per (task, arm, λ, draw): size, α quantiles, `task_*` / `obj_*` (Z / A_sp / C_sp, necessity, empties, accuracies, pass), timings |
| `out/<MODEL>/circuits/<task>/<arm>/<λ>[.d<k>].json` | {site: {latent: α}} + fit stats |
| `out/<MODEL>/profiles.jsonl.gz` | per W circuit, per member: activation, attribution and (attn) head drive by role |
| `out/<MODEL>/active/`, `members.jsonl.gz`, `train.jsonl`, `status.jsonl` | RND pools, members, curves, status |
| `out/<MODEL>/report.txt`, `report.json`, `figures/task_circuits_Z.{png,pdf}` | report and figure |
| `paper/figures/task-circuits.{png,pdf}` | `PAPER=1` figure (5.5 in): panel A size-faithfulness (W, U, W_ZA, random null), panel B competence |

## Code and shared changes

**Experiment files:**
- `task_common.py` (pure logic), `tasks.py` (tokenizer, schema, induction), `originals.py` (loaders for the original
  data), `blimp.py`, `make_tasks.py`;
- `engine.py` (GPU side; `head_drive`), `prepare.py`, `probe.py`, `calibrate.py`, `run.py`, `report.py`, `figure.py`;
- `smoke.sh`, `queue_calib.sh`, `queue_local.sh`, `config_smoke.json`.

**Tests:** 67 in total.
- `test_081_logic.py` (30): includes the list halves, the held-out cap and the presentation order (without
  replacement within an epoch).
- `test_081_tasks.py` (21): includes SFC's official split, disjoint generated lists, the cap and the recorded uncapped
  size.
- `test_081_engine_cpu.py` (15): the real fit path on a tiny CPU model for token / set / seq answers, the training
  stream handed to the engine, whole-pool chunked means, `HELDOUT_N`, W_ZA, RND, the contrast pool, attribution
  against a finite difference, the exact head-drive decomposition.
- `test_081_report_cpu.py` (1).

**`src/` changes**, all additive and off by default:
- `eval/floors.py`: `valid=` masks for the A / C means.
- `circuit/instrument/learned_mask.py`:
  - `logit_metric` "token" (default) | "set" | "seq" (teacher-forced multi-token answers as (good, bad)
    row pairs);
  - `pos_valid` / `neg_valid`;
  - `floor_means` / `pos_floor_means`: precomputed C / A floors, None = collected as before; provenance
    `floors_precomputed`.
- `eval/wcm_eval.py`: `score_task` (now including `seq_diff`), `passes_task`, `task_metric`, `read_rows`,
  `answer_logprob`, `pair_diff`, `read_anchor_logprobs`.
  - `score_task(means_A=, means_C=)`: precomputed A / C values, None = collected as before.
  - Each frame's [n_ho, vocab] log-probs are now reduced to per-item values as soon as they are read. Host memory is
    one frame (~1 GB at 5,000 held-out items), not nine; the numbers are unchanged (the existing tests pass as
    before).
- Tests:
  - `tests/eval/test_wcm_eval.py`: `TestTaskScorer` (9), `TestTaskScorerSeq` (3);
  - `tests/circuit/instrument/test_learned_mask.py`: `TestLogitSetMetricAndValidFloors` (7), `TestLogitSeqMetric` (4).
- Repo suite: **1612 passed, 21 skipped**.

## Runtime estimate (AN ESTIMATE; nothing has been timed on GPT-2)

The basis is 080's two smoke timing passes (TuringLLM, 64-token contexts): fit time = a0 + a1·sites + steps·(b0 +
b1·sites). Extrapolated to **48 masked sites at 400 steps**:
- **Per fit:** W ≈ 105–115 s; U ≈ 105–120 s; RND and W_ZA ≈ W.
  - GPT-2 small is smaller than TuringLLM (d_model 768), and the task prompts are 1–31 tokens, so these are upper-end
    numbers.
  - The 48 float32 dictionaries are a new memory cost (~9.7 GB resident).
  - The training budget is unchanged (400 × 4). The engine's pre-passes (natural targets and the gap
    normaliser's two closed-mask passes) now run over the 1,600 presented rows instead of 180, which adds an
    estimated 20–35%: **≈ 125–155 s per fit**.
- **Per scoring pass:** ~1.5–6 ms per held-out row with all 48 sites edited. This is bounded below by 080's fit
  per-site step cost (~2 µs per token-site forward) and above by 080's scorer (~10 µs per token-site pass), at ~12
  tokens per prompt.
- **Scoring** scales with the held-out size: 9 passes per circuit, 28 circuits per original-contrast task and 32 per
  generated-contrast task (W 4, U 4, RND 20, W_ZA 4).

  | held-out items | per circuit | per task (28 circuits) | tasks |
  |---|---|---|---|
  | 5,000 | ~1–4.5 min | ~0.5–2.1 h | ioi, greater_than, induction, sva_across_rc |
  | 1,728 | ~25–95 s | ~11–43 min | sva_within_rc |
  | 233–469 | ~3–25 s | ~2–12 min | sva_across_pp, factual_known, each BLiMP subset |
  | 25–85 | < 5 s | < 3 min | gender_pronoun, sva_simple |

  Summed over all 17 tasks (24,582 held-out items): **~2.6–10.5 h** of scoring (it was ~1–2 h at 80 items per task).
- **Profile:** the first 256 held-out items per W circuit (`PROFILE_N`), ~30–100 s each, so ~0.6–1.8 h over 68 W
  circuits.
- **Means and RND pools:** A / C over the whole pools (~95k prompts in all; sva_across_rc alone has 2 × 34,003) plus
  the active-latent pass: an estimated ~5–15 min over the run.
- **Prepare:** 9 generated-contrast tasks (690–908 training prompts each, 32 candidates per item): a few minutes, plus
  ~1–2 min of model and dictionary load.
- **Probe:** every training and held-out prompt plus the counterfactuals (~140k plain forwards): ~5–15 min.
- **Calibration:** 3 tasks × 10 fits at ~125–155 s, with validation sets of 400 (ioi, greater_than) and 5,000
  (sva_across_rc, capped from ~8,500): **~1.3–2.2 h**, or ~4–6.5 h with 3 learning rates per arm.
- **Main run:** 28 fits per original-contrast task, 32 per generated-contrast task.
  - All 17 competent: 512 fits ≈ 18–22 h, plus scoring 2.6–10.5 h and profiles 0.6–1.8 h, so **~21–34 h**.
  - ~14 competent (a guess) with the four 5,000-item tasks among them: ~18–29 h.
  - `N_DRAWS=3`: ~16–25 h for all 17. `HELDOUT_CAP=2000` (rerun make_tasks) would cut the 5,000-item tasks' scoring by
    60%.
- **Smoke:** ~15–25 min, unchanged in order of magnitude. Each fit has 48 presentations, scoring uses 200 held-out items
  (was 80), profiles 80, and the ioi stage means cover 1,600 + 1,600 prompts.

## Caveats and open questions

- **Generated contrast set (Daniel's rule applied to outputs).** The threshold and the 32-candidate depth are design
  choices, and W_ZA is the sensitivity check. If a task has items with no silent candidate (see the diagnostics), its
  pool is built from the rest.
- **Head attribution** decomposes an attention member's pre-activation over heads. It is not a head ablation. Members
  are weighted by first-order attribution (attribution patching in the circuit's Z frame).
- **BLiMP selection** spends 2 of 8 slots on NPI licensing, while determiner-noun agreement (88%, 87%) and binding's
  domain_1/2 are left out by the cap. The rule is mechanical; reorder if phenomenon coverage matters more.
- **SFC's official split leaks** (see **Splits**): most SFC test items are identical to a training item, and every
  simple / within_rc / rc test prompt is a training prompt. Held-out faithfulness on the SFC tasks is therefore close
  to in-sample. `SFC_SPLIT=prefix` is the prompt-disjoint alternative. This needs Daniel's call before the held-out
  numbers are quoted.
- **Task-file size.** The full datasets make the tracked task files larger: tasks/gpt2-small is 47 MB (sva_across_rc
  19.9 MB, induction 8.1 MB, greater_than 7.3 MB, ioi 5.7 MB) and tasks/turingllm is 41 MB. They are regenerable in
  ~20 s (gpt2-small) / ~2 min (turingllm) on the CPU; consider git-ignoring the large ones.
- **Sources with gaps:**
  - gender_pronoun (75 / 25) and sva_simple (150 / 85, 40 prompts) have small held-out sets;
  - Hanna's repository has no licence and no noun file (nouns from ACDC's copy);
  - the Mathwin notebook has no licence and no version control.
- **Multi-token scoring** is implemented and tested but unused on GPT-2, where every answer is a single token. It runs
  under `MODEL=turingllm` (factual_known).
- **Precision.** Fits run in float32 by default (~9.7 GB of dictionaries). On a 16 GB card a deep fit may be tight;
  `DTYPE=bfloat16` is the fallback, and a precision change has to be stated with the results.

## v2 local run

The first GPU run of 081, under protocol v2, on the RTX 5070 Ti (float32). It was stopped early (token budget), so it
covers IOI and greater-than in full and gender_pronoun's W arm only. Tables: `out_v2/tables.md` (from `v2_tables.py`),
`out_v2/gpt2-small/report.txt` / `report.json` (report.py). Logs: `out_v2/queue*.log`.

**v2 wiring.**
- `task_common.WCM_VERSION` (env, **default v2**; `WCM_VERSION=v1` reproduces the old fill). It maps to
  `train_fill="sparse"`, `sparse_grad="ste"`, passed by `engine.engine_kwargs` to `run_learned_mask`. The logit objective
  uses the same patchers as the latent objective, so the fill reaches its C and A terms. `sparse_fill_calls` = 38,400
  per 400-step fit (> 0 on every fit; 1,152 in the 12-step smoke). U runs the fill at α = 1.
- `train_means_source="eval_ref"` holds by construction: `engine.means(B)` is computed once per task and the same
  tensors go to the fit and to `score_task`. Rows record `train_means_source="scorer"`.
- `run.py` refuses an OUT whose circuits were fitted under another version (`OUT/WCM_VERSION`). New env knobs:
  `EVAL_BS` (64 here), `RND_LAMS`. `train.jsonl` now records the per-term curve.
- **src (additive, default-off):** `learned_mask` `logit_norm="gap_term"`. Each ablation term is divided by its own
  closed-mask gap², measured under that term's training fill and floored at `LOGIT_GAP_FLOOR`. It needs a logit
  objective with a dual / triple / pn floor. Provenance: `logit_term_scales`, `dual_norm_pos`. 4 tests
  (`TestLogitPerTermNorm`). Repo suite: 1815 passed, 3 skipped. 081 + 080: 115 passed. The 081 profile
  finite-difference test is pinned to v1 (its v2 toy circuit fails the ±1% difference on one member; not investigated).

**Competence** (`out_v2/probe`, held-out). 15 of 17 tasks are competent: IOI (logit diff 3.67, 100% correct) and
greater-than (probability diff 0.816, 98%).
- Not competent: `blimp_principle_A_c_command` (63% correct) and `blimp_wh_island` (54%).
- `blimp_principle_A_case_1` is competent, but it has **no silent contrast candidate**: prepare.py's pool is empty, so
  it cannot be fitted under the C rule.

**Calibration** (W only, lr 0.05, 400 steps, the training rows re-split 75 / 25; `out_v2/calib_*`).
- IOI W: λ 3e-4 passed (9,544 latents, Z / A / C 0.90 / 0.91 / 0.90, necessity 1.07); λ 1e-3 reached 0.46; λ 3e-3 gave
  0.02.
- **IOI fits are bimodal.** λ 1e-4 collapsed to Z 0.01 at 12.7k latents.
- U was not calibrated (time). Each task got a hand-written 3-λ grid (`config_v2_ioi.json`,
  `config_v2_greater_than.json`, `config_v2_small.json`), not calibrate.py's ADOPT.

**Normaliser A/B** (v2, gap vs gap_term, same grid).
- The rule was fixed before the gap_term rows were read: task-rule passes, then the median over fits of min(Z, A, C).
  - gap: 1 pass, median 0.198.
  - gap_term: 1 pass, median 0.052.
  - So the rule picks **gap**.
- **IOI: the per-term normaliser is a no-op.** Its closed gaps are zero 17.8, C 17.7, A 18.0 nats², because a
  contrastive logit difference goes to ~0 under every ablation. Yet the outcomes flipped (gap_term passed at 1e-4 with
  21k latents and collapsed at 3e-4), which is fit noise, not the normaliser.
- **Greater-than:** the gaps are 13.6 / 9.9 / 10.0, so gap_term weights A and C 1.37× more. It raised C_sp from 0.12 to
  0.92 at λ 1e-4, at a cost in Z (0.83 → 0.69).
- 080's suspected cause (b), a zero gap ~5× the mean gaps, does not arise on contrastive task metrics.

**Does v2 fix the A/C failure on outputs?** On these tasks A and C are not the bottleneck: in every held-out W fit, A
and C are at or above Z. The paired v1 control (`calib_v1_gap`, the same fits under the dense fill) goes as follows:

| fit | v1 Z / A / C | v2 Z / A / C |
|---|---|---|
| IOI 3e-4 | 0.48 / 0.39 / 0.55 | 0.90 / 0.91 / 0.90 |
| greater-than 1e-4, task metric | 0.01 / 0.00 / 0.00 | 0.83 / 0.74 / 0.12 |
| greater-than 1e-4, objective | 1.04 / 0.91 / 1.02 | 0.98 / 0.93 / 0.70 |
| greater-than 3e-4, objective | 1.03 / 0.61 / 0.71 | 0.95 / 1.03 / 1.13 |

This is consistent with v2 helping. But n = 3, and IOI's fit-to-fit noise is of the same size, so it is **not** a
demonstration.

**Held-out results** (5,000 items each, from disjoint name / noun lists; the full table is in `out_v2/tables.md`).
- **IOI: nothing passes.**
  - W: 4,174 latents Z 0.24; 9,460 Z 0.33; 20,961 Z **0.47** (A 0.52, C 0.51, necessity 1.05, accuracy 0.83). Z at
    α = 1 is ≈ 0.
  - U: 16k–41k latents, Z 0.04–0.09.
  - RND (9,460 latents, 3 draws): Z 0.01, pass 0/3.
  - Calibration's 9.5k pass (same names as the fit pool) did not reproduce on held-out names. The main fit differs
    too (1,600-item pool, not 1,198), and IOI outcomes are bimodal. A queued diagnostic was not run: re-scoring on
    training-list items (`v2_insample.py`), and 1,200 steps (`config_v2_ioi_steps1200.json`). One more data point:
    U's training-batch fit is ~0.6, but its held-out Z is 0.04.
- **Greater-than: metric mismatch.** The fit reproduces log P(>YY set) − log P(≤YY set); the registered pass rule
  uses Hanna's probability difference.
  - On the **task metric**, W fails at every size (11.4k: Z 0.61, A 0.90, C 0.91) and **U passes** (12.3k: 0.88 / 0.88
    / 0.94; 19.3k: 0.97 / 0.97 / 0.98).
  - On the **fitted objective**, W passes at 5,011 latents (0.97 / 1.03 / 1.00, necessity 1.00) and at 11.4k. U fails
    on necessity (0.68–0.78).
  - W's coefficients reproduce the log-odds without the probability mass on the year tokens (W's task Z at α = 1 is
    0.00). U cannot rescale, so it keeps the mass.
  - RND (5,011 latents, 3 draws): task Z 0.00, pass 0/3. On the objective, Z is 0.03–0.07 and A / C are 0.25–0.76.
  - To act on this, change one of the two: fit the probability difference, or register the rule on the objective.
- **U / W size** (claim 2):
  - IOI: neither arm reaches Z 0.8. At similar size, W 0.47 (21k) vs U 0.09 (16k).
  - Greater-than, task metric: reversed (U reaches 0.8 at ~11.6k, W never does).
  - Greater-than, objective: W reaches Z 0.8 at ~2.0k and U at ~9.3k, so **4.7×**. The smallest passing W is 5,011;
    U never passes.
- **gender_pronoun** (25 held-out items, W only, stopped before U / RND): W passes at every λ, at 564, 1,269 and 2,700
  latents (smallest: 0.92 / 1.22 / 1.23, necessity 0.96).

**Structure** (the primary W circuit is the best worst-ablation one, since none passes; head-drive enrichment, found at
≥ 2× the base rate).
- **IOI** (20,961 latents):
  - holds: S-inhibition 3.7× (share 0.62 vs base 0.17), induction 4.3×, duplicate-token 5.3×;
  - fails: name movers 1.07×, previous-token 0.71×;
  - at END, the top heads by attribution are 7.3, 7.9, 8.3, 8.6 and 8.10 (all four of Wang's S-inhibition heads in the
    top 5), and backup name mover 9.7 also appears.
- **Greater-than** (11,393 latents): Hanna's heads 4.3× (head 9.1 alone carries 20% of the anchor head weight), late
  MLPs 0.52 (≥ 0.5), and YY-concentrated members sit 5 layers before anchor-concentrated ones. All three hold.
- Caveat: these circuits do not pass, and the attribution is a decomposition of each member's pre-activation, not a
  head ablation.

**Not run:**
- the tasks sva_across_pp, sva_simple, factual_known and blimp_anaphor_number_agreement;
- gender_pronoun U / RND;
- induction, sva_within_rc, sva_across_rc and the other BLiMP subsets;
- every W_ZA arm;
- the diagnostics (`queue_v2_diag.sh`).

**Runtime** (one GPU process at a time):

| item | time |
|---|---|
| 400-step fit | 174 s (IOI), 148 s (greater-than), ~145 s (gender); peak 10.9 GB |
| score, 5,000 items at batch 64 | 352 s (IOI), 216 s (greater-than) |
| profile | 14–49 s |
| smoke | 15 min |
| prepare + probe | 6 min |
| W calibration, two normalisers | 62 min |
| v1 control | 14 min |
| IOI | 81 min |
| greater-than | 56 min |
| gender (W only) | ~10 min |
| **total** | **~4.2 h of GPU time** (15:20–19:31) |

**Resume** (resumable: finished circuits and rows are reloaded, and IOI / greater-than come back in a few minutes):
```bash
wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && SKIP_V1=1 bash experiments/081-task-circuits/queue_v2_main.sh >> experiments/081-task-circuits/out_v2/queue_main.log 2>&1 && bash experiments/081-task-circuits/queue_v2_diag.sh >> experiments/081-task-circuits/out_v2/queue_diag.log 2>&1'
PYTHONPATH=src python -X utf8 experiments/081-task-circuits/v2_tables.py      # refresh out_v2/tables.md
```
Note: the stopped gender invocation left an unterminated gzip member at the end of `out_v2/gpt2-small/members.jsonl.gz`
/ `profiles.jsonl.gz` only if it had written to them. report.py read both cleanly after the stop.
