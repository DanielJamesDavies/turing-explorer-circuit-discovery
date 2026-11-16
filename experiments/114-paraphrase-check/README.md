# 113 Paraphrase check: does the interpretability corpus hold paraphrases of one document?

A reviewer asked: "State whether the interpretability corpus contains paraphrases of the same document. If it does,
split held-out contexts by document group." This is a read-only check (2026-11-14). It changed no data, records or
paper text.

**Answer: yes.** 86.6% of TuringLLM's 24.8M interpretability windows come from T5 paraphrase files. Each file is the
paraphrased variants of one Phi-3 original, and the original is in the corpus too. The production held-out strongest
sets (094 / T2 `ctx_fresh`) remove exact, anchor-prefix and near copies, but they do not remove paraphrases. In a
388-target sample, **28–35% of held-out strongest rows paraphrase a training context of the same target** (TF-IDF
cos ≥ 0.3; random pairs pass at 0.02%). **93% of targets have at least one such row.** A held-out rebuild by document
group is feasible from stored data, with no refit.

Note: `113-coefficients-shape/` already uses the number 113. This folder uses the name given in the brief, so one of
the two may need renumbering.

## 1. How the corpus was generated

The generator code is **not in this repo** and not elsewhere on X: (a search of `X:\Projects` for `desc_`,
`paraphras`, `data-augmentation` and `chatgpt_paraphraser` found only unrelated hits). The construction is documented
in two places:

- `paper/resources/dan28-removed-appendix.md:29` (the removed App. A). Phi-3 Mini generated documents over
  18 subject areas, "recursively generating topics and sub-topics and producing structured documents".
- `dan28-removed-appendix.md:32`: "A fine-tuned T5-based paraphrase model … generating 11 paraphrased variants of
  each source document".
- `paper/main.tex:637` (model card): "each with 11 paraphrases~\citep{vorobev2023paraphraser}".
- `data/text_paths.txt` (95,682 lines). This is the source-file list. `corpus_src.npy`
  (085, `corpus_index.py:4,45`) holds each window's 0-based line index into it. I checked the alignment: at offset 0,
  paraphrase files have a median of 490 windows and originals 42 (about 11.7×). At offset 1 both are 60.

| tree | files | role |
|---|---|---|
| `text/phi-3-mini/dataset/subjects/items/A/items/B/items/C/desc_k.txt` | 43,225 | Phi-3 originals; k = 0..5 per leaf topic (14,179 leaves with 6 files, 1,512 with 2) |
| `text/data-augmentation/dataset/<same relative path>` | 42,535 | the T5 paraphrases of that exact original (**every** paraphrase file has a same-path original) |
| `phi-3-mini/.../short_stories`, `public_figures`, `textbooks` | 9,922 | originals, no paraphrase file |

So one prompt or topic gives two levels of near-same text:

- **Paraphrases (certain).** Each paraphrase file holds about 11 rewrites of one original, which is also in the corpus.
- **Siblings (possible).** The 6 `desc_k` files of one leaf topic are 6 Phi-3 documents on the same topic.

## 2. Group ids

None are stored per row on TuringLLM: `strong.doc` is −1 in every TL record. But every row stores its corpus id, and
`corpus_src.npy[id]` → `text_paths.txt` gives exact groups:

- **file**: the source file.
- **document group**: the relative path with the `phi-3-mini` / `data-augmentation` prefix stripped (the original
  plus its paraphrases; 53,147 groups).
- **topic group**: the dirname of that path (the 6 sibling documents).

## 3. Empirical check (`check.py`, `report.py`; output in `out/`)

The sample is 400 random T2 records, 388 of them usable (6,009 held-out strongest rows). For each held-out row I took
the maximum similarity over the target's 96 training-side rows (48 strongest-train + 48 mid-train):

- **TF-IDF cosine** on token unigrams and bigrams (IDF from 100k random windows);
- **8-gram share** (the 086 near-duplicate measure).

**Calibration.** For 200 originals, I compared one window against its own paraphrase file and against an unrelated
paraphrase file of the same size.

| max cos (p10 / p50 / p90 / p99) | |
|---|---|
| own paraphrase file | 0.174 / 0.407 / 0.592 / 0.740 |
| unrelated file | 0.019 / 0.029 / 0.054 / 0.093 |

Known paraphrases have a median 8-gram share of only 0.05, so the copy rule cannot catch them.

**Held-out rows (max over 96 training rows).**

| comparison | cos p10 / p50 / p90 / p99 |
|---|---|
| same target's training rows | 0.055 / 0.198 / 0.529 / 0.721 |
| another target's training rows | 0.020 / 0.033 / 0.060 / 0.117 |
| 96 random corpus windows | 0.024 / 0.037 / 0.066 / 0.123 |

**Share of held-out rows called paraphrases, by cos threshold.**

| cos ≥ | held rows | and best match in the same document group | false positive (other target) | recall on known paraphrases | targets with ≥ 1 |
|---|---|---|---|---|---|
| 0.2 | 49.5% | 32.6% | 0.10% | 87% | 95% |
| **0.3** | **34.6%** | **27.6%** | 0.02% | 75% | **93%** |
| 0.4 | 22.5% | 19.7% | 0.02% | 50% | 85% |

The p99-of-random threshold (0.093) flags 77.9% of rows. That count is inflated by topical similarity: a latent's
strongest contexts share a topic. It should not be quoted as a paraphrase rate. No held-out row reaches 8-gram share
0.5 (maximum 0.491), so the copy rule did its job.

**Provenance.** The held-out row's group also supplies a training row of the same target in these shares of rows:

| group | share of held-out rows |
|---|---|
| same file | 39.8% |
| same document group | 45.3% |
| same topic | 54.9% |

When the best match is in the same document group, its median cos is 0.427, close to the known-paraphrase median of
0.407. Otherwise the median is 0.128.

**Examples** (decoded in full in `out/report.md`). Each pair is a held-out row and its training match:

1. `9.resid.33013` (cos 0.84, 8-gram 0.49, same paraphrase file `…/items/8/items/17/items/6/desc_1.txt`). Held: "As
   advanced conversational agents, chatbots rely on Natural Language Understanding (NLU) …". Train: "To function as
   advanced conversational agents, chatbots rely on …".
2. `5.mlp.8113` (cos 0.84, same file `…/11/items/3/items/11/desc_5.txt`). Both rows list PLA, PHA and PBS polymers,
   reworded.
3. `4.attn.34415` (cos 0.83, same file `…/1/items/12/items/13/desc_0.txt`). Held: "Within the realm of numerical
   methods for mathematics, …". Train: "Several techniques are included in the realm of numerical methods …".
4. `9.attn.20456` and `1.mlp.16391` (cos 0.11, near the 0.093 threshold, different documents). These are the same
   topic at most, not paraphrases.

## 4. Is a group-level split feasible, and what does it cost?

**Yes, from stored metadata.** Corpus id → `corpus_src.npy` (085) → `text_paths.txt` gives the document and topic
groups. The T1 scan (`109…/pub-turingllm-T1-scan/…/scan_top512.pt`) holds each target's 512 strongest candidate ids.

If the held-out rows are drawn only from groups that supply no training row, the stored held-out rows that survive
are:

| rule | median rows left of 16 | targets below 8 |
|---|---|---|
| file-disjoint | 9 | 35.8% |
| document-group-disjoint | 8 | 45.6% |
| topic-disjoint | 6 | 56.2% |

Adding unused scan candidates from disjoint groups fixes the shortfall. Before the firing and copy checks there are
these candidates per target:

| rule | median candidates | targets with ≥ 16 | targets below 8 |
|---|---|---|---|
| document-group-disjoint | 313 | 96.9% | 2.8% |
| topic-disjoint | 251 | 96.1% | 3.1% |

They sit deeper in the ranking, so held-out strength must be reported as a covariate.

**Cost.**

- **Held-out-only rebuild (no refit).** Training rows and circuits stay fixed. The steps are a 086/094-style rebuild
  with a document-group clause (the dropped "source-file clause" of v2, widened to the original plus its paraphrases)
  followed by a re-score of main, sweep, nulls and so on, as DAN-157 was. By the 086 cost table this is under a pod
  hour per stage on 8×H100, or roughly a day locally.
- **Group-aware pool split (training changes).** Grouping the whole pool by document group changes the fit rows, so
  the production run would need a **refit**.

**Recommendation.** Answer the reviewer "yes" and run the held-out-only rebuild by document group (re-score, no
refit), reporting the pass rate on it next to the current one. Report the topic-disjoint rule as a sensitivity check.
Quote the paired difference. Mid-band and contrast sets were not checked here.

## Files

- `check.py`: sampling, calibration and similarity. Writes `out/rows.jsonl`, `out/calibration.jsonl` and
  `out/run.log`. Run as `.venv/bin/python -I experiments/113-paraphrase-check/check.py 400`; it takes about 25 min in
  WSL on /mnt/x.
- `report.py`: tables and decoded examples. Writes `out/report.md`.
- `out_run.log` is the log of a first run that failed (too many open files). It can be deleted.
- The scan-candidate count in §4 was an inline computation, not saved as a script; it uses the same group definitions
  as `check.py:groups()`.
