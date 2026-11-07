# 065 circuit describer: a short label for every TuringLLM circuit

## Question

Can every circuit of the 15k TuringLLM run (062) get a short, readable description of what its target latent fires
on, written from the circuit's own reading data and nothing else? The descriptions are labels for browsing and for
the appendix gallery, not findings. They are hypotheses: the 063 case studies found that readings taken from contexts
alone are often wrong until they are tested with minimal pairs.

## What the code does

1. **Report** (`report.py`, CPU). For one circuit it builds a compact text report from the explorer bundle
   (`outputs/explorer_bundle/2026-09-27_062-full/`: `reading.sqlite`, `explorer.sqlite`, `tokens/tokens.npy`):
   - the target: scores, peak tokens and their consistency, up to 5 distinct windows with the peak marked, weaker
     contexts, a contrast snippet where the target is silent, and its direct logit effect;
   - the top 15 members by contribution share at the target's anchor, hub latents dropped. Each member carries its
     specificity (strongest activation on the target's contexts over that on the contrast; ≥ 3 = discriminative),
     its own peak tokens, logits and one context.
2. **Describe** (`describe.py`). The report goes to Claude Sonnet through the Claude Code CLI in headless mode (`-p`,
   no tools, no hooks, no MCP, no session, structured JSON output against a schema). The answer has four fields, in
   this order: `description` (≤ 40 words), `label` (≤ 6 words), `trigger`, `condition`.
   - **trigger** = what the target fires on: `one_token`, `word_family`, `varied`, `number_format`, `unclear`. It is
     computed, not asked, when ≥ 80% of the peaks are one token after merging case, plural and prefix variants
     (`ONE_TOKEN_MIN`); the schema then allows only `one_token`.
   - **condition** = what else it needs: `none`, `within_word`, `construction`, `subject_area`, `entity_kind`,
     `unclear`, decided mainly from the contrast (`CONDITION_RULE`).
   - The system prompt forbids claims the report cannot support: that a circuit checks whether a fact is true, knows
     facts about particular entities, or combines separate concepts.
   - The report is fenced by a random token so corpus text cannot close the block, and is declared data, not
     instructions.
   - At most 8 CLI processes run at once, started at least 0.3 s apart (`MAX_WORKERS`, `LAUNCH_GAP_S`). More
     concurrent CLIs corrupted the CLI's own config file.
3. **Merge** (`combine.py`). Each `describe.py` run writes only its own `results/runs/<time>_<pid>.jsonl`.
   `combine.py` merges them into `results/descriptions.jsonl`, one record per circuit: a success beats a failure,
   otherwise the later record wins. The file is replaced atomically.
4. **Prompt tests** (`pick_holdout.py`, `score_prompts.py`). Expected answers for 29 circuits were written by hand
   from the reports before each prompt version was run (`results/test_expectations.json` for the single-type prompts
   v1–v4, `results/test_expectations_facets.json` for v5's two facets):
   - `original14`: 14 circuits chosen from v1's inconsistent answers;
   - `holdout15`: `pick_holdout.py --layers 6,7,8,9,10 --per-layer 3 --seed 0`, 15 passing circuits no prompt had
     seen.

   `score_prompts.py` counts exact matches and matches to an accepted alternative.

Prompt history: v1–v4 asked for one flat type (string, synonyms, syntax, gated_word, topic, composition, knowledge,
relation, format, unclear). Most confusions were one question in disguise: what does the target need besides the
token it fires on? v5 (`PROMPT_VERSION = 5`) splits that into trigger and condition, and drops the mechanism types.
`results/example_input.txt` shows a v1-era system prompt and report.

## Files and commands

| File | What |
|---|---|
| `report.py` | Builds one circuit's report from the explorer bundle (`--stats` prints its size and consistency) |
| `describe.py` | Sends reports to the CLI; selection by key or `--scope pass-l6 / pass-layers / fail-layers` |
| `combine.py` | Merges the run files into `results/descriptions.jsonl` |
| `pick_holdout.py` | Draws the held-out prompt-test circuits |
| `score_prompts.py` | Scores prompt versions against the hand-written expectations (`--facets` for v5) |
| `results/descriptions.jsonl` | The merged descriptions, one per circuit (the file other code reads) |
| `results/runs/*.jsonl` | The raw run files (history; kept) |
| `results/_v1/` | The first 128 descriptions under prompt v1 |
| `results/test_prompt_v2..v5.jsonl` | Prompt-test outputs |
| `results/run_v5_*.log` | Console logs of the v5 runs, by layer group and pass / fail |

```
python experiments/065-circuit-describer/describe.py 10.resid.15497 --dry-run     # print the full prompt, call nothing
python experiments/065-circuit-describer/describe.py --scope pass-layers --layers 5 --workers 8
python experiments/065-circuit-describer/describe.py --scope fail-layers --layers 0,1,2,3,4,5 --workers 8
python experiments/065-circuit-describer/combine.py
python experiments/065-circuit-describer/score_prompts.py            # v1-v4, single type
python experiments/065-circuit-describer/score_prompts.py --facets   # v5, trigger + condition
```

All of it is CPU-side. It needs the explorer bundle and the `claude` CLI on the path (or in the desktop app's install
folder, which `find_claude()` searches).

## Results

**Coverage** (`results/descriptions.jsonl`): 15,004 records, one per circuit of the 062 run, all successful, all
prompt v5, all answered by `claude-sonnet-5-5`. Every layer is covered (907 circuits at layer 0 to 1,357 at layer 9).
Median 5.4 s per call, median 4,503 input and 210 output tokens.

**Facet counts** over the 15,004:

| trigger | n | condition | n |
|---|---|---|---|
| one_token (computed) | 6,176 | within_word | 4,594 |
| word_family | 4,515 | construction | 3,627 |
| varied | 4,178 | unclear | 2,849 |
| number_format | 70 | subject_area | 2,077 |
| unclear | 65 | none | 1,764 |
| | | entity_kind | 93 |

**Prompt tests** (`score_prompts.py` output, rerun on CPU from the files above):

| prompt | set | exact | exact or accepted alternative |
|---|---|---|---|
| v1 | original14 | 6 / 14 | 7 / 14 |
| v2 | original14 | 7 / 14 | 10 / 14 |
| v3 | original14 | 2 / 14 | 6 / 14 |
| v4 | original14 | 7 / 14 | 11 / 14 |
| v4 | holdout15 | 9 / 15 | 13 / 15 |
| v5, both facets | original14 | 9 / 14 | 12 / 14 |
| v5, both facets | holdout15 | 12 / 15 | 15 / 15 |

v5 per facet: trigger 11 / 14 and 14 / 15 exact, condition 12 / 14 and 13 / 15 exact. v4 and v5 also cut the mean
output from 611–656 tokens (v1–v3) to 187 (v4) and 210 (v5).

**Where it is used.**
- The paper's random-circuit gallery takes its target labels from this file (`paper/main.tex:1501-1506`, table
  caption at `:1537`), through `experiments/072-random-gallery/gallery.py`, which also reuses `describe.py`'s CLI
  call for the member labels.
- The explorer's descriptions stage reads it (`src/export/explorer/stage_descriptions.py`).

## Caveats

- The labels are read off a text report by a language model. They are hypotheses, not verified readings. Nothing
  here tests them against the model.
- The expectations were written by hand by one reader; agreement with them measures consistency with that reading,
  not correctness.
- The descriptions are of the 062 (protocol v1) circuits and the explorer bundle built from them, not of the later v4
  refit.
