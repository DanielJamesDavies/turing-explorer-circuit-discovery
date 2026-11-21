# 125 — Random-circuit gallery on the production run (DAN-219, gallery decision)

The 072 gallery (`app:random-gallery`, `tab:random-gallery`) redrawn from the **production run** (WCM v4, TuringLLM
8k, 2026-11-07/08 pod) instead of protocol v1. Daniel approved the redraw. Nothing in `paper/` is edited here; the
proposed replacement text is at the bottom.

## Design (072's, production data throughout)

* **Draw.** 9 targets, one per depth band (L0-4 / L5-7 / L8-11) × site kind (attn / mlp / resid). Each is drawn
  uniformly from that cell's non-vacuous production circuits, whatever the outcome.
  * Population: the held-out-strongest row of each target, with merge.py's v2 vacuous rule (7,464 scored).
  * Draw: 072's code, `random.Random(20261012).choice(sorted(cell))`, in 072's cell order.
  * Cell sizes and the run's rates: `results/draw.json`.
* **Scores.** From the production eval row: Z / A / C, necessity, induce and size.
  * Pass: the production rule (merge.py: Z, A, C in [0.75, 1.5] and necessity ≥ 0.9).
  * Amplifier and near-threshold flags: `wcm_specificity.target_specificity` on the production spec rows, as in 111.
* **Members.** Contribution share at the target's anchor (|α| × mean activation at the anchor × decoder norm). This
  uses 110 `report.member_table`, imported unchanged, on the production circuit and its fresh deduplicated record.
  It ran **on the CPU** (Windows torch, about 330 s for all nine), so no GPU was used.
* **Hubs.** The explorer converter's `HUB_RULE` (`src/export/explorer/stage_reading.py`) has two parts:
  * Membership: member of more than 50% of circuits. This is counted over the **7,786 production circuits**
    (`results/member_freq.json`), giving 80 candidates.
  * Specificity: median pos_max / neg_max ≤ 1.5 over at least 10 circuits. This median exists only in the v1
    bundle's `reading_hub`, so it is taken from there. 56 candidates pass and are dropped as hubs.
  * The other 21 candidates have no v1 median, because they were not frequent in v1. As in the converter
    (is_hub NULL), they are kept. The other 3 candidates have a v1 median above 1.5 and are kept by the rule.
  * Example: `0.attn.4540` (document start) is in 83% of production circuits, but its v1 median ratio is 2.9, so it
    is not a hub and appears as a top member of `0.mlp.12578`.
* **Target labels.** The 065 circuit describer, prompt v5, unchanged: `describe.SYSTEM_PROMPT`, `schema_for` and
  `ask_claude`.
  * Its `build_report` reads the v1 bundle, so the report is rebuilt line for line in its format from production
    data: the production scores and member table, plus the fresh record's strongest, mid and contrast contexts.
    The reports are in `results/target_reports/`.
  * The target labels are therefore written for these nine production circuits, not taken from the run-wide v1
    descriptions.
* **Member labels.** 072's latent-only call, verbatim (`LATENT_SYSTEM_PROMPT`, `LATENT_SCHEMA`, `build_prompt`).
  * Its evidence is the latent's own peak tokens, strongest stored contexts and logit effect. This is a property of
    the SAE latent, not of the run, so it is read from the v1 bundle's `reading_latent`.
  * Every target and every member shown was present there. The computed fallback (`results/latents.json`) stayed
    empty.
* **Model and concurrency.** All 36 calls resolved to `claude-sonnet-5-5`, with 4 workers (at most 8).
* **CLI location.** 065 `find_claude` no longer finds the desktop app's CLI, which now lives at
  `%LOCALAPPDATA%/Claude-Data/claude_*/claude-code/<ver>/<hash>/claude.exe`. `gallery_v4.py` wraps it (065 is not
  edited).
* **Typography.** One fix on top of 072's `tex`: an opening single quote at a word start becomes a TeX open quote
  (`'by'` → `` `by' ``). The label text itself is kept verbatim.

```
PYTHONPATH=src python -X utf8 experiments/125-random-gallery-v4/gallery_v4.py     # STAGE=draw|hubs|members|labels|write
STAGE=labels DRY=1 ...                                                            # print one target + one member prompt
```

Each stage caches its output, so a rerun calls neither the model nor the CLI again. The member stage takes about
6 min on the CPU, the hub count about 4 min, and labels about 3 min.

**Outputs (`results/`):**

* `draw.json`: the draw, scores, cell sizes and run rates;
* `members/<key>.csv`: the full member contribution tables, with the hub flag;
* `target_reports/`;
* `target_labels.jsonl`, `member_labels.jsonl`: label caches;
* `gallery.json` (includes the hub table), `gallery.md`;
* `gallery_table.tex`: the table rows, in 072's format.

## Result: the draw (seed 20261012)

| Cell | Target | Outcome | Size | Z / A / C | Nec. | Ind. |
|---|---|---|---|---|---|---|
| L0-4 attn | 3.attn.29966 | pass | 646 | 0.95 / 1.02 / 0.99 | 1.00 | 1.38 |
| L0-4 mlp | 0.mlp.12578 | pass, amplifier | 39 | 1.11 / 1.43 / 1.35 | 0.94 | 0.35 |
| L0-4 resid | 0.resid.27712 | pass | 184 | 0.94 / 0.99 / 0.98 | 1.00 | 1.54 |
| L5-7 attn | 5.attn.10033 | pass | 566 | 0.90 / 1.01 / 0.91 | 1.00 | 1.16 |
| L5-7 mlp | 5.mlp.16658 | pass, amplifier | 336 | 0.98 / 1.03 / 1.04 | 1.00 | 0.64 |
| L5-7 resid | 7.resid.31916 | pass | 770 | 0.82 / 0.97 / 0.95 | 1.00 | 0.97 |
| L8-11 attn | 8.attn.7985 | **fail**, near-threshold | 2226 | 0.61 / 1.05 / 1.03 | 1.00 | 0.00 |
| L8-11 mlp | 9.mlp.11349 | pass | 636 | 0.92 / 1.01 / 0.99 | 1.00 | 0.99 |
| L8-11 resid | 10.resid.20721 | pass | 1238 | 0.87 / 0.99 / 1.06 | 1.00 | 0.91 |

**Pass count.** 8 of 9 pass, against the run's 6,565 / 7,464 = 88.0%. Under that rate the expected count is 7.9.
* The one failure, `8.attn.7985`, is a deep attention target and near-threshold. It fails on Z alone (0.61); A and
  C are in band. It is large (2,226 latents), induces nothing (0.00), and its label is unclear (varied tokens in
  academic science text).
* Amplifiers: 2 of 9 (`0.mlp.12578`, `5.mlp.16658`), against 29.0% over the run. Near-threshold: 1 of 9, against
  8.0%.

**Overlap with 072.** Only `0.resid.27712` is in both draws: the same key comes up in its cell again. It now passes
(v1: it failed on A). Its top members again include `0.mlp.205` (`chrom`) and `0.mlp.4393` (`fect`).

**Readings (hypotheses, as in 072).**
* Shallow circuits rest on a few detectors of the target's own token.
  * `0.mlp.12578` (hedging adverbs: essentially / simply / merely) gets 7.7% of its contribution from an
    `essentially` detector and 5.7% from a `simply` detector.
  * `5.mlp.16658` (word-start ` ret`) has three ` ret` detectors among its top members.
* Deep circuits spread over many members with about 1–2.5% each, but their top members still detect the target's
  token.
  * `10.resid.20721` (` random`): three ` random` detectors, 1.5–1.9% each.
  * `9.mlp.11349` (` sequ`): two ` sequ` detectors, 2.4% each.
  * `7.resid.31916` (the digit 4 in 14xx years): two detectors of the same digit in the same position, 1.8–1.9%.
* Condition-type circuits:
  * `5.attn.10033` (` by` after "replaced") rests on a "replaced by/with" member (spec 69) and a replacement-word
    member (spec 34).
  * `3.attn.29966` (social and political theme words in text about music and society) mixes music-theory members
    with generic ones. Its 50 hubs carry 27% of its contribution.

## Proposed replacement text for `app:random-gallery` (main.tex lines 1742–1783; not applied)

Replace the comment block, the three paragraphs, the table rows and the caption with the following. The table
preamble and header are unchanged; put `\midrule` after rows 3 and 6 as now.

```latex
\section{Circuits for Randomly Drawn Targets}
\label{app:random-gallery}
% DAN-219 (2026-11-16): redrawn on the production run. experiments/125-random-gallery-v4/gallery_v4.py (SEED 20261012,
% 072's draw rule on the production non-vacuous circuits) writes results/gallery_table.tex (the rows below),
% gallery.md and gallery.json. Target labels: 065 circuit describer (prompt v5) on each production circuit's report;
% member labels: 072's latent-only call (results/member_labels.jsonl); both claude-sonnet-5-5.
The case studies were chosen; the circuits in \cref{tab:random-gallery} were not.
We drew one target uniformly at random from each depth band and site kind of the production run (seed $20261012$, non-vacuous circuits, any outcome) and list each circuit's scores and its three largest members by contribution share at the target's anchor, excluding generic hub latents.
Eight of the nine pass under the production criterion (floor $0.75$), in line with the run's $88.0\%$.
The labels are hypotheses written by a language model, not verified readings: each target was labelled from its circuit's report (its strongest, weaker and contrast contexts, logit effect and largest members), and each member from its own strongest contexts and logit effect alone.

The shallow circuit for \texttt{0.mlp.12578} (hedging adverbs such as ``essentially'' and ``simply'') rests largely on detectors of those same tokens, with $7.7\%$ and $5.7\%$ of the contribution, while the deep circuits spread over many members with about $2\%$ each; their largest members still detect the target's own token (\texttt{9.mlp.11349}, \texttt{10.resid.20721}).
The one failure, the deep attention target \texttt{8.attn.7985}, is near-threshold and fails on Z alone ($0.61$; A and C are in band), with $2{,}226$ latents and a target the describer could not name.
Two of the nine circuits are flagged as amplifiers, against $29\%$ over the run (\cref{app:specificity}).
```

Caption, replacing the first sentence and the label sentence:

```latex
\caption{Circuits for nine randomly drawn targets of the production run, one per depth band (rows grouped L0--4, L5--7, L8--11) and site kind, seed $20261012$.
Size is the number of upstream latents; Z, A, C are faithfulness under the three ablation methods, Nec.\ necessity, and Ind.\ sufficiency to induce with coefficients, all on held-out contexts; ``--'' is an undefined score, which counts as a fail.
$^{a}$~flagged as an amplifier; $^{t}$~near-threshold target.
Labels are hypotheses written by Claude Sonnet 5.5, for targets from each circuit's report and for members from each latent's own strongest contexts and logit effect; members are ranked by contribution share at the target's anchor, generic hub latents excluded.}
```

Two pointers elsewhere in `main.tex` also change:

* Line 638: `\appxrow{app:random-gallery}{protocol v1}` becomes `{production run}`.
* Line 535: "random protocol-v1 targets in \cref{app:random-gallery}" becomes "random production targets in
  \cref{app:random-gallery}".

The $29\%$ amplifier rate is `wcm_specificity` over the 7,464 scored production targets (`draw.json`
`run.amp_rate` = 0.2901). It matches the TuringLLM figure in `app:specificity` (main.tex line 1340: $29.0\%$). The old
gallery text's $30\%$ was the v1 rate, which line 2203 still quotes for v1.
