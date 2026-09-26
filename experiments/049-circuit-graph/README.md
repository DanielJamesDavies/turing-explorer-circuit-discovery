# 049 — Circuit-level graph over the production tri-amp circuits

Daniel's question (2026-09-07, night): with ~16k circuits, treat circuits
as nodes and find the "organs" — causal relationships between circuits,
fan-in/fan-out, shared dependencies/sub-circuits.

Two datasets. **Snapshot** (01:05 UTC, 7,310 circuits, seeds L0–L3 +
L9–L11 only because of the interleave): `data/`, `tables/`, `results/`;
sections 1–8 below were written on it overnight and are kept as the
record. **Full set** (run complete 06:43 UTC, 15,046 circuits from 15,925
seeds): `data_full/`, `tables_full/`, `results_full/`; **section 0 is the
full-set summary and supersedes the snapshot numbers wherever they
differ.** Same scripts, `TABLES=…/tables_full OUT=…/results_full`.

Data layout: `data*/` (gitignored .pt shard stores + task_metrics),
`tables*/` (parquet: `members` one row per (circuit, member) with
amplitude and attribution; `circuits` one row per circuit with evals and
amp_stats), `results*/` (csv/json outputs).

## 0. FULL SET — 15,046 circuits, 5.10M member nodes (100% with amplitudes)

**Skip census.** 879 of 15,925 seeds (5.5%) produced no circuit; ALL are
attention seeds: L1 70%, L2 34%, L3 11%, L5 22%, L7 16%, L8 24%, others
1–6%; every mlp/resid seed fits. Reasons: no stored positive contexts
(dead latents) or fewer than 8 positives (no negatives for the triple
floor). Cost per fitted seed rises linearly with depth: L0 11 s → L11
86 s (p90 within 3 s of the median).

**Fan-out / fan-in.** 482k distinct member latents (82% of the
dictionary); 42% appear in one circuit; the top 1% of latents hold 71% of
all memberships; 651 latents sit in ≥10% of circuits, the top hub
`0.attn.5663` in 99.5%. Top-200 hubs: L0 125, L1 40, L2 26, ≥L3 9. Fan-in
median 321 (p10 101, p90 593, max 2,311), monotone in seed depth: L0 25,
L2 161, L4 260, L6 371, L8 464, L9–10 ~505, L11 452. 97.8% of members lie
below the seed, mean 5.4 layers down, 12.6% within one layer.

**Seed→seed graph.** 5,154 seeds are members of other seeds' circuits;
46,465 edges (3.1 per circuit); in-degree median 3, max 27; out-degree
max 4,531 (hub seeds). Longest chain **10 circuits**, one per layer:
`0.mlp.16730 → 4.attn.15114 → 4.resid.26536 → 6.attn.40549 → 7.attn.5010
→ 7.resid.11194 → 8.mlp.4085 → 9.mlp.5042 → 10.mlp.35327 → 11.mlp.40310`;
35% of circuits sit at depth ≥3 in the DAG (snapshot: 12%).

**Overlap.** Every pair shares ≥1 member; raw overlap correlates 0.89
with |A|·|B| (density artefact); lift removes it (−0.23). Naive Leiden
= 3 giant hub-defined communities. 155 near-duplicate pairs (Jaccard
≥0.5, 94 same-layer; e.g. five L8 attention seeds `8.attn.9500/10760/
19501/14368/17736` share one circuit; `6.mlp.13113 ~ 6.resid.11113`).

**Families, hubs excluded** (1,040 hubs = 58% of membership removed;
median 128 specific members/circuit): **87 families covering 95.2%** (39
with ≥100 circuits). Cross-layer: 85 of 87 span ≥3 seed layers, dominant
layer holds only 20% of a family. The largest are now organised by
MID-LAYER cores: fam 0 (921 circuits, seeds L4–L7, core L3–L5 attn+resid),
fam 1 (873, deep attention seeds L6–L11, core = six L5 resid latents),
fam 2 (819, L2–L4 attention seeds, core L1–L2), fam 3 (705, L10–L11
seeds, core resid L1/L3/L5/L10). Seeds of a family are of all three kinds.

**Stability (even vs odd shards).** Hub sets: Jaccard 1.00 / 0.97 / 0.96
(top 50 / 200 / 1000). Latent-pair lift: Spearman **0.84** over 1.6M
pairs; 41% of strong pairs are strong in the other half. Family cores:
47% match a core in the other half at Jaccard ≥0.5 (best 1.00, 0.88,
0.83, 0.67…), 47% at 0 — bimodal as in the snapshot, but now L0
token-twin cores replicate too (`0.mlp.7286/0.resid.7286…` 0.88).

**Routing (co-activation).** Cores that fire together on the corpus
(mutual top-128 PMI partners; random 0.00): fam 23 **1.00**, 24 1.00, 26
1.00, 27 1.00, 29 0.79, 16 0.71, 19 0.68, 22 0.46, 13 0.43, 4 0.39, 7
0.29 — all mid-layer resid/attn cores (L2–L8). The L0/L1 token cores and
the big L1–L5 cores of families 0–3: 0.00. Hubs among themselves: 0.09.

**Amplitudes.** 5.10M gains: median 1.056, 70% > 1, 10% > 2, 0.8% < 0.5.
Gain vs attribution corr 0.74. Same-layer members of L11 seeds boosted
3.2×. 26,910 latents with ≥10 memberships: 3,478 consistently boosted
(L11 mlp with gains 5–7 dominate), **5 consistently reduced**, 18 flips
(0.1%), 5,814 near-unity (22%).

**Labels (activation-gated, `label_hubs.py`).** The hubs are the
sequence scaffold: `0.attn.25914` fires on `<s>` (100%), `0.attn.4540`
`<s>` (97%), `0.attn.15350` `.` (100%), `0.mlp.9162` / `0.resid.21893`
`ing` (80%), `0.mlp.40345` `to`, `0.mlp.28097` digits; many others peak at
positions 0–1 with low token consistency (start-of-sequence/position
features). **The five consistently-REDUCED latents are the `<s>`/BOS
hubs** (`0.attn.4540`, `1.attn.12218`, `2.attn.21439`… all peak on `<s>`):
the circuits systematically damp the attention-sink features. Boosted
L11 mlp latents are function-word/next-token features (`the`, `and`,
`to`, `as`). Family cores label as topical: fam 8 core = international /
countries / Union; fam 9 core = conduct(ing) / chemical / reaction.

**Causal edges, snapshot sources L0–L3** (`causal_edges.py`, 297 edges,
B in L1–L3 and L9–L11; effect = fractional drop of B's seed on B's probes):
- A's whole circuit: **0.72** median; matched control circuit **0.64**;
  paired difference +0.05, edge > control on 67% of edges. Removing A
  minus the members it shares with B: 0.015. So the bulk is the L0 hubs
  every circuit carries, plus the directly shared members.
- **Hubs excluded**: A's circuit 0.001 median (mean 0.074; 14% of edges
  > 0.1, 7% > 0.5), control 0.000 (1% > 0.1), random 0.000. A minority of
  L0-source edges are genuine specific dependencies; most are not.
- A's seed latent alone: 0.003 (2% > 0.1). A token feature is one of
  ~300 inputs and is active on few of B's 64 probes — hence the
  A-ACTIVE conditional effect added for the full-set run.
- **B as a unit** (amplitude-weighted state of B's members): drops track
  the endpoint (Spearman 0.66); 0 "shortcut" cases (state broken,
  endpoint intact), 13 "bypass" cases (endpoint drops, B's members
  intact) — A can reach B's seed around B's own circuit.
- **Jacobian vs ablation** of A's seed: Spearman **0.84**, Pearson 0.54,
  first-order prediction = 0.65× the measured drop. Ranks well,
  underestimates magnitude by a third — the linear-attribution lesson at
  circuit level.
- Family cores (snapshot's 10 largest, all L0/L1-cored): core ablation
  ≤0.04 median except the token-twin core `0.attn.6888/0.mlp.1288/…`
  (0.11 median, 33% of seeds > 0.2) — vs 0.00 for random matched sets.
  The co-firing mid-layer cores were not in the top-10 by size; they are
  tested directly in the full-set run (`FAMS=…`).

**Causal edges, FULL SET, sources L3–L9 → B in L4–L11** (239 edges,
`results_full/causal_edges_full.jsonl`; effect = fractional drop of B's
seed on B's 64 probes; mid-layer sources are active on all 64, so the
A-active conditional equals the overall effect):

| condition (ablated on B's probes) | median | mean | edges > 0.1 | edges > 0.5 |
|---|---|---|---|---|
| A's seed latent alone | 0.002 | 0.02 | 5% | 0% |
| A's whole circuit | 0.79 | 0.74 | 99% | 83% |
| matched control circuit | 0.72 | 0.68 | 97% | 81% |
| **A's circuit, hubs excluded** | 0.013 | **0.19** | **31%** | **18%** |
| control circuit, hubs excluded | 0.002 | 0.01 | 3% | 0% |
| random matched set, hubs excluded | 0.000 | 0.00 | 0% | 0% |

- **A third of membership edges are real circuit-level dependencies**
  once the shared infrastructure is removed: 31% of edges drop B by more
  than 10% and 18% by more than half, against 3% / 0% for a size- and
  layer-matched control circuit and 0 for random latents. Paired: edge >
  control on 71% of edges. The rest of the edges are carried by the hubs
  (whole-circuit removal 0.79 vs control 0.72 — the same hub effect).
- **The dependency is strongest for shallow targets and fades with
  depth:** hub-excluded median effect by B layer — L4 **0.61**, L5 0.05,
  L6–L8 ~0.01, L9 0.03, L10–L11 ~0.01–0.02. Deep circuits have hundreds
  of inputs and no single upstream circuit is decisive; an L4 circuit has
  few and depends on them.
- **B as a unit**: member-state drops track the endpoint (Spearman
  0.77); 0 shortcut cases, 16 bypass cases (endpoint falls while B's own
  members stay intact — A reaches B's seed around B's circuit).
- **Jacobian vs ablation** improves with mid-layer sources: Spearman
  **0.89**, Pearson 0.89, first-order prediction = 0.82× the measured
  drop. Linear attribution ranks single-latent edges well and still
  under-predicts magnitude.
- Example edges: `3.mlp.20320 → 4.attn.7775` hub-excluded effect 0.99
  (control 0.00); `3.resid.22706 → 6.resid.2422` 0.82; `6.resid.21018 →
  9.attn.20814` 0.88; `5.resid.40724 → 8.mlp.30174` 0.39 (0.80 on the 22
  probes where A is active — the one edge where the conditional matters).
  And a negative one: `8.resid.29286 → 9.attn.24229` −0.44 (removing A's
  circuit RAISES B; A is an inhibitory dependency).

**Family cores, full set** (12 largest families, cores hub-excluded,
12 seeds each above the core): core-ablation medians 0.00–0.09; family 8
(L0 token-twin core `0.mlp.16181/0.resid.16181/…`) 0.09 with 17% of seeds
> 0.2, family 5 0.04 (17%), all others < 0.05 — random matched sets 0.00.
The largest families are heterogeneous and their 8-latent "cores" are
the most-shared members, not a unit. The co-firing mid-layer cores
(families 23, 24, 26, 27, 29, 16, 19, 22, 13, 4, 7) are tested
separately below.

**Family cores that co-fire, ablated** (`causal_families_cofire.jsonl`,
16 seeds per family, cores hub-excluded, random site-matched sets 0.00
throughout):

| family | what the core labels as | core-ablation median | seeds dropping > 20% |
|---|---|---|---|
| 23 | cuisine: ingred-ients, dishes, local | **0.11** | **25%** |
| 29 | L4–L7 resid/attn (unlabelled mid-layer set) | 0.07 | **31%** |
| 7 | programming: Python, threads, concurrency, algorithms | 0.06 | 6% |
| 16 | mathematics: algebraic, groups, open (topology), numbers | 0.06 | 0% |
| 22 | art: art, colors, oil paint/gesso | 0.045 | 6% |
| 24 | digits / dates (`0.attn.6888`, `0.mlp.1288`, `0.resid.1288`) | 0.03 | 12% |
| 4, 13, 19, 26, 27, 1, 3 | finance; geopolitics; grammar; narrative; … | 0.00–0.025 | 0–6% |

Reading: a family core is NECESSARY for a minority of the seeds that
share it, and never a bottleneck — removing 3–8 latents from circuits of
300 members costs a median of a few percent, with a tail of seeds that
lose 20–80%. Same shape as the edge test: influence is distributed, and
the specific dependencies are a minority of the membership structure.
The "organs" are real (they replicate, co-fire and carry causal weight
for some seeds) but they are shared VOCABULARY the circuits read from,
not modules whose removal switches a function off.

**What the families ARE (`labels_cofire.json`, full-set cores).** The
cores label as TOPIC DOMAINS and scaffolds:
- topic families: 4 finance/corporate (economic, derivatives, Chief
  Executive Officer); 7 programming (Python, developers, threads,
  concurrency, algorithms); 10 biology (species, speciation, habitat,
  thylakoid membrane); 13 geopolitics/war (Union, Renaissance,
  international, troops, Germany); 16 mathematics (algebraic, groups,
  open sets, real numbers); 19 grammar (verb, plural, relative clause,
  German compound); 22 art (art, colors, oil paint); 23 cuisine
  (ingredients, dishes, cuisine); 26 narrative (plot, protagonist); 24
  digits/dates.
- scaffold families: 8 = "the" after "of/from" in formula prose (five
  L0 latents at 98–100% consistency); 1 = position-0/1 fragments; 2 =
  "and" + implementation/compiler; 3 = how / interdisciplinary /
  conduct.

**Query tool (`query_circuits.py`)**, five texts: a query recruits 3–7%
of the circuits (465 prose → 1,013 code), and when a seed fires a median
65–70% of its members are live at or before its peak (amplitude-weighted
64–69%; p90 0.87). Content seeds are legible — "Paris" (L8 resid, 92% of
533 members live), "France", the "5" of 1905 (family 24), "Albert",
"theory", "def" (family 10/7), "if", "store", "drink" — and three
scaffold circuits fire on every query at fixed strength (an L9 resid
seed at position 1; two seeds on `<s>`), hidden with `--skip-pos 1`.

## 0b. Capabilities → sets of production circuits (`task_circuit_sets.py`)

For each task the model can do, the circuits whose SEEDS fire at the
prediction position on the task's prompts and not on the other tasks'
(rate ≥ 0.5, ranked by task-minus-other), then the task metric
logp(target) − logp(contrast) under removal of the set (hubs excluded)
vs size/site-matched random latents vs the other tasks' sets:

| task | prompts | selective circuits | full | remove set (no hubs) | random | other tasks' sets |
|---|---|---|---|---|---|---|
| greater-than | 255 | 20 | 2.29 | **−0.74** | 2.26 | 1.83 |
| agreement | 256 | 9 | 2.64 | **0.79** | 2.53 | 2.33 |
| year frame ("… in the year" → digit prefix vs " of") | 120 | 32 | 4.34 | **0.26** | 4.23 | 3.97 |
| list closure ("a, b, c, d, e," → " and" vs " or") | 120 | 16 | 3.50 | **1.77** | 3.41 | 3.49 |
| code colon ("if n < 2" → ":" vs ")") | 40 | 18 | 1.60 | **−1.58** | 1.59 | 1.55 |
| definition ("… is called" → term vs " the") | 13 | 13 | 1.53 | **−3.85** | 1.55 | 1.51 |

Four tasks are destroyed by removing ~15 circuits' latents; list closure
loses half; controls are inert. The sets are FAMILY-coherent: greater-
than = nine family-24 (digits) circuits at L0–L3 + three family-1 deep
attention circuits (L8–L10); list closure = family 17 (the comma /
apposition family, incl. the L4 circuit that fired on "Paris,");
agreement = two family-8 residual circuits + an 18-member L2 attention
circuit selective at 95%; year = the digit-prefix promoter (L10) + the
"published/titled" circuit (L9); code = the L10 operator circuit.
NOT well-posed: circuit-only sufficiency (everything else zero-filled)
— the all-zero baseline scores above the full model on some tasks, and
these circuits were fitted for their seeds, not for a task endpoint.
Task sufficiency needs a task-fitted circuit (044/047 runners).

## 0c. Greater-than mechanism (`gt_mechanism.py`)

Prompt "… from the year 1964 to the year 19": roles c1 "1", c2 "9",
tens "6", units "4", final f1 "1", f2 "9" (prediction). Position-
restricted removal of the set (hubs excluded), margin / P(>tens)−P(≤tens)
(full 2.28 / 0.74): everywhere −0.72/−0.02; **final "19" only −0.39/0.00**;
**first-year digits only 0.90/0.03**; tens digit only 2.08/0.47; "year"
tokens 2.40/0.78. Read at the first year, consumed at the end — Hanna's
geometry. Circuit roles from seed behaviour:
- **Year-format recognisers (load-bearing):** seeds peaking on the "9"
  of the century and the final "9", 100% of prompts, flat in tens, all
  members digits-family. `2.mlp.6540` alone drops the margin by 2.36 and
  `2.resid.687` by 2.20 — either destroys the task. The "inside a 19xx
  year" precondition.
- **Tens readers:** `3.resid.6422` is a clean DETECTOR (fires 98% at
  tens 2, 93% at 3, 0% at 1); `4.mlp.17443` graded (76% at tens 1 → 33%
  at 8); `3.resid.18945` graded down, output promotes 8/9; `4.mlp.20833`
  peaks on the tens/units digits.
- **Transport:** family-1 attention circuits at the final position
  (`9.attn.7712` peaks at f2 on 199/256, costs 1.20 alone), `10.attn.6471`
  reading the first year, `8.attn.12485`.
- **Output:** not in the set — seed DLAs over the digits are all small
  (≤0.2); the boosting is downstream in late MLPs that fire on many
  tasks and so were not task-selective seeds.
Difference from Hanna: only one clean per-digit detector, because the
selection favours format features (fire on every prompt) over tens-t
detectors (fire on 1/8).

**Members as a computation + the tens swap (`gt_variables.py`,
`results_full/gt_variables.json`).** Every member of the set (1,180
latents, hubs excluded) role-assigned from its activations: FRAME 85
(fire on 100% at the century "9"/final "19", flat in T), TENS READERS
225 (T-dependent at the tens digit — a GRADED magnitude code, not
per-digit detectors: `7.resid.40372` 13.9→2.0 from T=1→8, `7.resid.38758`
0.2→9.6, `6.resid.3320` peaks at 7), TRANSPORT 166 (T-dependent at the
final position; clean carrier `9.attn.17863` 1.2→3.6), OUTPUT 7 (L0–L2
monotone digit-DLA, T-independent "high digit" prior, net +1.30 for
digits > T). Swap: 60 same-noun receiver/donor pairs, |ΔT| ≥ 3; install
the donor's states of a member group at the receiver's positions; score
= fraction of the way the gap mass P(digit ∈ (min T, max T]) moves from
the receiver's value to the donor's. **Tens readers at the tens position
0.84 (argmax legal under the donor's T 85% vs 58% unpatched)**; all
members at that position 0.84; readers + transport 0.75; transport at
the final position alone 0.17; random site-matched latents given the
same values 0.12. The variable T lives in nameable members at one
token and overwriting them moves the model's threshold; the decision is
read from that position by late attention, not from swappable members
at the end.

## 0e. Circuits as ALGORITHMS: the census, and what it found (2026-09-13/14)

Daniel: "I want to see a circuit that computes something… multiple
concepts/variables figuring things out", then "let's find more circuits
as algorithms". Greater-than (§0c) is the worked positive; this section
is the search for more.

**`algorithm_census.py` — 993 seeds scored** (tok_cons<0.5, posctx_sup
>=0.9, >=30 members, + 6 calibration controls; 127 skipped as position-
0/1 attention sinks). Per seed on its 64 stored contexts: `ctx_dep`
(prefix replaced by another context's), `ord_dep` (prefix shuffled),
distal readers (members with >=50% activation mass at offsets <= -2
whose activation there predicts the seed), and a positional OCCLUSION
profile over offsets -2..-9 (one token replaced at a time; gated on
ord_dep>=0.4). score = ord_dep x occ_max x concentration x
sqrt(1+readers at the critical offset). Calibration: gt transport
`9.attn.7712` 0.31 > spatial concept `9.resid.37056` 0.13 >
human-agent `8.resid.16415` 0.00.

- **The signature is confined to ATTENTION seeds**: median score attn
  0.17, mlp 0.00, resid 0.00; peaks L5-L8. Attention is what moves
  information across positions, so this is the expected shape.
- **Negative controls behave**: the sharp single-position seeds are
  literal frames — `7.resid.37795` has ONE distinct token at its
  critical offset (`'one'x64`, the `one's` bigram), `7.resid.32488` has
  `'this'x52` and readers that are literal `'this'` detectors.
- **The distal positives are mostly MEMORISED SEQUENCE CONTINUATION**,
  not computation: `6.attn.9708` "I think, therefore **I** am";
  `7.attn.13805` the `7` of "…Handicapped Children **Act** of 1975";
  `8.attn.23585` the `(` of "Dual Language Immersion **(**DLI";
  `7.attn.8709` `universal` in "**Newton**'s laws of motion and
  universal gravitation" (readers ARE Newton detectors: `6.resid.23701`
  'Newton' 100%, `5.resid.13783` 'Newton'/'Ein' 95%);
  `5.attn.30381` the `1` of "Between the **14**th and **1**7th"
  (readers all '1'-detectors at 100%).
- **METHOD CORRECTION: a within-seed swap cannot work on corpus probe
  contexts.** The 64 contexts are highly redundant (the same sentence
  4-8x), so "low vs high activation" pairs differ by noise, not by a
  variable; every swap score came out ~0 INCLUDING on the controls.
  Algorithms in this model must be exposed with DESIGNED TASK datasets,
  as greater-than was. (`algorithm_inspect.py` runs the reading + swap.)

**Acronym candidate REFUTED (`acronym_probe.py`).** `8.attn.23585`
looked like acronym formation (a known published circuit type). The
model does emit acronyms (13/16 real names — the tokenizer merges
'RAM'/'CP'/'HR'/'ML'/'AB', so a merged token starting with the wanted
initial is a hit; 8/12 novel word combinations). But the single-variable
sweep (words 2-3 FIXED, word 1 varied over 16 initials x 5 tails)
scores **31/80 = 39%**, and success is determined by WORD IDENTITY, not
position: Dual/Quality/Public succeed under all five tails,
Hybrid/Vertical/Eastern/Joint/Basic/Legal/Native fail under all five.
Per-word lexical association, not a copy-the-initial algorithm.

**Agreement: the variable is REPRESENTED but not CAUSALLY CARRIED
(`agree_variables.py`, `results_full/agree_variables.json`).**
"The key from the books" — the verb agrees with the subject (3 back)
and must ignore the attractor (adjacent). 567 member latents over the
9-circuit set. Roles: **43 subject-number readers** with clean
selectivity (`7.resid.23740` 3.24 plural / 0.00 singular, sel +1.00;
`7.resid.23540` its mirror -0.87; `11.mlp.28789` gain 3.48, sel +0.61);
**38 number readers at the FINAL position**, several being the same
latents with FLIPPED sign (`7.resid.23740` +1.00 at the subject, -0.95
at the final position) — the subject-vs-attractor resolution;
**2 output members** with verb-number DLA (`7.resid.23740` +0.29
plural, `9.resid.28714` -0.33 singular).
Swap (60 pairs with OPPOSITE subject number; margin = logp(correct) -
logp(wrong), receiver 2.54, full flip -2.54):

  HARNESS CHECK zero all @ final    402   1.81  score 0.12  13% flipped
  HARNESS CHECK zero all @ subject  296   2.33        0.05   8%
  SUBJECT readers @ subject pos      43   2.55       -0.00   2%
  all members @ subject pos         296   2.47        0.01   5%
  FINAL-number readers @ final pos   38   2.63       -0.04   2%
  random matched @ final pos         38   2.23        0.06   8%

The harness bites (zeroing moves the margin) so the donor swap is a
REAL NULL: the number-selective members do no better than random.
**Interpretation: number is recoverable from the token itself
("key"/"keys") through paths a member-only patch leaves untouched, so
the members can represent it without carrying it. Greater-than differs
because "which tens digit" cannot be read off the token — it must be
computed, so the members had to carry it (swap 0.84).** The contrast
is the result: the method returns nulls, which is what makes the gt
positive meaningful.

## 0d. Are there interesting circuits in TuringLLM? (`interesting_circuits.py`, `interesting_report.py`, `tl_case_study.py`)

Hypothesis (Daniel, 2026-09-12): "TuringLLM doesn't have many circuits
we're interested in". Test: rank all 15k on abstract (low seed
top-token consistency) + specific (low hub share) + composed (layer
spread) + modulating (amplitude work) + depended-on (in-degree), each
z-scored WITHIN seed layer (global z just ranks "deep"); gate sup ≥ 0.9,
≥ 30 members; take the top 33 per layer (396); label each from its own
contexts; run the amp-aware held-out eval on them.

**Eval (385 scored):** held-out F0 with gains median **0.949**, 93% ≥
0.8; α=1 median 0.089; permuted-gain null 0.145; amplitude effect +0.82.
Several L11 resid seeds score −6 to −59 at α=1: members at natural gains
OVERSHOOT the seed and the fitted gains act as brakes.

**What they are (396 labelled):** 241 lexical (≥60% peak consistency,
mostly WORD-ASSEMBLY: trans[par]ent, unpre[ced]ented, nucleos[yn]thesis,
gram[mat]ical), 94 mixed (topic words, phrase frames), 61 non-lexical:
mostly syntactic-frame predictors at L9–L11 ("basis [for]", "by
[which]", "that [studies]") plus ~15 semantic-category seeds at L7–L10.

**Case studies with the fresh-prompt CLASS TEST** (10 unseen instances vs
10 matched non-instances in a frame absent from the contexts):
- `9.resid.37056` **physical-world/spatial**: 8/10 fire (terrain 5.6,
  landscape 5.3, space 5.1 …) vs 0/10 (argument, budget, recipe …). A
  concept. F0 0.90 / α=1 0.12.
- `8.resid.16415` **human agent**: 7/10 vs 0/10 in a definite-human
  frame ("respected the old ___"), 8/10 vs 2/10 in a quoted-noun frame,
  3/10 vs 0/10 in object position; nurse/pilot never fire (vocabulary
  gap). Strongest specific member `7.resid.9964` (93% at the seed's
  position, gain 2.2) has peak tokens -ian/-ator/manager: agentive
  morphology feeds the person feature. F0 0.85 / α=1 0.63.
- `8.resid.1629` looked like PLACE NAMES ("in Paris") — fires equally on
  "in detail / in general / in person / in advance": a complement-of-"in"
  frame feature. FAILS.
- `8.resid.26994` looked TEMPORAL ("over time") — fires equally on "over
  mountains / budgets / borders": complement-of-"over". FAILS.
Plus the greater-than task set (§0c) as the one algorithmic circuit.

Verdict: the population is dominated by word-assembly, syntactic-frame
and topic circuits; a real minority of concept circuits exists and the
sort + class test finds them; roughly half of concept-LOOKING seeds are
preposition-frame predictors that only the fresh-prompt control exposes.
Full census = label all 15k (~4 h local / ~20 min pod) + class-test the
non-lexical ones.

## Scripts

| script | what |
|---|---|
| `interesting_circuits.py` / `interesting_report.py` / `tl_case_study.py` | interestingness ranking (within-layer z), join with labels + amp eval, member-level case studies with the fresh-prompt class test |
| `task_circuit_sets.py` | task → selective production circuits (seed fires at the prediction position on the task, not on other tasks) + causal removal/sufficiency tests |
| `gt_mechanism.py` | greater-than mechanism: per-circuit role positions, tens dependence, digit DLA, members; positional and per-circuit ablations |
| `gt_variables.py` | greater-than as a computation: role-assign every member (frame / tens readers / transport / output) and the tens SWAP with random controls |
| `algorithm_census.py` | rank all non-lexical seeds by the algorithmic signature: order-dependence x positional occlusion x distal readers (§0e) |
| `algorithm_inspect.py` | read a census hit: contexts with critical positions marked, token variety there, readers with labels, reader swap vs control |
| `acronym_probe.py` | acronym capability probe: real names, novel combinations, and the single-variable sweep that refuted the candidate |
| `agree_variables.py` | agreement as a computation: subject vs attractor number readers, output DLA, the number SWAP with harness positive controls |
| `run_stats.py` | every stored per-circuit eval (bare cf/sup, amplitudes, post-analysis, cost) by layer/kind |
| `query_circuits.py` | **query → which circuits lit up**: run a text, encode every site, list every circuit whose seed fired with the fraction of its members active at/before the seed's peak (plain and amplitude-weighted) and its family; `--families` tally, `--sort frac`, `--skip-pos 1` to hide the `<s>`/first-word scaffold circuits; interactive when no `-q` |
| `causal_edges.py` | circuit-level causal test on the GPU: ablate source circuit A (seed / whole / minus shared / hubs excluded / matched control / random) on target B's probes; B read as endpoint and as member state; Jacobian vs ablation; family-core ablation (`FAMS=…`, `N_EDGES=0`) |
| `label_hubs.py` | activation-gated token labels for hubs, boosted/reduced latents and family cores |
| `extract_tables.py` | flatten shard stores → `tables/*.parquet` (27 s for 7.3k circuits) |
| `skip_census.py` | seeds attempted vs circuits by (layer, kind); seconds per seed |
| `graph_analysis.py` | fan-out/fan-in, layer ordering, seed→seed graph + chains, frequency-corrected overlap (lift), Leiden families |
| `families_nohub.py` | families with hub latents excluded (the first pass was dominated by L0 hubs) + near-duplicate pairs |
| `amplitude_profiles.py` | per-latent gain across all its circuits: boosted / reduced / flip / near-unity |

## 1. Skip census (`skip_census.py`) — which seeds cannot be fitted

Skips are ATTENTION seeds only, and early: L1-attn 70.5%, L2-attn 34.3%,
L3-attn 12.4%, L9–L11-attn 2–4%; every mlp/resid seed fits. Two reasons
from the per-seed logs: "empty probe dataset" (no stored positive
contexts: dead/near-dead latents, 355 of the first 519) and "floor
'triple' needs negctx but seed has none" (rare latents below the
8-positive minimum for negative mining, 164). Cost 0.2–0.3 s each. Total
7.1% of attempted seeds. Fit cost per seed: L0 12 s → L3 29 s → L9 72 s
→ L11 86 s (median; p90 within 4 s of median — the cost is set by layer).

## 2. Fan-out and fan-in (`graph_analysis.py`)

- **Fan-out is extremely heavy-tailed.** 313k distinct member latents;
  58% appear in exactly one circuit; the top 1% of latents hold 71% of
  all memberships; 677 latents sit in ≥10% of circuits. The top hub,
  `0.attn.5663`, is in 99.3% of circuits; the top 40 are all L0/L1
  (attn, mlp AND resid). The top-200 hubs by site: L0 122, L1 40, L2 22,
  ≥L3 16. This is the "universal latents" finding of the 32-circuit study
  at scale: a small L0/L1 infrastructure layer is linked into everything.
- **Fan-in (members per circuit)** median 276, p10 47, p90 647, max
  2,311; by seed layer 25 (L0) → 161 (L2) → ~500 (L9–L11). Attention
  seeds are the largest (298 vs mlp 273, resid 240).
- **Circuits are full-stack, not local.** 97.1% of members lie strictly
  below the seed's layer, 2.9% same-layer, 0% above; mean depth below
  the seed 6.5 layers; only 14.6% of members are within one layer of the
  seed. A deep seed's circuit reaches L0.

## 3. Seed→seed graph and chains

Edge A→B when A's seed latent is a member of B's circuit. 1,871 of 7,310
seeds are members of other circuits; 12,159 edges; in-degree median 2,
max 16; out-degree max 2,088 (hub seeds). Longest chain in the snapshot:
7 circuits (`0.resid.9895 → 1.mlp.12513 → 3.attn.30674 → 9.mlp.26300 →
10.mlp.21245 → 11.mlp.35673 → 11.resid.37588`); depth ≥3 for 11.8% of
circuits. Chains are truncated by the missing middle layers — recompute
on the full set.

## 4. Overlap: the density artefact, measured at scale

Every pair of circuits shares ≥1 member (the L0 hubs); Jaccard median
0.07. Raw overlap correlates 0.92 with |A|·|B| — the density artefact
from the 32-circuit study, now on 26.7M pairs. Lift = observed / bipartite
configuration-model expectation removes it (corr −0.19): median lift 1.09
(= chance), p99 8.2; 3.6% of pairs are "strong" (lift ≥3, overlap ≥5).
58 near-duplicate pairs (Jaccard ≥0.5).

**First-pass families (Leiden on the lift graph) were infrastructure,
not organs:** 4 communities of 1,000–2,200 circuits whose ≥50% cores were
65–87 L0/L1 hub latents each — the split was "which L0 set you use",
roughly by seed depth. Hence `families_nohub.py` (section 5).

## 5. Families with hubs excluded (`families_nohub.py`)

Hubs = latents in ≥5% of circuits: 1,100 latents holding 58.9% of all
membership (454 at L0, 182 L1, 134 L2, tapering to 1 at L11). Removing
them from the membership matrix leaves a median of 96 SPECIFIC members
per circuit; 31.8% of pairs still share ≥1 specific member and lift
becomes informative (median 1.52, p99 23.9).

Leiden at resolution 2 on the lift ≥3 graph: **74 families of ≥5
circuits covering 90.2%** (23 with ≥100 circuits, 22 with 20–99, 29 with
5–19). Families are CROSS-LAYER: 72 of 74 span ≥3 seed layers; the
dominant seed layer holds only a third of a family (median purity 0.33),
so they are mechanisms, not "same-layer" groups. Two kinds of core
(latents present in ≥50% of the family's circuits, or the 8 most-shared):

- **Mid-layer residual cores** (families 12, 13, 18, 24, 25, 26, 27, 5,
  9, 16): sets of 3–8 L5–L9 `resid` latents (e.g. `6.resid.19081 +
  7.resid.6912 + 8.resid.414`, present in 89–96% of family 27's 79
  circuits; `8.resid.26816 + 7.resid...` in family 25). Their families are
  DEEP seeds (L9–L11) of all three kinds. These are the organ candidates.
- **L0 token-identity cores** (families 3, 6, 7, 17, 21, 23, 28, 29): sets
  of L0 latents, often resid/mlp TWINS of the same index (`0.mlp.7286 +
  0.resid.7286`, `0.mlp.21371 + 0.resid.21371`, `0.mlp.13821 +
  0.resid.13821`), i.e. the same token feature seen by two SAEs. Their
  families are shallow seeds (L0–L2). These are "circuits that read the
  same token", not mechanisms.
- **The L11 core** (family 4: 247 circuits, 223 of them L11 resid seeds):
  `11.mlp.38334 (72%) + 11.mlp.35662 + 11.attn.16790` — exactly the
  consistently-boosted L11 mlp latents of section 6 (median gains 5.1–5.5).

58 near-duplicate circuit pairs (Jaccard ≥0.5): 44 same-layer, none are
resid/mlp twins of one index; the big ones are pairs of L9–L11 attention
or resid seeds sharing 200–540 members (e.g. `10.attn.14676 ~
10.attn.26748 ~ 11.attn.24297`): several seeds, one circuit.

## 5b. Stability (`stability.py`, even vs odd shards = independent seed samples)

| statistic | replication across halves |
|---|---|
| hub set, top 50 / 200 / 1000 | Jaccard **0.96 / 0.96 / 0.95** |
| latent-pair co-membership lift (822k pairs with ≥3 co-memberships in both) | Spearman **0.77** |
| strong pairs (lift ≥5) found in half 0 | 45% present in half 1, 36% strong there too |
| family cores (≥3 latents, families ≥10) | **bimodal**: 38% match a half-1 core at Jaccard ≥0.5, 54% at 0 |

The cores that replicate are the mid-layer residual ones (`6.resid.19081/
7.resid.6912/8.resid.414` at Jaccard 1.00; `1.resid.8861/3.resid.24606/
7.resid.37114/...` 0.80; `2.resid.35520/7.resid.26272/...` 0.75; the L11
core 0.60). The L0-based cores do not replicate. (The raw fan-out Spearman
of −0.15 is over all 313k latents including one-circuit singletons and is
not meaningful; the hub-set overlaps are the statistic.)

## 5c. Routing / co-activation (`routing_coact.py`)

Do cores FIRE together on real text? Fraction of core pairs that are
mutual top-128 PMI co-activation partners (pipeline `top_coactivation`
store), against site-matched random pairs (0.00 everywhere):

- mid-layer residual cores: family 12 **1.00**, 13 0.83, 18 **1.00**, 24
  0.90, 25 **1.00**, 26 0.86, 27 **1.00**, 10 0.50, 22 0.50, 5/8/9/16 0.39.
- L0 token cores: 0.00 (families 3, 6, 7, 17, 21, 23, 29) — alternative
  tokens, never co-active by construction.
- L11 core (family 4): 0.33 mutual, 0.33 either.
- the top-40 hubs among themselves: 0.09 mutual, 0.50 either.

Reading: the same families that replicate across halves are the ones
whose cores co-fire. The organs, on this half of the run, are **sets of
L5–L9 residual-stream latents that (a) recur as shared members across
families of deep seeds, (b) replicate in independent seed samples, and
(c) co-activate on the corpus**. The L0 cores fail (b) and (c): they are
the token layer, shared because circuits read the same tokens.

## 6. Amplitude profiles (`amplitude_profiles.py`)

2.37M member gains: median 1.05, 69% > 1, 9.7% > 2, only 0.9% < 0.5.
Gain tracks attribution (corr 0.74 with log gain). Gain grows with member
depth: L0 members 1.02 → L9 1.28 → L10 1.52 → L11 3.16 (same-layer
members of L11 seeds are boosted 3×); by distance below the seed it peaks
4–5 layers down (1.14) and returns to ~1.02 at 11 layers.

Per latent (≥10 memberships, 9,746 latents):
- **Consistently boosted: 1,272** — dominated by L11 mlp latents with
  medians 5–7 (e.g. `11.mlp.13730` n=178 median 6.6, `11.mlp.38334`
  n=303 median 5.5). The circuits lean on these far above natural level.
- **Consistently reduced: 5** — and they are the early-attention HUBS:
  `0.attn.4540` (n 4,113, median 0.52, reduced in 91% of its circuits),
  `0.attn.25914` (n 4,244, 0.56), `1.attn.12218`, `2.attn.5354`,
  `2.attn.30504`. Universal latents that circuits systematically damp —
  a candidate "infrastructure the model turns down" (attention-sink-like?
  to label).
- **Flip latents (used both ways): 13 (0.1%).** Practically none — a
  latent's gain is a stable property, not circuit-specific.
- **Near-unity everywhere: 2,410 (25%).** Pass-through members.
- The 30 highest fan-out hubs sit at gain ~1.06 with no flips: the
  infrastructure passes through at its natural level.

## Next (after the full run lands)

1. Rerun all scripts on the complete 15.9k set (chains and mid-layer
   families are the parts that change).
2. Causal circuit-level edges on the pod: ablate A (with amplitudes) on
   B's probes for the seed→seed edges; core-ablation across each family.
3. Label the hubs and family cores (activation-gated auto-interp).
4. Replicate family/hub statistics on arm two (neg-amp) — the stability
   check membership chaos demands.
