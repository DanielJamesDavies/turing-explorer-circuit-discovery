# circuit-tracer on its OWN turf: Gemma-2-2B + the default "gemma" scan

2026-08-21/22. Both sides read circuit-tracer's shipped transcoder files
(mwhanna/gemma-scope-transcoders = GemmaScope JumpReLU transcoders, 16k,
one L0 pick per layer) — NO weight conversion. 6 seeds (3 at L4, 3 at
L6), 48 train / 16 held-out windows, BOS prepended, position 0 excluded.

## Why this run exists
The Llama comparison (035-transcoder-compare) used OUR
conversion of EleutherAI TopK skip-transcoders. Its only real
vulnerability was "you dragged their tool onto foreign weights". Here
the tool runs on its own default model, dictionary and loader.

## Gates (all passed, all logged)
* Convention probed (`probe_gemma_tc.py`): input = HF mlp_in/(1+w_pre)
  = TL's unweighted hook (Gemma RMSNorm scales by (1+w)); target =
  post_ffw_ln(mlp_out); FVU 0.157 at pos>=1 (BOS dominates naive FVU);
  measured L0 83 vs advertised 88.
* Harness gate (`check_gtc.py`): identity exactly 0; ablation wired;
  per-layer L0 tracks the ladder.
* Agreement gate (`check_agreement_gtc.py`): TL-vs-HF differ by ~4e-4
  on the stream after layer-0 ATTENTION (configs identical; embeddings
  and resid_pre exact; `debug_agreement_gtc.py`). JumpReLU thresholds
  turn that into support flips for near-threshold features: 0.1-0.9% of
  L0. Gate stated as a DRIFT BUDGET (flips <5%, shared-feature error
  <1%) and passed. Not bit-exact; say so.
* Selfcheck: our members re-scored through the path used for their sets
  reproduce triamp400 on 5/6 seeds bit-exactly; the 6th differs on cf by
  0.6% (bf16 injection noise).

## A turf fact that frames everything: e0 is not zero
Per-layer transcoders cover only the MLP path. Ablating EVERY upstream
feature leaves seeds at 0-75% of natural (e0/a_pos: 75, 45, 7, 31, 0, 0
%). Attention, embeddings and the transcoder error carry the rest. Both
methods share this universe, so the comparison is fair, but raw
necessity reads low for everyone; quote universe-relative necessity
sup_rel = (a_pos - a_sup)/(a_pos - e0).

## Results

Overlap at matched size: median 43% of our members in their top-n
(Llama: 66%); their #1-ranked feature is one of ours on 5/6 seeds; all
but 4 of our 291 members appear somewhere in their ranking.

Function, all arms through one `score()` closure:

| arm | nodes (med) | ampF0 med | ampFM med | sup_rel | cf | both fill bands |
|---|---|---|---|---|---|---|
| tri-amp (ours) | 32 | **1.03** | 0.89 | 0.67 | 0.67 | 3/6 |
| gate-only (ours) | 69 | 0.88 | 0.66 | 0.85 | 0.55 | 3/6 |
| theirs, matched size | 32 | **0.17** | 0.08 | 0.53 | 0.36 | 0/6 |
| theirs, top-4n | 128 | 0.27 | 0.17 | | | 0/6 |
| theirs, top-16n | 512 | 0.42 | 0.32 | | | 1/6 |
| theirs, top-64n | 2048 | 0.52 | 0.50 | | | 1/6 |
| theirs, top-256n | 5376 | 0.77 | 0.76 | | | 1/6 |
| theirs, 20k (export cap) | 20000 | 0.78 | 0.82 | 0.99 | 0.25 | 3/6 |
| null (x3) | 32 | 0.00 | 0.00 | 0.00 | 0.00 | 0/6 |

## Reading
**The Llama result replicates at matched size, and sharpens into a
compactness curve.** At our size, attribution's top-n is necessary
(sup_rel 0.53 vs our 0.67 — the same mechanism is being pointed at) but
reconstructs the seed at 0.17 against our 1.03, landing in the fill
bands on 0/6 seeds where we land 3/6.

Unlike Llama (0.000 at every size), here their ranking DOES reach
faithfulness — at ~600x the node count: median 0.78 / 3-of-6 in band at
20,000 nodes, with a monotone climb through 4n, 16n, 64n, 256n. So on
their own turf the honest statement is not "attribution cannot find a
sufficient set" but "attribution's ordering needs two to three orders of
magnitude more nodes to reach the reconstruction a 32-node learned
circuit delivers". Graded JumpReLU activations and ~64k active features
per prompt make partial reconstruction reachable by big sets; TopK-32
did not.

Seed L4/10430 is the exception that proves the rule: their top-21
already reconstructs at 0.75 and crosses the band at 16n. It is also the
seed where 45% of the activation is out-of-universe, i.e. the in-universe
circuit is tiny and easy.

Attribution answers "what influences this feature"; the exam asks "what
set reproduces it". The gap between those two questions is measured here
in nodes: ~32 vs ~5,000-20,000.

## Caveats
* 6 seeds, one model, one transcoder set (per-layer, not CLT).
* "20k" is an EXPORT CAP on their ranking (36-54k features received
  nonzero attribution); the true full ranking may score higher still.
* Their sets rendered at alpha=1 (they emit no coefficients); gate-only
  is the matched control and beats them at every size up to ~256n.
* Agreement is within a stated drift budget, not bit-exact.
* Position aggregation of their edge weights (sum |edge| over positions
  and prompts) is our choice; other aggregations could reorder the tail.

## Files
`ours_gtc.py` (harness; `port_gtc.py` generates it from the SAE
harness), `check_gtc.py`, `probe_gemma_tc.py`, `gemma_tc_convention.json`,
`check_agreement_gtc.py`, `debug_agreement_gtc.py`, `theirs_gtc.py`,
`probe_attr_gtc.py`, `compare_gtc.py`; data `ours_gtc_rows.jsonl`,
`ours_gtc_members.jsonl`, `theirs_gtc_nodes.jsonl`, `compare_gtc_rows.jsonl`;
logs `run.log`, `theirs_run.log`, `score_theirs.log`, `sweep.log`.
Weights at `$HOME/gemma_tc` (native disk).


## Deep dive + consensus frontier (2026-08-26)

`gemma_deep_dive.py` -> `gemma_deepdive.pdf/png` + `deepdive_stats.md`
(CPU, from stored circuits) and a GPU CONSENSUS-FRONTIER sweep
(`CT_FREQ_SWEEP=48,...,1` in ours_gtc.py, arms `ct_published_f<t>` /
`ct_seed_rooted_f<t>`, log `freq_sweep.log`): each pruned circuit cut
by ITS OWN window-survival frequency (nodes in >= t of 48 windows) --
the strongest ordering their pipeline provides -- and every distinct
cut scored through the standard closure at alpha=1.

Findings: (1) their membership is mostly window-local -- L6 seeds have
ZERO nodes surviving all 48 windows, L4 stable cores are 0.1-3.6% of
the union; (2) the stable cores are NOT faithful (f0 ~0.06-0.4 at our
sizes), so stability and sufficiency come apart; (3) the consensus
frontier enters the band only at ~10^3-10^4 nodes (best steel-man:
median nodes-to-band 6,346 = ~148x ours, down from 425x under plain
truncation -- their own stability signal IS a better ordering than
frequency-head truncation, but the gap stays two orders of magnitude);
(4) recovering our circuit needs top-70..top-10,000 of their orderings
(SFC shallowest); (5) our fitted alphas have median 1.5-2.0 and up to
39x spread within a circuit -- why alpha=1 renderings undershoot and
the hybrid fits rescue them; (6) their sets concentrate on layer 0
(matched head ~88% layer-0) vs our mass spread across the cone.

### Qualitative anatomy + the echo test (2026-08-26)

`fetch_labels.py` caches Neuronpedia auto-interp labels (397 features:
our six circuits, their six matched heads, the seeds) ->
`neuronpedia_labels.jsonl`. `seed_anatomies.py` writes per-seed tables
`anatomy_L<l>_<f>.md` (alpha, their direct-edge rank, SFC rank,
window-survival count, label) + `anatomy_summary.md` +
`alpha_vs_rank.pdf/png`. `echo_vs_context.py` runs the quantitative
test -> `echo_vs_context.md` + figure.

ECHO = member label shares a content word or quoted token with the
SEED's label; CONTEXT = everything else. Label-lexical, descriptive,
inherits auto-interp error -- stated in the file.

CONFIRMED: attribution's ordering is dominated by echoes. Pooled over
291 members, median direct-edge rank 23 (echo) vs 166 (context),
z=-6.47, p=9.5e-11. And our sets are less echo-dense than their
equally sized head on 6/6 seeds (sign test p=0.031, median ratio
0.60) -- i.e. the difference between the methods is in WHICH nodes are
selected, matching the hybrid decomposition.

NOT CONFIRMED, retracted: the first reading of L6/6649 suggested our
ALPHAS preferentially amplify context members. Across all six seeds
that is false -- median alpha 1.57 (echo) vs 1.55 (context), z=0.13,
p=0.9; L4/12424 runs the other way (p=0.014). The single-seed top-5
was not representative. Do not claim amplitude tracks the echo/context
split.

Worked example for the paper (L6/6649, seed = the words "each"/"other"):
their top-8 are "each"-detectors climbing the stack; our set adds
pairs-of-entities, reciprocal-relationship and plural/present-verb
features that their ranking buries at 55-14,476. All 43 of our nodes
appear somewhere in their 20k ranking, so the disagreement is ordering
and selection, never existence.
