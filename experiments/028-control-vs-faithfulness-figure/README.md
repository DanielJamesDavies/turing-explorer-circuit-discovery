# centrepiece (2026-08-07)

*README generated from the scripts' docstrings; the scripts are the record.*

## `collector.py`

Centrepiece-figure data collector: cumulative pinned vs free
faithfulness as members are added in attribution order.

Per seed: one position-aware abl-ig_mean discovery (abs-p50 selection,
the July recipe), members ranked by |attribution| within each site-role
group, then the runner's own score-ranked per-site truncation evaluated
at a log ladder of per-site caps. Three scores per prefix:

  pinned_mean  members clamped to position-specific clean values,
               non-members at posctx means  (node selection)
  free_mean    members live, non-members at posctx means (freeM_dense,
               the SFC-aligned protocol of the July curves)
  free_zero    members live, non-members at zero (free0, the
               fill-independent anchor)

Rows land in curves.jsonl; the July scripts and data were lost in the
2026-07-22 teardown, so shapes should reproduce but numbers are fresh
measurements. Seeds: one shallow / mid / deep (L3 mlp, L7 resid,
L11 mlp — the original depth trio's hosts).

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/collector.py

## `collector2.py`

Centrepiece REFRESH collector (v2): 3 seeds per depth band, chosen
with the PANEL's seed selection (sorted pool head) so the weighted-
circuit markers from 029-panel land on the same seeds; tail
ladder points (16384/65536) added to smooth the free curve's final
rise. Rows -> curves2.jsonl; v1 single-seed data stays in curves2.jsonl.

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/collector2.py

## `render.py`

Render the centrepiece figure from curves.jsonl.

Three panels (shallow / mid / deep seed), x = members added in
attribution order (log), y = score. Per panel: pinned (blue, node
selection) and free under the mean fill (red, the circuit alone), with
the pinned-free gap shaded; free under zero fill as a faint dashed
companion so each free variant names its semantics. Slot 3 (teal) is
reserved for the weighted-circuit markers once matched-seed tri-amp
data exists (the depth-stratified run).

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/render.py

## `render2.py`

Render the REFRESHED centrepiece from curves2.jsonl: per-band median
curves (3 seeds) with individual seed traces ghosted, plus teal
weighted-circuit markers from 029-panel (triamp400 rows on the
same seeds; amplitudes applied, zero fill — different amplitude
semantics from the curves, named in the caption).

  PYTHONPATH=src python experiments/028-control-vs-faithfulness-figure/render2.py

## Result files

`curves.jsonl`, `curves2.jsonl`
