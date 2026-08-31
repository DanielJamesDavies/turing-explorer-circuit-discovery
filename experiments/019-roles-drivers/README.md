# D2.2(a,b) — Roles for drivers + the rankings archive (2026-08-01)

11 seeds (full D1 panel), arms R (abl-restoration PA) and D (cf-ig_mean
PA) discovered once each with negative_roles="include", exam per K in
{64,256,1024,4096} in two SIZE-MATCHED variants: include (top-K of the
full ranking) vs exclude (top-K among non-inhibitors). 176 rows in
`rows.jsonl`; tables in `summary_tables.md`; run log `run.log`.

**THE RANKINGS ARCHIVE**: full signed rankings saved per (arm, seed) as
`ranking_{R|D}_{comp}_{latent}.jsonl.gz` (score, layer, kind, idx, role,
round). This unblocks D4.3 (containment), D4.4 (anatomy), and D3.6's
containment gate without any rediscovery.

## Verdicts

1. **v1's headline ("excluding inhibitors collapses phi-cf 0.57 ->
   0.04") does NOT survive the frozen exam — it inverts at large K.**
   At K=64 exclusion is a cf wash (median delta 0.000 on both arms). At
   K >= 1024 the include variant OVERSHOOTS (cf medians 1.1-1.2) while
   exclude sits nearer 1: by calibrated |cf-1|, exclude is better on
   7-8/11 seeds at K=1024 (R 0.117 vs 0.185; D 0.174 vs 0.370). The v1
   collapse was an artifact of its regime (cf-local + per-site caps +
   in-distribution eval — the same regime D2.1 just showed is dead).
2. **The REAL role effect is necessity on attention seeds.** On both
   attn seeds the head is inhibitor-heavy (28-48% at K=64) and excluding
   them collapses phi-sup: 9/38734-D 0.899 -> 0.323, 27/6859-R 0.760 ->
   0.200. Everywhere else sup is unchanged (medians 0.95-1.0 both
   variants). Inhibitors are load-bearing for SUPPRESSION, on the kind
   where drive is hardest — not for drive itself.
3. **Role anatomy (part b)**: every full ranking is ~45-50% inhibitors
   regardless of depth or kind — but the HEAD is kind-structured: R's
   resid/mlp heads are nearly inhibitor-free at K=64 (0-5%; its greedy
   rounds admit supports first), D's carry 8-25%, and attn heads are
   inhibitor-heavy for both arms (28-48%). Head inhibitor fraction
   rises toward the global ~45% as K grows.

## Paper consequences

- The roles Q&A beat should now read: "Do inhibitors matter for
  drivers? For driving: no — at driver budgets excluding them changes
  nothing, and at large budgets it improves calibration. For
  suppression on attention seeds: decisively yes (sup 0.9 -> 0.2-0.3)."
- Retracts/supersedes the v1 include/exclude collapse wherever cited.
- The D2.2(c) hub null test stays deferred: no surprising generic-
  suppressor signal surfaced here that demands it.

## Caveats

- Exclusion is post-hoc on a ranking discovered WITH inhibitors
  participating in PA union/rounds — a discovery run with
  negative_roles="exclude" could differ (not tested; v1's comparison
  had the same property).
- pin0 columns are ~0 beyond L5 (known pin-dead-below-K=4096 from D1);
  the cf/sup columns carry the comparison.
- cf>1 overshoot at large K is the same over-drive defect tracked in
  the mask arc; |cf-1| is the calibrated read (T5).
