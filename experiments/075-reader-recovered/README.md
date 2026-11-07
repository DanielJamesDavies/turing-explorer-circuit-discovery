# 075 reader-recovered: does a circuit preserve what its target does downstream?

## Question

The production scores ask whether a circuit reproduces the target latent's own activation. A stricter standard
(judge a simulation by its downstream effect, measured as a KL rather than a correlation) asks whether the circuit's
version of the target has the same *effect* as the real one. Our targets act mainly on other latents, so this
measures recovery on the latents that read the target ("readers"), and also on the logits.

## What the code does

`reader_recovered.py`, per target, on TuringLLM through the 062 driver's `Runner` (imported via
`073-strict-eval/strict_eval.py`, which sets `OUT` to the 15k run, `062/out_full/out`, and `EVAL_SPLIT=explicit`):

1. **Readers.** On the 48 strongest *training* contexts, zero the target latent in the live stream and sum every
   downstream latent's activation (every site after the target's in causal order). The `N_READ` (20) downstream
   latents whose summed activation changes most are the readers; `sign_j` is the sign of that change. No held-out
   text is used to choose them.
2. **Target values.** On the 16 held-out strongest contexts, compute the value the target takes, at every position,
   in a circuit-only run (all other latents live elsewhere):
   - `wcm_C`: the production circuit under the contrast-mean fill, sparsity-preserving (the `freeN_topk` run);
   - `wcm_Z`: the production circuit under zero fill;
   - `rand_C`: the target's random-null circuit (`062/out_random_scale/circuits/<key>.d0.pt`) under C;
   - `empty_C`: the empty circuit under C;
   - `nat`: the target's own natural value (sanity check: must recover 1).
3. **Splice.** Run the live model again with only the target latent set to each of those values (an exact additive
   edit along its decoder direction, SAE error kept), and read the readers and the logits.
4. **Scores.**
   - `reader_rec = Σ_j sign_j (r_splice − r_ablate) / Σ_j sign_j (r_clean − r_ablate)`, summed over positions. 1 = the
     readers see the real target; 0 = they see a zeroed target.
   - `kl_rec = 1 − KL(clean ‖ splice) / KL(clean ‖ ablate)`, mean over positions, with `kl_ablate` =
     KL(clean ‖ ablate) reported alongside.
   - `val_corr_*`: Pearson r of each value with the natural one over all held-out positions.

**Targets:** the 747 random-null targets (those with a `<key>.d0.pt` random-null circuit), in 073's stratified,
interleaved order. Nothing in `out_full` is written.

## Files and commands

| File | What |
|---|---|
| `reader_recovered.py` | The test (GPU) |
| `out/rows.jsonl` | One row per target, the full run (747 rows) |
| `out_smoke/rows.jsonl`, `out_smoke2/rows.jsonl` | Smoke runs on the first 2 targets |
| `run.log` | Console log of the full run |

```
PYTHONPATH=src python experiments/075-reader-recovered/reader_recovered.py
#   env: LIMIT  N_READ (20)  OUT75 (default out/)
```

`REPORT=1` imports a `report` module that does not exist in this folder, so there is no report step; the numbers
below were computed directly from `out/rows.jsonl`.

## Results

From `out/rows.jsonl` (747 targets, 0 error rows, 20 readers on every target). Medians over targets with a defined
value; `reader_rec` is undefined on 26 targets whose readers do not move when the target is zeroed.

| value spliced in | reader_rec median (n) | kl_rec median (n) | value r with natural, median (n) |
|---|---|---|---|
| natural (sanity) | 1.000 (721) | 1.000 (747) | 1.000 (747) |
| WCM circuit, C fill | 1.105 (721) | 0.024 (747) | 0.68 (662) |
| WCM circuit, Z fill | 1.314 (721) | 0.000 (747) | 0.59 (700) |
| random-null circuit, C fill | 0.000 (721) | 0.000 (747) | 0.04 (110) |
| empty circuit, C fill | 0.000 (721) | 0.000 (747) | 0.04 (75) |

`reader_rec` of the WCM circuit under C, by depth band (median, n): L0-4 1.132 (249), L5-7 1.105 (247), L8-11 1.056
(225). Quartiles over all bands: 0.566, 1.105, 1.500. 210 of 721 lie in [0.8, 1.25] (random-null: 2 of 721). 83 are
exactly 0.

Reading:
- On the readers, the WCM circuit's version of the target carries the target's downstream effect, with a median
  slight overshoot (the target is reproduced somewhat above its natural level). The random-null and empty circuits
  carry none of it.
- The KL numbers say little: zeroing a single target barely moves the logits (median `kl_ablate` 4.5e-4 nats, 90th
  percentile 3.7e-3, max 0.079), so `kl_rec` is a ratio of two very small numbers.

## Status and caveats

- Not cited in the paper.
- The circuits and records are the 15k run's (protocol v1, `062/out_full/out`), not the v4 refit. Those records carry
  the known duplicate-context leakage in their held-out sets (experiment 085).
- "Readers" are chosen by the size of the change under a full zeroing, which favours large late latents: of the top
  10 readers stored per row (7,230 in all), 67% sit at residual sites of layers 9–11.
- Of the 83 targets at exactly 0, 82 have no defined value correlation (`val_corr_wcm_C` is null: the circuit's
  target value is constant on the held-out contexts), so the circuit-only run does not make the target fire there.
