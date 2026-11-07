# 040: three-arena method-comparison figure (tri-amp vs external methods)

An early comparison campaign, before WCM. It gathers the comparison rows of three earlier campaigns into one table,
draws faithfulness-vs-size and necessity-vs-size frontiers per arena, and computes the statistics behind them. Our
method here is tri-amp (the learned mask with the triple floor and fitted amplitudes, a predecessor of WCM) and
tri-mask (the same without amplitudes).

## Question

Within each arena, how do tri-amp and tri-mask compare with external circuit methods at matched size, and how many
nodes does each method need to reach the acceptance band?

## What the code does

- `collect.py` reads the rows files and writes `points.jsonl`, one row per (arena, method, seed, size variant) with
  n, f0 (zero-fill faithfulness), fm (mean-fill), sup (necessity) and cf. Sources:
  - **gemma-tc** (Gemma-2-2B, GemmaScope transcoders): `038-transcoder-compare-gemma/ours_gtc_rows.jsonl`;
  - **llama-tc** (Llama-3.2-1B, TopK skip-transcoders): `035-transcoder-compare/ours_llama_rows.jsonl`;
  - **turingllm** (home SAE bank): `029-panel/rows.jsonl` (tri-amp, tri-mask, nulls), `031-matrix/rows.jsonl`
    (ablation-gradient, cf-gradient and restoration rankings) and `034-ge-replication/rows.jsonl` (Ge et al.
    hierarchical attribution).

  Methods fall into four families: OURS, EXT (external methods as shipped: circuit-tracer, SFC attribution patching,
  the home attribution rankings, Ge et al.), HYB (an external selection with our fitted amplitudes) and LAD (random
  null, support-matched null, co-activation). The rows files use different field names for the same scores
  (ampF0 / ampFM, free0 / freeM, ampF0 / ampFMd); the mapping is in `collect.py`.
- `render.py` draws `comparison.pdf` / `.png`: 2 rows (f0, sup) × 3 arenas, log size on x, the band [0.8, 1.25]
  shaded, per-seed dots and median lines over log-clustered sizes. Methods are compared only within an arena.
- `stats.py` writes `stats.md`. Per arena and method: the median f0 at the size nearest tri-amp's (only within 3× of
  it) with a 10,000-resample bootstrap 95% CI over seeds; per-seed wins and losses against tri-amp by |f0 − 1|; and
  nodes-to-band, the smallest measured size with both f0 and fm in [0.8, 1.25] ("never" if none).

```
python experiments/040-comparison-figure/collect.py
python experiments/040-comparison-figure/stats.py
python experiments/040-comparison-figure/render.py
```

## Files

| File | What |
|---|---|
| `collect.py` | Builds `points.jsonl` from the 029 / 031 / 034 / 035 / 038 rows files |
| `points.jsonl` | The tidy table (999 rows) |
| `render.py` | Draws the frontier figure |
| `comparison.pdf`, `comparison.png` | The figure |
| `stats.py` | Matched-size medians, wins / losses, nodes-to-band |
| `stats.md` | Its output |

## Results

From `stats.md` (6 seeds per transcoder arena, 22 on TuringLLM):
- **gemma-tc.** Matched-size median f0: tri-amp 1.14 [0.79, 1.22]; circuit-tracer as published 0.01 [0.00, 0.10];
  SFC 0.14 [0.02, 0.67]; the hybrids sfc+amp 0.91 and ct-rooted+amp 0.87. tri-amp reaches the band at a median of 10
  nodes; no external method as shipped wins on any seed.
- **llama-tc.** tri-amp 0.94 [0.68, 1.43] at a median nodes-to-band of 71; circuit-tracer direct ranking, SFC and the
  nulls are at or near 0.00, and the hybrids sfc+amp (0.92) and coact+amp (0.82) win 4 of 6 seeds each.
- **turingllm.** tri-amp 0.96 [0.90, 1.02], nodes-to-band median 394 (4 of 22 seeds never reach it). The attribution
  rankings reach the band only at about 560,000–590,000 nodes (436–465× tri-amp's size), and Ge et al. never does.

## Status

Superseded. The figure compares tri-amp, not WCM, and several external arms here are our adaptations rather than
the published algorithms. The paper's method comparison is now built from `083-v1-comparison` (our own baseline
rows) and `107-true-baselines` (other papers' methods as published, on their own substrates, with WCM run there too).
The paper does not cite this folder. The frozen protocol for scaling this comparison is `042-protocol-freeze`.
