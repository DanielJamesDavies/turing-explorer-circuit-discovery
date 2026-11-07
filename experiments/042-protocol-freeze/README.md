# 042: frozen comparison protocol and 8×H100 run spec (pre-WCM)

A specification, not an experiment: no code and no results. `PROTOCOL.md` fixes what a scaled-up version of the
`040-comparison-figure` pilot would repeat, so that a large run would be the pilot's protocol at more seeds and
nothing else. It was written as a draft for sign-off.

## Question

What exactly should an 8×H100 run of the three-arena method comparison (tri-amp and tri-mask against circuit-tracer,
SFC, the home attribution rankings, Ge et al., hybrids and nulls) hold fixed, and how should it be sharded and set up?

## What the document fixes

`PROTOCOL.md`:
1. **The arenas:** gemma-tc (Gemma-2-2B, GemmaScope transcoders), llama-tc (Llama-3.2-1B, TopK skip-transcoders) and
   turingllm (home SAE bank, the 22-seed panel). Comparisons stay inside one arena. Fixed everywhere:
   `probe_sequence_count` 64, 8 windows per seed, band [0.8, 1.25] on both zero-fill and mean-fill.
2. **The arms**, by their exact tags in the rows files (OURS, EXT, HYB, LAD; the same families as 040).
3. **Gates before scoring:** circuit-tracer provenance at commit `8f1e2438`, pruning identity, harness identity,
   self-check, reproduction of a published graph, batch invariance. The harness is
   `038-transcoder-compare-gemma/ct_faithfulness.py`.
4. **Seed-survival policy** for circuit-tracer's pruning (published, seed-pinned and seed-rooted arms all reported).
5. **Metrics and statistics:** those of `040-comparison-figure/stats.py` (matched size within 3×, wins by |f0 − 1|,
   nodes-to-band, bootstrap CI over seeds).
6. **The run spec:** a proposed 48 seeds per transcoder arena with a deeper layer band; sharding by a
   `SEED_SHARD=<i>/<k>` option to be added to the comparison drivers; a wall-clock estimate from the pilot; the
   environment (pinned requirements, the circuit-tracer submodule, weights on local disk, allocator settings).
7. **A checklist of harness traps** met during the pilot (unregistered arm tags, silent "ALL DONE" on empty rows,
   timings taken while memory spills, and others).
8. **Size policy** for external circuits: natural size (`_med`, `_union`) and size matched to ours (`_matched`), both
   reported.

## Files

| File | What |
|---|---|
| `PROTOCOL.md` | The frozen protocol and the run spec |
| `requirements.txt` | `pip freeze` of the main environment (torch 2.8.0+cu128, transformers 5.1.0) |
| `requirements-ct.txt` | `pip freeze` of the circuit-tracer environment (torch 2.13.0, transformer-lens 3.2.1, circuit-tracer at `8f1e2438`) |

## Results

None. This folder holds no outputs, and no campaign in the repo carries out the run it specifies: the comparison
drivers in 035 and 038 have no `SEED_SHARD` option. Its environment notes and trap checklist were reused by the
tri-amp production run (`045-h100-production`, "H100 checklist").

## Status

Superseded. It freezes a comparison of tri-amp, a predecessor of WCM, and includes adapted versions of other methods.
The current method comparison is `083-v1-comparison` (our own baseline rows) and `107-true-baselines` (other papers'
methods as published, on their own substrates); the current method and evaluation protocol is `paper/protocol.md`
§0. The paper does not cite this folder.
