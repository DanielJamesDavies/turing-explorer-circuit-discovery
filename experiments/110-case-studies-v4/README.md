# 110 Case studies on the production run (WCM v4, TuringLLM 8k)

## Question

1. Does the paper's case study (9.resid.20419, classical antiquity) hold on the production (v4) circuit and the fresh,
   deduplicated records?
2. Which production circuits would make better case studies on physics, AI and computing?

## Inputs

All inputs are read only; nothing is written to the archives.

- Production circuits and rows: `109-pod-run-analysis/out/archives/pub-turingllm-T3-main/.../094-tl-v2-records/out_v2/main`
  (`circuits/*.pt`, `eval|spec.shard*.jsonl`).
- Fresh records: `.../pub-turingllm-T2-records/.../out_turingllm/ctx_fresh/<key>.pt`.
- The old run, for comparison: `062-h100-protocol-v1/out_full/out` and 063 `results_case/`.
- Tooling: 063 `validate.py`, `case_study.py`, `inspect_circuit.py`, `case_figure.py`. They are imported unchanged and
  pointed at the production setup by `lib110.use_production()` (`OUT`, `CTX_DIR`, `WCM_VERSION=v4`).
- Pass rule: `062/merge.py`, imported via 109's `lib109`.

**GPU.** The Windows Python is CPU-only (torch 2.9.1+cpu). GPU runs go through WSL:

```
wsl.exe -e bash -lc "cd '/mnt/x/.../2' && PYTHONPATH=src .venv/bin/python -X utf8 experiments/110-case-studies-v4/<script>"
```

The first `report.py` run of 9.resid.20419 (contribution table only) ran on the CPU. Everything else ran on the RTX
5070 Ti.

## Scripts

| Script | What | Output |
|---|---|---|
| `lib110.py` | paths, `use_production()`, score table (merge.py rule), circuit / record loaders | |
| `report.py` | 063 Browser on v4 circuits + fresh records: member contribution table, top-N members with own top contexts, old-circuit overlap / old rank | `out/reports/<key>.md`, `out/members/<key>.csv` |
| `case_v4.py` | 063 `case_study.py` unchanged: group shares, knock-outs vs rank-matched random, group edges, probes + circuit-alone reads. Default groups = 063's for 9.resid.20419 minus absent members. `GROUPS_JSON` for other targets | `out/case/<key>.json` |
| `case_figure_v4.py` | 063 figure at 5.5 in; band 0.75; circuit-alone (C) read added to (b,c) | `paper/figures/run2026_case_antiquity.pdf/.png` |
| `topic_scan.py` (CPU) | all 7,786 targets: share of strongest / contrast contexts with physics / AI / computing terms, anchor-is-term share, peak-token consistency, scores | `out/topic_scan.csv` |
| `windows.py` (CPU) | anchor windows + contrast snippets for a list of keys | stdout |
| `probes.py` | clean-model probes per candidate incl. the string-detector control (same word, other sense) | `out/probes/<key>.md` |
| `run_partb.py` | `report.py` + `probes.py` for all candidates in one model load | |

| `case_figure_generic.py` | 2-panel figure (knock-outs; probes clean vs circuit alone) for any worked case | `paper/figures/run2026_case_{fusion,ai,laws}.pdf/.png` |
| `case_figure_fusion_core.py` | 063's antiquity layout (core diagram, two probe panels, knock-outs) for 9.resid.13945 | `paper/figures/run2026_case_fusion_core.pdf/.png` |
| `groups_<key>.json` | role groups (from each member's own top contexts), story list, probes | |

## Results (2026-11-08)

**Part A: 9.resid.20419 on the production circuit.**
- The circuit passes with 945 latents (old 728). Jaccard with the old circuit is 0.30, but the shared latents carry
  70% of the contribution, and the top of the ranking is the same.
- The story groups together still break the target: 0.22 / 0.10 / 0.21 vs random 0.54 / 0.69 / 0.68.
- The classical-Greece group alone is weaker than before: 0.55 / 0.60 / 0.59 vs random 0.66 / 0.85 / 0.83 (old
  0.29 / 0.22 / 0.32). The early-civilisations group took over the leading role: 0.50 / 0.40 / 0.47, edge to classical
  0.25.
- New: run alone, the circuit reproduces the BC/AD contrast (BC years 75–109%, AD 12–21%).
- Details: `out/case/9.resid.20419_numbers.md`. Draft text: `draft_case_study_v4.md`.

**Part B.** See `shortlist.md`.
- 9.resid.13945 (magnetic-confinement fusion) is validated: 10 latents take it to 0.17 / −0.01 / 0.12 vs random
  0.79 / 0.99 / 0.89. It is an amplifier.
- 10.resid.15497 (eponymous laws) is still validated.
- 8.resid.31646 (AI) is a clean latent, but its circuit's knock-outs do not validate the named groups.
