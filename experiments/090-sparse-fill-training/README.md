# 090: training WCM's A and C terms under the scorer's sparsity-preserving fill

## Question

In protocol v1, WCM trains its A and C terms under a **dense** mean fill: every non-member latent at its mean, with no
Top-K. The pass rule scores A and C under the **sparsity-preserving** fill (`CircuitOnlyPatcher(respect_topk=True)`):
- every member at α x its live value (members are never evicted, even when more than K are active);
- the top (K − #active members) non-members by mean, at their mean;
- everything else 0.

089 showed that reweighting the loss reaches the dense A / C but not the scored one. Does training A and C under the
scorer's own fill lift the pass-rule A and C, without costing Z or the circuits that already pass?

## Design

**Targets** (`select_targets.py`, seed 20261025, CPU; `targets.json`, `targets.txt`), 20 in three groups, from the
062 production eval (held-out strongest, not vacuous):

| group | n | what |
|---|---|---|
| `fail89` | 8 | 089's failing circuits (Z and necessity pass, A and C < 0.8) |
| `attn` | 4 | deep (L5–11) attention targets with dense A ≥ 0.8 but rule A < 0.5: 2 near-threshold (clean rank ≥ 64: 6.attn.4595, 6.attn.32897) and 2 not (9.attn.29578, 11.attn.13198) |
| `control` | 8 | passing production circuits, one per band x kind cell for 8 of the 9 cells |

**Arms** (`run.py`). Everything else is production v1: γ_Z 1, γ_A = γ_C 0.25, rank-keep 3e-3, λ 1e-3, 400 steps, the
15k run's contexts, split and AdamW.

| arm | A / C training fill |
|---|---|
| `prod` | none: the 062 production circuit, re-scored |
| `dense` | the production dense fill, refitted on this GPU (the in-run baseline) |
| `sp_hard` | sparsity-preserving, hard keep mask (gradients only through kept entries) |
| `sp_ste` | sparsity-preserving, straight-through backward (forward hard, backward as if nothing were dropped) |
| `curriculum` | dense for steps 1–200, then `sp_hard` |

**Engine patch (`sp_engine.py`).** `src/` is not edited. `learned_mask.py` is read, two exact text edits are applied
in memory, and the result is exec'd as `circuit.instrument.learned_mask`:
1. in the mean-floor branch of the A and C patchers, re-impose the scorer's fill (`sp_keep_mask`, `sp_code`). The Z
   patcher has no floor and is untouched;
2. the training loop publishes its step (for `curriculum`).

The keep rule copies the scorer's: the ablation value cast to the code dtype, then to float32, members set to −inf,
`argsort(descending=True)`. At binary gates the training code equals the scorer's patched code exactly.

**Scoring** as 089: `driver.Runner.score` on the held-out strongest 16, and the same scorer on the training probe
(089 `score_train`). Each row also records an eviction diagnostic (`evict_stats`):
- `evict`: the share of active members a literal Top-K over members and means would drop. The scorer never drops a
  member;
- `>K`: the share of (site, anchor) pairs with more than K active members.

**Checks:**
- `test_sp_engine.py` (CPU, fake bank): the edits apply once; the training fill equals the scorer's fill (code and
  transform); more active members than K; `dense` mode is bitwise production; the Z patcher is untouched; hard blocks
  and ste passes gradients of dropped entries; the curriculum switch.
- `check_sites.py` (GPU, real model and bank; `check_sites.log`): at every upstream site of 6.mlp.23423 (19 sites)
  and 11.attn.13967 (33 sites), the training code equals the scorer's (code max diff 0.0, support equal at every site),
  in bf16 and fp32.
- `run.py CHECK=1`: the same equality inside the training engine, on the first target. Its output is not in any log
  in this folder.
- No-op check (`out/tables.md`): 090 `dense` reproduces 089 `prod_refit` exactly on all 8 shared targets (same n, Z, A
  and C), so refits at fixed settings are deterministic on this GPU.

## Files and commands

| File | What |
|---|---|
| `select_targets.py` | builds `targets.json` / `targets.txt` |
| `sp_engine.py` | the in-memory engine patch (`install`, `set_mode`) |
| `test_sp_engine.py` | CPU tests of the patch |
| `check_sites.py`, `check_targets.txt`, `check_sites.log` | site-by-site check against the scorer on the real model |
| `run.py` | fits and scoring, resumable per (target, arm); env `ARMS`, `OUT90`, `LIMIT`, `TARGETS`, `CHECK=1`, `SKIP_PROD=1` |
| `report.py` | `out/tables.md` |
| `smoke_targets.txt`, `smoke.log`, `out_smoke/` | a short-step smoke on 11.attn.13967 (all latents kept; not a result) |
| `run.log` | the full run (6,431 s) |

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/bin/python -m pytest experiments/090-sparse-fill-training/test_sp_engine.py -q
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/090-sparse-fill-training/check_sites.py 6.mlp.23423 11.attn.13967
PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/090-sparse-fill-training/run.py
python experiments/090-sparse-fill-training/report.py
```

## Results

`out/` is gitignored, so the outputs are not tracked. The numbers below are from `out/tables.md` and
`out/rows.jsonl` on the author's machine.

Rule pass counts (held-out strongest, post-Top-K read) and median circuit size:

| arm | all 20 | fail89 (8) | deep attn (4) | controls (8) | median n (all) |
|---|---|---|---|---|---|
| prod | 8/20 | 0/8 | 0/4 | 8/8 | 656 |
| dense | 7/20 | 0/8 | 0/4 | 7/8 | 652 |
| sp_hard | 14/20 | 6/8 | 0/4 | 8/8 | 545 |
| **sp_ste** | **15/20** | **6/8** | 1/4 | **8/8** | 560 |
| curriculum | 12/20 | 4/8 | 1/4 | 7/8 | 544 |

Paired against `dense` (`out/tables.md`, medians over the 20 targets):
- `sp_ste`: ΔZ +0.01, ΔA +0.20, ΔC +0.19, Δnecessity 0.00;
- passes gained / lost: 8 / 0;
- size ratio 0.86;
- 1 target with Z down by more than 0.1.

`sp_hard` is similar (ΔZ 0.00, ΔA +0.12, ΔC +0.15, 7 / 0, size 0.86).

Other observations:
- **Controls:** the sparse arms lower the controls' A and C slightly (`sp_ste` ΔA −0.17, ΔC −0.20) towards 1.0 from
  above (prod medians 1.17 / 1.10). All 8 still pass.
- **Deep attention:** A rises the most (`sp_ste` ΔA +2.12), but only 1 of the 4 passes (6.attn.4595, near-threshold).
- **Trade-off between the fills:** under the sparse arms the dense-fill A / C fall to about 0. The circuits now fit
  the scored fill instead of the dense one (dpass 0/20).
- **Eviction is small:** `evict` 0.016–0.026 median, `>K` 0.000 in every arm.
- **Fit cost** (median fit s / max peak MB): dense 60 / 8,700; sp_hard 77 / 9,413; sp_ste 76 / 8,752; curriculum
  66 / 9,414. Summed over the 20 targets, `sp_ste` takes 1.30x the `dense` fit time (from `t_fit` in
  `out/rows.jsonl`).

**Where it is used:**
- `paper/protocol.md` §12.1 lists this as the evidence for adopting protocol v2 (`sp_ste`, i.e. `train_fill
  "sparse"`, `sparse_grad "ste"`): rule pass 7/20 → 15/20, the 8 earlier failures 0/8 → 6/8, Z unchanged, controls
  still pass, circuits 0.84–0.86x the size, fit time 1.28x.
- `paper/main.tex` line 274 (comment) and lines 1767-1768 cite 090 for the v2 training fill.
- 091 and 092 build on `sp_ste` and its circuits.

## Status and caveats

- **Adopted, then ported.** The `sp_ste` fill is now native in `src/` (`learned_mask.py`: `sparse_keep_mask`,
  `sparse_code`; `WCM_VERSION=v2` in `062/driver.py`). Production v2 also trains against the **scorer's** means
  (`train_means_source = "eval_ref"`), which 090 did not: 090 used the engine's own training means. 093 validates the
  ported version.
- **`sp_engine.py` no longer applies to the current `src/`.** Its first edit anchors the v1 floor branch of
  `LearnedMaskPatcher.transform`. The v2 port moved that branch, and the anchor now occurs 0 times in
  `src/circuit/instrument/learned_mask.py` (checked by reading the file, not by running). `patched_source()`
  therefore raises "anchor found 0 times". The second edit still matches. The same breaks 091 and 092, which build on
  this patch (095 README).
- 20 targets, 12 of them chosen to fail. These are not population rates.
- Scored on the 15k run's as-run held-out sets, which carry the duplicate-context leakage (DAN-157). The comparison is
  paired on identical records.
