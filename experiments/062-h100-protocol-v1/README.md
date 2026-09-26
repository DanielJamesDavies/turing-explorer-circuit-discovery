# 062 — Protocol-v1 runs on H100s

2026-09-26. The first at-scale runs under the final protocol (DAN-75 spec). They run the protocol through the same
injection harness as 059 / 061 (`../059-context-pool/protocol_harness.py`), which reproduces local results exactly.
DAN-78 will later move the context protocol into the production path.

| Run | Targets | What | Output |
|---|---|---|---|
| **main** (stage 1) | 1,416 (`targets_stage1.txt`: 40 per layer × site-kind cell, 35 cells, + the 16 pilots) | contexts → fit (γ 0.25, λ 1e-3, rank-keep 3e-3, 32 + 16 contexts) → eval pass on held-out strongest and mid-band → specificity | `out/main/` |
| **sweep** (Figure 4 / DAN-76 / DAN-77) | 191 (`targets_sweep.txt`: 5 per cell + the 16 pilots) | WCM (λ 2.5e-4 … 4e-3) and unweighted masking (λ 1e-5 … 1e-3), 10 arms, eval on held-out strongest | `out/sweep/` |

The sweep reuses main's contexts, and main's circuits for its WCM λ = 1e-3 arm, so run main first (or `MODE=both`).

**Files:**
- `driver.py`: one process per GPU, `SHARD=i/k`, resumable, one file per target for contexts and circuits.
- `launch.sh`: one shard per GPU, retries per shard, staggered start-up.
- `merge.py`: summary, safe to run any time.
- `make_targets.py`: target lists (already generated and committed).

## Timings (local smoke test, RTX 5070 Ti)

Per target: contexts ~2 s (a one-off ~90 s cache warm-up on the first target), fit 23–43 s, eval 3–6 s,
specificity ~1 s. So ~30–50 s per target locally. On H100 expect about half that, so on 8×H100:

- **main (1,416 targets):** ~1–1.5 h.
- **sweep (191 × 10 fits + evals):** ~1–1.5 h.
- `merge.py` projects the full 15,046-target run from the measured throughput; the old estimate was ~13 h on
  4×H100.

## How to run

**1. Commit and push (local).** The pod pulls `multi-device` from GitHub, and none of this week's code is committed
yet.

```bash
git add -u
git add src tests paper experiments/049-circuit-graph experiments/055-close-contrast experiments/056-specificity experiments/057-wcm-edges experiments/058-split-ordering experiments/059-context-pool experiments/060-running-example-v1 experiments/061-lambda-sweep experiments/062-h100-protocol-v1
git commit -m "Protocol v1: stratified contexts, rank-keep, specificity, WCM edges, H100 driver"
git push origin multi-device
```

`*.pt`, logs and 049's tables are git-ignored, so this adds code, READMEs, target lists and small results.
Experiments 050–054 are left out; 052 holds ~120 MB of jsonl results.

**2. Start the pod and bootstrap.** RunPod H100 with the global volume (`inland_copper_crab`) at `/workspace`. The
global volume disallows chmod, so git lives on the pod disk and the volume holds only models / data /
`runs_store.tgz`.

```bash
git clone --branch multi-device https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git /root/turing && bash /root/turing/scripts/pod_bootstrap_global.sh
```

This copies models / data / discovery artifacts (including `seq_repr.pt`, which the close contrast selector needs)
from the volume to `/root/turing`, and builds the venv. Everything the run writes (`out/`, `logs/`) is on the pod
disk, which is wiped when the pod stops, so step 6 is not optional.

**3. Smoke test on the pod (~5 min, 1 GPU).** It checks everything loads and one target runs end to end.

```bash
cd /root/turing && OUT=experiments/062-h100-protocol-v1/out_smoke TARGETS=experiments/062-h100-protocol-v1/targets_smoke.txt MODE=main PYTHONPATH=src ./.venv/bin/python -X utf8 experiments/062-h100-protocol-v1/driver.py
```

Expect `2.attn.33479 ok n=309`, the same circuit as locally.

**4. Launch both runs** (8 GPUs; set `K` to the GPU count).

```bash
cd /root/turing && mkdir -p experiments/062-h100-protocol-v1/logs && chmod +x experiments/062-h100-protocol-v1/launch.sh && MODE=both K=8 nohup ./experiments/062-h100-protocol-v1/launch.sh > experiments/062-h100-protocol-v1/logs/launch.log 2>&1 &
```

**4b. Optional: continue to the full 15,046 targets (DAN-75) in the same session.** Run this after checking stage 1's
summary. `targets_full.txt` lists the stage-1 targets first, and resume works across shards, so the 1,416 already
done are skipped. Expect ~10–13 h on 8×H100 at the measured speed; `merge.py` prints a projection.

```bash
cd /root/turing && MODE=main TARGETS=experiments/062-h100-protocol-v1/targets_full.txt K=8 nohup ./experiments/062-h100-protocol-v1/launch.sh > experiments/062-h100-protocol-v1/logs/launch_full.log 2>&1 &
```

Integrity note: fix the pass rule (DAN-8) before reading the full run's results. Stage 1 is the calibration set; the
full run is the confirmatory one.

**5. Watch.**

```bash
tail -f /root/turing/experiments/062-h100-protocol-v1/logs/main.shard0.log
```

```bash
cd /root/turing && GPUS=8 PYTHONPATH=src ./.venv/bin/python experiments/062-h100-protocol-v1/merge.py | head -40
```

**6. Copy results to the volume before stopping the pod.** Contexts are small, circuits a few GB.

```bash
mkdir -p /workspace/results && cd /root/turing/experiments/062-h100-protocol-v1 && tar czf /workspace/results/062-out-$(date +%Y%m%d-%H%M).tgz out logs
```

Then download the tarball and unpack it into this folder locally (`out/`, `logs/`).

## What comes out

`out/summary.md` (from `merge.py`):
- **Bookkeeping:** targets ok / skipped / failed, thin targets, per-stage timings, and the full-run projection.
- **Circuits (the paper's first real headline):** the size distribution (share inside 10²–10³); faithfulness
  Z / A / C, necessity and sufficiency to induce on held-out strongest and mid-band; the illustrative band
  [0.8, 1.25]; by site kind and by layer.
- **Specificity:** the amplifier rate, by site kind; near-threshold targets (clean rank ≥ 64) vs the rest.
- **Sweep:** per method × λ medians, the per-target natural-scale cost by depth band, and
  `faithfulness_vs_size.png` (one row per depth band).

`out/targets.csv` has one row per target, for picking case studies and for DAN-19 / DAN-20.

**Not included** (later runs): the random-circuit baseline with fitted coefficients (DAN-17), stable-core refits
(DAN-18), wired edges (057), the attribution / external-method curves for Figure 4 (DAN-77), and the public
models.
