#!/usr/bin/env bash
# DAN-64 local smoke test (one RTX, one process). 8 targets (targets_smoke.txt, layers 1-11, all three kinds) x arms
# {close_silent, random_silent, distant_silent, fused_silent} x lambda 1e-3, each circuit scored on the full battery
# (the three held-out eval sets: close, random, distant), plus the random baseline (all four arms) on one target; then
# report.py's smoke checks (README section 7). close_silent at lambda 1e-3 is the production circuit, re-scored.
# Resumable: re-run to continue.
#   - fresh OUT: ~45-55 min (24 fresh fits + 4 random fits at ~45 s, 108 scorer runs at ~3 s)
#   - resuming out_smoke2 (the three-arm smoke, 2026-10-10): its close / random / distant sets, fits, eval rows and
#     random baselines are reused (their set files keep version 071-disjoint-v1); only fused is built, fitted (8 + 1
#     random), scored on the three eval sets (27 scorer runs), and each target's set rows are re-logged: ~10-12 min
# Writes to out_smoke2 (out_smoke holds the old 8-arm design's smoke; its cached sets are refused by version).
#
#   wsl -e bash -lc 'cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2" && bash experiments/071-contrast-sweep/smoke.sh'
#   env: OUT (experiments/071-contrast-sweep/out_smoke2)  SRC_OUT (062 out_full/out)
cd "$(dirname "$0")/../.." || exit 1
E=experiments/071-contrast-sweep
O=${OUT:-$E/out_smoke2}
mkdir -p "$O"
ORDER=list TARGETS=$E/targets_smoke.txt OUT=$O ARMS=close_silent,random_silent,distant_silent,fused_silent LAMS=0.001 \
  RANDOM_TARGETS=2.mlp.10238 PYTHONPATH=src .venv/bin/python -X utf8 -u $E/sweep.py >> "$O/run.log" 2>&1
echo "smoke exit $?" >> "$O/run.log"
tail -3 "$O/run.log"
SMOKE=1 OUT=$O PYTHONPATH=src .venv/bin/python -X utf8 $E/report.py
