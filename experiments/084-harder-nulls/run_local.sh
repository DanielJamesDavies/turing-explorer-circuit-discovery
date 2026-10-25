#!/usr/bin/env bash
# 084 harder nulls (DAN-181), local queue on the RTX 5070 Ti: TuringLLM, the frozen 126-target panel
# (targets_turingllm_nulls.txt: 14 per depth band x site kind from the 062 sweep targets), anchor + contrib nulls at
# WIN 8 with the production members in the pool, 1 draw each (4 on the first 63 = 7 per cell, the DAN-17 sweep
# targets), fitted and alpha = 1, plus the WCM and DAN-17 references on the same machine (65 stored DAN-17 circuits,
# 61 drawn + fitted here); then the report. Resumable at every (target, arm, draw, split): rerun after an interruption.
# On the pod the same work is chain.sh's TuringLLM track T4 (after the held-out rebuild).
#   wsl -e bash -lc 'cd "<repo>" && bash experiments/084-harder-nulls/run_local.sh > experiments/084-harder-nulls/run_local.log 2>&1'
# env: GPU [0]  WAIT_FOR [pgrep -f pattern of jobs to wait for; default: other experiment fits on this GPU]
#      PROFILE_FIRST [1]: profiles + coverage of all 126 first (cheap, no fits), so the pool definition can be checked
#      before the fits; any harder_nulls.py env passes through (N_TARGETS, SUB_N, SUB_DRAWS, NULLS, WIN, ...)
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
D=experiments/084-harder-nulls
export OUT84=${OUT84:-$D/out}
WAIT_FOR=${WAIT_FOR:-"062-h100-protocol-v1/(driver|random_null|stable_core|rescore_alpha1).py|079-resample-training/fit_score.py|083-v1-comparison/|073-strict-eval/strict_eval.py"}
mkdir -p "$OUT84"
n=0
while pgrep -f "$WAIT_FOR" > /dev/null; do
  [ $((n % 30)) = 0 ] && echo "[$(date +%H:%M)] waiting for: $(pgrep -af "$WAIT_FOR" | cut -c1-160 | head -3)"
  n=$((n + 1)); sleep 60
done
echo "[$(date +%H:%M)] 084 start (GPU $CUDA_VISIBLE_DEVICES, OUT84 $OUT84)"
if [ "${PROFILE_FIRST:-1}" = 1 ]; then
  PROFILE_ONLY=1 .venv/bin/python -X utf8 -u $D/harder_nulls.py
  echo "[$(date +%H:%M)] profiles exit $?"
fi
for attempt in 1 2 3; do
  .venv/bin/python -X utf8 -u $D/harder_nulls.py && break
  echo "[$(date +%H:%M)] harder_nulls exited nonzero (attempt $attempt); resuming in 60 s"
  sleep 60
done
echo "[$(date +%H:%M)] 084 fits done"
.venv/bin/python $D/report.py > /dev/null 2>&1 && echo "report: $OUT84/report.md"
