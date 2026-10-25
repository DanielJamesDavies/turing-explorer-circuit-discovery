#!/usr/bin/env bash
# 079 silent-position rerun: the pilot's 27 targets. prod / W0 / RS0.25 / RS1 are REUSED from out_smoke4 (circuits and
# rows), so only the four new arms are fitted:
#   RS1_S0.25, RS1_S1   the donor term at weight 1 plus the silent-position term at 0.25 / 1
#   RS0.25_S0.25        the lighter donor weight plus the silent term
#   W0_S1               the silent term ALONE (control: what it does without the donor term)
# ~27 x (3 x ~65 s + ~45 s) + scoring ~= 2 h on the RTX. Resumable: rerun this script after an interruption.
#   wsl -e bash -lc 'cd "<repo>" && nohup setsid bash experiments/079-resample-training/run_silent.sh > experiments/079-resample-training/silent.log 2>&1 < /dev/null & disown; sleep 3'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
D=experiments/079-resample-training
echo "[$(date +%H:%M)] 079 silent rerun start"
# out_silent_v2: the silent term as fixed 2026-10-18 (out_silent/ holds the first target's fits under the
# double-normalised version, kept only as the record of that bug).
ARMS=W0,RS0.25,RS1,RS1_S0.25,RS1_S1,RS0.25_S0.25,W0_S1 REUSE_FROM=$D/out_smoke4 OUT79=$D/out_silent_v2 \
  .venv/bin/python -X utf8 -u $D/fit_score.py
echo "[$(date +%H:%M)] 079 silent rerun exit $?"
OUT79=$D/out_silent_v2 .venv/bin/python $D/report.py > /dev/null 2>&1 && echo "report: $D/out_silent_v2/report.md"
