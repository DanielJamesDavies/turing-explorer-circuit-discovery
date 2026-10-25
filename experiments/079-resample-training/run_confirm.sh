#!/usr/bin/env bash
# 079 confirmation run (Daniel, 2026-10-23): 99 targets = 11 per band x kind, disjoint from the pilot's 27 (073's
# stratified order, positions 3..13 of every cell; all have random nulls).
# The arm was chosen on the pilot before this run (out_silent_v2/report.md): RS0.25_S0.25 had the best production
# pass (59%) with resample pass 37%, size 1.35x and all-position r 0.79 (W0 0.76). It is compared paired against W0,
# the production config refitted on this GPU; prod (the 15k circuit) is scored only.
# ~99 x (W0 ~46 s + RS ~70 s + scoring) ~= 5-6.5 h on the RTX. Resumable: rerun after an interruption.
#   wsl -e bash -lc 'cd "<repo>" && bash experiments/079-resample-training/run_confirm.sh > experiments/079-resample-training/confirm.log 2>&1'
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
D=experiments/079-resample-training
echo "[$(date +%H:%M)] 079 confirmation start"
CELL_OFFSET=3 PER_CELL=11 ARMS=W0,RS0.25_S0.25 OUT79=$D/out_confirm \
  .venv/bin/python -X utf8 -u $D/fit_score.py
echo "[$(date +%H:%M)] 079 confirmation exit $?"
OUT79=$D/out_confirm .venv/bin/python $D/report.py > /dev/null 2>&1 && echo "report: $D/out_confirm/report.md"
