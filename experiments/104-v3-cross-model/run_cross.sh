#!/usr/bin/env bash
# 104 part B: v2 vs v3 on the public models (Pythia-70m, GPT-2 small), local GPU, one GPU job at a time.
# Per model: select_targets.py (records from the local 069 smoke stores) -> MODE=main fits under WCM_VERSION=v2, then
# v3 (lambda 1e-3, 400 steps, SKIP_SPEC=1, ORDER=list, CURVE_STRIDE 20; each arm its own version-marked OUT, both on the
# same records via CTX_DIR) -> src/eval/donor_test.py on both arms (one shared R pool per model). Every step resumes
# (select: cached records + select.jsonl; driver: status rows; donor: rows). Rerun the same command after a stop.
#   wsl (repo root): bash experiments/104-v3-cross-model/run_cross.sh [pythia-70m gpt2-small]
set -u
cd "$(dirname "$0")/../.."
E=experiments/104-v3-cross-model
MODELS=${*:-"pythia-70m gpt2-small"}
mkdir -p "$E/logs"
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PY=.venv/bin/python
say() { echo "[104B $(date '+%F %T')] $*" | tee -a "$E/logs/run_cross.events"; }
for F in $MODELS; do
  O=$E/out/$F
  CTX=$O/rec/ctx
  T=$E/targets_$F.txt
  if [ ! -s "$T" ]; then
    say "$F select start"
    FAMILY=$F $PY -X utf8 -u $E/select_targets.py >> "$E/logs/select_$F.log" 2>&1 || { say "$F select FAILED rc=$?"; exit 1; }
  fi
  say "$F select done: $(wc -l < "$T") targets"
  for V in v2 v3; do
    say "$F fit $V start"
    FAMILY=$F DATA=data_public/$F STORES=outputs_public/$F OUT=$O/$V CTX_DIR=$CTX TARGETS=$T MODE=main \
      WCM_VERSION=$V ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20 SAE_PRELOAD=0 \
      $PY -X utf8 -u experiments/062-h100-protocol-v1/driver.py >> "$E/logs/fit_${F}_$V.log" 2>&1
    rc=$?
    say "$F fit $V end rc=$rc ($(cat $O/$V/main/status.shard*.jsonl 2> /dev/null | grep -c '"error"') error rows)"
    [ $rc = 0 ] || exit 1
  done
  for V in v2 v3; do
    say "$F donor $V start"
    FAMILY=$F DATA=data_public/$F STORES=outputs_public/$F OUT=$O/$V CTX_DIR=$CTX WCM_VERSION=$V SAE_PRELOAD=0 \
      DONOR_TARGETS=$T DONOR_ARMS=main DONOR_POOL=$O/random_pool.pt \
      $PY -X utf8 -u src/eval/donor_test.py >> "$E/logs/donor_${F}_$V.log" 2>&1
    rc=$?
    say "$F donor $V end rc=$rc ($(cat $O/$V/donor/rows*.jsonl 2> /dev/null | grep -c '"error"') error rows)"
    [ $rc = 0 ] || exit 1
  done
done
say "done $MODELS"
