#!/usr/bin/env bash
# D2 (decided 2026-10-24): PRELIMINARY lambda runs on the LOCAL GPU, before the pod pilot. Per model, ~30 targets x 3
# lambdas on the 069 smoke stores (outputs_public/<model>, data_public/<model>: wikitext, 2-4k sequences), then the
# size-only report (pilot_lambda.py, the pre-stated rule with the log-scale "nearest 494"). Nothing here is adopted:
# it says whether the pod pilot's grid brackets the size range, and gives local fit seconds per model (B10).
# The smoke stores are tiny, so contexts are often thin (arm A) or missing (too_few_contexts): read the sizes as an
# indication only; the pod pilot (S5 / S8, 100 targets x 3 lambdas on the real corpora) is what C3 decides on.
#
#   wsl -e bash -lc 'cd "<repo>" && bash experiments/082-public-runs/prelim_lambda_local.sh > experiments/082-public-runs/prelim/queue.log 2>&1'
#   MODELS="pythia-70m" N=30 bash ...           # one model; resumable (driver sweep rows are keyed per target / arm)
# env: MODELS ["pythia-70m gpt2-small"]  N [30]  SEED [0]  FRAME [fired]  GPU [0]  STEPS [400]
#      ARMS_<model> [pythia: W_0.0003,W_0.001,W_0.003; gpt2: W_0.001,W_0.003,W_0.01] (the pod pilot's grids)
#      SKIP_MISSING_SAES_<model> [gpt2: 1, pythia: 0] (a dictionary missing from the local HF cache = a missing site;
#      the log says which)  WAIT_FOR [pgrep pattern of GPU jobs to wait for]
# Estimate (local RTX 5070 Ti; pod plan §3.2 unit costs, sweep = fit + one held-out score, no specificity):
#   pythia-70m 90 fits x ~40 s = ~1.0 h;  gpt2-small 90 fits x ~210 s = ~5.3 h;  both ~6.5 h (est., UNVERIFIED on the
#   smoke stores: thin contexts make fits cheaper, the first GPT-2 fits at 400 steps have never been timed locally).
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-0}
PY=${PY:-.venv/bin/python}
E=experiments/082-public-runs
W=$E/prelim
MODELS=${MODELS:-"pythia-70m gpt2-small"}
N=${N:-30}
SEED=${SEED:-0}
FRAME=${FRAME:-fired}
STEPS=${STEPS:-400}
WAIT_FOR=${WAIT_FOR:-"062-h100-protocol-v1/(driver|random_null|stable_core|rescore_alpha1).py|084-harder-nulls/harder_nulls.py|083-v1-comparison/|073-strict-eval/strict_eval.py|086-heldout-rebuild/"}
mkdir -p "$W"
# pgrep -f also matches THIS script's own launcher (e.g. bash -lc '... WAIT_FOR="<pattern>" bash prelim...') whenever
# the pattern text appears in its command line, which blocked the queue twice on 2026-10-24. So the wait ignores this
# process, its ancestors and all of its descendants (command-substitution / pipeline subshells).
mine() {
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do echo "$p"; p=$(ps -o ppid= -p "$p" 2> /dev/null | tr -d ' '); done
  ps -eo pid=,ppid= 2> /dev/null | awk -v r=$$ '{ pp[$1] = $2 } END { for (p in pp) { q = p; while (q > 1 && (q in pp)) {
    if (q == r) { print p; break }; q = pp[q] } } }'
}
others() { pgrep -f "$WAIT_FOR" | grep -vxF -f <(mine); }
n=0
while [ -n "$(others)" ]; do
  [ $((n % 30)) = 0 ] && echo "[$(date +%H:%M)] waiting for: $(others | head -3 | xargs -r ps -o pid=,args= -p | cut -c1-160)"
  n=$((n + 1)); sleep 60
done
for F in $MODELS; do
  v=ARMS_${F//-/_}; d=W_0.0003,W_0.001,W_0.003; [ "$F" = gpt2-small ] && d=W_0.001,W_0.003,W_0.01
  ARMS=${!v:-$d}
  v=SKIP_MISSING_SAES_${F//-/_}; d=0; [ "$F" = gpt2-small ] && d=1
  SKIP=${!v:-$d}
  T=$W/$F
  echo "[$(date +%H:%M)] $F: targets ($FRAME, seed $SEED, $N pilot) -> $T"
  # a separate frame draw on the smoke stores (not the pod's frozen lists): its pilot list is the prelim sample
  "$PY" -X utf8 $E/make_targets_public.py --family "$F" --frame "$FRAME" --seed "$SEED" --stores "outputs_public/$F" \
    --out "$T" --n-pilot "$N" > "$T.targets.log" 2>&1 || { echo "  targets failed (see $T.targets.log)"; continue; }
  echo "[$(date +%H:%M)] $F: sweep $ARMS over $(wc -l < "$T/targets_${F}_pilot.txt") targets (steps $STEPS)"
  FAMILY=$F DATA=data_public/$F STORES=outputs_public/$F OUT=$W/out_$F TARGETS=$T/targets_${F}_pilot.txt MODE=sweep \
    SWEEP_ARMS=$ARMS ORDER=list STEPS=$STEPS SKIP_MISSING_SAES=$SKIP SAE_PRELOAD=0 \
    "$PY" -X utf8 -u experiments/062-h100-protocol-v1/driver.py >> "$W/sweep_$F.log" 2>&1
  echo "[$(date +%H:%M)] $F: sweep exit $? ($(grep -c '"error"' "$W"/out_$F/sweep/eval.*.jsonl 2> /dev/null | paste -sd+ | bc 2> /dev/null || echo ?) error rows)"
  grep -h "site skipped" "$W/sweep_$F.log" | sort -u | head -5
  "$PY" -X utf8 $E/pilot_lambda.py --family "$F" --out "$W/out_$F" --targets "$T/targets_${F}_pilot.txt" \
    --json "$W/out_$F/pilot_lambda.json" > "$W/report_$F.txt" 2>&1
  echo "[$(date +%H:%M)] $F: report $W/report_$F.txt"; tail -n 2 "$W/report_$F.txt"
done
echo "[$(date +%H:%M)] prelim done"
