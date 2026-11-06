#!/usr/bin/env bash
# Tonight's confirmation of the recommended latent setting (Daniel, 2026-11-03: "let's do it! Please set it up for
# tonight"): geometric lr x3.15 (+ the censored-activation term, CENSOR_T 0.05) against geometric x3.15 alone. Starts at
# START_AT (default 18:30); a watchdog kills every 062 driver / donor_test process at STOP_AT (default 08:30). Stages:
#   1  one process at a time, the 48 unused dev targets: finish dev48_geo_x3.15[_censor], then the fixes for its
#      losses: lr warm-up 0.1, the necessity term (NEC_W 1), both, both + censored
#   (4 at the end: seed 1 of both main arms on 102's check panel, R_SEED 1, if the night allows)
#   2  cross-model smoke: GPT-2 small and Pythia-70m, 104's 24 targets and records each, arms production v3 (constant
#      lr), geometric x3.15, + censored (into experiments/104-v3-cross-model/out/<model>/<arm>)
#   3  donor test (src/eval/donor_test.py, protocol v2 12.3: resample_pass = production AND donor recovery A, C in the
#      band) on every OUT above plus check_geo_x3.15[_censor] and dev48_*; each run with the OUT's own fit env (the
#      driver's OUT markers must match); DONOR_VARIANTS=fit (the pass rule's variant; ~3x faster than all three)
# Every step resumes. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_tonight.sh > experiments/107-true-baselines/latent_pilot/tonight.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
X104=experiments/104-v3-cross-model
PY=.venv/bin/python
now=$(date +%s)
START=$(date -d "today ${START_AT:-18:30}" +%s); [ "$START" -le "$now" ] && START=$now
STOP=$(date -d "today ${STOP_AT:-08:30}" +%s); [ "$STOP" -le "$START" ] && STOP=$(date -d "tomorrow ${STOP_AT:-08:30}" +%s)
say() { echo "[tonight] $(date +%H:%M:%S) $*"; }
say "waiting until $(date -d @$START +%H:%M); hard stop $(date -d @$STOP '+%a %H:%M')"
while [ "$(date +%s)" -lt "$START" ]; do sleep 30; done
(
  while [ "$(date +%s)" -lt "$STOP" ]; do sleep 30; done
  say "STOP_AT reached: killing driver / donor_test processes"
  pkill -TERM -f "experiments/062-h100-protocol-v1/driver.py" || true
  pkill -TERM -f "src/eval/donor_test.py" || true
) &
late() { [ "$(date +%s)" -ge "$STOP" ]; }
TL="PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 WCM_VERSION=v3 CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx"
GEO="LR_SCHEDULE=geometric LR_PEAK_MULT=3.15"
done_all() {   # done_all OUT TARGETS: every target has an ok status row (skip without loading the model, ~3-4 min)
  local ok n
  ok=$(cat "$1"/main/status.shard*.jsonl 2> /dev/null | grep -c '"ok": true')
  n=$(grep -c . "$2")
  [ "$ok" -ge "$n" ]
}
fit_tl() {   # name targets extra-env...
  local name=$1 targets=$2; shift 2
  late && return
  done_all "$E/out/$name" "$targets" && { say "skip $name (complete)"; return; }
  say "fit $name"
  env $TL $GEO "$@" MODE=main LAM=1e-3 OUT="$E/out/$name" TARGETS="$targets" CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 \
    $PY -u experiments/062-h100-protocol-v1/driver.py > "$E/logs_tonight_$name.log" 2>&1 || say "FAILED fit $name"
}
# ---- stage 1: ONE process at a time (2026-11-03 11:20: three GPU processes filled the 16 GB card and host RAM).
# First finish the daytime dev48 arms (resumable), then seed 1 of both arms on the check panel; seed 2 runs last.
CHECK=experiments/102-pass-improvement/check_targets.txt
D48=$E/targets_dev48.txt
fit_tl dev48_geo_x3.15 $D48
fit_tl dev48_geo_x3.15_censor $D48 CENSOR_T=0.05
# the fixes for geometric x3.15's dev48 losses (Daniel 2026-11-03 ~17:30: "Please can we try the fixes"):
# an lr warm-up (the x3.15 peak's early spikes: 10.mlp.2064, 11.mlp.39065), the necessity term (2.attn.27772: necessity
# 0.50 at 95 latents), both, and both + the censored term
fit_tl dev48_geo_x3.15_warmup $D48 LR_WARMUP=0.1
# (2026-11-03 21:10) the necessity term arms are dropped: at 23 / 48 it lost 7 passes (+1 / -7 vs geometric alone),
# necessity collapsed to -17..-50 on resid targets and circuits grew to 8-14k latents: the hinge is mis-specified for
# targets with an extreme empty-circuit value. Replaced by the promising combination, warm-up + censored.
fit_tl dev48_geo_x3.15_warmup_censor $D48 LR_WARMUP=0.1 CENSOR_T=0.05
say "stage 1 done"
# ---- stage 2: cross-model
for F in gpt2-small pythia-70m; do
  O=$X104/out/$F
  # (2026-11-04 00:47, Daniel) GPT-2 tonight: layers 0-8 only (16 of 24); the deep targets spill past the 16 GB card
  # and run on the H100s instead. Pythia keeps all 24.
  TG=$X104/targets_$F.txt
  [ "$F" = gpt2-small ] && TG=$E/targets_gpt2-small_shallow.txt
  # (2026-11-03 21:50) the arms follow the likely pick: warm-up (dev48 43 vs production 39, +4 / -0) and warm-up +
  # censored
  for arm in v3prod geo_x3.15_warmup geo_x3.15_warmup_censor; do
    late && break 2
    case $arm in
      v3prod) extra="" ;;
      geo_x3.15_warmup) extra="$GEO LR_WARMUP=0.1" ;;
      geo_x3.15_warmup_censor) extra="$GEO LR_WARMUP=0.1 CENSOR_T=0.05" ;;
    esac
    done_all "$O/$arm" "$TG" && { say "skip $F $arm (complete)"; continue; }
    say "fit $F $arm"
    # DEEP_BS=1 (2026-11-04 00:10 / 00:35): at the config's micro-batch 4 (and still at 2), GPT-2's deep targets
    # (~47 sites) spilled past the 16 GB card into shared memory (5-6 min per target); same effective batch,
    # churn-level membership noise only. Profile (00:39, one deep target, 60 steps): 11.3 GB allocated, 12.7 GB
    # reserved; with the CUDA context and Windows' own use that reaches the ~15.6 GB usable. expandable_segments cuts
    # the allocator's fragmentation (~1.4 GB of the reserve); numerics unchanged.
    env PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 FAMILY=$F DATA=data_public/$F STORES=outputs_public/$F \
      SAE_PRELOAD=0 WCM_VERSION=v3 $extra DEEP_BS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODE=main LAM=1e-3 OUT=$O/$arm CTX_DIR=$O/rec/ctx TARGETS=$TG \
      CURVE_STRIDE=20 ORDER=list SKIP_SPEC=1 $PY -X utf8 -u experiments/062-h100-protocol-v1/driver.py \
      > "$E/logs_tonight_${F}_$arm.log" 2>&1 || say "FAILED fit $F $arm"
  done
done
say "stage 2 done"
# ---- stage 3: donor tests (most important first)
donor_tl() {   # name targets extra-env...
  local name=$1 targets=$2; shift 2
  late && return
  [ -d "$E/out/$name/main" ] || { say "skip donor $name (no fits)"; return; }
  say "donor $name"
  env $TL $GEO "$@" OUT="$E/out/$name" DONOR_TARGETS="$targets" DONOR_ARMS=main DONOR_VARIANTS=fit \
    DONOR_POOL=experiments/073-strict-eval/out/random_pool.pt $PY -u src/eval/donor_test.py \
    > "$E/logs_tonight_donor_$name.log" 2>&1 || say "FAILED donor $name"
}
donor_tl dev48_geo_x3.15 $D48
donor_tl dev48_geo_x3.15_censor $D48 CENSOR_T=0.05
donor_tl dev48_geo_x3.15_warmup $D48 LR_WARMUP=0.1
donor_tl dev48_geo_x3.15_warmup_censor $D48 LR_WARMUP=0.1 CENSOR_T=0.05
donor_tl check_geo_x3.15 $CHECK
donor_tl check_geo_x3.15_censor $CHECK CENSOR_T=0.05
for F in gpt2-small pythia-70m; do
  O=$X104/out/$F
  TG=$X104/targets_$F.txt
  [ "$F" = gpt2-small ] && TG=$E/targets_gpt2-small_shallow.txt
  for arm in v3prod geo_x3.15_warmup geo_x3.15_warmup_censor; do
    late && break 2
    [ -d "$O/$arm/main" ] || continue
    case $arm in
      v3prod) extra="" ;;
      geo_x3.15_warmup) extra="$GEO LR_WARMUP=0.1" ;;
      geo_x3.15_warmup_censor) extra="$GEO LR_WARMUP=0.1 CENSOR_T=0.05" ;;
    esac
    say "donor $F $arm"
    env PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 FAMILY=$F DATA=data_public/$F STORES=outputs_public/$F \
      SAE_PRELOAD=0 WCM_VERSION=v3 $extra OUT=$O/$arm CTX_DIR=$O/rec/ctx DONOR_TARGETS=$TG \
      DONOR_ARMS=main DONOR_VARIANTS=fit DONOR_POOL=$O/random_pool.pt $PY -X utf8 -u src/eval/donor_test.py \
      > "$E/logs_tonight_donor_${F}_$arm.log" 2>&1 || say "FAILED donor $F $arm"
  done
done
# ---- stage 4: the likely pick on 102's check panel (warm-up, warm-up + censored), then their donor tests, if the
# night allows (the watchdog stops it at STOP_AT)
fit_tl check_geo_x3.15_warmup $CHECK LR_WARMUP=0.1
fit_tl check_geo_x3.15_warmup_censor $CHECK LR_WARMUP=0.1 CENSOR_T=0.05
donor_tl check_geo_x3.15_warmup $CHECK LR_WARMUP=0.1
donor_tl check_geo_x3.15_warmup_censor $CHECK LR_WARMUP=0.1 CENSOR_T=0.05
say "DONE"
kill %1 2> /dev/null
