#!/bin/bash
# DAN-64 pod launcher, modelled on 062's launch.sh: K GPUs x P sweep.py processes (SHARD=i/N, shard i on GPU i % K,
# SLOT = i // K) under CUDA MPS, retry per shard (a shard that dies restarts and resumes), one log per shard, then
# report.py. Run from the repo root on the pod, after README section 6 (contexts + production circuits in $SRC_OUT).
# Not to be started without Daniel's go-ahead on the exact command.
#
#   K=4 P=4 nohup ./experiments/071-contrast-sweep/launch_pod.sh > experiments/071-contrast-sweep/logs/launch.log 2>&1 &
#   (ARMS / LAMS / TARGETS / RANDOM_TARGETS / OUT / SRC_OUT set on this command reach every shard)
# Progress:  tail -f experiments/071-contrast-sweep/logs/run.shard0.log
# Report:    PYTHONPATH=src ./.venv/bin/python experiments/071-contrast-sweep/report.py   (any time; partial is fine)

K=${K:-4}             # GPUs
P=${P:-4}             # processes per GPU (MPS; 062 ran 6 per GPU on 8xH100)
N=$((K * P))
THREADS=${THREADS:-$(( $(nproc) / N ))}
[ "$THREADS" -gt 8 ] && THREADS=8
[ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
D=experiments/071-contrast-sweep
PY=${PY:-./.venv/bin/python}
export OUT=${OUT:-$D/out}
export SRC_OUT=${SRC_OUT:-$D/src_out/out}
TARGETS=${TARGETS:-$D/targets.txt}
export TARGETS
mkdir -p "$D/logs" "$OUT"

# preflight: every target's context record and production circuit must be in SRC_OUT (README section 6)
miss_ctx=0; miss_c=0
for k in $(cat "$TARGETS"); do
  [ -f "$SRC_OUT/ctx/$k.pt" ] || miss_ctx=$((miss_ctx + 1))
  [ -f "$SRC_OUT/main/circuits/$k.pt" ] || miss_c=$((miss_c + 1))
done
if [ $miss_ctx -gt 0 ] || [ $miss_c -gt 0 ]; then
  echo "preflight: $miss_ctx context records and $miss_c production circuits missing under $SRC_OUT; stopping"
  exit 1
fi
echo "preflight ok: $(wc -l < "$TARGETS") targets, records + circuits in $SRC_OUT"

# CUDA MPS (062: ~1.6x the throughput of plain time-slicing at 4-8 processes per GPU)
if [ "${MPS:-1}" = "1" ] && command -v nvidia-cuda-mps-control > /dev/null; then
  if pgrep -x nvidia-cuda-mps-control > /dev/null; then
    echo "MPS daemon already running"
  else
    nvidia-cuda-mps-control -d && echo "MPS daemon started"
  fi
fi

for i in $(seq 0 $((N - 1))); do
  (
    tries=0
    until CUDA_VISIBLE_DEVICES=$((i % K)) SLOT=$((i / K)) SHARD=$i/$N MODE=run \
        PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        $PY -X utf8 -u $D/sweep.py >> "$D/logs/run.shard$i.log" 2>&1; do
      tries=$((tries + 1))
      echo "shard $i exited nonzero (attempt $tries)" >> "$D/logs/run.shard$i.log"
      [ $tries -ge 5 ] && { echo "shard $i GIVING UP" >> "$D/logs/run.shard$i.log"; break; }
      sleep 30
    done
    echo "shard $i finished" >> "$D/logs/run.shard$i.log"
  ) &
  sleep ${STAGGER:-20}   # model + store loading; the first selection also builds the negctx token cache
done
wait
echo "all $N shards returned ($K GPUs x $P per GPU, $THREADS CPU threads each)"
PYTHONPATH=src $PY -X utf8 $D/report.py > "$D/logs/report.log" 2>&1 && echo "report: $OUT/report.md"
