#!/bin/bash
# ONE-COMMAND POD SETUP for the 082 chain (public tracks + TuringLLM track), written 2026-11-07 from the first 5x H100
# launch. It runs the two existing bootstraps with every fix that launch needed, then a GPU HEALTH CHECK, then prints
# the dry-run and launch commands (it never launches the chain itself).
#
#   git clone --branch multi-device https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git /root/turing \
#     && bash /root/turing/scripts/pod_quickstart.sh 2>&1 | tee /root/quickstart.log
#   (meanwhile, from the local PC:  scp -P <port> -i ~/.ssh/id_ed25519 tl-extras.tgz tl-extras.md5 root@<ip>:/root/ )
#
# Fixes built in (each cost time on 2026-11-07):
#   - HF_HOME / pip cache on the pod disk, not on the network volume (the template points both at /workspace);
#     HF_HUB_DISABLE_XET=1 (xet-read-token 404 for gpt2). Written to ~/.bashrc so every later shell (and the launch)
#     sees the same cache.
#   - public bootstrap with SKIP_TESTS=1 (34 CPU tests, 16 min, passed on this commit 2026-11-07); its GPU check runs.
#   - /workspace/data (13 GB, 3,031 small files + hidden .shard_indices/) copied 8-way parallel (cp -r: ~0.5 MB/s).
#   - tl-extras.tgz unpacked with --no-same-owner (made on Windows: tar's chown fails and the bootstrap stops).
#   - the two bootstraps run in parallel.
# GPU HEALTH CHECK: a 60 s matmul load on every GPU at once, sampling SM clock / temperature / power and the
#   thermal-slowdown flags, plus each GPU's lifetime thermal-slowdown counter. 2026-11-07's pod had a GPU stuck at
#   420-1,230 MHz and 86 C at 200 W (SW thermal slowdown); the launcher shards statically, so one slow GPU gates every
#   fit job. A FLAGGED GPU means: rent another pod (or leave that GPU out of FIT_GPUS / DATA_GPU_*).
#
# ENV: SKIP_TESTS [1]  BURN_S [60]  EXTRAS_WAIT_MIN [30] (how long to wait for /root/tl-extras.tgz)
set -uo pipefail
REPO=https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
BRANCH=${BRANCH:-multi-device}
PUB=/root/turing
TL=/root/turing-tl
LOGD=/root/quickstart
mkdir -p "$LOGD"
T0=$(date +%s)
say() { echo "[quickstart $(date +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 ))m] $*"; }

# ---------------------------------------------------------------- 0. environment (persisted for later shells)
grep -q 'HF_HUB_DISABLE_XET' ~/.bashrc 2> /dev/null || \
  echo 'export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 PIP_CACHE_DIR=/root/.cache/pip' >> ~/.bashrc
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 PIP_CACHE_DIR=/root/.cache/pip
say "env: HF_HOME=$HF_HOME, xet off, pip cache on the pod disk (also in ~/.bashrc)"

# ---------------------------------------------------------------- 1. volume check
miss=()
for p in /workspace/models /workspace/data /workspace/runs_store.tgz /workspace/results/062-full-20260927-1924.tgz; do
  [ -e "$p" ] || miss+=("$p")
done
[ ${#miss[@]} -eq 0 ] || { say "MISSING on the volume: ${miss[*]} (is inland_copper_crab mounted at /workspace?)"; exit 1; }
NG=$(nvidia-smi -L | wc -l)
say "volume ok; $NG GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | tr -s ' ')"

# ---------------------------------------------------------------- 2. public bootstrap (background)
[ -d "$PUB/.git" ] || git clone --branch "$BRANCH" "$REPO" "$PUB" || exit 1
say "public bootstrap started (log $LOGD/public.log)"
( SKIP_TESTS=${SKIP_TESTS:-1} EXPECT_GPUS=$NG bash "$PUB/scripts/pod_bootstrap_public.sh" > "$LOGD/public.log" 2>&1; \
  echo "rc=$?" > "$LOGD/public.rc" ) &
PUB_PID=$!

# ---------------------------------------------------------------- 3. TuringLLM tree: pre-stage the slow copies
[ -d "$TL/.git" ] || git clone --branch "$BRANCH" "$REPO" "$TL" || exit 1
if [ ! -d "$TL/models" ]; then
  say "copying /workspace/models (~15 GB) in the background"
  ( cp -r --no-preserve=mode,ownership /workspace/models "$TL/models.tmp" && mv "$TL/models.tmp" "$TL/models" ) &
  MOD_PID=$!
fi
if [ ! -f "$TL/data/.quickstart_copied" ]; then
  say "copying /workspace/data (13 GB, ~3k files) 8-way parallel"
  mkdir -p "$TL/data"
  (cd /workspace/data && ls -A | xargs -P 8 -I{} sh -c \
    '[ "$(stat -c %s "'"$TL"'/data/{}" 2> /dev/null)" = "$(stat -c %s "{}")" ] || cp -r --no-preserve=mode,ownership "{}" "'"$TL"'/data/"')
  nv=$(cd /workspace/data && find . -type f | wc -l); nl=$(cd "$TL/data" && find . -type f | wc -l)
  bv=$(du -sb /workspace/data | cut -f1); bl=$(du -sb "$TL/data" | cut -f1)
  say "data: $nl / $nv files, $bl / $bv bytes"
  [ "$nl" = "$nv" ] || { say "data copy INCOMPLETE (files differ); re-run this script (it skips finished files)"; exit 1; }
  touch "$TL/data/.quickstart_copied"
fi
[ -n "${MOD_PID:-}" ] && { wait "$MOD_PID" || { say "models copy FAILED"; exit 1; }; }
say "models: $(du -sh "$TL/models" | cut -f1)"

# ---------------------------------------------------------------- 4. the local-only extras (sent from the PC)
n=0
until [ -f /root/tl-extras.tgz ] && [ -f /root/tl-extras.md5 ] && (cd /root && md5sum -c --quiet tl-extras.md5 2> /dev/null); do
  [ $((n % 5)) = 0 ] && say "waiting for /root/tl-extras.tgz + .md5 (scp them from the PC); $n min so far"
  n=$((n + 1)); [ "$n" -gt "${EXTRAS_WAIT_MIN:-30}" ] && { say "no valid tl-extras after ${EXTRAS_WAIT_MIN:-30} min"; exit 1; }
  sleep 60
done
if [ ! -d "$TL/experiments/062-h100-protocol-v1/out_full/out/ctx" ]; then
  tar xzf /root/tl-extras.tgz --no-same-owner -C "$TL" || { say "extras unpack FAILED"; exit 1; }
fi
say "extras: md5 ok, unpacked ($(ls "$TL/experiments/062-h100-protocol-v1/out_full/out/ctx" | wc -l) as-run records)"

# ---------------------------------------------------------------- 5. TuringLLM bootstrap (skips 3 + 8 now)
say "TuringLLM bootstrap (log $LOGD/tl.log)"
ROOT=$TL TL_TRACK=1 bash "$PUB/scripts/pod_bootstrap_global.sh" > "$LOGD/tl.log" 2>&1
TL_RC=$?
wait "$PUB_PID"
PUB_RC=$(cut -d= -f2 "$LOGD/public.rc" 2> /dev/null || echo "?")
say "bootstraps: public rc $PUB_RC, TuringLLM rc $TL_RC"
grep -E "OK$|GPUs|download check" "$LOGD/public.log" | tail -5 | sed 's/^/   public | /'
grep -E "READY|main circuits|as-run records|corpus_hash" "$LOGD/tl.log" | sed 's/^/   tl     | /'
OK=1
[ "$PUB_RC" = 0 ] || { say "public bootstrap FAILED: tail $LOGD/public.log"; OK=0; }
[ "$TL_RC" = 0 ] || { say "TuringLLM bootstrap FAILED: tail $LOGD/tl.log"; OK=0; }
grep -q "as-run records: 15046 (expect 15046)" "$LOGD/tl.log" || { say "as-run records != 15046"; OK=0; }
grep -q "== local" "$LOGD/tl.log" || { say "corpus_hash md5 does not match the local file"; OK=0; }

# ---------------------------------------------------------------- 6. GPU health check (all GPUs loaded at once)
say "GPU health check: ${BURN_S:-60} s matmul load on all $NG GPUs"
BURN=$LOGD/burn.py
cat > "$BURN" << 'EOF'
import sys, time, torch
g, secs = int(sys.argv[1]), float(sys.argv[2])
torch.cuda.set_device(g)
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); b = torch.randn_like(a)
torch.cuda.synchronize(); t0 = time.time(); n = 0
while time.time() - t0 < secs:
    for _ in range(20): c = a @ b
    torch.cuda.synchronize(); n += 20
print("gpu %d: %.0f TFLOP/s bf16" % (g, n * 2 * 8192 ** 3 / (time.time() - t0) / 1e12))
EOF
for ((g = 0; g < NG; g++)); do "$PUB/.venv/bin/python" "$BURN" "$g" "${BURN_S:-60}" > "$LOGD/burn.$g.txt" 2>&1 & done
sleep $(( ${BURN_S:-60} / 2 ))
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown \
  --format=csv,noheader > "$LOGD/health_load.csv"
wait
BAD=()
TF_MED=$(cat "$LOGD"/burn.*.txt | grep -o "[0-9]* TFLOP" | cut -d' ' -f1 | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]+0}')
while IFS=, read -r i clk tmp pw swt hwt; do
  clk=${clk//[^0-9]/}; tmp=${tmp//[^0-9]/}
  life=$(nvidia-smi -q -d PERFORMANCE -i "$i" | awk -F: '/SW Thermal Slowdown/ && /us/ {gsub(/[^0-9]/, "", $2); print int($2 / 3.6e9)}' | head -1)
  tf=$(grep -o "[0-9]* TFLOP" "$LOGD/burn.$i.txt" | cut -d' ' -f1)
  flag=""
  # a full bf16 load holds an H100 SXM at its 700 W cap, ~1,365-1,400 MHz (2026-11-07, healthy pod: 679-688 TFLOP/s):
  # flag a GPU that is hot, or clearly below its neighbours (the bad one: 420-1,230 MHz, 86 C, ~200 W)
  { [ "${clk:-0}" -lt 1100 ] || [ "${tmp:-0}" -ge 80 ] || [ "${tf:-0}" -lt "$(( TF_MED * 85 / 100 ))" ]; } \
    && flag="  <== FLAGGED (hot, slow clock or < 85% of the median matmul rate)"
  case "$swt|$hwt" in *"Not Active|"*"Not Active") ;; *Active*) flag="  <== FLAGGED (thermal slowdown active)" ;; esac
  printf '   GPU %s: under load SM %s MHz, %s C, %s, matmul %s TFLOP/s, lifetime SW-thermal-slowdown %s h%s\n' \
    "$i" "$clk" "$tmp" "${pw# }" "${tf:-?}" "${life:-?}" "$flag"
  [ -n "$flag" ] && BAD+=("$i")
done < "$LOGD/health_load.csv"
if [ ${#BAD[@]} -gt 0 ]; then
  say "!!! FLAGGED GPU(s): ${BAD[*]} (slow clock / hot / thermal slowdown). Rent another pod, or leave them out."; OK=0
else
  say "all $NG GPUs healthy under load"
fi

# ---------------------------------------------------------------- 7. next commands (not run here)
FIT=$(seq -s, 2 $((NG - 1)))
echo
if [ "$OK" = 1 ]; then say "READY in $(( ($(date +%s) - T0) / 60 )) min. Next (each needs Daniel's yes):"
else say "NOT READY (see above). The commands, for after a fix:"; fi
cat << EOF
   dry run:  cd $PUB && DRY_RUN=1 TL_TRACK=1 TL_ROOT=$TL FIT_GPUS=$FIT bash experiments/082-public-runs/chain.sh > /root/dryrun.txt 2>&1; echo "rc \$?"; grep -cE "FAILED|MISSING|not found" /root/dryrun.txt
   launch:   tmux new -s chain   then
             cd $PUB && mkdir -p experiments/082-public-runs/logs && TL_TRACK=1 TL_ROOT=$TL FIT_GPUS=$FIT STOP_POD=1 nohup bash experiments/082-public-runs/chain.sh > experiments/082-public-runs/logs/chain.out 2>&1 &
   progress: cd $PUB && while true; do clear; TL_ROOT=$TL bash experiments/082-public-runs/progress.sh < /dev/null; sleep 60; done
EOF
[ "$OK" = 1 ]
