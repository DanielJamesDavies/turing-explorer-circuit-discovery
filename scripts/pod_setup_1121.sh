#!/bin/bash
# ONE-SHOT POD SETUP for the 2026-11-21 run (T7 + arena A + arena D), written after the first attempt that morning.
# Runs every setup step of agent-planning/2026-11-08-post-paris/02-pod-runbook-2026-11-21.md steps 1-3 (+ the gates of
# step 5) unattended, with the morning's fixes built in. It never launches the chain (pod_launch_1121.sh does that).
#
#   PC:   scp -P <port> -i ~/.ssh/id_ed25519 tl-extras.tgz tl-extras.md5 pod_setup_1121.sh pod_launch_1121.sh root@<ip>:/root/
#   pod:  tmux new -s setup     then     bash /root/pod_setup_1121.sh
#         (asks once for the HF read token: typed / pasted, not shown, kept in this process only, never written)
#
# Fixes built in (2026-11-21 morning):
#   - HF_HUB_ENABLE_HF_TRANSFER=0 (the image sets 1 and hf_transfer is not installed: every download failed)
#   - python3.10 from the image's own deadsnakes.list (add-apt-repository on top of it broke apt: duplicate Signed-By)
#   - arena D's Python = 3.12 (built and tested locally on 3.12)
#   - Hugging Face 429 rate limits: weights fetched one file at a time with waits; every download step is retried
#   - the old T7 archive from the failed launch (pub-turingllm-T7-dan159.tgz) is NOT restored (RESTORE_SKIP)
# Steps: 0 diagnosis logs from the volume -> /root/setup/diag.txt; 1 clone + quickstart (background); 2 arena clones;
#   3 python3.10; 4 arena A envs; 5 arena D envs; 6 arena D weights; 7 arena A assets; 8 wait for quickstart (GPU
#   health check); 9 restore + markers; 10 arena D and arena A gates (P0-P1). Summary: /root/setup/summary.txt.
# Safe to re-run: finished steps are skipped or fast (clones, venvs and cached downloads are reused).
set -u
LOG=/root/setup
mkdir -p "$LOG"
T0=$(date +%s)
say() { echo "[setup $(date +%H:%M:%S) +$(( ($(date +%s) - T0) / 60 ))m] $*" | tee -a "$LOG/summary.txt"; }
retry() {   # retry N SLEEP CMD...: run CMD until rc 0, at most N times
  local n=$1 s=$2 i
  shift 2
  for ((i = 1; i <= n; i++)); do
    "$@" && return 0
    [ "$i" -lt "$n" ] && { echo "[retry $i/$n failed, again in ${s} s]"; sleep "$s"; }
  done
  return 1
}
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 PIP_CACHE_DIR=/root/.cache/pip
R=/root/turing
REPO=https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
AA=experiments/107-true-baselines/arena_a
AD=experiments/107-true-baselines/arena_d

if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp 'HF read token (paste, then Enter; not shown): ' HF_TOKEN
  echo
fi
[ -n "${HF_TOKEN:-}" ] || { echo "no HF token given: stop"; exit 1; }
export HF_TOKEN
say "start (log dir $LOG)"
[ -f /root/tl-extras.tgz ] && [ -f /root/tl-extras.md5 ] \
  || say "NOTE: /root/tl-extras.tgz + .md5 not here yet; quickstart waits up to 30 min for them (scp them now)"

# ------------------------------------------------------------------ 0. diagnosis logs of the failed launch
mkdir -p /root/diag
for a in pub-arena-logs.tgz pub-turingllm-T7-dan159.tgz; do
  [ -f "/workspace/results/$a" ] && tar xzf "/workspace/results/$a" --no-same-owner -C /root/diag 2> /dev/null
done
{
  echo "files:"; find /root/diag -type f -name '*.log' | sort
  for f in $(find /root/diag -type f \( -name '*.log' -o -name 'events*' \) -path '*arena_a*' | sort | head -30) \
           $(find /root/diag -type f -name 'dan159_*.log' | sort); do
    echo; echo "===== $f"; tail -40 "$f"
  done
} > "$LOG/diag.txt" 2>&1
say "0 diagnosis: $(grep -c '^=====' "$LOG/diag.txt") logs -> $LOG/diag.txt"

# ------------------------------------------------------------------ 1. clone + quickstart (background)
[ -d "$R/.git" ] || git clone -q --branch multi-device "$REPO" "$R" || { say "1 clone FAILED"; exit 1; }
cd "$R" || exit 1
bash "$R/scripts/pod_quickstart.sh" > "$LOG/quickstart.log" 2>&1 &
QS=$!
say "1 quickstart running in the background (log $LOG/quickstart.log, ~40 min)"

# ------------------------------------------------------------------ 2. arena clones (pinned)
clone() {   # clone URL DIR COMMIT
  [ -d "$2/.git" ] || git clone -q "$1" "$2" || return 1
  git -C "$2" -c advice.detachedHead=false checkout -q "$3" && [ "$(git -C "$2" rev-parse HEAD)" = "$3" ]
}
ok=OK
clone https://github.com/OpenMOSS/Language-Model-SAEs.git third_party/Language-Model-SAEs 67bfb0b9f835f772f3af2aa176745c6b6135373a || ok=FAILED
clone https://github.com/jacobdunefsky/transcoder_circuits.git third_party/transcoder_circuits 8721ff06b64889e8cc50bb1c5e869c7f72b8303d || ok=FAILED
clone https://github.com/NainaniJatinZ/ScalableSAECircuits.git third_party/ScalableSAECircuits f54a4efce908432b8771ea671a248ee30ec2b3af || ok=FAILED
clone https://github.com/decoderesearch/circuit-tracer.git third_party/circuit-tracer 255d989e93be1299d53d67b59e9fd76a0e274b14 || ok=FAILED
say "2 arena clones: $ok"

# ------------------------------------------------------------------ 3. python3.10 (arena A's pinned envs)
if ! command -v python3.10 > /dev/null; then
  {
    if ls /etc/apt/sources.list.d/ | grep -q deadsnakes; then
      apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv
    else
      apt-get update -qq && apt-get install -y -qq software-properties-common \
        && add-apt-repository -y ppa:deadsnakes/ppa && apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv
    fi
  } > "$LOG/python310.log" 2>&1
fi
say "3 python: $(python3.10 --version 2>&1) / $(python3.12 --version 2>&1)"

# ------------------------------------------------------------------ 4. arena A envs (ge107 torch 2.3.1, tc107 torch 2.2.0)
retry 3 30 env ROOT="$R" TORCH_INDEX=cu121 PY310=python3.10 bash "$AA/setup_env.sh" both > "$LOG/arena_a_env.log" 2>&1
say "4 arena A envs: $(grep -c 'env ready' "$LOG/arena_a_env.log")/2 ready ($(grep -o 'ge107 ok: torch [^ |]*' "$LOG/arena_a_env.log" | tail -1); $(grep -o 'tc107 ok: torch [^ |]*' "$LOG/arena_a_env.log" | tail -1))"

# ------------------------------------------------------------------ 5. arena D envs (no weights yet)
retry 3 30 env ROOT="$R" PY=python3.12 bash "$AD/setup_env.sh" > "$LOG/arena_d_env.log" 2>&1
say "5 arena D envs: $(grep -c 'arena D envs ready' "$LOG/arena_d_env.log") ready ($(grep -o 'circuit-tracer [^ ]* commit [0-9a-f]\{7\}' "$LOG/arena_d_env.log" | tail -1))"

# ------------------------------------------------------------------ 6. arena D weights, one file at a time (HF 429s)
/root/venvs/ct107/bin/python - > "$LOG/arena_d_weights.log" 2>&1 << 'EOF'
import time
from huggingface_hub import snapshot_download, hf_hub_download
def retry(f, *a, **k):
    for i in range(40):
        try:
            return f(*a, **k)
        except Exception as e:
            if "429" not in str(e) and "Too Many Requests" not in str(e):
                raise
            w = min(10 * (i + 1), 120)
            print(f"429, waiting {w} s (try {i + 1})", flush=True)
            time.sleep(w)
    raise SystemExit("still 429 after 40 tries")
retry(snapshot_download, "google/gemma-2-2b", revision="c5ebcd40d208330abc697524c919956e692655cf", allow_patterns=["*.safetensors", "*.json", "tokenizer.model"], max_workers=1)
print("google ok", flush=True)
retry(snapshot_download, "unsloth/gemma-2-2b", revision="25319945f7fd83b8b903e12081777b7eef2ba993", max_workers=1)
print("unsloth ok", flush=True)
retry(snapshot_download, "mwhanna/gemma-scope-transcoders", revision="3412e19be8f72d47f8683cb92028a46513222b2b", allow_patterns=["config.yaml", "layer_*.safetensors"], max_workers=1)
print("PLT ok", flush=True)
for p in ["layer_0/width_16k/average_l0_105", "layer_6/width_16k/average_l0_70", "layer_12/width_16k/average_l0_82", "layer_18/width_16k/average_l0_74", "layer_24/width_16k/average_l0_73"]:
    retry(hf_hub_download, "google/gemma-scope-2b-pt-res", f"{p}/params.npz", revision="fd571b47c1c64851e9b1989792367b9babb4af63")
print("ALL WEIGHTS CACHED", flush=True)
EOF
# the script's own FETCH pass: everything cached now; it adds the Dallas graph JSON
retry 10 60 env ROOT="$R" PY=python3.12 FETCH=1 bash "$AD/setup_env.sh" >> "$LOG/arena_d_env.log" 2>&1
dallas=$([ -s "$AD/out/fidelity/gemma-fact-dallas-austin.json" ] && echo ok || echo MISSING)
say "6 arena D weights: $(grep -q 'ALL WEIGHTS CACHED' "$LOG/arena_d_weights.log" && echo 'all cached' || echo 'FAILED (see arena_d_weights.log)'); Dallas JSON $dallas"

# ------------------------------------------------------------------ 7. arena A assets (needs the public .venv)
until [ -f /root/quickstart/public.rc ] || ! kill -0 "$QS" 2> /dev/null; do sleep 20; done
pa=FAILED; po=FAILED
retry 10 60 env ROOT="$R" PY=./.venv/bin/python bash "$AA/fetch_assets.sh" pod > "$LOG/arena_a_assets.log" 2>&1 && pa=ok
retry 10 60 env ROOT="$R" PY=./.venv/bin/python bash "$AA/fetch_assets.sh" owt >> "$LOG/arena_a_assets.log" 2>&1 && po=ok
say "7 arena A assets: GPT-2 + transcoders + Kissane $pa; OWT shard $po"

# ------------------------------------------------------------------ 8. quickstart result + GPU health check
wait "$QS"
QRC=$?
grep -E 'bootstraps:|  GPU [0-9]+:|healthy|FLAGGED|READY|NOT READY|FAILED|INCOMPLETE' "$LOG/quickstart.log" | sed 's/^/    /' | tee -a "$LOG/summary.txt"
say "8 quickstart rc $QRC"
if [ "$QRC" != 0 ]; then say "STOP: quickstart not ready (see $LOG/quickstart.log); nothing restored"; exit 1; fi
# every .sh executable in both trees (git stores them 100644; 083/run_pod.sh ran ./launch.sh directly -> T7 did nothing
# in the morning's launch). Harmless where scripts are already called through bash.
find "$R" /root/turing-tl -name '*.sh' -not -path '*/third_party/*' -not -path '*/.venv/*' -exec chmod +x {} + 2> /dev/null
say "8b .sh files executable: launch.sh $([ -x /root/turing-tl/experiments/062-h100-protocol-v1/launch.sh ] && echo yes || echo NO)"

# ------------------------------------------------------------------ 9. restore + markers
ST=$R/experiments/082-public-runs/logs/state
if [ ! -f "$LOG/restore.done" ]; then
  RESTORE_SKIP='S10-stores|T7-dan159' bash "$R/scripts/pod_restore.sh" > "$LOG/restore.log" 2>&1 && touch "$LOG/restore.done"
fi
grep -E 'RESTORED|FAIL|md5' "$LOG/restore.log" | tail -3 | sed 's/^/    /' | tee -a "$LOG/summary.txt"
for t in T3 T4 T5 T6 T8 T10 T11 T12; do
  [ -f "$ST/turingllm.$t.done" ] || date '+%F %T restored (run 2026-11-07 finished)' > "$ST/turingllm.$t.done"
done
[ -f "$ST/turingllm.T7.done" ] && mv "$ST/turingllm.T7.done" "$ST/turingllm.T7.done.skipped-$(date +%H%M%S)"
say "9 restore: $([ -f "$LOG/restore.done" ] && echo ok || echo FAILED); TuringLLM markers: $(cd "$ST" && ls turingllm.*.done 2> /dev/null | sed 's/turingllm\.//; s/\.done//' | sort -V | tr '\n' ' ')(T7 must NOT be listed)"

# ------------------------------------------------------------------ 10. gates (P0-P1 of both arenas, GPU 0, seconds-minutes)
DW_STOP_AFTER=P1 DW_GPUS=0 bash "$AD/pod_stage_d.sh" > "$LOG/gate_arena_d.log" 2>&1
GD=$?
STOP_AT=P1 AA_GPUS=0 AA_MAX_H=0 bash "$AA/run_pod.sh" > "$LOG/gate_arena_a.log" 2>&1
GA=$?
say "10 gates: arena D rc $GD, arena A rc $GA (0 = ok; 2 = commands to approve, see the gate logs)"
[ "$GD" = 0 ] || tail -15 "$LOG/gate_arena_d.log" | sed 's/^/    D| /' | tee -a "$LOG/summary.txt"
if [ "$GA" != 0 ]; then
  tail -15 "$LOG/gate_arena_a.log" | sed 's/^/    A| /' | tee -a "$LOG/summary.txt"
  for f in "$AA"/out/pod/logs/P0/*.log "$AA"/out/pod/logs/P1/*.log; do
    [ -f "$f" ] && { echo "    A| --- $f"; tail -12 "$f" | sed 's/^/    A| /'; } | tee -a "$LOG/summary.txt"
  done
fi

# ------------------------------------------------------------------ 11. MPS check (the morning's arena A rc 1 came 31 s in,
# around its GPU checks, under the chain's MPS daemon): the baseline envs' old torch (2.3.1 / 2.2.0 cu121) + the main
# .venv, each doing real GPU work under MPS. The daemon is stopped again afterwards (the chain starts its own).
mps_test() {   # mps_test LABEL SOURCE-CMD PYTHON
  bash -c "$2 && CUDA_VISIBLE_DEVICES=0 timeout 120 $3 -I -B -c 'import torch; x = torch.randn(512, 512, device=\"cuda\"); print(torch.__version__, float((x @ x).sum()))'" \
    >> "$LOG/mps_check.log" 2>&1 && echo "$1 ok" || echo "$1 FAILED"
}
if command -v nvidia-cuda-mps-control > /dev/null; then
  nvidia-cuda-mps-control -d >> "$LOG/mps_check.log" 2>&1
  m1=$(mps_test ge107 "source /root/venvs/ge107/bin/ge_env.sh" python)
  m2=$(mps_test tc107 "source /root/venvs/tc107/bin/tc_env.sh" python)
  m3=$(mps_test main-venv "true" "$R/.venv/bin/python")
  echo quit | nvidia-cuda-mps-control >> "$LOG/mps_check.log" 2>&1
  say "11 under MPS: $m1, $m2, $m3 (any FAILED: launch with MPS=0; log $LOG/mps_check.log)"
else
  say "11 MPS check skipped (no nvidia-cuda-mps-control)"
fi

say "SETUP FINISHED. Send Claude: cat $LOG/summary.txt  and  cat $LOG/diag.txt. Do NOT launch before the T7 / arena A fixes."
