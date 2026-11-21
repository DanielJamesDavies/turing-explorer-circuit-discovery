#!/bin/bash
# LAUNCH for the 2026-11-21 run: the runbook's step-4 environment, automatic pre-flight and dry-run checks, then the
# chain. It launches ONLY if every check passes. STOP_POD defaults to 0 here (the morning's auto-stop after an arena
# failure wiped the pod's setup): stop the pod by hand when arena D has packed.
#
#   pod (inside tmux):  bash /root/pod_launch_1121.sh            # checks, then launches
#                       CHECK_ONLY=1 bash /root/pod_launch_1121.sh   # checks only
set -u
cd /root/turing || exit 1
export HF_HOME=/root/.cache/huggingface HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 PIP_CACHE_DIR=/root/.cache/pip \
  TL_TRACK=1 TL_ROOT=/root/turing-tl TL_STOP_AFTER=T7 TL_DAN159_STAGES="ref attr fitted exact" TL_083_P=4 \
  STOP_AFTER=S0 FIT_GPUS=1,2,3,4,5 DATA_GPU_pythia_70m=1 DATA_GPU_gpt2_small=1 \
  ARENA_A=1 ARENA_A_GPUS=0,1,2,3,4,5 ARENA_A_MAX_H=16 AA_DUNEFSKY_EXT=1 \
  ARENA_D=1 ARENA_D_GPUS=0,1,2,3,4,5
export STOP_POD=${STOP_POD:-0}
unset AA_LAMBDAS AA_LAMS

fail=0
chk() {   # chk LABEL TEST
  if eval "$2"; then echo "  PASS  $1"; else echo "  FAIL  $1"; fail=1; fi
}
ST=experiments/082-public-runs/logs/state
TLW=/root/turing-tl/experiments/082-public-runs/out_turingllm
LIST=$TLW/lists/targets_tl8k_sweep190.txt
echo "pre-flight:"
chk "no turingllm.T7.done marker" "[ ! -f $ST/turingllm.T7.done ]"
chk "T7 target list has >= 150 targets ($LIST)" "[ \"\$(grep -c . $LIST 2> /dev/null || echo 0)\" -ge 150 ]"
chk "T7 can start its shards (launch.sh via bash, or executable)" \
  "grep -q 'bash ./experiments/062-h100-protocol-v1/launch.sh' /root/turing-tl/experiments/083-v1-comparison/run_pod.sh || [ -x /root/turing-tl/experiments/062-h100-protocol-v1/launch.sh ]"
chk "arena A gate passed (P1.done)" "[ -f experiments/107-true-baselines/arena_a/out/pod/state/P1.done ]"
chk "arena D gate passed (P1.done)" "[ -f experiments/107-true-baselines/arena_d/out/pod/state/P1.done ]"

D=/root/dryrun.txt
DRY_RUN=1 bash experiments/082-public-runs/chain.sh > "$D" 2>&1
echo "dry run ($D):"
chk "T7 starts" "grep -q 'turingllm.T7: START' $D"
chk "all 5 T7 stages on GPUs 1,2,3,4,5" "[ \"\$(grep -c 'dan159 .* on GPUs 1,2,3,4,5,' $D)\" -ge 5 ]"
chk "S10 does not start" "! grep -q 'S10: START' $D"
chk "arena A: WCM v4, lambdas 2.5e-4,1e-3,4e-3,1.6e-2" "grep -q 'WCM v4 .*lambdas 2.5e-4,1e-3,4e-3,1.6e-2' $D"
chk "arena D: WCM v3" "grep -q 'arena D pod stage .*WCM v3' $D"
echo "STOP_POD=$STOP_POD"

if [ "$fail" != 0 ]; then echo "NOT LAUNCHED: a check failed (send Claude this output)"; exit 1; fi
if [ "${CHECK_ONLY:-0}" = 1 ]; then echo "all checks passed (CHECK_ONLY: not launched)"; exit 0; fi
echo "all checks passed: launching in 10 s (Ctrl-C to cancel)"
sleep 10
bash experiments/082-public-runs/chain.sh 2>&1 | tee -a /root/chain-2026-11-21.log
