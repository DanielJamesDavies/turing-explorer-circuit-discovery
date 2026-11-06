#!/bin/bash
# Arena D pod stage (IMPLEMENTATION_PLAN item D8; README §9): Caples et al.'s masks (D2) and attribution graphs (D1) on
# Gemma-2-2B, each with WCM on its own substrate, on the 8xH100 pod AFTER the main 082 chain and arena A. The chain calls
# it with ARENA_D=1 (082/chain.sh arena_d_stage, D9); it also runs on its own. Run from anywhere; it cds to the repo root.
# PROPOSAL ONLY until Daniel approves it on the pod ([[remote-pod-discuss-first]]). Scope (Daniel, 2026-11-05): tasks only.
#
#   DRY_RUN=1 bash experiments/107-true-baselines/arena_d/pod_stage_d.sh     # print every command, run nothing
#   DW_STOP_AFTER=P2 bash .../pod_stage_d.sh    # assets, envs and the gate only (prints the fetch / venv commands for
#                                               # approval when anything is missing, exit 2)
#   nohup bash .../pod_stage_d.sh > <DW_WORK>/pod_stage_d.out 2>&1 &
#
# STAGES (resumable: STATE/<P>.done skips a finished stage; inside a stage every step / job skips what is already written,
# and every runner resumes per unit: run_caples.sh per grid, wcm_gemma.py per (task, lambda, means, mode), ag_tasks.py per
# prompt, score_d.py per (circuit, substrate, means))
#   P0 assets  checks only: the pinned clones third_party/ScalableSAECircuits @ f54a4ef (their code + the SVA data) and
#              third_party/circuit-tracer @ 255d989 (their tests), the HF snapshots (google/gemma-2-2b fp32 @ c5ebcd40,
#              unsloth/gemma-2-2b bf16 @ 25319945, mwhanna PLTs @ 3412e19b, Gemma Scope res @ fd571b47 x 5), the
#              Neuronpedia Dallas JSON, HF_TOKEN present in this shell (checked, never printed). Missing -> the clone /
#              fetch commands are PRINTED for Daniel to approve and run; exit 2. This script never downloads.
#   P1 envs    ct107 / caples107 must exist (setup_env.sh: pip, PRINTED for approval when missing, exit 2). Gate: pip check
#              in both, CUDA visible in .venv, ct107 and caples107.
#   P2 gate    on DW_GATE_GPUS (default the first two of DW_GPUS, "0,1": the D2 half on 0 and the D1 half on 1 in parallel;
#              one GPU = the halves in turn; 2026-11-06, was GPU 0 alone), before
#              any fan-out. D2 half: weights check (CPU, informative: google fp32 -> bf16 vs unsloth, tensor by tensor),
#              caples_means.py = THEIR get_sae_means / get_sae_error_means RECOMPUTED on the pod's fp32 google weights
#              (the D2 C fill / eps_bar), cross-check vs the local bf16 file (informative), D6 identity (gsres, fp32,
#              bitwise vs raw HF), README §4.3 test 4 on the pod file, Caples grid A + caples_fidelity A (must PASS).
#              D1 half: ag_tasks.py prompts (CPU), D6 identity (PLT), ag_fidelity.py F1 F3 F4 F5 F6 in bf16 (unsloth,
#              informative), bf16 (google, informative) and fp32 (google, --strict, must pass), their own
#              test_gemma_2_2b (copy of the clone's tests, against the installed package; must pass).
#              A half that fails turns off that half's production jobs (DW_GATE_MODE=warn runs them anyway, labelled).
#   P3 jobs    one job per GPU from a shared queue (flock; a free GPU takes the next job), D2 first (plan §3 cut lines):
#              Caples dense, D2-WCM rc (caples mode per_position = the like-for-like row; native per_position; native
#              pooled), Caples B, C; then D1: D1-WCM (PLT, 4 SVA tasks x {pooled, per_position}), AG (4 tasks x 40 held
#              prompts, fp32 google, as published). Behavioural WCM = arena C's adopted setting (wcm_gemma.py run,
#              unchanged; 081 WCM_VERSION=v2 pinned, as arena C's runs), DW_LAMS = arena C's 9-lambda grid, 400 steps.
#   P4 score   caples_fidelity.py A,B,C (CPU), then the queue again: score_d.py all (ours: Z / A / C / necessity (a)
#              errors live + (b) all errors clean, both substrates; theirs: their SAEMasks on their eval batches, binary,
#              alpha dropped, labelled) per Caples grid and per D2-WCM configuration.
#   P5 pack    pod_summary_d.py (SUMMARY.md + summary.json), results tgz + md5 in CKPT, copied to RESULTS and md5-checked
#              there. P5 ALSO RUNS when an earlier stage failed (partial results are packed before the chain stops the pod).
#              Then, AFTER the results are verified, the AG graphs (when any exist) in their own TAG-arena-d-graphs.tgz +
#              md5 (pigz when present), copied and md5-checked the same way; the results tgz leaves them out. The hard
#              stop spares P5's own commands.
#
# ENV (defaults in brackets)
#   DRY_RUN [0]         1 = print every command; nothing is run, written or created
#   DW_STOP_AFTER [P5]  the last stage to run      DW_FROM [P0] stages before it are taken as done (no marker written)
#   STOP_EPOCH          no new step / job / unit once date +%s >= it (python runners: STOP_EPOCH env; ag_tasks.py: STOP_AT
#                       HH:MM derived from it). DW_HARD_GRACE [3600] s after it, every running job's process group is
#                       killed (a hung job cannot keep the pod up), then P5 packs
#   DW_WORK [107/arena_d/out/pod]  every output; LOGS = DW_WORK/logs, STATE = DW_WORK/state, cmds.tsv (one row per
#                       command: log, start, end, rc, script; the D10 re-cost), steps.tsv, gpu_mem.csv
#   DW_GPUS [0,...,7]   DW_GATE_GPUS [first two of DW_GPUS]   DW_GATE_MODE [strict | warn]
#   DW_D1 [1]           0 = no D1 (cut line 1: D2 only)   DW_D2_NATIVE [1]  0 = D2-WCM Caples mode only (cut line 2)
#   DW_LAMS [3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2] (arena C's grid)   DW_STEPS [400]
#   DW_TASKS [simple,within_rc,rc,nounpp] (D1)   DW_AG_LIMIT [0 = all 40 held prompts]   DW_AG_BATCH [64]
#   AG_SAVE_GRAPHS [1]  1 = the ag_* jobs run ag_tasks_graphs.py (= ag_tasks.py unchanged + every attribution graph saved
#                       as AGO/<task>/held/p<id>.graph.pt, float16 adjacency, ~0.1 GB each, ~16 GB for 4 x 40; per-prompt
#                       outputs identical to ag_tasks.py, checked on the tiny CPU model); 0 = ag_tasks.py, no graphs.
#                       AG_GRAPH_DTYPE [float16] (float32 = exact, ~2x). A prompt whose JSON exists is skipped on a rerun,
#                       so a graph is saved only for prompts run with AG_SAVE_GRAPHS=1
#   DW_CAPLES_GRIDS ["B C dense"] (A runs in the gate; D = "report, never match": add it here to run it)   DENSE_N [8]
#   DW_ID_DTYPE [float32] (D6 on the pod)   DW_LOCAL_MEANS [107/arena_d/out/caples_means/rc_gap6_bf16_SMOKE.pt: the local
#                       bf16 file, copied to the pod by hand (out/ is not in git); missing = cross-check skipped]
#   DW_HF_OFFLINE [0]   0 = the libraries may reach the hub (they read HF_TOKEN from the environment themselves; nothing
#                       here prints it); 1 = HF_HUB_OFFLINE=1 (needs refs/main in the cache for every by-name load)
#   CT_ENV [~/venvs/ct107] CAPLES_ENV [~/venvs/caples107] (score_d.py reaches caples107 at ~/venvs/caples107 itself)
#   PY [./.venv/bin/python] PY311 [python3.11]  HF_HUB_CACHE [HF_HOME/hub, else ~/.cache/huggingface/hub]
#   CKPT [/root/ckpt] RESULTS [/workspace/results] TAG [pub]   (P5: TAG-arena-d-results.tgz + TAG-arena-d.md5, and
#                       TAG-arena-d-graphs.tgz + TAG-arena-d-graphs.md5)
# Exit: 0 done (or stopped at DW_STOP_AFTER), 1 a stage failed, 2 waiting for Daniel (downloads / env builds to approve).
set -uo pipefail
shopt -s nullglob

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT" || exit 1
AD=experiments/107-true-baselines/arena_d
DRY_RUN=${DRY_RUN:-0}
STOP_AFTER=${DW_STOP_AFTER:-P5}
FROM=${DW_FROM:-P0}
STOP_EPOCH=${STOP_EPOCH:-99999999999}
HARD_GRACE=${DW_HARD_GRACE:-3600}
PY=${PY:-./.venv/bin/python}
PY311=${PY311:-python3.11}
WORK=${DW_WORK:-$AD/out/pod}
LOGS=$WORK/logs
STATE=$WORK/state
GPUS=${DW_GPUS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GL <<< "$GPUS"
NG=${#GL[@]}
GATE_GPUS=${DW_GATE_GPUS:-${GL[0]}${GL[1]:+,${GL[1]}}}   # default: the first two of DW_GPUS (one when only one)
IFS=',' read -r -a GG <<< "$GATE_GPUS"
GATE_MODE=${DW_GATE_MODE:-strict}
D1=${DW_D1:-1}
D2_NATIVE=${DW_D2_NATIVE:-1}
LAMS=${DW_LAMS:-3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2}
N_LAMS=$(echo "$LAMS" | tr ',' '\n' | grep -c .)
STEPS=${DW_STEPS:-400}
TASKS=${DW_TASKS:-simple,within_rc,rc,nounpp}
AG_LIMIT=${DW_AG_LIMIT:-0}
AG_BATCH=${DW_AG_BATCH:-64}
AG_K=40; [ "$AG_LIMIT" -gt 0 ] 2> /dev/null && AG_K=$AG_LIMIT
SAVE_GRAPHS=${AG_SAVE_GRAPHS:-1}
if [ "$SAVE_GRAPHS" = 1 ]; then AG_PY=ag_tasks_graphs.py; else AG_PY=ag_tasks.py; fi   # graphs: same per-prompt outputs
GZ="-z"; command -v pigz > /dev/null 2>&1 && GZ="-I pigz"   # the graphs archive (~16 GB): parallel gzip when there
CAPLES_GRIDS=${DW_CAPLES_GRIDS:-"B C dense"}
DENSE_N=${DENSE_N:-8}
ID_DTYPE=${DW_ID_DTYPE:-float32}
LOCAL_MEANS=${DW_LOCAL_MEANS:-$AD/out/caples_means/rc_gap6_bf16_SMOKE.pt}
HF_OFF=${DW_HF_OFFLINE:-0}
CT_ENV=${CT_ENV:-$HOME/venvs/ct107}
CAPLES_ENV=${CAPLES_ENV:-$HOME/venvs/caples107}
CT=$CT_ENV/bin/python
CAP=$CAPLES_ENV/bin/python
HF_HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
CKPT=${CKPT:-/root/ckpt}
RESULTS=${RESULTS:-/workspace/results}
TAG=${TAG:-pub}

MEANS=$WORK/caples_means/rc_gap6.pt          # their functions on the pod's fp32 google weights (the D2 fill)
RUNC=$WORK/caples_runs                       # run_caples.sh RUN (one clone copy per grid)
FID=$WORK/fidelity
CHK=$WORK/checks
AGO=$WORK/ag_tasks
W2=$WORK/wcm_d2
W1=$WORK/wcm_d1
SD=$WORK/score_d
SAVE=masks/sva/rc_train_saegap6_mean_error   # their save folder inside a grid's copy (run_caples.sh)

# pinned assets (README §1, setup_env.sh)
CAPLES_PIN=f54a4efce908432b8771ea671a248ee30ec2b3af
CT_PIN=255d989e93be1299d53d67b59e9fd76a0e274b14
G_SNAP=$HF_HUB/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf
U_SNAP=$HF_HUB/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993
PLT_SNAP=$HF_HUB/models--mwhanna--gemma-scope-transcoders/snapshots/3412e19be8f72d47f8683cb92028a46513222b2b
RES_SNAP=$HF_HUB/models--google--gemma-scope-2b-pt-res/snapshots/fd571b47c1c64851e9b1989792367b9babb4af63
RES_FILES="layer_0/width_16k/average_l0_105 layer_6/width_16k/average_l0_70 layer_12/width_16k/average_l0_82 layer_18/width_16k/average_l0_74 layer_24/width_16k/average_l0_73"
DALLAS=$AD/out/fidelity/gemma-fact-dallas-austin.json

# ag_tasks.py reads STOP_AT=HH:MM only: derived from STOP_EPOCH when that is under a day away (else none)
STOP_HHMM=""
if [ "$STOP_EPOCH" != 99999999999 ] && [ $((STOP_EPOCH - $(date +%s))) -lt 86000 ]; then
  STOP_HHMM=$(date -d "@$STOP_EPOCH" +%H:%M)
fi
# the environment of every python step. PYTHONPATH is dropped (the chain exports src: it must not shadow their imports);
# WCM_VERSION=v2 = 081 task_common's default, which arena C's adopted behavioural setting ran under (the chain exports
# v4, which 081 refuses). HF_TOKEN is inherited from the shell untouched and never appears in a command or a log.
BASE="env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WCM_VERSION=v2 GEMMA_HF_ID=google/gemma-2-2b HF_HUB_OFFLINE=$HF_OFF TORCH_THREADS=8 OMP_NUM_THREADS=8 \
STOP_EPOCH=$STOP_EPOCH STOP_AT=$STOP_HHMM"

# ------------------------------------------------------------------------------------------------- helpers
say() {
  local line="[arena_d $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$LOGS"; echo "$line" >> "$LOGS/events.log"; }
}
q() { printf '%q' "$1"; }
snum() { local s=${1#P}; echo "${s%%[!0-9]*}"; }
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
mark() { [ "$DRY_RUN" = 1 ] || { mkdir -p "$STATE"; date '+%F %T' > "$STATE/$1.$2"; }; }
nlines() { if [ -f "$1" ]; then grep -c . "$1"; else echo 0; fi; }
count() { echo $#; }
clone_ok() { [ -d "$1/.git" ] && [ "$(git -C "$1" rev-parse HEAD 2> /dev/null)" = "$2" ]; }

run_cmd() {   # run_cmd LOG CMD [NAME]: one command string through bash -c in its own process group (setsid: the hard
  #             stop kills the group), appended to LOG; one row per command in cmds.tsv. DRY_RUN prints it
  local log=$1 cmd=$2 name=${3:-cmd$$} t0 rc pid
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $cmd   >> $log"; return 0; fi
  mkdir -p "$(dirname "$log")" "$STATE/pids"
  echo "[$(date '+%F %T')] \$ $cmd" >> "$log"
  t0=$(date +%s)
  setsid -w bash -c "$cmd" >> "$log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$STATE/pids/$name"
  wait "$pid"
  rc=$?
  rm -f "$STATE/pids/$name"
  [ "$rc" = 0 ] || echo "[$(date '+%F %T')] exit $rc" >> "$log"
  printf '%s\t%s\t%s\t%s\t%s\n' "${log#"$LOGS"/}" "$t0" "$(date +%s)" "$rc" \
    "$({ echo "$cmd" | grep -oE '[A-Za-z0-9_]+\.(py|sh)[^>|&]*' || echo "${cmd:0:80}"; } | tail -n 1 | cut -c1-200)" \
    >> "$WORK/cmds.tsv"
  return "$rc"
}

passed() {   # passed N LOG: pytest's last summary line in LOG reads "N passed ..." with nothing failed / errored / skipped
  local last
  last=$(grep -E " in [0-9.]+s" "$2" 2> /dev/null | tail -1 | sed -E 's/^=+ //')
  echo "$last" | grep -qE "^$1 passed" && ! echo "$last" | grep -qE "failed|error|skipped"
}

step() {   # step LOG NAME DONE_TEST CMD: skip when DONE_TEST holds (resume); ok = exit 0 AND DONE_TEST afterwards
  local log=$1 name=$2 done_test=$3 cmd=$4
  if [ "$DRY_RUN" != 1 ] && eval "$done_test"; then say "  $name: done earlier, skipped"; return 0; fi
  if [ "$DRY_RUN" != 1 ] && late; then say "  $name: STOP_EPOCH reached, not started"; return 1; fi
  [ "$DRY_RUN" = 1 ] && echo "    # $name"
  run_cmd "$log" "$cmd" "$name" || { say "  $name: FAILED (exit; $log)"; return 1; }
  [ "$DRY_RUN" = 1 ] && return 0
  eval "$done_test" || { say "  $name: FAILED (exit 0 but its output check failed; $log)"; return 1; }
  say "  $name: ok"
}

stage() {   # stage P<n> TITLE FUNC: DW_STOP_AFTER / DW_FROM gating, markers, steps.tsv. rc 3 = after DW_STOP_AFTER
  local num=$1 title=$2 fn=$3 rc t0
  if [ "$(snum "$num")" -gt "$(snum "$STOP_AFTER")" ]; then say "$num $title: after DW_STOP_AFTER=$STOP_AFTER; stopping here"; return 3; fi
  if [ "$(snum "$num")" -lt "$(snum "$FROM")" ]; then say "$num $title: before DW_FROM=$FROM, taken as done"; return 0; fi
  if [ -f "$STATE/$num.done" ]; then
    say "$num $title: done earlier ($STATE/$num.done)"
    [ "$DRY_RUN" = 1 ] || return 0
    say "  (dry run: its commands are printed anyway)"
  fi
  if [ "$DRY_RUN" != 1 ] && late && [ "$num" != P5 ]; then say "$num $title: STOP_EPOCH reached; not started"; return 1; fi
  say "$num $title: START"
  mark "$num" start
  t0=$(date +%s)
  "$fn"
  rc=$?
  [ "$DRY_RUN" = 1 ] || printf '%s\t%s\t%s\t%s\n' "$num" "$t0" "$(date +%s)" "$rc" >> "$WORK/steps.tsv"
  case $rc in
    0) mark "$num" done; say "$num $title: DONE ($(( $(date +%s) - t0 )) s)" ;;
    2) say "$num $title: WAITING FOR APPROVAL (run the commands above, then rerun)" ;;
    *) say "$num $title: FAILED (rc $rc; logs in $LOGS/$num)" ;;
  esac
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- job queue
QN=(); QC=(); QD=()        # job names, commands (@GPU@ = the worker's GPU), done tests
qreset() { QN=(); QC=(); QD=(); }
qadd() { QN+=("$1"); QC+=("$2"); QD+=("$3"); }

q_worker() {   # q_worker STAGE GPU: take the next job index under flock until the queue is empty
  local st=$1 g=$2 i name cmd fails=0
  local L=$LOGS/$st
  while :; do
    i=$( { flock 9; local k; k=$(cat "$STATE/$st.qidx"); echo $((k + 1)) > "$STATE/$st.qidx"; echo "$k"; } 9> "$STATE/$st.qlock")
    [ "$i" -ge "${#QN[@]}" ] && break
    name=${QN[$i]}
    if eval "${QD[$i]}"; then echo "[$(date '+%F %T')] $name: done earlier, skipped" >> "$L/gpu$g.log"; continue; fi
    if late; then echo "[$(date '+%F %T')] $name: STOP_EPOCH reached, not started" >> "$L/gpu$g.log"; fails=$((fails + 1)); continue; fi
    echo "[$(date '+%F %T')] $name: START on GPU $g" >> "$L/gpu$g.log"
    cmd=${QC[$i]//@GPU@/$g}
    if run_cmd "$L/$name.log" "$cmd" "$name" && eval "${QD[$i]}"; then
      echo "[$(date '+%F %T')] $name: DONE" >> "$L/gpu$g.log"
    else
      echo "[$(date '+%F %T')] $name: FAILED or incomplete ($L/$name.log)" >> "$L/gpu$g.log"
      say "  job $name FAILED or incomplete on GPU $g ($L/$name.log)"
      fails=$((fails + 1))
    fi
  done
  return "$fails"
}

q_run() {   # q_run STAGE: the queue over every GPU of DW_GPUS (a dry run prints it in queue order); rc = failed jobs
  local st=$1 g j pids=() fails=0
  if [ "$DRY_RUN" = 1 ]; then
    for j in "${!QN[@]}"; do
      echo "  [job $((j + 1))/${#QN[@]} ${QN[$j]}: next free GPU of $GPUS]"
      run_cmd "$LOGS/$st/${QN[$j]}.log" "${QC[$j]//@GPU@/<gpu>}"
    done
    return 0
  fi
  mkdir -p "$LOGS/$st" "$STATE"
  echo 0 > "$STATE/$st.qidx"
  say "  $st: ${#QN[@]} jobs on GPUs $GPUS"
  for g in "${GL[@]}"; do q_worker "$st" "$g" & pids+=($!); done
  for j in "${pids[@]}"; do wait "$j" || fails=$((fails + $?)); done
  [ "$fails" = 0 ] || say "  $st: $fails job(s) failed or incomplete (a rerun resumes them)"
  return "$fails"
}

# ------------------------------------------------------------------------------------------------- P0 assets
assets_missing() {   # one line per missing item (read-only)
  clone_ok third_party/ScalableSAECircuits "$CAPLES_PIN" || echo "clone third_party/ScalableSAECircuits @ ${CAPLES_PIN:0:7}"
  [ "$D1" = 1 ] && { clone_ok third_party/circuit-tracer "$CT_PIN" || echo "clone third_party/circuit-tracer @ ${CT_PIN:0:7}"; }
  [ "$(count "$G_SNAP"/model-0000?-of-00003.safetensors)" = 3 ] && [ -f "$G_SNAP/tokenizer.json" ] \
    || echo "google/gemma-2-2b fp32 @ c5ebcd40"
  [ -f "$U_SNAP/model.safetensors" ] || echo "unsloth/gemma-2-2b bf16 @ 25319945"
  local f n=0
  for f in $RES_FILES; do [ -f "$RES_SNAP/$f/params.npz" ] && n=$((n + 1)); done
  [ "$n" = 5 ] || echo "Gemma Scope res x5 @ fd571b47 ($n / 5)"
  if [ "$D1" = 1 ]; then
    [ "$(count "$PLT_SNAP"/layer_*.safetensors)" = 26 ] || echo "PLT transcoders x26 @ 3412e19b"
    [ -s "$DALLAS" ] || echo "Neuronpedia Dallas graph JSON"
  fi
  if [ "$HF_OFF" != 1 ] && [ -z "${HF_TOKEN:-}" ]; then echo "HF_TOKEN in this shell's environment (gated google/gemma-2-2b)"; fi
  if [ "$HF_OFF" = 1 ]; then
    for f in google--gemma-2-2b unsloth--gemma-2-2b google--gemma-scope-2b-pt-res; do
      [ -f "$HF_HUB/models--$f/refs/main" ] || echo "refs/main of $f (DW_HF_OFFLINE=1 loads by name; or run with DW_HF_OFFLINE=0)"
    done
  fi
  { [ -x "$CT" ] && [ -x "$CAP" ]; } || command -v "$PY311" > /dev/null 2>&1 || echo "$PY311 (for setup_env.sh)"
}

print_p0_commands() {   # the clones and downloads, for Daniel to approve (never run here)
  local r
  r=$(q "$ROOT")
  echo "  --- P0 commands for Daniel to approve (pod shell; HF_TOKEN only in this shell's environment: never a file, never echoed) ---"
  echo "    \$ git clone https://github.com/NainaniJatinZ/ScalableSAECircuits.git third_party/ScalableSAECircuits && git -C third_party/ScalableSAECircuits checkout $CAPLES_PIN   # their code + SVA data"
  [ "$D1" = 1 ] && echo "    \$ git clone https://github.com/decoderesearch/circuit-tracer.git third_party/circuit-tracer && git -C third_party/circuit-tracer checkout $CT_PIN   # their tests (the package itself is pip-installed from git by setup_env.sh)"
  echo "    \$ command -v $PY311 || (apt-get update && apt-get install -y python3.11 python3.11-venv)   # only if missing"
  echo "    \$ read -rsp 'HF read token: ' HF_TOKEN && export HF_TOKEN; echo   # typed into this shell only"
  echo "    \$ ROOT=$r PY=$PY311 FETCH=1 bash $AD/setup_env.sh 2>&1 | tee $LOGS/P0/setup_env.log   # venvs ct107 + caples107 (pip, ~12 GB) + weights: google fp32 10.48 GB, unsloth bf16 5.25 GB, PLT 7.86 GB, Gemma Scope res 1.51 GB, Dallas JSON 2.75 MB (curl)"
  echo "  ---"
}

p0_assets() {
  local miss
  miss=$(assets_missing)
  if [ -n "$miss" ]; then
    say "  missing: $(echo "$miss" | paste -sd';' -)"
    print_p0_commands
    [ "$DRY_RUN" = 1 ] || return 2
  elif [ "$DRY_RUN" = 1 ]; then
    say "  every pinned asset is present here; the approval list, for the pod:"
    print_p0_commands
  fi
  [ -s "$LOCAL_MEANS" ] || say "  note: no local bf16 means file $LOCAL_MEANS (copy it to the pod for the cross-check; skipped otherwise)"
  return 0
}

# ------------------------------------------------------------------------------------------------- P1 envs
p1_envs() {
  if [ ! -x "$CT" ] || [ ! -x "$CAP" ] || [ "$DRY_RUN" = 1 ]; then
    echo "  --- P1 command for Daniel to approve (pip downloads ~12 GB; P0's FETCH=1 command builds the venvs too) ---"
    echo "    \$ ROOT=$(q "$ROOT") PY=$PY311 bash $AD/setup_env.sh 2>&1 | tee $LOGS/P1/setup_env.log"
    echo "  ---"
    if [ "$DRY_RUN" != 1 ]; then say "  ct107 / caples107 not built yet"; return 2; fi
  fi
  local L=$LOGS/P1/gate.log g=${GG[0]}
  run_cmd "$L" "$CT_ENV/bin/pip check && $CAPLES_ENV/bin/pip check" || return 1
  run_cmd "$L" "$BASE CUDA_VISIBLE_DEVICES=$g $PY -B -c 'import torch; assert torch.cuda.is_available(); print(\".venv torch\", torch.__version__, torch.cuda.get_device_name(0))'" || return 1
  run_cmd "$L" "$BASE CUDA_VISIBLE_DEVICES=$g $CT -I -B -c 'import torch; assert torch.cuda.is_available(); print(\"ct107 torch\", torch.__version__, torch.cuda.get_device_name(0))'" || return 1
  run_cmd "$L" "$BASE CUDA_VISIBLE_DEVICES=$g $CAP -I -B -c 'import torch; assert torch.cuda.is_available(); print(\"caples107 torch\", torch.__version__, torch.cuda.get_device_name(0))'" || return 1
}

# ------------------------------------------------------------------------------------------------- P2 gate
gate_d2() {   # gate_d2 GPU: the D2 half; rc 0 = pass (STATE/gate_d2.ok)
  local g=$1 L=$LOGS/P2 ok=1 pt="$PY -m pytest $AD/test_wcm_gemma.py -m gpu -q -s -rs -p no:cacheprovider"
  # informative (a mismatch never fails the gate)
  step "$L/weights.log" weights_check "[ -s $CHK/weights.json ]" \
    "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/pod_checks_d.py weights --out $CHK/weights.json" \
    || say "  weights check did not run (informative only)"
  step "$L/caples_means.log" caples_means "[ -s $MEANS ]" \
    "$BASE CUDA_VISIBLE_DEVICES=$g $CAP -I -u $AD/caples_means.py --device cuda --hf-cache $(q "$HF_HUB") --out $MEANS" || ok=0
  if [ "$DRY_RUN" = 1 ] || [ -s "$LOCAL_MEANS" ]; then
    step "$L/means_vs_local.log" means_vs_local "[ -s $CHK/means_vs_local.json ]" \
      "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/pod_checks_d.py means --pod $MEANS --local $LOCAL_MEANS --out $CHK/means_vs_local.json" \
      || say "  means cross-check did not run (informative only)"
  fi
  step "$L/d6_gsres.log" d6_gsres "passed 1 $L/d6_gsres.log" \
    "$BASE CUDA_VISIBLE_DEVICES=$g ARENA_D_ID_DTYPE=$ID_DTYPE $pt -k 'real_open_gate and gsres'" || ok=0
  if [ "$DRY_RUN" = 1 ] || [ -s "$MEANS" ]; then
    step "$L/test4.log" test4_real_caples "passed 1 $L/test4.log" \
      "$BASE CUDA_VISIBLE_DEVICES=$g ARENA_D_CAPLES_PT=$MEANS $pt -k real_caples" || ok=0
  else ok=0; fi
  step "$L/grid_A.log" caples_grid_A "[ -f $RUNC/A/DONE ]" \
    "$BASE GPU=$g RUN=$RUNC CAPLES_PY=$CAP PY=$PY DENSE_N=$DENSE_N bash $AD/run_caples.sh A" || ok=0
  step "$L/fidelity_A.log" caples_fidelity_A "grep -q '^ \"passed\": true' $FID/caples_A.json 2> /dev/null" \
    "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/caples_fidelity.py --runs $RUNC --grids A --out $FID/caples_A.json" || ok=0
  [ "$DRY_RUN" = 1 ] && return 0
  if [ "$ok" = 1 ]; then mark gate_d2 ok; rm -f "$STATE/gate_d2.fail"; say "  gate D2: PASS"; return 0; fi
  rm -f "$STATE/gate_d2.ok"; mark gate_d2 fail; say "  gate D2: FAIL (logs $L)"; return 1
}

gate_d1() {   # gate_d1 GPU: the D1 half; rc 0 = pass (STATE/gate_d1.ok)
  local g=$1 L=$LOGS/P2 ok=1 t n_t
  n_t=$(echo "$TASKS" | tr ',' '\n' | grep -c .)
  local pt="$PY -m pytest $AD/test_wcm_gemma.py -m gpu -q -s -rs -p no:cacheprovider"
  local agf="$BASE CUDA_VISIBLE_DEVICES=$g $CT -B -I -u $AD/ag_fidelity.py --checks F1,F3,F4,F5,F6 --device cuda"
  step "$L/ag_prompts.log" ag_prompts "[ \$(ls $AGO/prompts/*.json 2> /dev/null | wc -l) -ge $n_t ]" \
    "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/ag_tasks.py prompts --tasks $TASKS --out $AGO" || ok=0
  step "$L/d6_plt.log" d6_plt "passed 1 $L/d6_plt.log" \
    "$BASE CUDA_VISIBLE_DEVICES=$g ARENA_D_ID_DTYPE=$ID_DTYPE $pt -k 'real_open_gate and plt'" || ok=0
  step "$L/ag_fidelity_bf16_unsloth.log" ag_fidelity_bf16_unsloth "[ -s $FID/ag_fidelity_bf16_unsloth.json ]" \
    "$agf --dtype bfloat16 --hf-model unsloth/gemma-2-2b --out $FID/ag_fidelity_bf16_unsloth.json" \
    || say "  ag_fidelity bf16 unsloth did not run (informative only)"
  step "$L/ag_fidelity_bf16_google.log" ag_fidelity_bf16_google "[ -s $FID/ag_fidelity_bf16_google.json ]" \
    "$agf --dtype bfloat16 --out $FID/ag_fidelity_bf16_google.json" \
    || say "  ag_fidelity bf16 google did not run (informative only)"
  step "$L/ag_fidelity_fp32_google.log" ag_fidelity_fp32_google \
    "grep -q '^VERDICT: ALL PASS' $L/ag_fidelity_fp32_google.log 2> /dev/null" \
    "$agf --dtype float32 --strict --out $FID/ag_fidelity_fp32_google.json" || ok=0
  # their own test, from a copy of the clone's tests/ (so the INSTALLED package is what is tested; nothing runs in third_party/)
  step "$L/ct_test_gemma_2_2b.log" ct_test_gemma_2_2b "passed 1 $L/ct_test_gemma_2_2b.log" \
    "mkdir -p $WORK/ct_tests && cp -r third_party/circuit-tracer/tests third_party/circuit-tracer/pyproject.toml $WORK/ct_tests/ && cd $WORK/ct_tests && $BASE CUDA_VISIBLE_DEVICES=$g $CT -B -m pytest -q -rs -p no:cacheprovider tests/test_attributions_gemma.py -k 'test_gemma_2_2b and not clt'" \
    || ok=0
  [ "$DRY_RUN" = 1 ] && return 0
  if [ "$ok" = 1 ]; then mark gate_d1 ok; rm -f "$STATE/gate_d1.fail"; say "  gate D1: PASS"; return 0; fi
  rm -f "$STATE/gate_d1.ok"; mark gate_d1 fail; say "  gate D1: FAIL (logs $L)"; return 1
}

hf_refs() {   # record which revision every by-name load resolved (online loads follow main; informative)
  local f
  [ "$DRY_RUN" = 1 ] && { echo "    \$ (record) refs/main of the 4 Gemma repos vs their pins -> $CHK/hf_refs.txt"; return 0; }
  mkdir -p "$CHK"
  for f in google--gemma-2-2b unsloth--gemma-2-2b mwhanna--gemma-scope-transcoders google--gemma-scope-2b-pt-res; do
    echo "$f main=$(cat "$HF_HUB/models--$f/refs/main" 2> /dev/null || echo none) snapshots=$(ls "$HF_HUB/models--$f/snapshots" 2> /dev/null | paste -sd, -)"
  done > "$CHK/hf_refs.txt"
}

p2_gate() {
  local r2=0 r1=0 p2 p1
  if [ "$DRY_RUN" != 1 ] && [ -f "$STATE/gate_d2.ok" ] && { [ "$D1" != 1 ] || [ -f "$STATE/gate_d1.ok" ]; }; then
    say "  both gate halves passed earlier"; return 0
  fi
  if [ "${#GG[@]}" -ge 2 ] && [ "$D1" = 1 ] && [ "$DRY_RUN" != 1 ]; then      # the halves in parallel on two GPUs
    gate_d2 "${GG[0]}" & p2=$!
    gate_d1 "${GG[1]}" & p1=$!
    wait "$p2"; r2=$?; wait "$p1"; r1=$?
  else
    [ "$DRY_RUN" = 1 ] && echo "  [gate D2 half on GPU ${GG[0]}]"
    gate_d2 "${GG[0]}"; r2=$?
    if [ "$D1" = 1 ]; then
      [ "$DRY_RUN" = 1 ] && echo "  [gate D1 half on GPU ${GG[1]:-${GG[0]}}]"
      gate_d1 "${GG[1]:-${GG[0]}}"; r1=$?
    fi
  fi
  hf_refs
  [ "$DRY_RUN" = 1 ] && return 0
  if [ "$GATE_MODE" = warn ]; then
    { [ "$r2" = 0 ] && [ "$r1" = 0 ]; } || say "  DW_GATE_MODE=warn: the failed half's jobs run anyway, LABELLED gate-failed in the summary"
    return 0
  fi
  if [ "$r2" != 0 ] && { [ "$D1" != 1 ] || [ "$r1" != 0 ]; }; then return 1; fi
  return 0
}

half_on() {   # half_on d1|d2: that half's production jobs run (gate passed, or DW_GATE_MODE=warn, or a dry run)
  [ "$DRY_RUN" = 1 ] || [ -f "$STATE/gate_$1.ok" ] || [ "$GATE_MODE" = warn ]
}

# ------------------------------------------------------------------------------------------------- P3 jobs
wcm_cmd() {   # wcm_cmd FAMILY TASKS OUT [args]: wcm_gemma.py run (fp32, google weights) on @GPU@
  local fam=$1 tasks=$2 out=$3; shift 3
  echo "$BASE CUDA_VISIBLE_DEVICES=@GPU@ $PY -u $AD/wcm_gemma.py run --family $fam --tasks $tasks --lams $LAMS --steps $STEPS --out $out $*"
}
rows_ge() { [ "$(nlines "$1")" -ge "$2" ]; }    # rows_ge FILE N: at least N rows (eval.jsonl: one per finished lambda)

build_jobs() {
  local G t m
  qreset
  if half_on d2; then
    # order (plan §3: D2 first): the longest Caples job (dense, 3 x DENSE_N lambdas), the D2-WCM rows (Caples mode first:
    # the like-for-like row), then the other grids
    if [[ " $CAPLES_GRIDS " == *" dense "* ]]; then
      qadd caples_dense "$BASE GPU=@GPU@ RUN=$RUNC CAPLES_PY=$CAP PY=$PY DENSE_N=$DENSE_N bash $AD/run_caples.sh dense" \
        "[ -f $RUNC/dC/DONE ] && [ -f $RUNC/dB/DONE ] && [ -f $RUNC/dA/DONE ]"
    fi
    qadd wcm_d2_caples_pp "$(wcm_cmd gemma-2-2b-gsres-gap6 rc "$W2/caples_pp" --data caples --mode caples --means per_position --caples-means "$MEANS")" \
      "rows_ge $W2/caples_pp/eval.jsonl $N_LAMS"
    if [ "$D2_NATIVE" = 1 ]; then
      qadd wcm_d2_native_pp "$(wcm_cmd gemma-2-2b-gsres-gap6 rc "$W2/native_pp" --data caples --mode native --means per_position)" \
        "rows_ge $W2/native_pp/eval.jsonl $N_LAMS"
      qadd wcm_d2_native_pooled "$(wcm_cmd gemma-2-2b-gsres-gap6 rc "$W2/native_pooled" --data caples --mode native --means pooled)" \
        "rows_ge $W2/native_pooled/eval.jsonl $N_LAMS"
    fi
    for G in $CAPLES_GRIDS; do
      [ "$G" = dense ] && continue
      qadd "caples_$G" "$BASE GPU=@GPU@ RUN=$RUNC CAPLES_PY=$CAP PY=$PY bash $AD/run_caples.sh $G" "[ -f $RUNC/$G/DONE ]"
    done
  else say "  D2 jobs OFF (gate D2 failed)"; fi
  if [ "$D1" = 1 ]; then
    if half_on d1; then
      for t in ${TASKS//,/ }; do               # rc first (plan §1.1), then the others in DW_TASKS order
        [ "$t" = rc ] || continue
        for m in pooled per_position; do
          qadd "wcm_d1_${t}_$m" "$(wcm_cmd gemma-2-2b-plt "$t" "$W1/${t}_$m" --data sfc --means $m)" "rows_ge $W1/${t}_$m/eval.jsonl $N_LAMS"
        done
        qadd "ag_$t" "$BASE CUDA_VISIBLE_DEVICES=@GPU@ $CT -B -I -u $AD/$AG_PY run --tasks $t --split held --limit $AG_LIMIT --dtype float32 --batch-size $AG_BATCH --device cuda --out $AGO" \
          "[ \$(ls $AGO/$t/held/p*.json 2> /dev/null | wc -l) -ge $AG_K ]"
      done
      for t in ${TASKS//,/ }; do
        [ "$t" = rc ] && continue
        for m in pooled per_position; do
          qadd "wcm_d1_${t}_$m" "$(wcm_cmd gemma-2-2b-plt "$t" "$W1/${t}_$m" --data sfc --means $m)" "rows_ge $W1/${t}_$m/eval.jsonl $N_LAMS"
        done
        qadd "ag_$t" "$BASE CUDA_VISIBLE_DEVICES=@GPU@ $CT -B -I -u $AD/$AG_PY run --tasks $t --split held --limit $AG_LIMIT --dtype float32 --batch-size $AG_BATCH --device cuda --out $AGO" \
          "[ \$(ls $AGO/$t/held/p*.json 2> /dev/null | wc -l) -ge $AG_K ]"
      done
    else say "  D1 jobs OFF (gate D1 failed)"; fi
  fi
}

p3_jobs() {
  build_jobs
  [ "${#QN[@]}" -gt 0 ] || { say "  no production job (both gate halves failed?)"; return 1; }
  q_run P3
}

# ------------------------------------------------------------------------------------------------- P4 scoring
caples_dirs() {   # caples_dirs GRID: their per-lambda run folders of one grid, comma-joined
  if [ "$DRY_RUN" = 1 ]; then echo "<every $RUNC/$1/src/$SAVE/logit_diff_*_run>"; return; fi
  local d=("$RUNC/$1/src/$SAVE"/logit_diff_*_run)
  (IFS=,; echo "${d[*]}")
}
wcm_jsons() {     # wcm_jsons OUT: the circuit JSONs of one D2-WCM configuration, comma-joined
  if [ "$DRY_RUN" = 1 ]; then echo "<every circuit json under $1/circuits>"; return; fi
  find "$1/circuits" -name '*.json' 2> /dev/null | sort | paste -sd, -
}
score_cmd() {   # score_cmd OUT ARGS: score_d.py all (ours in .venv, then theirs in caples107, merged)
  local out=$1; shift
  echo "$BASE CUDA_VISIBLE_DEVICES=@GPU@ $PY -u $AD/score_d.py all --caples-means $MEANS --modes native,caples --means per_position --device cuda --out $out $*"
}
scored() { [ -s "$1/scores.jsonl" ] && [ -s "$1/theirs.json" ]; }

p4_score() {
  local rc=0 G grids c d
  if half_on d2; then
    run_cmd "$LOGS/P4/caples_fidelity.log" "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/caples_fidelity.py --runs $RUNC --grids A,B,C --out $FID/caples_fidelity.json" \
      caples_fidelity || rc=1
  fi
  qreset
  if half_on d2; then
    grids="A"
    for G in $CAPLES_GRIDS; do if [ "$G" = dense ]; then grids="$grids dC dB dA"; else grids="$grids $G"; fi; done
    for G in $grids; do
      if [ "$DRY_RUN" != 1 ] && [ ! -f "$RUNC/$G/DONE" ]; then say "  score: grid $G has no DONE marker; not scored"; rc=1; continue; fi
      qadd "score_caples_$G" "$(score_cmd "$SD/caples_$G" --caples-dirs "$(caples_dirs "$G")")" "scored $SD/caples_$G"
    done
    for c in caples_pp native_pp native_pooled; do
      [ "$D2_NATIVE" = 1 ] || [ "$c" = caples_pp ] || continue
      d=$(wcm_jsons "$W2/$c")
      if [ -z "$d" ]; then say "  score: no WCM circuit in $W2/$c; not scored"; rc=1; continue; fi
      qadd "score_wcm_$c" "$(score_cmd "$SD/wcm_$c" --wcm-circuits "$d" --no-controls)" "scored $SD/wcm_$c"
    done
  fi
  # D1 rows: ours = wcm_gemma.py's eval.jsonl (score_task, written by the fits); theirs for AG = ag_tasks.py's per-prompt
  # completeness / replacement. score_d.py is the residual family only (no PLT / AG path; open item, see README §9)
  [ "${#QN[@]}" -gt 0 ] || return "$rc"
  q_run P4 || rc=1
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- P5 pack
p5_pack() {
  local res=$TAG-arena-d-results.tgz
  run_cmd "$LOGS/P5/summary.log" "$BASE CUDA_VISIBLE_DEVICES= $PY -B $AD/pod_summary_d.py $WORK --lams $LAMS --tasks $TASKS --ag-k $AG_K" \
    P5_summary || say "  summary FAILED (packing anyway)"
  # the results tgz leaves the attribution graphs out (they go into their own archive below, AFTER the results are
  # verified on RESULTS: ~16 GB must never delay or sink the small results archive)
  run_cmd "$LOGS/P5/pack.log" "mkdir -p $CKPT && tar -czf $CKPT/$res --exclude=.git --exclude=$WORK/ct_tests --exclude='*.graph.pt' --exclude='*.graph.pt.tmp' $WORK $AD/env_freeze_ct107.txt $AD/env_freeze_caples107.txt && (cd $CKPT && md5sum $res > $TAG-arena-d.md5)" \
    P5_pack || return 1
  run_cmd "$LOGS/P5/pack.log" "mkdir -p $RESULTS && cp --no-preserve=mode,ownership $CKPT/$res $CKPT/$TAG-arena-d.md5 $RESULTS/ ; (cd $RESULTS && md5sum -c $TAG-arena-d.md5)" \
    P5_copy || { say "  copy / md5 check on $RESULTS failed"; return 1; }
  [ "$DRY_RUN" = 1 ] || say "  verified on $RESULTS: $res"
  # the AG graphs (ag_tasks_graphs.py, AG_SAVE_GRAPHS=1): AGO/<task>/held/p<id>.graph.pt, ~0.1 GB each (float16
  # adjacency), 4 tasks x 40 = ~16 GB; packed whenever any exist (also after AG_SAVE_GRAPHS=0 on a resumed run)
  local gr=$TAG-arena-d-graphs.tgz ng=0
  [ "$DRY_RUN" = 1 ] || ng=$(find "$AGO" -name 'p*.graph.pt' 2> /dev/null | wc -l)
  if [ "$DRY_RUN" = 1 ] || [ "$ng" -gt 0 ]; then
    [ "$DRY_RUN" = 1 ] || say "  $ng attribution graphs ($(du -ch $(find "$AGO" -name 'p*.graph.pt') 2> /dev/null | tail -n 1 | cut -f1)) -> $gr"
    run_cmd "$LOGS/P5/pack.log" "mkdir -p $CKPT && find $AGO -name 'p*.graph.pt' -print0 | sort -z | tar $GZ -cf $CKPT/$gr --null -T - && (cd $CKPT && md5sum $gr > $TAG-arena-d-graphs.md5)" \
      P5_pack_graphs || { say "  graphs archive FAILED (the results archive above is verified)"; return 1; }
    run_cmd "$LOGS/P5/pack.log" "cp --no-preserve=mode,ownership $CKPT/$gr $CKPT/$TAG-arena-d-graphs.md5 $RESULTS/ ; (cd $RESULTS && md5sum -c $TAG-arena-d-graphs.md5)" \
      P5_copy_graphs || { say "  graphs copy / md5 check on $RESULTS failed (the results archive above is verified)"; return 1; }
    [ "$DRY_RUN" = 1 ] || say "  verified on $RESULTS: $gr"
  fi
}

# ------------------------------------------------------------------------------------------------- main
say "arena D pod stage | $(git --no-pager log -1 --format='%h %s' 2> /dev/null) | GPUs $GPUS (gate $GATE_GPUS, $GATE_MODE) | D1 $D1 D2-native $D2_NATIVE | lambdas $LAMS | steps $STEPS | tasks $TASKS | Caples grids A(gate) $CAPLES_GRIDS | HF offline $HF_OFF | WORK $WORK | FROM $FROM STOP_AFTER $STOP_AFTER$([ "$STOP_EPOCH" != 99999999999 ] && echo " | STOP_EPOCH $(date -d "@$STOP_EPOCH" '+%a %H:%M')")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$LOGS" "$STATE/pids" || exit 1
  (                                                        # GPU memory every 30 s, for the D10 re-cost
    while true; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2> /dev/null \
        | awk -v t="$(date +%s)" '{ print t "," $0 }' >> "$WORK/gpu_mem.csv"
      sleep 30
    done
  ) > /dev/null 2>&1 < /dev/null &
  SAMPLER=$!
  WATCHDOG=
  if [ "$STOP_EPOCH" != 99999999999 ]; then
    (                                                      # hard stop: STOP_EPOCH + DW_HARD_GRACE kills every running job
      while [ "$(date +%s)" -lt $((STOP_EPOCH + HARD_GRACE)) ]; do sleep 30; done
      date -Is > "$STATE/HARD_STOPPED"
      # P5's pack / copy steps (pid files P5_*) are spared: a pack that straddles the hard stop is the point of it
      for f in "$STATE"/pids/*; do [[ "${f##*/}" == P5_* ]] || kill -TERM -- "-$(cat "$f")" 2> /dev/null; done
      sleep 30
      for f in "$STATE"/pids/*; do [[ "${f##*/}" == P5_* ]] || kill -KILL -- "-$(cat "$f")" 2> /dev/null; done
    ) > /dev/null 2>&1 < /dev/null &
    WATCHDOG=$!
  fi
  trap 'kill $SAMPLER $WATCHDOG 2> /dev/null' EXIT
fi

FAILS=0
if [ "$DRY_RUN" = 1 ]; then          # parse checks only (nothing is written)
  for f in wcm_gemma.py test_wcm_gemma.py caples_means.py caples_fidelity.py ag_fidelity.py ag_tasks.py score_d.py \
           pod_checks_d.py pod_summary_d.py $([ "$SAVE_GRAPHS" = 1 ] && echo ag_tasks_graphs.py); do
    if [ -x "$PY" ]; then
      "$PY" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$AD/$f" 2> /dev/null && say "  ok parse $f" \
        || { say "  PARSE ERROR $f"; FAILS=$((FAILS + 1)); }
    fi
  done
  for f in run_caples.sh setup_env.sh; do bash -n "$AD/$f" && say "  ok bash -n $f" || { say "  BASH ERROR $f"; FAILS=$((FAILS + 1)); }; done
fi
pack_and_exit() {   # pack_and_exit RC: P5 runs whatever happened before (unless nothing ran or DW_STOP_AFTER < P5)
  local rc=$1
  if [ "$(snum "$STOP_AFTER")" -ge 5 ] && { [ "$DRY_RUN" = 1 ] || [ -f "$STATE/P2.start" ]; }; then
    [ "$DRY_RUN" = 1 ] || rm -f "$STATE/P5.done"           # every run packs again (a resumed run adds results)
    stage P5 pack p5_pack || FAILS=$((FAILS + 1))
  fi
  [ "$rc" = 3 ] && { say "stopped after $STOP_AFTER as asked ($FAILS failed stages)"; [ "$FAILS" = 0 ]; exit $?; }
  say "arena D finished: rc $rc, $FAILS failed stages$([ "$FAILS" = 0 ] || echo ' (rerun to resume: finished work is skipped)')"
  [ "$rc" = 0 ] && [ "$FAILS" != 0 ] && rc=1
  exit "$rc"
}
if [ "$DRY_RUN" != 1 ] && [ -n "$(ls "$STATE"/gate_d*.fail 2> /dev/null)" ]; then
  rm -f "$STATE/P2.done"                                   # a failed gate half is retried (its passed steps are skipped)
fi
stage P0 assets p0_assets; rc=$?; [ "$rc" = 0 ] || pack_and_exit "$rc"
stage P1 envs p1_envs; rc=$?; [ "$rc" = 0 ] || pack_and_exit "$rc"
stage P2 gate p2_gate; rc=$?; [ "$rc" = 0 ] || pack_and_exit "$rc"
stage P3 jobs p3_jobs; rc=$?; [ "$rc" = 3 ] && pack_and_exit 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P4 scoring p4_score; rc=$?; [ "$rc" = 3 ] && pack_and_exit 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
pack_and_exit 0
