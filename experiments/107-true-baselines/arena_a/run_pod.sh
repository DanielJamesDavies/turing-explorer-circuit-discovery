#!/bin/bash
# Arena A pod stage (IMPLEMENTATION_PLAN item A10; README section 8): Ge et al. (A1), Dunefsky et al. (A2) and WCM on
# GPT-2 small (Kissane z-SAEs + Dunefsky transcoders), on the 8xH100 pod AFTER the main 082 chain. The chain calls it
# with ARENA_A=1 (082/chain.sh arena_a_stage, A11); it also runs on its own. Run from anywhere; it cds to the repo root.
# PROPOSAL ONLY until Daniel approves it on the pod ([[remote-pod-discuss-first]]).
#
#   DRY_RUN=1 bash experiments/107-true-baselines/arena_a/run_pod.sh      # print every command, run nothing
#   STOP_AT=P1 bash experiments/107-true-baselines/arena_a/run_pod.sh     # the gates only: prints the download / env
#                                                                         # commands for approval when anything is missing
#   nohup bash experiments/107-true-baselines/arena_a/run_pod.sh > <AA_WORK>/run_pod.out 2>&1 &
#
# STAGES (each resumable: a STATE/<P>.done marker skips a finished stage; inside a stage every runner skips the
# circuits / fits / rows already written, so a rerun resumes per circuit)
#   P0 assets   checks only: the two pinned clones (third_party/ is not in git), GPT-2, the Dunefsky transcoders, the
#               Kissane z-SAEs, the OWT parquet (all pinned revisions, in the HF cache). Anything missing -> the
#               download commands are PRINTED for Daniel to approve and run, and the stage exits 2: this script never
#               downloads. Then (local, no download): the /home/daniel HF-cache path the arena A modules hard-code is
#               linked to this machine's cache, and 035's LN-folded transcoders_ct are regenerated (convert_transcoders.py,
#               deterministic and asserted)
#   P1 envs     ge107 / tc107 must exist (setup_env.sh TORCH_INDEX=cu121: a pip download, so PRINTED for approval when
#               missing, exit 2). Gate: pip check in both, CUDA visible in both, Ge's own test_attributor.py (G0). D-F1
#               (Dunefsky R1) runs inside every run_dunefsky.py start-up
#   P2 data     corpus: the first AA_N_SEQ windows of the same pinned OWT source as the 082 S1 corpus (069
#               prepare_corpus.py, offline from the HF cache); AA_N_SEQ=0 uses AA_DATA_SRC (the 082 S1 corpus) as is.
#               Then data_pass_tcz.py (pass 1 + neg_ctx, float32) on GPU 0
#   P3 targets  make_targets.py (CPU): 48 targets, K = 8 prompts each, records (ctx/), MANIFEST.json
#   P4 baselines a GPU canary (1 Ge circuit at 3 tau, 1 Dunefsky circuit at 5 branches) first; then one worker per GPU:
#               run_ge.py (ge107) one process PER TARGET (the residual leak of their gate closure is VRAM on the GPU, so
#               the process boundary releases it; the RSS re-exec stays ON, --max-rss-gb AA_GE_RSS), then run_dunefsky.py
#               (tc107) on the worker's tc targets: num_branches 5..60 at num_iters 5, then the extended knob (100..300
#               branches on K = 2 prompts, LABEL "beyond published range": the paper's runs stop at 60; the scorer's
#               units carry num_branches, so > 60 = that label; AA_DUNEFSKY_EXT=0 drops it, cut line 1)
#   P4h held    (Daniel 2026-11-05 13:35) the baselines ALSO on held-out prompts: score_perprompt.py
#               --emit-heldout-prompts (CPU; the AA_HELD_K highest-peak held-out strongest rows per target, = K) into
#               AA_WORK/targets_held/prompts.jsonl, which run_ge.py / run_dunefsky.py read unchanged through --targets;
#               then the P4 worker scheme: Ge (same tau grid) into ge_heldout/, Dunefsky on the PUBLISHED branch grid
#               only (no > 60 knob on held-out) into dunefsky_heldout/
#   P5 WCM      driver_tcz.py (062's driver, WCM_VERSION v4, MODE=main) through 062/launch.sh on all GPUs (one shard per
#               GPU, P = AA_WCM_P), one launch per arm: scope a1 (z + z2 + tc, 48 targets) and a2 (tc only, the 24 tc
#               targets: Dunefsky's universe) x AA_LAMBDAS; each arm its own OUT
#   P6 scoring  one worker per GPU: score_arena_a.py scope a1 (Ge + WCM a1) then a2 (Dunefsky + WCM a2) on the worker's
#               targets, with --selftest --donor (one shared R pool, file-locked); then --emit-ge-jobs per scope (CPU) and
#               ge_recovery_tcz.py (ge107, --check) on the jobs, split over the GPUs
#   P6b perprompt  A8b ("their turf"; README A8b): one worker per GPU, score_perprompt.py scope a1 (WCM a1 arms:
#               wcm_full, wcm_fire, wcm_fire_top in arm AA_PP_TOPK_ARM; + Ge on the held prompts) then a2 (WCM a2 arms
#               + Dunefsky on the held prompts), on the train AND held prompt sets (AA_PP_PROMPTS each); the P4
#               baselines on their own (train) prompts are score_arena_a's home rows from P6 (--baseline-rows). Then the
#               shards merged into perprompt/all, its Ge recovery jobs (ge_recovery_tcz.py, per shard; WCM units +
#               Dunefsky's held-out circuits, the latter emitted by ge_jobs_heldout.py = score_arena_a's emitter), and
#               --summary (summary.md, curves.jsonl / .csv)
#   P7 pack     the score shards merged into score/all, --summary (CPU), results + stores tgz with md5 in CKPT, copied to
#               RESULTS and md5-checked there. P7 runs on EVERY run (its marker never skips it: a resumed or stopped
#               run adds results) and also after the time limit (soft or hard stop), on whatever finished: a missing
#               store / corpus / file is left out, a failed --summary is logged and the pack goes on
#
# TIME LIMIT (2026-11-06; the hard stop): a soft stop at STOP_EPOCH (no new target / arm / stage; the runners finish
# their unit) and a HARD stop AA_HARD_GRACE s later: every command this run started (each one is tagged with
# AA_RUN_TAG=<run id> in its environment, which every child inherits: launch.sh shards, Ge / Dunefsky / scorer workers)
# gets SIGTERM, then SIGKILL after AA_KILL_GRACE s; what was killed goes to LOGS/hard_stop.log (ps line per process) and
# events.log; no new command starts after it (except P7's); then P7 packs and the script exits 4. If run_pod.sh is
# still running AA_PACK_MAX_H after the hard stop (a hung P7), the watchdog kills P7's commands and run_pod.sh itself,
# so the chain always gets control back (arena D, STOP_POD). Default limit (AA_MAX_H=auto): 1.5 x the wall estimate
# from the README section 7 / plan section 5 costs, 2 h serial (P0-P3, the GPU-0 data pass) + 68 GPU-h / #GPUs
# (45-55 GPU-h + P4h / P6b 10.5-14.5 GPU-h, upper end): 15.75 h on 8 GPUs, 28.5 h on 4. Nothing is ever deleted.
#
# ENV (defaults in brackets)
#   DRY_RUN [0]        1 = print every command; nothing is run, written or created
#   STOP_AT [P7]       the last stage to run (STOP_AT=P3: P0-P3 only)
#   STOP_EPOCH         no new target / arm / stage once date +%s >= it (the runners finish the unit they are on); the
#                      hard stop follows AA_HARD_GRACE s later. Unset = derived from AA_MAX_H (soft = hard - grace)
#   AA_MAX_H [auto]    the hard stop, in hours after this script starts (decimals allowed; auto = 1.5 x the estimate,
#                      see TIME LIMIT; 0 = no limit unless STOP_EPOCH is set). With STOP_EPOCH too, the earlier wins
#   AA_HARD_GRACE [3600] s from the soft to the hard stop   AA_KILL_GRACE [30] s from SIGTERM to SIGKILL
#   AA_PACK_MAX_H [2]  hours P7 may take after a hard stop before the watchdog kills run_pod.sh too
#   AA_WATCH_EVERY [30] the watchdog's poll, s   AA_HARDSTOP_TEST [0] CPU-only test of the time limit (fake P4 process
#                      trees of sleeps, one ignoring SIGTERM, then a fake P7; "hang" = a P7 that hangs, for the
#                      backstop): nothing real runs; set AA_WORK to a scratch dir and AA_MAX_H to seconds' worth
#   AA_WORK [107/arena_a/out/pod]  every output, LOGS = AA_WORK/logs, STATE = AA_WORK/state, steps.tsv, gpu_mem.csv
#   AA_GPUS [0,...,7]  one worker / shard per GPU
#   AA_DATA_SRC [data_public/gpt2-small] the 082 S1 corpus; AA_N_SEQ [524288] windows of arena A's own corpus (0 = use
#                      AA_DATA_SRC as is: ~4M windows, a data pass of several H100-hours)
#   AA_DP_BATCH [512]  data-pass batch (069's config default)
#   AA_LAMBDAS [2.5e-4,5e-4,1e-3,2e-3,4e-3]  the WCM lambda arms: a PLACEHOLDER (coordinator 2026-11-05; tuning may
#                      change it); AA_LAMS is read as a fallback name   AA_SCOPES ["a1 a2"]
#   AA_WCM_VERSION [v4] (settled for arena A; the chain's WCM_VERSION is NOT inherited)   AA_WCM_P [1]   STAGGER [5]
#   AA_GE_RSS [9]      run_ge / ge_recovery --max-rss-gb (the re-exec stays on)   AA_GE_K [all = 8]   AA_GE_NTAU [30]
#   AA_DUNEFSKY_K [all = 8] AA_DUNEFSKY_BRANCHES [5,10,15,20,30,40,60] (the published range)
#   AA_DUNEFSKY_EXT [1] AA_DUNEFSKY_EXT_K [2] AA_DUNEFSKY_EXT_BRANCHES [100,150,200,300] ("beyond published range")
#   AA_BASELINE_DEVICE [cuda]  run_ge / ge_recovery --device (cpu for a local smoke: the ge107 / tc107 wheels there
#                      are CPU builds, and their pinned torch cannot drive the local sm_120 card)
#   AA_FROM            stages before it are taken as done (no marker written), e.g. AA_FROM=P4 for a smoke on
#                      existing stores / targets
#   AA_REDO            comma list of stages whose STATE/<P>.done marker is ignored, so they run again (their runners
#                      still skip finished rows), e.g. AA_REDO=P6b,P7 after a failed P6: P6b's baseline rows and P7's
#                      merge + summary are rebuilt from the new P6 shards
#   AA_DONOR_RECORDS [AA_TARGETS_DIR/ctx]  P6 --donor-records: the records the R donor pool is filtered against
#                      (donor_test.load_or_build_pool, >= 32 survivors). SMOKE ONLY: "run" = score_arena_a's small-
#                      corpus mode (the same 073 pool and exact / near filter, against the run's target records only,
#                      >= 16 survivors). The 2048-window smoke corpus leaves 28 of 512 donors against all 48 smoke
#                      records (2026-11-05); the pod corpus (524k windows) leaves ~500. Never set it on the pod
#   AA_STORES [AA_WORK/datapass]  an existing data pass (P2 then finds it and skips)
#   AA_TARGETS_DIR [AA_WORK/targets] an existing make_targets.py output (prompts.jsonl, ctx/, targets.txt), read only
#   AA_KEYS            smoke: an explicit comma list of target keys downstream of P3 (overrides AA_LIMIT)
#   AA_LIMIT           smoke: only the first N targets of targets.txt (and their tc subset) downstream of P3
#   AA_STEPS           an explicit driver STEPS override (smoke); it wins over CHOSEN.env's STEPS (unset = CHOSEN.env's
#                      STEPS if it sets one, else the protocol's 400)
#   AA_EXCLUDE [107/arena_a/tune_exclude.txt]  P3: make_targets.py EXCLUDE= (the WCM tuning dev targets never become
#                      production targets); a missing file fails P3
#   AA_TUNE_ENV [107/arena_a/out/tune/CHOSEN.env]  P5: the tuning choice (TUNING.md, tune_summary.py). Its KEY=VALUE
#                      lines (AA_RANK_W, NEC_W, STEPS, AA_TUNE_CONFIG only; '#' comments) go into every WCM fit's env,
#                      production included. Read once at P5 start and recorded in AA_WORK/wcm/TUNE_ENV; a later run
#                      with a different choice is refused (the fits would mix). Missing file = P5 FAILS, unless
#                      AA_ALLOW_UNTUNED=1 (then plain v4)
#   AA_PACK [1]        0 = P7 merges and summarises only (no tgz / RESULTS copy)
#   AA_HELD_K [8]      P4h: held-out prompts per target (= the training K)
#   AA_PP_PROMPTS [8]  P6b: prompts per set (train, held)   AA_PP_TAU_STRIDE [2] (Ge held units: every 2nd tau, the
#                      A8b cost basis)   AA_PP_TOPK_ARM [l1e-3 if in the grid, else the first]   AA_PP_CHECK [0]
#                      (1 = score_perprompt --check, the keep-all exactness check on the train prompts)
#   GE_ENV [~/venvs/ge107] TC_ENV [~/venvs/tc107] PY310 [python3.10] PY [./.venv/bin/python]
#   HF_HUB_CACHE [HF_HOME/hub, else ~/.cache/huggingface/hub]
#   CKPT [/root/ckpt] RESULTS [/workspace/results] TAG [pub]   (P7: TAG-arena-a-{results,stores}.tgz)
# Exit: 0 done (or stopped at STOP_AT), 1 a stage failed, 2 waiting for Daniel (downloads / env builds to approve),
#       4 stopped by the time limit (soft or hard stop) with stages unfinished, after P7 packed what finished.
set -uo pipefail
shopt -s nullglob

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
cd "$ROOT" || exit 1
A=experiments/107-true-baselines/arena_a
D62=experiments/062-h100-protocol-v1
D69=experiments/069-public-data-pass
DRY_RUN=${DRY_RUN:-0}
STOP_AT=${STOP_AT:-P7}
PY=${PY:-./.venv/bin/python}
WORK=${AA_WORK:-$A/out/pod}
LOGS=$WORK/logs
STATE=$WORK/state
GPUS=${AA_GPUS:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GL <<< "$GPUS"
NG=${#GL[@]}

# the time limit (header TIME LIMIT): HARD_EPOCH = the hard stop, STOP_EPOCH = the soft stop (99999999999 = none)
T0=$(date +%s)
RUN_ID=$T0.$$                                            # AA_RUN_TAG of this run's commands (P7's: RUN_ID.P7)
HARD_FLAG=$STATE/HARD_STOPPED.$RUN_ID                    # written at the hard stop (a new name per run: never removed)
MAIN_PID=$$
IN_P7=0
MAX_H=${AA_MAX_H:-auto}
HARD_GRACE=${AA_HARD_GRACE:-3600}
KILL_GRACE=${AA_KILL_GRACE:-30}
WATCH_EVERY=${AA_WATCH_EVERY:-30}
HS_TEST=${AA_HARDSTOP_TEST:-0}
h2s() { awk -v h="$1" 'BEGIN { printf "%d", h * 3600 }'; }
PACK_MAX_S=$(h2s "${AA_PACK_MAX_H:-2}")
HARD_EPOCH=""
LIMIT_NOTE="none"
[ -n "${STOP_EPOCH:-}" ] && HARD_EPOCH=$((STOP_EPOCH + HARD_GRACE)) && LIMIT_NOTE="STOP_EPOCH"
if [ "$MAX_H" != 0 ] && { [ -z "$HARD_EPOCH" ] || [ -n "${AA_MAX_H:-}" ]; }; then
  if [ "$MAX_H" = auto ]; then MAX_S=$(( (7200 + 68 * 3600 / NG) * 3 / 2 )); else MAX_S=$(h2s "$MAX_H"); fi
  if [ -z "$HARD_EPOCH" ] || [ $((T0 + MAX_S)) -lt "$HARD_EPOCH" ]; then
    HARD_EPOCH=$((T0 + MAX_S))
    LIMIT_NOTE="AA_MAX_H=$MAX_H: $(awk -v s="$MAX_S" 'BEGIN { printf "%.2f", s / 3600 }') h"
  fi
fi
if [ -n "$HARD_EPOCH" ]; then
  soft=$((HARD_EPOCH - HARD_GRACE)); [ "$soft" -lt "$T0" ] && soft=$T0
  { [ -z "${STOP_EPOCH:-}" ] || [ "$soft" -lt "$STOP_EPOCH" ]; } && STOP_EPOCH=$soft
fi
STOP_EPOCH=${STOP_EPOCH:-99999999999}
DATA_SRC=${AA_DATA_SRC:-data_public/gpt2-small}
N_SEQ=${AA_N_SEQ:-524288}
DP_BATCH=${AA_DP_BATCH:-512}
LAMS=${AA_LAMBDAS:-${AA_LAMS:-2.5e-4,5e-4,1e-3,2e-3,4e-3}}     # placeholder grid (tuning may change it)
SCOPES=${AA_SCOPES:-"a1 a2"}
WCMV=${AA_WCM_VERSION:-v4}
WCM_P=${AA_WCM_P:-1}
STAGGER=${STAGGER:-5}
GE_RSS=${AA_GE_RSS:-9}
GE_K=${AA_GE_K:-}
DF_EXT=${AA_DUNEFSKY_EXT:-1}
DF_EXT_K=${AA_DUNEFSKY_EXT_K:-2}
DF_EXT_BRANCHES=${AA_DUNEFSKY_EXT_BRANCHES:-100,150,200,300}
LIMIT=${AA_LIMIT:-}
KEYS=${AA_KEYS:-}
STEPS=${AA_STEPS:-}
EXCLUDE=${AA_EXCLUDE:-$A/tune_exclude.txt}
TUNE_ENV=${AA_TUNE_ENV:-$A/out/tune/CHOSEN.env}
ALLOW_UNTUNED=${AA_ALLOW_UNTUNED:-0}
TUNE_KV=""                                              # P5: the CHOSEN.env assignments (tune_env)
FROM=${AA_FROM:-P0}
REDO=${AA_REDO:-}
PACK=${AA_PACK:-1}
BDEV=${AA_BASELINE_DEVICE:-cuda}
GE_NTAU=${AA_GE_NTAU:-30}
DF_K=${AA_DUNEFSKY_K:-}
DF_BRANCHES=${AA_DUNEFSKY_BRANCHES:-5,10,15,20,30,40,60}
GE_ENV=${GE_ENV:-$HOME/venvs/ge107}
TC_ENV=${TC_ENV:-$HOME/venvs/tc107}
PY310=${PY310:-python3.10}
HF_HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
CKPT=${CKPT:-/root/ckpt}
RESULTS=${RESULTS:-/workspace/results}
TAG=${TAG:-pub}

if [ "$N_SEQ" = 0 ]; then CORPUS=$DATA_SRC; else CORPUS=$WORK/corpus; fi
STO=${AA_STORES:-$WORK/datapass}
TGT=${AA_TARGETS_DIR:-$WORK/targets}
GE_OUT=$WORK/ge
DF_OUT=$WORK/dunefsky
WCM_OUT=$WORK/wcm
SC=$WORK/score
CAN=$WORK/canary
HELD=$WORK/targets_held
GE_HELD=$WORK/ge_heldout
DF_HELD=$WORK/dunefsky_heldout
PP=$WORK/perprompt
HELD_K=${AA_HELD_K:-8}
PP_PROMPTS=${AA_PP_PROMPTS:-8}
PP_TAU_STRIDE=${AA_PP_TAU_STRIDE:-2}
PP_CHECK=${AA_PP_CHECK:-0}
case ",$LAMS," in *,1e-3,*) PP_TOPK_DEF=l1e-3 ;; *) PP_TOPK_DEF=l${LAMS%%,*} ;; esac
PP_TOPK=${AA_PP_TOPK_ARM:-$PP_TOPK_DEF}
if [ -n "$KEYS" ]; then LIST=$WORK/targets.keys.txt; LIST_TC=$WORK/targets_tc.keys.txt
elif [ -n "$LIMIT" ]; then LIST=$WORK/targets.limit.txt; LIST_TC=$WORK/targets_tc.limit.txt
else LIST=$TGT/targets.txt; LIST_TC=$WORK/targets_tc.txt; fi
N_TARGETS=48
K=8
N_DF_PRIMARY=$(echo "$DF_BRANCHES" | tr ',' '\n' | grep -c .)

# pinned assets (fetch_assets.sh / setup_env.sh / README section 1)
LMS_REV=67bfb0b9f835f772f3af2aa176745c6b6135373a
TCC_REV=8721ff06b64889e8cc50bb1c5e869c7f72b8303d
GPT2_SNAP=$HF_HUB/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
TC_SNAP=$HF_HUB/models--pchlenski--gpt2-transcoders/snapshots/4f3ca345572ce2d75018a60d10b4dca8bd899e99
Z_SNAP=$HF_HUB/models--ckkissane--attn-saes-gpt2-small-all-layers/snapshots/97e0da0e9182695e9813d15ea32184da83f5c874
OWT_REPO=apollo-research/Skylion007-openwebtext-tokenizer-gpt2
OWT_REV=f02886b54795e8acabceb637ca119f9ae8f19d3f
OWT_FILE=data/train-00000-of-00073.parquet
OWT_PARQUET=$HF_HUB/datasets--apollo-research--Skylion007-openwebtext-tokenizer-gpt2/snapshots/$OWT_REV/$OWT_FILE
HARDCODED_HUB=/home/daniel/.cache/huggingface/hub        # wcm_tcz.py, ge_dicts.py, dunefsky_checks.py read this path

# the settled .venv environment of the WCM side (run_pilot_a.sh ENVV, without the 16 GB card's DEEP_BS)
BASEENV="PYTHONPATH=src HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 ARENA_A_SCOPE=a1 FAMILY=gpt2-small-tcz \
SAE_PRELOAD=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WCM_VERSION=$WCMV DATA=$CORPUS STORES=$STO"
GE_SRC="source $GE_ENV/bin/ge_env.sh"
TC_SRC="source $TC_ENV/bin/tc_env.sh"

# ------------------------------------------------------------------------------------------------- helpers
say() {
  local line="[arena_a $(date +%H:%M:%S)] $*"
  echo "$line"
  [ "$DRY_RUN" = 1 ] || { mkdir -p "$LOGS"; echo "$line" >> "$LOGS/events.log"; }
}
q() { printf '%q' "$1"; }
snum() { local s=${1#P}; echo "${s%%[!0-9]*}"; }          # P4h -> 4 (sub-stages order with their parent)
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ] || [ -f "$HARD_FLAG" ]; }
mark() { [ "$DRY_RUN" = 1 ] || { mkdir -p "$STATE"; date '+%F %T' > "$STATE/$1.$2"; }; }
nlines() { if [ -f "$1" ]; then grep -c . "$1"; else echo 0; fi; }
count() { echo $#; }                                     # count GLOB...: nullglob, so no match = 0

run_cmd() {   # run_cmd LOG CMD: one command string through bash -c, appended to LOG; DRY_RUN prints it. Its environment
  #             carries AA_RUN_TAG (the hard stop finds and kills every process of the run by it); after the hard stop
  #             only P7's commands start
  local log=$1 cmd=$2
  if [ "$DRY_RUN" = 1 ]; then echo "    \$ $cmd   >> $log"; return 0; fi
  mkdir -p "$(dirname "$log")"
  if [ -f "$HARD_FLAG" ] && [ "$IN_P7" != 1 ]; then
    echo "[$(date '+%F %T')] HARD STOP: not started: \$ $cmd" >> "$log"; return 1
  fi
  echo "[$(date '+%F %T')] \$ $cmd" >> "$log"
  local t0 tag=$RUN_ID
  [ "$IN_P7" = 1 ] && tag=$RUN_ID.P7
  t0=$(date +%s)
  # stdin closed: callers loop over key lists on stdin. In the background + wait, so a signal to run_pod.sh runs its
  # trap at once (a foreground child would defer it until the child ends)
  AA_RUN_TAG=$tag bash -c "$cmd" >> "$log" 2>&1 < /dev/null &
  wait "$!"
  local rc=$?
  [ "$rc" = 0 ] || echo "[$(date '+%F %T')] exit $rc" >> "$log"
  # one row per command (log, start, end, rc, the command's script + args head): the per-step timings for A12
  printf '%s\t%s\t%s\t%s\t%s\n' "${log#"$LOGS"/}" "$t0" "$(date +%s)" "$rc" \
    "$({ echo "$cmd" | grep -oE '[A-Za-z0-9_]+\.py[^>|&]*' || echo "$cmd" | grep -oE '[A-Za-z0-9_]+\.sh[^>|&]*' \
       || echo "${cmd:0:80}"; } | tail -n 1 | cut -c1-200)" >> "$WORK/cmds.tsv"
  return "$rc"
}

shard_keys() {   # shard_keys LIST I: the keys on line n with (n - 1) % NG == I
  [ -f "$1" ] || return 0
  awk -v n="$NG" -v i="$2" 'NF && (NR - 1) % n == i' "$1"
}

keys_or_placeholder() {   # keys of shard I of LIST, or a placeholder in a dry run before LIST exists
  if [ -f "$1" ]; then shard_keys "$1" "$2"
  elif [ "$DRY_RUN" = 1 ] && [ -n "$KEYS" ]; then           # AA_KEYS: the list make_lists would write
    local t
    t=$(echo "$KEYS" | tr ',' '\n' | grep .)
    [ "$1" = "$LIST_TC" ] && t=$(echo "$t" | grep -F .tc.)
    echo "$t" | awk -v n="$NG" -v i="$2" 'NF && (NR - 1) % n == i'
  elif [ "$DRY_RUN" = 1 ]; then echo "<each key of $(basename "$1") line n with (n-1)%$NG==$2>"; fi
}

join_keys() { paste -sd, -; }

stage() {   # stage P<n> TITLE FUNC: STOP_AT gating, markers, steps.tsv (the re-cost reads it). rc 3 = STOP_AT
  local num=$1 title=$2 fn=$3 rc t0
  if [ "$(snum "$num")" -gt "$(snum "$STOP_AT")" ]; then say "$num $title: after STOP_AT=$STOP_AT; stopping here"; return 3; fi
  if [ "$(snum "$num")" -lt "$(snum "$FROM")" ]; then say "$num $title: before AA_FROM=$FROM, taken as done"; return 0; fi
  if [ -f "$STATE/$num.done" ] && [[ ",$REDO," == *",$num,"* ]]; then
    say "$num $title: done earlier, but in AA_REDO=$REDO: running it again (finished rows are skipped)"
  elif [ -f "$STATE/$num.done" ] && [ "$num" = P7 ]; then
    say "$num $title: done earlier; every run packs again (a resumed or stopped run adds results)"
  elif [ -f "$STATE/$num.done" ]; then
    say "$num $title: done earlier ($STATE/$num.done)"
    [ "$DRY_RUN" = 1 ] || return 0
    say "  (dry run: its commands are printed anyway)"
  fi
  # P7 runs after the time limit too (it packs what finished)
  if [ "$DRY_RUN" != 1 ] && late && [ "$num" != P7 ]; then say "$num $title: STOP_EPOCH reached; not started"; return 1; fi
  say "$num $title: START"
  mark "$num" start
  t0=$(date +%s)
  "$fn"
  rc=$?
  if [ "$DRY_RUN" != 1 ]; then printf '%s\t%s\t%s\t%s\n' "$num" "$t0" "$(date +%s)" "$rc" >> "$WORK/steps.tsv"; fi
  case $rc in
    0) mark "$num" done; say "$num $title: DONE ($(( $(date +%s) - t0 )) s)" ;;
    2) say "$num $title: WAITING FOR APPROVAL (run the commands above, then rerun)" ;;
    *) say "$num $title: FAILED (rc $rc; logs in $LOGS/$num)" ;;
  esac
  return "$rc"
}

run_workers() {   # run_workers FUNC: FUNC I on every GPU index in parallel (sequentially in a dry run); rc = failures
  local fn=$1 i pids=() fails=0
  for i in $(seq 0 $((NG - 1))); do
    if [ "$DRY_RUN" = 1 ]; then echo "  [GPU ${GL[$i]}]"; "$fn" "$i"; continue; fi
    "$fn" "$i" &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait "$i" || fails=$((fails + 1)); done
  [ "$fails" = 0 ] || say "  $fails of $NG workers reported a failure"
  return "$fails"
}

# ------------------------------------------------------------------------------------------------- P0 assets
clone_ok() { [ -d "$1/.git" ] && [ "$(git -C "$1" rev-parse HEAD 2> /dev/null)" = "$2" ]; }

assets_missing() {   # one line per missing pinned asset (read-only)
  clone_ok third_party/Language-Model-SAEs "$LMS_REV" || echo "clone third_party/Language-Model-SAEs @ ${LMS_REV:0:7}"
  clone_ok third_party/transcoder_circuits "$TCC_REV" || echo "clone third_party/transcoder_circuits @ ${TCC_REV:0:7}"
  [ -f "$GPT2_SNAP/model.safetensors" ] && [ -f "$GPT2_SNAP/tokenizer.json" ] || echo "GPT-2 @ 607a30d"
  [ "$(count "$TC_SNAP"/final_sparse_autoencoder_gpt2-small_blocks.*.ln2.hook_normalized_24576.pt)" = 12 ] \
    || echo "Dunefsky transcoders x12 @ 4f3ca345"
  [ "$(count "$Z_SNAP"/gpt2-small_L*_Hcat_z_*_cfg.json)" = 12 ] && [ "$(count "$Z_SNAP"/gpt2-small_L*_Hcat_z_*.pt)" = 12 ] \
    || echo "Kissane z-SAEs x12 (+ cfg) @ 97e0da0"
  if [ "$N_SEQ" = 0 ]; then [ -f "$DATA_SRC/corpus_manifest.json" ] || echo "corpus $DATA_SRC (082 S1)"
  else [ -f "$OWT_PARQUET" ] || echo "OWT $OWT_FILE @ ${OWT_REV:0:8} (082 S1 fetches it; else fetch_assets.sh owt)"; fi
  command -v "$PY310" > /dev/null 2>&1 || [ -f "$GE_ENV/bin/ge_env.sh" ] || echo "$PY310 (for setup_env.sh)"
}

print_p0_commands() {   # the downloads, for Daniel to approve (never run here)
  local r
  r=$(q "$ROOT")
  echo "  --- P0 commands for Daniel to approve (pod shell; HF_TOKEN only from this shell's environment, never a file) ---"
  echo "    \$ git clone https://github.com/OpenMOSS/Language-Model-SAEs.git third_party/Language-Model-SAEs && git -C third_party/Language-Model-SAEs checkout $LMS_REV   # ~122 MB"
  echo "    \$ git clone https://github.com/jacobdunefsky/transcoder_circuits.git third_party/transcoder_circuits && git -C third_party/transcoder_circuits checkout $TCC_REV   # ~29 MB"
  echo "    \$ ROOT=$r PY=$PY bash $A/fetch_assets.sh pod   # GPT-2 0.55 GB (cached already if the 082 bootstrap fetched it) + transcoders 1.81 GB + Kissane 2.12 GB"
  [ "$N_SEQ" = 0 ] || echo "    \$ ROOT=$r PY=$PY bash $A/fetch_assets.sh owt   # only if the 082 S1 corpus step has not cached it: 243 MB"
  echo "    \$ command -v $PY310 || (apt-get update && apt-get install -y python3.10 python3.10-venv)   # only if missing; check the template first"
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
  # the HF cache path that wcm_tcz.py / ge_dicts.py / dunefsky_checks.py hard-code (no download; a symlink)
  if [ "$HF_HUB" != "$HARDCODED_HUB" ] && [ ! -e "$HARDCODED_HUB" ]; then
    run_cmd "$LOGS/P0/hf_link.log" "mkdir -p $(dirname "$HARDCODED_HUB") && ln -s $(q "$HF_HUB") $HARDCODED_HUB" || return 1
  fi
  # 035's LN-folded transcoders (the WCM site model reads them; not in git): regenerated offline, asserted equivalent
  if [ "$DRY_RUN" = 1 ] || [ ! -f experiments/035-transcoder-compare/transcoders_ct/meta.json ]; then
    run_cmd "$LOGS/P0/convert_transcoders.log" "CUDA_VISIBLE_DEVICES=${GL[0]} HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $PY -X utf8 experiments/035-transcoder-compare/convert_transcoders.py" \
      || return 1
  fi
}

# ------------------------------------------------------------------------------------------------- P1 envs
p1_envs() {
  if [ ! -f "$GE_ENV/bin/ge_env.sh" ] || [ ! -f "$TC_ENV/bin/tc_env.sh" ] || [ "$DRY_RUN" = 1 ]; then
    echo "  --- P1 command for Daniel to approve (pip downloads, ~3.9 GB: torch 2.3.1+cu121 / 2.2.0+cu121 + the pins) ---"
    echo "    \$ ROOT=$(q "$ROOT") TORCH_INDEX=cu121 PY310=$PY310 GE_ENV=$GE_ENV TC_ENV=$TC_ENV bash $A/setup_env.sh both 2>&1 | tee $LOGS/P1/setup_env.log"
    echo "  ---"
    if [ "$DRY_RUN" != 1 ]; then say "  ge107 / tc107 not built yet"; return 2; fi
  fi
  local L=$LOGS/P1/gate.log
  run_cmd "$L" "$GE_ENV/bin/pip check && $TC_ENV/bin/pip check" || return 1
  if [ "$BDEV" = cuda ]; then
    run_cmd "$L" "$GE_SRC && CUDA_VISIBLE_DEVICES=${GL[0]} python -I -B -c 'import torch; assert torch.cuda.is_available(); print(\"ge107 torch\", torch.__version__, torch.cuda.get_device_name(0))'" || return 1
    run_cmd "$L" "$TC_SRC && CUDA_VISIBLE_DEVICES=${GL[0]} python -I -B -c 'import torch; assert torch.cuda.is_available(); print(\"tc107 torch\", torch.__version__, torch.cuda.get_device_name(0))'" || return 1
  fi
  run_cmd "$L" "$GE_SRC && CUDA_VISIBLE_DEVICES= python -B -m pytest -q -p no:cacheprovider third_party/Language-Model-SAEs/tests/intergration/test_attributor.py" || return 1
}

# ------------------------------------------------------------------------------------------------- P2 data
p2_data() {
  if [ "$N_SEQ" != 0 ]; then
    if [ "$DRY_RUN" != 1 ] && [ -f "$CORPUS/corpus_manifest.json" ]; then say "  corpus $CORPUS exists (manifest)"
    elif [ "$DRY_RUN" != 1 ] && [ "$(count "$CORPUS"/*.npy)" -gt 0 ]; then
      say "  $CORPUS holds shards but no manifest (an interrupted run): move it away, then rerun"; return 1
    else
      run_cmd "$LOGS/P2/corpus.log" "HF_HUB_OFFLINE=1 PYTHONPATH=src $PY -X utf8 $D69/prepare_corpus.py --family gpt2-small --out $CORPUS --hf-dataset $OWT_REPO --hf-revision $OWT_REV --hf-pattern $OWT_FILE --token-field input_ids --max-sequences $N_SEQ --source-name 'OpenWebText (Skylion007), apollo-research gpt2-tokenised, file 0 of 73, first $N_SEQ windows (arena A)' --licence 'UNCONFIRMED: Skylion007/openwebtext card CC0-1.0; the apollo copy has no card licence'" \
        || return 1
    fi
  fi
  if [ "$DRY_RUN" != 1 ] && [ -f "$STO/neg_ctx.pt" ] && [ -f "$STO/data_pass_manifest.json" ]; then
    say "  data pass $STO exists"; return 0
  fi
  run_cmd "$LOGS/P2/datapass.log" "env $BASEENV CUDA_VISIBLE_DEVICES=${GL[0]} $PY -X utf8 -u $A/data_pass_tcz.py --data $CORPUS --out $STO --batch-size $DP_BATCH --dtype float32"
}

# ------------------------------------------------------------------------------------------------- P3 targets
p3_targets() {
  if [ "$DRY_RUN" = 1 ] || [ "$(nlines "$TGT/targets.txt")" -lt "$N_TARGETS" ] || [ ! -f "$TGT/MANIFEST.json" ]; then
    if [ ! -s "$EXCLUDE" ]; then
      say "  EXCLUDE file $EXCLUDE is missing or empty: the tuning dev targets could become production targets"
      [ "$DRY_RUN" = 1 ] || return 1
    fi
    run_cmd "$LOGS/P3/make_targets.log" "env $BASEENV CUDA_VISIBLE_DEVICES= OUT=$TGT EXCLUDE=$EXCLUDE $PY -X utf8 -I -B $A/make_targets.py" || return 1
  fi
  if [ "$DRY_RUN" != 1 ] && [ -s "$EXCLUDE" ]; then           # belt and braces: no dev target among the picks
    local bad
    bad=$(grep -xF -f <(tr -s ' \t' '\n\n' < "$EXCLUDE" | grep .) "$TGT/targets.txt")
    [ -z "$bad" ] || { say "  P3: tuning dev targets among the picks: $(echo "$bad" | paste -sd' ' -)"; return 1; }
  fi
  make_lists
}

make_lists() {   # the active lists downstream of P3 (AA_LIMIT cuts them; the tc subset for Dunefsky and WCM a2)
  if [ "$DRY_RUN" = 1 ]; then
    if [ -n "$KEYS" ]; then echo "    \$ echo $KEYS | tr , '\\n' > $LIST"
    elif [ -n "$LIMIT" ]; then echo "    \$ head -n $LIMIT $TGT/targets.txt > $LIST"; fi
    echo "    \$ grep -F .tc. $LIST > $LIST_TC"
    return 0
  fi
  [ -f "$TGT/targets.txt" ] || return 1
  mkdir -p "$WORK"
  if [ -n "$KEYS" ]; then
    local k
    for k in ${KEYS//,/ }; do                                # every key must be one of the targets dir's picks
      grep -qxF "$k" "$TGT/targets.txt" || { say "  AA_KEYS: $k is not in $TGT/targets.txt"; return 1; }
    done
    echo "$KEYS" | tr ',' '\n' | grep . > "$LIST" || return 1
  elif [ -n "$LIMIT" ]; then head -n "$LIMIT" "$TGT/targets.txt" > "$LIST" || return 1; fi
  grep -F .tc. "$LIST" > "$LIST_TC"
  say "  active lists: $(nlines "$LIST") targets, $(nlines "$LIST_TC") tc"
}

# ------------------------------------------------------------------------------------------------- P4 baselines
ge_cmd() {   # ge_cmd GPU KEY OUT [extra args]
  local g=$1 key=$2 out=$3; shift 3
  echo "$GE_SRC && CUDA_VISIBLE_DEVICES=$g python -I -B $A/run_ge.py --targets $TGT --out $out --keys $key --device $BDEV --max-rss-gb $GE_RSS --n-tau $GE_NTAU${GE_K:+ --k $GE_K} $*"
}

df_cmd() {   # df_cmd GPU KEYS OUT [extra args] (their code picks cuda when the env's torch sees a GPU)
  local g=$1 keys=$2 out=$3; shift 3
  echo "$TC_SRC && CUDA_VISIBLE_DEVICES=$g python -I -B $A/run_dunefsky.py --targets $TGT --out $out --keys $keys $*"
}

p4_canary() {   # one small circuit of each runner on the first GPU before 8 workers start (both runners were CPU-only so far)
  local k1 kt
  k1=$(keys_or_placeholder "$LIST" 0 | head -n 1); kt=$(keys_or_placeholder "$LIST_TC" 0 | head -n 1)   # = line 1
  case $k1 in ""|"<"*) k1="<first key of $(basename "$LIST")>" ;; esac
  case $kt in ""|"<"*) kt="<first key of $(basename "$LIST_TC")>" ;; esac
  local L=$LOGS/P4/canary.log ok=1
  if [ "$DRY_RUN" = 1 ] || [ ! -f "$CAN/ge/circuits/$k1/p0.pt" ]; then
    run_cmd "$L" "$(ge_cmd "${GL[0]}" "$k1" "$CAN/ge" --k 1 --n-tau 3)" || ok=0
    [ "$DRY_RUN" = 1 ] || [ -f "$CAN/ge/circuits/$k1/p0.pt" ] || ok=0
  fi
  if [ "$DRY_RUN" = 1 ] || [ ! -f "$CAN/dunefsky/circuits/$kt/p0.b5.i5.pt" ]; then
    run_cmd "$L" "$(df_cmd "${GL[0]}" "$kt" "$CAN/dunefsky" --k 1 --branches 5)" || ok=0
    [ "$DRY_RUN" = 1 ] || [ -f "$CAN/dunefsky/circuits/$kt/p0.b5.i5.pt" ] || ok=0
  fi
  [ "$ok" = 1 ] || { say "  GPU canary FAILED ($L): no worker started"; return 1; }
}

p4_worker() {   # p4_worker I: Ge on the worker's targets (one process per target), then Dunefsky on its tc targets
  local i=$1 g=${GL[$1]} key keys rc=0
  local L=$LOGS/P4/gpu$g.log
  while read -r key; do
    [ -n "$key" ] || continue
    if [ "$DRY_RUN" != 1 ] && late; then echo "STOP_EPOCH: no new target" >> "$L"; break; fi
    run_cmd "$L" "$(ge_cmd "$g" "$key" "$GE_OUT")" || rc=1
  done < <(keys_or_placeholder "$LIST" "$i")
  keys=$(keys_or_placeholder "$LIST_TC" "$i" | join_keys)
  if [ -n "$keys" ] && { [ "$DRY_RUN" = 1 ] || ! late; }; then
    run_cmd "$L" "$(df_cmd "$g" "$keys" "$DF_OUT" --branches $DF_BRANCHES --iters 5${DF_K:+ --k $DF_K})" || rc=1
    if [ "$DF_EXT" = 1 ]; then                           # LABEL: "beyond published range" (num_branches > 60)
      [ "$DRY_RUN" = 1 ] && echo "    # Dunefsky extended knob, LABEL 'beyond published range' (their runs stop at 60 branches):"
      run_cmd "$L" "$(df_cmd "$g" "$keys" "$DF_OUT" --k "$DF_EXT_K" --branches "$DF_EXT_BRANCHES" --iters 5)" || rc=1
    fi
  fi
  return "$rc"
}

p4_complete() {   # every expected circuit file is there (errors are not saved, so they show up here)
  [ "$DRY_RUN" = 1 ] && { echo "    \$ (check) $GE_OUT: targets x ${GE_K:-$K} circuits; $DF_OUT: tc targets x ${DF_K:-$K} x $N_DF_PRIMARY branch values$([ "$DF_EXT" = 1 ] && echo " + $DF_EXT_K x ext")"; return 0; }
  local key ng=0 nd=0 ne=0 kge=${GE_K:-$K} kdf=${DF_K:-$K} n_ext
  n_ext=$(echo "$DF_EXT_BRANCHES" | tr ',' '\n' | grep -c .)
  while read -r key; do ng=$((ng + $(count "$GE_OUT/circuits/$key"/p*.pt))); done < "$LIST"
  while read -r key; do
    nd=$((nd + $(count "$DF_OUT/circuits/$key"/p*.b*.i5.pt)))
  done < "$LIST_TC"
  local want_g=$(( $(nlines "$LIST") * kge )) want_d
  want_d=$(( $(nlines "$LIST_TC") * (kdf * N_DF_PRIMARY + (DF_EXT == 1 ? DF_EXT_K * n_ext : 0)) ))
  say "  P4 circuits: Ge $ng / $want_g, Dunefsky $nd / $want_d (status rows: $GE_OUT/status.jsonl, $DF_OUT/status.jsonl)"
  ne=$(cat "$GE_OUT/status.jsonl" "$DF_OUT/status.jsonl" 2> /dev/null | grep -c '"status": "error"')
  [ "$ne" = 0 ] || say "  P4: $ne error rows in the status files (a rerun retries them)"
  [ "$ng" -ge "$want_g" ] && [ "$nd" -ge "$want_d" ]
}

p4_baselines() {
  p4_canary || return 1
  run_workers p4_worker
  local rc=$?
  [ "$rc" = 0 ] || rc=1                                   # a failed-worker COUNT of 2 / 3 must not read as stage()'s
  p4_complete || return 1                                 # 2 (approval) / 3 (STOP_AT)
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- P4h held baselines
p4h_worker() {   # p4h_worker I: Ge on the worker's targets' held prompts (one process per target), then Dunefsky (the
  #                published branch grid only) on its tc targets' held prompts. --targets / --k come last (argparse: last wins)
  local i=$1 g=${GL[$1]} key keys rc=0
  local L=$LOGS/P4h/gpu$g.log
  while read -r key; do
    [ -n "$key" ] || continue
    if [ "$DRY_RUN" != 1 ] && late; then echo "STOP_EPOCH: no new target" >> "$L"; break; fi
    run_cmd "$L" "$(ge_cmd "$g" "$key" "$GE_HELD" --targets "$HELD" --k "$HELD_K")" || rc=1
  done < <(keys_or_placeholder "$LIST" "$i")
  keys=$(keys_or_placeholder "$LIST_TC" "$i" | join_keys)
  if [ -n "$keys" ] && { [ "$DRY_RUN" = 1 ] || ! late; }; then
    run_cmd "$L" "$(df_cmd "$g" "$keys" "$DF_HELD" --branches $DF_BRANCHES --iters 5 --targets "$HELD" --k "$HELD_K")" || rc=1
  fi
  return "$rc"
}

p4h_held() {
  local all_keys
  if [ "$DRY_RUN" = 1 ]; then
    all_keys=$( { [ -n "$KEYS" ] && echo "$KEYS"; } || echo "<every key of $(basename "$LIST")>")
  else all_keys=$(join_keys < "$LIST"); fi
  if [ "$DRY_RUN" = 1 ] || [ "$(nlines "$HELD/prompts.jsonl")" -lt "$(nlines "$LIST")" ]; then
    run_cmd "$LOGS/P4h/emit.log" "env $BASEENV CUDA_VISIBLE_DEVICES= ARENA_A_CTX_DIRS=$TGT/ctx $PY -X utf8 -u $A/score_perprompt.py --emit-heldout-prompts $HELD --keys $all_keys --n-prompts $HELD_K" \
      || return 1
  fi
  run_workers p4h_worker
  local rc=$?
  [ "$rc" = 0 ] || rc=1                                   # a failed-worker count, not stage()'s 2 / 3
  [ "$DRY_RUN" = 1 ] && { echo "    \$ (check) $GE_HELD: targets x $HELD_K circuits; $DF_HELD: tc targets x $HELD_K x $N_DF_PRIMARY"; return "$rc"; }
  local key ng=0 nd=0 want_g want_d
  while read -r key; do ng=$((ng + $(count "$GE_HELD/circuits/$key"/p*.pt))); done < "$LIST"
  while read -r key; do nd=$((nd + $(count "$DF_HELD/circuits/$key"/p*.b*.i5.pt))); done < "$LIST_TC"
  want_g=$(( $(nlines "$LIST") * HELD_K )); want_d=$(( $(nlines "$LIST_TC") * HELD_K * N_DF_PRIMARY ))
  say "  P4h circuits: Ge held $ng / $want_g, Dunefsky held $nd / $want_d"
  [ "$ng" -ge "$want_g" ] && [ "$nd" -ge "$want_d" ] || return 1
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- P5 WCM
arm_list() { if [ "$1" = a2 ]; then echo "$LIST_TC"; else echo "$LIST"; fi; }

fits_done() {   # fits_done OUT LIST: an ok / skip status row for every target (as run_pilot_a.sh)
  local n ok f=("$1"/main/status.shard*.jsonl)
  n=$(nlines "$2")
  [ ${#f[@]} -gt 0 ] || return 1
  ok=$(cat "${f[@]}" | grep -c -E '"ok": true|"skip"')
  [ "$n" -gt 0 ] && [ "$ok" -ge "$n" ]
}

tune_env() {   # TUNE_KV = the CHOSEN.env assignments (whitelisted keys, plain values); rc 1 = refuse P5
  local line key val rec kv=() steps_seen=""
  if [ ! -f "$TUNE_ENV" ]; then
    if [ "$ALLOW_UNTUNED" = 1 ]; then say "  no tuning choice ($TUNE_ENV): AA_ALLOW_UNTUNED=1, so plain v4"; TUNE_KV=""; rec=UNTUNED
    else
      say "  P5 REFUSED: the WCM tuning choice $TUNE_ENV is missing (TUNING.md: tune_summary.py writes it; copy it to"
      say "  the pod). Set AA_TUNE_ENV=<file>, or AA_ALLOW_UNTUNED=1 to fit plain v4 on purpose"
      [ "$DRY_RUN" = 1 ] && { say "  (dry run: continuing as if AA_ALLOW_UNTUNED=1)"; TUNE_KV=""; return 0; }
      return 1
    fi
  else
    while IFS= read -r line || [ -n "$line" ]; do
      line=${line%%#*}; line=$(echo "$line" | tr -d ' \t\r')
      [ -n "$line" ] || continue
      key=${line%%=*}; val=${line#*=}
      case $key in AA_RANK_W|NEC_W|STEPS|AA_TUNE_CONFIG) ;;
        *) say "  P5 REFUSED: $TUNE_ENV sets $key (allowed: AA_RANK_W NEC_W STEPS AA_TUNE_CONFIG)"; return 1 ;; esac
      [[ "$line" == *=* ]] && [[ "$val" =~ ^[A-Za-z0-9_.+-]*$ ]] \
        || { say "  P5 REFUSED: bad line in $TUNE_ENV: $line"; return 1; }
      if [ "$key" = STEPS ] && [ -n "$STEPS" ]; then
        say "  CHOSEN.env STEPS=$val overridden by AA_STEPS=$STEPS"; continue
      fi
      [ "$key" = STEPS ] && steps_seen=$val
      kv+=("$key=$val")
    done < "$TUNE_ENV"
    TUNE_KV=${kv[*]}
    rec="$TUNE_KV"
    say "  WCM tuning choice ($TUNE_ENV): ${TUNE_KV:-<none>}${steps_seen:+ (STEPS $steps_seen: honoured)}"
  fi
  [ -n "$STEPS" ] && rec="$rec AA_STEPS=$STEPS"
  [ "$DRY_RUN" = 1 ] && return 0
  mkdir -p "$WCM_OUT"
  if [ -f "$WCM_OUT/TUNE_ENV" ] && [ "$(cat "$WCM_OUT/TUNE_ENV")" != "$rec" ]; then
    say "  P5 REFUSED: $WCM_OUT was fitted under '$(cat "$WCM_OUT/TUNE_ENV")', this run is '$rec' (move it away to refit)"
    return 1
  fi
  echo "$rec" > "$WCM_OUT/TUNE_ENV"
}

p5_wcm() {
  local scope lam name O list fails=0 gu
  tune_env || return 1
  for scope in $SCOPES; do
    case $scope in a1|a2) ;; *) say "  WCM: unknown scope $scope (a1 / a2)"; return 1 ;; esac
    list=$(arm_list "$scope")
    if [ "$DRY_RUN" != 1 ] && [ "$(nlines "$list")" = 0 ]; then say "  WCM $scope: no targets in $list; skipped"; continue; fi
    for lam in ${LAMS//,/ }; do
      name=${scope}_l$lam; O=$WCM_OUT/$name
      if [ "$DRY_RUN" != 1 ] && fits_done "$O" "$list"; then say "  WCM $name: every target fitted"; continue; fi
      if [ "$DRY_RUN" != 1 ] && late; then say "  WCM $name: STOP_EPOCH reached"; fails=$((fails + 1)); continue; fi
      say "  WCM $name on GPUs $GPUS (P=$WCM_P)"
      run_cmd "$LOGS/P5/$name.launch.log" "env $BASEENV GPUS=$GPUS P=$WCM_P STAGGER=$STAGGER MERGE=0 LOGDIR=$LOGS/P5/$name PY=$PY SCRIPT=$A/driver_tcz.py ARENA_A_SCOPE=$scope OUT=$O CTX_DIR=$TGT/ctx TARGETS=$list MODE=main LAM=$lam ORDER=list SKIP_SPEC=1 CURVE_STRIDE=20${TUNE_KV:+ $TUNE_KV}${STEPS:+ STEPS=$STEPS} bash $D62/launch.sh" \
        || fails=$((fails + 1))
      if [ "$DRY_RUN" != 1 ]; then
        gu=$(cat "$LOGS/P5/$name"/*.log 2> /dev/null | grep -c "GIVING UP")
        [ "$gu" = 0 ] || { say "  WCM $name: $gu shard(s) gave up (see $LOGS/P5/$name)"; fails=$((fails + 1)); continue; }
        fits_done "$O" "$list" || { say "  WCM $name: not every target has an ok row"; fails=$((fails + 1)); }
      fi
    done
  done
  [ "$fails" = 0 ]
}

# ------------------------------------------------------------------------------------------------- P6 scoring
wcm_spec() {   # wcm_spec SCOPE: label=dir,... for score_arena_a.py --wcm (label l<lambda>)
  local lam out=()
  for lam in ${LAMS//,/ }; do out+=("l$lam=$WCM_OUT/${1}_l$lam"); done
  (IFS=,; echo "${out[*]}")
}

score_cmd() {   # score_cmd GPU SCOPE OUTDIR [args]: the scorer under the settled env
  local g=$1 scope=$2 out=$3; shift 3
  local stop=""
  [ "$STOP_EPOCH" != 99999999999 ] && stop=" --stop-epoch $STOP_EPOCH"
  echo "env $BASEENV CUDA_VISIBLE_DEVICES=$g ARENA_A_SCOPE=$scope ARENA_A_CTX_DIRS=$TGT/ctx OUT=$out $PY -X utf8 -u $A/score_arena_a.py --scope $scope --out $out $*$stop"
}

p6_worker() {   # p6_worker I: scope a1 (Ge + WCM a1), then a2 (Dunefsky + WCM a2), on the worker's targets
  local i=$1 g=${GL[$1]} keys rc=0
  local L=$LOGS/P6/gpu$g.log
  local pool="--donor --donor-pool $SC/random_pool.pt --donor-records ${AA_DONOR_RECORDS:-$TGT/ctx} --selftest"
  keys=$(keys_or_placeholder "$LIST" "$i" | join_keys)
  if [ -n "$keys" ] && [[ " $SCOPES " == *" a1 "* ]]; then
    run_cmd "$L" "$(score_cmd "$g" a1 "$SC/a1.s$i" --ge "$GE_OUT" --wcm "$(wcm_spec a1)" --keys "$keys" $pool)" || rc=1
  fi
  keys=$(keys_or_placeholder "$LIST_TC" "$i" | join_keys)
  if [ -n "$keys" ] && [[ " $SCOPES " == *" a2 "* ]]; then
    run_cmd "$L" "$(score_cmd "$g" a2 "$SC/a2.s$i" --dunefsky "$DF_OUT" --wcm "$(wcm_spec a2)" --keys "$keys" $pool)" || rc=1
  fi
  return "$rc"
}

p6_recovery_worker() {   # p6_recovery_worker I: Ge's leaf-sum recovery (ge107) on job shard I
  local i=$1 g=${GL[$1]}
  local J=$SC/jobs/ge_jobs.s$i.jsonl
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ awk -v n=$NG -v i=$i 'NR % n == i' $SC/jobs/ge_jobs.a1.jsonl $SC/jobs/ge_jobs.a2.jsonl > $J"
  else
    mkdir -p "$SC/jobs" "$SC/all"
    cat "$SC/jobs"/ge_jobs.a*.jsonl 2> /dev/null | awk -v n="$NG" -v i="$i" 'NR % n == i' > "$J.tmp" && mv "$J.tmp" "$J"
    [ -s "$J" ] || return 0
  fi
  run_cmd "$LOGS/P6/recovery.gpu$g.log" "$GE_SRC && CUDA_VISIBLE_DEVICES=$g python -I -B $A/ge_recovery_tcz.py --jobs $J --out $SC/all/ge_recovery.s$i.jsonl --check --device $BDEV --max-rss-gb $GE_RSS"
}

p6_score() {
  local rc=0 all_keys tc_keys
  [ -n "${AA_DONOR_RECORDS:-}" ] && say "  SMOKE: R donor pool filtered against --donor-records $AA_DONOR_RECORDS (AA_DONOR_RECORDS), not $TGT/ctx"
  run_workers p6_worker || rc=1
  if [ "$DRY_RUN" = 1 ]; then all_keys="<every key of $(basename "$LIST")>"; tc_keys="<every key of $(basename "$LIST_TC")>"
  else all_keys=$(join_keys < "$LIST"); tc_keys=$(join_keys < "$LIST_TC"); fi
  # the A11 jobs for WCM and Dunefsky circuits (no model; CPU)
  if [[ " $SCOPES " == *" a1 "* ]]; then
    run_cmd "$LOGS/P6/emit.log" "$(score_cmd "" a1 "$SC/jobs" --wcm "$(wcm_spec a1)" --keys "$all_keys" --emit-ge-jobs)" || rc=1
  fi
  if [[ " $SCOPES " == *" a2 "* ]]; then
    run_cmd "$LOGS/P6/emit.log" "$(score_cmd "" a2 "$SC/jobs" --dunefsky "$DF_OUT" --wcm "$(wcm_spec a2)" --keys "$tc_keys" --emit-ge-jobs)" || rc=1
  fi
  run_workers p6_recovery_worker || rc=1
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- P6b per-prompt
pp_cmd() {   # pp_cmd GPU SCOPE OUTDIR [args]: score_perprompt.py under the scorer's env
  local g=$1 scope=$2 out=$3; shift 3
  local stop=""
  [ "$STOP_EPOCH" != 99999999999 ] && stop=" --stop-epoch $STOP_EPOCH"
  echo "env $BASEENV CUDA_VISIBLE_DEVICES=$g ARENA_A_SCOPE=$scope ARENA_A_CTX_DIRS=$TGT/ctx OUT=$out $PY -X utf8 -u $A/score_perprompt.py --scope $scope --out $out $*$stop"
}

p6b_worker() {   # p6b_worker I: a1 (WCM a1 + Ge held), then a2 (WCM a2 + Dunefsky held), train + held prompts
  local i=$1 g=${GL[$1]} keys rc=0
  local L=$LOGS/P6b/gpu$g.log
  local common="--sets train,held --n-prompts $PP_PROMPTS --topk-arms $PP_TOPK$([ "$PP_CHECK" = 1 ] && echo ' --check')"
  keys=$(keys_or_placeholder "$LIST" "$i" | join_keys)
  if [ -n "$keys" ] && [[ " $SCOPES " == *" a1 "* ]]; then
    run_cmd "$L" "$(pp_cmd "$g" a1 "$PP/a1.s$i" --wcm "$(wcm_spec a1)" --ge-held "$GE_HELD" --tau-stride "$PP_TAU_STRIDE" --keys "$keys" $common)" || rc=1
  fi
  keys=$(keys_or_placeholder "$LIST_TC" "$i" | join_keys)
  if [ -n "$keys" ] && [[ " $SCOPES " == *" a2 "* ]]; then
    run_cmd "$L" "$(pp_cmd "$g" a2 "$PP/a2.s$i" --wcm "$(wcm_spec a2)" --dunefsky-held "$DF_HELD" --keys "$keys" $common)" || rc=1
    # Ge recovery jobs of Dunefsky's held-out circuits (score_arena_a's emitter, uid "pp|..."; no model, CPU)
    run_cmd "$L" "env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 ARENA_A_BOS=1 ARENA_A_SCOPE=a2 CUDA_VISIBLE_DEVICES= ARENA_A_CTX_DIRS=$TGT/ctx $PY -X utf8 -B $A/ge_jobs_heldout.py --dunefsky-held $DF_HELD --out $PP/a2.s$i/ge_jobs.dfheld.jsonl --keys $keys" || rc=1
  fi
  return "$rc"
}

p6b_recovery_worker() {   # Ge recovery of shard I's per-prompt units (its own a1 / a2 ge_jobs.pp.jsonl: a stable
  #                           job -> worker assignment, so a resumed run never re-runs a job in another shard)
  local i=$1 g=${GL[$1]} s jobs=""
  for s in a1/ge_jobs.pp a2/ge_jobs.pp a2/ge_jobs.dfheld; do      # WCM per-prompt units + Dunefsky held circuits
    local f=$PP/${s%%/*}.s$i/${s#*/}.jsonl
    if [ "$DRY_RUN" = 1 ] || [ -s "$f" ]; then jobs="$jobs --jobs $f"; fi
  done
  [ -n "$jobs" ] || return 0
  run_cmd "$LOGS/P6b/recovery.gpu$g.log" "$GE_SRC && CUDA_VISIBLE_DEVICES=$g python -I -B $A/ge_recovery_tcz.py$jobs --out $PP/all/ge_recovery.pp.s$i.jsonl --check --device $BDEV --max-rss-gb $GE_RSS"
}

p6b_perprompt() {
  local rc=0 f
  run_workers p6b_worker || rc=1
  # merge the shards (rows, jobs, checks) and the P6 score rows (the train-prompt baselines = their home rows)
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ cat $PP/a{1,2}.s*/{rows,ge_jobs.pp,ge_jobs.dfheld,check}.jsonl > $PP/all/<same name>.jsonl; cat $SC/a{1,2}.s*/rows.jsonl > $PP/all/baseline_rows.jsonl"
  else
    mkdir -p "$PP/all"
    for f in rows ge_jobs.pp ge_jobs.dfheld check; do
      local parts=("$PP"/a1.s*/"$f".jsonl "$PP"/a2.s*/"$f".jsonl)
      if [ ${#parts[@]} -gt 0 ]; then cat "${parts[@]}" > "$PP/all/$f.jsonl.tmp" && mv "$PP/all/$f.jsonl.tmp" "$PP/all/$f.jsonl"
      else : > "$PP/all/$f.jsonl"; fi
    done
    local br=("$SC"/a1.s*/rows.jsonl "$SC"/a2.s*/rows.jsonl)
    if [ ${#br[@]} -gt 0 ]; then cat "${br[@]}" > "$PP/all/baseline_rows.jsonl"; else : > "$PP/all/baseline_rows.jsonl"; fi
  fi
  [ "$DRY_RUN" = 1 ] || mkdir -p "$PP/all"
  run_workers p6b_recovery_worker || rc=1
  run_cmd "$LOGS/P6b/summary.log" "env $BASEENV CUDA_VISIBLE_DEVICES= $PY -X utf8 -u $A/score_perprompt.py --summary --out $PP/all --baseline-rows $PP/all/baseline_rows.jsonl" || rc=1
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- P7 pack
p7_pack() {   # tolerates partial outputs (a stopped run): missing parts are left out, a failed --summary is logged
  local f d rc=0
  wait_hard_kill
  IN_P7=1                                                  # its commands start (and are tagged apart) after a hard stop
  if [ "$DRY_RUN" = 1 ]; then
    echo "    \$ cat $SC/a{1,2}.s*/{rows,selftest,donor,donor_raw}.jsonl > $SC/all/<same name>.jsonl   # merge the score shards"
  else
    mkdir -p "$SC/all"
    for f in rows selftest donor donor_raw; do
      local parts=("$SC"/a1.s*/"$f".jsonl "$SC"/a2.s*/"$f".jsonl)
      if [ ${#parts[@]} -gt 0 ]; then cat "${parts[@]}" > "$SC/all/$f.jsonl.tmp" && mv "$SC/all/$f.jsonl.tmp" "$SC/all/$f.jsonl"; fi
    done
  fi
  run_cmd "$LOGS/P7/summary.log" "env $BASEENV CUDA_VISIBLE_DEVICES= $PY -X utf8 -u $A/score_arena_a.py --summary --out $SC/all" \
    || { say "  summary FAILED (packing anyway)"; rc=1; }
  [ "$PACK" = 1 ] || { say "  AA_PACK=0: no tgz / RESULTS copy"; return "$rc"; }
  local res=$TAG-arena-a-results.tgz sto=$TAG-arena-a-stores.tgz
  local big=() files=$res cps
  if [ "$DRY_RUN" = 1 ] || [ -d "$STO" ]; then big+=("$STO"); fi
  if [ "$N_SEQ" != 0 ] && { [ "$DRY_RUN" = 1 ] || [ -d "$CORPUS" ]; }; then big+=("$CORPUS"); fi
  if [ ${#big[@]} -gt 0 ]; then files="$res $sto"; else say "  no data pass / corpus yet: the results tgz only"; fi
  # results tgz: a missing path is skipped (--ignore-failed-read; a stopped run), and tar's rc 1 ("file changed as we
  # read it": the GPU sampler's csv, the event log) is kept; rc 2 is a failure
  local tar_res="{ tar -czf $CKPT/$res --ignore-failed-read --exclude=$STO --exclude=$WORK/corpus $WORK $A/env_freeze_ge.txt $A/env_freeze_tc.txt experiments/035-transcoder-compare/transcoders_ct/meta.json; [ \$? -le 1 ]; }"
  local tar_sto=""
  [ ${#big[@]} -gt 0 ] && tar_sto=" && tar -czf $CKPT/$sto ${big[*]}"
  run_cmd "$LOGS/P7/pack.log" "mkdir -p $CKPT && $tar_res$tar_sto && (cd $CKPT && md5sum $files > $TAG-arena-a.md5)" \
    || return 1
  cps=$(for f in $files; do printf '%s ' "$CKPT/$f"; done)
  run_cmd "$LOGS/P7/pack.log" "mkdir -p $RESULTS && cp --no-preserve=mode,ownership $cps$CKPT/$TAG-arena-a.md5 $RESULTS/ ; (cd $RESULTS && md5sum -c $TAG-arena-a.md5)" \
    || { say "  copy / md5 check on $RESULTS failed"; return 1; }
  [ "$DRY_RUN" = 1 ] || say "  verified on $RESULTS: $files"
  return "$rc"
}

# ------------------------------------------------------------------------------------------------- time limit
tagged() {   # tagged TAG: the pids whose environment holds AA_RUN_TAG=TAG exactly (every command of the run and its
  #            children; run_pod.sh itself, the sampler and the watchdog are not tagged)
  grep -lzx "AA_RUN_TAG=$1" /proc/[0-9]*/environ 2> /dev/null | awk -F/ '{ print $3 }' | grep -vx "$BASHPID"
}

kill_tagged() {   # kill_tagged WHY TAG...: SIGTERM every tagged process (two passes: a restart loop may spawn one), SIGKILL
  #                 what is left after AA_KILL_GRACE s; every process is logged (ps line) to LOGS/hard_stop.log
  local why=$1 t p n=0 nk=0 w=0 left pass
  shift
  local K=$LOGS/hard_stop.log
  mkdir -p "$LOGS"
  for pass in 1 2; do
    for t in "$@"; do
      for p in $(tagged "$t"); do
        { printf '[%s] %s: SIGTERM ' "$(date '+%F %T')" "$why"; ps -o pid=,ppid=,etime=,args= -p "$p" 2> /dev/null | cut -c1-400; } >> "$K"
        kill -TERM "$p" 2> /dev/null && n=$((n + 1))
      done
    done
    sleep 2
  done
  while [ "$w" -lt "$KILL_GRACE" ]; do
    left=$(for t in "$@"; do tagged "$t"; done)
    [ -n "$left" ] || break
    sleep 1; w=$((w + 1))
  done
  for pass in 1 2 3; do
    for t in "$@"; do
      for p in $(tagged "$t"); do
        { printf '[%s] %s: SIGKILL (alive %s s after SIGTERM) ' "$(date '+%F %T')" "$why" "$KILL_GRACE"
          ps -o pid=,ppid=,etime=,args= -p "$p" 2> /dev/null | cut -c1-400; } >> "$K"
        kill -KILL "$p" 2> /dev/null && nk=$((nk + 1))
      done
    done
    sleep 1
  done
  left=$(for t in "$@"; do tagged "$t"; done | paste -sd' ' -)
  say "  $why: $n process(es) sent SIGTERM, $nk SIGKILL after ${KILL_GRACE} s$([ -n "$left" ] && echo "; STILL ALIVE: $left")" \
    "(each one: $K)"
}

hard_watchdog() {   # (background) the hard stop at HARD_EPOCH, then the backstop AA_PACK_MAX_H later
  local alive
  while [ "$(date +%s)" -lt "$HARD_EPOCH" ]; do
    kill -0 "$MAIN_PID" 2> /dev/null || return 0; sleep "$WATCH_EVERY"
  done
  kill -0 "$MAIN_PID" 2> /dev/null || return 0
  date '+%F %T' > "$HARD_FLAG"
  alive=$(tagged "$RUN_ID" | wc -l)
  say "HARD STOP ($LIMIT_NOTE; soft stop was $(date -d "@$STOP_EPOCH" '+%a %H:%M:%S')): $alive running process(es) of this run;" \
    "SIGTERM, then SIGKILL after ${KILL_GRACE} s; no new command except P7's"
  [ "$alive" = 0 ] || kill_tagged "HARD STOP" "$RUN_ID"
  date '+%F %T' > "$HARD_FLAG.killed"                      # wait_hard_kill: P7 / the exit wait for this
  say "HARD STOP: done; P7 packs what finished, then run_pod.sh exits (backstop: killed if still running in $PACK_MAX_S s)"
  local t_end=$(( $(date +%s) + PACK_MAX_S ))
  while [ "$(date +%s)" -lt "$t_end" ]; do
    kill -0 "$MAIN_PID" 2> /dev/null || return 0; sleep "$WATCH_EVERY"
  done
  kill -0 "$MAIN_PID" 2> /dev/null || return 0
  say "HARD STOP BACKSTOP: run_pod.sh still running $PACK_MAX_S s after the hard stop (P7 hung?):" \
    "killing every command of the run (P7's too), then run_pod.sh itself (pid $MAIN_PID)"
  kill_tagged "BACKSTOP" "$RUN_ID" "$RUN_ID.P7"
  kill -TERM "$MAIN_PID" 2> /dev/null
  sleep "$KILL_GRACE"
  if kill -0 "$MAIN_PID" 2> /dev/null; then say "BACKSTOP: SIGKILL run_pod.sh $MAIN_PID"; kill -KILL "$MAIN_PID" 2> /dev/null; fi
  kill "$SAMPLER" 2> /dev/null
}

wait_hard_kill() {   # after a hard stop: wait until the watchdog has finished killing (SIGKILL round included), so
  #                    P7 never packs beside a dying process and the exit never cuts the kill short
  [ "$DRY_RUN" != 1 ] && [ -f "$HARD_FLAG" ] && [ -n "${WATCHDOG:-}" ] || return 0
  local w=0
  while [ ! -f "$HARD_FLAG.killed" ] && kill -0 "$WATCHDOG" 2> /dev/null && [ "$w" -lt $((KILL_GRACE + 120)) ]; do
    sleep 1; w=$((w + 1))
  done
}

on_signal() {   # run_pod.sh got SIGTERM / SIGINT / SIGHUP: no new command, its workers and every tagged process stopped
  trap - TERM INT HUP
  [ -f "$HARD_FLAG" ] || echo "signal $(date '+%F %T')" > "$HARD_FLAG"
  say "run_pod.sh got a signal: stopping its workers and every running command of this run"
  local j
  for j in $(jobs -p); do [ "$j" = "${WATCHDOG:-}" ] || kill -TERM "$j" 2> /dev/null; done
  kill_tagged "SIGNAL" "$RUN_ID" "$RUN_ID.P7"
  kill "${SAMPLER:-}" "${WATCHDOG:-}" 2> /dev/null
  exit 143
}

# AA_HARDSTOP_TEST (CPU only, nothing real): fake P4 workers, each a process tree like a real one (a launch.sh-style
# restart loop around a sleep, a background sleep, a child that ignores SIGTERM), and a fake P7 (hang = a P7 that hangs)
t_p4_worker() {
  run_cmd "$LOGS/P4/gpu${GL[$1]}.log" "( until sleep 600; do :; done ) & sleep 600 & bash -c 'trap \"\" TERM; sleep 600' & wait"
}
t_p4() { run_workers t_p4_worker || return 1; }
t_p5() { say "  TEST P5: must not start (the soft stop)"; return 0; }
t_p7() {
  wait_hard_kill
  IN_P7=1
  run_cmd "$LOGS/P7/pack.log" "echo packed at \$(date +%s) > $WORK/packed.txt" || return 1
  if [ "$HS_TEST" = hang ]; then run_cmd "$LOGS/P7/pack.log" "sleep 600"; fi
}

# ------------------------------------------------------------------------------------------------- main
say "arena A pod stage | $(git --no-pager log -1 --format='%h %s' 2> /dev/null) | WCM $WCMV | GPUs $GPUS | lambdas $LAMS (placeholder) | scopes $SCOPES | corpus $CORPUS (N_SEQ $N_SEQ) | stores $STO | targets $TGT | baselines on $BDEV | WORK $WORK | FROM $FROM STOP_AT $STOP_AT$([ -n "$LIMIT" ] && echo " | LIMIT $LIMIT")$([ -n "$KEYS" ] && echo " | KEYS $KEYS")$([ "$DRY_RUN" = 1 ] && echo ' | DRY RUN')"
if [ -n "$HARD_EPOCH" ]; then
  if [ "$LIMIT_NOTE" = STOP_EPOCH ]; then
    say "time limit: soft stop STOP_EPOCH $(date -d "@$STOP_EPOCH" '+%a %F %H:%M'), hard stop (kill) AA_HARD_GRACE=$HARD_GRACE s later; then P7"
  else
    say "time limit: $LIMIT_NOTE after the start: soft stop at +$(( (STOP_EPOCH - T0) / 60 )) min, hard stop (kill) at +$(( (HARD_EPOCH - T0) / 60 )) min; then P7"
  fi
else say "time limit: none (AA_MAX_H=0, no STOP_EPOCH)"; fi
SAMPLER=
WATCHDOG=
if [ "$DRY_RUN" != 1 ]; then
  mkdir -p "$LOGS" "$STATE" || exit 1
  (                                                        # GPU memory every 30 s, for the re-cost (A12)
    while true; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2> /dev/null \
        | awk -v t="$(date +%s)" '{ print t "," $0 }' >> "$WORK/gpu_mem.csv"
      sleep 30
    done
  ) > /dev/null 2>&1 < /dev/null &                         # detached from stdout: a pipe reader is not held open
  SAMPLER=$!
  if [ -n "$HARD_EPOCH" ]; then hard_watchdog < /dev/null & WATCHDOG=$!; fi
  trap 'wait_hard_kill; kill $SAMPLER $WATCHDOG 2> /dev/null' EXIT
  trap on_signal TERM INT HUP
fi

FAILS=0
stopped_note() { if [ -f "$HARD_FLAG" ]; then echo "hard stop: the running commands were killed"; else echo "soft stop: no new unit"; fi; }
finish() {   # finish RC: STOP_AT (3) is a normal end; a stage cut by the time limit: P7 packs what finished, exit 4
  local rc=$1
  [ "$rc" = 3 ] && { say "stopped after $STOP_AT as asked ($FAILS failed stages)"; [ "$FAILS" = 0 ]; exit $?; }
  if [ "$rc" != 0 ] && [ "$rc" != 2 ] && [ "$DRY_RUN" != 1 ] && late; then
    say "time limit reached ($(stopped_note)) before P4: P7 packs what finished"
    stage P7 pack p7_pack
    say "arena A STOPPED by its time limit: exit 4"
    exit 4
  fi
  exit "$rc"
}
if [ "$HS_TEST" != 0 ]; then                               # AA_HARDSTOP_TEST: the time-limit test, CPU only
  [ "$DRY_RUN" = 1 ] && { say "AA_HARDSTOP_TEST needs DRY_RUN=0"; exit 1; }
  say "AA_HARDSTOP_TEST=$HS_TEST: fake P4 / P5 / P7 (no GPU, nothing real), WORK $WORK"
  stage P4 "TEST fake workers" t_p4; rc=$?; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
  stage P5 "TEST after the soft stop" t_p5; rc=$?; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
  stage P7 "TEST pack" t_p7; rc=$?; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
  if [ "$FAILS" != 0 ] && late; then say "arena A STOPPED by its time limit ($(stopped_note)): $FAILS stages incomplete; exit 4"; exit 4; fi
  say "TEST finished: $FAILS failed stages"; [ "$FAILS" = 0 ]; exit $?
fi
stage P0 assets p0_assets; rc=$?; [ "$rc" = 0 ] || finish "$rc"
stage P1 envs p1_envs; rc=$?; [ "$rc" = 0 ] || finish "$rc"
stage P2 "corpus + data pass" p2_data; rc=$?; [ "$rc" = 0 ] || finish "$rc"
stage P3 targets p3_targets; rc=$?; [ "$rc" = 0 ] || finish "$rc"
if [ -f "$STATE/P3.done" ] || [ "$(snum "$FROM")" -gt 3 ]; then  # P3 skipped (marker / AA_FROM): the active lists
  if [ "$DRY_RUN" = 1 ]; then [ "$(snum "$FROM")" -gt 3 ] && make_lists
  elif [ -n "$KEYS" ] || [ ! -f "$LIST" ] || [ ! -f "$LIST_TC" ]; then make_lists || finish 1; fi
fi
stage P4 baselines p4_baselines; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P4h "held-out baselines" p4h_held; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P5 WCM p5_wcm; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P6 scoring p6_score; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P6b "per-prompt (A8b)" p6b_perprompt; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
stage P7 pack p7_pack; rc=$?; [ "$rc" = 3 ] && finish 3; [ "$rc" = 0 ] || FAILS=$((FAILS + 1))
if [ "$FAILS" != 0 ] && [ "$DRY_RUN" != 1 ] && late; then
  say "arena A STOPPED by its time limit ($(stopped_note)): $FAILS stages incomplete, P7 packed what finished (rerun to resume); exit 4"
  exit 4
fi
say "arena A finished: $FAILS failed stages$([ "$FAILS" = 0 ] || echo ' (rerun to resume: finished work is skipped)')"
[ "$FAILS" = 0 ]
