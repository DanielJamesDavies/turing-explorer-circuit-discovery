#!/usr/bin/env bash
# Arena D2 (README §5.2, plan D5): Caples et al.'s interp_bottlenecks.py, UNMODIFIED, on Gemma-2-2B (fp32,
# google/gemma-2-2b by name, as theirs), over the lambda grids of README §0.3, each in its own `git clone --local`
# copy of third_party/ScalableSAECircuits. Never runs anything in third_party/ (their script writes masks/ relative to
# the working directory, which would overwrite the authors' anchors there, README §0.4).
#   grid  --start/--end_threshold  steps  role
#   A     0.1    10                5      scripts/sae_gap.sh as committed       (anchored)
#   B     0.005  0.1               5      the spec's anchors                    (anchored)
#   C     5e-5   0.005             5                                            (anchored)
#   D     0.01   0.2               5      the Colab grid (report, never match)
#   dense three sub-grids dC / dB / dA over C's, B's and A's ranges with DENSE_N steps each (default 8): the curve
#         [ours: the grid choice; their script and spacing]
# Per grid: clone -> check HEAD = f54a4ef and a clean tree -> move the clone's committed masks/ aside (mv, so masks/ is
# fresh and every JSON there is ours) -> run their script from inside the copy -> DONE marker when every lambda's
# <lam>_run.json exists. A grid without DONE reruns whole (their script has no per-lambda resume). No new grid after
# STOP_EPOCH. Then caples_fidelity.py scores A / B / C against the anchors (CPU).
# Their script picks cuda if available and their helpers default to cuda:0: one grid process per GPU, chosen with
# GPU=<index> (CUDA_VISIBLE_DEVICES). Weights: google/gemma-2-2b is gated; huggingface_hub reads HF_TOKEN from the
# environment by itself (never written to a file or printed here); --config_file /nonexistent keeps their script from
# reading a token file. --hf_cache = $HF_HOME/hub: their main() sets HF_HOME to it only AFTER huggingface_hub is
# imported (no effect on sae-lens' downloads) and passes it as TL's cache_dir (the hub cache, where the model lives).
#   WSL / pod, repo root:
#     GPU=0 bash experiments/107-true-baselines/arena_d/run_caples.sh A
#     GPU=1 bash .../run_caples.sh B C          (sequential on GPU 1)
#     DRY_RUN=1 bash .../run_caples.sh A B C D dense
set -u
cd "$(dirname "$0")/../../.."
REPO=$(pwd)
AD=experiments/107-true-baselines/arena_d
SRC_REPO=$REPO/third_party/ScalableSAECircuits
PIN=f54a4efce908432b8771ea671a248ee30ec2b3af
RUN=${RUN:-$REPO/$AD/out/caples_runs}
CAPLES_PY=${CAPLES_PY:-$HOME/venvs/caples107/bin/python}
PY=${PY:-$REPO/.venv/bin/python}
GPU=${GPU:-0}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
DENSE_N=${DENSE_N:-8}
STOP_EPOCH=${STOP_EPOCH:-99999999999}
DRY_RUN=${DRY_RUN:-0}
SAVE=masks/sva/rc_train_saegap6_mean_error
say() { echo "[run_caples] $(date +%H:%M:%S) $*"; }
late() { [ "$(date +%s)" -ge "$STOP_EPOCH" ]; }
GRIDS=("$@")
[ ${#GRIDS[@]} -eq 0 ] && GRIDS=(A B C)
expand() {   # grid name -> "name start end steps" lines
  case $1 in
    A) echo "A 0.1 10 5" ;;
    B) echo "B 0.005 0.1 5" ;;
    C) echo "C 5e-05 0.005 5" ;;
    D) echo "D 0.01 0.2 5" ;;
    dense) echo "dC 5e-05 0.005 $DENSE_N"; echo "dB 0.005 0.1 $DENSE_N"; echo "dA 0.1 10 $DENSE_N" ;;
    *) say "unknown grid $1"; return 1 ;;
  esac
}
n_json() { ls "$1/$SAVE"/logit_diff_*_run/*_run.json 2> /dev/null | wc -l; }
FAILS=0
say "grids ${GRIDS[*]}; run root $RUN; GPU $GPU; HF_HOME $HF_HOME; dry run $DRY_RUN"
if [ "$DRY_RUN" != 1 ]; then
  [ -x "$CAPLES_PY" ] || { say "no caples107 python at $CAPLES_PY"; exit 1; }
  [ "$(git -C "$SRC_REPO" rev-parse HEAD 2> /dev/null)" = "$PIN" ] || { say "third_party clone is not at $PIN"; exit 1; }
  if [ "${HF_HUB_OFFLINE:-0}" != 1 ] && [ -z "${HF_TOKEN:-}" ]; then
    say "WARNING: HF_TOKEN is not set and HF_HUB_OFFLINE != 1: the gated google/gemma-2-2b download will fail"
  fi
fi
for G in "${GRIDS[@]}"; do
  expand "$G" > /dev/null || { FAILS=$((FAILS + 1)); continue; }
  while read -r NAME S E N; do
    D=$RUN/$NAME
    C=$D/src
    CMD=(env WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 "$CAPLES_PY" -u
         interp_bottlenecks.py --task sva/rc_train --model_name google/gemma-2-2b --sae_gap 6 --use_mask
         --use_mean_error --mean_mask --per_token_mask --run_training_thresholds --start_threshold "$S"
         --end_threshold "$E" --n_threshold_steps "$N" --portion_of_data 0.4 --hf_cache "$HF_HOME/hub"
         --config_file /nonexistent)
    if [ -f "$D/DONE" ]; then say "skip $NAME (done)"; continue; fi
    if late; then say "skip $NAME (STOP_EPOCH reached)"; continue; fi
    if [ "$DRY_RUN" = 1 ]; then
      say "DRY $NAME: git clone --local --no-hardlinks $SRC_REPO $C; check HEAD = ${PIN:0:7} + clean;" \
          "mv $C/masks $C/masks_from_clone; cd $C && ${CMD[*]} > $D/run.log 2>&1;" \
          "DONE when $N x $SAVE/logit_diff_<lam>_run/<lam>_run.json"
      continue
    fi
    mkdir -p "$D"
    if [ ! -d "$C/.git" ]; then
      git clone -q --local --no-hardlinks "$SRC_REPO" "$C" || { say "FAILED clone $NAME"; FAILS=$((FAILS + 1)); continue; }
      head=$(git -C "$C" rev-parse HEAD)
      dirty=$(git -C "$C" status --porcelain)
      { echo "clone of $SRC_REPO at $(date -Is)"; echo "HEAD $head"; echo "status: ${dirty:-clean}"; } > "$D/provenance.txt"
      if [ "$head" != "$PIN" ] || [ -n "$dirty" ]; then say "FAILED $NAME: copy HEAD $head / dirty"; FAILS=$((FAILS + 1)); continue; fi
      mv "$C/masks" "$C/masks_from_clone"
    fi
    # every tracked file outside masks/ must still be the pinned commit's (a resumed copy included)
    if [ "$(git -C "$C" rev-parse HEAD)" != "$PIN" ] || \
       [ -n "$(git -C "$C" status --porcelain -- . ':(exclude)masks' ':(exclude)masks_from_clone')" ]; then
      say "FAILED $NAME: the copy differs from $PIN outside masks/"; FAILS=$((FAILS + 1)); continue
    fi
    say "start $NAME (S $S, E $E, $N steps) on GPU $GPU"
    t0=$(date +%s)
    ( cd "$C" && "${CMD[@]}" ) > "$D/run.log" 2>&1
    rc=$?
    nj=$(n_json "$C")
    printf "%s\t%s\t%s\t%s\t%s\n" "$NAME" "$t0" "$(date +%s)" "$rc" "$nj" >> "$RUN/steps.tsv"
    if [ $rc -ne 0 ] || [ "$nj" -lt "$N" ]; then
      say "FAILED $NAME (exit $rc, $nj / $N run JSONs; $D/run.log)"; FAILS=$((FAILS + 1)); continue
    fi
    date -Is > "$D/DONE"
    say "done $NAME ($(( $(date +%s) - t0 )) s)"
  done < <(expand "$G")
done
if [ "$DRY_RUN" = 1 ]; then
  say "DRY fidelity: CUDA_VISIBLE_DEVICES= $PY -B $AD/caples_fidelity.py --runs $RUN --grids A,B,C"
else
  have=""
  for g in A B C; do [ -f "$RUN/$g/DONE" ] && have="$have,$g"; done
  if [ -n "$have" ]; then
    CUDA_VISIBLE_DEVICES= "$PY" -B "$AD/caples_fidelity.py" --runs "$RUN" --grids "${have#,}" \
      > "$RUN/fidelity.log" 2>&1 || say "FAILED fidelity"
    cat "$RUN/fidelity.log"
  fi
fi
say "DONE ($FAILS failed)"
[ "$FAILS" -eq 0 ]
