#!/bin/bash
# 103 chain smoke (WSL, local GPU): the 082 chain under WCM_VERSION=v2cr in its smoke mode (LIMIT=2, 20 steps, P=1,
# GPU 0) through the new stages: TuringLLM T3 main -> T4 sweep (W_0.001, W_0.004, U_0.001) -> T10 without-R arm ->
# T11 donor main -> T12 donor sweep (+ the T10 arm); Pythia S4 records -> S6 main -> unweighted -> donor ->
# donor_unweighted -> S10 pack. Stages not under test are pre-marked done in smoke_chain/logs/state (T0-T2: the 082
# smoke_v3 run's fresh records and lists are read, never written; T5-T8; Pythia S5 and the S6 alpha1 / nulls /
# stable core / 073 arms).
#   PHASE=tl  bash experiments/103-donor-stage/run_smoke_chain.sh   # the TuringLLM track only (Pythia stops at S3)
#   PHASE=all bash experiments/103-donor-stage/run_smoke_chain.sh   # then Pythia (TL stages are done: skipped)
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
S3=$PWD/experiments/082-public-runs/smoke_v3/out_turingllm
export LIMIT=2 TL_TRACK=1 MODELS=pythia-70m WCM_VERSION=v2cr WORK=experiments/103-donor-stage/smoke_chain \
  TL_CTX_FRESH="$S3/ctx_fresh" TL_LISTS="$S3/lists"
[ "${PHASE:-all}" = tl ] && export STOP_AFTER_pythia_70m=S3
bash experiments/082-public-runs/chain.sh
