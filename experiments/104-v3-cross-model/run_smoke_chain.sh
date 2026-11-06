#!/bin/bash
# 104 part C: the 082 chain smoke under WCM_VERSION=v3 (the chain's default since 2026-11-01) on all three models, as
# 103's run_smoke_chain.sh (LIMIT=2, 20 steps, P=1, GPU 0): TuringLLM T3 main + alpha1 -> T4 sweep (W_0.001, W_0.004,
# U_0.001) -> T10 without-R arm (v2) -> T11 donor main -> T12 donor sweep (+ T10); Pythia S4 records -> S6 main ->
# unweighted -> donor -> donor_unweighted; GPT-2 S7 records -> S9 main -> donor; S10 pack. Stages not under test are
# pre-marked done in smoke_chain/logs/state (as 103: TuringLLM T0-T2 read 082 smoke_v3's fresh records and lists, never
# written; T5-T8; Pythia S5 and the S6 alpha1 / nulls / stable core / 073 arms; GPT-2 S8 and the same S9 arms).
#   PHASE=tl  bash experiments/104-v3-cross-model/run_smoke_chain.sh   # the TuringLLM track only (public tracks stop at S3)
#   PHASE=all bash experiments/104-v3-cross-model/run_smoke_chain.sh   # then both public tracks (TL stages are done)
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
S3=$PWD/experiments/082-public-runs/smoke_v3/out_turingllm
export LIMIT=2 TL_TRACK=1 MODELS="pythia-70m gpt2-small" WORK=${WORK:-experiments/104-v3-cross-model/smoke_chain} \
  TL_CTX_FRESH="$S3/ctx_fresh" TL_LISTS="$S3/lists"
unset WCM_VERSION                      # the chain's own default (v3) is under test
if [ "${PHASE:-all}" = tl ]; then export STOP_AFTER_pythia_70m=S3 STOP_AFTER_gpt2_small=S3; fi
bash experiments/082-public-runs/chain.sh
