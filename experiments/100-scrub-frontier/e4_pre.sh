#!/bin/bash
# 100 E4 pre-flight (WSL, repo root): (1) a v2CR refit through e4_engine (all E4 switches off) vs Run A's v2CR circuit;
# (2) a 20-step smoke of every E4 mechanism on one dev target (no random control; scores meaningless).
set -u
cd "/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
X=experiments/100-scrub-frontier
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
printf "2.attn.34021\n" > $X/targets_e4one.txt
.venv/bin/python -u $X/run.py --engine e4 --check-refit --check-arm v2cr_1e-3 --check-ref $X/out/main/circuits \
    --targets $X/targets_e4one.txt --out $X/out/e4_check > $X/logs/e4_check.log 2>&1
echo "check exit $?"
.venv/bin/python -u $X/run.py --engine e4 --no-rand --steps 20 --targets $X/targets_e4one.txt \
    --arms v2cr-w0.5_1e-3,v2cr-mixCAR_1e-3,rsAC_1e-3,v2cr-k4_1e-3,v2cr-cap2_1e-3 --out $X/out/e4_smoke > $X/logs/e4_smoke.log 2>&1
echo "smoke exit $?"
