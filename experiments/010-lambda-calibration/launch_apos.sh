#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, wherever it is checked out
source .venv/bin/activate
export PYTHONPATH=src
# 10 seeds each in a SHALLOW, MID and DEEP component. The script resumes, so
# the 5+5 already measured at comps 8 and 32 are reused; this adds 5+10+5=20.
# Three components (not two) so the exponent beta can be tested for being
# SHARED - if it is, only the per-component intercept needs calibrating and a
# new seed costs zero extra runs.
stdbuf -oL -eL env COMPONENTS=8,25,32 N_PER_COMPONENT=10 \
  python experiments/010-lambda-calibration/within_component.py 2>&1 \
  | grep --line-buffered -vE 'CF-Capture|CFaithfulness|UserWarning|Consider using|mean_m'
