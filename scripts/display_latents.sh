#!/usr/bin/env bash
# Run with: bash scripts/analyze.sh  (or: scripts/analyze.sh 5 resid 12345)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/display_latents.py "$@"
