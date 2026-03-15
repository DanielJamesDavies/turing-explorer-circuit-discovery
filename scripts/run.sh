#!/usr/bin/env bash
# Run with: bash scripts/run.sh  (not: python scripts/run.sh)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/main.py
