#!/usr/bin/env bash
# Run with: bash scripts/ablation_matrix.sh <layer> <kind> <latent>
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/ablation_sensitivity.py "$@"
