#!/usr/bin/env bash
# Run with: bash scripts/search.sh  (not: python scripts/search.sh)
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/search_latents.py --run_patch_clamp
