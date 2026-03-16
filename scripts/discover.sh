#!/usr/bin/env bash
# Run with: bash scripts/discover.sh  (not: python scripts/discover.sh)
#
# Usage:
#   bash scripts/discover.sh                  # Run discovery with saved candidates
#   bash scripts/discover.sh --reselect       # Re-run candidate selection and then discovery
#   bash scripts/discover.sh --n-seeds 32     # Run selection for 32 seeds
#
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/discover_circuits.py --reselect "$@"
