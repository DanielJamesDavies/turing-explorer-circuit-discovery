#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)/src"
python src/build_search_cache.py "$@"
