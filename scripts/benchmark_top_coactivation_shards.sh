#!/usr/bin/env bash
set -euo pipefail

# Benchmarks top-coactivation reducer sharding by running the pipeline with a
# baseline single-process reduce and then one or more target-sharded configs.
#
# Examples:
#   bash scripts/benchmark_top_coactivation_shards.sh
#   SHARDS_LIST="2 4 8" BENCH_N_DATA_SHARDS=16 bash scripts/benchmark_top_coactivation_shards.sh
#
# Logs are written under runs/top_coactivation_shard_bench/<timestamp>/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT_DIR/src"

SHARDS_LIST="${SHARDS_LIST:-2 4 8}"
BENCH_N_DATA_SHARDS="${BENCH_N_DATA_SHARDS:-}"
RUN_LABEL="${RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-runs/top_coactivation_shard_bench/$RUN_LABEL}"
SHARD_OUTPUT_BASE="${SHARD_OUTPUT_BASE:-outputs/top_coactivation_reduce_shards}"
CONFIG_PATH="$ROOT_DIR/config.yaml"
CONFIG_BACKUP="$(mktemp)"

cp "$CONFIG_PATH" "$CONFIG_BACKUP"
restore_config() {
  cp "$CONFIG_BACKUP" "$CONFIG_PATH"
  rm -f "$CONFIG_BACKUP"
}
trap restore_config EXIT

mkdir -p "$LOG_DIR"

patch_config() {
  local backend="$1"
  local shards="$2"
  local shard_dir="$3"
  BENCH_BACKEND="$backend" \
  BENCH_SHARDS="$shards" \
  BENCH_SHARD_DIR="$shard_dir" \
  BENCH_N_DATA_SHARDS="$BENCH_N_DATA_SHARDS" \
  python - <<'PY'
import os
from pathlib import Path

import yaml

path = Path("config.yaml")
with path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

top = cfg["latents"]["top_coactivation"]
top["reduce_backend"] = os.environ["BENCH_BACKEND"]
top["reduce_shards"] = int(os.environ["BENCH_SHARDS"])
top["reduce_shard_output_dir"] = os.environ["BENCH_SHARD_DIR"] or None

n_data_shards = os.environ.get("BENCH_N_DATA_SHARDS", "")
if n_data_shards:
    cfg["data"]["n_shards"] = int(n_data_shards)

with path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_case() {
  local name="$1"
  local backend="$2"
  local shards="$3"
  local shard_dir="$4"
  local log_path="$LOG_DIR/$name.log"

  echo "[bench] $name backend=$backend shards=$shards shard_dir=${shard_dir:-<none>}"
  patch_config "$backend" "$shards" "$shard_dir"
  python src/main.py 2>&1 | tee "$log_path"
}

run_case "single_process" "single_process" "1" ""

for shards in $SHARDS_LIST; do
  run_case "target_sharded_${shards}" "target_sharded" "$shards" "$SHARD_OUTPUT_BASE/$RUN_LABEL/shards_${shards}"
done

echo "[bench] complete. Logs: $LOG_DIR"
