#!/usr/bin/env bash
# Latent pilot (2026-11-02, Daniel: "do the small local pilot"): does the geometric lr schedule (arena C, WCM_ADAPTER.md
# section 8) change production latent WCM? The 062 driver, MODE=main, WCM_VERSION=v3 (production), on 102's check-panel
# records (062 out_full/out/ctx, explicit split), 9 targets (targets.txt: the 6 failing check targets whose zero term
# ended highest + 3 passing, one per band, seed 20261102), 400 steps, CURVE_STRIDE 1 (the full curve), SKIP_SPEC=1.
# Arms, each its own OUT, serial: geo_l1e-3 (LR_SCHEDULE=geometric, production λ), geo_l3e-3 (geometric, λ ×3: the
# geometric lr integral is ~0.32× constant's, so this keeps the sparsity budget), const_l1e-3 (production, the control).
# Stops before an arm once END_BY (epoch) passes. WSL (repo root):
#   nohup setsid bash experiments/107-true-baselines/latent_pilot/run_pilot.sh > experiments/107-true-baselines/latent_pilot/pilot.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../../.."
E=experiments/107-true-baselines/latent_pilot
END_BY=${END_BY:-$(date -d "today 15:45" +%s)}
for spec in geo:1e-3 geo:3e-3 const:1e-3; do
  kind=${spec%%:*}; lam=${spec#*:}
  [ "$(date +%s)" -ge "$END_BY" ] && { echo "[pilot] END_BY passed before $spec"; break; }
  sched=""; [ "$kind" = geo ] && sched=geometric
  echo "[pilot] $(date +%H:%M:%S) start $spec"
  PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MODE=main WCM_VERSION=v3 LAM=$lam LR_SCHEDULE=$sched \
    OUT="$E/out/${kind}_l${lam}" CTX_DIR=experiments/062-h100-protocol-v1/out_full/out/ctx TARGETS="$E/targets.txt" \
    CURVE_STRIDE=1 ORDER=list SKIP_SPEC=1 .venv/bin/python -u experiments/062-h100-protocol-v1/driver.py \
    || echo "[pilot] FAILED $spec"
  echo "[pilot] $(date +%H:%M:%S) end $spec"
done
echo "[pilot] $(date +%H:%M:%S) DONE"
