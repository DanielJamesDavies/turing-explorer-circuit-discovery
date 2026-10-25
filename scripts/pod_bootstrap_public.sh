#!/bin/bash
# Fresh-pod bootstrap for the PUBLIC-model runs (Pythia-70m, GPT-2 small; DAN-39 B7 / stage S0 of
# dev-notes/2026-10-23-pod-plan-public-models.md). Unlike pod_bootstrap_global.sh it copies NO TuringLLM data: the
# models and SAE dictionaries come from the Hugging Face Hub, the corpora are prepared by the chain (stage S1).
# The global volume (/workspace) is only the destination of the chain's verified tars; nothing here writes to it.
#
#   git clone --branch multi-device https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git /root/turing \
#     && bash /root/turing/scripts/pod_bootstrap_public.sh 2>&1 | tee /root/bootstrap_public.log
#
# Steps: 1 code (multi-device, pod disk)  2 venv over the system torch, transformers pinned to the version the 070
# adapter tests passed on (5.1.0)  3 native extensions  4 HF downloads (models + 18 + 48 dictionaries, ~13 GB)
# 5 adapter tests + 069 corpus tests (CPU), then the same models and every dictionary on the GPU vs CPU
# 6 summary: git log -1, versions, GPUs (checkpoint C0).
# Idempotent / resumable: every step skips what is already done (the venv by a pin stamp, the downloads by the HF
# cache); re-run after any failure. Env: BRANCH (multi-device)  ROOT (/root/turing)  NO_RESET=1 (keep local commits
# instead of resetting to origin)  SKIP_TESTS=1  SKIP_GPU_CHECK=1  EXPECT_GPUS (warn if fewer GPUs are visible)
# CONFIG=<file> (copied to config.yaml; default: keep the repo's config.yaml, which the 068 gates and the 069 smoke
# data passes ran with).
set -eo pipefail
REPO=https://github.com/DanielJamesDavies/turing-explorer-circuit-discovery.git
BRANCH=${BRANCH:-multi-device}
ROOT=${ROOT:-/root/turing}
PY=$ROOT/.venv/bin/python
# The pins. transformers 5.1.0: the version the 070 adapter tests (tests/model/test_public_adapters.py) passed on
# (070 README); huggingface_hub / safetensors / tokenizers: the local versions those tests ran with (2026-10-23).
PINS="transformers==5.1.0 tokenizers==0.22.2 huggingface_hub==1.24.0 safetensors==0.8.0 zstandard \
pydantic==2.12.5 pydantic_core==2.41.5 numpy==2.5.1 pyyaml pyarrow pandas matplotlib tqdm rich psutil pytest"
export PYTHONPATH=src
export HF_HUB_DISABLE_TELEMETRY=1

echo "== 1/6 code: $BRANCH onto the pod disk ($ROOT)"
if [ ! -d "$ROOT/.git" ]; then
  git clone --branch "$BRANCH" "$REPO" "$ROOT"
fi
cd "$ROOT"
git fetch origin "$BRANCH"
git checkout "$BRANCH"
if [ "${NO_RESET:-0}" = "1" ]; then
  echo "  NO_RESET=1: keeping the working tree"
else
  git reset --hard "origin/$BRANCH"
fi
git --no-pager log --oneline -1
if [ -n "${CONFIG:-}" ]; then
  cp "$CONFIG" config.yaml && echo "  config.yaml <- $CONFIG"
fi
mkdir -p logs_public

echo "== 2/6 venv over the system torch (pins: $PINS)"
STAMP=$ROOT/.venv/.public_pins
if [ ! -x "$PY" ]; then
  python3 -m venv --system-site-packages "$ROOT/.venv"
fi
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$PINS" ]; then
  # shellcheck disable=SC2086  # PINS is a word list on purpose
  "$PY" -m pip install -q $PINS
  echo "$PINS" > "$STAMP"
else
  echo "  pins already installed (stamp $STAMP)"
fi
"$PY" - <<'PYCHECK'
import transformers
assert transformers.__version__ == "5.1.0", "transformers %s, expected the pinned 5.1.0" % transformers.__version__
import zstandard, huggingface_hub, safetensors, torch  # noqa: F401,E401
print("  imports ok; transformers", transformers.__version__, "| torch", torch.__version__)
PYCHECK

echo "== 3/6 native extensions (as pod_bootstrap_global.sh)"
if (cd "$ROOT/src/native" && "$PY" setup.py build_ext --inplace > "$ROOT/logs_public/native_build.log" 2>&1); then
  echo "  built: $(cd "$ROOT/src/native" && ls ./*.so 2>/dev/null | tr '\n' ' ')"
else
  echo "  (native build failed, see logs_public/native_build.log; PyTorch fallbacks work, S2 checks the CUDA kernel)"
fi

echo "== 4/6 HF downloads: EleutherAI/pythia-70m, gpt2, 18 Pythia + 48 GPT-2 dictionaries (resumable)"
HF_HUB_OFFLINE=0 "$PY" scripts/pod_public_check.py download
df -h "${HF_HOME:-$HOME/.cache/huggingface}" | tail -1

echo "== 5/6 tests: adapters (070) + corpus tool (069), then the GPU check"
if [ "${SKIP_TESTS:-0}" != "1" ]; then
  # The adapter tests run their models on CPU (devices are fixed in the test) and SKIP a family whose files are not
  # cached; with everything downloaded nothing may be skipped, so a skip is treated as a failure here.
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= "$PY" -m pytest tests/model/test_public_adapters.py \
    tests/pipeline/test_069_token_field.py -q -rs -p no:cacheprovider 2>&1 | tee logs_public/bootstrap_tests.log
  if grep -qiE "^SKIPPED|[0-9]+ skipped" logs_public/bootstrap_tests.log; then
    echo "  FAIL: tests skipped (missing HF files or zstandard), see logs_public/bootstrap_tests.log"; exit 1
  fi
fi
if [ "${SKIP_GPU_CHECK:-0}" != "1" ]; then
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 "$PY" scripts/pod_public_check.py gpu --device cuda:0 \
    2>&1 | grep -v -E "LOAD REPORT|UNEXPECTED|^Key |^-+\+|^$|Notes:|can be ignored" | tee logs_public/bootstrap_gpu.log
fi

echo "== 6/6 summary (checkpoint C0)"
git --no-pager log -1 --format='  commit %H%n  %ad  %s' --date=iso
"$PY" - <<'PYSUM'
import importlib, torch
for m in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "zstandard", "numpy", "pyarrow"):
    print("  %-16s %s" % (m, importlib.import_module(m).__version__))
print("  %-16s %s (CUDA %s)" % ("torch", torch.__version__, torch.version.cuda))
n = torch.cuda.device_count()
print("  GPUs            ", n, "x", torch.cuda.get_device_name(0) if n else "-")
PYSUM
md5sum config.yaml | sed 's/^/  config.yaml md5 /'
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ -n "${EXPECT_GPUS:-}" ] && [ "$NGPU" -lt "$EXPECT_GPUS" ]; then
  echo "  WARNING: $NGPU GPUs visible, expected $EXPECT_GPUS"
fi
echo ""
echo "Bootstrap complete. The local HEAD must equal the commit above (plan risk table). Next:"
echo "  nvidia-cuda-mps-control -d      # MPS (062: P=6 under MPS ~1.6x); chain.sh also starts it unless MPS=0"
echo "  cd $ROOT && DRY_RUN=1 bash experiments/082-public-runs/chain.sh      # print the plan, run nothing"
echo "  cd $ROOT && STOP_AFTER=S4 nohup bash experiments/082-public-runs/chain.sh > logs_public/chain.log 2>&1 &"
echo "  bash experiments/082-public-runs/progress.sh"
