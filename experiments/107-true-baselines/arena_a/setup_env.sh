#!/bin/bash
# Arena A (DAN-192 / DAN-200): the two pinned environments for the baselines' official code. Run in WSL (or on the pod).
# WRITTEN 2026-11-03, NOT RUN: it downloads packages, so it waits for Daniel's approval of README.md section 1.
#
#   ge107  Ge et al. 2024, OpenMOSS Language-Model-SAEs v0.1.0 (third_party/Language-Model-SAEs @67bfb0b)
#          python 3.10 (their requires-python "==3.10.*"); the pins are their pdm.lock (torch 2.3.1, transformers 4.43.1,
#          ...). transformer_lens = THEIR VENDORED FORK (third_party/Language-Model-SAEs/TransformerLens): the attributor
#          calls fork-only APIs (run_with_ref_cache, mount_hooked_modules), so upstream TL is never installed here.
#   tc107  Dunefsky et al. 2024, transcoder_circuits @8721ff0. python 3.10; their requirements.txt pins (torch 2.2.0,
#          transformer_lens 1.11.0, datasets 2.12.0, einops 0.7.0, setuptools 67.7.2, wandb 0.16.0, huggingface-hub
#          0.17.3) plus era pins of ours for what they leave open (transformers 4.35.2, numpy 1.26.4, pyarrow 14.0.2,
#          fsspec 2023.6.0, ...): labelled [ours] in README.md section 2.
#
# Neither clone is pip-installed (an install would write *.egg-info / build/ into third_party/). Both are put on
# PYTHONPATH by the shims bin/ge_env.sh and bin/tc_env.sh, which also set PYTHONDONTWRITEBYTECODE=1 so no __pycache__
# lands in the clones, and HF_HUB_OFFLINE=1 so nothing is fetched at run time.
#
# TORCH_INDEX=cpu (default; local: the Blackwell card needs torch >= 2.7, so the baselines' local checks run on the CPU)
# TORCH_INDEX=cu121 (pod, H100). Usage:  bash setup_env.sh [ge|tc|both]
set -euo pipefail
ROOT=${ROOT:-"/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"}
HERE="$ROOT/experiments/107-true-baselines/arena_a"
LMS="$ROOT/third_party/Language-Model-SAEs"
TCC="$ROOT/third_party/transcoder_circuits"
PY310=${PY310:-python3.10}
TORCH_INDEX=${TORCH_INDEX:-cpu}
WHICH=${1:-both}
GE_ENV=${GE_ENV:-$HOME/venvs/ge107}
TC_ENV=${TC_ENV:-$HOME/venvs/tc107}
IDX="https://download.pytorch.org/whl/$TORCH_INDEX"

# guards: the clones must be at their pinned commits and untouched
( cd "$LMS" && test "$(git rev-parse HEAD)" = 67bfb0b9f835f772f3af2aa176745c6b6135373a ) \
  || { echo "Language-Model-SAEs is not at v0.1.0 (67bfb0b)"; exit 1; }
( cd "$TCC" && test "$(git rev-parse HEAD)" = 8721ff06b64889e8cc50bb1c5e869c7f72b8303d ) \
  || { echo "transcoder_circuits is not at 8721ff0"; exit 1; }

write_shim() {   # $1 env dir, $2 shim name, $3 PYTHONPATH
  printf '%s\n' \
    "# source me: arena A baseline env ($2), written by setup_env.sh" \
    "source \"$1/bin/activate\"" \
    "export PYTHONPATH=\"$3\"" \
    "export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled" \
    > "$1/bin/$2"
}

setup_ge() {
  "$PY310" -m venv "$GE_ENV"
  "$GE_ENV/bin/pip" install --quiet --upgrade "pip<25"
  "$GE_ENV/bin/pip" install --quiet "torch==2.3.1" --index-url "$IDX"
  # pdm.lock pins (67bfb0b) for everything the circuit module, sae.py, config.py and the vendored TL import, plus pytest
  # for their tests/intergration/test_attributor.py. Server / training-only extras (fastapi, pymongo, openai, tiktoken,
  # uvicorn) are not imported by the circuit path and are left out.
  "$GE_ENV/bin/pip" install --quiet --prefer-binary "torch==2.3.1" \
    "transformers==4.43.1" "tokenizers==0.19.1" "huggingface-hub==0.23.3" "accelerate==0.31.0" "safetensors==0.4.3" \
    "datasets==2.19.2" "pyarrow==16.1.0" "numpy==1.26.4" "pandas==2.2.2" "einops==0.8.0" "fancy-einsum==0.0.3" \
    "jaxtyping==0.2.29" "typeguard==2.13.3" "beartype==0.18.5" "better-abc==0.0.3" "rich==13.7.1" \
    "sentencepiece==0.2.0" "tqdm==4.66.4" "typing-extensions==4.12.2" "networkx==3.3" "pydantic==2.7.4" \
    "packaging==24.1" "pyyaml==6.0.1" "tomlkit==0.12.5" "msgpack==1.0.8" "tensorboardX==2.6.2.2" "wandb==0.17.1" \
    "matplotlib==3.9.0" "plotly==5.22.0" "pytest==8.2.2"
  "$GE_ENV/bin/pip" check
  "$GE_ENV/bin/pip" freeze > "$HERE/env_freeze_ge.txt"
  write_shim "$GE_ENV" ge_env.sh "$LMS/src:$LMS/TransformerLens"
  # -I ignores PYTHONDONTWRITEBYTECODE, so -B is needed too (else __pycache__ lands in the clones; fixed 2026-11-03)
  ( source "$GE_ENV/bin/ge_env.sh" && cd "$HERE" && python -I -B -c "
import sys; sys.path[:0] = '''$LMS/src:$LMS/TransformerLens'''.split(':')
import torch, transformer_lens, lm_saes.circuit.transformer as T
from transformer_lens.hook_points import HookedRootModule
assert hasattr(HookedRootModule, 'run_with_ref_cache'), 'not the OpenMOSS TL fork'
print('ge107 ok: torch', torch.__version__, '| TL fork from', transformer_lens.__file__)" \
    && python -B -m pytest -q -p no:cacheprovider "$LMS/tests/intergration/test_attributor.py" )
  echo "Ge env ready: source $GE_ENV/bin/ge_env.sh"
}

setup_tc() {
  "$PY310" -m venv "$TC_ENV"
  "$TC_ENV/bin/pip" install --quiet --upgrade "pip<25"
  "$TC_ENV/bin/pip" install --quiet "torch==2.2.0" --index-url "$IDX"
  # their requirements.txt, then [ours] era pins for what it leaves open (the resolver would otherwise take 2026
  # releases that need huggingface-hub >= 0.30 or numpy 2, which torch 2.2.0 and hub 0.17.3 predate)
  "$TC_ENV/bin/pip" install --quiet --prefer-binary -r "$TCC/requirements.txt" "torch==2.2.0" \
    "transformers==4.35.2" "tokenizers==0.15.0" "numpy==1.26.4" "pyarrow==14.0.2" "fsspec==2023.6.0" \
    "pandas==2.1.4" "matplotlib==3.8.2" "plotly==5.18.0" "accelerate==0.25.0" "safetensors==0.4.1" \
    "jaxtyping==0.2.25" "typeguard==2.13.3" "beartype==0.14.1" "fancy-einsum==0.0.3" "rich==13.7.0" \
    "sentencepiece==0.1.99" "tqdm==4.66.1" "pytest==7.4.3"
  "$TC_ENV/bin/pip" check
  "$TC_ENV/bin/pip" freeze > "$HERE/env_freeze_tc.txt"
  write_shim "$TC_ENV" tc_env.sh "$TCC"
  ( source "$TC_ENV/bin/tc_env.sh" && cd "$HERE" && python -I -B -c "
import sys; sys.path.insert(0, '''$TCC''')
import torch, transformer_lens
from sae_training.sparse_autoencoder import SparseAutoencoder
from transcoder_circuits.circuit_analysis import greedy_get_top_paths, paths_to_graph, make_sae_feature_vector
from importlib.metadata import version
assert version('transformer_lens') == '1.11.0', version('transformer_lens')
print('tc107 ok: torch', torch.__version__, '| TL', transformer_lens.__file__)" )
  echo "Dunefsky env ready: source $TC_ENV/bin/tc_env.sh"
}

case "$WHICH" in
  ge) setup_ge ;;
  tc) setup_tc ;;
  both) setup_ge; setup_tc ;;
  *) echo "usage: setup_env.sh [ge|tc|both]"; exit 2 ;;
esac
