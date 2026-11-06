#!/bin/bash
# Arena C (DAN-191): the environment for the official SFC code at third_party/feature-circuits@7fbd82b.
# Run in WSL. Pins from specs/sfc.md §8 / §12.5, with one labelled change:
#   torch 2.8.0+cu128 instead of torch<2.6. The RTX 5070 Ti (Blackwell, sm_120) needs CUDA 12.8 wheels (torch >= 2.7).
#   The <2.6 pin only guarded t.load's weights_only default; TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 (sfc_env.sh) restores
#   the old default without touching their code.
set -euo pipefail
ROOT="/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"
ENV=${SFC_ENV:-$HOME/venvs/sfc107}
FC="$ROOT/third_party/feature-circuits"

python3 -m venv "$ENV"
"$ENV/bin/pip" install --quiet --upgrade pip
"$ENV/bin/pip" install --quiet "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/pip" install --quiet "nnsight==0.3.7" "numpy<2"
# requirements.txt minus its unpinned dictionary_learning line (installed from the pinned local clone below)
grep -v '^dictionary_learning' "$FC/requirements.txt" > /tmp/sfc_requirements.txt
"$ENV/bin/pip" install --quiet -r /tmp/sfc_requirements.txt "torch==2.8.0" "nnsight==0.3.7" "numpy<2"
"$ENV/bin/pip" install --quiet "$FC/dictionary_learning"   # @61ac634, the companion commit
# The resolver otherwise takes 2026 releases (numpy 2.5, transformers 5.x, huggingface_hub 2.x), which nnsight 0.3.7
# (2024-11) predates. Pin the HF stack to its era (transformers 4.47.1, 2024-12) and numpy < 2 as their requirements say.
"$ENV/bin/pip" install --quiet "torch==2.8.0" "nnsight==0.3.7" "numpy<2" "transformers==4.47.1" \
  "tokenizers>=0.21,<0.22" "huggingface_hub>=0.26,<0.28" "accelerate==1.2.1" "datasets==3.2.0" "safetensors<0.6" \
  "scipy==1.14.1" "scikit-learn==1.5.2" "pandas==2.2.3" "numba==0.60.0" "llvmlite==0.43.0" "matplotlib==3.9.4" \
  "pyarrow==18.1.0" "umap-learn==0.5.7"
"$ENV/bin/pip" check
"$ENV/bin/pip" freeze > "$ROOT/experiments/107-true-baselines/arena_c/env_freeze.txt"
"$ENV/bin/python" - <<'EOF'
import torch, nnsight, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("nnsight", nnsight.__version__, "numpy", numpy.__version__)
x = torch.randn(64, 64, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo "SFC env ready: $ENV"
