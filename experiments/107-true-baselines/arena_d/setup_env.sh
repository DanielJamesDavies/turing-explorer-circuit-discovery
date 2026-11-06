#!/bin/bash
# Arena D (DAN-201 / DAN-202 / DAN-203): environments for circuit-tracer@255d989 (D1) and ScalableSAECircuits@f54a4ef
# (D2). DRAFT, written 2026-11-03, NOT RUN. Needs Daniel's approval before the first run (pip installs; FETCH=1 also
# downloads the weights listed in README §1).
#
#   bash setup_env.sh                 # build both venvs, freeze, smoke-import (no weights)
#   FETCH=1 bash setup_env.sh         # also fetch the weights (google/gemma-2-2b needs HF_TOKEN in the environment)
#   FETCH=1 CLT=1 bash setup_env.sh   # plus the 28.5 GB CLT-426k (stretch)
#
# Pins (README §1): ct107 = Python 3.11 (their CI), torch 2.13.0 cu128, transformer-lens 3.2.1 (the last release that
# allows transformers < 5, which is what their CI resolved on 2026-10-14), transformers 4.57.3, nnsight 0.8.0rc1,
# huggingface_hub 0.36.2. caples107 = Python 3.11, sae-lens 5.3.3 + transformer-lens 2.11.0 (both the latest on
# 2025-01-27, the HEAD commit date), transformers 4.47.1, torch 2.8.0 cu128 (labelled: the 5070 Ti needs CUDA 12.8).
set -euo pipefail

ROOT=${ROOT:-"/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"}
HERE="$ROOT/experiments/107-true-baselines/arena_d"
VENVS=${VENVS:-$HOME/venvs}
# DEVIATION (built 2026-11-03): WSL has no python3.11 (system 3.12.3, no uv), so both venvs use 3.12. On the pod set
# PY=python3.11 if present. ct107's torch 2.13.0 is not on the cu128 index (it stops at 2.11.0): it comes from PyPI
# (CUDA 13.0 wheels, 2.13.0+cu130), as in 038's working venv. caples107 needed pyarrow 18.1.0 (pyarrow >= 19 needs
# numpy 2; arena C pins the same).
PY=${PY:-python3.12}
CT_ENV=$VENVS/ct107
CAPLES_ENV=$VENVS/caples107
CT_COMMIT=255d989e93be1299d53d67b59e9fd76a0e274b14
CAPLES_COMMIT=f54a4efce908432b8771ea671a248ee30ec2b3af
TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------- D1: circuit-tracer
if [ ! -x "$CT_ENV/bin/python" ]; then
  "$PY" -m venv "$CT_ENV"
  "$CT_ENV/bin/pip" install --quiet --upgrade pip
  "$CT_ENV/bin/pip" install --quiet "torch==2.13.0"          # PyPI default = +cu130 (cu128 index tops at 2.11.0)
  # non-editable install from git at the pinned commit (spec §7.2), so local edits cannot leak in (038's F1 FAIL)
  "$CT_ENV/bin/pip" install --quiet \
    "circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer@$CT_COMMIT" \
    "torch==2.13.0" "transformer-lens==3.2.1" "transformers==4.57.3" "nnsight==0.8.0rc1" \
    "huggingface_hub==0.36.2" "pytest>=8"
fi
"$CT_ENV/bin/pip" check
"$CT_ENV/bin/pip" freeze > "$HERE/env_freeze_ct107.txt"
"$CT_ENV/bin/python" - <<EOF
import json, pathlib, importlib.metadata as md
import torch, transformer_lens, transformers, circuit_tracer
from circuit_tracer import ReplacementModel, attribute            # noqa: F401
from circuit_tracer.graph import prune_graph, compute_graph_scores  # noqa: F401
d = md.distribution("circuit-tracer")
du = json.loads(d.read_text("direct_url.json") or "{}")
commit = du.get("vcs_info", {}).get("commit_id")
print("circuit-tracer", d.version, "commit", commit)
assert commit == "$CT_COMMIT", "F1 provenance: wrong commit"
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| TL", transformer_lens.__version__ if hasattr(transformer_lens, "__version__") else md.version("transformer-lens"),
      "| transformers", transformers.__version__)
EOF

# ---------------------------------------------------------------- D2: ScalableSAECircuits (Caples et al.)
# Their repo has no requirements file; the Colab installs sae-lens / transformer-lens unpinned. These are the
# versions current at their HEAD commit (2025-01-27). Their code imports wandb, datasets, pandas, matplotlib, einops.
if [ ! -x "$CAPLES_ENV/bin/python" ]; then
  "$PY" -m venv "$CAPLES_ENV"
  "$CAPLES_ENV/bin/pip" install --quiet --upgrade pip
  "$CAPLES_ENV/bin/pip" install --quiet "torch==2.8.0" --index-url "$TORCH_INDEX"
  "$CAPLES_ENV/bin/pip" install --quiet "torch==2.8.0" "sae-lens==5.3.3" "transformer-lens==2.11.0" \
    "transformers==4.47.1" "numpy<2" "datasets<3" "huggingface_hub>=0.26,<0.28" "wandb" "pandas" "einops" \
    "matplotlib" "tqdm" "pyarrow==18.1.0"
fi
"$CAPLES_ENV/bin/pip" check
"$CAPLES_ENV/bin/pip" freeze > "$HERE/env_freeze_caples107.txt"
WANDB_MODE=disabled "$CAPLES_ENV/bin/python" - <<'EOF'
import re
from collections import defaultdict
import pandas as pd
import sae_lens, transformer_lens, torch
from sae_lens import SAE, HookedSAETransformer  # noqa: F401
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
print("sae_lens", sae_lens.__version__, "| torch", torch.__version__, torch.cuda.is_available())
# The exact selection code of interp_bottlenecks.py:138-159 (gap 6, L0 target 100), on sae-lens' own directory
df = pd.DataFrame.from_records({k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}).T
df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)
m = df.loc["gemma-scope-2b-pt-res"]["saes_map"]
pat = re.compile(r"layer_(\d+)/width_16k/average_l0_(\d+)")
by = defaultdict(list)
for s in m:
    g = pat.search(s)
    if g and m[s] is not None:
        by[int(g.group(1))].append((s, int(g.group(2))))
pick = {L: min(v, key=lambda x: abs(x[1] - 100))[0] for L, v in by.items()}
want = {0: 105, 6: 70, 12: 82, 18: 74, 24: 73}
for L, l0 in want.items():
    print(L, pick[L])
    assert pick[L].endswith(f"average_l0_{l0}"), (L, pick[L])
EOF

# ---------------------------------------------------------------- weights (explicit, off by default)
if [ "${FETCH:-0}" = "1" ]; then
  "$CT_ENV/bin/python" - <<'EOF'
import os
from huggingface_hub import hf_hub_download, snapshot_download
# 1. official Gemma-2-2B (gated; HF_TOKEN must be set in the environment, never written to a file)
snapshot_download("google/gemma-2-2b", revision="c5ebcd40d208330abc697524c919956e692655cf",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer.model"])
# 2. unsloth bf16 mirror (already cached locally; on the pod for the bf16 cross-check and 038's route)
snapshot_download("unsloth/gemma-2-2b", revision="25319945f7fd83b8b903e12081777b7eef2ba993")
# 3. PLT transcoders: only what circuit-tracer itself fetches (no features/*.bin)
snapshot_download("mwhanna/gemma-scope-transcoders", revision="3412e19be8f72d47f8683cb92028a46513222b2b",
                  allow_patterns=["config.yaml", "layer_*.safetensors"])
# 4. Gemma Scope residual SAEs, Caples' gap-6 picks
for p in ["layer_0/width_16k/average_l0_105", "layer_6/width_16k/average_l0_70", "layer_12/width_16k/average_l0_82",
          "layer_18/width_16k/average_l0_74", "layer_24/width_16k/average_l0_73"]:
    hf_hub_download("google/gemma-scope-2b-pt-res", f"{p}/params.npz",
                    revision="fd571b47c1c64851e9b1989792367b9babb4af63")
# 5. CLT-426k (stretch)
if os.environ.get("CLT") == "1":
    snapshot_download("mntss/clt-gemma-2-2b-426k", revision="b1e9ab376d07c90d780bf20b5fb1a0c89bd0f5e7",
                      allow_patterns=["config.yaml", "*.safetensors"])
EOF
  # 6. Neuronpedia Dallas graph (F3 reference), into this folder's out/
  mkdir -p "$HERE/out/fidelity"
  curl -sfL -o "$HERE/out/fidelity/gemma-fact-dallas-austin.json" \
    "https://neuronpedia-attrib.s3.us-east-1.amazonaws.com/user-graphs/cmao6p1xd0000ypedw0qhlhob/gemma-fact-dallas-austin.json"
fi

echo "arena D envs ready: $CT_ENV  $CAPLES_ENV  (Caples code: run from a COPY of third_party/ScalableSAECircuits @ $CAPLES_COMMIT)"
