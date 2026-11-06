#!/bin/bash
# Arena A: the weight / data downloads of README.md section 1, as commands. WRITTEN 2026-11-03, NOT RUN: waits for
# Daniel's approval. On a pod these are proposed to Daniel, never run autonomously.
#
#   bash fetch_assets.sh local     # D1 Kissane attention SAEs (2.12 GB)          -> WSL HF cache
#   bash fetch_assets.sh owt       # D5 (optional) OWT apollo shard 0 (243 MB)    -> WSL HF cache
#   bash fetch_assets.sh pod       # pod: GPT-2 + Dunefsky transcoders + Kissane (4.48 GB) -> the pod's HF cache
#   bash fetch_assets.sh verify    # sizes of every file against the HF API listing of 2026-11-03 (no download)
#
# Pip packages (D2-D4) are fetched by setup_env.sh, not here. Revisions are pinned; nothing is fetched unpinned.
set -euo pipefail
ROOT=${ROOT:-"/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/2"}
PY=${PY:-"$ROOT/.venv/bin/python"}

KISSANE_REPO=ckkissane/attn-saes-gpt2-small-all-layers
KISSANE_REV=97e0da0e9182695e9813d15ea32184da83f5c874
# name stems: the version suffix (_v4 ... _v9) and L1 / lr / dc vary per layer, so the files are listed exactly
KISSANE_STEMS=(
  gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5
  gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4
  gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7
  gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6
  gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9
  gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9
  gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9
)
TC_REPO=pchlenski/gpt2-transcoders
TC_REV=4f3ca345572ce2d75018a60d10b4dca8bd899e99
GPT2_REV=607a30d783dfa663caf39e06633721c8d4cfcd7e
OWT_REPO=apollo-research/Skylion007-openwebtext-tokenizer-gpt2
OWT_REV=f02886b54795e8acabceb637ca119f9ae8f19d3f

kissane_files() { for s in "${KISSANE_STEMS[@]}"; do printf '%s.pt\n%s_cfg.json\n' "$s" "$s"; done; }
tc_files() { for l in $(seq 0 11); do
  printf 'final_sparse_autoencoder_gpt2-small_blocks.%d.ln2.hook_normalized_24576.pt\n' "$l"; done; }

fetch() {   # $1 repo, $2 revision, $3 repo_type, then file names on stdin
  HF_HUB_OFFLINE=0 "$PY" -I -c "
import sys
from huggingface_hub import hf_hub_download
repo, rev, rtype = sys.argv[1:4]
for f in sys.stdin.read().split():
    print(hf_hub_download(repo, f, revision=rev, repo_type=rtype))
" "$1" "$2" "$3"
}

verify() {   # HF API sizes vs the cached files (metadata only)
  HF_HUB_OFFLINE=0 "$PY" -I -c "
import os, sys
from huggingface_hub import HfApi, try_to_load_from_cache
api = HfApi()
for repo, rev in [('$KISSANE_REPO', '$KISSANE_REV'), ('$TC_REPO', '$TC_REV')]:
    info = api.model_info(repo, revision=rev, files_metadata=True)
    tot = miss = 0
    for s in info.siblings:
        if not (s.rfilename.endswith('.pt') or s.rfilename.endswith('_cfg.json')) or '/' in s.rfilename \
           or 'log_feature_sparsity' in s.rfilename:
            continue
        p = try_to_load_from_cache(repo, s.rfilename, revision=rev)
        ok = isinstance(p, str) and os.path.getsize(p) == s.size
        tot += s.size; miss += 0 if ok else 1
        if not ok: print('  missing / size mismatch:', s.rfilename, s.size)
    print(repo, rev[:8], 'bytes', tot, 'missing', miss)
"
}

case "${1:-}" in
  local)  kissane_files | fetch "$KISSANE_REPO" "$KISSANE_REV" model ;;
  owt)    echo data/train-00000-of-00073.parquet | fetch "$OWT_REPO" "$OWT_REV" dataset ;;
  pod)    printf 'config.json\ngeneration_config.json\nmodel.safetensors\ntokenizer.json\ntokenizer_config.json\nvocab.json\nmerges.txt\n' \
            | fetch gpt2 "$GPT2_REV" model
          tc_files | fetch "$TC_REPO" "$TC_REV" model
          kissane_files | fetch "$KISSANE_REPO" "$KISSANE_REV" model ;;
  verify) verify ;;
  *) echo "usage: fetch_assets.sh local|owt|pod|verify"; exit 2 ;;
esac
