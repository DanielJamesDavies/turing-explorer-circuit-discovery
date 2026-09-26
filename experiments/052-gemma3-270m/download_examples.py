"""Download every examples.safetensors for the chosen SAE set (16k, L0 big,
all 18 layers x att/mlp/res = 54 files, ~733 MB each, ~40 GB) into the HF
cache, 8 files in parallel. Resumable: finished files are skipped.

  python experiments/052-gemma3-270m/download_examples.py
"""
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
import gemmascope2 as GS  # noqa: E402

patterns = ["%s/examples.safetensors" % GS.path(k, l) for l in range(18) for k in ("att", "mlp", "res")]
t0 = time.time()
root = snapshot_download(GS.REPO, allow_patterns=patterns, max_workers=int(os.environ.get("WORKERS", 8)))
n = sum(1 for p in patterns if (Path(root) / p).exists())
print("%d/%d examples files present under %s | %.0fs" % (n, len(patterns), root, time.time() - t0))
