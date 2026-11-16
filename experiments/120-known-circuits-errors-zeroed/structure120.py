"""120 stage 2 (GPU): 117's structure test (structure117.main, unchanged) on the error-zeroed circuits, with every
upstream SAE error zeroed in the attribution frame (common120.install_struct_error_zero).

Frames, as in 117 but error-free: "circuit" (fitted alphas, non-members 0, errors 0), "null" (20 random matched draws,
same pool rule as 117: latents active in the unmodified model on 512 training-split prompts), "full" = the ERROR-FREE
full model (every latent live, every upstream error 0). The errors-live full model is 117's own "full" row.

  wsl: N_PROMPTS=128 N_DRAWS=20 PYTHONPATH=src .venv/bin/python -X utf8 -u experiments/120-known-circuits-errors-zeroed/structure120.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common120 as C  # noqa: E402

os.environ["OUT"] = str(C.WCM)
os.environ.setdefault("STRUCT_OUT", str(C.WCM / "dan215" / "structure.jsonl"))

import lib117 as L  # noqa: E402
import structure117 as S  # noqa: E402

C.install_struct_error_zero(L)

if __name__ == "__main__":
    print("[120 struct] error-zeroed attribution; circuits from %s" % S.OUT, flush=True)
    S.main()
    print("[120 struct] error-zero site calls %d" % C.STATS["struct_calls"], flush=True)
    print("EXIT 0", flush=True)
