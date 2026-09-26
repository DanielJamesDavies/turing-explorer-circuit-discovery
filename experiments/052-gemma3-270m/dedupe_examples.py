"""DEDUPE the Gemma Scope 2 example stores.

Only resid_post SAEs ship examples.safetensors (18 files, one per layer; the
attn_out / mlp_out folders contain config + params only). Each file carries a
`tokens` corpus [N, 256] (~400 MB) plus per-latent arrays. This script:

  1. hashes every file's `tokens` tensor -> how many DISTINCT corpora exist
  2. writes each distinct corpus once:   corpus_<hash12>.npy  (int32 [N, 256])
  3. writes per-layer compact arrays:    res_L<l>.npz  (activations float32,
     seq_ids int32, positions int16, feature_frequencies, logit_effects,
     top/bottom tokens + logits, and the corpus hash they index)

Output dir: EXAMPLES_DIR (default ~/gemmascope2_examples). The HF cache copies
are left untouched.

  python experiments/052-gemma3-270m/dedupe_examples.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
import gemmascope2 as GS  # noqa: E402

OUT = Path(os.environ.get("EXAMPLES_DIR", str(Path.home() / "gemmascope2_examples")))
OUT.mkdir(parents=True, exist_ok=True)
summary = {"layers": {}, "corpora": {}}
src_bytes = 0
for layer in range(18):
    p = hf_hub_download(GS.REPO, GS.path("res", layer) + "/examples.safetensors")
    src_bytes += os.path.getsize(p)
    with safe_open(p, "np") as f:
        toks = f.get_tensor("tokens")
        h = hashlib.sha1(np.ascontiguousarray(toks).tobytes()).hexdigest()[:12]
        cpath = OUT / ("corpus_%s.npy" % h)
        if h not in summary["corpora"]:
            if not cpath.exists():
                np.save(cpath, toks.astype(np.int32))
            summary["corpora"][h] = {"shape": list(toks.shape), "first_layer": layer}
        arrs = {k: f.get_tensor(k) for k in f.keys() if k != "tokens"}
    sid = arrs["seq_ids"]
    np.savez(OUT / ("res_L%d.npz" % layer),
             activations=arrs["activations"].astype(np.float32), seq_ids=sid.astype(np.int32),
             positions=arrs["positions"].astype(np.int16),
             feature_frequencies=arrs["feature_frequencies"], logit_effects=arrs["logit_effects"].astype(np.float32),
             top_tokens=arrs["top_tokens"], top_logits=arrs["top_logits"],
             bottom_tokens=arrs["bottom_tokens"], bottom_logits=arrs["bottom_logits"],
             corpus=np.array(h))
    summary["layers"][layer] = {"corpus": h, "seq_id_max": int(sid.max()), "n_tokens_rows": int(toks.shape[0])}
    print("layer %2d | corpus %s | seq_ids max %d of %d rows" % (layer, h, int(sid.max()), toks.shape[0]), flush=True)

out_bytes = sum(q.stat().st_size for q in OUT.iterdir() if q.is_file())
summary.update(src_gb=round(src_bytes / 1e9, 2), out_gb=round(out_bytes / 1e9, 2))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("\n%d distinct corpora across 18 layers | source %.1f GB -> deduped store %.1f GB at %s"
      % (len(summary["corpora"]), src_bytes / 1e9, out_bytes / 1e9, OUT))
