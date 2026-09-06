"""CONVENTION PROBE for GemmaScope ATTENTION and RESIDUAL SAEs on
Gemma-2-2B (HF), the way 037/gemma_loader.py probed the MLP ones:
which tensor was each SAE trained on? Verified by FVU and L0 against the
advertised L0, not assumed.

Candidates:
  att: (a) o_proj INPUT  (concatenated head outputs, d = n_heads*head_dim)
       (b) o_proj OUTPUT (attention block output, d_model)
       (c) post_attention_layernorm OUTPUT
  res: (d) decoder-layer OUTPUT hidden state (resid_post)
       (e) decoder-layer INPUT hidden state  (resid_pre)
The SAE's W_enc input width rules out the impossible ones outright.

  LAYER=4 TIER=2 python experiments/048-gemma-tasks/probe_kinds.py
"""
import os
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
CACHE = Path(os.environ.get("SAE_CACHE", str(Path.home() / "gemmascope")))
LAYER = int(os.environ.get("LAYER", 4))
TIER = int(os.environ.get("TIER", 2))
TEXT = ("The Eiffel Tower is in Paris, the capital of France. Machine learning "
        "models process text one token at a time. When Mary and John went to "
        "the store, John gave a drink to Mary.")
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
dev = torch.device("cuda")


def ladder(repo, layer):
    pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % layer)
    return sorted({int(m.group(1)) for f in list_repo_files(repo) for m in [pat.search(f)] if m})


def load(kind, layer, l0):
    local = CACHE / ("%s_layer_%d_w16k_l0_%d.npz" % (kind, layer, l0))
    if not local.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(REPOS[kind], "layer_%d/width_16k/average_l0_%d/params.npz" % (layer, l0))
        tmp = local.with_suffix(".tmp"); tmp.write_bytes(Path(p).read_bytes()); tmp.replace(local)
    z = np.load(local)
    return {k: torch.tensor(z[k], dtype=torch.float32, device=dev) for k in z.files}


def fvu_l0(sae, x):
    # drop position 0 (BOS): Gemma's attention-sink position carries an
    # anomalous norm; GemmaScope trained without it and circuit-tracer
    # ignores it — including it inflates FVU at deeper layers by 10x+
    x = x[:, 1:].float().reshape(-1, x.shape[-1])
    if x.shape[-1] != sae["W_enc"].shape[0]:
        return None
    pre = x @ sae["W_enc"] + sae["b_enc"]
    c = pre * (pre > sae["threshold"])
    rec = c @ sae["W_dec"] + sae["b_dec"]
    fvu = float(((x - rec) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    return fvu, float((c > 0).float().sum(-1).mean())


tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(dev).eval()
ids = tok(TEXT, return_tensors="pt")["input_ids"].to(dev)
blk = model.model.layers[LAYER]
cap = {}
hs = [
    blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("att_a_oproj_in", i[0].detach())),
    blk.self_attn.o_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("att_b_oproj_out", o.detach())),
    blk.post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("att_c_post_attn_ln", o.detach())),
    blk.register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("res_e_layer_in", a[0].detach()), with_kwargs=True),
    blk.register_forward_hook(lambda m, a, o: cap.__setitem__("res_d_layer_out", (o[0] if isinstance(o, tuple) else o).detach())),
]
with torch.no_grad():
    model(ids)
for h in hs:
    h.remove()
print("captured:", {k: tuple(v.shape[-1:]) for k, v in cap.items()})

for kind in ("att", "res"):
    lad = ladder(REPOS[kind], LAYER)
    l0 = lad[min(TIER, len(lad) - 1)]
    sae = load(kind, LAYER, l0)
    print("\n%s L%d | ladder %s | tier %d -> advertised L0 %d | W_enc in-dim %d"
          % (kind, LAYER, lad, TIER, l0, sae["W_enc"].shape[0]))
    for name, x in cap.items():
        if not name.startswith(kind):
            continue
        r = fvu_l0(sae, x)
        print("  %-24s %s" % (name, "dim mismatch" if r is None else "FVU %.3f | L0 %.1f" % r))
