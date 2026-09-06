"""DIRECT LOGIT ATTRIBUTION of the attention members at one layer (name-mover test): does each feature's
decoder direction (through o_proj and the unembedding, final-norm weight
applied) promote the name it fires on? Rank ordering over the 24 names is
scale-free so the RMSNorm's per-token scale does not matter. Also reports which head each feature's decoder direction lives in.

  LAYER=22 python experiments/048-gemma-tasks/dla_att_members.py
"""
import os
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path("experiments/048-gemma-tasks")
MODEL_ID = "unsloth/gemma-2-2b"
DEV = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEV).eval()
dens = json.load(open(HERE / os.environ.get("DENSITY", "ioi_att_mlp_gemma_density.json")))
data = torch.load(HERE / "ioi_gemma_prompts.pt", weights_only=False)
names = sorted({r["meta"]["io"] for r in data["rows"]} | {r["meta"]["s"] for r in data["rows"]})
name_ids = [tok.encode(" " + n, add_special_tokens=False) for n in names]
assert all(len(i) == 1 for i in name_ids), name_ids
name_ids = torch.tensor([i[0] for i in name_ids], device=DEV)
print("names:", len(names))

# END firing counts per IO name for the chosen layer (255 prompts, one hook)
import sys
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
LAYER = int(os.environ.get("LAYER", 22))
DENSITY = os.environ.get("DENSITY", "ioi_att_mlp_gemma_density.json")
pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % LAYER)
lad = sorted({int(m.group(1)) for f in list_repo_files("google/gemma-scope-2b-pt-att") for m in [pat.search(f)] if m})
z = np.load(Path.home() / "gemmascope" / ("att_layer_%d_w16k_l0_%d.npz" % (LAYER, lad[min(2, len(lad) - 1)])))
sae = {k: torch.tensor(z[k], device=DEV, dtype=torch.float32) for k in z.files}
feats = sorted(d["latent"] for d in dens if d["kind"] == "att" and d["layer"] == LAYER)
alpha = {d["latent"]: d["alpha"] for d in dens if d["kind"] == "att" and d["layer"] == LAYER}
cap = {}
h = model.model.layers[LAYER].self_attn.o_proj.register_forward_pre_hook(lambda m, i: cap.__setitem__("x", i[0].detach()))
fire = {f: Counter() for f in feats}
with torch.no_grad():
    for r in data["rows"]:
        model(torch.tensor([r["ids"]], device=DEV))
        pre = cap["x"][0, -1] @ sae["W_enc"] + sae["b_enc"]
        c = pre * (pre > sae["threshold"])
        for f in feats:
            if c[f] > 0:
                fire[f][r["meta"]["io"]] += 1
h.remove()

Wo = model.model.layers[LAYER].self_attn.o_proj.weight          # (d_model, n_heads*head_dim)
g = 1.0 + model.model.norm.weight                                 # Gemma RMSNorm: (1 + w)
U = model.lm_head.weight                                          # (vocab, d_model)
head_dim = model.config.head_dim
print("\natt L%d members: DLA over the %d names (rank of the feature's top-firing IO name; 1 = promotes it most)" % (LAYER, len(names)))
print("  %6s | %5s | %-7s | %4s | %-8s %-8s | %s" % ("latent", "alpha", "fires@IO", "rank", "top DLA", "2nd DLA", "head (argmax |W_dec| block)"))
ranks = []
for f in feats:
    d = sae["W_dec"][f]                                           # (n_heads*head_dim)
    resid = Wo @ d                                                 # (d_model)
    logits = U @ (resid * g)                                       # (vocab)
    nl = logits[name_ids]
    order = torch.argsort(nl, descending=True).tolist()
    top_name = fire[f].most_common(1)[0][0] if fire[f] else None
    rank = order.index(names.index(top_name)) + 1 if top_name else -1
    blocks = d.view(-1, head_dim).norm(dim=-1)
    ranks.append((rank, sum(fire[f].values())))
    print("  %6d | %5.2f | %-7s | %4s | %-8s %-8s | head %d (%.0f%% of norm)"
          % (f, alpha[f], "%s:%d" % (top_name, fire[f][top_name]) if top_name else "-", rank if rank > 0 else "-",
             names[order[0]], names[order[1]], int(blocks.argmax()), 100 * float(blocks.max() ** 2 / (blocks ** 2).sum())))
r1 = sum(1 for r, n in ranks if r == 1)
r3 = sum(1 for r, n in ranks if 0 < r <= 3)
print("\n  rank 1: %d / %d | rank <= 3: %d / %d | chance rank-1 = 1/%d" % (r1, len(ranks), r3, len(ranks), len(names)))
