"""IS ZERO-FILL AT AN ATTENTION SITE A REMOVAL ON GEMMA 3?

Gemma's decoder layer applies post_attention_layernorm AFTER o_proj, and our
att site is o_proj's INPUT. Per layer, on IOI prompts (real tokens only, BOS
excluded), with x = the clean o_proj input:

  y_clean   = o_proj(x)
  y_empty   = o_proj(x + (0 - c) @ W_dec)        every latent zero-filled
  out_clean = post_attention_layernorm(y_clean)
  unfrozen  = post_attention_layernorm(y_empty)  what the fitter did
  frozen    = y_empty / rms(y_clean) * (1 + w)   FREEZE_NORM=1

  pre-norm shrink   ||y_empty|| / ||y_clean||    how much the edit removes
  unfrozen ratio    ||unfrozen|| / ||out_clean|| ~1 => rescaled back up
  frozen ratio      ||frozen||  / ||out_clean||  should equal the shrink
  cos(unfrozen, out_clean)                       what direction survives

EXACTNESS CHECK first: with no edit, the frozen formula must reproduce the
module's output (validates the (1 + weight) RMSNorm form and eps).

  python experiments/052-gemma3-270m/norm_diag.py
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gemmascope2 as GS  # noqa: E402

MODEL = os.environ.get("MODEL", "unsloth/gemma-3-270m")
N = int(os.environ.get("N", 64))
DEV = torch.device("cuda")
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
LAYERS = model.model.layers
rng = random.Random(0)
NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Joseph", "Jim", "Peter",
         "Paul", "Bob", "Alice", "Sarah", "Emma", "David", "Lucy", "Anna", "Kate", "Mark"]
TPL = ["Then, {X} and {Y} went to the {p}. {S} gave a {o} to",
       "When {X} and {Y} got a {o} at the {p}, {S} decided to give it to",
       "After {X} and {Y} went to the {p}, {S} gave a {o} to"]
prompts = []
for _ in range(N):
    io, s = rng.sample(NAMES, 2)
    x, y = (io, s) if rng.random() < 0.5 else (s, io)
    prompts.append(rng.choice(TPL).format(X=x, Y=y, S=s, p=rng.choice(["store", "park", "school"]),
                                          o=rng.choice(["drink", "book", "letter"])))

SAE = {l: GS.load_sae("att", l, device=DEV, dtype=torch.float32) for l in range(len(LAYERS))}
cap = {}
hs = [LAYERS[l].self_attn.o_proj.register_forward_pre_hook(
      lambda m, i, _l=l: cap.__setitem__(_l, i[0].detach())) for l in range(len(LAYERS))]

acc = {l: {"shrink": [], "unfrozen": [], "frozen": [], "cos": [], "exact": 0.0} for l in range(len(LAYERS))}
for pr in prompts:
    ids = tok(pr, return_tensors="pt")["input_ids"].to(DEV)
    model(ids)
    for l, blk in enumerate(LAYERS):
        norm = blk.post_attention_layernorm
        W = blk.self_attn.o_proj.weight.float()
        x = cap[l][0, 1:].float()                                  # [T-1, 1024], BOS dropped
        t = SAE[l]
        pre = x @ t["W_enc"] + t["b_enc"]
        c = pre * (pre > t["threshold"])
        x_empty = x - c @ t["W_dec"]                               # zero-fill every latent
        y_clean, y_empty = F.linear(x, W), F.linear(x_empty, W)
        out_clean = norm(y_clean.to(torch.bfloat16)).float()
        unfrozen = norm(y_empty.to(torch.bfloat16)).float()
        rms = torch.sqrt(y_clean.pow(2).mean(-1, keepdim=True) + norm.eps)
        w1 = 1.0 + norm.weight.float()
        frozen = y_empty / rms * w1
        mine_clean = y_clean / rms * w1                            # exactness: must equal out_clean
        acc[l]["exact"] = max(acc[l]["exact"], float((mine_clean - out_clean).abs().max() / out_clean.abs().max()))
        nc = out_clean.norm(dim=-1)
        acc[l]["shrink"].append(float((y_empty.norm(dim=-1) / y_clean.norm(dim=-1)).mean()))
        acc[l]["unfrozen"].append(float((unfrozen.norm(dim=-1) / nc).mean()))
        acc[l]["frozen"].append(float((frozen.norm(dim=-1) / nc).mean()))
        acc[l]["cos"].append(float(F.cosine_similarity(unfrozen, out_clean, dim=-1).mean()))
for h in hs:
    h.remove()

print("EXACTNESS (frozen formula vs module, no edit; relative max error per layer): max %.2e"
      % max(a["exact"] for a in acc.values()))
print("\n%5s | %15s | %15s | %13s | %20s" % ("layer", "pre-norm shrink", "unfrozen ratio", "frozen ratio", "cos(unfrozen, clean)"))
for l in range(len(LAYERS)):
    a = acc[l]
    print("%5d | %15.3f | %15.3f | %13.3f | %20.3f"
          % (l, np.mean(a["shrink"]), np.mean(a["unfrozen"]), np.mean(a["frozen"]), np.mean(a["cos"])))
print("\nmedian over layers: shrink %.3f | unfrozen %.3f | frozen %.3f | cos %.3f"
      % tuple(np.median([np.mean(acc[l][k]) for l in acc]) for k in ("shrink", "unfrozen", "frozen", "cos")))
