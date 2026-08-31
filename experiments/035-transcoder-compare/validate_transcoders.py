"""End-to-end validation of the converted transcoders on REAL GPT-2
activations: fed the full ln_2 output (the convention the conversion
targets), each transcoder should predict its layer's MLP output well.

Reports L0, relative error and fraction of variance unexplained per
layer. Run after convert_transcoders.py; it is the check that the two
folds (input pre-bias, LayerNorm affine) are right in practice and not
just on random vectors.

  python validate_transcoders.py
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, GPT2LMHeadModel

HERE = Path(__file__).parent
TEXT = ("The Eiffel Tower is in Paris, the capital of France. "
        "Machine learning models process text one token at a time. "
        "In 1969, astronauts walked on the surface of the Moon.")

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").float().eval()
ids = tok(TEXT, return_tensors="pt")["input_ids"]

print("layer |    L0 | rel_err |   FVU")
rows = []
for L in range(12):
    sd = {k: v.float() for k, v in
          load_file(str(HERE / ("transcoders_ct/layer_%d.safetensors" % L))).items()}
    cap = {}

    def hook(mod, inp, out):
        cap["xin"] = inp[0]
        cap["out"] = out

    h = model.transformer.h[L].mlp.register_forward_hook(hook)
    with torch.no_grad():
        model(ids)
    h.remove()

    feats = F.relu(cap["xin"] @ sd["W_enc"].T + sd["b_enc"])
    rec = feats @ sd["W_dec"] + sd["b_dec"]
    tgt = cap["out"]
    l0 = float((feats > 0).sum(-1).float().mean())
    rel = float((rec - tgt).norm() / tgt.norm())
    fvu = float(((rec - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum())
    rows.append((L, l0, rel, fvu))
    print("  %2d  | %5.1f |   %.3f | %.3f" % (L, l0, rel, fvu))

bad = [r for r in rows if r[3] > 0.6]
print("\nlayers with FVU > 0.6:", [r[0] for r in bad] or "none")
print("median FVU: %.3f" % sorted(r[3] for r in rows)[len(rows) // 2])
