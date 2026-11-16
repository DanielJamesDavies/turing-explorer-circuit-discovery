"""117 setup smoke (CPU only): load GPT-2 small + one OAI v5 SAE (attn-out L9), run one IOI prompt from 081's task
file, print the IO-vs-S logit difference and the number of attn-out L9 latents active at END.

  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=src python -X utf8 experiments/117-known-circuits-gpt2/smoke_cpu.py
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import config  # noqa: E402
from model.inference import Inference  # noqa: E402
from sae.public_bank import PublicSAEBank  # noqa: E402

config.model.family = "gpt2-small"
dev = torch.device("cpu")
inf = Inference(dev, family="gpt2-small")
bank = PublicSAEBank(device=dev, family="gpt2-small", preload=False)
task = json.load(open(ROOT / "experiments/081-task-circuits/tasks/gpt2-small/ioi.json"))
row = task["rows"][task["split"]["train"][0]]
tok = torch.tensor([row["tokens"]], dtype=torch.long)
kinds = list(bank.kinds)
cap = {}


def hook(layer_idx, acts):
    if layer_idx == 9:
        cap["x"] = acts[kinds.index("attn")]


inf.disable_compile()
with torch.no_grad():
    out = inf.forward(tok, activations_callback=hook, return_activations=False, tokenize_final=False,
                      all_logits=True)
lg = out[1] if isinstance(out, (tuple, list)) else out
lp = torch.log_softmax(lg[0, -1].float(), -1)
ld = float(lp[row["good"][0]] - lp[row["bad"][0]])
ta, ti = bank.encode(cap["x"], "attn", 9)
n_act = int((ta[0, -1] > 0).sum())
print("prompt:", row["prompt"])
print("logit diff IO-S: %.3f | attn L9 latents active at END: %d (k=%d)" % (ld, n_act, bank.k))
print("SMOKE_OK" if ld > 0 and n_act > 0 else "SMOKE_FAIL")
