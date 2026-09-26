"""ARE THE CIRCUIT'S LATE-ATTENTION MEMBERS INDUCTION HEADS?

The word test showed they are NOT per-word copiers (purity at chance),
which is what an induction head should look like: it copies whatever
token follows the match, so its output is content-general and the answer
identity rides on the ATTENTION PATTERN. This tests the pattern directly.

For each attention member: the HEAD it lives in (attention SAEs are
trained on the o_proj INPUT, the concatenated head outputs, so each
latent's decoder splits into per-head blocks — the head holding most of
the norm is the member's head), then, for that head, the attention paid
FROM the prediction position TO:

  ans_first   the first-copy occurrence of the ANSWER (match + 1)
              -> the induction-head signature
  prev_first  the first-copy occurrence of the match word
              -> a same-token / duplicate-token head
  self/last   the current token
  other       everything else (baseline)

Reported per head, averaged over prompts, against the uniform baseline
1/n_positions.

  MEMBERS=induction_am_all_gemma_members.jsonl LAYERS=19,22,23 \
    python experiments/050-gemma-circuits/induction_head_test.py
"""
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
import gemma_loader as G  # noqa: E402

MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
TIER = int(os.environ.get("TIER", 2))
MEMBERS = os.environ.get("MEMBERS", "induction_am_all_gemma_members.jsonl")
PROMPTS = os.environ.get("PROMPTS", "induction_gemma_prompts.pt")
WANT = [int(x) for x in os.environ.get("LAYERS", "19,22,23").split(",") if x != ""]
N_PROMPTS = int(os.environ.get("N_PROMPTS", 100))
DEV = torch.device("cuda")
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}

tok = AutoTokenizer.from_pretrained(MODEL_ID)
# eager attention so attentions are returned
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32,
                                             attn_implementation="eager").to(DEV).eval()
H = model.config.num_attention_heads
HD = model.config.head_dim
print("Gemma: %d attention heads x head_dim %d = %d (att SAE input dim)" % (H, HD, H * HD))

rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    if kind == "att" and int(layer) in WANT:
        members[int(layer)] = {int(i): float(a) for i, a in d.items()}
data = torch.load(HERE / PROMPTS, weights_only=False)
rows = data["rows"][:N_PROMPTS]
print("%s | %d prompts | attention members: %s"
      % (MEMBERS, len(rows), {l: len(d) for l, d in sorted(members.items())}))

_TC = {}


def tc(layer):
    if layer not in _TC:
        pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % layer)
        lad = sorted({int(m.group(1)) for f in list_repo_files(REPOS["att"]) for m in [pat.search(f)] if m})
        p = G.CACHE / ("att_layer_%d_w16k_l0_%d.npz" % (layer, lad[min(TIER, len(lad) - 1)]))
        z = np.load(p)
        _TC[layer] = {k: torch.tensor(z[k], device=DEV).float() for k in z.files}
    return _TC[layer]


# ---- which head does each member live in? -------------------------------------
head_of = {}
for layer, d in members.items():
    W = tc(layer)["W_dec"]                       # [d_sae, n_heads*head_dim]
    for li in d:
        blocks = W[li].view(H, HD).norm(dim=-1)
        head_of[(layer, li)] = (int(blocks.argmax()), float((blocks.max() ** 2 / (blocks ** 2).sum())))
print("\nhead assignment (share of decoder norm in the dominant head):")
for layer in sorted(members):
    hs = defaultdict(list)
    for li in members[layer]:
        h, s = head_of[(layer, li)]
        hs[h].append(s)
    print("  L%-2d %s" % (layer, {h: "%d members (norm share %.2f)" % (len(v), float(np.mean(v)))
                                  for h, v in sorted(hs.items())}))


def roles_for(r):
    words = [tok.decode([t]).strip() for t in r["ids"]]
    seq, k = r["meta"]["seq"], r["meta"]["k"]
    pos_first, j = {}, 0
    for p, w in enumerate(words):
        if j < len(seq) and w == seq[j]:
            pos_first[seq[j]] = p; j += 1
    return pos_first.get(seq[k]), pos_first.get(seq[k - 1])


# ---- attention from END ---------------------------------------------------------
attn = defaultdict(lambda: defaultdict(list))     # (layer, head) -> role -> [attn]
with torch.no_grad():
    for r in rows:
        ans_p, prev_p = roles_for(r)
        if ans_p is None or prev_p is None:
            continue
        ids = torch.tensor([r["ids"]], device=DEV)
        out = model(ids, output_attentions=True)
        T = ids.shape[1]
        for layer in members:
            A = out.attentions[layer][0, :, -1, :].float().cpu().numpy()   # [H, T] from END
            for h in range(H):
                a = A[h]
                attn[(layer, h)]["ans_first"].append(float(a[ans_p]))
                attn[(layer, h)]["prev_first"].append(float(a[prev_p]))
                attn[(layer, h)]["self"].append(float(a[T - 1]))
                mask = np.ones(T, dtype=bool); mask[[ans_p, prev_p, T - 1, 0]] = False
                attn[(layer, h)]["other"].append(float(a[mask].mean()) if mask.any() else 0.0)
                attn[(layer, h)]["bos"].append(float(a[0]))

print("\nATTENTION FROM THE PREDICTION POSITION (mean over %d prompts). An induction head puts"
      "\nmass on ans_first; a duplicate-token head on prev_first. 'members' = circuit members in that head."
      % len(rows))
print("  %-10s %8s %10s %10s %8s %8s  %s" % ("head", "members", "ans_first", "prev_first", "self", "bos", "other"))
best = []
for (layer, h), d in sorted(attn.items()):
    n_mem = sum(1 for (l, li), (hh, _) in head_of.items() if l == layer and hh == h)
    m = {k: float(np.mean(v)) for k, v in d.items()}
    best.append((m["ans_first"], layer, h, n_mem, m))
    print("  L%-2d H%-6d %8d %10.3f %10.3f %8.3f %8.3f  %.4f"
          % (layer, h, n_mem, m["ans_first"], m["prev_first"], m["self"], m["bos"], m["other"]))
best.sort(reverse=True)
print("\nTOP INDUCTION HEADS (most attention to ans_first):")
for a, layer, h, n_mem, m in best[:6]:
    print("  L%d.H%d | attn to ans_first %.3f (%.0fx the 'other' baseline %.4f) | circuit members in this head: %d"
          % (layer, h, a, a / max(m["other"], 1e-6), m["other"], n_mem))
json.dump({"%d.%d" % (l, h): m for _, l, h, _, m in best}, open(HERE / "induction_head_attention.json", "w"), indent=1)
print("->", HERE / "induction_head_attention.json")
