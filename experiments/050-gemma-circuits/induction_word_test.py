"""IS THE INDUCTION CIRCUIT'S LATE-ATTENTION BLOCK A BANK OF WORD COPIERS?

The same test that identified Gemma's IOI name movers (048/dla_att_members.py),
applied to induction. For every member of a chosen kind/layer that fires at
the PREDICTION position:

  purity     of the prompts where it fires at END, the share whose ANSWER is
             its single most common answer word (chance = the most frequent
             word's share, ~1/40 here)
  DLA rank   rank of that word among the dataset's answer vocabulary when the
             member's decoder direction is pushed through the unembedding
             (1 = it promotes its own word above every other candidate)

A bank of per-word copiers looks like: high purity, rank 1, one member per
word — which is what the IOI name movers did (22 of 24 at rank 1).

  MEMBERS=induction_am_all_gemma_members.jsonl KIND=att LAYERS=19,22,23 \
    python experiments/050-gemma-circuits/induction_word_test.py
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
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
KIND = os.environ.get("KIND", "att")
WANT = [int(x) for x in os.environ.get("LAYERS", "19,22,23").split(",") if x != ""]
MIN_FIRE = int(os.environ.get("MIN_FIRE", 10))
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    if kind == KIND and int(layer) in WANT:
        members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
data = torch.load(HERE / PROMPTS, weights_only=False)
rows = data["rows"]
answers = sorted({r["target"] for r in rows})
aw = {t: tok.decode([t]).strip() for t in answers}
freq = Counter(r["target"] for r in rows)
chance = max(freq.values()) / len(rows)
print("%s | %d prompts | %d answer words | chance purity %.3f | sites %s"
      % (MEMBERS, len(rows), len(answers), chance, sorted(members)))

_TC = {}


def tc(site):
    if site not in _TC:
        kind, layer = site
        if kind == "mlp":
            p = G.CACHE / ("layer_%d_w%s_l0_%d.npz" % (layer, G.WIDTH, G.tier_l0(layer, TIER)))
        else:
            pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % layer)
            lad = sorted({int(m.group(1)) for f in list_repo_files(REPOS[kind]) for m in [pat.search(f)] if m})
            p = G.CACHE / ("%s_layer_%d_w16k_l0_%d.npz" % (kind, layer, lad[min(TIER, len(lad) - 1)]))
        z = np.load(p)
        _TC[site] = {k: torch.tensor(z[k], device=DEV).to(DTYPE) for k in z.files}
    return _TC[site]


# ---- fire pattern at END ------------------------------------------------------
caps = {}
hs = []
for site in members:
    def mk(_s):
        def cap(x):
            t = tc(_s)
            pre = x[0, -1] @ t["W_enc"] + t["b_enc"]
            c = pre * (pre > t["threshold"])
            idx = torch.tensor(sorted(members[_s]), device=DEV)
            caps[_s] = c[idx].float().cpu()
        return cap
    kind, layer = site
    blk = model.model.layers[layer]
    f = mk(site)
    if kind == "mlp":
        hs.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, _c=f: _c(o)))
    elif kind == "att":
        hs.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, _c=f: _c(i[0])))
    else:
        hs.append(blk.register_forward_hook(lambda m, a, o, _c=f: _c(o[0] if isinstance(o, tuple) else o)))

fire = defaultdict(Counter)
with torch.no_grad():
    for r in rows:
        caps.clear()
        model(torch.tensor([r["ids"]], device=DEV))
        for site, d in members.items():
            a = caps[site]
            for j, li in enumerate(sorted(d)):
                if float(a[j]) > 0:
                    fire[(site, li)][r["target"]] += 1
for h in hs:
    h.remove()

# ---- DLA over the answer vocabulary -------------------------------------------
W_U = model.lm_head.weight.detach().float()
NORM = (1.0 + model.model.norm.weight.detach().float())
ans_t = torch.tensor(answers, device=DEV)


def dla_answers(site, idx):
    kind, layer = site
    d = tc(site)["W_dec"][idx].float()
    if kind == "att":
        d = model.model.layers[layer].self_attn.o_proj.weight.detach().float() @ d
    return (W_U[ans_t] @ (d * NORM)).cpu().numpy()


rank1 = tot = 0
print("\n  %-16s %5s %7s %8s %6s %6s  %s" % ("member", "alpha", "fires", "top word", "purity", "rank", "top-3 promoted words"))
rows_out = []
for site, d in sorted(members.items()):
    for li, alpha in sorted(d.items()):
        c = fire[(site, li)]
        n = sum(c.values())
        if n < MIN_FIRE:
            continue
        top_tok, top_n = c.most_common(1)[0]
        purity = top_n / n
        lg = dla_answers(site, li)
        order = np.argsort(-lg)
        rank = int(np.where(np.array(answers)[order] == top_tok)[0][0]) + 1
        rank1 += int(rank == 1); tot += 1
        rows_out.append(dict(kind=site[0], layer=site[1], latent=li, alpha=alpha, fires=n,
                             top_word=aw[top_tok], purity=purity, rank=rank))
        print("  %-3s L%-2d %6d %5.2f %7d %8s %6.2f %6d  %s"
              % (site[0], site[1], li, alpha, n, aw[top_tok], purity, rank,
                 " ".join(aw[answers[j]] for j in order[:3])))
print("\n  members tested: %d | rank 1: %d (%.0f%%) | rank <= 3: %d | chance rank-1 = 1/%d"
      % (tot, rank1, 100 * rank1 / max(tot, 1), sum(1 for r in rows_out if r["rank"] <= 3), len(answers)))
if rows_out:
    p = np.array([r["purity"] for r in rows_out])
    print("  purity: median %.2f | max %.2f | >= 0.5: %d | chance %.3f"
          % (np.median(p), p.max(), int((p >= 0.5).sum()), chance))
    print("  distinct top words covered: %d of %d" % (len({r["top_word"] for r in rows_out}), len(answers)))
json.dump(rows_out, open(HERE / MEMBERS.replace("_members.jsonl", "_wordtest_%s.json" % KIND), "w"), indent=1)
print("->", MEMBERS.replace("_members.jsonl", "_wordtest_%s.json" % KIND))
