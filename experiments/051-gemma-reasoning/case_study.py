"""CASE STUDY of one two-hop inference, feature by feature.

Default: "The main language of the country that contains Munich is"
  A = Munich, bridge = GERMANY (never written), answer = German.

Prints, for the circuit's bridge features:
  1. per-bridge profile on prompts that NAME each entity (no city present)
  2. TOKEN-BY-TOKEN activation on the two-hop prompt — where in the prompt
     does the unsaid intermediate switch on?
  3. the same feature on a matched prompt with a different bridge (Milan)
  4. direct logit attribution: what does the feature's output direction
     promote in the vocabulary?
  5. the swap, on this single prompt: top-5 next tokens before and after
     replacing bridge-1 features with bridge-2 features

  B1=Germany B2=China CITY=Munich CITY2=Milan FAMILY=country_language \
    python experiments/051-gemma-reasoning/case_study.py
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
MEMBERS = os.environ.get("MEMBERS", "twohop_am_all_gemma_members.jsonl")
BRIDGE = os.environ.get("BRIDGE", "twohop_am_all_gemma_bridge.json")
B1 = os.environ.get("B1", "Germany")
B2 = os.environ.get("B2", "China")
CITY = os.environ.get("CITY", "Munich")
CITY2 = os.environ.get("CITY2", "Milan")
FAMILY = os.environ.get("FAMILY", "country_language")
MIN_RATIO = float(os.environ.get("MIN_RATIO", 3.0))
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
rec = json.loads(open(HERE / MEMBERS).readline())
alphas = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    alphas[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
bj = json.load(open(HERE / BRIDGE))
two = json.load(open(HERE / "twohop_rows.json"))
probes = json.load(open(HERE / "bridge_probes.json"))
bridges = sorted(probes)

feats = defaultdict(list)          # bridge -> [((kind,layer), latent, alpha)]
for r in bj["top"]:
    if r["ratio"] >= MIN_RATIO and r["selectivity"] >= 0.4:
        feats[r["bridge"]].append(((r["kind"], r["layer"]), r["latent"],
                                   alphas[(r["kind"], r["layer"])][r["latent"]]))
print("CASE STUDY: %s (%s) vs %s (%s), family %s" % (B1, CITY, B2, CITY2, FAMILY))
print("  %s features in the circuit: %s" % (B1, ["%s L%d.%d" % (s[0], s[1], li) for s, li, _ in feats[B1]]))
print("  %s features in the circuit: %s" % (B2, ["%s L%d.%d" % (s[0], s[1], li) for s, li, _ in feats[B2]]))

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


WATCH = [(s, li) for b in (B1, B2) for s, li, _ in feats[b]]
caps = {}
hs = []
for site in {s for s, _ in WATCH}:
    def mk(_s):
        def cap(x):
            t = tc(_s)
            pre = x[0] @ t["W_enc"] + t["b_enc"]
            caps[_s] = (pre * (pre > t["threshold"])).float().cpu().numpy()   # [T, d_sae]
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


def acts(text):
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
    caps.clear()
    with torch.no_grad():
        model(ids)
    return [tok.decode([t]) for t in ids[0].tolist()], {k: caps[s][:, li] for k, (s, li) in
                                                        zip([(s, li) for s, li in WATCH], WATCH)}


# ---- 1. per-bridge profile ------------------------------------------------------
print("\n[1] PER-BRIDGE PROFILE — mean activation at the final token of prompts that NAME each")
print("    entity ('the government of X'); no city is mentioned. Top 5 of %d entities:" % len(bridges))
prof = {k: {} for k in WATCH}
for b in bridges:
    vals = defaultdict(list)
    for p in probes[b]:
        _, a = acts(p)
        for k in WATCH:
            vals[k].append(float(a[k][-1]))
    for k in WATCH:
        prof[k][b] = float(np.mean(vals[k]))
for (s, li) in WATCH:
    top = sorted(prof[(s, li)].items(), key=lambda kv: -kv[1])[:5]
    print("    %-3s L%-2d %6d : %s" % (s[0], s[1], li, "  ".join("%s %.2f" % t for t in top)))

# ---- 2/3. token-by-token on the two-hop prompt ------------------------------------
row = next(r for r in two if r["meta"]["city"] == CITY and r["meta"]["family"] == FAMILY)
row2 = next(r for r in two if r["meta"]["city"] == CITY2 and r["meta"]["family"] == FAMILY)
for label, r in (("%s (bridge %s)" % (CITY, B1), row), ("%s (bridge %s)" % (CITY2, r_b2 := row2["meta"]["bridge"]), row2)):
    words, a = acts(r["prompt"])
    print("\n[2] TOKEN-BY-TOKEN — %s" % label)
    print("    prompt: %r  -> answer %r" % (r["prompt"], r["meta"]["answer"]))
    hdr = "    %-22s" % "token"
    for (s, li) in WATCH:
        hdr += " %10s" % ("%s%d.%d" % (s[0][0], s[1], li))
    print(hdr)
    for p, w in enumerate(words):
        line = "    %2d %-19s" % (p, repr(w)[:19])
        for k in WATCH:
            v = float(a[k][p])
            line += " %10s" % ("%.2f" % v if v > 0.005 else ".")
        print(line)

for h in hs:
    h.remove()

# ---- 4. what does the feature promote? -------------------------------------------
W_U = model.lm_head.weight.detach().float()
NORM = (1.0 + model.model.norm.weight.detach().float())
print("\n[4] DIRECT LOGIT ATTRIBUTION — top tokens promoted by each feature's output direction")
for (s, li) in WATCH:
    kind, layer = s
    d = tc(s)["W_dec"][li].float()
    if kind == "att":
        d = model.model.layers[layer].self_attn.o_proj.weight.detach().float() @ d
    lg = W_U @ (d * NORM)
    top = torch.topk(lg, 8).indices.tolist()
    print("    %-3s L%-2d %6d : %s" % (kind, layer, li, " ".join(repr(tok.decode([t])) for t in top)))

# ---- 5. the swap on this one prompt ------------------------------------------------
off = defaultdict(list)
for s, li, _ in feats[B1]:
    off[tuple(s)].append(li)
on = defaultdict(dict)
for s, li, _ in feats[B2]:
    on[tuple(s)][li] = next(r["match"] for r in bj["top"]
                            if r["bridge"] == B2 and r["latent"] == li and r["layer"] == s[1])


class Edit:
    def __init__(self, off, on):
        self.off, self.on, self.handles = off, on, []

    def __enter__(self):
        for site in set(self.off) | set(self.on):
            kind, layer = site
            blk = model.model.layers[layer]

            def edit(x, _s=site):
                t = tc(_s)
                pre = x @ t["W_enc"] + t["b_enc"]
                c = pre * (pre > t["threshold"])
                chat = c.clone()
                for i in self.off.get(_s, []):
                    chat[0, 1:, i] = 0.0
                for i, v in self.on.get(_s, {}).items():
                    chat[0, 1:, i] = float(v)
                delta = (chat - c).to(x.dtype) @ t["W_dec"]
                delta[:, 0] = 0
                return x + delta
            if kind == "mlp":
                self.handles.append(blk.post_feedforward_layernorm.register_forward_hook(
                    lambda m, i, o, _e=edit: _e(o)))
            elif kind == "att":
                self.handles.append(blk.self_attn.o_proj.register_forward_pre_hook(
                    lambda m, i, _e=edit: (_e(i[0]),) + tuple(i[1:])))
            else:
                self.handles.append(blk.register_forward_hook(
                    lambda m, a, o, _e=edit: (_e(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else _e(o)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def top5(text, edit=None):
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
    with torch.no_grad():
        if edit is None:
            lg = model(ids).logits[0, -1].float()
        else:
            with edit:
                lg = model(ids).logits[0, -1].float()
    pr = torch.softmax(lg, -1)
    t = torch.topk(pr, 5)
    return [(tok.decode([int(i)]), float(p)) for p, i in zip(t.values, t.indices)], lg


print("\n[5] THE SWAP on this single prompt: %r" % row["prompt"])
b_top, b_lg = top5(row["prompt"])
p_top, p_lg = top5(row["prompt"], Edit(dict(off), dict(on)))
print("    before: " + "  ".join("%r %.3f" % t for t in b_top))
print("    after : " + "  ".join("%r %.3f" % t for t in p_top))
ans1 = row["meta"]["answer"]
ans2 = next(r["meta"]["answer"] for r in two if r["meta"]["bridge"] == B2 and r["meta"]["family"] == FAMILY)


def tid(word):
    a = tok(row["prompt"], return_tensors="pt")["input_ids"][0].tolist()
    b = tok(row["prompt"] + " " + word, return_tensors="pt")["input_ids"][0].tolist()
    return b[len(a)]


t1, t2 = tid(ans1), tid(ans2)
print("    log-odds %r - %r : before %+.2f  ->  after %+.2f   (shift %+.2f)"
      % (ans1, ans2, float(torch.log_softmax(b_lg, -1)[t1] - torch.log_softmax(b_lg, -1)[t2]),
         float(torch.log_softmax(p_lg, -1)[t1] - torch.log_softmax(p_lg, -1)[t2]),
         float(torch.log_softmax(b_lg, -1)[t1] - torch.log_softmax(b_lg, -1)[t2])
         - float(torch.log_softmax(p_lg, -1)[t1] - torch.log_softmax(p_lg, -1)[t2])))
