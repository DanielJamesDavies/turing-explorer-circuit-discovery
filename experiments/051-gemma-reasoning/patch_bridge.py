"""SWAP THE INTERMEDIATE, CHANGE THE CONCLUSION.

bridge_features.py found members that fire when, and only when, the model
must COMPUTE their entity — e.g. an L22 attention latent that switches on
for Munich/Hamburg/Frankfurt (bridge Germany) and not for Milan/Naples
(bridge Italy), while the word "Germany" never appears.

If those members ARE the model's intermediate rather than a correlate of
it, then replacing them should change the answer: on a Munich prompt,
turn Germany's features off and Japan's features on, and the model should
prefer Tokyo over Berlin — without the prompt changing at all.

Per ordered pair (B1 -> B2) of bridges with strong features:
  baseline   log p(answer_B1) - log p(answer_B2) on B1's prompts
  patched    same, with B1 features zeroed and B2 features injected at the
             positions where B1's features were active
  control    the same number of RANDOM circuit members, zeroed/injected
             with matched magnitudes (does any large edit flip it?)

Success = the margin moves toward B2, ideally past zero (argmax flips).

  MEMBERS=twohop_am_all_gemma_members.jsonl BRIDGE=twohop_am_all_gemma_bridge.json \
    N_PAIRS=12 python experiments/051-gemma-reasoning/patch_bridge.py
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
MIN_RATIO = float(os.environ.get("MIN_RATIO", 3.0))
N_PAIRS = int(os.environ.get("N_PAIRS", 12))
SCALE = float(os.environ.get("SCALE", 1.0))     # multiplier on the donor activation
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
rng = np.random.default_rng(0)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
bj = json.load(open(HERE / BRIDGE))
two = json.load(open(HERE / "twohop_rows.json"))

# bridge -> [(site, latent)] for members that passed the ratio bar
by_bridge = defaultdict(list)
for r in bj["top"]:
    if r["ratio"] >= MIN_RATIO and r["selectivity"] >= 0.4:
        by_bridge[r["bridge"]].append(((r["kind"], r["layer"]), r["latent"], r["match"]))
strong = {b: v for b, v in by_bridge.items() if v}
print("%s | bridges with strong features: %d (%s)"
      % (MEMBERS, len(strong), ", ".join("%s:%d" % (b, len(v)) for b, v in sorted(strong.items()))))

# answers per bridge (the fact the model should retrieve)
ans_of = {}
for r in two:
    ans_of.setdefault((r["meta"]["family"], r["meta"]["bridge"]), r["meta"]["answer"])
prompts_of = defaultdict(list)
for r in two:
    prompts_of[(r["meta"]["family"], r["meta"]["bridge"])].append(r)

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


class Edit:
    """Set chosen latents to given values (or 0) at chosen positions, writing
    back through the decoder: x <- x + (chat - c) @ W_dec. Also records where
    the 'off' latents were active, so the donor lands in the same places."""

    def __init__(self, off, on, pos=None):
        self.off, self.on, self.pos = off, on, pos      # {site: [idx]}, {site: {idx: val}}
        self.handles = []
        self.active = defaultdict(list)

    def __enter__(self):
        sites = set(self.off) | set(self.on)
        for site in sites:
            kind, layer = site
            blk = model.model.layers[layer]

            def edit(x, _s=site):
                t = tc(_s)
                pre = x @ t["W_enc"] + t["b_enc"]
                c = pre * (pre > t["threshold"])
                chat = c.clone()
                mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
                if self.pos is None:
                    mask[1:] = True
                else:
                    mask[self.pos] = True
                for i in self.off.get(_s, []):
                    col = chat[0, :, i]
                    self.active[_s].append((i, (col > 0).nonzero().flatten().tolist()))
                    chat[0, mask, i] = 0.0
                for i, v in self.on.get(_s, {}).items():
                    chat[0, mask, i] = float(v)
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


def margin(prompt, t1, t2, off=None, on=None):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    with torch.no_grad():
        if off or on:
            with Edit(off or {}, on or {}):
                lg = model(ids).logits[0, -1].float()
        else:
            lg = model(ids).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return float(lp[t1] - lp[t2]), int(lg.argmax())


def first_after(prefix, word):
    a = tok(prefix, return_tensors="pt")["input_ids"][0].tolist()
    b = tok(prefix + word, return_tensors="pt")["input_ids"][0].tolist()
    return b[len(a)] if (b[:len(a)] == a and len(b) > len(a)) else None


all_members = [(s, li) for s, d in members.items() for li in d]
pairs, done = [], set()
for fam in {r["meta"]["family"] for r in two}:
    bs = [b for b in strong if (fam, b) in prompts_of]
    for i, b1 in enumerate(bs):
        for b2 in bs[i + 1:]:
            if ans_of[(fam, b1)] == ans_of[(fam, b2)]:
                continue
            pairs.append((fam, b1, b2))
rng.shuffle(pairs)
pairs = pairs[:N_PAIRS]
print("testing %d (family, B1 -> B2) pairs\n" % len(pairs))

print("  %-16s %-11s -> %-11s %8s %9s %9s %8s  flips" % ("family", "B1", "B2", "base", "patched", "control", "shift"))
res = []
for fam, b1, b2 in pairs:
    rows = prompts_of[(fam, b1)][:6]
    a1, a2 = ans_of[(fam, b1)], ans_of[(fam, b2)]
    off = defaultdict(list)
    for site, li, _ in strong[b1]:
        off[tuple(site)].append(li)
    on = defaultdict(dict)
    for site, li, m in strong[b2]:
        on[tuple(site)][li] = m * SCALE
    n_edit = sum(len(v) for v in off.values()) + sum(len(v) for v in on.values())
    # control: same number of random members, injected with matched magnitude
    pick = [all_members[i] for i in rng.choice(len(all_members), min(n_edit, len(all_members)), replace=False)]
    c_on = defaultdict(dict)
    mags = [m for _, _, m in strong[b2]] or [3.0]
    for s, li in pick:
        c_on[s][li] = float(rng.choice(mags))
    base_v, patch_v, ctrl_v, flips = [], [], [], 0
    for r in rows:
        t1, t2 = first_after(r["prompt"], " " + a1), first_after(r["prompt"], " " + a2)
        if t1 is None or t2 is None or t1 == t2:
            continue
        b_, _ = margin(r["prompt"], t1, t2)
        p_, am = margin(r["prompt"], t1, t2, off=dict(off), on=dict(on))
        c_, _ = margin(r["prompt"], t1, t2, on=dict(c_on))
        base_v.append(b_); patch_v.append(p_); ctrl_v.append(c_); flips += int(p_ < 0)
    if not base_v:
        continue
    B, P, C = float(np.mean(base_v)), float(np.mean(patch_v)), float(np.mean(ctrl_v))
    res.append(dict(family=fam, b1=b1, b2=b2, base=B, patched=P, control=C, shift=B - P,
                    flips=flips, n=len(base_v), n_edit=n_edit))
    print("  %-16s %-11s -> %-11s %8.2f %9.2f %9.2f %8.2f  %d/%d"
          % (fam, b1, b2, B, P, C, B - P, flips, len(base_v)))

if res:
    sh = np.array([r["shift"] for r in res])
    cs = np.array([r["base"] - r["control"] for r in res])
    print("\n  SHIFT toward B2's answer: median %+.2f (control %+.2f) | pairs where the patch moved it > 1.0: %d/%d"
          % (float(np.median(sh)), float(np.median(cs)), int((sh > 1.0).sum()), len(res)))
    print("  argmax FLIPPED to B2's answer on %d of %d prompts"
          % (sum(r["flips"] for r in res), sum(r["n"] for r in res)))
json.dump(res, open(HERE / "patch_bridge_results.json", "w"), indent=1)
print("->", HERE / "patch_bridge_results.json")
