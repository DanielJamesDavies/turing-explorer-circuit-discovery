"""DOES THE TWO-HOP CIRCUIT CONTAIN THE UNSAID INTERMEDIATE?

The prompt says Dallas and asks for a capital; the answer is Austin; the
bridge TEXAS is never written. If reasoning is happening in the members,
some member should represent Texas while the model is answering.

Three measurements per member:

  1. BRIDGE PROFILE (explicit prompts, `bridge_probes.json`): activation
     at the final token of prompts ending in each bridge entity ("the
     government of Texas"). Gives each member a preferred bridge and a
     selectivity = preferred / total.
  2. FIRING ON TWO-HOP PROMPTS, where the bridge never appears: mean
     activation on prompts whose bridge IS the member's preferred bridge
     vs prompts whose bridge is not. The ratio is the effect we are
     after — a bridge feature fires when the model must COMPUTE that
     bridge, without the word being present.
  3. CITY GENERALISATION: each bridge has several cities (Dallas,
     Houston, San Antonio -> Texas). A genuine bridge feature fires for
     ALL of them; a city feature fires for one. Reported as the fraction
     of the preferred bridge's cities on which the member fires.

Controls: (a) the one-hop prompts, where the bridge IS written — a
bridge feature should fire there too, and more strongly; (b) shuffled
labels, i.e. the same statistics against a random preferred bridge.

  MEMBERS=twohop_am_all_gemma_members.jsonl \
    python experiments/051-gemma-reasoning/bridge_features.py
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
TOP = int(os.environ.get("TOP", 25))
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
two = json.load(open(HERE / "twohop_rows.json"))
one = json.load(open(HERE / "onehop_rows.json"))
probes = json.load(open(HERE / "bridge_probes.json"))
bridges = sorted(probes)
print("%s | %d members over %d sites | %d two-hop prompts | %d bridges | EF_ho %s"
      % (MEMBERS, sum(len(d) for d in members.values()), len(members), len(two), len(bridges), rec.get("EF_ho")))

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


caps = {}
hs = []
for site in members:
    def mk(_s):
        def cap(x):
            t = tc(_s)
            pre = x[0] @ t["W_enc"] + t["b_enc"]
            c = pre * (pre > t["threshold"])
            idx = torch.tensor(sorted(members[_s]), device=DEV)
            caps[_s] = c[:, idx].float().cpu().numpy()      # [T, n_members]
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

KEYS = [(s, li) for s in sorted(members) for li in sorted(members[s])]
KI = {k: j for j, k in enumerate(KEYS)}
NM = len(KEYS)


def run(text, mode="last"):
    """member activations for one prompt: at the final token, or max over positions."""
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
    caps.clear()
    with torch.no_grad():
        model(ids)
    v = np.zeros(NM, dtype=np.float32)
    for s in members:
        a = caps[s]
        arr = a[-1] if mode == "last" else a[1:].max(axis=0)   # skip BOS for max
        for j, li in enumerate(sorted(members[s])):
            v[KI[(s, li)]] = arr[j]
    return v


# ---- 1. bridge profile from explicit prompts ------------------------------------
print("\n[1/3] bridge profile over %d bridges x %d prompts ..." % (len(bridges), len(probes[bridges[0]])), flush=True)
prof = np.zeros((len(bridges), NM), dtype=np.float32)
for bi, b in enumerate(bridges):
    prof[bi] = np.mean([run(p, "last") for p in probes[b]], axis=0)
pref = prof.argmax(axis=0)
sel = prof.max(axis=0) / np.maximum(prof.sum(axis=0), 1e-6)
print("      members with a preferred bridge (any activation): %d of %d | selectivity median %.2f"
      % (int((prof.max(axis=0) > 0).sum()), NM, float(np.median(sel[prof.max(axis=0) > 0]))))

# ---- 2. firing on two-hop prompts (bridge never written) -------------------------
print("[2/3] two-hop prompts (%d) ..." % len(two), flush=True)
two_act = np.stack([run(r["prompt"], "max") for r in two])          # [P, NM]
two_bridge = np.array([bridges.index(r["meta"]["bridge"]) for r in two])
two_city = [r["meta"]["city"] for r in two]
match = np.zeros(NM); nonmatch = np.zeros(NM)
for j in range(NM):
    m = two_bridge == pref[j]
    match[j] = two_act[m, j].mean() if m.any() else 0.0
    nonmatch[j] = two_act[~m, j].mean() if (~m).any() else 0.0
ratio = match / np.maximum(nonmatch, 1e-6)

# city generalisation: fraction of the preferred bridge's cities where it fires
cities_of = defaultdict(set)
for r in two:
    cities_of[bridges.index(r["meta"]["bridge"])].add(r["meta"]["city"])
gen = np.zeros(NM)
for j in range(NM):
    cs = cities_of.get(pref[j], set())
    if not cs:
        continue
    hit = 0
    for c in cs:
        idx = [i for i, r in enumerate(two) if r["meta"]["city"] == c]
        hit += int(any(two_act[i, j] > 0 for i in idx))
    gen[j] = hit / len(cs)

# ---- 3. one-hop control (bridge IS written) --------------------------------------
print("[3/3] one-hop control (%d) ..." % len(one), flush=True)
one_act = np.stack([run(r["prompt"], "max") for r in one])
one_bridge = np.array([bridges.index(r["meta"]["bridge"]) for r in one])
one_match = np.zeros(NM)
for j in range(NM):
    m = one_bridge == pref[j]
    one_match[j] = one_act[m, j].mean() if m.any() else 0.0
for h in hs:
    h.remove()

# ---- shuffled-label null ----------------------------------------------------------
rng = np.random.default_rng(0)
null_ratio = np.zeros(NM)
for j in range(NM):
    fake = rng.integers(0, len(bridges))
    m = two_bridge == fake
    nm = two_act[~m, j].mean() if (~m).any() else 0.0
    null_ratio[j] = (two_act[m, j].mean() if m.any() else 0.0) / max(nm, 1e-6)

live = prof.max(axis=0) > 0
print("\nBRIDGE-MATCH EFFECT on two-hop prompts (activation when the model must COMPUTE the")
print("member's preferred bridge, vs when it must compute a different one):")
print("  members with a bridge profile: %d | median ratio %.2f | shuffled-label null %.2f"
      % (int(live.sum()), float(np.median(ratio[live])), float(np.median(null_ratio[live]))))
for thr in (1.5, 2.0, 3.0):
    print("    ratio >= %.1f: %3d members (null %3d)" % (thr, int((ratio[live] >= thr).sum()), int((null_ratio[live] >= thr).sum())))

order = np.argsort(-(ratio * np.log1p(match)))
rows_out = []
print("\nTOP %d BRIDGE FEATURES (fire when the unsaid bridge is theirs):" % TOP)
print("  %-16s %5s %-12s %5s %8s %9s %6s %8s" % ("member", "alpha", "bridge", "sel", "two-match", "two-other", "ratio", "cities"))
for j in order[:TOP]:
    if not live[j]:
        continue
    (kind, layer), li = KEYS[j]
    rows_out.append(dict(kind=kind, layer=layer, latent=li, bridge=bridges[pref[j]],
                         selectivity=float(sel[j]), match=float(match[j]), nonmatch=float(nonmatch[j]),
                         ratio=float(ratio[j]), city_gen=float(gen[j]), one_hop=float(one_match[j])))
    print("  %-3s L%-2d %8d %5.2f %-12s %5.2f %8.2f %9.2f %6.1f %7.0f%%"
          % (kind, layer, li, members[(kind, layer)][li], bridges[pref[j]], sel[j],
             match[j], nonmatch[j], ratio[j], 100 * gen[j]))
json.dump(dict(bridges=bridges, top=rows_out,
               summary=dict(n_members=NM, n_live=int(live.sum()),
                            median_ratio=float(np.median(ratio[live])),
                            median_null=float(np.median(null_ratio[live])))),
          open(HERE / MEMBERS.replace("_members.jsonl", "_bridge.json"), "w"), indent=1)
print("\n->", MEMBERS.replace("_members.jsonl", "_bridge.json"))
