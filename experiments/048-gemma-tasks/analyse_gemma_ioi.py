"""POSITION-RESOLVED MEMBER PROFILES for the Gemma IOI circuit.

IOI (Wang et al. 2022) is positional: duplicate-token / induction heads
read the repeated subject (S2) against its first occurrence (S1),
S-inhibition heads carry "not S" to END, name movers at END copy IO.
Roles here: IO (indirect object's first mention), S1 (subject's first
mention), S2 (subject's second mention), END (prediction position),
other. Members are classified by where their activation mass sits.

  PYTHONPATH=src python experiments/048-gemma-tasks/analyse_gemma_ioi.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
import gemma_loader as G  # noqa: E402

MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
TIER = int(os.environ.get("TIER", 2))
TOP = int(os.environ.get("TOP", 25))
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()

MEMBERS = os.environ.get("MEMBERS", "ioi_att_mlp_res_gemma_members.jsonl")
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = (k.split("/") if "/" in k else ("mlp", k))
    members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
data = torch.load(HERE / "ioi_gemma_prompts.pt", weights_only=False)
rows = data["rows"]
print("Gemma IOI | %s | %d prompts | %d members over %d sites | EF_ho %.3f"
      % (MEMBERS, len(rows), sum(len(d) for d in members.values()), len(members), rec["EF_ho"]))

_TC = {}


def tc(site):
    if site not in _TC:
        kind, layer = site
        if kind == "mlp":
            p = G.CACHE / ("layer_%d_w%s_l0_%d.npz" % (layer, G.WIDTH, G.tier_l0(layer, TIER)))
        else:
            import re
            from huggingface_hub import list_repo_files
            pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % layer)
            lad = sorted({int(m.group(1)) for f in list_repo_files(REPOS[kind]) for m in [pat.search(f)] if m})
            p = G.CACHE / ("%s_layer_%d_w16k_l0_%d.npz" % (kind, layer, lad[min(TIER, len(lad) - 1)]))
        z = np.load(p)
        _TC[site] = {k: torch.tensor(z[k], device=DEV).to(DTYPE) for k in z.files}
    return _TC[site]


def roles_for(r):
    ids, words = r["ids"], [tok.decode([t]) for t in r["ids"]]
    io, s = r["meta"]["io"], r["meta"]["s"]
    roles = ["other"] * len(ids)
    seen_s = 0
    for p, w in enumerate(words):
        if w.strip() == io and "IO" not in roles:
            roles[p] = "IO"
        elif w.strip() == s:
            seen_s += 1
            roles[p] = "S1" if seen_s == 1 else "S2"
    roles[len(ids) - 1] = "END"
    return roles


caps = {}
hs = []
for site in members:
    def mk(_s):
        def cap(x):
            t = tc(_s)
            pre = x @ t["W_enc"] + t["b_enc"]
            c = pre * (pre > t["threshold"])
            idx = torch.tensor(sorted(members[_s]), device=DEV)
            caps[_s] = c[0, :, idx].float().cpu()
        return cap
    kind, layer = site
    blk = model.model.layers[layer]
    cap_fn = mk(site)
    if kind == "mlp":
        hs.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, _c=cap_fn: _c(o)))
    elif kind == "att":
        hs.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, _c=cap_fn: _c(i[0])))
    else:
        hs.append(blk.register_forward_hook(lambda m, a, o, _c=cap_fn: _c(o[0] if isinstance(o, tuple) else o)))

role_sum = defaultdict(lambda: defaultdict(float))
fire = defaultdict(int)
with torch.no_grad():
    for r in rows:
        roles = roles_for(r)
        caps.clear()
        model(torch.tensor([r["ids"]], device=DEV))
        for site, d in members.items():
            a = caps.get(site)
            if a is None:
                continue
            for j, li in enumerate(sorted(d)):
                v = a[:, j]
                if float(v.max()) > 0:
                    fire[(site, li)] += 1
                for p, ro in enumerate(roles):
                    if p > 0 and float(v[p]) > 0:      # BOS excluded
                        role_sum[(site, li)][ro] += float(v[p])
for h in hs:
    h.remove()

out = []
for site, d in members.items():
    for li, alpha in d.items():
        rs = role_sum[(site, li)]
        tot = sum(rs.values())
        if tot == 0:
            out.append({"kind": site[0], "layer": site[1], "latent": li, "alpha": alpha, "fires": 0,
                        "dominant": "silent", "shares": {}})
            continue
        dom = max(rs, key=rs.get)
        out.append({"kind": site[0], "layer": site[1], "latent": li, "alpha": alpha, "fires": fire[(site, li)],
                    "dominant": dom, "shares": {k: v / tot for k, v in rs.items()}})

by = defaultdict(list)
for r in out:
    by[r["dominant"]].append(r)
print("\nMEMBERS BY DOMINANT ROLE:")
print("  %-8s %5s | by layer" % ("role", "n"))
for ro, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
    cnt = defaultdict(int)
    for r in lst:
        cnt["%s%d" % (r["kind"][0], r["layer"])] += 1
    print("  %-8s %5d | %s" % (ro, len(lst), dict(sorted(cnt.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))))))
for k in ("IO", "S1", "S2", "END"):
    conc = [r for r in out if r["shares"].get(k, 0) >= 0.6]
    cnt = defaultdict(int)
    for r in conc:
        cnt["%s%d" % (r["kind"][0], r["layer"])] += 1
    print("  >=60%% mass on %-3s: %3d members | by kind+layer %s" % (k, len(conc), dict(sorted(cnt.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))))))
print("\nTOP %d concentrated END members (name-mover candidates), by alpha*fires:" % TOP)
end = sorted([r for r in out if r["shares"].get("END", 0) >= 0.6],
             key=lambda r: -(r["alpha"] * r["fires"]))[:TOP]
for r in end:
    print("  %-4s L%-2d %6d | alpha %5.2f | fires %3d/%d" % (r["kind"], r["layer"], r["latent"], r["alpha"], r["fires"], len(rows)))
print("\nTOP %d concentrated S2 members (duplicate-token / S-inhibition candidates):" % TOP)
s2 = sorted([r for r in out if r["shares"].get("S2", 0) >= 0.6],
            key=lambda r: -(r["alpha"] * r["fires"]))[:TOP]
for r in s2:
    print("  %-4s L%-2d %6d | alpha %5.2f | fires %3d/%d" % (r["kind"], r["layer"], r["latent"], r["alpha"], r["fires"], len(rows)))
outp = MEMBERS.replace("_members.jsonl", "_member_roles.json")
json.dump(out, open(HERE / outp, "w"), indent=1)
print("\n->", outp)
