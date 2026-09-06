"""Per-position DENSITY + NAME-IDENTITY test for a Gemma IOI circuit (att/mlp sites).
Fixes the mass-share bias of analyse_gemma_ioi.py (END is one position vs ~11 "other").

  MEMBERS=ioi_att_mlp_gemma_members.jsonl PYTHONPATH=src python experiments/048-gemma-tasks/analyse_ioi_density.py

(1) density: mean activation at role position / mean over 'other' positions
(2) name movers: for att members, does firing at END partition prompts by IO name?
(3) S-inhibition/dup-token: do S2 members fire at S2 regardless of which name S is?
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

HERE = Path("experiments/048-gemma-tasks")
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
import gemma_loader as G  # noqa: E402

MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
TIER = 2
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
MEMBERS = os.environ.get("MEMBERS", "ioi_att_mlp_gemma_members.jsonl")
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
data = torch.load(HERE / "ioi_gemma_prompts.pt", weights_only=False)
rows = data["rows"]
names = Counter(r["meta"]["io"] for r in rows)
print("IO names:", len(names), dict(names.most_common(8)))
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


def roles_for(r):
    words = [tok.decode([t]) for t in r["ids"]]
    io, s = r["meta"]["io"], r["meta"]["s"]
    roles = ["other"] * len(words)
    seen = 0
    for p, w in enumerate(words):
        if w.strip() == io and "IO" not in roles:
            roles[p] = "IO"
        elif w.strip() == s:
            seen += 1
            roles[p] = "S1" if seen == 1 else "S2"
    roles[-1] = "END"
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
    f = mk(site)
    if kind == "mlp":
        hs.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, _c=f: _c(o)))
    else:
        hs.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, _c=f: _c(i[0])))

dens = defaultdict(lambda: defaultdict(list))
end_fire_names = defaultdict(Counter)
end_fire_s = defaultdict(Counter)
s2_fire_names = defaultdict(Counter)
with torch.no_grad():
    for r in rows:
        roles = roles_for(r)
        caps.clear()
        model(torch.tensor([r["ids"]], device=DEV))
        for site, d in members.items():
            a = caps[site]
            for j, li in enumerate(sorted(d)):
                v = a[:, j].numpy()
                for ro in ("IO", "S1", "S2", "END", "other"):
                    ps = [p for p, x in enumerate(roles) if x == ro and p > 0]
                    if ps:
                        dens[(site, li)][ro].append(float(v[ps].mean()))
                if v[-1] > 0:
                    end_fire_names[(site, li)][r["meta"]["io"]] += 1
                    end_fire_s[(site, li)][r["meta"]["s"]] += 1
                s2p = [p for p, x in enumerate(roles) if x == "S2"]
                if s2p and v[s2p[0]] > 0:
                    s2_fire_names[(site, li)][r["meta"]["s"]] += 1
for h in hs:
    h.remove()


def purity(c):
    n = sum(c.values())
    return (c.most_common(1)[0][1] / n, n) if n else (0.0, 0)


out = []
for (site, li), dd in dens.items():
    m = {ro: float(np.mean(v)) for ro, v in dd.items()}
    oth = m.get("other", 0) + 1e-6
    out.append(dict(site=site, li=li, alpha=members[site][li], dens={ro: m[ro] / oth for ro in m if ro != "other"}, m=m))

print("\nDENSITY (mean act at role / mean act at other positions). Members with END density >= 3:")
end = sorted([o for o in out if o["dens"].get("END", 0) >= 3], key=lambda o: -o["dens"]["END"])
cnt = Counter("%s%d" % (o["site"][0][0], o["site"][1]) for o in end)
print("  n=%d by site %s" % (len(end), dict(cnt)))
for o in end[:25]:
    c = end_fire_names[(o["site"], o["li"])]
    pu, n = purity(c)
    pus, _ = purity(end_fire_s[(o["site"], o["li"])])
    print("  %-3s L%-2d %6d | alpha %.2f | END dens %6.1f | END fires %3d | IO-name purity %.2f (top %s) | S-name purity %.2f"
          % (o["site"][0], o["site"][1], o["li"], o["alpha"], o["dens"]["END"], n, pu, c.most_common(1)[0][0] if n else "-", pus))

print("\nNAME-IDENTITY TEST at END (attention members firing at END on >=10 prompts): IO-name purity vs S-name purity")
rows_ = []
for (site, li), c in end_fire_names.items():
    n = sum(c.values())
    if site[0] == "att" and n >= 10:
        rows_.append((site, li, n, purity(c)[0], purity(end_fire_s[(site, li)])[0], c.most_common(1)[0][0]))
rows_.sort(key=lambda x: -x[3])
byl = defaultdict(list)
for s, li, n, pi, ps, top in rows_:
    byl[s[1]].append((pi, ps))
for l in sorted(byl):
    print("  att L%-2d n=%2d | mean IO-purity %.2f | mean S-purity %.2f | n IO-purity>=0.9: %d"
          % (l, len(byl[l]), np.mean([p[0] for p in byl[l]]), np.mean([p[1] for p in byl[l]]), sum(p[0] >= 0.9 for p in byl[l])))
print("  chance purity ~ %.2f (most common of %d names)" % (max(names.values()) / sum(names.values()), len(names)))
print("\n  att L22 members individually:")
for s, li, n, pi, ps, top in rows_:
    if s[1] == 22:
        print("    %6d | END fires %3d | IO purity %.2f -> '%s' | S purity %.2f" % (li, n, pi, top, ps))

print("\nS2 members: name-independence (S-name purity at S2; low = duplicate-token-like)")
s2 = sorted([o for o in out if o["dens"].get("S2", 0) >= 3], key=lambda o: -o["dens"]["S2"])
cnt = Counter("%s%d" % (o["site"][0][0], o["site"][1]) for o in s2)
print("  n=%d with S2 density>=3 by site %s" % (len(s2), dict(cnt)))
for o in s2[:20]:
    pu, n = purity(s2_fire_names[(o["site"], o["li"])])
    print("  %-3s L%-2d %6d | alpha %.2f | S2 dens %6.1f | S2 fires %3d | S-name purity %.2f"
          % (o["site"][0], o["site"][1], o["li"], o["alpha"], o["dens"]["S2"], n, pu))

json.dump([dict(kind=o["site"][0], layer=o["site"][1], latent=o["li"], alpha=o["alpha"], density=o["dens"], mean_act=o["m"],
                end_io_purity=purity(end_fire_names[(o["site"], o["li"])])[0],
                end_fires=sum(end_fire_names[(o["site"], o["li"])].values()),
                s2_sname_purity=purity(s2_fire_names[(o["site"], o["li"])])[0]) for o in out],
          open(HERE / MEMBERS.replace("_members.jsonl", "_density.json"), "w"), indent=1)
print("->", MEMBERS.replace("_members.jsonl", "_density.json"))
