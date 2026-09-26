"""TWO VARIABLES, ONE COMPUTATION: entity x relation -> answer.

The two-hop circuit answers f(entity, relation):
    (Germany, capital)  -> Berlin       (Germany, language) -> German
    (China,   capital)  -> Beijing      (China,   language) -> Chinese
The ENTITY is inferred from a city (never written); the RELATION is read
from the question ("capital" vs "language"). bridge_features.py showed
the entity lives in specific members and is causal. This script:

  1. finds the RELATION members: activation on language prompts vs
     capital prompts over the SAME entities (ratio), split by whether
     they peak at the relation word or at the answer position
  2. groups the circuit into ENTITY / RELATION / BOTH / OTHER members
  3. the 2x2 patch on one prompt: swap the entity, swap the relation,
     swap both, and read the log-probs of all four candidate answers.
     If the circuit computes f(entity, relation), the answer should
     track the installed combination.

  python experiments/051-gemma-reasoning/twohop_variables.py
Env: B1=Germany B2=China CITY=Munich REL_MIN (3.0) SCALE (1.0)
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
B1, B2, CITY = os.environ.get("B1", "Germany"), os.environ.get("B2", "China"), os.environ.get("CITY", "Munich")
REL_MIN = float(os.environ.get("REL_MIN", 3.0))
SCALE = float(os.environ.get("SCALE", 1.0))
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
KEYS = [(s, li) for s in sorted(members) for li in sorted(members[s])]
KI = {k: j for j, k in enumerate(KEYS)}
NM = len(KEYS)
two = json.load(open(HERE / "twohop_rows.json"))
cap_rows = [r for r in two if r["meta"]["family"] == "country_capital"]
lang_rows = [r for r in two if r["meta"]["family"] == "country_language"]
ents = sorted({r["meta"]["bridge"] for r in cap_rows} & {r["meta"]["bridge"] for r in lang_rows})
print("%d members | capital prompts %d | language prompts %d | shared entities %d" % (NM, len(cap_rows), len(lang_rows), len(ents)))

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
            caps[_s] = c[:, idx].float().cpu().numpy()      # [T, n]
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


def run(text):
    """[T, NM] member activations, plus the decoded tokens."""
    ids = tok(text, return_tensors="pt")["input_ids"].to(DEV)
    caps.clear()
    with torch.no_grad():
        model(ids)
    T = ids.shape[1]
    A = np.zeros((T, NM), dtype=np.float32)
    for s in members:
        for j, li in enumerate(sorted(members[s])):
            A[:, KI[(s, li)]] = caps[s][:, j]
    return A, [tok.decode([t]) for t in ids[0].tolist()]


# ---- 1. relation members: language vs capital over the same entities ----------------
def rel_word_pos(words, fam):
    key = "language" if fam == "country_language" else "capital"
    for p, w in enumerate(words):
        if key in w.lower():
            return p
    return None


act_end = {"cap": [], "lang": []}; act_rel = {"cap": [], "lang": []}; act_max = {"cap": [], "lang": []}
for tag, rows in (("cap", cap_rows), ("lang", lang_rows)):
    for r in rows:
        A, words = run(r["prompt"])
        act_end[tag].append(A[-1]); act_max[tag].append(A[1:].max(axis=0))
        p = rel_word_pos(words, r["meta"]["family"])
        act_rel[tag].append(A[p] if p is not None else np.zeros(NM))
for d in (act_end, act_rel, act_max):
    for tag in d:
        d[tag] = np.stack(d[tag])
m_lang, m_cap = act_max["lang"].mean(0), act_max["cap"].mean(0)
rel_ratio = np.where(m_lang > m_cap, m_lang / np.maximum(m_cap, 1e-3), -m_cap / np.maximum(m_lang, 1e-3))
# relation must be consistent across entities: fraction of entities where the preferred family wins
ent_of = {"cap": np.array([ents.index(r["meta"]["bridge"]) if r["meta"]["bridge"] in ents else -1 for r in cap_rows]),
          "lang": np.array([ents.index(r["meta"]["bridge"]) if r["meta"]["bridge"] in ents else -1 for r in lang_rows])}
consist = np.zeros(NM)
for j in range(NM):
    wins = 0; n = 0
    for e in range(len(ents)):
        a = act_max["lang"][ent_of["lang"] == e, j].mean() if (ent_of["lang"] == e).any() else 0
        b = act_max["cap"][ent_of["cap"] == e, j].mean() if (ent_of["cap"] == e).any() else 0
        if a > 0 or b > 0:
            n += 1; wins += int((a > b) == (rel_ratio[j] > 0))
    consist[j] = wins / max(n, 1)
bj = json.load(open(HERE / BRIDGE))
ent_ratio = {(r["kind"], r["layer"], r["latent"]): r["ratio"] for r in bj["top"]}
ent_of_member = {(r["kind"], r["layer"], r["latent"]): r["bridge"] for r in bj["top"]}


def kind_of(j):
    (s, li) = KEYS[j]
    er = ent_ratio.get((s[0], s[1], li), 1.0)
    is_ent = er >= 3.0
    is_rel = abs(rel_ratio[j]) >= REL_MIN and consist[j] >= 0.8
    return ("BOTH" if is_ent and is_rel else "ENTITY" if is_ent else "RELATION" if is_rel else "other")


groups = defaultdict(list)
for j in range(NM):
    groups[kind_of(j)].append(j)
print("\n[2] MEMBER GROUPS: %s" % {k: len(v) for k, v in groups.items()})
print("\n[1] RELATION MEMBERS (activation on language vs capital prompts, same %d entities; consistency = share of entities agreeing):" % len(ents))
print("  %-16s %5s %9s %9s %7s %6s %9s %9s" % ("member", "alpha", "lang", "capital", "ratio", "consis", "@relword", "@END"))
for j in sorted(groups["RELATION"] + groups["BOTH"], key=lambda j: -abs(rel_ratio[j])):
    (s, li) = KEYS[j]
    pref = "lang" if rel_ratio[j] > 0 else "cap"
    print("  %-3s L%-2d %8d %5.2f %9.2f %9.2f %+7.1f %6.2f %9.2f %9.2f   -> %s%s"
          % (s[0], s[1], li, members[s][li], m_lang[j], m_cap[j], rel_ratio[j], consist[j],
             act_rel[pref][:, j].mean(), act_end[pref][:, j].mean(),
             "LANGUAGE" if rel_ratio[j] > 0 else "CAPITAL", " (also entity: %s)" % ent_of_member.get((s[0], s[1], li), "") if j in groups["BOTH"] else ""))
for h in hs:
    h.remove()

# ---- 3. the 2x2 patch --------------------------------------------------------------------
lang_members = {KEYS[j]: float(act_max["lang"][:, j].mean()) for j in groups["RELATION"] if rel_ratio[j] > 0}
cap_members = {KEYS[j]: float(act_max["cap"][:, j].mean()) for j in groups["RELATION"] if rel_ratio[j] < 0}
ent = defaultdict(dict)
for r in bj["top"]:
    if r["ratio"] >= 3.0 and r["selectivity"] >= 0.4:
        ent[r["bridge"]][((r["kind"], r["layer"]), r["latent"])] = r["match"]


class Edit:
    def __init__(self, off, on):
        self.off, self.on, self.handles = off, on, []

    def __enter__(self):
        for site in {s for s, _ in self.off} | {s for s, _ in self.on}:
            kind, layer = site
            blk = model.model.layers[layer]

            def edit(x, _s=site):
                t = tc(_s)
                pre = x @ t["W_enc"] + t["b_enc"]
                c = pre * (pre > t["threshold"])
                chat = c.clone()
                for (s2, i) in self.off:
                    if s2 == _s:
                        chat[0, 1:, i] = 0.0
                for (s2, i), v in self.on.items():
                    if s2 == _s:
                        chat[0, 1:, i] = float(v) * SCALE
                delta = (chat - c).to(x.dtype) @ t["W_dec"]
                delta[:, 0] = 0
                return x + delta
            if kind == "mlp":
                self.handles.append(blk.post_feedforward_layernorm.register_forward_hook(lambda m, i, o, _e=edit: _e(o)))
            elif kind == "att":
                self.handles.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, _e=edit: (_e(i[0]),) + tuple(i[1:])))
            else:
                self.handles.append(blk.register_forward_hook(lambda m, a, o, _e=edit: (_e(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else _e(o)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


row = next(r for r in lang_rows if r["meta"]["city"] == CITY)
prompt = row["prompt"]
ans = {("%s" % B1, "lang"): next(r["meta"]["answer"] for r in lang_rows if r["meta"]["bridge"] == B1),
       ("%s" % B1, "cap"): next(r["meta"]["answer"] for r in cap_rows if r["meta"]["bridge"] == B1),
       ("%s" % B2, "lang"): next(r["meta"]["answer"] for r in lang_rows if r["meta"]["bridge"] == B2),
       ("%s" % B2, "cap"): next(r["meta"]["answer"] for r in cap_rows if r["meta"]["bridge"] == B2)}
base_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()


def tid(word):
    b = tok(prompt + " " + word, return_tensors="pt")["input_ids"][0].tolist()
    return b[len(base_ids)]


cand = {k: tid(v) for k, v in ans.items()}


def logprobs(off=(), on=None):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(DEV)
    with torch.no_grad():
        if off or on:
            with Edit(list(off), on or {}):
                lg = model(ids).logits[0, -1].float()
        else:
            lg = model(ids).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    return {k: float(lp[t]) for k, t in cand.items()}, int(lg.argmax())


conds = {
    "none": ((), {}),
    "entity %s->%s" % (B1, B2): (list(ent[B1]), dict(ent[B2])),
    "relation lang->cap": (list(lang_members), dict(cap_members)),
    "both": (list(ent[B1]) + list(lang_members), {**ent[B2], **cap_members}),
}
print("\n[3] THE 2x2 on %r" % prompt)
print("    installed variables -> log p of each candidate answer (argmax in caps):")
hdr = "    %-24s" % "condition" + "".join(" %12s" % ("%s/%s" % k) for k in cand)
print(hdr + "   top token")
res = {}
for name, (off, on) in conds.items():
    lp, am = logprobs(off, on)
    res[name] = lp
    best = max(lp, key=lp.get)
    print("    %-24s" % name + "".join(" %12s" % (("%.2f" % lp[k]).upper() if k == best else "%.2f" % lp[k]) for k in cand)
          + "   %r" % tok.decode([am]))
# control: same number of random members for the 'both' edit
n_edit = len(conds["both"][0]) + len(conds["both"][1])
pick = [KEYS[i] for i in rng.choice(NM, min(n_edit, NM), replace=False)]
mags = list(cap_members.values()) + list(ent[B2].values())
ctrl_on = {k: float(rng.choice(mags)) for k in pick}
lp, am = logprobs((), ctrl_on)
print("    %-24s" % "control (random)" + "".join(" %12.2f" % lp[k] for k in cand) + "   %r" % tok.decode([am]))
json.dump({"prompt": prompt, "answers": {"%s/%s" % k: v for k, v in ans.items()}, "results": {n: {"%s/%s" % k: v for k, v in d.items()} for n, d in res.items()},
           "groups": {g: ["%s L%d.%d" % (KEYS[j][0][0], KEYS[j][0][1], KEYS[j][1]) for j in v] for g, v in groups.items() if g != "other"}},
          open(HERE / "twohop_variables.json", "w"), indent=1)
print("\n->", HERE / "twohop_variables.json")
