"""INDUCTION MECHANISM in the Gemma circuit. Prompt: six words A B C D E F
then the first k repeated (A B C); the answer is the word that followed
the last repeated word in the first copy.

Published mechanism (Olsson et al. 2022): a PREVIOUS-TOKEN head writes
"I am preceded by X" at each first-copy position; an INDUCTION head at
the prediction position attends to the position AFTER the earlier copy
of the current token and copies what it finds.

Roles per position:
  first_copy  any token of the first copy
  prev_first  the FIRST-copy occurrence of the last prompt word (the
              "match" position the induction head looks up)
  ans_first   the FIRST-copy occurrence of the ANSWER (match + 1) — the
              position an induction head must read
  repeat      second-copy tokens before the last
  END         the prediction position (= second occurrence of the match)

Per member: mean activation by role, the role it peaks at, and its
COPY SCORE = mean over prompts of (logit it adds to that prompt's answer
token − to the contrast token) via direct logit attribution, which is
what a copying feature must do. Members are ranked by alpha × |copy|.

  MEMBERS=induction_amr_s2_gemma_members.jsonl TIER=2 \
    python experiments/050-gemma-circuits/analyse_induction.py
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
TOP = int(os.environ.get("TOP", 20))
MEMBERS = os.environ.get("MEMBERS", "induction_amr_s2_gemma_members.jsonl")
PROMPTS = os.environ.get("PROMPTS", "induction_gemma_prompts.pt")
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
ROLES = ["prev_first", "ans_first", "first_copy", "repeat", "END"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
rec = json.loads(open(HERE / MEMBERS).readline())
members = {}
for k, d in rec["alphas"].items():
    kind, layer = k.split("/")
    members[(kind, int(layer))] = {int(i): float(a) for i, a in d.items()}
data = torch.load(HERE / PROMPTS, weights_only=False)
rows = data["rows"]
print("%s | %d prompts | %d members over %d sites | EF_ho %s"
      % (MEMBERS, len(rows), sum(len(d) for d in members.values()), len(members), rec.get("EF_ho")))

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
    """positions -> role. Tokens are ' word' pieces; find the two copies by
    matching the meta sequence against decoded tokens."""
    ids = r["ids"]
    words = [tok.decode([t]).strip() for t in ids]
    seq, k = r["meta"]["seq"], r["meta"]["k"]
    roles = ["other"] * len(ids)
    # first copy = first occurrence of each seq word, in order
    pos_first, j = {}, 0
    for p, w in enumerate(words):
        if j < len(seq) and w == seq[j]:
            pos_first[seq[j]] = p; roles[p] = "first_copy"; j += 1
    match_word, ans_word = seq[k - 1], seq[k]
    if match_word in pos_first:
        roles[pos_first[match_word]] = "prev_first"
    if ans_word in pos_first:
        roles[pos_first[ans_word]] = "ans_first"
    for p in range(max(pos_first.values(), default=0) + 1, len(ids)):
        roles[p] = "repeat"
    roles[len(ids) - 1] = "END"
    return roles


# ---- direct logit attribution helper (decoder direction -> vocabulary) ----
W_U = model.lm_head.weight.detach().float()
NORM = (1.0 + model.model.norm.weight.detach().float())


def dla(site, idx):
    kind, layer = site
    d = tc(site)["W_dec"][idx].float()
    if kind == "att":                      # o_proj input -> residual stream
        d = model.model.layers[layer].self_attn.o_proj.weight.detach().float() @ d
    return W_U @ (d * NORM)


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
    elif kind == "att":
        hs.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, i, _c=f: _c(i[0])))
    else:
        hs.append(blk.register_forward_hook(lambda m, a, o, _c=f: _c(o[0] if isinstance(o, tuple) else o)))

# Per-member DLA, restricted to the answer/contrast tokens that occur in the
# dataset, so the CONTRIBUTION (activation x direction x alpha) can be
# accumulated inside the prompt loop.
vocab_ids = sorted({r["target"] for r in rows} | {r["contrast"] for r in rows})
vi = {t: j for j, t in enumerate(vocab_ids)}
DLA = {}
for site, d in members.items():
    for li in d:
        DLA[(site, li)] = dla(site, li)[torch.tensor(vocab_ids, device=DEV)].float().cpu().numpy()

role_act = defaultdict(lambda: defaultdict(float))
peak_role = defaultdict(Counter)
fire_end = defaultdict(int)
contrib = defaultdict(float)          # act at END x (dla[answer] - dla[contrast])
n = len(rows)
with torch.no_grad():
    for r in rows:
        roles = roles_for(r)
        caps.clear()
        model(torch.tensor([r["ids"]], device=DEV))
        for site, d in members.items():
            a = caps[site]
            for j, li in enumerate(sorted(d)):
                v = a[:, j].numpy()
                if v.max() <= 0:
                    continue
                key = (site, li)
                peak_role[key][roles[int(v.argmax())]] += 1
                if v[-1] > 0:
                    fire_end[key] += 1
                    dd = DLA[key]
                    contrib[key] += float(v[-1]) * float(dd[vi[r["target"]]] - dd[vi[r["contrast"]]])
                for ro in set(roles):
                    idxs = [p for p, x in enumerate(roles) if x == ro and p > 0]
                    if idxs:
                        role_act[key][ro] += float(v[idxs].max())
for h in hs:
    h.remove()

# copy score: does the member's output direction promote the answer over the contrast?
copy = {}
for site, d in members.items():
    for li in d:
        lg = dla(site, li)
        vals = [float(lg[r["target"]] - lg[r["contrast"]]) for r in rows]
        copy[(site, li)] = float(np.mean(vals))

out = []
for site, d in members.items():
    for li, alpha in d.items():
        key = (site, li)
        pr = peak_role[key].most_common(1)
        out.append(dict(kind=site[0], layer=site[1], latent=li, alpha=alpha,
                        peak=pr[0][0] if pr else "silent", n_peak=pr[0][1] if pr else 0,
                        act={ro: role_act[key][ro] / n for ro in ROLES},
                        end_fires=fire_end[key], copy=copy[key],
                        contrib=alpha * contrib[key] / n))

print("\nMEMBERS BY PEAK ROLE:")
by = Counter(o["peak"] for o in out)
for ro, c in by.most_common():
    sub = [o for o in out if o["peak"] == ro]
    kinds = Counter("%s%d" % (o["kind"][0], o["layer"]) for o in sub)
    print("  %-11s %4d | %s" % (ro, c, dict(sorted(kinds.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))))))

print("\nCOPY SCORE (DLA on that prompt's answer minus contrast; a copying feature is positive):")
print("  all members: median %+.3f | >0: %.2f | top decile %+.3f"
      % (np.median([o["copy"] for o in out]), np.mean([o["copy"] > 0 for o in out]),
         np.percentile([o["copy"] for o in out], 90)))
for ro in ROLES:
    sub = [o["copy"] for o in out if o["peak"] == ro]
    if sub:
        print("  peaking at %-11s n=%3d | median copy %+.3f | frac > 0 %.2f" % (ro, len(sub), np.median(sub), np.mean(np.array(sub) > 0)))

print("\nCONTRIBUTION at END (alpha x activation x [DLA(answer) - DLA(contrast)]), summed = the")
print("circuit's own account of how the answer gets promoted:")
cs = np.array([o["contrib"] for o in out])
print("  total %+.3f | positive members %d (sum %+.3f) | negative %d (sum %+.3f)"
      % (cs.sum(), int((cs > 0).sum()), cs[cs > 0].sum(), int((cs < 0).sum()), cs[cs < 0].sum()))

print("\nTOP %d COPYING MEMBERS (by contribution at END; the induction-head signature):" % TOP)
for o in sorted(out, key=lambda o: -o["contrib"])[:TOP]:
    print("  %-3s L%-2d %6d | alpha %5.2f | contrib %+7.3f | fires at END %3d/%d | act END %6.2f | act ans_first %5.2f prev_first %5.2f | peak %s"
          % (o["kind"], o["layer"], o["latent"], o["alpha"], o["contrib"], o["end_fires"], n,
             o["act"]["END"], o["act"]["ans_first"], o["act"]["prev_first"], o["peak"]))

print("\nTOP %d MEMBERS AT ans_first (the position an induction head must READ; previous-token signature):" % TOP)
for o in sorted(out, key=lambda o: -(o["alpha"] * o["act"]["ans_first"]))[:TOP]:
    print("  %-3s L%-2d %6d | alpha %5.2f | act ans_first %6.2f | prev_first %5.2f | END %5.2f | contrib %+.3f | peak %s"
          % (o["kind"], o["layer"], o["latent"], o["alpha"], o["act"]["ans_first"], o["act"]["prev_first"],
             o["act"]["END"], o["contrib"], o["peak"]))

print("\nROLE PROFILE of the whole circuit (mean max-activation per prompt at each role):")
for ro in ROLES:
    v = np.array([o["act"][ro] for o in out])
    print("  %-11s mean %6.3f | members with act > 0.5 there: %3d" % (ro, v.mean(), int((v > 0.5).sum())))

json.dump(out, open(HERE / MEMBERS.replace("_members.jsonl", "_roles.json"), "w"), indent=1)
print("\n->", MEMBERS.replace("_members.jsonl", "_roles.json"))
