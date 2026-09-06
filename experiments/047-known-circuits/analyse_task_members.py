"""POSITION-RESOLVED MEMBER PROFILES for a task circuit: where does each
member latent fire on the task prompts? Classifies members by the token
ROLE at which they are most active, so the fitted tri-amp circuit can be
compared with the published circuit's structure:

  greater-than (Hanna et al. 2023): attention heads move the YY digits of
    the first year to the final position; MLPs at the final position
    compute "greater than". Roles: y1_c1 y1_c2 (century), y1_tens,
    y1_units, mid (" to the year"), y2_c1, y2_c2 (= prediction position).
  agreement (Marks et al. 2025 SFC): subject-number features on the
    subject, distractor-number features on the PP noun, verb-number
    features at the final position. Roles: subject, distractor, final,
    other.

  TASK=gt|agree PYTHONPATH=src python experiments/047-known-circuits/analyse_task_members.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, "src")
from hardware import detect_devices, is_fast_memory, should_compile
from model.hooks import multi_patch
from model.inference import Inference
from model.tokenizer import Tokenizer
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
from sae.dense import sparse_topk_to_dense

HERE = Path(__file__).parent
TASK = os.environ.get("TASK", "gt")
RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/"
                "Runs/20260531-152059-37117a33/20260531-152059-37117a33")
TOP = int(os.environ.get("TOP", 40))

load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices()
device = devices[0]
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(),
               compile=should_compile())
tok = Tokenizer()
inference.disable_compile()

data = torch.load(HERE / ("%s_clusters.pt" % TASK), weights_only=False)
members = {}
for line in open(HERE / os.environ.get("MEMBERS", "%s_members.jsonl" % TASK)):
    r = json.loads(line)
    for site, d in r["alphas"].items():
        lyr, knd = site.split("/")
        members[(int(lyr), knd)] = {int(i): float(a) for i, a in d.items()}
n_mem = sum(len(d) for d in members.values())
print("task %s | %d prompts | %d members over %d sites"
      % (TASK, len(data["prompts"]), n_mem, len(members)))


def roles_for(prompt, toks):
    """role label per token position."""
    words = [tok.decode([t]) for t in toks]
    roles = ["other"] * len(toks)
    if TASK == "gt":
        # digits: first four digit tokens = year 1, last two = year 2 prefix
        digit_pos = [i for i, w in enumerate(words) if w.strip().isdigit()]
        names = ["y1_c1", "y1_c2", "y1_tens", "y1_units"]
        for i, p in enumerate(digit_pos[:4]):
            roles[p] = names[i]
        for i, p in enumerate(digit_pos[4:6]):
            roles[p] = ["y2_c1", "y2_c2"][i]
        for p in range(digit_pos[3] + 1, digit_pos[4]):
            roles[p] = "mid"
    else:
        roles[1] = "subject"                 # "The <subject> ..."
        roles[len(toks) - 1] = "distractor"  # PP noun is the last token
        # the verb position is the anchor + 1 (not in the prompt); the
        # prediction is READ at the last token, so label it 'final' too
        roles[len(toks) - 1] = "distractor/final"
    return roles


class Capture:
    """Encode at member sites, keep member activations at every position."""

    def __init__(self):
        self.acts = {}      # site -> (B, T, n_members_at_site)

    def __call__(self, model):
        return multi_patch(model, self.transform)

    def transform(self, layer_idx, kind, x):
        site = (layer_idx, kind)
        d = members.get(site)
        if not d:
            return x
        ta, ti = bank.encode(x, kind, layer_idx)
        dense = sparse_topk_to_dense(ta, ti, bank.d_sae, dtype=torch.float32)
        idx = torch.tensor(sorted(d), device=dense.device)
        self.acts[site] = dense[..., idx].detach().cpu()
        return x


# accumulate per member: sum of activation by role, count of prompts firing
role_sum = defaultdict(lambda: defaultdict(float))
role_cnt = defaultdict(lambda: defaultdict(int))
fire_any = defaultdict(int)
prompts = data["prompts"]
anchors = data.get("anchors")
for i, prompt in enumerate(prompts):
    toks = tok.encode(prompt)
    roles = roles_for(prompt, toks)
    cap = Capture()
    with torch.no_grad():
        inference.forward(torch.tensor([toks], dtype=torch.long, device=device),
                          patcher=cap, grad_enabled=False,
                          return_activations=False, tokenize_final=False)
    for site, d in members.items():
        a = cap.acts.get(site)
        if a is None:
            continue
        ids = sorted(d)
        for j, li in enumerate(ids):
            v = a[0, :, j]
            if float(v.max()) > 0:
                fire_any[(site, li)] += 1
            for p, r in enumerate(roles):
                if p < v.shape[0] and float(v[p]) > 0:
                    role_sum[(site, li)][r] += float(v[p])
                    role_cnt[(site, li)][r] += 1

rows = []
for site, d in members.items():
    for li, alpha in d.items():
        k = (site, li)
        rs = role_sum[k]
        tot = sum(rs.values())
        if tot == 0:
            rows.append((site, li, alpha, 0, "silent", 0.0, {}))
            continue
        dom = max(rs, key=rs.get)
        share = {r: rs[r] / tot for r in rs}
        rows.append((site, li, alpha, fire_any[k], dom, rs[dom] / tot, share))

# summary by dominant role, and top members by firing prevalence
by_role = defaultdict(list)
for r in rows:
    by_role[r[4]].append(r)
print("\nMEMBERS BY DOMINANT ROLE (where their activation mass sits on the task prompts):")
for role, lst in sorted(by_role.items(), key=lambda kv: -len(kv[1])):
    layers = sorted(set(r[0][0] for r in lst))
    print("  %-18s %4d members | layers %s" % (role, len(lst), layers))
print("\nTOP %d MEMBERS by prompts-fired (alpha, dominant role, role share, layer/kind/latent):" % TOP)
for r in sorted(rows, key=lambda r: -r[3])[:TOP]:
    site, li, alpha, nf, dom, dshare, share = r
    others = ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in
                       sorted(share.items(), key=lambda kv: -kv[1])[:3])
    print("  L%-2d %-5s %6d | alpha %5.2f | fires on %3d/%d | %s" % (
        site[0], site[1], li, alpha, nf, len(prompts), others))
out = HERE / (os.environ.get("MEMBERS", "%s_members.jsonl" % TASK).replace("_members.jsonl", "_member_roles.json"))
json.dump([{"layer": r[0][0], "kind": r[0][1], "latent": r[1], "alpha": r[2],
            "fires": r[3], "dominant": r[4], "shares": r[6]} for r in rows],
          open(out, "w"), indent=1)
print("\n->", out)
