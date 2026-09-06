"""TRI-AMP BEHAVIOUR CIRCUIT on Gemma-2-2B + GemmaScope MLP SAEs (16k
JumpReLU, TIER sparsity variant), logit endpoint, all 26 layers.

Same recipe as 044/behaviour_runner.py on TuringLLM, ported to the
Gemma substrate of 037/ours_gemma.py (hooks on
post_feedforward_layernorm output; intervention x <- x + (chat - c) @
W_dec so the SAE error passes through untouched):

  objective   reproduce the FULL model's log p(target) - log p(contrast)
              (contrastive; IOI: IO vs S, agreement: correct vs wrong)
              or log p(target) when there is no contrast (greater-than)
  floor       zero (non-members -> 0 at every site), free amplitudes
  sites       all 26 MLP SAE layers
  split       75/25 train/held-out; EF on held-out; matched amp-null;
              alpha=1 control; task metric under circuit-only execution
  members     -> <task>_gemma_members.jsonl (layer -> {latent: alpha})

Prompts are RIGHT-padded with per-row anchors (left-padding corrupts
the model). BOS prepended, as Gemma expects.

  TASK=ioi|gt|agree N=256 STEPS=400 LAM=3e-3 TIER=2 \
    python experiments/048-gemma-tasks/fit_gemma_task.py
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
import gemma_loader as G   # noqa: E402  (SAE download/cache + tier ladder)

TASK = os.environ.get("TASK", "ioi")
N = int(os.environ.get("N", 256))
STEPS = int(os.environ.get("STEPS", 400))
LAM = float(os.environ.get("LAM", 3e-3))
LR = float(os.environ.get("LR", 0.05))
TIER = int(os.environ.get("TIER", 2))
FIT_BS = int(os.environ.get("FIT_BS", 4))
N_NULL = int(os.environ.get("N_NULL", 2))
MODEL_ID = os.environ.get("GEMMA_MODEL", "unsloth/gemma-2-2b")
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
rng = random.Random(0)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
N_LAYERS = len(model.model.layers)
KINDS = [k for k in os.environ.get("KINDS", "att,mlp,res").split(",") if k]
REPOS = {"att": "google/gemma-scope-2b-pt-att", "res": "google/gemma-scope-2b-pt-res"}
SITES = [(k, l) for k in KINDS for l in range(N_LAYERS)]
_TC = {}


def _ladder(kind, layer):
    import re
    from huggingface_hub import list_repo_files
    pat = re.compile(r"layer_%d/width_16k/average_l0_(\d+)/" % layer)
    return sorted({int(m.group(1)) for f in list_repo_files(REPOS[kind])
                   for m in [pat.search(f)] if m})


def tc(site):
    """SAE params for (kind, layer) at TIER, cached on disk and on GPU."""
    if site not in _TC:
        kind, layer = site
        if kind == "mlp":
            l0 = G.tier_l0(layer, TIER)
            p = G.CACHE / ("layer_%d_w%s_l0_%d.npz" % (layer, G.WIDTH, l0))
            if not p.exists():
                G.load_sae(layer, l0)
        else:
            lad = _ladder(kind, layer)
            l0 = lad[min(TIER, len(lad) - 1)]
            p = G.CACHE / ("%s_layer_%d_w16k_l0_%d.npz" % (kind, layer, l0))
            if not p.exists():
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(REPOS[kind], "layer_%d/width_16k/average_l0_%d/params.npz" % (layer, l0))
                tmp = p.with_suffix(".tmp"); tmp.write_bytes(Path(src).read_bytes()); tmp.replace(p)
        z = np.load(p)
        _TC[site] = {k: torch.tensor(z[k], device=DEV).to(DTYPE) for k in z.files}
    return _TC[site]


D_SAE = tc(SITES[0])["W_enc"].shape[1]
print("Gemma %d layers | kinds %s | %d sites | SAE width %d | tier %d"
      % (N_LAYERS, KINDS, len(SITES), D_SAE, TIER), flush=True)


def features(site, x):
    t = tc(site)
    pre = x @ t["W_enc"] + t["b_enc"]
    return pre * (pre > t["threshold"])


def _edit(site, x, fn):
    """x <- x + (chat - c) @ W_dec on positions >= 1 (BOS untouched)."""
    c = features(site, x)
    chat = fn(c)
    delta = (chat - c).to(x.dtype) @ tc(site)["W_dec"]
    delta[:, 0] = 0
    return x + delta


class Runner:
    def __init__(self, transforms):
        self.transforms, self.handles = transforms, []

    def __enter__(self):
        for site, fn in self.transforms.items():
            kind, layer = site
            blk = model.model.layers[layer]
            if kind == "mlp":
                self.handles.append(blk.post_feedforward_layernorm.register_forward_hook(
                    lambda m, i, o, _s=site, _f=fn: _edit(_s, o, _f)))
            elif kind == "att":
                self.handles.append(blk.self_attn.o_proj.register_forward_pre_hook(
                    lambda m, i, _s=site, _f=fn: (_edit(_s, i[0], _f),) + tuple(i[1:])))
            else:
                def res_hook(m, a, o, _s=site, _f=fn):
                    if isinstance(o, tuple):
                        return (_edit(_s, o[0], _f),) + tuple(o[1:])
                    return _edit(_s, o, _f)
                self.handles.append(blk.register_forward_hook(res_hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def value(tokens, anchors, targets, contrasts, transforms, grad):
    """log p(target) [- log p(contrast)] at each row's anchor -> [B]."""
    with torch.set_grad_enabled(grad), Runner(transforms):
        lg = model(tokens.to(DEV)).logits
    b = torch.arange(tokens.shape[0], device=DEV)
    lp = torch.log_softmax(lg[b, anchors.to(DEV)].float(), -1)
    v = lp[b, targets.to(DEV)]
    if contrasts is not None:
        v = v - lp[b, contrasts.to(DEV)]
    return v


# ---------------------------------------------------------------- data
def enc(text):
    return tok(text, return_tensors="pt")["input_ids"][0].tolist()


def tok_after(prefix, word):
    a, b = enc(prefix), enc(prefix + word)
    return b[len(a)] if (b[:len(a)] == a and len(b) == len(a) + 1) else None


def build():
    rows = []
    if TASK == "ioi":
        NAMES = ["Mary", "John", "Tom", "James", "Dan", "Martin", "Amy", "Joseph",
                 "Jim", "Peter", "Paul", "Bob", "Alice", "Sarah", "Emma", "David",
                 "Lucy", "Anna", "Kate", "Mark", "Steve", "Laura", "Jack", "Sam"]
        PLACES = ["store", "school", "park", "office", "hospital", "garden", "station"]
        OBJECTS = ["drink", "book", "ring", "bone", "necklace", "snack", "letter"]
        TPL = ["Then, {X} and {Y} went to the {place}. {S} gave a {obj} to",
               "When {X} and {Y} got a {obj} at the {place}, {S} decided to give it to",
               "After {X} and {Y} went to the {place}, {S} gave a {obj} to"]
        seen = set()
        while len(rows) < N:
            io, s = rng.sample(NAMES, 2)
            abba = rng.random() < 0.5
            x, y = (io, s) if abba else (s, io)
            prompt = rng.choice(TPL).format(X=x, Y=y, S=s, place=rng.choice(PLACES),
                                            obj=rng.choice(OBJECTS))
            if prompt in seen:
                continue
            seen.add(prompt)
            ti, ts = tok_after(prompt, " " + io), tok_after(prompt, " " + s)
            if ti is None or ts is None:
                continue
            rows.append({"prompt": prompt, "target": ti, "contrast": ts,
                         "meta": {"order": "ABBA" if abba else "BABA", "io": io, "s": s}})
    elif TASK == "gt":
        while len(rows) < N:
            c = rng.choice(["16", "17", "18", "19"]); yy = rng.randint(11, 88)
            prompt = "The %s lasted from the year %s%02d to the year %s" % (
                rng.choice(["war", "empire", "dynasty", "siege", "festival",
                            "expedition", "reign"]), c, yy, c)
            ids = [tok_after(prompt, str(d)) for d in range(10)]
            if any(i is None for i in ids) or any(r["prompt"] == prompt for r in rows):
                continue
            with torch.no_grad():
                lg = model(torch.tensor([enc(prompt)], device=DEV)).logits[0, -1].float()
            p = torch.softmax(lg, -1)[ids]
            d = int(p.argmax()); tens = yy // 10
            if d <= tens:
                continue
            rows.append({"prompt": prompt, "target": ids[d], "contrast": None,
                         "meta": {"tens": tens, "digit_ids": ids}})
    else:
        pairs = [("key", "keys"), ("book", "books"), ("car", "cars"), ("teacher", "teachers"),
                 ("dog", "dogs"), ("report", "reports"), ("student", "students"),
                 ("plan", "plans"), ("river", "rivers"), ("idea", "ideas")]
        pps = ["on the {n}", "near the {n}", "behind the {n}", "of the {n}", "beside the {n}"]
        verbs = [("is", "are"), ("was", "were"), ("has", "have")]
        seen = set()
        while len(rows) < N:
            ss, sp = rng.choice(pairs); ds, dp = rng.choice(pairs)
            if ss == ds:
                continue
            plural = rng.random() < 0.5
            prompt = "The %s %s" % (sp if plural else ss, rng.choice(pps).format(n=ds if plural else dp))
            vs, vp = rng.choice(verbs)
            if (prompt, vs) in seen:
                continue
            seen.add((prompt, vs))
            cor, wr = (vp, vs) if plural else (vs, vp)
            tc_, tw = tok_after(prompt, " " + cor), tok_after(prompt, " " + wr)
            if tc_ is None or tw is None:
                continue
            rows.append({"prompt": prompt, "target": tc_, "contrast": tw,
                         "meta": {"plural": plural}})
    # competence filter with the intact model (contrast > 0, or target argmax digit)
    keep = []
    for r in rows:
        ids = enc(r["prompt"])
        with torch.no_grad():
            lg = model(torch.tensor([ids], device=DEV)).logits[0, -1].float()
        if r["contrast"] is not None:
            r["nat_margin"] = float(lg[r["target"]] - lg[r["contrast"]])
            if r["nat_margin"] <= 0:
                continue
        r["ids"] = ids
        keep.append(r)
    return keep


rows = build()
n = len(rows)
n_tr = int(n * 0.75)
T = max(len(r["ids"]) for r in rows)
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
tokens = torch.tensor([r["ids"] + [pad] * (T - len(r["ids"])) for r in rows], dtype=torch.long)
anchors = torch.tensor([len(r["ids"]) - 1 for r in rows], dtype=torch.long)
targets = torch.tensor([r["target"] for r in rows], dtype=torch.long)
contrasts = (torch.tensor([r["contrast"] for r in rows], dtype=torch.long)
             if rows[0]["contrast"] is not None else None)
print("task %s | %d prompts kept (%d train / %d held-out) | max len %d | contrastive=%s"
      % (TASK, n, n_tr, n - n_tr, T, contrasts is not None), flush=True)
sl = lambda t, a, b: None if t is None else t[a:b]
tr = (tokens[:n_tr], anchors[:n_tr], targets[:n_tr], sl(contrasts, 0, n_tr))
ho = (tokens[n_tr:], anchors[n_tr:], targets[n_tr:], sl(contrasts, n_tr, n))

# natural targets (full model) for train rows
with torch.no_grad():
    nat_tr = torch.cat([value(tr[0][s:s + 16], tr[1][s:s + 16], tr[2][s:s + 16],
                              sl(tr[3], s, s + 16), {}, False)
                        for s in range(0, n_tr, 16)])
tnorm = max(float((nat_tr ** 2).mean()), 1e-6)


def mean_value(split, transforms):
    tk, an, tg, ct = split
    with torch.no_grad():
        vals = torch.cat([value(tk[s:s + 16], an[s:s + 16], tg[s:s + 16],
                                sl(ct, s, s + 16), transforms, False)
                          for s in range(0, tk.shape[0], 16)])
    return float(vals.mean())


def circuit_transforms(members, use_amps=True):
    tr_ = {}
    for site in SITES:
        d = members.get(site, {})
        idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
        al = torch.tensor([d[int(i)] if use_amps else 1.0 for i in idx.tolist()],
                          device=DEV, dtype=torch.float32)
        def fn(c, _idx=idx, _al=al):
            chat = torch.zeros_like(c)
            if len(_idx):
                chat[..., _idx] = c[..., _idx] * _al.to(c.dtype)
            return chat
        tr_[site] = fn
    return tr_


def fit():
    params = {}
    for site in SITES:
        th = torch.full((D_SAE,), 2.0, device=DEV, requires_grad=True)
        ps = torch.full((D_SAE,), math.log(math.e - 1.0), device=DEV, requires_grad=True)
        params[site] = (th, ps)
    opt = torch.optim.AdamW([p for pr in params.values() for p in pr], lr=LR, weight_decay=0.05)
    temp = [1.0]
    t0 = time.time()
    for step in range(STEPS):
        temp[0] = 1.0 * (0.05 ** (step / max(STEPS - 1, 1)))
        s0 = (step * FIT_BS) % n_tr
        tk, an, tg, ct = (tr[0][s0:s0 + FIT_BS], tr[1][s0:s0 + FIT_BS],
                          tr[2][s0:s0 + FIT_BS], sl(tr[3], s0, s0 + FIT_BS))
        nat = nat_tr[s0:s0 + FIT_BS]
        opt.zero_grad()
        trf = {}
        for site in SITES:
            def fn(c, _s=site):
                th, ps = params[_s]
                m = torch.sigmoid(th / temp[0])
                return (m * F.softplus(ps)).to(c.dtype) * c      # zero floor
            trf[site] = fn
        v = value(tk, an, tg, ct, trf, True)
        (((v - nat.to(DEV)) ** 2).mean() / tnorm).backward()
        pen = 0.0
        for site in SITES:
            th, ps = params[site]
            m = torch.sigmoid(th / temp[0])
            pen = pen + LAM * m.sum()
        pen.backward()
        opt.step()
        if step % 100 == 0:
            print("  step %d | data %.4f | %.0fs" % (step, float(v.mean()), time.time() - t0), flush=True)
    out = {}
    with torch.no_grad():
        for site in SITES:
            th, ps = params[site]
            keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(as_tuple=True)[0]
            out[site] = {int(i): float(a) for i, a in
                          zip(keep.tolist(), F.softplus(ps)[keep].tolist())}
    return out


t0 = time.time()
members = fit()
n_mem = sum(len(d) for d in members.values())
m_full = mean_value(ho, {})
m_empty = mean_value(ho, circuit_transforms({}))
m_circ = mean_value(ho, circuit_transforms(members))
m_a1 = mean_value(ho, circuit_transforms(members, use_amps=False))
den = m_full - m_empty
ef = lambda m: (m - m_empty) / den
per_kind = {k: sum(len(d) for s_, d in members.items() if s_[0] == k) for k in KINDS}
print("\n%s on Gemma-2-2B | kinds %s | %d members %s | fit %.0fs" % (TASK, KINDS, n_mem, per_kind, time.time() - t0))
print("  frame                       value    EF")
print("  full model               %8.3f  %6.3f" % (m_full, 1.0))
print("  empty                    %8.3f  %6.3f" % (m_empty, 0.0))
print("  circuit + amplitudes     %8.3f  %6.3f" % (m_circ, ef(m_circ)))
print("  circuit at alpha=1       %8.3f  %6.3f" % (m_a1, ef(m_a1)))
for j in range(N_NULL):
    r2 = random.Random(100 + j)
    nullm = {}
    for site, d in members.items():
        ids = r2.sample(range(D_SAE), len(d)); amps = list(d.values()); r2.shuffle(amps)
        nullm[site] = dict(zip(ids, amps))
    mn = mean_value(ho, circuit_transforms(nullm))
    print("  amp-null %d               %8.3f  %6.3f" % (j, mn, ef(mn)))
print("  (value = %s on held-out)" % ("log p(target) - log p(contrast)" if contrasts is not None else "log p(target)"))
TAG = os.environ.get("TAG", "_".join(KINDS))
with open(HERE / ("%s_%s_gemma_members.jsonl" % (TASK, TAG)), "w") as fh:
    fh.write(json.dumps({"task": TASK, "tier": TIER, "lam": LAM, "steps": STEPS,
                         "n_members": n_mem, "EF_ho": round(ef(m_circ), 4),
                         "EF_alpha1": round(ef(m_a1), 4),
                         "kinds": KINDS,
                         "alphas": {"%s/%d" % s_: {str(i): round(a, 4) for i, a in d.items()}
                                    for s_, d in members.items() if d}}) + "\n")
torch.save({"rows": rows, "n_tr": n_tr}, HERE / ("%s_gemma_prompts.pt" % TASK))
print("-> %s_%s_gemma_members.jsonl" % (TASK, TAG))
