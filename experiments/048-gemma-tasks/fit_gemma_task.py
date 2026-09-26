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
OUT = Path(os.environ.get("OUT", str(HERE)))   # where members/prompts are written
sys.path.insert(0, str(HERE.parent / "037-gemmascope"))
import gemma_loader as G   # noqa: E402  (SAE download/cache + tier ladder)
# SAE_SOURCE: gs1 = GemmaScope 1 for Gemma-2 (npz, TIER ladder; the default,
# unchanged) | gs2 = Gemma Scope 2 for Gemma 3 (safetensors via
# 052-gemma3-270m/gemmascope2.py; width/L0 from GS2_WIDTH / GS2_L0, TIER unused).
SAE_SOURCE = os.environ.get("SAE_SOURCE", "gs1")
if SAE_SOURCE == "gs2":
    sys.path.insert(0, str(HERE.parent / "052-gemma3-270m"))
    import gemmascope2 as GS2   # noqa: E402
# SUB_BDEC=1 encodes (x - b_dec) instead of x (052/sae_recon.py measures which
# convention reproduces the configured L0). The edit delta is unaffected.
SUB_BDEC = os.environ.get("SUB_BDEC", "0") == "1"
# FREEZE_NORM=1: Gemma applies post_attention_layernorm (RMSNorm) AFTER o_proj,
# i.e. after our att site. Unfrozen, a zero-filled att site is not a removal:
# whatever survives the edit (SAE error + b_dec + members) is rescaled back to
# full attention-output magnitude. Frozen, that norm divides by the RMS of the
# PRE-EDIT o_proj output of the same forward pass (detached), so an edit that
# shrinks the input shrinks the output. Default 0 = historical behaviour.
# (mlp sites sit after post_feedforward_layernorm, so they need no freeze.)
FREEZE_NORM = os.environ.get("FREEZE_NORM", "0") == "1"
_ATT_IN = {}

TASK = os.environ.get("TASK", "ioi")
N = int(os.environ.get("N", 256))
STEPS = int(os.environ.get("STEPS", 400))
LAM = float(os.environ.get("LAM", 3e-3))
LR = float(os.environ.get("LR", 0.05))
TIER = int(os.environ.get("TIER", 2))
# FLOOR: what a NON-member latent takes.
#   "zero" — the historical zero-fill. Fine for att/mlp sites (the residual
#            stream still flows), but at RES sites it deletes the
#            SAE-explained part of the whole stream, so the circuit would
#            have to rebuild it: measured EF 0.008 on induction with
#            att+mlp+res (2026-09-09).
#   "mean" — SFC's mean ablation: non-members sit at their mean over the
#            training prompts, so the circuit explains the DEVIATION from
#            the mean. Required for res sites.
#   "dual" — BOTH, scored every step (our engine's dual/triple floor minus
#            the negctx term, which needs negative contexts these task
#            datasets do not have): a member must earn its place under
#            zero-fill AND mean-fill. Weight on the mean term = FLOOR_W.
FLOOR = os.environ.get("FLOOR", "zero")
FLOOR_W = float(os.environ.get("FLOOR_W", 0.25))
FLOOR_TERMS = ({"zero": [("zero", 1.0)], "mean": [("mean", 1.0)],
                "dual": [("zero", 1.0), ("mean", FLOOR_W)]}[FLOOR])
NEEDS_MEANS = any(f == "mean" for f, _ in FLOOR_TERMS)
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
# LAYER SELECTION. All three kinds at all 26 layers is 11.3 GB of SAEs in
# bf16 (att 134 MB + mlp 151 + res 151 per layer) on top of the 5.2 GB
# model — over a 16 GB card. LAYER_STRIDE=2 keeps all three kinds at every
# other layer (14 layers incl. the last, 6.1 GB) which still covers L22
# where the name movers sit; LAYERS=0,4,8,... overrides explicitly.
LAYER_STRIDE = int(os.environ.get("LAYER_STRIDE", 1))
LAYERS = ([int(x) for x in os.environ["LAYERS"].split(",") if x != ""] if os.environ.get("LAYERS")
          else sorted(set(list(range(0, N_LAYERS, LAYER_STRIDE)) + [N_LAYERS - 1])))
SITES = [(k, l) for k in KINDS for l in LAYERS]
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
        if SAE_SOURCE == "gs2":
            _TC[site] = GS2.load_sae(kind, layer, device=DEV, dtype=DTYPE)
            return _TC[site]
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
print("%s: %d layers | kinds %s | layers %s | %d sites | SAE width %d | %s"
      % (MODEL_ID, N_LAYERS, KINDS, LAYERS, len(SITES), D_SAE,
         "Gemma Scope 2 %s/l0 %s sub_bdec=%s" % (GS2.WIDTH, GS2.L0, SUB_BDEC) if SAE_SOURCE == "gs2"
         else "GemmaScope 1 tier %d" % TIER), flush=True)


def features(site, x):
    t = tc(site)
    if SUB_BDEC:
        x = x - t["b_dec"]
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
                def att_pre(m, i, _s=site, _f=fn, _l=layer):
                    if FREEZE_NORM:
                        _ATT_IN[_l] = i[0]
                    return (_edit(_s, i[0], _f),) + tuple(i[1:])
                self.handles.append(blk.self_attn.o_proj.register_forward_pre_hook(att_pre))
                if FREEZE_NORM:
                    def frozen_norm(m, a, o, _l=layer, _blk=blk):
                        y = a[0]
                        yc = F.linear(_ATT_IN[_l], _blk.self_attn.o_proj.weight)
                        rms = torch.sqrt(yc.float().pow(2).mean(-1, keepdim=True) + m.eps).detach()
                        return (y.float() / rms * (1.0 + m.weight.float())).to(o.dtype)
                    self.handles.append(blk.post_attention_layernorm.register_forward_hook(frozen_norm))
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
    rows_path = os.environ.get("ROWS")
    if rows_path:
        # Externally generated dataset (050-gemma-circuits/make_datasets.py):
        # [{prompt, target(int id), contrast(int id or null), meta}]. Falls
        # through to the same competence filter as the built-in tasks.
        rows = json.load(open(rows_path))
        for r in rows:
            r.setdefault("meta", {})
            r["contrast"] = None if r.get("contrast") is None else int(r["contrast"])
            r["target"] = int(r["target"])
    elif TASK == "ioi":
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


SITE_MEANS = {}


def collect_site_means(tokens, anchors, bs=8):
    """Mean dense code per site over the TRAIN prompts, real positions only
    (1..anchor; BOS excluded because it is never intervened, padding
    excluded because it would drag the mean toward the pad token)."""
    acc = {s: torch.zeros(D_SAE, device=DEV, dtype=torch.float32) for s in SITES}
    cnt = 0.0
    for i in range(0, int(tokens.shape[0]), bs):
        tk = tokens[i:i + bs].to(DEV)
        an = anchors[i:i + bs].to(DEV)
        pos = torch.arange(tk.shape[1], device=DEV)[None, :]
        keep = ((pos >= 1) & (pos <= an[:, None])).float()          # [B, T]
        cnt += float(keep.sum())

        def mk(_s):
            def fn(c, _k=keep):
                acc[_s] += (c.float() * _k[..., None]).sum(dim=(0, 1))
                return c                                            # identity: delta = 0
            return fn
        with torch.no_grad(), Runner({s: mk(s) for s in SITES}):
            model(tk)
    for s in SITES:
        SITE_MEANS[s] = (acc[s] / max(cnt, 1.0)).to(DTYPE)
    nz = {s: int((SITE_MEANS[s] > 0).sum()) for s in SITES}
    print("site means over %d token slots | non-zero entries per site: min %d median %d max %d"
          % (cnt, min(nz.values()), int(np.median(list(nz.values()))), max(nz.values())), flush=True)


def floor_for(site, c, mode):
    """The value non-members take, broadcast to c's shape."""
    if mode == "mean":
        return SITE_MEANS[site].to(c.dtype).expand_as(c)
    return torch.zeros_like(c)


def circuit_transforms(members, use_amps=True, floor="zero"):
    tr_ = {}
    for site in SITES:
        d = members.get(site, {})
        idx = torch.tensor(sorted(d), device=DEV, dtype=torch.long)
        al = torch.tensor([d[int(i)] if use_amps else 1.0 for i in idx.tolist()],
                          device=DEV, dtype=torch.float32)
        def fn(c, _idx=idx, _al=al, _s=site, _f=floor):
            chat = floor_for(_s, c, _f).clone()
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

        def trf_for(mode):
            t = {}
            for site in SITES:
                def fn(c, _s=site, _m=mode):
                    th, ps = params[_s]
                    m = torch.sigmoid(th / temp[0])
                    kept = (m * F.softplus(ps)).to(c.dtype) * c
                    if _m == "zero":
                        return kept
                    return kept + ((1.0 - m).to(c.dtype) * floor_for(_s, c, _m))
                t[site] = fn
            return t

        # One data term per floor: each is a separate forward+backward so the
        # graphs never coexist (the dual floor otherwise doubles peak VRAM).
        vs = {}
        for mode, w in FLOOR_TERMS:
            v_m = value(tk, an, tg, ct, trf_for(mode), True)
            (w * ((v_m - nat.to(DEV)) ** 2).mean() / tnorm).backward()
            vs[mode] = float(v_m.mean())
        v = v_m
        pen = 0.0
        for site in SITES:
            th, ps = params[site]
            m = torch.sigmoid(th / temp[0])
            pen = pen + LAM * m.sum()
        pen.backward()
        opt.step()
        if step % 100 == 0:
            print("  step %d | %s | %.0fs" % (step, " ".join("%s %.3f" % kv for kv in vs.items()),
                                              time.time() - t0), flush=True)
    out = {}
    with torch.no_grad():
        for site in SITES:
            th, ps = params[site]
            keep = (torch.sigmoid(th / temp[0]) > 0.5).nonzero(as_tuple=True)[0]
            out[site] = {int(i): float(a) for i, a in
                          zip(keep.tolist(), F.softplus(ps)[keep].tolist())}
    return out


# Site means are always collected: they are the mean-fill EVALUATION frame
# even when the fit itself used the zero floor (full eval matrix).
collect_site_means(tr[0], tr[1])
t0 = time.time()
members = fit()
fit_s = time.time() - t0
n_mem = sum(len(d) for d in members.values())
m_full = mean_value(ho, {})
per_kind = {k: sum(len(d) for s_, d in members.items() if s_[0] == k) for k in KINDS}
print("\n%s on %s | kinds %s | floor %s | %d members %s | fit %.0fs"
      % (TASK, MODEL_ID, KINDS, FLOOR, n_mem, per_kind, fit_s))
print("  frame                        zero-fill        mean-fill")
print("  %-22s %8.3f          %8.3f" % ("full model", m_full, m_full))
scores = {}
for fl in ("zero", "mean"):
    e = mean_value(ho, circuit_transforms({}, floor=fl))
    c_ = mean_value(ho, circuit_transforms(members, floor=fl))
    a_ = mean_value(ho, circuit_transforms(members, use_amps=False, floor=fl))
    nulls = []
    for j in range(N_NULL):
        r2 = random.Random(100 + j)
        nullm = {}
        for site, d in members.items():
            ids = r2.sample(range(D_SAE), len(d)); amps = list(d.values()); r2.shuffle(amps)
            nullm[site] = dict(zip(ids, amps))
        nulls.append(mean_value(ho, circuit_transforms(nullm, floor=fl)))
    den = m_full - e
    scores[fl] = dict(empty=e, circ=c_, a1=a_, nulls=nulls,
                      EF=(c_ - e) / den if abs(den) > 1e-9 else None,
                      EF_a1=(a_ - e) / den if abs(den) > 1e-9 else None,
                      EF_nulls=[(x - e) / den if abs(den) > 1e-9 else None for x in nulls])
f = lambda k, fn: "  %-22s %8.3f (EF %5s)  %8.3f (EF %5s)" % (
    k, scores["zero"][fn], "%.3f" % scores["zero"]["EF" + ("" if fn == "circ" else "_a1")] if scores["zero"]["EF"] is not None else "-",
    scores["mean"][fn], "%.3f" % scores["mean"]["EF" + ("" if fn == "circ" else "_a1")] if scores["mean"]["EF"] is not None else "-")
print("  %-22s %8.3f          %8.3f" % ("empty", scores["zero"]["empty"], scores["mean"]["empty"]))
print(f("circuit + amplitudes", "circ"))
print(f("circuit at alpha=1", "a1"))
for j in range(N_NULL):
    print("  %-22s %8.3f (EF %5.3f)  %8.3f (EF %5.3f)"
          % ("amp-null %d" % j, scores["zero"]["nulls"][j], scores["zero"]["EF_nulls"][j],
             scores["mean"]["nulls"][j], scores["mean"]["EF_nulls"][j]))
PRIMARY = FLOOR_TERMS[0][0]
m_circ, m_a1 = scores[PRIMARY]["circ"], scores[PRIMARY]["a1"]
ef = lambda m: (m - scores[PRIMARY]["empty"]) / (m_full - scores[PRIMARY]["empty"])
print("  (value = %s on held-out; EF denominators are each frame's own empty; primary frame = %s)"
      % ("log p(target) - log p(contrast)" if contrasts is not None else "log p(target)", PRIMARY))
TAG = os.environ.get("TAG", "_".join(KINDS))
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / ("%s_%s_gemma_members.jsonl" % (TASK, TAG)), "w") as fh:
    fh.write(json.dumps({"task": TASK, "model": MODEL_ID, "sae_source": SAE_SOURCE,
                         "sae": ({"width": GS2.WIDTH, "l0": GS2.L0, "sub_bdec": SUB_BDEC}
                                 if SAE_SOURCE == "gs2" else {"tier": TIER}),
                         "tier": TIER, "lam": LAM, "lr": LR, "steps": STEPS, "freeze_norm": FREEZE_NORM,
                         "n_members": n_mem, "EF_ho": round(ef(m_circ), 4),
                         "EF_alpha1": round(ef(m_a1), 4),
                         "floor": FLOOR, "floor_w": FLOOR_W, "layers": LAYERS,
                         "scores": {k: {kk: (round(vv, 4) if isinstance(vv, float) else
                                             [round(x, 4) for x in vv] if isinstance(vv, list) else vv)
                                        for kk, vv in v.items()} for k, v in scores.items()},
                         "kinds": KINDS,
                         "alphas": {"%s/%d" % s_: {str(i): round(a, 4) for i, a in d.items()}
                                    for s_, d in members.items() if d}}) + "\n")
torch.save({"rows": rows, "n_tr": n_tr}, OUT / ("%s_gemma_prompts.pt" % TASK))
print("-> %s_%s_gemma_members.jsonl" % (TASK, TAG))
