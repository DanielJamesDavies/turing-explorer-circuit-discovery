"""WHAT IS IN A GPT-2 CIRCUIT? — structure, then the members themselves.

Ranks members by ATTRIBUTION (d seed / d w_i = grad x activation at the anchor,
summed over the train positives — the same measure the fitter uses for roles),
then reads the strongest drivers and suppressors off Neuronpedia: its
auto-interp label, its firing rate, and a top context.

Labels are HINTS ONLY: in the concept-circuit work 3 of 8 auto-interp labels
were wrong without activation gating, so the firing rate and the context excerpt
are printed next to every label.

  FILE=latent_circuits_lam1e-2.jsonl SEED=resid-post:6:100 \
      python experiments/053-gpt2-topk/inspect_circuit.py
Env: FILE | SEED (all in file if unset) | TOP (12) | LABELS (1)
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gpt2saes as G  # noqa: E402
import neuronpedia as NP  # noqa: E402

FILE = os.environ.get("FILE", "latent_circuits_lam1e-2.jsonl")
WANT = os.environ.get("SEED", "")
TOP = int(os.environ.get("TOP", 12))
LABELS = os.environ.get("LABELS", "1") == "1"
N_POS = int(os.environ.get("N_POS", 48)); N_TR_FRAC = float(os.environ.get("N_TR_FRAC", 0.75))
RNG = int(os.environ.get("RNG", 0))
DEV = torch.device("cuda")
DTYPE = getattr(torch, os.environ.get("DTYPE", "float32"))

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=DTYPE).to(DEV).eval()
for p_ in model.parameters():
    p_.requires_grad_(False)
_SAE = {}


def sae(site):
    if site not in _SAE:
        _SAE[site] = G.load_sae(site[0], site[1], device=DEV, dtype=DTYPE)
    return _SAE[site]


class _Stop(Exception):
    pass


def _reg(hs, site, fn_x):
    mod = G.module_for(model, site[0], site[1])
    if site[0] == "resid-mid":
        hs.append(mod.register_forward_pre_hook(lambda m, a: (fn_x(a[0]),) + tuple(a[1:])))
    elif site[0] == "mlp-out":
        hs.append(mod.register_forward_hook(lambda m, a, o: fn_x(o)))
    else:
        def post(m, a, o):
            x = o[0] if isinstance(o, tuple) else o
            y = fn_x(x)
            return (y,) + tuple(o[1:]) if isinstance(o, tuple) else y
        hs.append(mod.register_forward_hook(post))


def attribution(seed, members, toks, anchors):
    """d(seed pre-act at anchor)/d w_i for each member, w_i = 1 (grad x act)."""
    kind, layer, idx = seed
    msets = {s: torch.tensor(sorted(d), device=DEV, dtype=torch.long) for s, d in members.items()}
    acc = {s: torch.zeros(len(v), device=DEV) for s, v in msets.items()}
    for s0 in range(0, toks.shape[0], 4):
        tk_, an_ = toks[s0:s0 + 4], anchors[s0:s0 + 4]
        W_ = {s: torch.ones(len(v), device=DEV, requires_grad=True) for s, v in msets.items()}
        out, hs = {}, []

        def mk(_s):
            def fn(x):
                t = sae(_s)
                xn, std, mu = G.norm_in(x)
                c = G.encode(t, xn)
                ch = c.clone()
                ch[..., msets[_s]] = c[..., msets[_s]] * W_[_s].to(c.dtype)
                return x + ((ch - c) @ t["W_dec"]) * std
            return fn
        for s in msets:
            _reg(hs, s, mk(s))

        def read(x):
            t = sae((kind, layer))
            xn, _, _ = G.norm_in(x)
            out["pre"] = ((xn.float() - t["b_dec"].float()) @ t["W_enc"][:, idx].float()
                          + t["b_enc"][idx].float())
            raise _Stop()
        _reg(hs, (kind, layer), read)
        try:
            with torch.enable_grad():
                model(tk_.to(DEV))
        except _Stop:
            pass
        finally:
            for h in hs:
                h.remove()
        v = out["pre"][torch.arange(tk_.shape[0], device=DEV), an_.to(DEV)].sum()
        g = torch.autograd.grad(v, [W_[s] for s in msets], allow_unused=True)
        for s, gi in zip(msets, g):
            if gi is not None:
                acc[s] += gi.detach().float()
    return {s: {int(i): float(a) for i, a in zip(msets[s].tolist(), acc[s].tolist())} for s in msets}


def describe(kind, layer, idx):
    """Neuronpedia label + firing rate + a top context, for one latent."""
    try:
        d = NP.feature(kind, layer, idx)
    except Exception as e:
        return "(neuronpedia: %s)" % type(e).__name__, None, ""
    lab = [e.get("description") for e in (d.get("explanations") or [])]
    acts = d.get("activations") or []
    ctx = ""
    if acts:
        a = max(acts, key=lambda r: r.get("maxValue", 0))
        j = int(a.get("maxValueTokenIndex", 0)); ts = a.get("tokens", [])
        ctx = "".join(ts[max(0, j - 7):j]) + "[[" + (ts[j] if j < len(ts) else "") + "]]"
        ctx = ctx.replace("\n", "\\n")[-60:]
    return (lab[0] if lab else "(no label)"), d.get("frac_nonzero"), ctx


recs = [json.loads(l) for l in open(HERE / FILE)]
for rec in recs:
    if WANT and rec["seed"] != WANT:
        continue
    kind, layer, idx = rec["seed"].split(":"); layer, idx = int(layer), int(idx)
    members = {(k.split("/")[0], int(k.split("/")[1])): {int(i): a for i, a in d.items()}
               for k, d in rec["members"].items()}
    n = rec["n_members"]
    print("\n" + "=" * 104)
    print("SEED %s | %d members | free0 %.3f | freeM_tk %.3f | phi_sup %.3f | phi_cf %.3f | lam %g"
          % (rec["seed"], n, rec["F0"], rec["FMd_tk"], (rec.get("table1") or {}).get("sup_role", float("nan")),
             (rec.get("table1") or {}).get("cf_alpha_role", float("nan")), rec["lam"]))
    if rec.get("label"):
        print("seed label (hint): %s" % rec["label"])

    # ---- structure
    print("\nMEMBERS BY SITE (rows = layer, columns = kind)")
    print("  %-5s %9s %9s %9s %9s %7s" % ("layer", *G.KINDS, "total"))
    per_layer = Counter()
    for l in range(layer + 1):
        row = [len(members.get((k, l), {})) for k in G.KINDS]
        if sum(row):
            per_layer[l] = sum(row)
            print("  L%-4d %9d %9d %9d %9d %7d" % (l, *row, sum(row)))
    kk = {k: sum(len(d) for s, d in members.items() if s[0] == k) for k in G.KINDS}
    print("  %-5s %9d %9d %9d %9d %7d" % ("all", *[kk[k] for k in G.KINDS], n))
    amps = [a for d in members.values() for a in d.values()]
    amps.sort()
    print("\nalpha: median %.2f | p10 %.2f | p90 %.2f | >2x natural: %d (%.0f%%) | <0.5x: %d (%.0f%%)"
          % (amps[len(amps) // 2], amps[len(amps) // 10], amps[9 * len(amps) // 10],
             sum(1 for a in amps if a > 2), 100 * sum(1 for a in amps if a > 2) / len(amps),
             sum(1 for a in amps if a < 0.5), 100 * sum(1 for a in amps if a < 0.5) / len(amps)))
    depth = [l for (k_, l), d in members.items() for _ in d]
    print("depth: members sit %.1f layers below the seed on average (min %d, max %d)"
          % (layer - sum(depth) / len(depth), layer - max(depth), layer - min(depth)))

    # ---- members themselves
    rng = random.Random(RNG + idx)
    rows = NP.contexts(kind, layer, idx, N_POS)
    rng.shuffle(rows)
    ids, anc = [], []
    for toks_, a_, v_ in rows:
        t_ = NP.to_ids(tok, toks_)[:a_ + 1]
        if len(t_) >= 2:
            ids.append(t_); anc.append(len(t_) - 1)
    T = max(len(i) for i in ids)
    toks_t = torch.tensor([i + [tok.eos_token_id] * (T - len(i)) for i in ids], dtype=torch.long)
    anch = torch.tensor(anc)
    n_tr = max(2, int(round(len(ids) * N_TR_FRAC)))
    attr = attribution((kind, layer, idx), members, toks_t[:n_tr], anch[:n_tr])
    flat = [(s, i, a, members[s][i]) for s, d in attr.items() for i, a in d.items()]
    flat.sort(key=lambda r: -r[2])
    pos_n = sum(1 for r in flat if r[2] >= 0)
    print("\nroles by attribution: %d drivers / %d suppressors" % (pos_n, len(flat) - pos_n))

    def show(title, rs):
        print("\n%s" % title)
        print("  %-16s %8s %6s %8s  %s" % ("site/latent", "attrib", "alpha", "fires", "label / context"))
        for s, i, a, al in rs:
            lab, fq, ctx = describe(s[0], s[1], i) if LABELS else ("", None, "")
            print("  %-16s %8.2f %6.2f %8s  %s" % ("%s/%d #%d" % (s[0], s[1], i), a, al,
                                                   ("%.4f" % fq) if fq else "-", lab[:62]))
            if ctx:
                print("  %-16s %8s %6s %8s  ...%s" % ("", "", "", "", ctx))
    show("STRONGEST DRIVERS", flat[:TOP])
    show("STRONGEST SUPPRESSORS", flat[-min(TOP // 2, len(flat)):][::-1])
