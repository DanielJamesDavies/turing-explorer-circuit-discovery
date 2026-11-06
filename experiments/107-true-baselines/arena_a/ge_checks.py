"""Arena A1 (DAN-192): fidelity checks for Ge et al.'s own code (OpenMOSS lm_saes v0.1.0, unmodified) on the public GPT-2
dictionaries (ge_dicts.py). CPU, in ~/venvs/ge107:

  source ~/venvs/ge107/bin/ge_env.sh
  CUDA_VISIBLE_DEVICES= python -I -B experiments/107-true-baselines/arena_a/ge_checks.py format|g1|g2|g3b|g4

  g2 / g4 (2026-11-05) use ge_core.py; see their docstrings and LOG_2026-11-05_cpu.md.

  format  the dictionary files and conventions, measured: Kissane z-SAE forward (b_dec centring or not; value biases
          folded into b_O as TL's default fold_value_biases=True, or not), FVU / L0 per layer on OWT; parity of the
          lm_saes conversions against the source formulas (activations and reconstructions).
  g1      ge.md F1, Theorem 3.1 at the full graph, gate OFF: one plain backward through THEIR linear graph (their
          apply_sae splice, their detach_hook on hook_attn_scores, their Attributor.cache_nodes forward); the gate is
          not involved (calling their attributor with threshold -inf is the same backward, but its read-out enumerates
          ~8M (position, latent) entries into networkx). Roots = a transcoder latent, a z latent and the IO logit.
          Euler sum over every leaf (token + position embeddings, every bias, every dictionary error) vs the root.
          Two rows: LN as shipped (scale live: not linear, a gap is expected) and LN frozen (their detach_hook on
          ln1/ln2/ln_final hook_scale; [ours], ge.md A3): must match to 1e-4 relative.
  g3b     ge.md F3(b): a set-separating toy through their HierachicalAttributor: in-pass gating keeps {D} at tau = 2,
          post-hoc pruning keeps {A, D}; signed gate drops E (attr -1) at tau = 0.5. (F3(a), their own unit test, is
          run by setup_env.sh: 3/3 pass.)
Writes out/ge_checks/<check>.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LMS = ROOT / "third_party" / "Language-Model-SAEs"
# `python -I` ignores PYTHONPATH, so the clone's paths go on sys.path here (as setup_env.sh's import check does)
for _p in (str(LMS / "TransformerLens"), str(LMS / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT = HERE / "out" / "ge_checks"
OWT = Path("/home/daniel/.cache/huggingface/hub/datasets--apollo-research--Skylion007-openwebtext-tokenizer-gpt2/"
           "snapshots/f02886b54795e8acabceb637ca119f9ae8f19d3f/data/train-00000-of-00073.parquet")

import torch  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from transformer_lens.hook_points import HookedRootModule, HookPoint  # noqa: E402

from lm_saes.circuit.attributors import HierachicalAttributor  # noqa: E402
from lm_saes.circuit.context import apply_sae  # noqa: E402
from lm_saes.circuit.graph import Node  # noqa: E402
from lm_saes.circuit.transformer import hierarchical_attribute_transformer_with_saes  # noqa: E402
from lm_saes.utils.hooks import detach_hook  # noqa: E402

import ge_dicts as GD  # noqa: E402

BOS = 50256


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("%s.json" % name)).write_text(json.dumps(obj, indent=1, default=str))


def owt_tokens(n_rows=8, seq=128):
    import pyarrow.parquet as pq
    t = pq.read_table(OWT).slice(0, n_rows).to_pydict()
    col = [k for k in t if "id" in k or "token" in k][0]
    rows = [[BOS] + list(r[: seq - 1]) for r in t[col]]
    return torch.tensor(rows, dtype=torch.long), col


def fvu(x, xh):
    x, xh = x.reshape(-1, x.shape[-1]), xh.reshape(-1, xh.shape[-1])
    return float(((x - xh) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


# ------------------------------------------------------------------------------------------- format
def check_format():
    toks, col = owt_tokens()
    res = dict(owt_column=col, tokens=list(toks.shape), z={}, tc={})
    with torch.no_grad():
        m_fold = HookedTransformer.from_pretrained("gpt2", device="cpu")                       # TL defaults
        m_nofold = HookedTransformer.from_pretrained("gpt2", device="cpu", fold_value_biases=False)
        names = lambda n: ("hook_z" in n) or ("ln2.hook_normalized" in n) or ("hook_mlp_out" in n)
        _, c_f = m_fold.run_with_cache(toks, names_filter=names)
        _, c_n = m_nofold.run_with_cache(toks, names_filter=lambda n: "hook_z" in n)
        for l in range(GD.N_LAYER):
            r = GD.load_z_raw(l)
            row = {"d_sae": r["W_enc"].shape[1]}
            for conv, cache in (("z_folded_bV", c_f), ("z_with_bV", c_n)):
                x = cache["blocks.%d.attn.hook_z" % l][:, 1:].flatten(-2)                     # BOS excluded
                for cent in (True, False):
                    acts = torch.relu(((x - r["b_dec"]) if cent else x) @ r["W_enc"] + r["b_enc"])
                    xh = acts @ r["W_dec"] + r["b_dec"]
                    row["%s_%s" % (conv, "centred" if cent else "uncentred")] = dict(
                        fvu=fvu(x, xh), l0=float((acts > 0).float().sum(-1).mean()))
            sae = GD.z_sae(l)
            z4 = c_f["blocks.%d.attn.hook_z" % l]
            f_ref, out_ref = GD.source_forward_ref("z", l)(z4.flatten(-2))
            f_ours = sae.encode(z4.flatten(-2))
            row["lm_saes_parity"] = dict(max_dacts=float((f_ours - f_ref).abs().max()),
                                         max_drecon=float((sae.forward(z4).flatten(-2) - out_ref).abs().max()))
            res["z"][l] = row
            del sae
            t = GD.tc_sae(l)
            xin = c_f["blocks.%d.ln2.hook_normalized" % l]
            yout = c_f["blocks.%d.hook_mlp_out" % l]
            f_ref, out_ref = GD.source_forward_ref("tc", l)(xin)
            res["tc"][l] = dict(max_dacts=float((t.encode(xin, label=yout) - f_ref).abs().max()),
                                max_drecon=float((t.forward(xin, label=yout) - out_ref).abs().max()),
                                fvu_mlp_out=fvu(yout[:, 1:], out_ref[:, 1:]),
                                l0=float((f_ref[:, 1:] > 0).float().sum(-1).mean()))
            del t
            print("L%d z %s | tc %s" % (l, {k: (round(v["fvu"], 3), round(v["l0"], 1)) for k, v in row.items()
                                            if isinstance(v, dict) and "fvu" in v},
                                       {k: round(v, 6) for k, v in res["tc"][l].items()}), flush=True)
            print("    z parity", row["lm_saes_parity"], flush=True)
    save("format", res)


# ------------------------------------------------------------------------------------------- G1
IOI = "When Mary and John went to the store, John gave the bag to"


def _leaf_params(model, saes):
    out = []
    for n, p in model.named_parameters():
        last = n.split(".")[-1]
        if last in ("W_E", "W_pos") or last.startswith("b_"):
            out.append((n, p))
    for s in saes:
        out += [("%s.encoder.bias" % s.cfg.hook_point_in, s.encoder.bias),
                ("%s.decoder.bias" % s.cfg.hook_point_out, s.decoder.bias)]
    return out


def check_g1():
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    saes = GD.all_saes()
    for s in saes:
        s.encoder.bias.requires_grad_(True)
        s.decoder.bias.requires_grad_(True)
    toks = model.to_tokens(IOI)                                                              # BOS prepended
    T = toks.shape[1]
    pos = T - 1
    io = model.to_single_token(" Mary")
    # clean values and dictionary errors (forward values with the splice equal the clean ones)
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)
        err = {}
        for s in saes:
            if s.cfg.hook_point_out.endswith("hook_mlp_out"):
                l = int(s.cfg.hook_point_out.split(".")[1])
                x, y = cache["blocks.%d.ln2.hook_normalized" % l], cache["blocks.%d.hook_mlp_out" % l]
                err[("tc", l)] = y - s.forward(x, label=y)
            else:
                l = int(s.cfg.hook_point_in.split(".")[1])
                z = cache["blocks.%d.attn.hook_z" % l]
                err[("z", l)] = z - s.forward(z)
        acts = {}
        for s in saes:
            kind = "tc" if s.cfg.hook_point_out.endswith("hook_mlp_out") else "z"
            l = int(s.cfg.hook_point_in.split(".")[1])
            x = (cache["blocks.%d.ln2.hook_normalized" % l], cache["blocks.%d.hook_mlp_out" % l]) if kind == "tc" \
                else (cache["blocks.%d.attn.hook_z" % l], None)
            acts[(kind, l)] = s.encode(x[0], label=x[1]) if kind == "tc" else s.encode(x[0].flatten(-2))
    f_tc = acts[("tc", 8)][0, pos]
    f_z = acts[("z", 9)][0, pos]
    roots = {"tc8.%d" % int(f_tc.argmax()): Node("blocks.8.hook_mlp_out.sae.hook_feature_acts",
                                                 "0.%d.%d" % (pos, int(f_tc.argmax()))),
             "z9.%d" % int(f_z.argmax()): Node("blocks.9.attn.hook_z.sae.hook_feature_acts",
                                               "0.%d.%d" % (pos, int(f_z.argmax()))),
             "logit_Mary": Node(None, "0.%d.%d" % (pos, io))}
    candidates = [Node(GD.feature_node_name(s)) for s in saes]
    leaves = _leaf_params(model, saes)
    W_O = {l: model.blocks[l].attn.W_O.detach() for l in range(12)}
    res = dict(prompt=IOI, T=T, rows=[])
    for ln_mode in ("shipped", "ln_frozen"):
        for rname, target in roots.items():
            model.zero_grad(set_to_none=True)
            for _, p in leaves:
                p.grad = None
            grads = {}

            def keep(name):
                def h(t, hook):
                    t.retain_grad()
                    grads[name] = t
                    return t
                return h
            fwd = [("blocks.%d.hook_resid_mid" % l, keep(("mid", l))) for l in range(12)]
            fwd += [("blocks.%d.hook_resid_post" % l, keep(("post", l))) for l in range(12)]
            if ln_mode == "ln_frozen":                                    # [ours] A3: their detach_hook on the LN scales
                fwd += [("blocks.%d.ln1.hook_scale" % l, detach_hook) for l in range(12)]
                fwd += [("blocks.%d.ln2.hook_scale" % l, detach_hook) for l in range(12)]
                fwd += [("ln_final.hook_scale", detach_hook)]
            t0 = time.time()
            # Gate off = a plain backward through THEIR linear graph: their apply_sae splice (context.py) and their
            # detach_hook on every hook_attn_scores (transformer.py:70), their Attributor.cache_nodes forward, one
            # target.backward(). (Calling their attributor with threshold -inf gives the same backward, but its read-out
            # then enumerates every (position, latent) entry of 24 dictionaries into networkx: ~8M nodes, impractical.)
            with apply_sae(model, saes):
                with model.hooks(fwd_hooks=fwd + [("blocks.%d.attn.hook_attn_scores" % i, detach_hook)
                                                  for i in range(12)]):
                    c = HierachicalAttributor(model).cache_nodes(toks, [target])
                    root = c[target]
                    root.backward()
            dt = time.time() - t0
            a_t = float(root)
            parts = {"embed": 0.0, "bias_model": 0.0, "bias_dict": 0.0, "err_tc": 0.0, "err_z": 0.0}
            for n, p in leaves:
                if p.grad is None:
                    continue
                v = float((p.detach() * p.grad).sum())
                key = "embed" if n.split(".")[-1] in ("W_E", "W_pos") else ("bias_dict" if "coder.bias" in n
                                                                           else "bias_model")
                parts[key] += v
            for (kind, l), e in err.items():
                if kind == "tc":
                    g = grads[("post", l)].grad
                    parts["err_tc"] += 0.0 if g is None else float((e * g).sum())
                else:
                    g = grads[("mid", l)].grad
                    if g is not None:
                        parts["err_z"] += float(torch.einsum("bphd,hdm,bpm->", e, W_O[l], g))
            S = sum(parts.values())
            row = dict(ln=ln_mode, root=rname, a_t=a_t, leaf_sum=S, rel_gap=abs(S - a_t) / max(abs(a_t), 1e-12),
                       parts=parts, seconds=dt,
                       passed=(abs(S - a_t) / max(abs(a_t), 1e-12) <= 1e-4) if ln_mode == "ln_frozen" else None)
            res["rows"].append(row)
            print("%-10s %-12s a_t %.6f  leaf sum %.6f  rel gap %.2e  %.1fs  %s" % (
                ln_mode, rname, a_t, S, row["rel_gap"], dt, {k: round(v, 4) for k, v in parts.items()}), flush=True)
    save("g1", res)


# ------------------------------------------------------------------------------------------- G3(b)
class Toy(HookedRootModule):
    """x = 1; A = x; B_k = A (k = 1..4); D = 3x; E = -x; t = sum_k B_k + D + E = 6. attr: A 4, B_k 1, D 3, E -1."""

    def __init__(self):
        super().__init__()
        self.hook_A, self.hook_B, self.hook_D, self.hook_E = HookPoint(), HookPoint(), HookPoint(), HookPoint()
        self.setup()

    def forward(self, x):
        A = self.hook_A(x * 1.0)
        B = self.hook_B(A.repeat(4))
        D = self.hook_D(3.0 * x)
        E = self.hook_E(-1.0 * x)
        return (B.sum() + D + E).reshape(1)


def _kept(circuit, target, tau=None):
    out = {}
    for n, d in circuit.nodes(data=True):
        if n == target:
            continue
        if tau is None or d["attribution"] > tau:
            out["%s[%s]" % (n.hook_point, n.reduction)] = d["attribution"]
    return out


def check_g3b():
    model = Toy()
    cands = [Node("hook_A"), Node("hook_B"), Node("hook_D"), Node("hook_E")]
    target = Node(None, "0")
    res = {}
    for name, tau in (("hier_tau2", 2.0), ("hier_tau0.5", 0.5)):
        x = torch.tensor([1.0], requires_grad=True)
        res[name] = _kept(HierachicalAttributor(model).attribute(x, target, cands, threshold=tau), target)
    x = torch.tensor([1.0], requires_grad=True)
    ungated = HierachicalAttributor(model).attribute(x, target, cands, threshold=-math.inf)
    res["ungated"] = _kept(ungated, target)
    res["posthoc_tau2"] = _kept(ungated, target, tau=2.0)
    res["posthoc_tau0.5"] = _kept(ungated, target, tau=0.5)
    exp = {"hier_tau2": {"hook_D[0]": 3.0},
           "posthoc_tau2": {"hook_A[0]": 4.0, "hook_D[0]": 3.0},
           "hier_tau0.5": {"hook_A[0]": 4.0, "hook_B[0]": 1.0, "hook_B[1]": 1.0, "hook_B[2]": 1.0, "hook_B[3]": 1.0,
                           "hook_D[0]": 3.0},
           "ungated": {"hook_A[0]": 4.0, "hook_B[0]": 1.0, "hook_B[1]": 1.0, "hook_B[2]": 1.0, "hook_B[3]": 1.0,
                       "hook_D[0]": 3.0, "hook_E[0]": -1.0}}
    res["passed"] = {k: (res[k] == v) for k, v in exp.items()}
    res["expected"] = exp
    save("g3b", res)
    for k in exp:
        print("%-14s %s  got %s" % (k, "PASS" if res["passed"][k] else "FAIL", res[k]))


# ------------------------------------------------------------------------------------------- G2
def _g1_roots(model, st, toks):
    T = toks.shape[1]
    pos = T - 1
    f_tc = st["acts"]["blocks.8.hook_mlp_out.sae.hook_feature_acts"][0, pos]
    f_z = st["acts"]["blocks.9.attn.hook_z.sae.hook_feature_acts"][0, pos]
    io = model.to_single_token(" Mary")
    return {"tc8.%d" % int(f_tc.argmax()): Node("blocks.8.hook_mlp_out.sae.hook_feature_acts",
                                                "0.%d.%d" % (pos, int(f_tc.argmax()))),
            "z9.%d" % int(f_z.argmax()): Node("blocks.9.attn.hook_z.sae.hook_feature_acts",
                                              "0.%d.%d" % (pos, int(f_z.argmax()))),
            "logit_Mary": Node(None, "0.%d.%d" % (pos, io))}


def check_g2():
    """ge.md F2: Theorem 3.1 on gated G'. Their hierarchical_attribute_transformer_with_saes at tau (LN frozen) gives N'
    and the in-pass leaf sum (embeddings + biases + errors, gradient only through N'); the linearised forward
    (ge_core.linear_forward) with only N' kept must give the same root. Plus one 'all kept' row per root: the linearised
    forward with every node equals the clean root (a check of linear_forward itself)."""
    import ge_core as GC
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    saes = GD.all_saes()
    leaves = GC.prepare(model, saes)
    cands = GC.candidates(saes)
    toks = model.to_tokens(IOI)
    st = GC.clean_state(model, saes, toks)
    res = dict(prompt=IOI, T=toks.shape[1], rows=[])
    for rname, target in _g1_roots(model, st, toks).items():
        a_t = GC.root_value(st, target)
        ones = {hp: torch.ones_like(a) for hp, a in st["acts"].items()}
        lin_all = GC.linear_forward(model, saes, toks, st, ones, target)
        row = dict(root=rname, tau_rel=None, a_t=a_t, linear_root=lin_all, rel_gap=abs(lin_all - a_t) / abs(a_t))
        res["rows"].append(row)
        print("%-12s all kept      a_t %.6f  linear %.6f  rel gap %.2e" % (rname, a_t, lin_all, row["rel_gap"]),
              flush=True)
        for tau_rel in (1e-3, 1e-2, 1e-1):
            t0 = time.time()
            nodes, parts = GC.hier_pass(model, saes, toks, target, cands, tau_rel * a_t, leaves, st)
            S = sum(parts.values())
            lin = GC.linear_forward(model, saes, toks, st, GC.masks_from_nodes(nodes, st, target), target)
            gap = abs(lin - S) / max(abs(lin), 1e-12)
            # the A11 path (fixed-mask backward, used for post-hoc recovery) with mask = N' must reproduce the in-pass
            # parts exactly: in-pass gating is a fixed mask N' on the gradient
            _, parts_m = GC.ungated_pass(model, saes, toks, target, cands, leaves, st,
                                         masks=GC.masks_from_nodes(nodes, st, target))
            a11_gap = max(abs(parts_m[k] - parts[k]) for k in parts) / max(abs(S), 1e-12)
            row = dict(root=rname, tau_rel=tau_rel, a_t=a_t, n_nodes=len(nodes), in_pass_leaf_sum=S, linear_root=lin,
                       rel_gap=gap, rel_gap_vs_a_t=abs(lin - S) / abs(a_t), parts=parts, a11_parts=parts_m,
                       a11_rel_gap=a11_gap, seconds=time.time() - t0, passed=gap <= 1e-4 and a11_gap <= 1e-4)
            res["rows"].append(row)
            print("%-12s tau/a_t %.0e  N' %6d  leaf sum %.6f  linear %.6f  rel gap %.2e  A11-path gap %.2e  %s  %.1fs"
                  % (rname, tau_rel, len(nodes), S, lin, gap, a11_gap, "PASS" if row["passed"] else "FAIL",
                     row["seconds"]), flush=True)
    res["passed"] = all(r.get("passed", r["rel_gap"] <= 1e-4) for r in res["rows"])
    save("g2", res)
    print("G2", "PASS" if res["passed"] else "FAIL")


# ------------------------------------------------------------------------------------------- G4
NAMES = [" Mary", " John", " Tom", " James", " Dan", " Sid", " Martin", " Amy", " Kate", " Paul", " Mark", " Sarah",
         " Lisa", " David", " Michael", " Anna", " Laura", " Peter", " Emma", " George"]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
OBJECTS = ["ring", "bone", "basketball", "computer", "necklace", "drink", "snack", "book"]
# Wang et al. 2022 BABA templates (their ioi_dataset.py), prompt cut before the final IO name; ABBA swaps the first pair
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to",
]


def ioi_prompts(model, n_each=10, seed=0):
    """ge.md A12 [ours]: Wang et al. templates, single-token names, kept only if GPT-2's top-1 is the IO name; n_each
    ABBA and n_each BABA, seeded, deterministic."""
    import random
    rng = random.Random(seed)
    names = [n for n in NAMES if len(model.to_tokens(n, prepend_bos=False)[0]) == 1]
    out = {"ABBA": [], "BABA": []}
    tries = 0
    while min(len(v) for v in out.values()) < n_each:
        tries += 1
        assert tries < 5000, "IOI generation stalled"
        tmpl = rng.choice(BABA_TEMPLATES)
        a, b = rng.sample(names, 2)
        kind = rng.choice(["ABBA", "BABA"])
        if len(out[kind]) >= n_each:
            continue
        t = tmpl
        if kind == "ABBA":                                       # swap the first [B] and [A]
            i, j = t.index("[B]"), t.index("[A]")
            t = t[:i] + "[A]" + t[i + 3:j] + "[B]" + t[j + 3:]
        text = (t.replace("[PLACE]", rng.choice(PLACES)).replace("[OBJECT]", rng.choice(OBJECTS))
                .replace(" [A]", a).replace(" [B]", b).replace("[A]", a.strip()).replace("[B]", b.strip()))
        if any(p["text"] == text for v in out.values() for p in v):
            continue
        toks = model.to_tokens(text)
        with torch.no_grad():
            top = int(model(toks)[0, -1].argmax())
        io = model.to_single_token(a)
        if top == io:
            out[kind].append(dict(text=text, io=a, s=b, kind=kind, template=BABA_TEMPLATES.index(tmpl)))
    return [p for k in ("ABBA", "BABA") for p in out[k]], tries


def _curve_compare(curves, key, n_grid=60):
    """hierarchical vs post-hoc recovery at matched node counts: each curve linearly interpolated in log(nodes) over the
    overlap of the two curves' node ranges; mean over prompts on a common log grid (each grid point averages the
    prompts whose overlap covers it)."""
    import numpy as np
    per, lo_all, hi_all = [], [], []
    for c in curves:
        h = sorted((r["n"], r[key]) for r in c["hier"] if r["n"] > 0)
        p = sorted((r["n"], r[key]) for r in c["posthoc"] if r["n"] > 0)
        lo, hi = max(h[0][0], p[0][0]), min(h[-1][0], p[-1][0])
        if lo >= hi:
            per.append(dict(prompt=c["i"], overlap=None))
            continue
        g = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
        ih = np.interp(np.log(g), np.log([x for x, _ in h]), [y for _, y in h])
        ip = np.interp(np.log(g), np.log([x for x, _ in p]), [y for _, y in p])
        d = ih - ip
        per.append(dict(prompt=c["i"], overlap=[float(lo), float(hi)], frac_hier_ge=float((d >= -1e-6).mean()),
                        min_diff=float(d.min()), mean_diff=float(d.mean()), dominates=bool((d >= -1e-6).all()),
                        h=(h, p)))
        lo_all.append(lo)
        hi_all.append(hi)
    G = np.exp(np.linspace(np.log(min(lo_all)), np.log(max(hi_all)), n_grid))
    mh, mp, cnt = [], [], []
    for gx in G:
        vh, vp = [], []
        for r in per:
            if r["overlap"] is None or not (r["overlap"][0] <= gx <= r["overlap"][1]):
                continue
            h, p = r["h"]
            vh.append(np.interp(np.log(gx), np.log([x for x, _ in h]), [y for _, y in h]))
            vp.append(np.interp(np.log(gx), np.log([x for x, _ in p]), [y for _, y in p]))
        cnt.append(len(vh))
        mh.append(float(np.mean(vh)) if vh else float("nan"))
        mp.append(float(np.mean(vp)) if vp else float("nan"))
    for r in per:
        r.pop("h", None)
    ok = [i for i in range(n_grid) if cnt[i] > 0]
    mean_dom = all(mh[i] >= mp[i] - 1e-6 for i in ok)
    n_dom = sum(1 for r in per if r.get("dominates"))
    n_mostly = sum(1 for r in per if r.get("frac_hier_ge", 0) >= 0.5)
    return dict(grid=[float(x) for x in G], mean_hier=mh, mean_posthoc=mp, n_prompts_at_grid=cnt,
                mean_curve_dominates=mean_dom, prompts_dominating=n_dom, prompts_mostly_ge=n_mostly,
                n_prompts=len(per), per_prompt=per,
                passed=mean_dom and n_mostly > len(per) / 2)


def check_g4():
    """ge.md F4 (their Fig. 2b): 20 IOI prompts (A12), root = IO logit at the last position (A10), 30 tau log-spaced in
    tau / a_t (A7), hierarchical = their hierarchical_attribute_transformer_with_saes at each tau (LN frozen), post-hoc =
    the ungated attributions cut at the same tau, recovery of the post-hoc set by the A11 fixed-mask re-run. Nodes =
    feature nodes in N' excluding the root (A13). Recovery read two ways: interp (embeddings only, A11 / A5 literal)
    and nonerr (embeddings + biases; the bias mass is large on this substrate, G1)."""
    import ge_core as GC
    torch.manual_seed(0)
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    prompts, tries = ioi_prompts(model)
    save("g4_prompts", dict(prompts=prompts, tries=tries))
    print("IOI prompts: %d (%d draws)" % (len(prompts), tries), flush=True)
    saes = GD.all_saes()
    leaves = GC.prepare(model, saes)
    cands = GC.candidates(saes)
    part_file = OUT / "g4_curves.jsonl"
    done = {}
    if part_file.exists():                                   # resumable per prompt
        for line in part_file.read_text().splitlines():
            r = json.loads(line)
            done[r["i"]] = r
    for i, pr in enumerate(prompts):
        if i in done:
            continue
        t0 = time.time()
        toks = model.to_tokens(pr["text"])
        pos = toks.shape[1] - 1
        target = Node(None, "0.%d.%d" % (pos, model.to_single_token(pr["io"])))
        st = GC.clean_state(model, saes, toks)
        a_t = GC.root_value(st, target)
        assert a_t > 0, (pr, a_t)
        attrs, parts_u = GC.ungated_pass(model, saes, toks, target, cands, leaves, st)
        n_pos = sum(int((a > 0).sum()) for a in attrs.values() if a is not None)
        taus = GC.tau_grid(a_t)
        hier, post = [], []
        for tau in taus:
            nodes, parts = GC.hier_pass(model, saes, toks, target, cands, tau, leaves, st)
            hier.append(dict(tau=tau, n=len(nodes), parts=parts, **GC.recovery(parts, a_t)))
            S = GC.posthoc_nodes(attrs, tau, target)
            _, parts_m = GC.ungated_pass(model, saes, toks, target, cands, leaves, st,
                                         masks=GC.masks_from_nodes(S, st, target))
            post.append(dict(tau=tau, n=len(S), parts=parts_m, **GC.recovery(parts_m, a_t)))
        rec = dict(i=i, text=pr["text"], kind=pr["kind"], T=toks.shape[1], a_t=a_t, n_positive_ungated=n_pos,
                   ungated_parts=parts_u, g1_rel_gap=abs(sum(parts_u.values()) - a_t) / a_t, hier=hier, posthoc=post,
                   seconds=time.time() - t0)
        with part_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        done[i] = rec
        print("%2d %s a_t %.3f  T %d  nodes hier %d..%d post %d..%d (pos %d)  G1 gap %.1e  %.0fs  RSS %.1f GB" % (
            i, pr["kind"], a_t, toks.shape[1], hier[-1]["n"], hier[0]["n"], post[-1]["n"], post[0]["n"], n_pos,
            rec["g1_rel_gap"], rec["seconds"], GC.rss_gb()), flush=True)
        GC.maybe_reexec(9.0)
    curves = [done[i] for i in sorted(done)]
    res = dict(n_prompts=len(curves), max_g1_rel_gap=max(c["g1_rel_gap"] for c in curves),
               interp=_curve_compare(curves, "interp"), nonerr=_curve_compare(curves, "nonerr"),
               total=_curve_compare(curves, "total"))
    res["passed"] = res["interp"]["passed"]
    save("g4", res)
    for k in ("interp", "nonerr", "total"):
        r = res[k]
        print("G4 %-6s mean curve dominates %s; prompts dominating %d / %d, mostly >= %d / %d -> %s" % (
            k, r["mean_curve_dominates"], r["prompts_dominating"], r["n_prompts"], r["prompts_mostly_ge"],
            r["n_prompts"], "PASS" if r["passed"] else "FAIL"))


if __name__ == "__main__":
    {"format": check_format, "g1": check_g1, "g2": check_g2, "g3b": check_g3b, "g4": check_g4}[sys.argv[1]]()
