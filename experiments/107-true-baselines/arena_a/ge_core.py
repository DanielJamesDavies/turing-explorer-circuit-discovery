"""Arena A1 (DAN-192 / DAN-197): the passes through Ge et al.'s own code (OpenMOSS lm_saes v0.1.0, unmodified) that the
fidelity checks (ge_checks.py g2 / g4) and the production runner (run_ge.py) share. Run in ~/venvs/ge107 with
`python -I -B` (the paths below replace PYTHONPATH, which -I ignores).

  hier_pass      THEIR hierarchical_attribute_transformer_with_saes(model, saes, input, target, candidates, threshold),
                 called as shipped; around it, [ours, labelled]: LN scale frozen (their detach_hook on ln1 / ln2 /
                 ln_final hook_scale, ge.md A3) and retain_grad on resid_mid / resid_post to read the error leaves.
  ungated_pass   the same linear graph (their apply_sae splice, their detach_hook on every hook_attn_scores, their
                 Attributor.cache_nodes forward, one target.backward()) with no gate: what their attributor computes at
                 threshold -inf before its read-out (which would enumerate every (position, latent) entry into networkx).
                 Post-hoc sets (ge.md section 3.6) are thresholds of these attributions.
  masked_pass    ge.md A11 [ours]: the backward with a fixed keep mask (gradient only through the kept set), through
                 the same kind of HookPoint backward hook their gate uses. Recovery of a post-hoc set is read here.
  linear_forward ge.md F2 [ours]: the linearised model run forward (LN scales and patterns at their clean values, ReLU
                 gates at the clean pattern, errors as clean constants) with only the kept nodes' decoder writes.

Leaf parts (every pass): embed (W_E, W_pos), bias_model (every b_* of GPT-2), bias_dict (dictionary encoder / decoder
biases), err_tc, err_z. Their sum is the root (Theorem 3.1) when LN is frozen.
"""
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LMS = ROOT / "third_party" / "Language-Model-SAEs"
for _p in (str(LMS / "TransformerLens"), str(LMS / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from lm_saes.circuit.attributors import HierachicalAttributor  # noqa: E402
from lm_saes.circuit.context import apply_sae  # noqa: E402
from lm_saes.circuit.graph import Node  # noqa: E402
from lm_saes.circuit.transformer import hierarchical_attribute_transformer_with_saes  # noqa: E402
from lm_saes.utils.hooks import detach_hook, retain_grad_hook  # noqa: E402

import ge_dicts as GD  # noqa: E402

N_LAYER = 12


def kind_layer(sae):
    if sae.cfg.hook_point_out.endswith("hook_mlp_out"):
        return "tc", int(sae.cfg.hook_point_out.split(".")[1])
    return "z", int(sae.cfg.hook_point_in.split(".")[1])


def ln_frozen_hooks():
    """[ours, ge.md A3] their own detach_hook on every LayerNorm scale"""
    h = [("blocks.%d.ln%d.hook_scale" % (l, k), detach_hook) for l in range(N_LAYER) for k in (1, 2)]
    return h + [("ln_final.hook_scale", detach_hook)]


def scores_detached():
    """as their transformer.py:70"""
    return [("blocks.%d.attn.hook_attn_scores" % i, detach_hook) for i in range(N_LAYER)]


def leaf_params(model, saes):
    out = []
    for n, p in model.named_parameters():
        last = n.split(".")[-1]
        if last in ("W_E", "W_pos") or last.startswith("b_"):
            out.append((n, p))
    for s in saes:
        out += [("%s.encoder.bias" % s.cfg.hook_point_in, s.encoder.bias),
                ("%s.decoder.bias" % s.cfg.hook_point_out, s.decoder.bias)]
    return out


def prepare(model, saes):
    """leaf parameters take gradients (to read their attributions); every other weight is frozen. This changes no
    activation gradient (the activations still require grad through W_E / W_pos), only what is accumulated."""
    leaves = leaf_params(model, saes)
    ids = {id(p) for _, p in leaves}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    for s in saes:
        for p in s.parameters():
            p.requires_grad_(id(p) in ids)
    return leaves


def candidates(saes):
    """feature acts only (README section 3: the arena's OV-only scope drops the v0.1.0 default hook_attn_scores)"""
    return [Node(GD.feature_node_name(s)) for s in saes]


@torch.no_grad()
def clean_state(model, saes, toks):
    """clean forward: logits, dictionary errors, feature acts (ReLU gates), LN scales, attention patterns"""
    logits, cache = model.run_with_cache(toks)
    st = dict(logits=logits, err={}, acts={}, scale={}, pattern={})
    for s in saes:
        kind, l = kind_layer(s)
        hp = GD.feature_node_name(s)
        if kind == "tc":
            x, y = cache["blocks.%d.ln2.hook_normalized" % l], cache["blocks.%d.hook_mlp_out" % l]
            st["err"][("tc", l)] = y - s.forward(x, label=y)
            st["acts"][hp] = s.encode(x, label=y)
        else:
            z = cache["blocks.%d.attn.hook_z" % l]
            st["err"][("z", l)] = z - s.forward(z)
            st["acts"][hp] = s.encode(z.flatten(-2))
    for l in range(N_LAYER):
        for k in (1, 2):
            n = "blocks.%d.ln%d.hook_scale" % (l, k)
            st["scale"][n] = cache[n]
        st["pattern"][l] = cache["blocks.%d.attn.hook_pattern" % l]
    st["scale"]["ln_final.hook_scale"] = cache["ln_final.hook_scale"]
    st["W_O"] = {l: model.blocks[l].attn.W_O.detach() for l in range(N_LAYER)}
    return st


def root_value(st, target: Node) -> float:
    if target.hook_point is None:
        return float(target.reduce(st["logits"]))
    return float(target.reduce(st["acts"][target.hook_point]))


def _zero(leaves):
    for _, p in leaves:
        p.grad = None


def _keep_resid(grads):
    def keep(name):
        def h(t, hook):
            t.retain_grad()
            grads[name] = t
            return t
        return h
    return ([("blocks.%d.hook_resid_mid" % l, keep(("mid", l))) for l in range(N_LAYER)]
            + [("blocks.%d.hook_resid_post" % l, keep(("post", l))) for l in range(N_LAYER)])


def leaf_parts(leaves, st, grads):
    parts = {"embed": 0.0, "bias_model": 0.0, "bias_dict": 0.0, "err_tc": 0.0, "err_z": 0.0}
    for n, p in leaves:
        if p.grad is None:
            continue
        v = float((p.detach() * p.grad).sum())
        key = "embed" if n.split(".")[-1] in ("W_E", "W_pos") else ("bias_dict" if "coder.bias" in n else "bias_model")
        parts[key] += v
    for (kind, l), e in st["err"].items():
        if kind == "tc":
            g = grads[("post", l)].grad
            parts["err_tc"] += 0.0 if g is None else float((e * g).sum())
        else:
            g = grads[("mid", l)].grad
            if g is not None:
                parts["err_z"] += float(torch.einsum("bphd,hdm,bpm->", e, st["W_O"][l], g))
    return parts


def hier_pass(model, saes, toks, target, cands, tau, leaves, st):
    """THEIR function, unmodified, at threshold tau. Returns ({(hook_point, 'b.pos.feat'): attribution}, leaf parts).

    Memory glue [ours, no effect on any value]: their gate hook keeps the candidate tensor in a closure (v) that the
    tensor's own autograd graph references through the HookPoint's full backward hook, a cycle through C++ that Python's
    gc cannot free (~120 MB per pass at T = 16 on 24 dictionaries, measured). After the pass has been read, the feature
    tensors' storage and grads are released (a plain torch forward hook on each sae.hook_feature_acts captures them)."""
    _zero(leaves)
    grads, held = {}, []
    handles = [s.hook_feature_acts.register_forward_hook(lambda m, i, o: held.append(o)) for s in saes]
    handles += [s.hook_feature_acts.register_forward_pre_hook(lambda m, i: held.extend(i)) for s in saes]
    try:
        with model.hooks(fwd_hooks=_keep_resid(grads) + ln_frozen_hooks()):
            circuit = hierarchical_attribute_transformer_with_saes(model, saes, toks, target, cands, threshold=tau)
        nodes = {(n.hook_point, n.reduction): float(d["attribution"]) for n, d in circuit.nodes(data=True)
                 if n != target}
        parts = leaf_parts(leaves, st, grads)
    finally:
        for h in handles:
            h.remove()
        for t in held:
            t.grad = None
            t.data = torch.empty(0, dtype=t.dtype, device=t.device)
    return nodes, parts


def _mask_hook(m):
    def h(grad, hook):
        return (grad * m,)
    return h


def ungated_pass(model, saes, toks, target, cands, leaves, st, masks=None):
    """the gate-off backward through their linear graph (or, with masks {hook_point: [1, T, d] 0/1}, the A11 fixed-mask
    backward). Returns ({hook_point: attribution [T, d] for batch 0}, leaf parts)."""
    _zero(leaves)
    grads = {}
    fwd = _keep_resid(grads) + ln_frozen_hooks() + [(c.hook_point, retain_grad_hook) for c in cands]
    bwd = [] if masks is None else [(hp, _mask_hook(m)) for hp, m in masks.items()]
    with apply_sae(model, saes):
        with model.hooks(fwd_hooks=fwd + scores_detached(), bwd_hooks=bwd):
            cache = HierachicalAttributor(model).cache_nodes(toks, cands + [target])
            cache[target].backward()
            attrs = {}
            for c in cands:
                g = cache.grad(c)
                attrs[c.hook_point] = None if g is None else (g * cache[c]).detach()[0]
    return attrs, leaf_parts(leaves, st, grads)


def masks_from_nodes(nodes, st, target):
    """{(hook_point, 'b.pos.feat')} -> {hook_point: [1, T, d] float mask}; the root entry is always kept (A8)"""
    m = {hp: torch.zeros_like(a) for hp, a in st["acts"].items()}
    for hp, red in list(nodes) + ([(target.hook_point, target.reduction)] if target.hook_point is not None else []):
        b, p, f = map(int, red.split("."))
        m[hp][b, p, f] = 1.0
    return m


def posthoc_nodes(attrs, tau, target):
    """S = {v : attr_v > tau} after the ungated pass (their read-out rule, signed, strict), root excluded"""
    out = {}
    for hp, a in attrs.items():
        if a is None:
            continue
        for p, f in (a > tau).nonzero().tolist():
            red = "0.%d.%d" % (p, f)
            if hp == target.hook_point and red == target.reduction:
                continue
            out[(hp, red)] = float(a[p, f])
    return out


@torch.no_grad()
def linear_forward(model, saes, toks, st, masks, target):
    """ge.md F2: the linearised model forward with only the masked-in nodes' decoder writes (LN scales and patterns at
    their clean values, ReLU gates at the clean pattern, errors and biases kept)."""
    fwd, feats, xin = [], {}, {}
    for n, v in st["scale"].items():
        fwd.append((n, lambda t, hook, v=v: v))
    for l, v in st["pattern"].items():
        fwd.append(("blocks.%d.attn.hook_pattern" % l, lambda t, hook, v=v: v))
    for s in saes:
        kind, l = kind_layer(s)
        hp = GD.feature_node_name(s)
        m = masks[hp] * (st["acts"][hp] > 0).float()
        e = st["err"][(kind, l)]

        def pre(x, s=s):
            if s.cfg.apply_decoder_bias_to_pre_encoder:
                x = x - s.decoder.bias
            return s.encoder(x)
        if kind == "z":
            def hz(t, hook, s=s, m=m, e=e, hp=hp, pre=pre):
                f = m * pre(t.flatten(-2))
                feats[hp] = f
                return s.decoder(f).view(t.shape) + e
            fwd.append(("blocks.%d.attn.hook_z" % l, hz))
        else:
            def hin(t, hook, l=l):
                xin[l] = t
                return t

            def hout(t, hook, s=s, m=m, e=e, hp=hp, l=l, pre=pre):
                f = m * pre(xin[l])
                feats[hp] = f
                return s.decoder(f) + e
            fwd += [("blocks.%d.ln2.hook_normalized" % l, hin), ("blocks.%d.hook_mlp_out" % l, hout)]
    logits = model.run_with_hooks(toks, fwd_hooks=fwd)
    if target.hook_point is None:
        return float(target.reduce(logits))
    return float(target.reduce(feats[target.hook_point]))


def tau_grid(a_t, n=30, lo=1e-5, hi=0.99):
    """ge.md A7 [ours: the range]: n values log-spaced in tau / a_t from lo to hi. hi < 1 because their gate also applies
    to the root's own tensor (grad 1 there, so v * grad = a_t must exceed tau or the whole circuit is empty)."""
    return [a_t * math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (n - 1)) for i in range(n)]


def recovery(parts, a_t):
    """A11 / A5: interp = interpretable leaves only (token + position embeddings, the G4 adaptation) / a_t;
    nonerr = every non-error leaf (embeddings + biases) / a_t; total = the in-pass root of G' / a_t."""
    s = sum(parts.values())
    return dict(interp=parts["embed"] / a_t, nonerr=(parts["embed"] + parts["bias_model"] + parts["bias_dict"]) / a_t,
                total=s / a_t)


def rss_gb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0


def maybe_reexec(limit_gb: float):
    """a residual leak of a few MB per hierarchical pass remains after hier_pass's release (autograd nodes kept by the
    cycle). Callers are resumable per unit of work, so past limit_gb the process re-executes itself (same argv, same
    -I -B) and carries on from what is written."""
    if limit_gb and rss_gb() > limit_gb:
        print("RSS %.1f GB > %.1f GB: re-exec to release leaked autograd memory" % (rss_gb(), limit_gb), flush=True)
        import os
        os.execv(sys.executable, [sys.executable, "-I", "-B"] + sys.argv)
