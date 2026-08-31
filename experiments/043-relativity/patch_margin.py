"""Add the FIRING-MARGIN objective to learned_mask (method-list #11).

margin_topk=k switches the seed tap from
    seed_pre = x @ w + b
to
    seed_pre = (x @ w + b) - tau,   tau = k-th largest site pre-act
so the fitted/targeted quantity is the seed's margin over its SAE's
top-k cutoff -- competition-aware: if the masked stream raises rivals,
tau rises and the loss objects, which the value objective cannot see.
Because the change lives in the tap, _natural() automatically yields
natural-margin targets and every consumer (dual_norm, floors, loss)
works in the margin frame unchanged. Off by default; the value path is
bit-identical when unset. Same module-global pattern as _SIGNED_AMP.
Run from repo root.
"""
import ast

P = "src/circuit/instrument/learned_mask.py"
s = open(P, encoding="utf-8").read()

# 1. module global next to _SIGNED_AMP
anchor = "_SIGNED_AMP = [False]"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + (
    "\n# FIRING MARGIN (2026-08-30): when set, holds\n"
    "# (W_enc_site, b_eff_site, k) and the seed tap reports the seed's\n"
    "# MARGIN over the site's k-th largest pre-activation instead of the\n"
    "# raw pre-activation.\n"
    "_MARGIN = [None]"))

# 2. the seed tap
old = "            self.seed_pre = x @ w + b"
assert s.count(old) == 1
s = s.replace(old, """            _pre = x @ w + b
            if _MARGIN[0] is not None:
                _We, _be, _k = _MARGIN[0]
                _full = x.to(_We.dtype) @ _We.T + _be
                _tau = torch.topk(_full, _k, dim=-1).values[..., -1]
                _pre = _pre - _tau.to(_pre.dtype)
            self.seed_pre = _pre""")

# 3. kwarg + activation in run_learned_mask
old = "    signed_amplitude: bool = False,"
assert s.count(old) == 1
s = s.replace(old, old + "\n    margin_topk: \"Optional[int]\" = None,")

old = "    _SIGNED_AMP[0] = bool(signed_amplitude)"
assert s.count(old) == 1
s = s.replace(old, old + """
    if margin_topk is not None:
        _sae_m = bank.saes[seed_kind][seed_layer]
        _MARGIN[0] = (_sae_m.encoder.weight.detach(),
                      _sae_m._get_bias_eff().detach(), int(margin_topk))
    else:
        _MARGIN[0] = None""")

open(P, "w", encoding="utf-8", newline="").write(s)
ast.parse(s)
print("margin objective added")
