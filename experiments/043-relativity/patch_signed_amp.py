"""Add signed_amplitude to learned_mask: alpha = psi (raw, signed)
instead of softplus(psi). Members may then carry NEGATIVE amplitudes
(brakes discovered by the fit itself). Off by default; the softplus
path is bit-identical when the flag is False. Run from repo root.
"""
import ast

P = "src/circuit/instrument/learned_mask.py"
s = open(P, encoding="utf-8").read()

anchor = ('OBJECTIVES = ("pos", "contrast", "negctx", "inject", "raise", '
          '"pin", "logit",')
assert s.count(anchor) == 1
s = s.replace(anchor, (
    "# SIGNED AMPLITUDE (2026-08-30): alpha = psi raw instead of\n"
    "# softplus(psi), so the fit can assign NEGATIVE amplitudes --\n"
    "# members that push the seed down (brakes discovered inside the\n"
    "# standard fit). Per-call flag set by run_learned_mask; the patcher\n"
    "# and penalties read it through _amp_of().\n"
    "_SIGNED_AMP = [False]\n\n\n"
    "def _amp_of(psi):\n"
    "    if _SIGNED_AMP[0]:\n"
    "        return psi\n"
    "    return torch.nn.functional.softplus(psi)\n\n\n") + anchor)

old = ("                alpha = torch.nn.functional.softplus(amp).to(\n"
       "                    device=dense.device, dtype=dense.dtype)")
assert s.count(old) == 1
s = s.replace(old, ("                alpha = _amp_of(amp).to(\n"
                    "                    device=dense.device, "
                    "dtype=dense.dtype)"))

old = ('    seed_vector: "Optional[Tuple[torch.Tensor, torch.Tensor]]"'
       ' = None,')
assert s.count(old) == 1
s = s.replace(old, old + "\n    signed_amplitude: bool = False,")

old = '    if objective == "raise" and float(raise_gamma) <= 1.0:'
assert s.count(old) == 1
s = s.replace(old, "    _SIGNED_AMP[0] = bool(signed_amplitude)\n" + old)

old = "        _psi1 = math.log(math.expm1(1.0))"
assert s.count(old) == 1
s = s.replace(old, ("        _psi1 = (1.0 if signed_amplitude\n"
                    "                 else math.log(math.expm1(1.0)))"))

old = ("                    ((1.0 - torch.sigmoid(thetas[s]))\n"
       "                     * (torch.nn.functional.softplus(a) - 1.0)"
       ".abs()).sum()")
assert s.count(old) == 1
s = s.replace(old, ("                    ((1.0 - torch.sigmoid(thetas[s]))\n"
                    "                     * (_amp_of(a) - 1.0).abs()).sum()"))

old = ("                        (torch.sigmoid(thetas[s])\n"
       "                         * (torch.nn.functional.softplus(a) - 1.0)"
       ".abs()).sum()")
assert s.count(old) == 1
s = s.replace(old, ("                        (torch.sigmoid(thetas[s])\n"
                    "                         * (_amp_of(a) - 1.0).abs())"
                    ".sum()"))

old = "                alpha = torch.nn.functional.softplus(a_here)"
assert s.count(old) == 1
s = s.replace(old, "                alpha = _amp_of(a_here)")

open(P, "w", encoding="utf-8", newline="").write(s)
ast.parse(s)
print("signed_amplitude added; softplus path unchanged when False")
