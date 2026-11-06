"""090: the production mask engine with an optional SPARSITY-PRESERVING fill for its A and C training terms.

src is NOT edited. As in 079's rs_engine / 089's run.py, src/circuit/instrument/learned_mask.py is read, exact text
edits are applied in memory, and the result is exec'd as circuit.instrument.learned_mask (driver.Runner's method
builder imports it from there).

THE FILL. The pass rule scores A and C with CircuitOnlyPatcher(respect_topk=True, topk=K)._respect_topk_fill
(src/eval/ablation_faithfulness.py):
  members      at alpha x their live value (ALL kept, even when more than K are active: members are never evicted)
  budget       K - (number of members with a nonzero value at that position), clamped at 0, per (batch, position)
  fill         the top-`budget` NON-members ranked by their mean (the ablation value), at that mean
  everything else exactly 0.
(So it is not "Top-K of members + means": the K-cut applies only to the mean fill. The brief described a Top-K over
the whole code; the scorer's actual rule is reproduced here, since the brief also required an exact match to it.)

Training analogue, at each A / C patcher site (the engine's patcher / patcher_pos: the ones built with floors; the Z
patcher has floors=None and is untouched):
  code   = m * alpha * dense + (1 - m) * floor          (the engine's soft code, as in production)
  member = m > bin_threshold                            (the selection rule; under anneal == theta > 0)
  H      = member | (non-member AND rank-by-floor < budget), budget from the members' nonzero live values
  hard   code_sp = H * code                             (gradients only through kept entries)
  ste    code_sp = code + (H * code - code).detach()    (forward hard, backward as if nothing were dropped)
and the engine's delta (code_sp - dense) is decoded as usual. With binary gates (m in {0, 1}) code_sp equals the
scorer's patched code exactly (test_sp_engine.py). The rank uses the scorer's exact recipe: the ablation value cast
to the code dtype, then to float32, members set to -inf, torch.argsort(descending=True).

SPFILL["mode"]: "dense" (production, bit-identical forward), "hard", "ste", "curriculum" (dense for step <
SPFILL["switch"] (0-based, 200 = steps 1-200), then hard). SPFILL["step"] is set by the training loop (edit 2).
"""
import sys
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "src" / "circuit" / "instrument" / "learned_mask.py"
MODNAME = "circuit.instrument.learned_mask"
MODES = ("dense", "hard", "ste", "curriculum")

EDITS = [
    # 1. the free-amplitude branch with a mean floor (A and C patchers): re-impose the scorer's sparsity-preserving fill
    ("                if floor is not None:\n"
     "                    delta_code = delta_code + (1.0 - m) * self._const(\n"
     "                        \"floor\", (layer_idx, kind), floor, dense)\n",
     "                if floor is not None:\n"
     "                    delta_code = delta_code + (1.0 - m) * self._const(\n"
     "                        \"floor\", (layer_idx, kind), floor, dense)\n"
     "                    if SPFILL[\"mode\"] != \"dense\":\n"
     "                        delta_code = _sp_delta(self, (layer_idx, kind), dense, delta_code, m, alpha, floor)\n"),
    # 2. the training loop publishes its step (curriculum)
    ("        for step in range(int(steps)):\n",
     "        for step in range(int(steps)):\n"
     "            SPFILL[\"step\"] = step\n"),
]

FOOTER = '''

# ---------------------------------------------------------------------------------------------- 090 additions
SPFILL = {"mode": "dense", "step": 0, "switch": 200, "k": None}


def sp_keep_mask(dense, member, alpha, floor_c, K):
    """H [B, T, d_sae] bool: the entries the scorer's sparsity-preserving fill keeps. dense [B, T, d] the live code,
    member [d] bool, alpha [d] (any dtype; a member counts as active where alpha * dense != 0 in the code dtype),
    floor_c [d] the ablation value already in the code dtype (the scorer's mean_vector after its .to(dtype))."""
    D = int(dense.shape[-1])
    mean_rank = floor_c.clone().to(torch.float32)
    mean_rank[member] = float("-inf")
    ranked = torch.argsort(mean_rank, descending=True)
    rank_of = torch.empty_like(ranked)
    rank_of[ranked] = torch.arange(D, device=ranked.device)
    a_c = alpha.to(device=dense.device, dtype=dense.dtype)
    act = ((dense * a_c) != 0) & member.view(1, 1, D)
    budget = (int(K) - act.sum(dim=-1)).clamp(min=0)
    fill = (rank_of.view(1, 1, D) < budget.unsqueeze(-1)) & ~member.view(1, 1, D)
    return fill | member.view(1, 1, D)


def sp_code(code, H, mode):
    """The sparsity-preserving code from the soft code and the keep mask: hard = H * code (gradients only through kept
    entries); ste = forward H * code, backward identity."""
    Hc = H.to(code.dtype)
    if mode == "hard":
        return code * Hc
    if mode == "ste":
        return code + (code * Hc - code).detach()
    raise ValueError("sp_code mode must be hard or ste, got %r" % (mode,))


def _sp_delta(self, site, dense, delta_code, m, alpha, floor):
    mode = SPFILL["mode"]
    if mode == "curriculum":
        if int(SPFILL["step"]) < int(SPFILL["switch"]):
            return delta_code
        mode = "hard"
    K = int(SPFILL["k"] if SPFILL["k"] is not None else self.bank.k)
    f = self._const("floor", site, floor, dense)
    with torch.no_grad():
        member = (m.detach() > self.bin_threshold).to(dense.device)
        H = sp_keep_mask(dense, member, alpha.detach(), f, K)
        st = SPFILL.setdefault("stats", {})
        st["calls"] = st.get("calls", 0) + 1
    # the code itself (algebraically delta_code + dense), built directly so a bf16 stream rounds once, as the scorer
    # does: a member lands on alpha x dense and a filled non-member on its mean, exactly, at binary gates
    code = m * alpha * dense + (1.0 - m) * f
    return sp_code(code, H, mode) - dense
'''


def patched_source():
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("090: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:200]))
        text = text.replace(anchor, repl)
    return text + FOOTER


def install():
    """Exec the patched engine as circuit.instrument.learned_mask (idempotent: re-installs a fresh module)."""
    text = patched_source()
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(text, "<090 patched %s>" % SRC.name, "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def set_mode(mod, mode, switch=200, k=None):
    if mode not in MODES:
        raise ValueError("mode must be one of %s, got %r" % (MODES, mode))
    mod.SPFILL.update(mode=mode, switch=int(switch), k=k, step=0, stats={})
