"""079 engine: the production mask engine with a FOURTH training term, the donor-resample ablation.

src is not edited. ``install()`` reads src/circuit/instrument/learned_mask.py, applies the exact text edits in EDITS
(each must match exactly once, or install fails loudly: a changed src never runs silently under an old patch), execs
the result as ``circuit.instrument.learned_mask`` and puts it in sys.modules. ablation_gradient imports
run_learned_mask at call time, so every fit after install() goes through the patched engine; with
``RESAMPLE["weight"] == 0`` the patched engine builds no fourth patcher and adds no term, so it is the production
engine (the edits only add code behind that switch).

The fourth term. A patcher over the SAME thetas and amplitudes as the Z / C / A patchers, whose floor (the value a
fully masked latent takes) is the natural sparse code of a DONOR context, position by position, instead of zero or a
mean: code = m * alpha * dense + (1 - m) * donor_code. Donors are redrawn at every forward of this term (one donor row
per training row) from the target's donor pool (DonorFloors). Same data, anchors, target and normaliser as the other
terms: loss = w_R / mean(target^2) * mse(seed pre-activation at the anchors, natural). A member must now hold the target
up when everything it does not keep is a REAL other context, which is what 073 measures (0.1% of production circuits
pass it).

The silent-position term (added 2026-10-17, after the pilot's all-position trace: RS circuits hit the anchor exactly
but over-fire a few tokens later in the same passage, 66-90% of their squared error at positions where the target is
naturally silent). With ``SILENT["weight"] > 0``, every data term on the training activating contexts (every ablation,
the donor one included) is also charged ``w_S * relu(pre - max(natural pre, 0))^2`` averaged over the positions where
the target's natural post-Top-K read is 0 (anchors excluded). It is added inside data_loss, so the engine scales it
with that term: by (the term's weight) / mean(target^2), exactly like the anchor error. The effective silent weight
is therefore w_S under Z, w_S * gamma under C and A, and w_S * w_R under the donor term. See silent_penalty().

Differences from the 073 resample EVAL (stated, not hidden): training has no Top-K re-imposition (as for the C / A
means: "no Top-K re-imposed in training", protocol v1), so a donor's entries for latents the mask keeps are not
dropped and no budget is enforced; at m = 1 the (1 - m) donor term vanishes for that latent anyway.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "circuit" / "instrument" / "learned_mask.py"
MODNAME = "circuit.instrument.learned_mask"

# (anchor, replacement) pairs; each anchor must occur exactly once in src.
EDITS = [
    # 1. every forward of a patcher that has on_forward() (the donor patcher) draws its donors first
    ("def _forward_preact(inference: Any, patcher: LearnedMaskPatcher,\n"
     "                    tokens: torch.Tensor, grad: bool) -> torch.Tensor:\n",
     "def _forward_preact(inference: Any, patcher: LearnedMaskPatcher,\n"
     "                    tokens: torch.Tensor, grad: bool) -> torch.Tensor:\n"
     "    if hasattr(patcher, \"on_forward\"):                      # 079: donor draw\n"
     "        patcher.on_forward(int(tokens.shape[0]))\n"),
    # 2. build the fourth patcher (same thetas / amps) right after the posctx patcher
    ("        if pos_floor else None)\n",
     "        if pos_floor else None)\n"
     "    # 079: the donor-resample patcher (off unless RESAMPLE['weight'] > 0)\n"
     "    patcher_rs = (RESAMPLE[\"make\"](bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas, amps,\n"
     "                                  code_dtype, binarize, keep_threshold)\n"
     "                  if float(RESAMPLE.get(\"weight\", 0.0)) > 0 else None)\n"),
    # 3. its term, next to the posctx term (pos objective with the dual/triple floor)
    ("                                      patcher_pos))\n",
     "                                      patcher_pos))\n"
     "                    if patcher_rs is not None:                # 079: fourth term\n"
     "                        terms.append((pt_tr, pa_tr, ptgt_tr,\n"
     "                                      float(RESAMPLE[\"weight\"]) / dn, patcher_rs))\n"),
    # 4. its name in the training curve
    ("                            term_names.append(\"pos\")\n",
     "                            term_names.append(\"pos\")\n"
     "                        elif pat_t is not None and getattr(pat_t, \"is_resample\", False):\n"
     "                            term_names.append(\"resample\")\n"),
    # 5. anneal temperature kept in step with the other patchers
    ("                    patcher_pos.temperature = t_now\n",
     "                    patcher_pos.temperature = t_now\n"
     "                if patcher_rs is not None:\n"
     "                    patcher_rs.temperature = t_now\n"),
    # 6. the silent-position term (2026-10-17): on the training activating contexts, every data term (every
    #    ablation, including the donor one) is also charged for lifting the target where it is naturally silent
    ("        loss = ((vals - tgt) ** 2).mean()\n",
     "        loss = ((vals - tgt) ** 2).mean()\n"
     "        if float(SILENT.get(\"weight\", 0.0)) > 0 and tokens is pt_tr:      # 079: silent positions\n"
     "            loss = loss + SILENT[\"fn\"](pre, tk, an)\n"),
]
FOOTER = ("\n\n# 079: switches read at fit time (rs_engine.RESAMPLE / SILENT are these dicts)\n"
          "RESAMPLE = {\"weight\": 0.0, \"make\": None}\n"
          "SILENT = {\"weight\": 0.0, \"fn\": None}\n")

RESAMPLE: Dict[str, Any] = {}
_MOD: Optional[types.ModuleType] = None


def patched_source() -> str:
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("079 rs_engine: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:120]))
        text = text.replace(anchor, repl)
    return text + FOOTER


def install() -> types.ModuleType:
    """Exec the patched engine as circuit.instrument.learned_mask (idempotent). Returns the module."""
    global _MOD, RESAMPLE
    if _MOD is not None:
        return _MOD
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod                     # before exec: dataclasses / typing resolve the module by name
    exec(compile(patched_source(), "<079 patched %s>" % SRC.name, "exec"), mod.__dict__)
    pkg.learned_mask = mod
    RESAMPLE = mod.RESAMPLE
    _MOD = mod
    return mod


# ----------------------------------------------------------------------------------------------- donor floors
class DonorFloors:
    """The donor patcher's floors: {site: dense [B, T, d_sae]} for the CURRENT donor rows. ``codes`` holds every
    donor's natural sparse code per site, {site: (acts [N, T, k], idx [N, T, k])} (CPU or GPU). draw(B) picks B rows
    (with replacement, a seeded generator); get(site) densifies that site for those rows on demand."""

    def __init__(self, codes: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]], d_sae: int,
                 device: Any, seed: int = 0):
        self.codes, self.d_sae, self.device = codes, int(d_sae), device
        self.n = int(next(iter(codes.values()))[0].shape[0]) if codes else 0
        self.gen = torch.Generator().manual_seed(int(seed))
        self.rows: Optional[torch.Tensor] = None
        self.n_draws = 0

    def __bool__(self) -> bool:                   # LearnedMaskPatcher does `floors or {}`
        return True

    def __contains__(self, site) -> bool:
        return site in self.codes

    def keys(self):
        return self.codes.keys()

    def draw(self, B: int) -> None:
        self.rows = torch.randint(self.n, (int(B),), generator=self.gen)
        self.n_draws += 1

    def get(self, site, default=None):
        c = self.codes.get(site)
        if c is None:
            return default
        if self.rows is None:
            raise RuntimeError("DonorFloors.get before draw()")
        acts, idx = c
        r = self.rows.to(acts.device)
        a = acts[r].to(self.device, torch.float32)
        i = idx[r].to(self.device)
        dense = torch.zeros(a.shape[0], a.shape[1], self.d_sae, device=self.device, dtype=torch.float32)
        return dense.scatter_(-1, i, a)


def make_factory(mod: types.ModuleType, floors: DonorFloors):
    """RESAMPLE['make'] for one target: builds the donor patcher (a LearnedMaskPatcher subclass of the PATCHED
    module, so the engine's own transform runs; only the floor lookup and its cache differ)."""

    class DonorPatcher(mod.LearnedMaskPatcher):
        is_resample = True

        def on_forward(self, B: int) -> None:
            self.floors.draw(B)

        def _const(self, tag, site, t, like):
            if tag == "floor":                     # per-forward donor floor: never cached
                return t.to(device=like.device, dtype=like.dtype)
            return super()._const(tag, site, t, like)

    def make(bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas, amps, code_dtype, binarize, keep_threshold):
        return DonorPatcher(bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas=deltas, amps=amps,
                            code_dtype=code_dtype, floors=floors, binarize=binarize, bin_threshold=keep_threshold)

    return make


# ----------------------------------------------------------------------------------------------- silent positions
def silent_penalty(pre: torch.Tensor, silent: torch.Tensor, floor: torch.Tensor, weight: float) -> torch.Tensor:
    """weight * mean over rows of [ mean over the row's silent positions of relu(pre - floor)^2 ].

    Returned UNNORMALISED, in the same units as data_loss's anchor term ((vals - tgt)^2).mean(): the engine multiplies
    the whole data_loss by (term weight) / dual_norm, so the silent term is normalised exactly like the anchor error it
    is added to. (The first version also divided by mean(target^2) itself: normalised twice, i.e. scaled by
    1 / mean(target^2), which made it dominate weak targets ~8x at a_pos 0.35 and vanish on strong ones; caught on the
    2026-10-18 rerun's first target and fixed before any result was used.)

    pre     [B, T] the masked forward's target pre-activation (graph-connected)
    silent  [B, T] bool: positions where the target's natural post-Top-K read is 0, anchors excluded
    floor   [B, T] max(natural pre-activation, 0): only a LIFT above the natural value (and above 0) is charged,
            so a position where the target sits just below its Top-K cut is not pushed further down"""
    excess = torch.relu(pre.float() - floor.to(pre.device, torch.float32)) * silent.to(pre.device, torch.float32)
    per_row = (excess ** 2).sum(-1) / silent.sum(-1).clamp(min=1).to(pre.device, torch.float32)
    return float(weight) * per_row.mean()


def make_silent_fn(inference: Any, bank: Any, site: Tuple[int, str], latent: int, w_seed: torch.Tensor,
                   b_seed: torch.Tensor, weight: float):
    """SILENT['fn'] for one target. The natural per-position reads of each training batch are computed once, from
    the batch's own tokens (so they always match what the engine fed), and cached by token content."""
    from eval.wcm_eval import NaturalTap
    cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    stats = {"calls": 0, "batches": 0}

    def fn(pre: torch.Tensor, tk: torch.Tensor, an: torch.Tensor) -> torch.Tensor:
        key = (int(tk.shape[0]), hash(tuple(tk.flatten().tolist())))
        if key not in cache:
            with torch.no_grad():
                p = NaturalTap(bank, site, w_seed, b_seed, latent)
                inference.forward(tk, patcher=p, grad_enabled=False, return_activations=False, tokenize_final=False)
                nat_pre, nat_tk = p.seed_pre.detach().float(), p.tap_tk.detach().float()
                B = int(nat_pre.shape[0])
                silent = nat_tk <= 0
                a = an[:B].to(silent.device).long().clamp(0, silent.shape[1] - 1)
                silent[torch.arange(B, device=silent.device), a] = False
                cache[key] = (silent, nat_pre.clamp(min=0))
            stats["batches"] += 1
        stats["calls"] += 1
        silent, floor = cache[key]
        return silent_penalty(pre, silent, floor, weight)

    fn.stats = stats
    return fn
