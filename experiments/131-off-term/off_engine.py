"""131 engine: the production WCM v4 mask engine with a FIFTH training term, the off term O.

src is not edited. ``install()`` reads src/circuit/instrument/learned_mask.py, applies the exact text edits in EDITS
(each must match exactly once, or install fails loudly: a changed src never runs silently under an old patch), execs
the result as ``circuit.instrument.learned_mask`` and puts it in sys.modules (079 rs_engine / 095 rs2_engine's
mechanics). ablation_gradient imports run_learned_mask at call time, so every fit after install() goes through the
patched engine. With ``OFF["weight"] == 0`` the patched engine builds no off patcher and adds no term: the edits only
add code behind that switch, so it is the production engine (CPU test (a) checks the circuit and loss curve are
bit-identical to the unpatched module's).

The off term. A patcher over the SAME thetas, amplitudes and gate as the Z / C / A / R patchers, run on the target's
48 TRAINING contrast contexts (the rows the R term uses as donors, eval_reference's C rows = rec["neg"][:48]; never the
16 held-out contrast rows the scorer and the donor test read). Its floor is R's: every latent outside the circuit takes
the natural sparse code of ANOTHER training contrast context at the same position (donor rows redrawn every forward,
never the row itself), under the sparsity-preserving donor fill (donor_keep_mask, = 073 ResamplePatcher's fill at
binary gates); members sit at alpha x their live value on that contrast context. This is the training twin of the
donor test's pool-C silent-firing read (src/eval/donor_test.py: silent bases, members alpha x live, non-members from
another silent contrast context). Penalty at EVERY position of those rows:

    O = w_O / dual_norm * mean_{rows, positions} relu(pre - max(natural pre, 0))^2

pre = the seed's pre-activation under this ablation, natural pre = the full model's on the same row (computed once per
fit, no grad), dual_norm = mean(natural anchor pre^2) of the activating training rows (the engine's shared normaliser
of Z / C / A / R). Only a lift above the natural value (and above 0) is charged: a row where the target sits just
below its Top-K cut is not pushed further down. 079's silent-position term S used the same hinge and normaliser on the
naturally silent positions of the ACTIVATING rows; O moves it onto the contrast contexts themselves.

Rows per forward: the engine's own micro-batch slicing (rows s..s+micro_bs of the 48, s = micro_index x micro_bs mod
48), so O sees as many rows per step as the other terms and cycles through all 48.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "circuit" / "instrument" / "learned_mask.py"
MODNAME = "circuit.instrument.learned_mask"
OFF_SEED = 20261118                    # the off patcher's donor generator (R keeps its own, 20261017)

# (anchor, replacement) pairs; each anchor must occur exactly once in src.
EDITS = [
    # 1. build the off patcher (same thetas / amps / gate, R's donor codes) right before the necessity patcher
    ("    # NECESSITY TERM (2026-11-03): the same thetas, inverted (members removed,\n",
     "    # 131: the off patcher (off unless OFF['weight'] > 0)\n"
     "    patcher_off = (OFF[\"make\"](bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas, amps, code_dtype,\n"
     "                               binarize, keep_threshold, train_fill, sparse_grad, donor_pool, resample_tokens,\n"
     "                               inference, device)\n"
     "                   if float(OFF.get(\"weight\", 0.0)) > 0 else None)\n"
     "    # NECESSITY TERM (2026-11-03): the same thetas, inverted (members removed,\n"),
    # 2. its term, right after the resample term (same normaliser dual_norm)
    ("                        terms += [(pt_tr, pa_tr, ptgt_tr,\n"
     "                                   float(resample_weight) / dn / int(resample_k),\n"
     "                                   patcher_donor)] * int(resample_k)\n",
     "                        terms += [(pt_tr, pa_tr, ptgt_tr,\n"
     "                                   float(resample_weight) / dn / int(resample_k),\n"
     "                                   patcher_donor)] * int(resample_k)\n"
     "                    if patcher_off is not None:                    # 131: the off term (contrast rows)\n"
     "                        terms.append((patcher_off.off_tokens, None, None, float(OFF[\"weight\"]) / dn,\n"
     "                                      patcher_off))\n"),
    # 3. its name in the training curve
    ("                        elif pat_t is not None and pat_t is patcher_nec:\n",
     "                        elif pat_t is not None and getattr(pat_t, \"is_off\", False):\n"
     "                            term_names.append(\"off\")\n"
     "                        elif pat_t is not None and pat_t is patcher_nec:\n"),
    # 4. its loss: all positions of the contrast rows (no anchors, no target, no rank / censor reads)
    ("    def data_loss(tokens, anchors, targets, micro_index, pat=None) -> torch.Tensor:\n"
     "        pat = pat if pat is not None else patcher\n",
     "    def data_loss(tokens, anchors, targets, micro_index, pat=None) -> torch.Tensor:\n"
     "        pat = pat if pat is not None else patcher\n"
     "        if getattr(pat, \"is_off\", False):                     # 131: the off term\n"
     "            return OFF[\"loss\"](pat, tokens, micro_index, micro_bs)\n"),
    # 5. anneal temperature kept in step with the other patchers
    ("                if patcher_donor is not None:\n"
     "                    patcher_donor.temperature = t_now\n",
     "                if patcher_donor is not None:\n"
     "                    patcher_donor.temperature = t_now\n"
     "                if patcher_off is not None:\n"
     "                    patcher_off.temperature = t_now\n"),
]
FOOTER = ("\n\n# 131: switches read at fit time (off_engine.configure sets them)\n"
          "OFF = {\"weight\": 0.0, \"make\": None, \"loss\": None}\n")

_MOD: Optional[types.ModuleType] = None
STATS: Dict[str, Any] = {}


def patched_source() -> str:
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("131 off_engine: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:160]))
        text = text.replace(anchor, repl)
    return text + FOOTER


def install() -> types.ModuleType:
    """Exec the patched engine as circuit.instrument.learned_mask (idempotent). Returns the module."""
    global _MOD
    if _MOD is not None:
        return _MOD
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod                     # before exec: dataclasses / typing resolve the module by name
    exec(compile(patched_source(), "<131 patched %s>" % SRC.name, "exec"), mod.__dict__)
    pkg.learned_mask = mod
    mod.OFF.update(make=_make, loss=_loss)
    _MOD = mod
    return mod


def configure(weight: float, seed: int = OFF_SEED) -> None:
    """Set the off term for the next fits (weight 0 = the production engine) and reset the per-fit stats."""
    mod = install()
    mod.OFF["weight"] = float(weight)
    mod.OFF["seed"] = int(seed)
    STATS.clear()
    STATS.update(weight=float(weight), built=0, calls=0, rows_seen=set(), n_off=None, lift_share_sum=None,
                 excess_sum=None, nat_pos_share=None, self_draws=0, draws=0)


def stats_summary() -> Dict[str, Any]:
    """The last fit's off-term stats as plain values (one host sync, after the fit)."""
    s = dict(STATS)
    calls = max(int(s.get("calls") or 0), 1)
    rows = s.pop("rows_seen", set())
    out = {k: v for k, v in s.items() if k not in ("lift_share_sum", "excess_sum")}
    out["rows_seen"] = len(rows)
    out["max_row"] = max(rows) if rows else None
    out["lift_share_mean"] = (float(s["lift_share_sum"]) / calls) if s.get("lift_share_sum") is not None else None
    out["excess_mean"] = (float(s["excess_sum"]) / calls) if s.get("excess_sum") is not None else None
    return out


# ----------------------------------------------------------------------------------------------- the patcher
def _classes(mod):
    if getattr(mod, "_131_classes", None) is not None:
        return mod._131_classes

    class OffPool(mod.DonorPool):
        """R's donor codes (shared, not recaptured), its own generator; draw(B) never picks a row for itself: base
        row i takes donor (i + 1 + U{0..n-2}) mod n. ``base`` is set by the loss before each forward."""

        def __init__(self, codes, d_sae, device, seed):
            super().__init__(codes, d_sae, device, seed=seed)
            self.base: Optional[torch.Tensor] = None

        def draw(self, B: int) -> None:
            if self.base is None or self.n < 2:
                raise RuntimeError("131 OffPool.draw without base rows (or fewer than 2 contrast rows)")
            base = self.base[:int(B)]
            off = torch.randint(self.n - 1, (int(base.shape[0]),), generator=self.gen)
            self.rows = (base + 1 + off) % self.n
            self.n_draws += 1
            STATS["draws"] = STATS.get("draws", 0) + 1
            STATS["self_draws"] = STATS.get("self_draws", 0) + int((self.rows == base).sum())

    class OffPatcher(mod.LearnedMaskPatcher):
        is_off = True

    mod._131_classes = (OffPool, OffPatcher)
    return mod._131_classes


def _make(bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas, amps, code_dtype, binarize, keep_threshold,
          train_fill, sparse_grad, donor_pool, tokens, inference, device):
    """OFF['make']: the off patcher for one fit. Needs the R term (its donor codes = the 48 contrast-train rows)."""
    mod = _MOD
    if donor_pool is None or tokens is None:
        raise ValueError("131: the off term needs the resample term's donor contexts (resample_weight > 0)")
    OffPool, OffPatcher = _classes(mod)
    toks = tokens.to(device)
    if int(donor_pool.n) != int(toks.shape[0]):
        raise RuntimeError("131: donor codes (%d rows) != off tokens (%d rows)" % (donor_pool.n, toks.shape[0]))
    pool = OffPool(donor_pool.codes, bank.d_sae, device, int(mod.OFF.get("seed", OFF_SEED)))
    p = OffPatcher(bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas=deltas, amps=amps,
                   code_dtype=code_dtype, binarize=binarize, bin_threshold=keep_threshold, tap_seed=True,
                   train_fill=train_fill, sparse_grad=sparse_grad, donors=pool)
    p.off_tokens = toks
    p.off_inference = inference
    nat = mod._natural(inference, bank, toks, seed_layer, seed_kind, w_seed, b_seed, code_dtype=code_dtype).float()
    p.off_floor = nat.clamp(min=0)
    STATS["built"] = STATS.get("built", 0) + 1
    STATS["n_off"] = int(toks.shape[0])
    STATS["nat_pos_share"] = float((nat > 0).float().mean())
    STATS["nat_pre_max"] = float(nat.max())
    STATS["off_tokens_sum"] = int(toks.long().sum())           # identity check against eval_reference (run.py)
    hook = STATS.get("on_make")
    if hook is not None:
        hook(p)
    return p


def _loss(pat, tokens, micro_index, micro_bs) -> torch.Tensor:
    """OFF['loss']: mean over the rows' positions of relu(pre - max(natural pre, 0))^2 (unnormalised: the engine
    multiplies by w_O / dual_norm)."""
    mod = _MOD
    n = int(tokens.shape[0])
    s = (int(micro_index) * int(micro_bs)) % max(n, 1)
    rows = torch.arange(s, min(s + int(micro_bs), n))
    pat.donors.base = rows
    pre = mod._forward_preact(pat.off_inference, pat, tokens[rows.to(tokens.device)], grad=True)
    floor = pat.off_floor[rows.to(pat.off_floor.device)].to(pre.device)
    ex = torch.relu(pre.float() - floor)
    with torch.no_grad():
        STATS["calls"] = STATS.get("calls", 0) + 1
        STATS.setdefault("rows_seen", set()).update(int(r) for r in rows.tolist())
        ls, es = (ex > 0).float().mean(), ex.mean()
        STATS["lift_share_sum"] = ls if STATS.get("lift_share_sum") is None else STATS["lift_share_sum"] + ls
        STATS["excess_sum"] = es if STATS.get("excess_sum") is None else STATS["excess_sum"] + es
    hook = STATS.get("on_loss")
    if hook is not None:
        hook(pat, rows, ex)
    return (ex ** 2).mean()
