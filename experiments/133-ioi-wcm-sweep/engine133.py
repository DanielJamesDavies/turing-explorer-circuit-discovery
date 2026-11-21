"""133 engine: the 081 v4 mask engine (src learned_mask, run_learned_mask with objective "logit") with two in-memory
additions, both behind switches that are OFF by default (then the engine is src's, unchanged):

1. PAIRED R donors (PAIR["on"]). src's DonorPool.draw picks donor rows at random from the donor pool; 081 v4 used
   the 1,600 training items' ABC prompts in pool order, so a row's donor was another item's ABC prompt, often of a
   different length (38.7% shorter -> pad-position codes fill the row's tail). Paired: the donor tokens are the
   PRESENTED rows' own ABC prompts (same template, same length; 133 run.py passes them aligned with pos_tokens), and
   draw(B) takes rows s..s+B-1, s = the micro-batch start that data_loss computes (recorded in PAIR["s"]). So every
   row's non-members are filled with its own counterfactual's codes (causal-scrubbing style resample).

2. The TASK OFF TERM (OFF["weight"] > 0). A fifth patcher over the same thetas / amplitudes / gates, in the Z frame
   (non-members 0, error kept, as patcher_zero), run on the presented rows' own ABC prompts, scored for the CLEAN
   item's answers (IO vs S):
       hinge:  O = w / dn * mean relu(D_circ(cf) - D_full(cf))^2
       sq:     O = w / dn * mean (D_circ(cf) - D_full(cf))^2          (a weighted W_CF-like match on all rows)
   D = logit(IO) - logit(S) at the anchor, D_full(cf) = the full model's (no grad, OFF["target"]), dn = the engine's
   shared data-term normaliser (logit_norm "gap"). The hinge charges only the circuit producing MORE of the clean
   answer on the counterfactual than the model does (the task analogue of 131's latent off term).

src is not edited: install() applies exact text edits (each anchor must match once) and execs the result as
circuit.instrument.learned_mask (131 off_engine's mechanics).
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

EDITS = [
    # 1. build the off patcher (Z frame) right before the necessity patcher
    ("    # NECESSITY TERM (2026-11-03): the same thetas, inverted (members removed,\n",
     "    # 133: the task off patcher (off unless OFF133['weight'] > 0), Z frame\n"
     "    patcher_off133 = None\n"
     "    if float(OFF133.get(\"weight\", 0.0)) > 0:\n"
     "        patcher_off133 = LearnedMaskPatcher(bank, thetas, seed_layer, seed_kind, w_seed, b_seed,\n"
     "                                            deltas=deltas, amps=amps, code_dtype=code_dtype, floors=None,\n"
     "                                            binarize=binarize, bin_threshold=keep_threshold, tap_seed=_tap)\n"
     "        patcher_off133.is_off133 = True\n"
     "    # NECESSITY TERM (2026-11-03): the same thetas, inverted (members removed,\n"),
    # 2. its term, right after the resample term
    ("                        terms += [(pt_tr, pa_tr, ptgt_tr,\n"
     "                                   float(resample_weight) / dn / int(resample_k),\n"
     "                                   patcher_donor)] * int(resample_k)\n",
     "                        terms += [(pt_tr, pa_tr, ptgt_tr,\n"
     "                                   float(resample_weight) / dn / int(resample_k),\n"
     "                                   patcher_donor)] * int(resample_k)\n"
     "                    if patcher_off133 is not None:                 # 133: the task off term\n"
     "                        terms.append((pt_tr, pa_tr, ptgt_tr, float(OFF133[\"weight\"]) / dn, patcher_off133))\n"),
    # 3. its name in the training curve
    ("                        elif pat_t is not None and pat_t is patcher_nec:\n",
     "                        elif pat_t is not None and getattr(pat_t, \"is_off133\", False):\n"
     "                            term_names.append(\"off\")\n"
     "                        elif pat_t is not None and pat_t is patcher_nec:\n"),
    # 4. data_loss: record the micro-batch start (paired donors); the off term's loss
    ("    def data_loss(tokens, anchors, targets, micro_index, pat=None) -> torch.Tensor:\n"
     "        pat = pat if pat is not None else patcher\n",
     "    def data_loss(tokens, anchors, targets, micro_index, pat=None) -> torch.Tensor:\n"
     "        pat = pat if pat is not None else patcher\n"
     "        PAIR133[\"s\"] = (micro_index * micro_bs) % max(int(tokens.shape[0]), 1)   # 133: paired donors\n"
     "        if getattr(pat, \"is_off133\", False):                  # 133: the task off term\n"
     "            return OFF133[\"loss\"](inference, pat, micro_index, micro_bs)\n"),
    # 6. keep the final gates / amplitudes for the soft-vs-hard diagnostic (soft_eval)
    ("    # ---- selection ----------------------------------------------------------\n",
     "    LAST133.update(thetas=thetas, amps=amps, w_seed=w_seed, b_seed=b_seed, seed_layer=seed_layer,\n"
     "                   seed_kind=seed_kind, code_dtype=code_dtype, keep_threshold=keep_threshold,\n"
     "                   binarize=binarize, temperature=float(getattr(patcher, 'temperature', 1.0)),\n"
     "                   tap=_tap)\n"
     "    # ---- selection ----------------------------------------------------------\n"),
    # 5. anneal temperature kept in step
    ("                if patcher_donor is not None:\n"
     "                    patcher_donor.temperature = t_now\n",
     "                if patcher_donor is not None:\n"
     "                    patcher_donor.temperature = t_now\n"
     "                if patcher_off133 is not None:\n"
     "                    patcher_off133.temperature = t_now\n"),
]
FOOTER = ("\n\n# 133: switches read at fit time (engine133.configure sets them)\n"
          "OFF133 = {\"weight\": 0.0, \"loss\": None}\n"
          "PAIR133 = {\"on\": False, \"s\": 0}\n"
          "LAST133 = {}\n")


def soft_eval(inference, bank, tokens, anchors, good, bad, bs: int = 64) -> Dict[str, float]:
    """After a fit: the circuit's mean D in the Z frame (non-members 0, error live) on the given rows, computed by the
    TRAINING forward (LearnedMaskPatcher) with (a) the final soft gates at the final temperature, as the last training
    steps saw them, and (b) the same gates hard-thresholded at keep_threshold (= the exported circuit). (b) vs the
    scorer's Z checks the two forward paths agree; (a) vs (b) measures what sub-threshold gates carried in training."""
    mod = _MOD
    L = mod.LAST133
    if not L.get("thetas"):
        return {}
    out = {}
    with torch.no_grad():
        hard = {s: torch.where(torch.sigmoid(t) > L["keep_threshold"], torch.full_like(t, 50.0),
                               torch.full_like(t, -50.0)) for s, t in L["thetas"].items()}
        for name, th, temp in (("soft", L["thetas"], L["temperature"]), ("hard", hard, 1.0)):
            p = mod.LearnedMaskPatcher(bank, th, L["seed_layer"], L["seed_kind"], L["w_seed"], L["b_seed"],
                                       amps=L["amps"], code_dtype=L["code_dtype"], floors=None,
                                       binarize=L["binarize"], bin_threshold=L["keep_threshold"], tap_seed=L["tap"])
            p.temperature = temp
            vals = []
            for s0 in range(0, int(tokens.shape[0]), bs):
                lg = mod._forward_logits(inference, p, tokens[s0:s0 + bs], grad=False)
                vals.append(mod._target_logprob(lg, anchors[s0:s0 + bs], good[s0:s0 + bs], bad[s0:s0 + bs]).float().cpu())
            out[name] = float(torch.cat(vals).mean())
        sig = {s: torch.sigmoid(t / max(L["temperature"], 1e-4)) for s, t in L["thetas"].items()}
        keep = {s: torch.sigmoid(t) > L["keep_threshold"] for s, t in L["thetas"].items()}
        out["soft_mass_nonmembers"] = float(sum(float(g[~keep[s]].sum()) for s, g in sig.items()))
        out["soft_mass_members"] = float(sum(float(g[keep[s]].sum()) for s, g in sig.items()))
        out["temperature"] = L["temperature"]
    L.clear()
    return out

_MOD: Optional[types.ModuleType] = None
STATS: Dict[str, Any] = {}


def patched_source() -> str:
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("133 engine: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:160]))
        text = text.replace(anchor, repl)
    return text + FOOTER


def install() -> types.ModuleType:
    global _MOD
    if _MOD is not None:
        return _MOD
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(patched_source(), "<133 patched %s>" % SRC.name, "exec"), mod.__dict__)
    pkg.learned_mask = mod
    mod.OFF133.update(loss=_off_loss)
    orig_draw = mod.DonorPool.draw

    def draw(self, B: int) -> None:
        if not mod.PAIR133["on"]:
            return orig_draw(self, B)
        s = int(mod.PAIR133["s"])
        self.rows = (s + torch.arange(int(B))) % self.n
        self.n_draws += 1
        STATS["paired_draws"] = STATS.get("paired_draws", 0) + 1

    mod.DonorPool.draw = draw
    _MOD = mod
    return mod


def configure(pair: bool = False, off_weight: float = 0.0, off_mode: str = "hinge", off_data: Dict = None) -> None:
    """Switches for the next fit. off_data: tokens [N, T], anchors [N], good [N], bad [N], target [N] (the full
    model's D on those rows), aligned with the presented rows (pos_tokens)."""
    mod = install()
    assert off_mode in ("hinge", "sq")
    mod.PAIR133["on"] = bool(pair)
    mod.PAIR133["s"] = 0
    mod.OFF133["weight"] = float(off_weight)
    mod.OFF133["mode"] = off_mode
    mod.OFF133["data"] = off_data
    STATS.clear()
    STATS.update(pair=bool(pair), off_weight=float(off_weight), off_mode=off_mode, off_calls=0, off_rows=set(),
                 off_excess_sum=0.0, off_pos_share_sum=0.0, paired_draws=0)


def _off_loss(inference, pat, micro_index: int, micro_bs: int) -> torch.Tensor:
    mod = _MOD
    d = mod.OFF133["data"]
    n = int(d["tokens"].shape[0])
    s = (int(micro_index) * int(micro_bs)) % n
    sl = slice(s, s + int(micro_bs))
    tk = d["tokens"][sl]
    lg = mod._forward_logits(inference, pat, tk, grad=True)
    v = mod._target_logprob(lg, d["anchors"][sl], d["good"][sl], d["bad"][sl])
    tg = d["target"][sl].to(v.device, v.dtype)
    diff = v - tg
    with torch.no_grad():
        STATS["off_calls"] += 1
        STATS["off_rows"].update(range(s, min(n, s + int(micro_bs))))
        STATS["off_excess_sum"] += float(diff.mean())
        STATS["off_pos_share_sum"] += float((diff > 0).float().mean())
    if mod.OFF133["mode"] == "hinge":
        return (torch.relu(diff) ** 2).mean()
    return (diff ** 2).mean()


def stats_summary() -> Dict[str, Any]:
    s = dict(STATS)
    rows = s.pop("off_rows", set())
    c = max(int(s.get("off_calls") or 0), 1)
    s["off_rows_seen"] = len(rows)
    s["off_excess_mean"] = s.pop("off_excess_sum", 0.0) / c
    s["off_pos_share_mean"] = s.pop("off_pos_share_sum", 0.0) / c
    return s
