"""100 E4 engine: 095's rs2 engine (079 donor term + 092 drop_floor + the v2 sparsity-preserving donor fill) with four
more switches, all default-off. src is not edited: the patched text is exec'd in memory as
circuit.instrument.learned_mask, exactly as 079 / 095 do.

  E4["make2"], E4["w2"]  a SECOND donor-resample term (its own DonorFloors, same thetas / amps; the sparsity-preserving
                         donor fill applies to it too). Used for RA (activating donors) next to RESAMPLE's RC.
  E4["drop_pos"]         drop the A term (the posctx-mean patcher); with 092's SPEED["drop_floor"] (drop C) this gives
                         the all-resample arm Z + RA + RC.
  E4["k"]                k donor draws per training row per step: every donor term is entered k times at weight w / k
                         (each entry is its own forward, so each redraws its donors; 079's on_forward).
  E4["amp_cap"]          cap the amplitude alpha = min(softplus(psi), cap) everywhere _amp_of is read (forward,
                         penalties, the exported amplitudes).
With every E4 switch at its default the patched source differs from rs2's only by code behind these switches;
run.py --check-arm verifies a v2CR refit through this engine against Run A's v2CR circuit (bit-identical expected).
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "095-v2-resample-appendix"))
import rs2_engine  # noqa: E402

MODNAME = rs2_engine.MODNAME
SRC = rs2_engine.SRC

EXTRA = [
    # 1. the second donor patcher, built right after 079's patcher_rs
    ("                  if float(RESAMPLE.get(\"weight\", 0.0)) > 0 else None)\n",
     "                  if float(RESAMPLE.get(\"weight\", 0.0)) > 0 else None)\n"
     "    patcher_rs2 = (E4[\"make2\"](bank, thetas, seed_layer, seed_kind, w_seed, b_seed, deltas, amps,\n"
     "                               code_dtype, binarize, keep_threshold)\n"
     "                   if E4.get(\"make2\") is not None and float(E4.get(\"w2\", 0.0)) > 0 else None)  # 100 E4\n"),
    # 2. its anneal temperature
    ("                if patcher_rs is not None:\n"
     "                    patcher_rs.temperature = t_now\n",
     "                if patcher_rs is not None:\n"
     "                    patcher_rs.temperature = t_now\n"
     "                if patcher_rs2 is not None:                          # 100 E4\n"
     "                    patcher_rs2.temperature = t_now\n"),
    # 3. term edits, right after 092's drop_floor filter
    ("                    terms = [t_ for t_ in terms if t_[4] is not patcher]\n",
     "                    terms = [t_ for t_ in terms if t_[4] is not patcher]\n"
     "                if E4[\"drop_pos\"]:                                   # 100 E4: no A (posctx mean) term\n"
     "                    terms = [t_ for t_ in terms if t_[4] is not patcher_pos]\n"
     "                if patcher_rs is not None and int(E4[\"k\"]) > 1:      # 100 E4: k donor draws per row per step\n"
     "                    _rs = [t_ for t_ in terms if t_[4] is patcher_rs]\n"
     "                    terms = [t_ for t_ in terms if t_[4] is not patcher_rs]\n"
     "                    for t_ in _rs:\n"
     "                        terms += [t_[:3] + (t_[3] / int(E4[\"k\"]), t_[4])] * int(E4[\"k\"])\n"
     "                if patcher_rs2 is not None:                          # 100 E4: the second donor term (RA)\n"
     "                    terms += [(pt_tr, pa_tr, ptgt_tr, float(E4[\"w2\"]) / dn / int(E4[\"k\"]), patcher_rs2)] * int(E4[\"k\"])\n"),
    # 4. amplitude cap
    ("    return torch.nn.functional.softplus(psi)\n",
     "    if E4.get(\"amp_cap\"):                                       # 100 E4: bounded gains\n"
     "        return torch.nn.functional.softplus(psi).clamp(max=float(E4[\"amp_cap\"]))\n"
     "    return torch.nn.functional.softplus(psi)\n"),
]
FOOTER = '''

# ---------------------------------------------------------------------------------------------- 100 E4 additions
E4 = {"drop_pos": False, "k": 1, "make2": None, "w2": 0.0, "amp_cap": None}
'''


def patched_source():
    text = rs2_engine.patched_source()
    for anchor, repl in EXTRA:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("100 e4: anchor found %d times (need 1):\n%r" % (n, anchor[:200]))
        text = text.replace(anchor, repl)
    return text + FOOTER


def install(tag="100e4"):
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(patched_source(), "<%s patched %s>" % (tag, SRC.name), "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def reset(mod):
    rs2_engine.reset(mod)
    mod.E4.update(drop_pos=False, k=1, make2=None, w2=0.0, amp_cap=None)
