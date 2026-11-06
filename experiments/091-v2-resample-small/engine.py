"""091: one in-memory patch of src/circuit/instrument/learned_mask.py carrying BOTH experiment-local patches:

  090 sp_engine   the sparsity-preserving A / C training fill (SPFILL; "ste" = v2)
  079 rs_engine   the donor-resample fourth term (RESAMPLE) and the silent-position term (SILENT)

src is not edited. The anchors of the two patches do not overlap (090: the floor branch of LearnedMaskPatcher.transform
and the training-loop header; 079: _forward_preact, the patcher build, the term list, the term names, the anneal
temperature and data_loss), so both EDITS lists apply in sequence, each anchor still matching exactly once.

ONE re-derived edit. The donor patcher (079's DonorPatcher) is a LearnedMaskPatcher WITH floors (its floor is the
donor's dense code [B, T, d_sae], not a [d_sae] mean), so 090's edit 1 alone would also route the donor term through
the sparsity-preserving fill (and sp_keep_mask's rank assumes a [d] mean vector). The brief asks for R exactly as in
079 (no Top-K re-imposition on the donor term: 079 rs_engine docstring), so the combined edit 1 skips patchers with
is_resample = True. For every other patcher it is 090's edit verbatim. Equivalence checks (run.py CHECK=1):
  w_R = w_S = 0, SPFILL ste      == 090 sp_engine ste      (same circuit, same training curve)
  SPFILL dense, w_R = w_S = 0.25 == 079 rs_engine RS0.25_S0.25
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "090-sparse-fill-training"))
sys.path.insert(0, str(EXP / "079-resample-training"))
import sp_engine  # noqa: E402
import rs_engine  # noqa: E402

SRC = sp_engine.SRC
MODNAME = sp_engine.MODNAME
assert rs_engine.SRC == SRC

SP_EDIT1_OLD = sp_engine.EDITS[0]
SP_EDIT1 = (SP_EDIT1_OLD[0], SP_EDIT1_OLD[1].replace(
    "                    if SPFILL[\"mode\"] != \"dense\":\n",
    "                    if SPFILL[\"mode\"] != \"dense\" and not getattr(self, \"is_resample\", False):  # 091\n"))
assert SP_EDIT1[1] != SP_EDIT1_OLD[1], "091: 090 edit-1 replacement text changed"
EDITS = [SP_EDIT1] + list(sp_engine.EDITS[1:]) + list(rs_engine.EDITS)


def patched_source():
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("091: anchor found %d times (need 1):\n%r" % (n, anchor[:200]))
        text = text.replace(anchor, repl)
    return text + sp_engine.FOOTER + rs_engine.FOOTER


def install(source=None, tag="091"):
    """Exec a patched engine as circuit.instrument.learned_mask (always a fresh module). source: None = the combined
    patch; or sp_engine.patched_source() / rs_engine.patched_source() for the equivalence checks."""
    text = patched_source() if source is None else source
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(text, "<%s patched %s>" % (tag, SRC.name), "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def reset(mod):
    if hasattr(mod, "SPFILL"):
        sp_engine.set_mode(mod, "dense")
    if hasattr(mod, "RESAMPLE"):
        mod.RESAMPLE.update(weight=0.0, make=None)
        mod.SILENT.update(weight=0.0, fn=None)
