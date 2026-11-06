"""095: exp 092's test-3 engine (Z + A + R_C, R under the sparsity-preserving donor fill, no C term), ported onto the
CURRENT src engine, where protocol v2 is native (src/circuit/instrument/learned_mask.py train_fill="sparse",
sparse_grad="ste"; the driver's WCM_VERSION=v2 also sets train_means_source="eval_ref").

Why a port and not 092's sx_engine as-is. sx_engine builds on 090's sp_engine, whose edit 1 anchors the PRE-v2 floor
branch of LearnedMaskPatcher.transform. The v2 port moved that branch under `else:` (one indent deeper), so the anchor
no longer exists and sx_engine.source_v2r() fails loudly on today's src. 092's v2 was also 090's sp_ste with the
engine's own A / C means, while production v2 (and 093's v2 arm, which this experiment pairs against) trains against
the scorer's means (eval_ref). Running the 092 patch on the old src would therefore not be "v2 with R in place of C".

What is reused verbatim (no re-typing):
  079 rs_engine.EDITS + FOOTER   the donor-resample patcher, its term (weight / dual_norm), name, anneal temperature
  092 sx_engine.EDITS[3]         drop_floor: the C term (patcher = the contrast-mean floor) is removed from the terms
  092 sp_keep_mask_donor         the donor keep mask (source text cut out of sx_engine.FOOTER and exec'd here)
Re-derived (one edit, the only one that touches the new src): the donor patcher's sparsity-preserving fill now hooks
the native v2 branch. 092's _sp_delta_donor called sp_code(code, H, SPFILL["mode"]="ste") with K = bank.k; here it
calls the native sparse_code(code, H, "ste") with K = bank.k (sparse_code's "ste" is sp_code's "ste" verbatim). The
donor patcher itself is built by 079's factory with train_fill left at "dense", so without SPEED["sp_donor"] it is
091's raw R; the A patcher (patcher_pos) keeps the native v2 fill and the eval_ref A mean.

Switches (all off = the native src engine with a dormant R term: CHECK=1 in run.py refits one 093 v2 target through
this module and compares it with 093's circuit):
  SPEED["drop_floor"]  no C term          SPEED["sp_donor"]  R under the donor sparsity-preserving fill
  RESAMPLE["weight"] / ["make"]           079's donor term (set by 091 run.setup_terms)
"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent
SRC = REPO / "src" / "circuit" / "instrument" / "learned_mask.py"
MODNAME = "circuit.instrument.learned_mask"
sys.path.insert(0, str(EXP / "079-resample-training"))
sys.path.insert(0, str(EXP / "092-v2-speedups"))
import rs_engine  # noqa: E402
import sx_engine  # noqa: E402  (imports 090 sp_engine; nothing is installed at import)

assert rs_engine.SRC == SRC, (rs_engine.SRC, SRC)

DROP_FLOOR_EDIT = sx_engine.EDITS[3]
assert "drop_floor" in DROP_FLOOR_EDIT[1], "092 edit 4 (drop_floor) moved"

DONOR_OLD = ("                if floor is not None and self.train_fill == \"sparse\":\n"
             "                    delta_code = self._sparse_delta((layer_idx, kind), dense, m, alpha, floor)\n")
DONOR_NEW = ("                if floor is not None and getattr(self, \"is_resample\", False) and SPEED[\"sp_donor\"]:\n"
             "                    delta_code = _sp_delta_donor(self, (layer_idx, kind), dense, m, alpha, floor)  # 092 t3\n"
             "                elif floor is not None and self.train_fill == \"sparse\":\n"
             "                    delta_code = self._sparse_delta((layer_idx, kind), dense, m, alpha, floor)\n")

EDITS = list(rs_engine.EDITS) + [DROP_FLOOR_EDIT, (DONOR_OLD, DONOR_NEW)]


def _donor_mask_source():
    """092's sp_keep_mask_donor, as text, from sx_engine.FOOTER (so the exact 092 function runs)."""
    f = sx_engine.FOOTER
    a = f.index("def sp_keep_mask_donor(")
    b = f.index("def _sp_delta_donor(")
    return f[a:b].rstrip() + "\n"


FOOTER = '''

# ---------------------------------------------------------------------------------------------- 095 additions
SPEED = {"drop_floor": False, "sp_donor": False}
DONOR_STATS = {"calls": 0}


''' + _donor_mask_source() + '''

def _sp_delta_donor(self, site, dense, m, alpha, floor):
    """092 _sp_delta_donor on the native v2 helpers: the donor code at binary gates is members at alpha x live, then
    the donor's non-member active entries, largest first, up to K - #active members; forward hard, backward
    straight-through (sparse_code "ste" == 092's sp_code "ste")."""
    K = int(self.bank.k)
    f = self._const("floor", site, floor, dense)
    with torch.no_grad():
        member = (m.detach() > self.bin_threshold).to(dense.device)
        H = sp_keep_mask_donor(dense, member, alpha.detach(), f, K)
        DONOR_STATS["calls"] += 1
    code = m * alpha * dense + (1.0 - m) * f
    return sparse_code(code, H, "ste") - dense
'''


def patched_source():
    text = SRC.read_text(encoding="utf-8")
    for anchor, repl in EDITS:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("095: anchor found %d times (need 1) in %s:\n%r" % (n, SRC, anchor[:200]))
        text = text.replace(anchor, repl)
    return text + rs_engine.FOOTER + FOOTER


def install(tag="095"):
    """Exec the patched engine as circuit.instrument.learned_mask (a fresh module each call)."""
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(patched_source(), "<%s patched %s>" % (tag, SRC.name), "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def reset(mod):
    mod.SPEED.update(drop_floor=False, sp_donor=False)
    mod.DONOR_STATS["calls"] = 0
    mod.RESAMPLE.update(weight=0.0, make=None)
    mod.SILENT.update(weight=0.0, fn=None)
