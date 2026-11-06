"""102 engine: 100's e4 engine (= 095's rs2 engine + four default-off E4 switches) exec'd on a FROZEN copy of
src/circuit/instrument/learned_mask.py, plus one default-off 102 switch (warm-started gains, for the two-stage arm).

Why a frozen base text. 095 / 100 exec'd their patches on the LIVE src/circuit/instrument/learned_mask.py. On
2026-10-31 that file is being edited by another agent (r-port, experiment 101: the native R port), so the live text
no longer equals the one 095 / 100 ran (sha256 b9a6142b... in 100's source manifests). 099's source snapshot
(experiments/099-recursive-causal-scrubbing/out/panel18/source_snapshot/src/circuit/instrument/learned_mask.py) has
exactly that hash, so this module reads it instead of src. Nothing in 095 / 100 / src is edited or re-typed: the
edits and footers are 079 rs_engine's, 092 sx_engine's (via 095 rs2_engine.EDITS / FOOTER) and 100 e4_engine's
EXTRA / FOOTER, applied in the same order as rs2_engine.patched_source() then e4_engine.patched_source().

102 switch (default off; with it off the patched text differs from e4's only by code behind the switch):
  P102["amp_init"]  {site: {latent: alpha}}: the free amplitudes start at softplus^-1(alpha) for these latents instead
                    of softplus^-1(1) (two-stage arm: the base circuit's gains, membership fixed by the support
                    restriction).
"""
import hashlib
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "100-scrub-frontier"))
import e4_engine  # noqa: E402  (imports 095 rs2_engine -> 079 rs_engine, 092 sx_engine; nothing installed)

rs2_engine = e4_engine.rs2_engine
rs_engine = rs2_engine.rs_engine
MODNAME = rs2_engine.MODNAME
SRC = rs2_engine.SRC
FROZEN = (EXP / "099-recursive-causal-scrubbing" / "out" / "panel18" / "source_snapshot" / "src" / "circuit"
          / "instrument" / "learned_mask.py")
FROZEN_SHA256 = "b9a6142b4bfb1f248fc6135579b981c4bc018706d08383837a20c1dd55e68221"   # 095 / 100's learned_mask.py

AMP_OLD = "    params = list(thetas.values()) + list(deltas.values()) + list(amps.values())\n"
AMP_NEW = ("    if amps and P102.get(\"amp_init\"):                            # 102 two-stage: warm-started gains\n"
           "        with torch.no_grad():\n"
           "            for _st102, _vals102 in P102[\"amp_init\"].items():\n"
           "                if _st102 not in amps:\n"
           "                    continue\n"
           "                for _ix102, _v102 in _vals102.items():\n"
           "                    _v102 = float(_v102)\n"
           "                    if not signed_amplitude:\n"
           "                        _v102 = _v102 if _v102 > 20 else math.log(math.expm1(max(_v102, 1e-6)))\n"
           "                    amps[_st102][int(_ix102)] = _v102\n"
           "                    P102[\"amp_init_n\"] += 1\n"
           + AMP_OLD)
EXTRA = [(AMP_OLD, AMP_NEW)]
FOOTER = '''

# ---------------------------------------------------------------------------------------------- 102 additions
P102 = {"amp_init": None, "amp_init_n": 0}
'''


def frozen_text():
    raw = FROZEN.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != FROZEN_SHA256:
        raise RuntimeError("102: frozen learned_mask.py hash %s != %s" % (h, FROZEN_SHA256))
    return raw.decode("utf-8")


def _apply(text, edits, tag):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError("102 (%s): anchor found %d times (need 1):\n%r" % (tag, n, anchor[:200]))
        text = text.replace(anchor, repl)
    return text


def rs2_source(text):
    """095 rs2_engine.patched_source() applied to `text` instead of the live src file."""
    return _apply(text, rs2_engine.EDITS, "rs2") + rs_engine.FOOTER + rs2_engine.FOOTER


def e4_source(text):
    """100 e4_engine.patched_source() on top of rs2_source(text)."""
    return _apply(rs2_source(text), e4_engine.EXTRA, "e4") + e4_engine.FOOTER


def patched_source():
    return _apply(e4_source(frozen_text()), EXTRA, "102") + FOOTER


def install(tag="102"):
    import circuit.instrument as pkg
    mod = types.ModuleType(MODNAME)
    mod.__file__ = str(SRC)
    mod.__package__ = "circuit.instrument"
    sys.modules[MODNAME] = mod
    exec(compile(patched_source(), "<%s patched frozen %s>" % (tag, FROZEN.name), "exec"), mod.__dict__)
    pkg.learned_mask = mod
    return mod


def reset(mod):
    e4_engine.reset(mod)
    mod.P102.update(amp_init=None, amp_init_n=0)
