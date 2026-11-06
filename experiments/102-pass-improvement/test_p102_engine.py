"""CPU checks for p102_engine (no model, no GPU).

1. The frozen text has 095 / 100's hash.
2. p102's e4_source(frozen) == e4_engine.patched_source() run with rs2_engine.SRC pointed (in memory) at the frozen
   file: the layering re-uses 095 / 100's edits in their order, nothing re-typed.
3. The 102 text differs from e4's only by the amp_init block and the P102 footer.
4. The patched module compiles and execs; P102 defaults off.

  PYTHONPATH=src python experiments/102-pass-improvement/test_p102_engine.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p102_engine as P  # noqa: E402


def main():
    txt = P.frozen_text()
    mine = P.e4_source(txt)
    live_src = P.rs2_engine.SRC
    try:
        P.rs2_engine.SRC = P.FROZEN
        ref = P.e4_engine.patched_source()
    finally:
        P.rs2_engine.SRC = live_src
    assert mine == ref, "e4_source(frozen) != e4_engine.patched_source() on the frozen file"
    full = P.patched_source()
    assert full.replace(P.AMP_NEW, P.AMP_OLD).replace(P.FOOTER, "") == mine, "102 adds more than amp_init + footer"
    mod = P.install(tag="102test")
    assert mod.P102 == {"amp_init": None, "amp_init_n": 0}
    assert mod.E4["k"] == 1 and mod.SPEED == {"drop_floor": False, "sp_donor": False}
    P.reset(mod)
    print("test_p102_engine: OK (frozen sha %s..., %d chars patched)" % (P.FROZEN_SHA256[:12], len(full)))


if __name__ == "__main__":
    main()
