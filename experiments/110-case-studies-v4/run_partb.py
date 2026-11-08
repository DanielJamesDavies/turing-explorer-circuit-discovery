"""Part B screen in ONE model load: member reports (report.py) and probes (probes.py) for every candidate.

  PYTHONPATH=src python -X utf8 experiments/110-case-studies-v4/run_partb.py    env: KEYS (default probes.PROBES), N_MEMBERS
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib110 as L  # noqa: E402

L.use_production()
os.environ.setdefault("N_MEMBERS", "25")
import probes as P  # noqa: E402
import report as R  # noqa: E402


def main():
    import validate as VAL
    keys = [k for k in os.environ.get("KEYS", ",".join(P.PROBES)).split(",") if k]
    Vd = VAL.Validator()
    B = R.make_browser(Vd.G)
    d, dm = L.scores("strong"), L.scores("mid")
    for sub in ("reports", "members", "probes"):
        (L.RESULTS / sub).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for key in keys:
        if key in P.PROBES:
            lines = ["# %s — probes (clean model) on the production setup" % key]
            Vd.probes(key, {"probes": P.PROBES[key]}, lines)
            (L.RESULTS / "probes" / ("%s.md" % key)).write_text("\n".join(lines) + "\n", encoding="utf-8")
        path = L.RESULTS / "reports" / ("%s.md" % key)
        if not path.exists() or os.environ.get("FORCE"):
            txt, cdf = R.report(B, key, d.loc[key], dm.loc[key] if key in dm.index else None)
            path.write_text(txt, encoding="utf-8")
            cdf.to_csv(L.RESULTS / "members" / ("%s.csv" % key), index=False)
        print("%-16s done %.0fs" % (key, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
