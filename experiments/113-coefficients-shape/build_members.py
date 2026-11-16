"""Read every archived production circuit once and cache its members (out/members_<model>_<arm>.parquet), then check
the member counts against the eval rows' n.

  python -I -X utf8 experiments/113-coefficients-shape/build_members.py [--refresh]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib113 as C  # noqa: E402

if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    jobs = [(m, "main") for m in C.MODELS] + [("turingllm", p.name) for p in sorted(C.TL_SWEEP.iterdir()) if p.is_dir()]
    for model, arm in jobs:
        df = C.members(model, arm, refresh=refresh)
        n = df.groupby("seed").size()
        print("%-11s %-10s circuits %5d members %8d alpha missing %d" % (
            model, arm, n.size, len(df), int(df.alpha.isna().sum())))
        if arm == "main":
            ev = C.L.main_eval(model)
            ev = ev[ev.held == "strong"].set_index("seed")
            both = n.index.intersection(ev.index)
            bad = (n[both] != ev.loc[both, "n"].astype(int)).sum()
            print("    eval rows %d, with a circuit %d, member count != eval n: %d" % (len(ev), len(both), bad))
