"""Run every 109 table / figure script in order (CPU, a few seconds each). Missing archives print "not yet available".

  python -X utf8 experiments/109-pod-run-analysis/run_all.py
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["t1_headline.py", "t2_exclusions.py", "t3_alpha1.py", "t4_robustness.py", "t5_unweighted_pythia.py",
           "t6_tl_sweep.py", "t7_pilot.py", "t8_triad.py"]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for s in SCRIPTS:
        print("\n==== %s" % s, flush=True)
        sys.argv = [str(HERE / s)]
        runpy.run_path(str(HERE / s), run_name="__main__")
