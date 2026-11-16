"""Run every 113 script in order (CPU, about 6 minutes on the first run while build_members caches the circuits).
  python -I -X utf8 experiments/113-coefficients-shape/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for s in ("build_members.py", "c1_alpha.py", "c23_stored.py", "c4_band.py", "c5_effective.py", "c6_overlap.py"):
    print("=== %s" % s, flush=True)
    subprocess.run([sys.executable, "-I", "-X", "utf8", str(HERE / s)], check=True)
