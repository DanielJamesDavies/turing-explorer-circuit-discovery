"""Run every 111 script in order and log the printed numbers to out/run_all.log (about 1 min on the CPU).

  python -X utf8 experiments/111-failure-anatomy/run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["a1_anatomy.py", "a2_specificity.py", "a3_curves.py", "a4_runinfo.py", "a5_stable_bands.py"]

(HERE / "out").mkdir(exist_ok=True)
with open(HERE / "out" / "run_all.log", "w", encoding="utf-8") as log:
    for s in SCRIPTS:
        log.write("==== %s\n" % s)
        log.flush()
        r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / s)], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            sys.exit("%s failed (see out/run_all.log)" % s)
print("done:", HERE / "out" / "run_all.log")
