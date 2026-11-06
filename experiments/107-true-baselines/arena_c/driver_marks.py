"""Arena C (DAN-194): 062's production driver (experiments/062-h100-protocol-v1/driver.py, unchanged) on the Marks
family, for latent targets under WCM_VERSION=v3. The family is registered (and SAEBank(...)'s dispatch pointed at
MarksSAEBank) before the driver is imported, because the driver configures the family's stores at import.
The embedding is out of scope by default (MarksSAEBank puts a NullSAE at (0, embed) unless MARKS_EMBED=1), so a latent
target's upstream sites are its upstream attn / mlp / resid sites (WCM_ADAPTER.md sections 6-7).

  FAMILY=pythia-70m-deduped-marks DATA=data_public/pythia-70m STORES=<data_pass_marks.py --out> OUT=<dir> \
  TARGETS=<file of L.kind.i keys> MODE=main WCM_VERSION=v3 ORDER=list SKIP_SPEC=1 HF_HUB_OFFLINE=1 PYTHONPATH=src \
  ./.venv/bin/python experiments/107-true-baselines/arena_c/driver_marks.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import wcm_marks as WM  # noqa: E402

if os.environ.get("FAMILY", WM.FAMILY) != WM.FAMILY:
    raise SystemExit("driver_marks.py runs FAMILY=%s only" % WM.FAMILY)
os.environ["FAMILY"] = WM.FAMILY
WM.register(patch_bank_dispatch=True)
sys.path.insert(0, str(ROOT / "experiments" / "062-h100-protocol-v1"))
import driver  # noqa: E402

if __name__ == "__main__":
    driver.main()
