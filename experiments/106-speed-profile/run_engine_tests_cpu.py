"""106: run src's learned-mask engine test files (CPU, toy geometry) against the PATCHED engine copy.

The fastlm module is registered as circuit.instrument.learned_mask before pytest imports the tests, so every test
exercises the prototype. Batched-term flags are excluded: the tests' stub inference stores 4 rows per stream and cannot
take stacked rows (the GPU bench checks those).

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=src python experiments/106-speed-profile/run_engine_tests_cpu.py
  env: FLAGS (default early_exit,donor_gpu,rank_nosync)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

import pytest  # noqa: E402

import fastlm  # noqa: E402

FLAGS = [f for f in os.environ.get("FLAGS", "early_exit,donor_gpu,rank_nosync").split(",") if f]
mod = fastlm.install(FLAGS)
sys.modules["circuit.instrument.learned_mask"] = mod
import circuit.instrument as CI  # noqa: E402
CI.learned_mask = mod
print("engine under test:", mod.__name__, FLAGS, flush=True)
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider",
                      str(REPO / "tests" / "circuit" / "instrument" / "test_learned_mask.py"),
                      str(REPO / "tests" / "circuit" / "instrument" / "test_learned_mask_resample.py")]))
